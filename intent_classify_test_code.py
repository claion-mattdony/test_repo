#!/usr/bin/env python3
import json
import re
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pandas as pd
from pydantic import BaseModel, Field, ValidationError


# -----------------------------
# 로깅 (필요 시 수준/포맷 조정)
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("pipeline")


# -----------------------------
# 설정
# -----------------------------
EXPAND_URL = "http://99.1.82.207:8080/llm-studio/v1/api/task/generate/syncapi/busan_ai_llm/query_expand"
INTENT_URL = "http://99.1.82.207:8080/llm-studio/v1/api/task/generate/syncapi/busan_ai_llm/intent_classify"

HEADERS = {"Content-Type": "application/json"}

INPUT_XLSX = Path("test.xlsx")
INPUT_COLUMN = "질의 내용"  # ← 공백 포함
RESULTS_JSONL = Path("results.jsonl")   # 두 단계 모두 성공
FAILED_JSONL = Path("failed.jsonl")     # 실패만

TIMEOUT_SEC = 30.0
RETRY = 2
PAUSE = 0.15  # 요청 간 간격(서버 보호용). 필요 시 0으로.


# -----------------------------
# 모델 & 파서
# -----------------------------
class QueryInfo(BaseModel):
    query_complete: Optional[str] = Field(default=None)
    search_queries: Optional[List[str]] = Field(default=None)

def parse_query_answer(answer: str) -> Dict[str, Any]:
    """
    쿼리 확장 응답(answer)에서 query_complete 등을 추출.
    사용자가 제공한 정규식/모델을 그대로 사용.
    실패 시 {'error': "..."} 형태 반환.
    """
    try:
        logger.info("get_parsed_query called")
        pattern = r"\{\s*(\"|\')query_complete.*?\s*\]\s*\}"
        match = re.search(pattern, answer, re.DOTALL)
        if not match:
            return {"error": "no_json_block_matched"}
        json_string = match.group(0)
        parsed = QueryInfo.model_validate_json(json_string).model_dump()
        return parsed
    except ValidationError as ve:
        return {"error": f"validation_error: {ve}"}
    except Exception as e:
        return {"error": f"parse_exception: {repr(e)}"}


class Intent(BaseModel):
    no: int | str
    intent_name: str

class IntentInfo(BaseModel):
    intent: Optional[Intent] = Field(default=None)

def parse_intent_answer(answer: str) -> Dict[str, Any]:
    """
    의도분석 응답(answer)에서 intent 객체를 추출.
    사용자가 제공한 정규식/모델을 그대로 사용.
    실패 시 {'error': "..."} 형태 반환.
    """
    try:
        logger.info("get_parsed_intent called")
        pattern = r"\{\s*(\"|\')intent.*?\s*\}*\s*\}"
        match = re.search(pattern, answer, re.DOTALL)
        if not match:
            return {"error": "no_json_block_matched"}
        json_string = match.group(0)
        parsed = IntentInfo.model_validate_json(json_string).model_dump()
        return parsed
    except ValidationError as ve:
        return {"error": f"validation_error: {ve}"}
    except Exception as e:
        return {"error": f"parse_exception: {repr(e)}"}


# -----------------------------
# HTTP 유틸
# -----------------------------
def post_json(client: httpx.Client, url: str, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any], float]:
    """
    POST -> (status_code, json_body_or_text_wrapped, elapsed_sec)
    JSON 파싱 실패 시 {'raw_text': resp.text}로 감싼다.
    """
    start = time.time()
    resp = client.post(url, headers=HEADERS, json=payload, timeout=TIMEOUT_SEC)
    elapsed = round(time.time() - start, 3)
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}
    return resp.status_code, body, elapsed

def robust_post(client: httpx.Client, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    재시도 포함 POST. 항상 dict 반환.
    """
    attempt = 0
    last_exc = None
    while attempt <= RETRY:
        try:
            status, body, elapsed = post_json(client, url, payload)
            return {"status_code": status, "elapsed_sec": elapsed, "response": body}
        except Exception as e:
            last_exc = e
            attempt += 1
            if attempt <= RETRY:
                time.sleep(0.5)
    return {"status_code": None, "elapsed_sec": None, "error": repr(last_exc), "response": None}


# -----------------------------
# 파일 저장
# -----------------------------
def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# -----------------------------
# 메인 로직
# -----------------------------
def main() -> None:
    if not INPUT_XLSX.exists():
        raise FileNotFoundError(f"입력 파일 없음: {INPUT_XLSX}")

    df = pd.read_excel(INPUT_XLSX, dtype=str)
    if INPUT_COLUMN not in df.columns:
        raise KeyError(f"엑셀에 '{INPUT_COLUMN}' 컬럼이 없습니다.")

    queries = [q for q in df[INPUT_COLUMN].fillna("").tolist() if q.strip()]
    if not queries:
        print("보낼 질의가 없습니다.")
        return

    results_ok: List[Dict[str, Any]] = []
    results_failed: List[Dict[str, Any]] = []

    with httpx.Client() as client:
        total = len(queries)
        for i, original_query in enumerate(queries, start=1):
            # -------------------------
            # 1) 쿼리 확장 요청
            # -------------------------
            expand_payload = {"user_query": original_query, "stream": False}
            expand_call = robust_post(client, EXPAND_URL, expand_payload)

            # HTTP 실패 체크
            if not (expand_call.get("status_code") and 200 <= expand_call["status_code"] < 300):
                results_failed.append({
                    "stage": "expand_api",
                    "reason": "http_error",
                    "original_user_query": original_query,
                    "expand_request": expand_payload,
                    "expand_result": expand_call,
                })
                if PAUSE: time.sleep(PAUSE)
                continue

            # 응답에서 answer 추출
            expand_body = expand_call["response"] or {}
            llm_result = (expand_body or {}).get("llm_result", {})
            answer = llm_result.get("answer")
            if not isinstance(answer, str):
                results_failed.append({
                    "stage": "expand_parse",
                    "reason": "answer_not_string",
                    "original_user_query": original_query,
                    "expand_request": expand_payload,
                    "expand_result": expand_call,
                })
                if PAUSE: time.sleep(PAUSE)
                continue

            # 2) 파싱 (query_complete 얻기)
            parsed_query = parse_query_answer(answer)
            if "error" in parsed_query or not parsed_query.get("query_complete"):
                results_failed.append({
                    "stage": "expand_parse",
                    "reason": parsed_query.get("error") or "missing_query_complete",
                    "original_user_query": original_query,
                    "expand_request": expand_payload,
                    "expand_result": expand_call,
                    "parsed_query": parsed_query,
                })
                if PAUSE: time.sleep(PAUSE)
                continue

            query_complete = parsed_query["query_complete"]

            # -------------------------
            # 3) 의도분석 요청 (user_query=query_complete)
            # -------------------------
            intent_payload = {"user_query": query_complete}
            intent_call = robust_post(client, INTENT_URL, intent_payload)

            if not (intent_call.get("status_code") and 200 <= intent_call["status_code"] < 300):
                results_failed.append({
                    "stage": "intent_api",
                    "reason": "http_error",
                    "original_user_query": original_query,
                    "query_complete": query_complete,
                    "expand_request": expand_payload,
                    "expand_result": expand_call,
                    "parsed_query": parsed_query,
                    "intent_request": intent_payload,
                    "intent_result": intent_call,
                })
                if PAUSE: time.sleep(PAUSE)
                continue

            # 의도분석 응답에서 answer 추출
            intent_body = intent_call["response"] or {}
            intent_llm_result = (intent_body or {}).get("llm_result", {})
            intent_answer = intent_llm_result.get("answer")
            if not isinstance(intent_answer, str):
                results_failed.append({
                    "stage": "intent_parse",
                    "reason": "answer_not_string",
                    "original_user_query": original_query,
                    "query_complete": query_complete,
                    "expand_request": expand_payload,
                    "expand_result": expand_call,
                    "parsed_query": parsed_query,
                    "intent_request": intent_payload,
                    "intent_result": intent_call,
                })
                if PAUSE: time.sleep(PAUSE)
                continue

            # 4) 파싱 (intent 추출)  ← 신규 단계
            parsed_intent = parse_intent_answer(intent_answer)
            if "error" in parsed_intent or not parsed_intent.get("intent"):
                results_failed.append({
                    "stage": "intent_parse",
                    "reason": parsed_intent.get("error") or "missing_intent",
                    "original_user_query": original_query,
                    "query_complete": query_complete,
                    "expand_request": expand_payload,
                    "expand_result": expand_call,
                    "parsed_query": parsed_query,
                    "intent_request": intent_payload,
                    "intent_result": intent_call,
                    "parsed_intent": parsed_intent,
                })
                if PAUSE: time.sleep(PAUSE)
                continue

            # 두 단계 모두 성공 → results.jsonl 저장용 레코드
            results_ok.append({
                "original_user_query": original_query,
                "expand": {
                    "request": expand_payload,
                    "result": expand_call,
                },
                "parsed_query": parsed_query,     # {"query_complete": ..., "search_queries": [...]}
                "intent": {
                    "request": intent_payload,
                    "result": intent_call,
                },
                "parsed_intent": parsed_intent,   # {"intent": {"no": ..., "intent_name": ...}}
            })

            # 진행상황 출력
            if i % 10 == 0 or i == total:
                print(f"[{i}/{total}] 진행… 성공 {len(results_ok)}건, 실패 {len(results_failed)}건")

            if PAUSE:
                time.sleep(PAUSE)

    # 저장
    if results_ok:
        write_jsonl(results_ok, RESULTS_JSONL)
        print(f"✅ 성공 {len(results_ok)}건을 '{RESULTS_JSONL}'에 저장했습니다.")
    else:
        print("성공 케이스가 없습니다.")

    if results_failed:
        write_jsonl(results_failed, FAILED_JSONL)
        print(f"❗ 실패 {len(results_failed)}건을 '{FAILED_JSONL}'에 저장했습니다.")
    else:
        print("실패 케이스가 없습니다.")


if __name__ == "__main__":
    main()
