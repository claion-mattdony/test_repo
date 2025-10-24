# -*- coding: utf-8 -*-
"""
엑셀의 "질문" 컬럼으로 API 비동기 테스트 러너
- httpx.AsyncClient(timeout=30) 사용
- 성공: JSONL (요청한 포맷)
- 실패: JSONL (에러 정보 포함)
"""

import time  # <-- 추가
import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pandas as pd


# -----------------------------
# 설정(필요 시 조절)
# -----------------------------
CONCURRENCY = 10         # 동시에 요청할 최대 개수
TIMEOUT_SEC = 30         # httpx timeout
RETRIES = 2              # 실패 시 재시도 횟수
BACKOFF_BASE = 0.8       # 지수 백오프 기본(초)
ENCODING = "utf-8"       # 파일 저장 인코딩
EXCEL_ENGINE = None      # 필요 시 "openpyxl" 등 지정 가능


# -----------------------------
# 주신 케이스 구성 (파이썬 dict 로 그대로 사용)
# -----------------------------
CASES: Dict[str, Dict[str, Any]] = {
    "case1_1": {
        "input_files": [
            "test_datas/251017_내부자료_테스트셋_질의응답.xlsx"
        ],
        "output_files": [
            "outputs/1_1_inner_results.jsonl"
        ],
        "err_files": [
            "outputs/1_1_inner_errs.jsonl"
        ],
        "url": "http://99.1.82.183:18080/api/v1/qdrant/retrieve",
        "headers": {"Content-Type": "application/json", "Accept": "application/json"},
        "bodys": {
            "collection_name": "busan_city_20251013_latechunking",
            "reranking": {
                "is_active": True,
                "target": "text",
                "include_summary": False
            },
            "filter": {
                "key": "group",
                "value": [
                    "내부자료"
                ]
            }
        }
    },
    "case1_2": {
        "input_files": [
            "test_datas/251017_내부자료_테스트셋_질의응답.xlsx",
            "test_datas/251017_온나라_테스트셋_질의응답.xlsx"
        ],
        "output_files": [
            "outputs/1_2_inner_results.jsonl",
            "outputs/1_2_onnara_results.jsonl"
        ],
        "err_files": [
            "outputs/1_2_inner_errs.jsonl",
            "outputs/1_2_onnara_errs.jsonl"
        ],
        "url": "http://99.1.82.183:18080/api/v1/qdrant/retrieve",
        "headers": {"Content-Type": "application/json", "Accept": "application/json"},
        "bodys": {
            "collection_name": "busan_city_20251013_latechunking",
            "reranking": {
                "is_active": True,
                "target": "text",
                "include_summary": False
            },
            "filter": {
                "key": "group",
                "value": [
                    "내부자료",
                    "온나라"
                ]
            }
        }
    },
    "case2_1": {
        "input_files": [
            "test_datas/251017_내부자료_테스트셋_질의응답.xlsx"
        ],
        "output_files": [
            "outputs/2_1_results.jsonl"
        ],
        "err_files": [
            "outputs/2_1_errs.jsonl"
        ],
        "url": "http://99.1.82.183:18080/api/v1/qdrant/retrieve",
        "headers": {"Content-Type": "application/json", "Accept": "application/json"},
        "bodys": {
            "collection_name": "busan_inner_onnara_512_20251023",
            "reranking": {
                "is_active": True,
                "target": "text",
                "include_summary": False
            },
            "filter": {
                "key": "group",
                "value": [
                    "내부자료"
                ]
            }
        }
    },
    "case2_2": {
        "input_files": [
            "test_datas/251017_내부자료_테스트셋_질의응답.xlsx",
            "test_datas/251017_온나라_테스트셋_질의응답.xlsx"
        ],
        "output_files": [
            "outputs/2_2_inner_results.jsonl",
            "outputs/2_2_onnara_results.jsonl"
        ],
        "err_files": [
            "outputs/2_2_inner_errs.jsonl",
            "outputs/2_2_onnara_errs.jsonl"
        ],
        "url": "http://99.1.82.183:18080/api/v1/qdrant/retrieve",
        "headers": {"Content-Type": "application/json", "Accept": "application/json"},
        "bodys": {
            "collection_name": "busan_inner_onnara_512_20251023",
            "reranking": {
                "is_active": True,
                "target": "text",
                "include_summary": False
            },
            "filter": {
                "key": "group",
                "value": [
                    "내부자료",
                    "온나라"
                ]
            }
        }
    },
    "case2_3": {
        "input_files": [
            "test_datas/251017_내부자료_테스트셋_질의응답.xlsx"
        ],
        "output_files": [
            "outputs/2_3_results.jsonl"
        ],
        "err_files": [
            "outputs/2_3_errs.jsonl"
        ],
        "url": "http://99.1.82.184:18080/api/v1/qdrant/retrieve",
        "headers": {"Content-Type": "application/json", "Accept": "application/json"},
        "bodys": {
            "collection_name": "busan_inner_onnara_512_20251023",
            "reranking": {
                "is_active": True,
                "target": "text",
                "include_summary": True
            },
            "filter": {
                "key": "group",
                "value": [
                    "내부자료"
                ]
            }
        }
    },
    "case2_4": {
        "input_files": [
            "test_datas/251017_내부자료_테스트셋_질의응답.xlsx",
            "test_datas/251017_온나라_테스트셋_질의응답.xlsx"
        ],
        "output_files": [
            "outputs/2_4_inner_results.jsonl",
            "outputs/2_4_onnara_results.jsonl"
        ],
        "err_files": [
            "outputs/2_4_inner_errs.jsonl",
            "outputs/2_4_onnara_errs.jsonl"
        ],
        "url": "http://99.1.82.184:18080/api/v1/qdrant/retrieve",
        "headers": {"Content-Type": "application/json", "Accept": "application/json"},
        "bodys": {
            "collection_name": "busan_inner_onnara_512_20251023",
            "reranking": {
                "is_active": True,
                "target": "text",
                "include_summary": True
            },
            "filter": {
                "key": "group",
                "value": [
                    "내부자료",
                    "온나라"
                ]
            }
        }
    }
}


# -----------------------------
# 유틸
# -----------------------------
def ensure_dir_for_file(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def to_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    ensure_dir_for_file(path)
    with open(path, "a", encoding=ENCODING) as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_dataframe(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path, engine=EXCEL_ENGINE)
    # 필요한 컬럼 보정/확인
    required = ["파일명", "본문", "질문", "답변수정", "참고 내용"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"엑셀({excel_path})에 필수 컬럼이 없습니다: {missing}")
    return df


@dataclass
class CaseIO:
    input_path: str
    out_path: str
    err_path: str


def pair_io(case_conf: Dict[str, Any]) -> List[CaseIO]:
    ins = case_conf["input_files"]
    outs = case_conf["output_files"]
    errs = case_conf["err_files"]
    if not (len(ins) == len(outs) == len(errs)):
        raise ValueError(
            f"input/output/err 파일 개수가 맞지 않습니다: "
            f"{len(ins)} != {len(outs)} != {len(errs)}"
        )
    return [CaseIO(i, o, e) for i, o, e in zip(ins, outs, errs)]


def build_body(base_body: Dict[str, Any], question: str) -> Dict[str, Any]:
    # 질문을 bodys 최상위에 "queries": "<질문 문자열>" 로 주입
    body = dict(base_body)  # shallow copy
    body["queries"] = [str(question)] if question is not None else [""]
    return body


def success_record(row: pd.Series, api_json: Any, latency: float | None) -> Dict[str, Any]:
    return {
        "파일명": str(row["파일명"]),
        "본문": str(row["본문"]),
        "질문": str(row["질문"]),
        "답변수정": str(row["답변수정"]),
        "참고 내용": str(row["참고 내용"]),
        "results": api_json,
        "latency": round(latency, 4) if latency is not None else None,
    }


def error_record(row: Optional[pd.Series], err_type: str, message: str, status: Optional[int] = None) -> Dict[str, Any]:
    base = {
        "파일명": str(row["파일명"]) if row is not None and "파일명" in row else "",
        "본문": str(row["본문"]) if row is not None and "본문" in row else "",
        "질문": str(row["질문"]) if row is not None and "질문" in row else "",
        "답변수정": str(row["답변수정"]) if row is not None and "답변수정" in row else "",
        "참고 내용": str(row["참고 내용"]) if row is not None and "참고 내용" in row else "",
        "error": {
            "type": err_type,
            "status": status,
            "message": message,
        },
    }
    return base


async def request_with_retry(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    json_body: Dict[str, Any],
) -> Tuple[bool, Any, Optional[int], Optional[str], Optional[float]]:
    """
    반환: (ok, response_json_or_none, status_code_or_none, error_message_or_none, latency_seconds_or_none)
    latency: 최종 시도 1회에 대한 왕복 시간(초)
    """
    for attempt in range(RETRIES + 1):
        try:
            start = time.perf_counter()
            resp = await client.post(url, headers=headers, json=json_body)
            latency = time.perf_counter() - start

            if 200 <= resp.status_code < 300:
                try:
                    return True, resp.json(), resp.status_code, None, latency
                except Exception:
                    return True, resp.text, resp.status_code, None, latency
            else:
                err_text = resp.text
                if attempt < RETRIES:
                    await asyncio.sleep((BACKOFF_BASE * (2 ** attempt)))
                    continue
                return False, None, resp.status_code, f"HTTP {resp.status_code}: {err_text}", None
        except httpx.ReadTimeout:
            if attempt < RETRIES:
                await asyncio.sleep((BACKOFF_BASE * (2 ** attempt)))
                continue
            return False, None, None, "timeout", None
        except Exception as e:
            if attempt < RETRIES:
                await asyncio.sleep((BACKOFF_BASE * (2 ** attempt)))
                continue
            return False, None, None, f"exception: {repr(e)}", None

    return False, None, None, "unknown", None



async def process_one_row(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    base_body: Dict[str, Any],
    row: pd.Series,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    question = row.get("질문", None)
    if pd.isna(question) or str(question).strip() == "":
        return None, None

    json_body = build_body(base_body, str(question))

    async with sem:
        ok, resp_json, status, err_msg, latency = await request_with_retry(
            client=client, url=url, headers=headers, json_body=json_body
        )

    if ok:
        return success_record(row, resp_json, latency), None
    else:
        return None, error_record(
            row,
            "http_status" if status else ("timeout" if err_msg == "timeout" else "exception"),
            err_msg or "",
            status,
        )



async def process_case(case_name: str, conf: Dict[str, Any]) -> None:
    print(f"[{case_name}] 시작")
    io_pairs = pair_io(conf)

    # HTTP 클라이언트 & 세마포어
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(timeout=TIMEOUT_SEC) as client:
        for io in io_pairs:
            print(f"  - 입력: {io.input_path}")
            df = load_dataframe(io.input_path)

            # 출력 파일 경로 준비(존재하면 append, 새로 만들 디렉토리 ensure)
            ensure_dir_for_file(io.out_path)
            ensure_dir_for_file(io.err_path)

            tasks = []
            for _, row in df.iterrows():
                tasks.append(
                    process_one_row(
                        sem=sem,
                        client=client,
                        url=conf["url"],
                        headers=conf["headers"],
                        base_body=conf["bodys"],
                        row=row,
                    )
                )

            success_batch: List[Dict[str, Any]] = []
            error_batch: List[Dict[str, Any]] = []

            # 결과 수시 flush (메모리 과다 방지)
            FLUSH_EVERY = 200

            for idx, coro in enumerate(asyncio.as_completed(tasks), start=1):
                success_row, error_row = await coro
                if success_row:
                    success_batch.append(success_row)
                if error_row:
                    error_batch.append(error_row)

                if (idx % FLUSH_EVERY) == 0:
                    if success_batch:
                        to_jsonl(io.out_path, success_batch)
                        success_batch.clear()
                    if error_batch:
                        to_jsonl(io.err_path, error_batch)
                        error_batch.clear()

            # 잔여 flush
            if success_batch:
                to_jsonl(io.out_path, success_batch)
            if error_batch:
                to_jsonl(io.err_path, error_batch)

            print(f"  - 완료: {io.input_path} -> results:{io.out_path}, errs:{io.err_path}")

    print(f"[{case_name}] 종료\n")


async def main(selected_cases: Optional[List[str]] = None) -> None:
    """
    selected_cases: 처리할 케이스 이름 리스트. None이면 모든 케이스 처리
    """
    case_names = selected_cases or list(CASES.keys())
    for name in case_names:
        await process_case(name, CASES[name])


if __name__ == "__main__":
    # 예시) 모든 케이스 실행
    # 특정 케이스만 실행하려면: asyncio.run(main(["case1_1", "case2_4"]))
    asyncio.run(main(["case1_1", "case1_2", "case2_1", "case2_2"]))
