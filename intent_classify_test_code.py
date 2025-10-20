#!/usr/bin/env python3
import json
import time
from pathlib import Path

import httpx
import pandas as pd

URL = "http://99.1.82.207:8080/llm-studio/v1/api/task/generate/syncapi/busan_ai_llm/intent_classify"
INPUT_XLSX = Path("test.xlsx")
OUTPUT_JSONL = Path("test.jsonl")  # 전체 결과 저장
FAILED_JSONL = Path("failed.jsonl")  # 실패 결과만 저장

HEADERS = {"Content-Type": "application/json"}

TIMEOUT_SEC = 30.0
RETRY = 2
SLEEP_BETWEEN = 0.2


def send_request(client: httpx.Client, user_query: str) -> dict:
    """user_query를 전송하고 결과를 dict로 반환."""
    payload = {"user_query": user_query}
    attempt = 0
    last_exc = None

    while attempt <= RETRY:
        try:
            start = time.time()
            resp = client.post(URL, headers=HEADERS, json=payload, timeout=TIMEOUT_SEC)
            elapsed = time.time() - start

            try:
                body = resp.json()
            except Exception:
                body = {"raw_text": resp.text}

            return {
                "user_query": user_query,
                "status_code": resp.status_code,
                "elapsed_sec": round(elapsed, 3),
                "response": body,
            }
        except Exception as e:
            last_exc = e
            attempt += 1
            if attempt <= RETRY:
                time.sleep(0.5)

    return {
        "user_query": user_query,
        "status_code": None,
        "elapsed_sec": None,
        "error": repr(last_exc),
    }


def write_jsonl(records: list[dict], path: Path):
    """records(list of dict)를 JSONL로 저장."""
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    if not INPUT_XLSX.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {INPUT_XLSX}")

    df = pd.read_excel(INPUT_XLSX, dtype=str)
    if "질의내용" not in df.columns:
        raise KeyError("엑셀에 '질의내용' 컬럼이 없습니다.")

    queries = [q for q in df["질의내용"].fillna("").tolist() if q.strip()]
    if not queries:
        print("보낼 질의가 없습니다.")
        return

    all_results = []
    failed_results = []

    with httpx.Client() as client:
        total = len(queries)
        for i, q in enumerate(queries, start=1):
            result = send_request(client, q)
            all_results.append(result)

            # 실패 조건: status_code가 None이거나 200~299 범위가 아닐 때
            if not (result.get("status_code") and 200 <= result["status_code"] < 300):
                failed_results.append(result)

            if i % 10 == 0 or i == total:
                print(f"[{i}/{total}] 진행중… 실패 {len(failed_results)}건")

            if SLEEP_BETWEEN > 0:
                time.sleep(SLEEP_BETWEEN)

    # 전체 결과 저장
    write_jsonl(all_results, OUTPUT_JSONL)

    # 실패한 요청만 별도 저장
    if failed_results:
        write_jsonl(failed_results, FAILED_JSONL)
        print(
            f"❗ 실패한 요청 {len(failed_results)}건을 '{FAILED_JSONL}'에 저장했습니다."
        )
    else:
        print("✅ 모든 요청이 성공했습니다. 실패 요청 없음.")

    print(f"전체 결과 {len(all_results)}건을 '{OUTPUT_JSONL}'에 저장했습니다.")


if __name__ == "__main__":
    main()
