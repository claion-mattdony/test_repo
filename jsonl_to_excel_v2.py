# -*- coding: utf-8 -*-
"""
JSONL -> Excel (details + match_per_question)
- details: 레코드(질문)당 retrieved_documents 를 행으로 저장
  - is_same_file: 레코드 '파일명' == payload.file 정확 일치 여부 (정규화 없음)
- match_per_question: 질문 1건 당, 하나라도 일치하는 파일이 있었는지 여부(Boolean)

필요 패키지: pandas, openpyxl
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd


def safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def normalize_base_fields(rec: Dict[str, Any]) -> Dict[str, Any]:
    """기본 필드 표준화 + '참고 내용' -> '참고내용' 통일(값은 그대로 사용)"""
    return {
        "파일명": rec.get("파일명", ""),
        "본문": rec.get("본문", ""),
        "질문": rec.get("질문", ""),
        "답변수정": rec.get("답변수정", ""),
        "참고내용": rec.get("참고내용", rec.get("참고 내용", "")),
        "latency": rec.get("latency", None),
    }


def extract_results(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    rec["results"] 예상 구조:
      {"results": [ {"query": ..., "retrieved_documents": [...]}, ... ]}
    반환: [{"query": str, "docs": [doc, ...]}, ...]
    """
    results_root = rec.get("results", None)
    if results_root is None:
        return []

    if isinstance(results_root, dict):
        items = results_root.get("results", [])
    elif isinstance(results_root, list):
        items = results_root
    else:
        return []

    normalized = []
    for it in items:
        q = it.get("query", "")
        docs = it.get("retrieved_documents", []) or []
        normalized.append({"query": q, "docs": docs})
    return normalized


def truncate(s: Optional[str], limit: int) -> Optional[str]:
    if s is None:
        return None
    s = str(s)
    if limit <= 0:
        return s
    return (s[:limit] + "…") if len(s) > limit else s


def convert_jsonl_to_excel(
    input_path: str,
    output_path: str,
    topk: int = 5,
    truncate_text: int = 500,
) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"입력 파일이 없습니다: {input_path}")

    detail_rows: List[Dict[str, Any]] = []
    match_rows: List[Dict[str, Any]] = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            # 기본 parse 에러 처리
            try:
                rec = json.loads(line)
            except Exception as e:
                # details에 파싱 에러 남김
                detail_rows.append(
                    {
                        "rec_id": line_no,
                        "파일명": "",
                        "질문": f"[JSONDecodeError@line{line_no}]",
                        "latency": None,
                        "block_index": "",
                        "rank": "",
                        "doc_id": "",
                        "score": "",
                        "group": "",
                        "payload_file": "",
                        "payload_page_num": "",
                        "parent_text": "",
                        "text": "",
                        "summary": "",
                        "is_same_file": "",
                        "parse_error": repr(e),
                    }
                )
                # match_per_question에도 남김
                match_rows.append(
                    {
                        "rec_id": line_no,
                        "파일명": "",
                        "질문": f"[JSONDecodeError@line{line_no}]",
                        "latency": None,
                        "any_exact_match": "",
                        "parse_error": repr(e),
                    }
                )
                continue

            base = normalize_base_fields(rec)
            rec_id = line_no
            parsed_blocks = extract_results(rec)

            # ---------- details 작성 (Top-K만 저장) ----------
            for block_idx, block in enumerate(parsed_blocks):
                docs = block["docs"][: topk if topk and topk > 0 else None]
                for rank, doc in enumerate(docs, start=1):
                    payload = doc.get("payload", {}) or {}
                    payload_file = payload.get("file", "")

                    row = {
                        "rec_id": rec_id,
                        "파일명": base["파일명"],
                        "질문": base["질문"],
                        "latency": base["latency"],
                        "block_index": block_idx,
                        "rank": rank,
                        "doc_id": doc.get("id", ""),
                        "score": doc.get("score", ""),
                        "group": payload.get("group", ""),
                        "payload_file": payload_file,
                        "payload_page_num": payload.get("page_num", ""),
                        "parent_text": truncate(
                            payload.get("parent_text", ""), truncate_text
                        ),
                        "text": truncate(payload.get("text", ""), truncate_text),
                        "summary": truncate(payload.get("summary", ""), truncate_text),
                        # 정확 일치 비교(정규화 없음)
                        "is_same_file": (base["파일명"] == payload_file),
                    }
                    detail_rows.append(row)

            # ---------- match_per_question 작성 (전체 문서 기준으로 일치 여부만) ----------
            # any_exact_match는 **전체 retrieved_documents**(Top-K 제한 없이) 기준으로 판단
            any_exact_match = False
            for block in parsed_blocks:
                for doc in block["docs"]:
                    payload_file = safe_get(doc, "payload", "file", default="")
                    if base["파일명"] == payload_file:
                        any_exact_match = True
                        break
                if any_exact_match:
                    break

            match_rows.append(
                {
                    "rec_id": rec_id,
                    "파일명": base["파일명"],
                    "질문": base["질문"],
                    "latency": base["latency"],
                    "any_exact_match": any_exact_match,
                }
            )

    # DataFrame 구성
    df_details = pd.DataFrame(
        detail_rows,
        columns=[
            "rec_id",
            "파일명",
            "질문",
            "latency",
            "block_index",
            "rank",
            "doc_id",
            "score",
            "group",
            "payload_file",
            "payload_page_num",
            "parent_text",
            "text",
            "summary",
            "is_same_file",
            "parse_error",
        ],
    )

    df_match = pd.DataFrame(
        match_rows,
        columns=[
            "rec_id",
            "파일명",
            "질문",
            "latency",
            "any_exact_match",
            "parse_error",
        ],
    )

    # 저장
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_details.to_excel(writer, sheet_name="details", index=False)
        df_match.to_excel(writer, sheet_name="match_per_question", index=False)

    print(f"완료: {input_path} -> {output_path}")
    print(f" - details rows: {len(df_details)}")
    print(f" - match_per_question rows: {len(df_match)}")


def main():
    parser = argparse.ArgumentParser(
        description="JSONL 결과를 엑셀(details + match_per_question)로 변환"
    )
    parser.add_argument("--input", required=True, help="입력 JSONL 경로")
    parser.add_argument("--output", required=True, help="출력 XLSX 경로")
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="details 시트에 저장할 Top-K 문서 수 (0=전체)",
    )
    parser.add_argument(
        "--truncate-text", type=int, default=500, help="텍스트 컬럼 최대 길이(0=무제한)"
    )
    args = parser.parse_args()

    convert_jsonl_to_excel(
        input_path=args.input,
        output_path=args.output,
        topk=args.topk,
        truncate_text=args.truncate_text,
    )


if __name__ == "__main__":
    main()
