# .key file 복사되도록 Dockerifle 수정
import asyncio
import os

from busan_bot_ui import BusanBotUIManager
from db_manager import DBManager

from app.core import settings
from app.core.logger import get_logger
from app.services import (
    RAG,
    LLMStudio,
    Task,
    encode_token,
    get_group_filter,
    get_service_filter,
)

logger = get_logger(__file__)

TASK_QUEUE_SIZE: int = settings.TASK_QUEUE_SIZE
POLLING_INTERVER = settings.POLLING_INTERVER
RAG_INTENT_CODES = settings.RAG_INTENT_CODES

queue = asyncio.Queue(maxsize=TASK_QUEUE_SIZE)
task_client = Task()


async def poll_task(interval: float = POLLING_INTERVER):
    while True:
        try:
            fetched_task = await task_client.get_task()
            if fetched_task is not None:
                await queue.put(fetched_task)
                logger.info(
                    f"Input ID={fetched_task.get('task').get('id')} / size={queue.qsize()}"
                )

        except Exception as e:
            logger.error(f"[Error - poll_task]: {e}", exc_info=True)

        await asyncio.sleep(interval)


def get_dialog_history(
    use_dialog: bool,
    total_cnt: int,
    api_with_dialog: bool,
    dialog_cnt: int,
    dialog_history: list = [],
) -> list:
    """
    use_dialog >> 대화이력 조회 여부
    api_with_dialog >> 사용하고 하는 API에서 대회이력 사용 여부
    total_cnt >> 대화이력 전체 조회 개수 (질문-답변 쌍)
    dialog_cnt >> 사용하고 하는 API에서 사용할 대회이력 개수 (질문-답변 쌍)
    dialog_history >> 조회된 대화이력 / len(dialog_history) == total_cnt * 2
    """
    if (
        (use_dialog is False)
        or (api_with_dialog is False)
        or (total_cnt <= 0)
        or (dialog_cnt <= 0)
        or len(dialog_history) <= 0
    ):
        return []
    else:
        max_length = dialog_cnt if total_cnt >= dialog_cnt else total_cnt
        return dialog_history[-(max_length * 2) :]


async def handle_task(worker_id: str):
    pid = os.getpid()
    logger.info(f"Worker #{worker_id} Start")

    # DB Manager
    db_manager = DBManager(settings.DATABASE_URL)
    logger.info(f"Initializing DBManager with URL: {settings.DATABASE_URL}")

    # NAVERWORKS BOT Manager
    bot_manager = BusanBotUIManager(
        config_path=settings.CONFIG_PATH,
        static_messages_path=settings.STATIC_MESSAGES_PATH,
    )
    logger.info(f"[PID {pid}] Initialized BusanBotUIManager")

    # LLMStudio
    llm_studio = LLMStudio()
    logger.info(f"[PID {pid}] Initialized LLMStudio")
    llm_studio_gen_answer_apis = {
        # 일상대화
        101: llm_studio.get_conversation,
        # 질의응답
        201: llm_studio.get_qa_response,
        # 정책계획 보고서 초안 작성
        301: llm_studio.get_draft_policy_plan,
        # 보도자료 초안 작성
        302: llm_studio.get_draft_press,
        # 인사말씀 초안 작성
        303: llm_studio.get_draft_speech,
        # 행사 시나리오 초안 작성
        304: llm_studio.get_draft_event,
        # 월간업무보고 초안 작성
        305: llm_studio.get_draft_monthly_report,
        # 주간업무보고 초안 작성
        306: llm_studio.get_draft_weekly_report,
        # 의회보고자료 초안 작성
        307: llm_studio.get_draft_congress,
        # 국/과장 보고자료 초안 작성
        308: llm_studio.get_draft_report,
        # 시행문 초안 작성
        309: llm_studio.get_draft_letter,
        # 백서 초안 작성
        310: llm_studio.get_draft_annual_report,
        # 주간정책자료 초안 작성
        311: llm_studio.get_draft_weekly_policy,
        # 민감정보
        501: llm_studio.get_conversation,
        # 미분류
        900: llm_studio.get_conversation,
        # 콘텐츠 요약
        # 99999: llm_studio.get_content_summary,  # // TODO: API 사용 여부 확인
    }

    # RAG
    rag = RAG()
    logger.info(f"[PID {pid}] Initialized RAG")

    while True:
        task_info = await queue.get()
        logger.info(
            f"[PID {pid}] Received new task from queue: {task_info.get('task').get('id')}"
        )

        chat_session_id = task_info.get("chat_id")
        task = task_info.get("task")

        task_id = task.get("id")
        bot_id = task.get("headers").get("x-works-botid")

        body = task.get("body")
        user_id = body.get("source").get("user_id")
        content = body.get("content")
        content_type = content.get("type")
        message = content.get("text")

        status = "success"
        is_rag = False
        rag_psgs = None

        await bot_manager.write_log(
            task_id=task_id,
            user_id=user_id,
            bot_id=bot_id,
            log={
                "WAS": "AI Service",
                "STEP": "01. Task WAS 메세지큐 수신",
                "BODY": content,
            },
        )
        await bot_manager.write_admin_log(
            task_id=task_id,
            user_id=user_id,
            bot_id=bot_id,
            log={"AL-USER-MESSAGE": message},
        )

        try:
            if content_type == "text":
                # DB 대화이력 조회
                dialog_history = []
                if settings.USE_DIALOG_HISTORY:
                    try:
                        logger.debug(
                            f"[PID {pid}][{task_id=}] Fetching chat history for session: {chat_session_id}"
                        )
                        dialog_pairs = (
                            await db_manager.get_chat_histories_by_session_id(
                                chat_session_id=chat_session_id,
                                count=settings.DIALOG_HISTORY_COUNT,
                            )
                        )
                        for d in dialog_pairs:
                            if d.user_message is not None and d.bot_message is not None:
                                dialog_history.append(
                                    {"role": "user", "content": d.user_message}
                                )
                                dialog_history.append(
                                    {"role": "assistant", "content": d.bot_message}
                                )
                        log_body = {"result": "success", "detail": dialog_history}
                    except Exception as e:
                        err_message = "DB 대화이력 조회 ERR"
                        log_body = {"result": "err", "detail": e}
                        logger.error(f"[PID {pid}][{task_id=}] {err_message}: {e}")
                        raise e
                    finally:
                        await bot_manager.write_log(
                            task_id=task_id,
                            user_id=user_id,
                            bot_id=bot_id,
                            log={
                                "WAS": "AI Service",
                                "STEP": "02. DB 대화이력 조회",
                                "BODY": log_body,
                            },
                        )

                # DB 사용자 질의 저장
                try:
                    logger.debug(f"[PID {pid}][{task_id=}] Saving user chat history")
                    chat_number: int = await db_manager.add_user_chat_history(
                        chat_session_id=chat_session_id,
                        message=message,
                    )
                    log_body = {"result": "success", "detail": None}
                except Exception as e:
                    err_message = "DB 사용자 질의 저장 ERR"
                    log_body = {"result": "err", "detail": e}
                    logger.error(f"[PID {pid}][{task_id=}] {err_message}: {e}")
                    raise e
                finally:
                    await bot_manager.write_log(
                        task_id=task_id,
                        user_id=user_id,
                        bot_id=bot_id,
                        log={
                            "WAS": "AI Service",
                            "STEP": "03. DB 사용자 질의 저장",
                            "BODY": log_body,
                        },
                    )

                # NAVERWORKS 답변생성중 메세지 발신
                try:
                    logger.debug(
                        f"[PID {pid}][{task_id=}] Sending generate answer message"
                    )

                    end_point = (
                        f"api/v1/stream/generate?chat_session_id={chat_session_id}"
                    )
                    stream_url = f"{settings.SERVICE_WAS_IP}:{settings.SERVICE_WAS_PORT}/{end_point}"
                    token = encode_token(
                        {
                            "chat_session_id": chat_session_id,
                            "chat_number": chat_number,
                            "stream_url": stream_url,
                        }
                    )
                    web_view_uri = f"{settings.WEB_VIEW_URL}?token={token}"

                    await bot_manager.send_wait_message(
                        bot_id=bot_id,
                        user_id=user_id,
                        uri=web_view_uri,
                    )

                    log_body = {"result": "success", "detail": None}
                except Exception as e:
                    err_message = "NAVERWORKS 답변생성중 메세지 발신 ERR"
                    log_body = {"result": "err", "detail": e}
                    logger.error(f"[PID {pid}][{task_id=}] {err_message}: {e}")
                    raise e
                finally:
                    await bot_manager.write_log(
                        task_id=task_id,
                        user_id=user_id,
                        bot_id=bot_id,
                        log={
                            "WAS": "AI Service",
                            "STEP": "04. NAVERWORKS 답변생성중 메세지 발신",
                            "BODY": log_body,
                        },
                    )

                # 의도분석 API 요청
                try:
                    logger.debug(
                        f"[PID {pid}][{task_id=}] Requesting intent analysis for message"
                    )
                    intent_response = await llm_studio.get_intent(
                        user_query=message,
                        dialog_history=get_dialog_history(
                            use_dialog=settings.USE_DIALOG_HISTORY,
                            total_cnt=settings.DIALOG_HISTORY_COUNT,
                            api_with_dialog=settings.DIALOG_USAGE_LIST[0],
                            dialog_cnt=settings.DIALOG_USAGE_COUNT[0],
                            dialog_history=dialog_history,
                        ),
                    )
                    log_body = {"result": "success", "detail": intent_response}
                except Exception as e:
                    err_message = "LLM WAS 의도분석 API 요청 ERR"
                    log_body = {"result": "err", "detail": e}
                    logger.error(f"[PID {pid}][{task_id=}] {err_message}: {e}")
                    raise e
                finally:
                    await bot_manager.write_log(
                        task_id=task_id,
                        user_id=user_id,
                        bot_id=bot_id,
                        log={
                            "WAS": "AI Service",
                            "STEP": "05_01. LLM WAS 의도분석 API 요청",
                            "BODY": log_body,
                        },
                    )

                # 의도분석 API 요청 결과 파싱
                try:
                    intent_info = await llm_studio.get_parsed_intent(
                        intent_response.get("llm_result").get("answer")
                    )
                    intent = intent_info.get("intent")
                    intent_no = int(intent.get("no"))
                    log_body = {"result": "success", "detail": intent}
                    logger.info(
                        f"[PID {pid}][{task_id=}] Intent analysis result: {intent}"
                    )
                except Exception as e:
                    err_message = "의도분석 API 요청 결과 파싱 ERR"
                    log_body = {"result": "err", "detail": e}
                    logger.error(f"[PID {pid}][{task_id=}] {err_message}: {e}")
                    raise e
                finally:
                    await bot_manager.write_log(
                        task_id=task_id,
                        user_id=user_id,
                        bot_id=bot_id,
                        log={
                            "WAS": "AI Service",
                            "STEP": "05_02. 의도분석 API 요청 결과 파싱",
                            "BODY": log_body,
                        },
                    )

                # RAG 검색여부 확인
                if intent_no in RAG_INTENT_CODES:
                    # 질의확장 API 요청
                    try:
                        logger.debug(f"[PID {pid}][{task_id=}] Requesting query expand")
                        expand_response = await llm_studio.get_expanded_query(
                            user_query=message,
                            dialog_history=get_dialog_history(
                                use_dialog=settings.USE_DIALOG_HISTORY,
                                total_cnt=settings.DIALOG_HISTORY_COUNT,
                                api_with_dialog=settings.DIALOG_USAGE_LIST[1],
                                dialog_cnt=settings.DIALOG_USAGE_COUNT[1],
                                dialog_history=dialog_history,
                            ),
                        )
                        log_body = {"result": "success", "detail": expand_response}
                    except Exception as e:
                        err_message = "질의확장 API 요청 ERR"
                        log_body = {"result": "err", "detail": e}
                        logger.error(f"[PID {pid}][{task_id=}] {err_message}: {e}")
                        raise e
                    finally:
                        await bot_manager.write_log(
                            task_id=task_id,
                            user_id=user_id,
                            bot_id=bot_id,
                            log={
                                "WAS": "AI Service",
                                "STEP": "06_01. LLM WAS 질의확장 API 요청",
                                "BODY": log_body,
                            },
                        )

                    # 질의확장 API 요청 결과 파싱
                    queries = []
                    try:
                        query_info = await llm_studio.get_parsed_query(
                            expand_response.get("llm_result").get("answer")
                        )
                        query_complete = query_info.get("query_complete")
                        search_queries = query_info.get("search_queries")
                        queries.append(query_complete)

                        for query in search_queries:
                            if query not in queries:  # 중복제거
                                queries.append(query)
                        log_body = {"result": "success", "detail": queries}
                        logger.info(
                            f"[PID {pid}][{task_id=}] Expanded query result: {queries}"
                        )
                    except Exception as e:
                        err_message = "질의확장 API 요청 결과 파싱 ERR"
                        log_body = {"result": "err", "detail": e}
                        logger.error(f"[PID {pid}][{task_id=}] {err_message}: {e}")
                        raise e
                    finally:
                        await bot_manager.write_log(
                            task_id=task_id,
                            user_id=user_id,
                            bot_id=bot_id,
                            log={
                                "WAS": "AI Service",
                                "STEP": "06_02. 질의확장 API 요청 결과 파싱",
                                "BODY": log_body,
                            },
                        )

                    try:
                        if queries:
                            logger.debug(
                                f"[PID {pid}][{task_id=}] Performing RAG document retrieval"
                            )
                            # 검색 조건 추가
                            filters = []
                            if settings.USE_SERVICE_FILTER:
                                ## LLM Studio 서비스 분류 API 요청 > service 제약조건 확인
                                try:
                                    service_response = await llm_studio.get_service_category(
                                        user_query=message,
                                        dialog_history=get_dialog_history(
                                            use_dialog=settings.USE_DIALOG_HISTORY,
                                            total_cnt=settings.DIALOG_HISTORY_COUNT,
                                            api_with_dialog=settings.DIALOG_USAGE_LIST[
                                                2
                                            ],
                                            dialog_cnt=settings.DIALOG_USAGE_COUNT[2],
                                            dialog_history=dialog_history,
                                        ),
                                    )
                                    log_body = {
                                        "result": "success",
                                        "detail": service_response,
                                    }
                                except Exception as e:
                                    err_message = "서비스 분류 API 요청 ERR"
                                    log_body = {"result": "err", "detail": e}
                                    logger.error(
                                        f"[PID {pid}][{task_id=}] {err_message}: {e}"
                                    )
                                    raise e
                                finally:
                                    await bot_manager.write_log(
                                        task_id=task_id,
                                        user_id=user_id,
                                        bot_id=bot_id,
                                        log={
                                            "WAS": "AI Service",
                                            "STEP": "06_03. 서비스 분류 API 요청",
                                            "BODY": log_body,
                                        },
                                    )

                                try:
                                    service_info = await llm_studio.get_parsed_service(
                                        service_response.get("llm_result").get("answer")
                                    )
                                    log_body = {
                                        "result": "success",
                                        "detail": service_info,
                                    }
                                except Exception as e:
                                    err_message = "서비스 분류 API 요청 결과 파싱 ERR"
                                    log_body = {"result": "err", "detail": e}
                                    logger.error(
                                        f"[PID {pid}][{task_id=}] {err_message}: {e}"
                                    )
                                    raise e
                                finally:
                                    await bot_manager.write_log(
                                        task_id=task_id,
                                        user_id=user_id,
                                        bot_id=bot_id,
                                        log={
                                            "WAS": "AI Service",
                                            "STEP": "06_04. 서비스 분류 API 요청 결과 파싱",
                                            "BODY": log_body,
                                        },
                                    )
                                service_filter: list = await get_service_filter(
                                    service_info.get("service", [])
                                )
                                if len(service_filter) > 0:
                                    filters.append(service_filter)

                            if settings.USE_GROUP_FILTER:
                                ## "질의 원문 + 정제된 질의" > group 제약조건 확인
                                group_filter: list = await get_group_filter(
                                    [message, queries[0]]
                                )
                                if len(group_filter) > 0:
                                    filters.append(group_filter)

                            # RAG 검색 API 요청
                            retrieve_response = await rag.retrieve_documents(
                                queries=queries,
                                filters=filters if filters else None,
                            )
                            retrieve_results = retrieve_response.get("results")

                            # Retrieval 결과를 llm api에 맞춰 변형
                            ref_data = await rag.get_ref_documents(
                                retrieve_results, settings.REF_LIMIT
                            )
                            if ref_data:
                                is_rag = True
                                rag_psgs = [
                                    {
                                        "id": k,
                                        "passage": (
                                            f"[파일명: {v.get('file')}]\n{v.get('parent_text')}"
                                        ),
                                    }
                                    for k, v in ref_data.items()
                                ]
                            log_body = {
                                "result": "success",
                                "detail": {
                                    "ref_docs": rag_psgs,
                                    "total_docs": retrieve_response,
                                },
                            }

                    except Exception as e:
                        err_message = "RAG 검색 API 요청 ERR"
                        log_body = {"result": "err", "detail": e}
                        logger.error(f"[PID {pid}][{task_id=}] {err_message}: {e}")
                        raise e
                    finally:
                        await bot_manager.write_log(
                            task_id=task_id,
                            user_id=user_id,
                            bot_id=bot_id,
                            log={
                                "WAS": "AI Service",
                                "STEP": "06_05. RAG 검색 API 요청",
                                "BODY": log_body,
                            },
                        )
                        await bot_manager.write_admin_log(
                            task_id=task_id,
                            user_id=user_id,
                            bot_id=bot_id,
                            log={
                                "AL-VDB-CHUNKS": [
                                    {"id": k, "parent_text": v.get("parent_text")}
                                    for k, v in ref_data.items()
                                ]
                            },
                        )
                # 검색결과 Table 포함된 경우 상태 확인 및 추가 API 요청
                # // TODO: 테이블 병합 알고리즘 필요
                # // TODO: 검색 결과에 따라 rag_psgs 확인

                # 최종답변 생성 API 요청
                try:
                    logger.debug(f"[PID {pid}][{task_id=}] Generating final response")
                    gen_ans_response = await llm_studio_gen_answer_apis[intent_no](
                        user_query=message,
                        is_rag=is_rag,
                        stream=True,
                        rag_psgs=rag_psgs,
                        chat_session_id=chat_session_id,
                        dialog_history=get_dialog_history(
                            use_dialog=settings.USE_DIALOG_HISTORY,
                            total_cnt=settings.DIALOG_HISTORY_COUNT,
                            api_with_dialog=settings.DIALOG_USAGE_LIST[3],
                            dialog_cnt=settings.DIALOG_USAGE_COUNT[3],
                            dialog_history=dialog_history,
                        ),
                    )
                    llm_result = gen_ans_response.get("llm_result")
                    final_message: str = llm_result.get("answer")  # 생성된 답변
                    logger.info(f"[PID {pid}][{task_id=}] LLM message generated")
                    log_body = {"result": "success", "detail": llm_result}
                except Exception as e:
                    err_message = "LLM WAS 최종답변 생성 API 요청 ERR"
                    log_body = {"result": "err", "detail": e}
                    logger.error(f"[PID {pid}][{task_id=}] {err_message}: {e}")
                    raise e
                finally:
                    await bot_manager.write_log(
                        task_id=task_id,
                        user_id=user_id,
                        bot_id=bot_id,
                        log={
                            "WAS": "AI Service",
                            "STEP": "07_01. LLM WAS 최종답변 생성 API 요청",
                            "BODY": log_body,
                        },
                    )

                # 참고자료있는 경우 추가
                try:
                    annotations = llm_result.get("annotations")
                    ref_prefix = "\n📌 참고링크"

                    # RAG 검색 청크를 사용한 경우
                    annotaion_info_list: list[dict] = []
                    if is_rag and rag_psgs and annotations:
                        for annotation in annotations:
                            rag_id: list[str] = annotation.get("rag_id")
                            for _id in rag_id:
                                file = ref_data[_id].get("file")
                                page_num = ref_data[_id].get("page_num")
                                url = ref_data[_id].get("download_url")
                                # 같은 파일의 경우 파일 제목 및 다운로드 버튼은 하나만 표시
                                for annotaion_info in annotaion_info_list:
                                    if annotaion_info["file"] == file:
                                        annotaion_info["page_nums"].append(page_num)
                                        annotaion_info["page_nums"].sort()  # 오름차순
                                        break
                                else:
                                    annotaion_info_list.append(
                                        {
                                            "file": file,
                                            "page_nums": [page_num],
                                            "url": url,
                                        }
                                    )
                        ref_texts = f"\n{ref_prefix}\n" + "\n".join(
                            [
                                f"ㆍ{annotaion_info['file']} (p.{', '.join(map(str, annotaion_info['page_nums']))})\n{annotaion_info['url']}"
                                for annotaion_info in annotaion_info_list
                            ]
                        )
                        final_message += ref_texts

                        logger.debug(
                            f"[PID {pid}][{task_id=}] Add references end of message"
                        )
                    log_body = {"result": "success", "detail": final_message}
                except Exception as e:
                    err_message = "참고자료있는 경우 추가 ERR"
                    log_body = {"result": "err", "detail": e}
                    logger.error(f"[PID {pid}][{task_id=}] {err_message}: {e}")
                    raise e
                finally:
                    await bot_manager.write_log(
                        task_id=task_id,
                        user_id=user_id,
                        bot_id=bot_id,
                        log={
                            "WAS": "AI Service",
                            "STEP": "07_02. 참고자료있는 경우 추가",
                            "BODY": log_body,
                        },
                    )

                # DB 최종답변 저장
                try:
                    logger.debug(f"[PID {pid}][{task_id=}] Saving final message")
                    await db_manager.add_bot_chat_history(
                        chat_session_id=chat_session_id,
                        chat_number=chat_number,
                        message=final_message,
                    )
                    log_body = {"result": "success", "detail": None}
                except Exception as e:
                    err_message = "DB 최종답변 저장 ERR"
                    log_body = {"result": "err", "detail": e}
                    logger.error(f"[PID {pid}][{task_id=}] {err_message}: {e}")
                    raise e
                finally:
                    await bot_manager.write_log(
                        task_id=task_id,
                        user_id=user_id,
                        bot_id=bot_id,
                        log={
                            "WAS": "AI Service",
                            "STEP": "08. DB 최종답변 저장",
                            "BODY": log_body,
                        },
                    )

                # NAVERWORKS 최종답변 메세지 발신
                try:
                    await bot_manager.send_bot_answer_message(
                        bot_id=bot_id,
                        user_id=user_id,
                        bot_message=final_message,
                        uri=web_view_uri,
                    )
                    logger.debug(
                        f"[PID {pid}][{task_id=}] Sending final message to user"
                    )

                    ## 참조링크 다운로드 버튼 전송
                    for download_url in annotaion_info_list:
                        await bot_manager.send_file_download_button_ui_message(
                            bot_id=bot_id,
                            user_id=user_id,
                            file_name=download_url.get("file"),
                            uri=download_url.get("url"),
                        )
                        logger.debug(
                            f"[PID {pid}][{task_id=}] Sending download button to user"
                        )
                    log_body = {"result": "success", "detail": None}
                except Exception as e:
                    err_message = "NAVERWORKS 최종답변 메세지 발신 ERR"
                    log_body = {"result": "err", "detail": e}
                    logger.error(f"[PID {pid}][{task_id=}] {err_message}: {e}")
                    raise e
                finally:
                    await bot_manager.write_log(
                        task_id=task_id,
                        user_id=user_id,
                        bot_id=bot_id,
                        log={
                            "WAS": "AI Service",
                            "STEP": "09. NAVERWORKS 최종답변 메세지 발신",
                            "BODY": log_body,
                        },
                    )
                    await bot_manager.write_admin_log(
                        task_id=task_id,
                        user_id=user_id,
                        bot_id=bot_id,
                        log={"AL-BOT-MESSAGE": final_message},
                    )
            else:  # file 업로드인 경우
                logger.info(f"[PID {pid}][{task_id=}] Handling file upload task")

        except Exception as e:
            status = "failed"
            logger.error(f"[Error - handle_task][{task_id=}]: {e}", exc_info=True)

            # "사용자 + 관리자" NAVER WORKS 오류 메시지 전송
            error_message = (
                "🚨 답변 생성중 오류가 발생했습니다. 관리자에게 문의해주세요."
            )
            await bot_manager.send_error_message(
                bot_id=bot_id,
                user_id=user_id,
                error_message=error_message,
                task_id=task_id,
            )
        finally:
            logger.error(f">>> [PID {pid}][{task_id=}] Worker # {worker_id} finished")
            await task_client.update_task(task_id=task_id, payload={"status": status})
            queue.task_done()
