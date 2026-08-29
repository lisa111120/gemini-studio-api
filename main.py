import os
import time
import json
import asyncio
import itertools
from typing import List, Optional, Any

import httpx

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Gemini REST OpenAI Compatible Router"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# =========================================================
# API KEYS
# =========================================================

keys_env = os.environ.get(
    "GEMINI_API_KEYS",
    ""
)

API_KEYS = [
    key.strip()
    for key in keys_env.split(",")
    if key.strip()
]


key_cycle = (
    itertools.cycle(API_KEYS)
    if API_KEYS
    else None
)


def get_next_key():

    if not key_cycle:

        raise HTTPException(
            status_code=500,
            detail="请在环境变量中设置 GEMINI_API_KEYS"
        )

    return next(key_cycle)


# =========================================================
# GEMINI
# =========================================================

GEMINI_BASE_URL = (
    "https://generativelanguage.googleapis.com"
)

GEMINI_API_VERSION = "v1beta"


# =========================================================
# REQUEST MODELS
# =========================================================

class ChatMessage(BaseModel):

    model_config = ConfigDict(
        extra="allow"
    )

    role: str

    content: Any = None

    name: Optional[str] = None

    tool_call_id: Optional[str] = None

    tool_calls: Optional[List[Any]] = None


class ChatRequest(BaseModel):

    model_config = ConfigDict(
        extra="allow"
    )

    # =====================================================
    # MODEL 完全透传
    # =====================================================

    model: str

    messages: List[ChatMessage]

    stream: Optional[bool] = True


    # =====================================================
    # 默认 HIGH
    # =====================================================

    thinking_level: Optional[str] = "high"


    # =====================================================
    # OpenAI 常见参数
    # =====================================================

    temperature: Optional[float] = None

    top_p: Optional[float] = None

    top_k: Optional[int] = None

    max_tokens: Optional[int] = None

    max_completion_tokens: Optional[int] = None

    stop: Optional[Any] = None

    seed: Optional[int] = None


# =========================================================
# CONTENT
# =========================================================

def extract_text_content(
    content: Any
) -> str:

    if content is None:
        return ""

    if isinstance(
        content,
        str
    ):
        return content


    if isinstance(
        content,
        list
    ):

        text_parts = []

        for item in content:

            if not isinstance(
                item,
                dict
            ):
                continue


            item_type = item.get(
                "type"
            )


            if item_type in (
                "text",
                "input_text",
            ):

                text_parts.append(
                    str(
                        item.get(
                            "text",
                            ""
                        )
                    )
                )


        return "\n".join(
            text_parts
        )


    return str(content)


# =========================================================
# OPENAI MESSAGES → GEMINI REST
# =========================================================

def transform_messages(
    messages: List[ChatMessage]
):

    system_parts = []

    contents = []


    for msg in messages:

        role = msg.role

        text = extract_text_content(
            msg.content
        )


        # =================================================
        # SYSTEM / DEVELOPER
        # =================================================

        if role in (
            "system",
            "developer",
        ):

            if text:
                system_parts.append(
                    text
                )

            continue


        # =================================================
        # USER
        # =================================================

        if role == "user":

            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": text
                        }
                    ]
                }
            )

            continue


        # =================================================
        # ASSISTANT
        # =================================================

        if role == "assistant":

            contents.append(
                {
                    "role": "model",
                    "parts": [
                        {
                            "text": text
                        }
                    ]
                }
            )

            continue


        # =================================================
        # TOOL / UNKNOWN
        #
        # 暂时按 user 文本传递
        # =================================================

        contents.append(
            {
                "role": "user",
                "parts": [
                    {
                        "text": text
                    }
                ]
            }
        )


    # Gemini 要求最终最好有 user 内容
    if not contents:

        contents.append(
            {
                "role": "user",
                "parts": [
                    {
                        "text": " "
                    }
                ]
            }
        )


    system_instruction = None


    if system_parts:

        system_instruction = {
            "parts": [
                {
                    "text":
                        "\n\n".join(
                            system_parts
                        )
                }
            ]
        }


    return (
        system_instruction,
        contents,
    )


# =========================================================
# GEMINI BODY
# =========================================================

def build_gemini_body(
    req: ChatRequest
):

    (
        system_instruction,
        contents,
    ) = transform_messages(
        req.messages
    )


    generation_config = {}


    # =====================================================
    # THINKING
    # =====================================================

    thinking_level = (
        req.thinking_level
        or "high"
    )


    if thinking_level not in (
        "low",
        "medium",
        "high",
    ):

        thinking_level = "high"


    # Gemini 3.x
    generation_config[
        "thinkingConfig"
    ] = {
        "thinkingLevel":
            thinking_level
    }


    # =====================================================
    # MAX TOKENS
    # =====================================================

    max_tokens = None


    if (
        req.max_completion_tokens
        is not None
    ):

        max_tokens = (
            req.max_completion_tokens
        )

    elif req.max_tokens is not None:

        max_tokens = (
            req.max_tokens
        )


    if max_tokens is not None:

        generation_config[
            "maxOutputTokens"
        ] = max_tokens


    # =====================================================
    # STOP
    # =====================================================

    if req.stop is not None:

        if isinstance(
            req.stop,
            str
        ):

            generation_config[
                "stopSequences"
            ] = [
                req.stop
            ]

        elif isinstance(
            req.stop,
            list
        ):

            generation_config[
                "stopSequences"
            ] = req.stop


    # =====================================================
    # NOTE
    #
    # Gemini 3.7 官方迁移文档建议移除
    # temperature / top_p / top_k。
    #
    # 所以这里故意不传。
    # =====================================================


    body = {

        "contents":
            contents,

        "generationConfig":
            generation_config,

        "safetySettings":
            [
                {
                    "category":
                        "HARM_CATEGORY_HATE_SPEECH",

                    "threshold":
                        "BLOCK_NONE",
                },
                {
                    "category":
                        "HARM_CATEGORY_HARASSMENT",

                    "threshold":
                        "BLOCK_NONE",
                },
                {
                    "category":
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT",

                    "threshold":
                        "BLOCK_NONE",
                },
                {
                    "category":
                        "HARM_CATEGORY_DANGEROUS_CONTENT",

                    "threshold":
                        "BLOCK_NONE",
                },
            ],
    }


    if system_instruction:

        body[
            "systemInstruction"
        ] = system_instruction


    return body


# =========================================================
# HELPERS
# =========================================================

def make_completion_id():

    return (
        "chatcmpl-gemini-"
        + str(
            int(
                time.time()
                * 1000
            )
        )
    )


def make_sse(
    payload: dict
):

    return (
        "data: "
        + json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n\n"
    )


def extract_gemini_text(
    data: dict
) -> str:

    text_parts = []


    candidates = data.get(
        "candidates"
    ) or []


    for candidate in candidates:

        content = (
            candidate.get(
                "content"
            )
            or {}
        )

        parts = (
            content.get(
                "parts"
            )
            or []
        )


        for part in parts:

            text = part.get(
                "text"
            )

            if text:

                text_parts.append(
                    text
                )


    return "".join(
        text_parts
    )


def extract_finish_reason(
    data: dict
):

    candidates = (
        data.get(
            "candidates"
        )
        or []
    )


    if not candidates:
        return None


    reason = candidates[0].get(
        "finishReason"
    )


    mapping = {
        "STOP":
            "stop",

        "MAX_TOKENS":
            "length",

        "SAFETY":
            "content_filter",

        "RECITATION":
            "content_filter",
    }


    return mapping.get(
        reason,
        None
    )


# =========================================================
# ROOT
# =========================================================

@app.get("/")
async def root():

    return {

        "status":
            "running",

        "service":
            "Gemini REST OpenAI Compatible Router",

        "loaded_keys_count":
            len(API_KEYS),

        "model_mode":
            "passthrough",

        "default_thinking_level":
            "high",

        "backend":
            "Gemini REST",
    }


# =========================================================
# OPTIONS
# =========================================================

@app.options("/{path:path}")
async def options_handler(
    path: str
):

    return Response(
        status_code=204,
        headers={
            "Access-Control-Allow-Origin":
                "*",

            "Access-Control-Allow-Methods":
                "*",

            "Access-Control-Allow-Headers":
                "*",
        }
    )


# =========================================================
# MODELS
# =========================================================

@app.get("/models")
@app.get("/v1/models")
async def models():

    if not API_KEYS:

        raise HTTPException(
            status_code=500,
            detail="未配置 GEMINI_API_KEYS"
        )


    last_error = None


    for attempt in range(
        len(API_KEYS)
    ):

        api_key = get_next_key()


        try:

            url = (
                f"{GEMINI_BASE_URL}/"
                f"{GEMINI_API_VERSION}/"
                f"models"
            )


            timeout = httpx.Timeout(
                30.0
            )


            async with httpx.AsyncClient(
                timeout=timeout
            ) as client:


                response = (
                    await client.get(
                        url,
                        headers={
                            "x-goog-api-key":
                                api_key
                        },
                    )
                )


            if not response.is_success:

                raise RuntimeError(
                    f"Gemini models HTTP "
                    f"{response.status_code}: "
                    f"{response.text}"
                )


            data = response.json()


            result = []


            for model in (
                data.get(
                    "models"
                )
                or []
            ):

                name = model.get(
                    "name",
                    ""
                )


                if name.startswith(
                    "models/"
                ):

                    name = name[7:]


                if not name:
                    continue


                result.append(
                    {
                        "id":
                            name,

                        "object":
                            "model",

                        "created":
                            0,

                        "owned_by":
                            "google",
                    }
                )


            return {
                "object":
                    "list",

                "data":
                    result,
            }


        except Exception as e:

            last_error = e


            print(
                "[Gemini Models Error] "
                f"attempt={attempt + 1}/"
                f"{len(API_KEYS)} "
                f"error={repr(e)}"
            )


            if (
                attempt
                < len(API_KEYS) - 1
            ):

                await asyncio.sleep(
                    0.5
                )


    raise HTTPException(
        status_code=503,
        detail={
            "message":
                "模型列表请求失败",

            "error":
                str(last_error),
        }
    )


# =========================================================
# CHAT COMPLETIONS
# =========================================================

@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatRequest
):


    if not API_KEYS:

        raise HTTPException(
            status_code=500,
            detail="未配置 GEMINI_API_KEYS"
        )


    # =====================================================
    # MODEL 完全透传
    # =====================================================

    target_model = req.model


    body = build_gemini_body(
        req
    )


    # =====================================================
    # STREAM
    # =====================================================

    if req.stream:


        async def event_stream():


            completion_id = (
                make_completion_id()
            )


            created = int(
                time.time()
            )


            last_error = None


            for attempt in range(
                len(API_KEYS)
            ):


                api_key = get_next_key()


                got_any_gemini_data = False

                sent_openai_start = False


                try:


                    # =========================================
                    # model 原样塞进 URL
                    # =========================================

                    url = (
                        f"{GEMINI_BASE_URL}/"
                        f"{GEMINI_API_VERSION}/"
                        f"models/{target_model}:"
                        f"streamGenerateContent"
                        f"?alt=sse"
                    )


                    print(
                        "[Gemini REST Start] "
                        f"model={target_model} "
                        f"attempt={attempt + 1}/"
                        f"{len(API_KEYS)}"
                    )


                    timeout = httpx.Timeout(
                        connect=20.0,

                        read=None,

                        write=30.0,

                        pool=20.0,
                    )


                    async with httpx.AsyncClient(
                        timeout=timeout
                    ) as client:


                        async with client.stream(

                            "POST",

                            url,

                            headers={
                                "x-goog-api-key":
                                    api_key,

                                "Content-Type":
                                    "application/json",

                                "Accept":
                                    "text/event-stream",
                            },

                            json=body,

                        ) as response:


                            print(
                                "[Gemini REST HTTP] "
                                f"model={target_model} "
                                f"status="
                                f"{response.status_code}"
                            )


                            # =================================
                            # Google HTTP ERROR
                            # =================================

                            if not response.is_success:


                                raw_error = (
                                    await response.aread()
                                )


                                error_text = (
                                    raw_error.decode(
                                        "utf-8",
                                        errors="replace"
                                    )
                                )


                                raise RuntimeError(
                                    f"Gemini HTTP "
                                    f"{response.status_code}: "
                                    f"{error_text}"
                                )


                            # =================================
                            # GEMINI SSE
                            # =================================

                            async for line in (
                                response.aiter_lines()
                            ):


                                if not line:
                                    continue


                                # SSE comment
                                if line.startswith(
                                    ":"
                                ):
                                    continue


                                if not line.startswith(
                                    "data:"
                                ):
                                    continue


                                raw = (
                                    line[5:]
                                    .strip()
                                )


                                if not raw:
                                    continue


                                try:

                                    data = (
                                        json.loads(
                                            raw
                                        )
                                    )

                                except Exception:

                                    print(
                                        "[Gemini SSE Parse Skip] "
                                        f"{raw[:500]}"
                                    )

                                    continue


                                got_any_gemini_data = True


                                # =============================
                                # 第一次 Gemini 真正返回数据
                                #
                                # 再给 Chatbox 发 assistant
                                # =============================

                                if not sent_openai_start:


                                    first_payload = {

                                        "id":
                                            completion_id,

                                        "object":
                                            "chat.completion.chunk",

                                        "created":
                                            created,

                                        "model":
                                            target_model,

                                        "choices":
                                            [
                                                {
                                                    "index":
                                                        0,

                                                    "delta":
                                                        {
                                                            "role":
                                                                "assistant",

                                                            "content":
                                                                "",
                                                        },

                                                    "finish_reason":
                                                        None,
                                                }
                                            ],
                                    }


                                    yield make_sse(
                                        first_payload
                                    )


                                    sent_openai_start = True


                                # =============================
                                # TEXT
                                # =============================

                                text = (
                                    extract_gemini_text(
                                        data
                                    )
                                )


                                if text:


                                    payload = {

                                        "id":
                                            completion_id,

                                        "object":
                                            "chat.completion.chunk",

                                        "created":
                                            created,

                                        "model":
                                            target_model,

                                        "choices":
                                            [
                                                {
                                                    "index":
                                                        0,

                                                    "delta":
                                                        {
                                                            "content":
                                                                text
                                                        },

                                                    "finish_reason":
                                                        None,
                                                }
                                            ],
                                    }


                                    yield make_sse(
                                        payload
                                    )


                                # =============================
                                # FINISH
                                # =============================

                                finish_reason = (
                                    extract_finish_reason(
                                        data
                                    )
                                )


                                if finish_reason:


                                    finish_payload = {

                                        "id":
                                            completion_id,

                                        "object":
                                            "chat.completion.chunk",

                                        "created":
                                            created,

                                        "model":
                                            target_model,

                                        "choices":
                                            [
                                                {
                                                    "index":
                                                        0,

                                                    "delta":
                                                        {},

                                                    "finish_reason":
                                                        finish_reason,
                                                }
                                            ],
                                    }


                                    yield make_sse(
                                        finish_payload
                                    )


                                    yield (
                                        "data: [DONE]\n\n"
                                    )


                                    print(
                                        "[Gemini REST Done] "
                                        f"model={target_model}"
                                    )


                                    return


                            # =================================
                            # SSE 正常关闭，但没有 finish
                            # =================================

                            if got_any_gemini_data:


                                if not sent_openai_start:


                                    first_payload = {

                                        "id":
                                            completion_id,

                                        "object":
                                            "chat.completion.chunk",

                                        "created":
                                            created,

                                        "model":
                                            target_model,

                                        "choices":
                                            [
                                                {
                                                    "index":
                                                        0,

                                                    "delta":
                                                        {
                                                            "role":
                                                                "assistant",

                                                            "content":
                                                                "",
                                                        },

                                                    "finish_reason":
                                                        None,
                                                }
                                            ],
                                    }


                                    yield make_sse(
                                        first_payload
                                    )


                                finish_payload = {

                                    "id":
                                        completion_id,

                                    "object":
                                        "chat.completion.chunk",

                                    "created":
                                        created,

                                    "model":
                                        target_model,

                                    "choices":
                                        [
                                            {
                                                "index":
                                                    0,

                                                "delta":
                                                    {},

                                                "finish_reason":
                                                    "stop",
                                            }
                                        ],
                                }


                                yield make_sse(
                                    finish_payload
                                )


                                yield (
                                    "data: [DONE]\n\n"
                                )


                                return


                            raise RuntimeError(
                                "Gemini SSE 连接已关闭，"
                                "但没有返回任何数据"
                            )


                except Exception as e:


                    last_error = e


                    print(
                        "[Gemini REST Stream Error] "
                        f"model={target_model} "
                        f"attempt={attempt + 1}/"
                        f"{len(API_KEYS)} "
                        f"error_type={type(e).__name__} "
                        f"error={repr(e)}"
                    )


                    # =========================================
                    # 一个 Gemini 数据都没收到
                    #
                    # 可以切换下一个 Key
                    # =========================================

                    if (
                        not got_any_gemini_data
                        and
                        attempt
                        < len(API_KEYS) - 1
                    ):

                        await asyncio.sleep(
                            0.5
                        )

                        continue


                    # =========================================
                    # 已经向用户输出过了
                    #
                    # 不再重试
                    # =========================================

                    if sent_openai_start:


                        error_payload = {

                            "id":
                                completion_id,

                            "object":
                                "chat.completion.chunk",

                            "created":
                                created,

                            "model":
                                target_model,

                            "choices":
                                [
                                    {
                                        "index":
                                            0,

                                        "delta":
                                            {
                                                "content":
                                                    (
                                                        "\n\n"
                                                        "[Gemini 连接中断: "
                                                        f"{str(e)}]"
                                                    )
                                            },

                                        "finish_reason":
                                            None,
                                    }
                                ],
                        }


                        yield make_sse(
                            error_payload
                        )


                    break


            # =================================================
            # ALL KEYS FAILED
            # =================================================

            if not sent_openai_start:


                first_payload = {

                    "id":
                        completion_id,

                    "object":
                        "chat.completion.chunk",

                    "created":
                        created,

                    "model":
                        target_model,

                    "choices":
                        [
                            {
                                "index":
                                    0,

                                "delta":
                                    {
                                        "role":
                                            "assistant",

                                        "content":
                                            "",
                                    },

                                "finish_reason":
                                    None,
                            }
                        ],
                }


                yield make_sse(
                    first_payload
                )


            error_payload = {

                "id":
                    completion_id,

                "object":
                    "chat.completion.chunk",

                "created":
                    created,

                "model":
                    target_model,

                "choices":
                    [
                        {
                            "index":
                                0,

                            "delta":
                                {
                                    "content":
                                        (
                                            "[Gemini 请求失败: "
                                            f"{str(last_error)}]"
                                        )
                                },

                            "finish_reason":
                                None,
                        }
                    ],
            }


            yield make_sse(
                error_payload
            )


            finish_payload = {

                "id":
                    completion_id,

                "object":
                    "chat.completion.chunk",

                "created":
                    created,

                "model":
                    target_model,

                "choices":
                    [
                        {
                            "index":
                                0,

                            "delta":
                                {},

                            "finish_reason":
                                "stop",
                        }
                    ],
            }


            yield make_sse(
                finish_payload
            )


            yield (
                "data: [DONE]\n\n"
            )


        return StreamingResponse(

            event_stream(),

            media_type=(
                "text/event-stream"
            ),

            headers={

                "Cache-Control":
                    "no-cache, no-transform",

                "Connection":
                    "keep-alive",

                "X-Accel-Buffering":
                    "no",

                "Access-Control-Allow-Origin":
                    "*",
            },
        )


    # =====================================================
    # NON STREAM
    # =====================================================

    last_error = None


    for attempt in range(
        len(API_KEYS)
    ):


        api_key = get_next_key()


        try:


            url = (
                f"{GEMINI_BASE_URL}/"
                f"{GEMINI_API_VERSION}/"
                f"models/{target_model}:"
                f"generateContent"
            )


            timeout = httpx.Timeout(
                120.0
            )


            async with httpx.AsyncClient(
                timeout=timeout
            ) as client:


                response = (
                    await client.post(

                        url,

                        headers={
                            "x-goog-api-key":
                                api_key,

                            "Content-Type":
                                "application/json",
                        },

                        json=body,
                    )
                )


            if not response.is_success:

                raise RuntimeError(
                    f"Gemini HTTP "
                    f"{response.status_code}: "
                    f"{response.text}"
                )


            data = response.json()


            text = extract_gemini_text(
                data
            )


            finish_reason = (
                extract_finish_reason(
                    data
                )
                or "stop"
            )


            return JSONResponse(
                content={

                    "id":
                        make_completion_id(),

                    "object":
                        "chat.completion",

                    "created":
                        int(
                            time.time()
                        ),

                    "model":
                        target_model,

                    "choices":
                        [
                            {
                                "index":
                                    0,

                                "message":
                                    {
                                        "role":
                                            "assistant",

                                        "content":
                                            text,
                                    },

                                "finish_reason":
                                    finish_reason,
                            }
                        ],
                }
            )


        except Exception as e:


            last_error = e


            print(
                "[Gemini REST Error] "
                f"model={target_model} "
                f"attempt={attempt + 1}/"
                f"{len(API_KEYS)} "
                f"error_type={type(e).__name__} "
                f"error={repr(e)}"
            )


            if (
                attempt
                < len(API_KEYS) - 1
            ):

                await asyncio.sleep(
                    0.5
                )


    raise HTTPException(

        status_code=503,

        detail={

            "message":
                "所有 Gemini API Key 请求均失败",

            "model":
                target_model,

            "error":
                str(
                    last_error
                ),
        }
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    import uvicorn


    port = int(
        os.environ.get(
            "PORT",
            8080
        )
    )


    uvicorn.run(

        app,

        host="0.0.0.0",

        port=port,
    )
