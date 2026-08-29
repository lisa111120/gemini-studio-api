import os
import time
import json
import itertools
import asyncio
from typing import List, Optional, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

from google import genai
from google.genai import types


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Gemini OpenAI Compatible Router"
)


# =========================================================
# CORS
#
# 对齐你 Cloudflare Worker 的行为。
# 手机端 Chatbox / WebView 需要这个。
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


CLIENT_POOL = [
    genai.Client(api_key=key)
    for key in API_KEYS
]


client_cycle = (
    itertools.cycle(CLIENT_POOL)
    if CLIENT_POOL
    else None
)


def get_next_client():

    if not client_cycle:

        raise HTTPException(
            status_code=500,
            detail="请在环境变量中设置 GEMINI_API_KEYS"
        )

    return next(client_cycle)


# =========================================================
# SAFETY
#
# 与你原 Cloudflare Worker 的 BLOCK_NONE 行为保持一致
# =========================================================

GLOBAL_SAFETY_SETTINGS = [

    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),

    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),

    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),

    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
]


# =========================================================
# OPENAI REQUEST MODELS
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
    # MODEL 必填
    #
    # Chatbox 传什么，就原样使用什么。
    #
    # 不映射
    # 不替换
    # 不白名单
    # 不 DEFAULT_MODEL
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

    presence_penalty: Optional[float] = None

    frequency_penalty: Optional[float] = None

    n: Optional[int] = None


# =========================================================
# MESSAGE CONTENT
# =========================================================

def extract_text_content(content: Any) -> str:

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if not isinstance(item, dict):
                continue

            item_type = item.get("type")


            # OpenAI text
            if item_type == "text":

                text_parts.append(
                    str(
                        item.get(
                            "text",
                            ""
                        )
                    )
                )


            # 某些客户端可能使用 input_text
            elif item_type == "input_text":

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
# OPENAI MESSAGES → GEMINI
#
# 对齐你的 CF Worker：
#
# system    → system_instruction
# user      → user
# assistant → model
#
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
        # SYSTEM
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
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text=text
                        )
                    ]
                )
            )

            continue


        # =================================================
        # ASSISTANT
        # =================================================

        if role == "assistant":

            contents.append(
                types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=text
                        )
                    ]
                )
            )

            continue


        # =================================================
        # TOOL
        #
        # 当前先作为文本继续传递，
        # 避免直接让整个请求 422。
        # =================================================

        if role == "tool":

            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text=text
                        )
                    ]
                )
            )

            continue


        # =================================================
        # UNKNOWN ROLE
        # =================================================

        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=text
                    )
                ]
            )
        )


    system_instruction = None


    if system_parts:

        system_instruction = types.Content(
            role="user",
            parts=[
                types.Part(
                    text="\n\n".join(
                        system_parts
                    )
                )
            ]
        )


    # Gemini 某些情况下不能只有 system_instruction
    if not contents:

        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=" "
                    )
                ]
            )
        )


    return (
        system_instruction,
        contents,
    )


# =========================================================
# GENERATION CONFIG
# =========================================================

def build_config(
    req: ChatRequest,
    system_instruction=None,
):

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


    thinking_config = types.ThinkingConfig(
        thinking_level=thinking_level
    )


    # =====================================================
    # CONFIG
    # =====================================================

    config_args = {

        "safety_settings":
            GLOBAL_SAFETY_SETTINGS,

        "thinking_config":
            thinking_config,
    }


    # =====================================================
    # SYSTEM
    # =====================================================

    if system_instruction is not None:

        config_args[
            "system_instruction"
        ] = system_instruction


    # =====================================================
    # TEMPERATURE
    # =====================================================

    if req.temperature is not None:

        config_args[
            "temperature"
        ] = req.temperature


    # =====================================================
    # TOP P
    # =====================================================

    if req.top_p is not None:

        config_args[
            "top_p"
        ] = req.top_p


    # =====================================================
    # TOP K
    # =====================================================

    if req.top_k is not None:

        config_args[
            "top_k"
        ] = req.top_k


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

        config_args[
            "max_output_tokens"
        ] = max_tokens


    # =====================================================
    # STOP
    # =====================================================

    if req.stop is not None:

        if isinstance(
            req.stop,
            str
        ):

            config_args[
                "stop_sequences"
            ] = [
                req.stop
            ]

        elif isinstance(
            req.stop,
            list
        ):

            config_args[
                "stop_sequences"
            ] = req.stop


    # =====================================================
    # SEED
    # =====================================================

    if req.seed is not None:

        config_args[
            "seed"
        ] = req.seed


    return types.GenerateContentConfig(
        **config_args
    )


# =========================================================
# OPENAI IDS
# =========================================================

def make_completion_id():

    return (
        "chatcmpl-gemini-"
        + str(
            int(
                time.time() * 1000
            )
        )
    )


# =========================================================
# SSE HELPER
# =========================================================

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


# =========================================================
# ROOT
# =========================================================

@app.get("/")
async def root():

    return {

        "status":
            "running",

        "service":
            "Gemini OpenAI Compatible Router",

        "loaded_keys_count":
            len(CLIENT_POOL),

        "model_mode":
            "passthrough",

        "default_thinking_level":
            "high",
    }


# =========================================================
# OPTIONS
#
# FastAPI CORS middleware 已经会处理，
# 这里额外留一个兜底。
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
#
# 同时支持：
#
# /models
# /v1/models
#
# 从 Google Gemini 获取真实列表。
# =========================================================

@app.get("/models")
@app.get("/v1/models")
async def models():

    if not CLIENT_POOL:

        raise HTTPException(
            status_code=500,
            detail="未配置 GEMINI_API_KEYS"
        )


    last_error = None

    retry_count = len(
        CLIENT_POOL
    )


    for attempt in range(
        retry_count
    ):

        current_client = (
            get_next_client()
        )


        try:

            model_items = []


            pager = (
                await current_client.aio.models.list()
            )


            async for model in pager:

                model_name = getattr(
                    model,
                    "name",
                    None
                )


                if not model_name:
                    continue


                if model_name.startswith(
                    "models/"
                ):

                    model_name = (
                        model_name[7:]
                    )


                model_items.append(
                    {
                        "id":
                            model_name,

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
                    model_items,
            }


        except Exception as e:

            last_error = e


            print(
                "[Gemini Models Error] "
                f"attempt={attempt + 1}/{retry_count} "
                f"error_type={type(e).__name__} "
                f"error={repr(e)}"
            )


            if (
                attempt
                < retry_count - 1
            ):

                await asyncio.sleep(
                    0.5
                )


    raise HTTPException(
        status_code=503,
        detail={
            "message":
                "Gemini 模型列表请求失败",

            "error":
                str(last_error),
        },
    )


# =========================================================
# CHAT COMPLETIONS
#
# 同时支持：
#
# /chat/completions
# /v1/chat/completions
#
# =========================================================

@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatRequest
):


    # =====================================================
    # KEY CHECK
    # =====================================================

    if not CLIENT_POOL:

        raise HTTPException(
            status_code=500,
            detail="未配置 GEMINI_API_KEYS"
        )


    # =====================================================
    # MODEL 完全透传
    #
    # 这是最重要的要求：
    #
    # Chatbox:
    # gemini-3.6-flash
    #
    # ↓
    #
    # Gemini SDK:
    # model="gemini-3.6-flash"
    #
    # 不进行任何模型名转换。
    # =====================================================

    target_model = req.model


    # =====================================================
    # MESSAGES
    # =====================================================

    (
        system_instruction,
        contents,
    ) = transform_messages(
        req.messages
    )


    # =====================================================
    # CONFIG
    # =====================================================

    config = build_config(
        req,
        system_instruction,
    )


    retry_count = len(
        CLIENT_POOL
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


            # =================================================
            # IMPORTANT
            #
            # 不在这里提前发送 role chunk。
            #
            # 跟你 Cloudflare Worker 一样：
            # 等 Gemini 真正开始成功返回之后，
            # 再发送第一帧 role。
            #
            # 这样如果第一个 Key 连接失败，
            # 仍然可以安全切 Key。
            # =================================================


            for attempt in range(
                retry_count
            ):


                current_client = (
                    get_next_client()
                )


                gemini_started = False

                openai_started = False


                try:


                    response_stream = (
                        await current_client.aio.models.generate_content_stream(
                            model=target_model,
                            contents=contents,
                            config=config,
                        )
                    )


                    async for chunk in response_stream:


                        text = getattr(
                            chunk,
                            "text",
                            None
                        )


                        # Gemini 已经真实响应
                        gemini_started = True


                        # =====================================
                        # OpenAI 第一帧
                        #
                        # 对齐你的 CF Worker：
                        #
                        # role=assistant
                        # content=""
                        # =====================================

                        if not openai_started:


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


                            openai_started = True


                        # =====================================
                        # TEXT
                        # =====================================

                        if not text:
                            continue


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


                    # =================================================
                    # Gemini 正常完成，但可能一个文本 chunk 都没有
                    # =================================================

                    if not openai_started:


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


                    # =================================================
                    # FINISH
                    # =================================================

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


                except Exception as e:


                    last_error = e


                    print(
                        "[Gemini Stream Error] "
                        f"model={target_model} "
                        f"attempt={attempt + 1}/{retry_count} "
                        f"error_type={type(e).__name__} "
                        f"error={repr(e)}"
                    )


                    # =================================================
                    # Gemini 还没真正开始返回
                    #
                    # → 可以安全换 Key
                    # =================================================

                    if not gemini_started:


                        if (
                            attempt
                            < retry_count - 1
                        ):

                            await asyncio.sleep(
                                0.5
                            )

                            continue


                        # 所有 Key 都失败
                        break


                    # =================================================
                    # 已经开始生成
                    #
                    # 不能重新请求别的 Key，
                    # 否则回答会从头重复。
                    # =================================================

                    if not openai_started:


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
                                                    "\n\n"
                                                    "[Gemini 流式连接中断: "
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


            # =================================================
            # 所有 Key 在开始生成前全部失败
            #
            # SSE 已经建立，所以不能再改 HTTP 状态。
            # 输出 OpenAI 风格结束。
            # =================================================


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


        # =================================================
        # STREAM RESPONSE
        # =================================================

        return StreamingResponse(

            event_stream(),

            media_type="text/event-stream",

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
        retry_count
    ):


        current_client = (
            get_next_client()
        )


        try:


            response = (
                await current_client.aio.models.generate_content(
                    model=target_model,
                    contents=contents,
                    config=config,
                )
            )


            response_text = (
                getattr(
                    response,
                    "text",
                    None
                )
                or ""
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
                                            response_text,
                                    },

                                "finish_reason":
                                    "stop",
                            }
                        ],
                },

                headers={
                    "Access-Control-Allow-Origin":
                        "*"
                }
            )


        except Exception as e:


            last_error = e


            print(
                "[Gemini Error] "
                f"model={target_model} "
                f"attempt={attempt + 1}/{retry_count} "
                f"error_type={type(e).__name__} "
                f"error={repr(e)}"
            )


            if (
                attempt
                < retry_count - 1
            ):

                await asyncio.sleep(
                    0.5
                )


    # =====================================================
    # NON STREAM 全部 KEY 失败
    # =====================================================

    raise HTTPException(
        status_code=503,
        detail={
            "message":
                "所有 Gemini API Key 请求均失败",

            "model":
                target_model,

            "error_type":
                (
                    type(
                        last_error
                    ).__name__
                    if last_error
                    else None
                ),

            "error":
                (
                    str(
                        last_error
                    )
                    if last_error
                    else "Unknown error"
                ),
        },
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
