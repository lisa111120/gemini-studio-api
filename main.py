import os
import time
import json
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel


# =========================================================
# GEMINI API
# =========================================================

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
GEMINI_API_VERSION = "v1beta"


# =========================================================
# API KEYS
#
# Zeabur 环境变量：
#
# GEMINI_API_KEYS=key1,key2,key3
#
# 支持一个或多个 Key
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


# =========================================================
# KEY 轮换
#
# 每次新请求从不同 Key 开始。
#
# 如果当前 Key 请求失败：
# 自动继续下一个 Key。
# =========================================================

_key_index = 0
_key_lock = asyncio.Lock()


async def get_key_order() -> List[str]:

    global _key_index

    if not API_KEYS:

        raise HTTPException(
            status_code=500,
            detail="请在环境变量中设置 GEMINI_API_KEYS"
        )

    async with _key_lock:

        start = _key_index

        _key_index = (
            _key_index + 1
        ) % len(API_KEYS)

    return (
        API_KEYS[start:]
        + API_KEYS[:start]
    )


# =========================================================
# HTTP CLIENT
#
# 使用一个长期存在的 httpx client。
#
# 不再使用 google-genai Python SDK。
# 直接请求 Gemini REST API。
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.http = httpx.AsyncClient(
        headers={
            "User-Agent":
                "gemini-openai-router/1.0"
        },
        follow_redirects=True,
    )

    print(
        f"[Startup] loaded_keys={len(API_KEYS)}",
        flush=True
    )

    try:

        yield

    finally:

        await app.state.http.aclose()


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Gemini REST OpenAI Compatible Router",
    lifespan=lifespan,
)


# =========================================================
# CORS
#
# 对齐之前正常工作的 Cloudflare Worker。
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
# OPENAI REQUEST STRUCTURE
# =========================================================

class ChatMessage(BaseModel):

    role: str

    content: Any = None

    name: Optional[str] = None

    tool_call_id: Optional[str] = None

    tool_calls: Optional[List[Any]] = None


class ChatRequest(BaseModel):

    # =====================================================
    # 模型完全透传
    #
    # Chatbox 传：
    #
    # gemini-3.7-flash
    #
    # 就请求：
    #
    # gemini-3.7-flash
    #
    # 不替换
    # 不映射
    # 不设默认模型
    # =====================================================

    model: str

    messages: List[ChatMessage]


    # =====================================================
    # 非常重要
    #
    # 不默认 True。
    #
    # stream=true
    #     → SSE
    #
    # stream=false
    #     → JSON
    #
    # 没传 stream
    #     → JSON
    #
    # 这个行为与正常工作的 CF Worker 一致。
    # =====================================================

    stream: Optional[bool] = None


    # =====================================================
    # THINKING
    #
    # 默认 HIGH
    # =====================================================

    thinking_level: Optional[str] = "high"


    # =====================================================
    # OPENAI 常见参数
    # =====================================================

    temperature: Optional[float] = None

    top_p: Optional[float] = None

    top_k: Optional[int] = None

    max_tokens: Optional[int] = None

    max_completion_tokens: Optional[int] = None

    stop: Optional[Any] = None

    seed: Optional[int] = None

    stream_options: Optional[
        Dict[str, Any]
    ] = None


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


    # =====================================================
    # OpenAI content array
    #
    # [
    #   {
    #       "type": "text",
    #       "text": "hello"
    #   }
    # ]
    # =====================================================

    if isinstance(
        content,
        list
    ):

        texts: List[str] = []

        for item in content:

            if not isinstance(
                item,
                dict
            ):
                continue

            if item.get(
                "type"
            ) in (
                "text",
                "input_text",
            ):

                texts.append(
                    str(
                        item.get(
                            "text",
                            ""
                        )
                    )
                )

        return "\n".join(
            texts
        )

    return str(content)


# =========================================================
# OPENAI MESSAGES → GEMINI
#
# system     → systemInstruction
# developer  → systemInstruction
# user       → user
# assistant  → model
# =========================================================

def transform_messages(
    messages: List[ChatMessage]
):

    system_parts: List[str] = []

    contents: List[
        Dict[str, Any]
    ] = []


    for msg in messages:

        text = extract_text_content(
            msg.content
        )


        # =================================================
        # SYSTEM
        # =================================================

        if msg.role in (
            "system",
            "developer",
        ):

            if text:

                system_parts.append(
                    text
                )

            continue


        # =================================================
        # ASSISTANT
        # =================================================

        if msg.role == "assistant":

            contents.append(
                {
                    "role":
                        "model",

                    "parts":
                        [
                            {
                                "text":
                                    text
                            }
                        ],
                }
            )

            continue


        # =================================================
        # USER
        # =================================================

        if msg.role == "user":

            contents.append(
                {
                    "role":
                        "user",

                    "parts":
                        [
                            {
                                "text":
                                    text
                            }
                        ],
                }
            )

            continue


        # =================================================
        # TOOL
        #
        # 当前按文本继续放进上下文。
        # =================================================

        if msg.role == "tool":

            contents.append(
                {
                    "role":
                        "user",

                    "parts":
                        [
                            {
                                "text":
                                    text
                            }
                        ],
                }
            )

            continue


        # =================================================
        # UNKNOWN ROLE
        # =================================================

        contents.append(
            {
                "role":
                    "user",

                "parts":
                    [
                        {
                            "text":
                                text
                        }
                    ],
            }
        )


    # =====================================================
    # 防止只有 system message
    # =====================================================

    if not contents:

        contents.append(
            {
                "role":
                    "user",

                "parts":
                    [
                        {
                            "text":
                                " "
                        }
                    ],
            }
        )


    # =====================================================
    # SYSTEM INSTRUCTION
    # =====================================================

    system_instruction = None


    if system_parts:

        system_instruction = {

            "parts":
                [
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
# GEMINI REQUEST BODY
# =========================================================

def build_gemini_body(
    req: ChatRequest
) -> Dict[str, Any]:

    (
        system_instruction,
        contents,
    ) = transform_messages(
        req.messages
    )


    # =====================================================
    # THINKING
    # =====================================================

    thinking_level = (
        req.thinking_level
        or "high"
    ).lower()


    if thinking_level not in (
        "minimal",
        "low",
        "medium",
        "high",
    ):

        thinking_level = "high"


    generation_config: Dict[
        str,
        Any
    ] = {

        "thinkingConfig":
            {
                "thinkingLevel":
                    thinking_level
            }
    }


    # =====================================================
    # MAX TOKENS
    # =====================================================

    max_tokens = (

        req.max_completion_tokens

        if (
            req.max_completion_tokens
            is not None
        )

        else req.max_tokens
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
    # SEED
    # =====================================================

    if req.seed is not None:

        generation_config[
            "seed"
        ] = req.seed


    # =====================================================
    # Gemini 3 官方建议保持默认 temperature。
    #
    # 所以这里不会把 Chatbox 的
    # temperature / top_p / top_k 强行塞进去。
    # =====================================================


    body: Dict[str, Any] = {

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


    if system_instruction is not None:

        body[
            "systemInstruction"
        ] = system_instruction


    return body


# =========================================================
# OPENAI COMPLETION ID
# =========================================================

def make_completion_id() -> str:

    return (
        "chatcmpl-"
        + str(
            int(
                time.time()
                * 1_000_000
            )
        )
    )


# =========================================================
# SSE
# =========================================================

def sse(
    payload: Dict[str, Any]
) -> str:

    return (
        "data: "
        + json.dumps(
            payload,
            ensure_ascii=False,
            separators=(
                ",",
                ":"
            ),
        )
        + "\n\n"
    )


# =========================================================
# FINISH REASON
# =========================================================

def map_finish_reason(
    reason: Optional[str]
) -> Optional[str]:

    if reason is None:

        return None


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
        reason.lower()
    )


# =========================================================
# GEMINI CANDIDATE → OPENAI CHOICE
#
# 这里按照你正常工作的 CF Worker 的结构：
#
# role
# content
# tool_calls
# logprobs
# finish_reason
# =========================================================

def candidate_to_openai(
    candidate: Dict[str, Any],
    key: str,
) -> Dict[str, Any]:

    message: Dict[str, Any] = {

        "role":
            "assistant",

        "content":
            [],
    }


    tool_calls: List[
        Dict[str, Any]
    ] = []


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


        # =================================================
        # FUNCTION CALL
        # =================================================

        function_call = (
            part.get(
                "functionCall"
            )
        )


        if function_call:

            tool_calls.append(
                {
                    "id":
                        (
                            function_call.get(
                                "id"
                            )
                            or (
                                "call_"
                                + str(
                                    int(
                                        time.time()
                                        * 1_000_000
                                    )
                                )
                            )
                        ),

                    "type":
                        "function",

                    "function":
                        {
                            "name":
                                (
                                    function_call.get(
                                        "name"
                                    )
                                    or ""
                                ),

                            "arguments":
                                json.dumps(
                                    (
                                        function_call.get(
                                            "args"
                                        )
                                        or {}
                                    ),

                                    ensure_ascii=False,

                                    separators=(
                                        ",",
                                        ":"
                                    ),
                                ),
                        },
                }
            )

            continue


        # =================================================
        # TEXT
        # =================================================

        if (
            "text" in part
            and
            part.get(
                "text"
            ) is not None
        ):

            message[
                "content"
            ].append(
                str(
                    part.get(
                        "text"
                    )
                )
            )


    # =====================================================
    # CF Worker 使用：
    #
    # \n\n|>
    #
    # 拼接多个 part。
    # =====================================================

    message[
        "content"
    ] = (

        "\n\n|>".join(
            message[
                "content"
            ]
        )

        or None
    )


    if tool_calls:

        message[
            "tool_calls"
        ] = tool_calls


    return {

        "index":
            (
                candidate.get(
                    "index"
                )
                or 0
            ),

        key:
            message,

        "logprobs":
            None,

        "finish_reason":
            (
                "tool_calls"

                if tool_calls

                else map_finish_reason(
                    candidate.get(
                        "finishReason"
                    )
                )
            ),
    }


# =========================================================
# TOKEN USAGE
# =========================================================

def transform_usage(
    metadata: Optional[
        Dict[str, Any]
    ]
) -> Optional[
    Dict[str, Any]
]:

    if not metadata:

        return None


    prompt_tokens = (
        metadata.get(
            "promptTokenCount"
        )
        or 0
    )


    candidate_tokens = (
        metadata.get(
            "candidatesTokenCount"
        )
        or 0
    )


    tool_tokens = (
        metadata.get(
            "toolUsePromptTokenCount"
        )
        or 0
    )


    thought_tokens = (
        metadata.get(
            "thoughtsTokenCount"
        )
        or 0
    )


    total_tokens = (
        metadata.get(
            "totalTokenCount"
        )
    )


    if total_tokens is None:

        total_tokens = (
            prompt_tokens
            + candidate_tokens
            + tool_tokens
            + thought_tokens
        )


    usage: Dict[str, Any] = {

        "prompt_tokens":
            prompt_tokens,

        "completion_tokens":
            (
                candidate_tokens
                + tool_tokens
                + thought_tokens
            ),

        "total_tokens":
            total_tokens,
    }


    if thought_tokens:

        usage[
            "completion_tokens_details"
        ] = {

            "reasoning_tokens":
                thought_tokens
        }


    return usage


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
# FAVICON
#
# 防止你之前日志一直出现：
#
# GET /favicon.ico 405
# =========================================================

@app.get("/favicon.ico")
async def favicon():

    return Response(
        status_code=204
    )


# =========================================================
# MODELS
#
# 同时支持：
#
# /models
# /v1/models
#
# Chatbox API Host 如果填写：
#
# https://xxx.zeabur.app/v1
#
# 就会调用：
#
# /v1/models
# =========================================================

@app.get("/models")
@app.get("/v1/models")
async def models():

    key_order = (
        await get_key_order()
    )


    last_error: Optional[
        Exception
    ] = None


    client: httpx.AsyncClient = (
        app.state.http
    )


    for attempt, api_key in enumerate(
        key_order,
        start=1
    ):

        try:

            url = (
                f"{GEMINI_BASE_URL}/"
                f"{GEMINI_API_VERSION}/"
                f"models"
            )


            response = (
                await client.get(

                    url,

                    headers={
                        "x-goog-api-key":
                            api_key
                    },

                    timeout=httpx.Timeout(
                        30.0
                    ),
                )
            )


            if not response.is_success:

                raise RuntimeError(
                    "Gemini models HTTP "
                    f"{response.status_code}: "
                    f"{response.text}"
                )


            google_data = (
                response.json()
            )


            result = []


            for item in (
                google_data.get(
                    "models"
                )
                or []
            ):


                name = (
                    item.get(
                        "name"
                    )
                    or ""
                )


                if name.startswith(
                    "models/"
                ):

                    name = (
                        name[7:]
                    )


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


        except Exception as exc:

            last_error = exc


            print(
                "[Models Error] "
                f"attempt={attempt}/"
                f"{len(key_order)} "
                f"error={exc!r}",
                flush=True
            )


            if attempt < len(
                key_order
            ):

                await asyncio.sleep(
                    0.4
                )


    raise HTTPException(

        status_code=503,

        detail=(
            "模型列表请求失败: "
            f"{last_error}"
        ),
    )


# =========================================================
# CHAT COMPLETIONS
#
# 同时支持：
#
# /chat/completions
# /v1/chat/completions
# =========================================================

@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatRequest
):


    # =====================================================
    # MODEL 完全透传
    # =====================================================

    target_model = req.model


    # =====================================================
    # 只有明确 stream=true 才走 SSE
    #
    # 这是之前最关键的兼容问题。
    # =====================================================

    is_stream = (
        req.stream is True
    )


    # =====================================================
    # 构造 Gemini 请求
    # =====================================================

    body = build_gemini_body(
        req
    )


    # =====================================================
    # 取得本次请求的 Key 顺序
    # =====================================================

    key_order = (
        await get_key_order()
    )


    client: httpx.AsyncClient = (
        app.state.http
    )


    print(
        "[Chat Request] "
        f"model={target_model} "
        f"stream_received={req.stream!r} "
        f"effective_stream={is_stream}",
        flush=True
    )


    # =====================================================
    # STREAM
    # =====================================================

    if is_stream:


        async def event_stream():


            completion_id = (
                make_completion_id()
            )


            last_error: Optional[
                Exception
            ] = None


            # =================================================
            # 一旦已经给 Chatbox 发出内容，
            # 就不能换 Key 从头生成。
            # =================================================

            sent_any_openai_chunk = False


            # =================================================
            # KEY RETRY
            # =================================================

            for (
                attempt,
                api_key
            ) in enumerate(
                key_order,
                start=1
            ):


                got_google_event = False

                sent_first_chunk = False


                final_choices: Dict[
                    int,
                    Dict[str, Any]
                ] = {}


                final_usage: Optional[
                    Dict[str, Any]
                ] = None


                url = (
                    f"{GEMINI_BASE_URL}/"
                    f"{GEMINI_API_VERSION}/"
                    f"models/{target_model}:"
                    f"streamGenerateContent"
                    f"?alt=sse"
                )


                try:


                    print(
                        "[Gemini Stream Start] "
                        f"model={target_model} "
                        f"attempt={attempt}/"
                        f"{len(key_order)}",
                        flush=True
                    )


                    timeout = httpx.Timeout(

                        connect=20.0,

                        # 高思考模式可能较慢。
                        # 不在这里强制截断生成。
                        read=None,

                        write=30.0,

                        pool=20.0,
                    )


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

                        timeout=timeout,

                    ) as response:


                        print(
                            "[Gemini Stream HTTP] "
                            f"model={target_model} "
                            f"status="
                            f"{response.status_code}",
                            flush=True
                        )


                        # =====================================
                        # GOOGLE ERROR
                        # =====================================

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
                                "Gemini HTTP "
                                f"{response.status_code}: "
                                f"{error_text}"
                            )


                        # =====================================
                        # GOOGLE SSE
                        # =====================================

                        async for line in (
                            response.aiter_lines()
                        ):


                            # =================================
                            # 空行
                            # =================================

                            if not line:

                                continue


                            # =================================
                            # SSE comment
                            # =================================

                            if line.startswith(
                                ":"
                            ):

                                continue


                            # =================================
                            # 只处理 data:
                            # =================================

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


                            # =================================
                            # JSON
                            # =================================

                            try:

                                google_chunk = (
                                    json.loads(
                                        raw
                                    )
                                )

                            except json.JSONDecodeError as exc:

                                raise RuntimeError(
                                    "Gemini SSE JSON "
                                    "解析失败: "
                                    f"{raw[:500]}"
                                ) from exc


                            # =================================
                            # CF Worker 会要求 candidates 存在
                            # =================================

                            candidates = (
                                google_chunk.get(
                                    "candidates"
                                )
                            )


                            if candidates is None:

                                raise RuntimeError(
                                    "Gemini 返回无 "
                                    "candidates 的流事件: "
                                    f"{raw[:500]}"
                                )


                            got_google_event = True


                            # =================================
                            # USAGE
                            # =================================

                            if google_chunk.get(
                                "usageMetadata"
                            ):

                                final_usage = (
                                    transform_usage(
                                        google_chunk.get(
                                            "usageMetadata"
                                        )
                                    )
                                )


                            # =================================
                            # 第一帧
                            #
                            # 完全按照你的 CF Worker：
                            #
                            # delta:
                            # {
                            #   role: assistant,
                            #   content: ""
                            # }
                            # =================================

                            if (
                                candidates
                                and
                                not sent_first_chunk
                            ):


                                first_choice = (
                                    candidate_to_openai(
                                        candidates[0],
                                        "delta"
                                    )
                                )


                                first_choice[
                                    "delta"
                                ] = {

                                    "role":
                                        "assistant",

                                    "content":
                                        "",
                                }


                                first_choice[
                                    "finish_reason"
                                ] = None


                                first_payload = {

                                    "id":
                                        completion_id,

                                    "choices":
                                        [
                                            first_choice
                                        ],

                                    "created":
                                        int(
                                            time.time()
                                        ),

                                    "model":
                                        target_model,

                                    "object":
                                        "chat.completion.chunk",
                                }


                                yield sse(
                                    first_payload
                                )


                                sent_first_chunk = True

                                sent_any_openai_chunk = True


                            # =================================
                            # CANDIDATES
                            # =================================

                            for candidate in candidates:


                                choice = (
                                    candidate_to_openai(
                                        candidate,
                                        "delta"
                                    )
                                )


                                finish_reason = (
                                    choice.get(
                                        "finish_reason"
                                    )
                                )


                                # =================================
                                # 正文 chunk 不立即带结束状态
                                # =================================

                                choice[
                                    "finish_reason"
                                ] = None


                                delta = (
                                    choice.get(
                                        "delta"
                                    )
                                    or {}
                                )


                                # 第一帧后，
                                # 后续不再重复 role

                                delta.pop(
                                    "role",
                                    None
                                )


                                # =================================
                                # 与 CF Worker 一致：
                                #
                                # 只要 delta 有 content 字段
                                # 就发送。
                                # =================================

                                if (
                                    "content"
                                    in delta
                                ):


                                    payload = {

                                        "id":
                                            completion_id,

                                        "choices":
                                            [
                                                choice
                                            ],

                                        "created":
                                            int(
                                                time.time()
                                            ),

                                        "model":
                                            target_model,

                                        "object":
                                            "chat.completion.chunk",
                                    }


                                    yield sse(
                                        payload
                                    )


                                    sent_any_openai_chunk = True


                                # =================================
                                # 暂存最终状态
                                # =================================

                                final_choice = {

                                    "index":
                                        choice.get(
                                            "index",
                                            0
                                        ),

                                    "delta":
                                        {},

                                    "logprobs":
                                        None,

                                    "finish_reason":
                                        finish_reason,
                                }


                                final_choices[
                                    final_choice[
                                        "index"
                                    ]
                                ] = final_choice


                    # =================================================
                    # GOOGLE STREAM 正常关闭
                    # =================================================

                    if got_google_event:


                        # =============================================
                        # 极端情况下 Google 有事件
                        # 但没有 candidate 正文
                        # =============================================

                        if not sent_first_chunk:


                            first_payload = {

                                "id":
                                    completion_id,

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

                                            "logprobs":
                                                None,

                                            "finish_reason":
                                                None,
                                        }
                                    ],

                                "created":
                                    int(
                                        time.time()
                                    ),

                                "model":
                                    target_model,

                                "object":
                                    "chat.completion.chunk",
                            }


                            yield sse(
                                first_payload
                            )


                            sent_any_openai_chunk = True


                        # =============================================
                        # stream_options.include_usage
                        # =============================================

                        include_usage = bool(

                            req.stream_options

                            and

                            req.stream_options.get(
                                "include_usage"
                            )
                        )


                        # =============================================
                        # FINAL CHUNK
                        # =============================================

                        if final_choices:


                            for index in sorted(
                                final_choices
                            ):


                                final_choice = (
                                    final_choices[
                                        index
                                    ]
                                )


                                if (
                                    final_choice[
                                        "finish_reason"
                                    ]
                                    is None
                                ):

                                    final_choice[
                                        "finish_reason"
                                    ] = "stop"


                                final_payload: Dict[
                                    str,
                                    Any
                                ] = {

                                    "id":
                                        completion_id,

                                    "choices":
                                        [
                                            final_choice
                                        ],

                                    "created":
                                        int(
                                            time.time()
                                        ),

                                    "model":
                                        target_model,

                                    "object":
                                        "chat.completion.chunk",
                                }


                                if (
                                    include_usage
                                    and
                                    final_usage
                                ):

                                    final_payload[
                                        "usage"
                                    ] = final_usage


                                yield sse(
                                    final_payload
                                )


                        else:


                            final_payload = {

                                "id":
                                    completion_id,

                                "choices":
                                    [
                                        {
                                            "index":
                                                0,

                                            "delta":
                                                {},

                                            "logprobs":
                                                None,

                                            "finish_reason":
                                                "stop",
                                        }
                                    ],

                                "created":
                                    int(
                                        time.time()
                                    ),

                                "model":
                                    target_model,

                                "object":
                                    "chat.completion.chunk",
                            }


                            if (
                                include_usage
                                and
                                final_usage
                            ):

                                final_payload[
                                    "usage"
                                ] = final_usage


                            yield sse(
                                final_payload
                            )


                        # =============================================
                        # DONE
                        # =============================================

                        yield (
                            "data: [DONE]\n\n"
                        )


                        print(
                            "[Gemini Stream Done] "
                            f"model={target_model}",
                            flush=True
                        )


                        return


                    # =================================================
                    # HTTP 200 但 Gemini 一个事件都没给
                    # =================================================

                    raise RuntimeError(
                        "Gemini SSE 已关闭，"
                        "但没有返回任何事件"
                    )


                # =====================================================
                # STREAM ERROR
                # =====================================================

                except Exception as exc:


                    last_error = exc


                    print(
                        "[Gemini Stream Error] "
                        f"model={target_model} "
                        f"attempt={attempt}/"
                        f"{len(key_order)} "
                        f"error={exc!r}",
                        flush=True
                    )


                    # ================================================
                    # 一个 Google 事件都没收到，
                    # 并且还没有向 Chatbox 输出任何 chunk：
                    #
                    # 可以安全换下一个 Key。
                    # ================================================

                    if (

                        not sent_any_openai_chunk

                        and
                        not got_google_event

                        and
                        attempt < len(
                            key_order
                        )
                    ):


                        await asyncio.sleep(
                            0.4
                        )


                        continue


                    # ================================================
                    # 已经开始输出以后不能换 Key，
                    # 否则会从头重复回答。
                    # ================================================

                    break


            # =========================================================
            # ALL KEYS FAILED
            #
            # StreamingResponse 一旦建立以后 HTTP 已经是 200，
            # 此时只能正常结束 SSE。
            # =========================================================

            if not sent_any_openai_chunk:


                yield sse(
                    {
                        "id":
                            completion_id,

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

                                    "logprobs":
                                        None,

                                    "finish_reason":
                                        None,
                                }
                            ],

                        "created":
                            int(
                                time.time()
                            ),

                        "model":
                            target_model,

                        "object":
                            "chat.completion.chunk",
                    }
                )


            # =========================================================
            # ERROR TEXT
            # =========================================================

            yield sse(
                {
                    "id":
                        completion_id,

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
                                                f"{last_error}]"
                                            )
                                    },

                                "logprobs":
                                    None,

                                "finish_reason":
                                    None,
                            }
                        ],

                    "created":
                        int(
                            time.time()
                        ),

                    "model":
                        target_model,

                    "object":
                        "chat.completion.chunk",
                }
            )


            # =========================================================
            # FINAL
            # =========================================================

            yield sse(
                {
                    "id":
                        completion_id,

                    "choices":
                        [
                            {
                                "index":
                                    0,

                                "delta":
                                    {},

                                "logprobs":
                                    None,

                                "finish_reason":
                                    "stop",
                            }
                        ],

                    "created":
                        int(
                            time.time()
                        ),

                    "model":
                        target_model,

                    "object":
                        "chat.completion.chunk",
                }
            )


            yield (
                "data: [DONE]\n\n"
            )


        # =============================================================
        # RETURN SSE
        # =============================================================

        return StreamingResponse(

            event_stream(),

            media_type=
                "text/event-stream",

            headers={

                "Cache-Control":
                    "no-cache, no-transform",

                "Connection":
                    "keep-alive",

                "X-Accel-Buffering":
                    "no",
            },
        )


    # =========================================================
    # NON STREAM
    #
    # stream=false
    #
    # 或者 Chatbox 根本没有发送 stream
    #
    # 都必须返回 JSON。
    # =========================================================

    last_error: Optional[
        Exception
    ] = None


    for (
        attempt,
        api_key
    ) in enumerate(
        key_order,
        start=1
    ):


        try:


            url = (
                f"{GEMINI_BASE_URL}/"
                f"{GEMINI_API_VERSION}/"
                f"models/{target_model}:"
                f"generateContent"
            )


            print(
                "[Gemini Generate Start] "
                f"model={target_model} "
                f"attempt={attempt}/"
                f"{len(key_order)}",
                flush=True
            )


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

                    timeout=httpx.Timeout(
                        180.0
                    ),
                )
            )


            print(
                "[Gemini Generate HTTP] "
                f"model={target_model} "
                f"status="
                f"{response.status_code}",
                flush=True
            )


            # =================================================
            # GOOGLE ERROR
            # =================================================

            if not response.is_success:

                raise RuntimeError(
                    "Gemini HTTP "
                    f"{response.status_code}: "
                    f"{response.text}"
                )


            google_data = (
                response.json()
            )


            candidates = (
                google_data.get(
                    "candidates"
                )
                or []
            )


            # =================================================
            # OPENAI CHOICES
            # =================================================

            choices = [

                candidate_to_openai(
                    candidate,
                    "message"
                )

                for candidate
                in candidates
            ]


            # =================================================
            # 没有 candidate
            # =================================================

            if not choices:


                block_reason = (
                    (
                        google_data.get(
                            "promptFeedback"
                        )
                        or {}
                    )
                    .get(
                        "blockReason"
                    )
                )


                choices = [
                    {
                        "index":
                            0,

                        "message":
                            None,

                        "logprobs":
                            None,

                        "finish_reason":
                            (
                                "content_filter"

                                if block_reason

                                else "stop"
                            ),
                    }
                ]


            # =================================================
            # OPENAI RESPONSE
            # =================================================

            result: Dict[
                str,
                Any
            ] = {

                "id":
                    make_completion_id(),

                "choices":
                    choices,

                "created":
                    int(
                        time.time()
                    ),

                "model":
                    target_model,

                "object":
                    "chat.completion",
            }


            # =================================================
            # USAGE
            # =================================================

            usage = (
                transform_usage(
                    google_data.get(
                        "usageMetadata"
                    )
                )
            )


            if usage:

                result[
                    "usage"
                ] = usage


            print(
                "[Gemini Generate Done] "
                f"model={target_model}",
                flush=True
            )


            return JSONResponse(
                content=result
            )


        # =====================================================
        # NON STREAM ERROR
        # =====================================================

        except Exception as exc:


            last_error = exc


            print(
                "[Gemini Generate Error] "
                f"model={target_model} "
                f"attempt={attempt}/"
                f"{len(key_order)} "
                f"error={exc!r}",
                flush=True
            )


            # =================================================
            # 自动换下一个 Key
            # =================================================

            if attempt < len(
                key_order
            ):

                await asyncio.sleep(
                    0.4
                )


    # =========================================================
    # 所有 Key 都失败
    # =========================================================

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
        },
    )


# =========================================================
# START SERVER
#
# Zeabur 会读取 PORT 环境变量。
# =========================================================

if __name__ == "__main__":

    import uvicorn


    port = int(
        os.environ.get(
            "PORT",
            "8080"
        )
    )


    uvicorn.run(

        app,

        host="0.0.0.0",

        port=port,
    )
