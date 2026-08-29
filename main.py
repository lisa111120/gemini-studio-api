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
# Zeabur:
#
# GEMINI_API_KEYS=key1,key2,key3
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
# KEY ROTATION
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
# OPENAI REQUEST
# =========================================================

class ChatMessage(BaseModel):

    role: str

    content: Any = None

    name: Optional[str] = None

    tool_call_id: Optional[str] = None

    tool_calls: Optional[List[Any]] = None


class ChatRequest(BaseModel):

    # =====================================================
    # MODEL 完全透传
    # =====================================================

    model: str

    messages: List[ChatMessage]


    # =====================================================
    # stream=true 才流式
    #
    # false / 没传：
    # 返回普通 JSON
    # =====================================================

    stream: Optional[bool] = None


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
        # SYSTEM / DEVELOPER
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

            role = "model"

        else:

            # user / tool / unknown
            role = "user"


        contents.append(
            {
                "role":
                    role,

                "parts":
                    [
                        {
                            "text":
                                text
                        }
                    ],
            }
        )


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
# GEMINI BODY
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
# ID
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
# 判断 candidate 是否真的有内容
#
# 这是本次最关键的修改之一。
#
# candidate 存在但：
#
# parts=[]
#
# 或
#
# text=""
#
# 不再认为是成功回答。
# =========================================================

def candidate_has_payload(
    candidate: Dict[str, Any]
) -> bool:

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


        # 工具调用也属于有效输出
        if part.get(
            "functionCall"
        ):

            return True


        text = part.get(
            "text"
        )


        if (
            isinstance(
                text,
                str
            )
            and
            text != ""
        ):

            return True


    return False


def response_has_payload(
    candidates: List[
        Dict[str, Any]
    ]
) -> bool:

    return any(
        candidate_has_payload(
            candidate
        )
        for candidate in candidates
    )


# =========================================================
# GEMINI CANDIDATE → OPENAI
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

                            or

                            (
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
# USAGE
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
# =========================================================

@app.get("/favicon.ico")
async def favicon():

    return Response(
        status_code=204
    )


# =========================================================
# MODELS
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


                if name:

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


            # =================================================
            # 保持原来的重试流程
            # =================================================

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
# =========================================================

@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatRequest
):


    # =====================================================
    # MODEL 完全透传
    # =====================================================

    target_model = (
        req.model
    )


    # =====================================================
    # STREAM
    # =====================================================

    is_stream = (
        req.stream is True
    )


    # =====================================================
    # GEMINI BODY
    # =====================================================

    body = (
        build_gemini_body(
            req
        )
    )


    # =====================================================
    # KEY 顺序
    # =====================================================

    key_order = (
        await get_key_order()
    )


    client: httpx.AsyncClient = (
        app.state.http
    )


    # =====================================================
    # 每一个 Chatbox 请求都有新的 ID
    #
    # 你重新点重试时，
    # 这里一定会生成一个新 ID。
    # =====================================================

    completion_id = (
        make_completion_id()
    )


    print(

        "[Chat Request] "
        f"id={completion_id} "
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


            last_error: Optional[
                Exception
            ] = None


            # =================================================
            # 一旦已经真正给 Chatbox 输出，
            # 就不能切 Key 从头重生成。
            # =================================================

            sent_any_openai_chunk = False


            # =================================================
            # 保持你现有的多 Key / 繁忙重试流程
            # =================================================

            for (
                attempt,
                api_key
            ) in enumerate(
                key_order,
                start=1
            ):


                got_google_event = False

                saw_payload = False

                sent_first_chunk = False


                final_choices: Dict[
                    int,
                    Dict[str, Any]
                ] = {}


                final_usage: Optional[
                    Dict[str, Any]
                ] = None


                last_google_chunk: Optional[
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
                        f"id={completion_id} "
                        f"model={target_model} "
                        f"attempt={attempt}/"
                        f"{len(key_order)}",

                        flush=True
                    )


                    timeout = httpx.Timeout(

                        connect=20.0,

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
                            f"id={completion_id} "
                            f"model={target_model} "
                            f"status={response.status_code}",

                            flush=True
                        )


                        # =====================================
                        # HTTP ERROR
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
                        # GEMINI SSE
                        # =====================================

                        async for line in (
                            response.aiter_lines()
                        ):


                            if not line:

                                continue


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


                            got_google_event = True

                            last_google_chunk = (
                                google_chunk
                            )


                            candidates = (
                                google_chunk.get(
                                    "candidates"
                                )
                                or []
                            )


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
                            # CANDIDATES
                            # =================================

                            for candidate in candidates:


                                index = (
                                    candidate.get(
                                        "index"
                                    )
                                    or 0
                                )


                                finish_reason = (
                                    map_finish_reason(

                                        candidate.get(
                                            "finishReason"
                                        )
                                    )
                                )


                                # =================================
                                # 即使这一帧没有正文，
                                # 仍然保存最后 finish reason
                                # =================================

                                if finish_reason is not None:

                                    final_choices[
                                        index
                                    ] = {

                                        "index":
                                            index,

                                        "delta":
                                            {},

                                        "logprobs":
                                            None,

                                        "finish_reason":
                                            finish_reason,
                                    }


                                # =================================
                                # 关键：
                                #
                                # candidate 存在但没正文，
                                # 不算成功。
                                # =================================

                                if not candidate_has_payload(
                                    candidate
                                ):

                                    continue


                                saw_payload = True


                                # =================================
                                # 第一个真正有内容的 candidate
                                # 才给 Chatbox 发送第一帧。
                                #
                                # 这样空回复仍然可以安全重试。
                                # =================================

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


                                    sent_first_chunk = True

                                    sent_any_openai_chunk = True


                                # =================================
                                # 正文
                                # =================================

                                choice = (
                                    candidate_to_openai(

                                        candidate,

                                        "delta"
                                    )
                                )


                                choice_finish_reason = (
                                    choice.get(
                                        "finish_reason"
                                    )
                                )


                                choice[
                                    "finish_reason"
                                ] = None


                                delta = (
                                    choice.get(
                                        "delta"
                                    )
                                    or {}
                                )


                                delta.pop(
                                    "role",
                                    None
                                )


                                # =================================
                                # 只发送真正有内容的 chunk
                                # =================================

                                if (

                                    delta.get(
                                        "content"
                                    )
                                    is not None

                                    or

                                    delta.get(
                                        "tool_calls"
                                    )
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


                                final_choices[
                                    index
                                ] = {

                                    "index":
                                        index,

                                    "delta":
                                        {},

                                    "logprobs":
                                        None,

                                    "finish_reason":
                                        choice_finish_reason,
                                }


                    # =================================================
                    # 关键修改：
                    #
                    # Gemini HTTP 200
                    # SSE 也正常结束
                    #
                    # 但如果从头到尾没有任何真正正文，
                    # 不能返回空成功。
                    #
                    # 抛异常，让它进入下面原有重试流程。
                    # =================================================

                    if not saw_payload:


                        raw_debug = (

                            json.dumps(
                                last_google_chunk,
                                ensure_ascii=False
                            )

                            if last_google_chunk

                            else "null"
                        )


                        print(

                            "[Gemini Empty Stream] "
                            f"id={completion_id} "
                            f"model={target_model} "
                            f"attempt={attempt}/"
                            f"{len(key_order)} "
                            f"raw={raw_debug}",

                            flush=True
                        )


                        raise RuntimeError(

                            "Gemini HTTP 200 / SSE 正常结束，"
                            "但没有返回任何可用正文或工具调用"
                        )


                    # =================================================
                    # INCLUDE USAGE
                    # =================================================

                    include_usage = bool(

                        req.stream_options

                        and

                        req.stream_options.get(
                            "include_usage"
                        )
                    )


                    # =================================================
                    # FINAL
                    # =================================================

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


                    yield (
                        "data: [DONE]\n\n"
                    )


                    print(

                        "[Gemini Stream Done] "
                        f"id={completion_id} "
                        f"model={target_model}",

                        flush=True
                    )


                    return


                # =====================================================
                # ERROR / EMPTY RESPONSE
                # =====================================================

                except Exception as exc:


                    last_error = exc


                    print(

                        "[Gemini Stream Error] "
                        f"id={completion_id} "
                        f"model={target_model} "
                        f"attempt={attempt}/"
                        f"{len(key_order)} "
                        f"error={exc!r}",

                        flush=True
                    )


                    # =================================================
                    # 保持你的原有繁忙/多 Key 重试流程
                    #
                    # 还没有给 Chatbox 输出任何内容：
                    #
                    # 等 0.4 秒
                    # → 换下一个 Key
                    #
                    # 这里没有修改。
                    # =================================================

                    if (

                        not sent_any_openai_chunk

                        and

                        attempt < len(
                            key_order
                        )
                    ):


                        await asyncio.sleep(
                            0.4
                        )


                        continue


                    # =================================================
                    # 已经输出过正文：
                    #
                    # 不切 Key。
                    # =================================================

                    break


            # =========================================================
            # 所有 Key 都失败
            #
            # 绝不再空白结束。
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
    # =========================================================

    last_error: Optional[
        Exception
    ] = None


    # =========================================================
    # 保持原来的多 Key / 繁忙重试流程
    # =========================================================

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
                f"id={completion_id} "
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
                f"id={completion_id} "
                f"model={target_model} "
                f"status={response.status_code}",

                flush=True
            )


            # =================================================
            # HTTP ERROR
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
            # 关键修改：
            #
            # HTTP 200 但：
            #
            # candidates=[]
            #
            # 或者 candidates 有对象，
            # 但没有任何正文 / tool call。
            #
            # 以前：
            # 返回空 assistant → Chatbox 0 秒空回。
            #
            # 现在：
            # 当作本次请求失败
            # → 进入原来的多 Key 重试流程。
            # =================================================

            if (

                not candidates

                or

                not response_has_payload(
                    candidates
                )
            ):


                raw_debug = (
                    json.dumps(
                        google_data,
                        ensure_ascii=False
                    )
                )


                print(

                    "[Gemini Empty Response] "
                    f"id={completion_id} "
                    f"model={target_model} "
                    f"attempt={attempt}/"
                    f"{len(key_order)} "
                    f"raw={raw_debug}",

                    flush=True
                )


                raise RuntimeError(

                    "Gemini HTTP 200，"
                    "但没有返回任何可用正文或工具调用"
                )


            # =================================================
            # NORMAL RESPONSE
            # =================================================

            choices = [

                candidate_to_openai(
                    candidate,
                    "message"
                )

                for candidate
                in candidates
            ]


            result: Dict[
                str,
                Any
            ] = {

                "id":
                    completion_id,

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
                f"id={completion_id} "
                f"model={target_model}",

                flush=True
            )


            return JSONResponse(
                content=result
            )


        # =====================================================
        # ERROR / EMPTY RESPONSE
        # =====================================================

        except Exception as exc:


            last_error = exc


            print(

                "[Gemini Generate Error] "
                f"id={completion_id} "
                f"model={target_model} "
                f"attempt={attempt}/"
                f"{len(key_order)} "
                f"error={exc!r}",

                flush=True
            )


            # =================================================
            # 不改你的重试流程：
            #
            # 失败
            # → 0.4 秒
            # → 下一 Key
            # =================================================

            if attempt < len(
                key_order
            ):

                await asyncio.sleep(
                    0.4
                )


    # =========================================================
    # 所有 Key 全部失败
    #
    # 不再返回成功空消息。
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
# START
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
