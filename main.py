import os
import time
import json
import itertools
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Any

from google import genai
from google.genai import types


app = FastAPI(title="Gemini OpenAI Compatible Router")


# =========================================================
# API KEYS
# =========================================================

keys_env = os.environ.get("GEMINI_API_KEYS", "")

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
# OPENAI REQUEST STRUCTURE
# =========================================================

class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatRequest(BaseModel):

    # =====================================================
    # 完全透传
    #
    # Chatbox 写什么，这里就是什么
    # =====================================================

    model: str

    messages: List[ChatMessage]

    stream: Optional[bool] = True

    # 默认最高思考
    thinking_level: Optional[str] = "high"

    # Chatbox 可能会传这些 OpenAI 参数
    # 暂时接收，但不强制使用
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None


# =========================================================
# CONFIG
# =========================================================

def build_config(req: ChatRequest):

    thinking_level = req.thinking_level

    if thinking_level not in (
        "low",
        "medium",
        "high",
        None,
    ):
        thinking_level = "high"


    thinking_config = None

    if thinking_level:

        thinking_config = types.ThinkingConfig(
            thinking_level=thinking_level
        )


    config_args = {
        "safety_settings": GLOBAL_SAFETY_SETTINGS,
        "thinking_config": thinking_config,
    }


    # =====================================================
    # 兼容 Chatbox 传来的生成参数
    # =====================================================

    if req.temperature is not None:
        config_args["temperature"] = req.temperature

    if req.top_p is not None:
        config_args["top_p"] = req.top_p


    # OpenAI 新旧参数都兼容

    max_tokens = (
        req.max_completion_tokens
        if req.max_completion_tokens is not None
        else req.max_tokens
    )

    if max_tokens is not None:
        config_args["max_output_tokens"] = max_tokens


    return types.GenerateContentConfig(
        **config_args
    )


# =========================================================
# MESSAGE → PROMPT
# =========================================================

def build_prompt(messages: List[ChatMessage]):

    parts = []

    for msg in messages:

        content = msg.content


        # 普通文本
        if isinstance(content, str):

            text = content


        # OpenAI content array
        elif isinstance(content, list):

            text_parts = []

            for item in content:

                if not isinstance(item, dict):
                    continue

                if item.get("type") == "text":
                    text_parts.append(
                        str(item.get("text", ""))
                    )

            text = "\n".join(text_parts)


        else:

            text = str(content)


        parts.append(
            f"{msg.role}: {text}"
        )


    return "\n".join(parts)


# =========================================================
# ROOT
# =========================================================

@app.get("/")
async def root():

    return {
        "status": "running",
        "service": "Gemini OpenAI Compatible Router",
        "loaded_keys_count": len(CLIENT_POOL),
        "model_mode": "passthrough",
        "default_thinking_level": "high",
    }


# =========================================================
# MODELS
#
# 同时兼容：
#
# /models
# /v1/models
#
# Chatbox 有时会请求 /models
# =========================================================

@app.get("/models")
@app.get("/v1/models")
async def models():

    now = int(time.time())

    return {
        "object": "list",

        # 不对白名单做限制
        #
        # 这里返回一个占位 Gemini 模型，
        # 只是让 Chatbox 的模型接口检查通过。
        #
        # 实际聊天时 model 仍然完全透传。
        "data": [
            {
                "id": "gemini",
                "object": "model",
                "created": now,
                "owned_by": "google",
            }
        ],
    }


# =========================================================
# CHAT
#
# 同时兼容：
#
# /chat/completions
# /v1/chat/completions
# =========================================================

@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):


    # =====================================================
    # API KEY 检查
    # =====================================================

    if not CLIENT_POOL:

        raise HTTPException(
            status_code=500,
            detail="未配置 GEMINI_API_KEYS",
        )


    # =====================================================
    # MODEL 完全透传
    # =====================================================
    #
    # 没有：
    #
    # DEFAULT_MODEL
    # model mapping
    # model whitelist
    # model replacement
    #
    # Chatbox:
    #
    # gemini-3.6-flash
    #
    # Gemini SDK:
    #
    # model="gemini-3.6-flash"
    #
    # =====================================================

    target_model = req.model


    prompt_text = build_prompt(
        req.messages
    )


    config = build_config(req)


    retry_count = len(CLIENT_POOL)


    # =====================================================
    # STREAM
    # =====================================================

    if req.stream:


        async def event_stream():

            last_error = None


            completion_id = (
                "chatcmpl-gemini-"
                + str(int(time.time() * 1000))
            )


            created = int(time.time())


            # =================================================
            # OpenAI 标准第一帧
            # =================================================

            first_payload = {

                "id": completion_id,

                "object": "chat.completion.chunk",

                "created": created,

                "model": target_model,

                "choices": [
                    {
                        "index": 0,

                        "delta": {
                            "role": "assistant"
                        },

                        "finish_reason": None,
                    }
                ],
            }


            yield (
                "data: "
                + json.dumps(
                    first_payload,
                    ensure_ascii=False
                )
                + "\n\n"
            )


            # =================================================
            # 多 KEY 重试
            # =================================================

            for attempt in range(retry_count):


                current_client = get_next_client()


                # 是否已经从 Gemini 收到正文
                started = False


                try:


                    response_stream = (
                        await current_client.aio.models.generate_content_stream(
                            model=target_model,
                            contents=prompt_text,
                            config=config,
                        )
                    )


                    async for chunk in response_stream:


                        text = getattr(
                            chunk,
                            "text",
                            None
                        )


                        if not text:
                            continue


                        started = True


                        payload = {

                            "id": completion_id,

                            "object": "chat.completion.chunk",

                            "created": created,

                            "model": target_model,

                            "choices": [
                                {
                                    "index": 0,

                                    "delta": {
                                        "content": text
                                    },

                                    "finish_reason": None,
                                }
                            ],
                        }


                        yield (
                            "data: "
                            + json.dumps(
                                payload,
                                ensure_ascii=False
                            )
                            + "\n\n"
                        )


                    # =================================================
                    # 正常结束
                    # =================================================

                    finish_payload = {

                        "id": completion_id,

                        "object": "chat.completion.chunk",

                        "created": created,

                        "model": target_model,

                        "choices": [
                            {
                                "index": 0,

                                "delta": {},

                                "finish_reason": "stop",
                            }
                        ],
                    }


                    yield (
                        "data: "
                        + json.dumps(
                            finish_payload,
                            ensure_ascii=False
                        )
                        + "\n\n"
                    )


                    yield "data: [DONE]\n\n"


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
                    # 已经输出正文
                    #
                    # 不能重新生成，否则文本重复
                    # =================================================

                    if started:


                        error_payload = {

                            "id": completion_id,

                            "object": "chat.completion.chunk",

                            "created": created,

                            "model": target_model,

                            "choices": [
                                {
                                    "index": 0,

                                    "delta": {
                                        "content": (
                                            "\n\n"
                                            "[Gemini 流式连接中断: "
                                            f"{str(e)}]"
                                        )
                                    },

                                    "finish_reason": None,
                                }
                            ],
                        }


                        yield (
                            "data: "
                            + json.dumps(
                                error_payload,
                                ensure_ascii=False
                            )
                            + "\n\n"
                        )


                        finish_payload = {

                            "id": completion_id,

                            "object": "chat.completion.chunk",

                            "created": created,

                            "model": target_model,

                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop",
                                }
                            ],
                        }


                        yield (
                            "data: "
                            + json.dumps(
                                finish_payload,
                                ensure_ascii=False
                            )
                            + "\n\n"
                        )


                        yield "data: [DONE]\n\n"


                        return


                    # =================================================
                    # 一个 token 都没输出
                    #
                    # 可以安全切换下一个 KEY
                    # =================================================

                    if attempt < retry_count - 1:

                        await asyncio.sleep(0.5)

                        continue


            # =================================================
            # 所有 KEY 都失败
            # =================================================

            error_payload = {

                "id": completion_id,

                "object": "chat.completion.chunk",

                "created": created,

                "model": target_model,

                "choices": [
                    {
                        "index": 0,

                        "delta": {
                            "content": (
                                "[Gemini 请求失败: "
                                f"{str(last_error)}]"
                            )
                        },

                        "finish_reason": None,
                    }
                ],
            }


            yield (
                "data: "
                + json.dumps(
                    error_payload,
                    ensure_ascii=False
                )
                + "\n\n"
            )


            finish_payload = {

                "id": completion_id,

                "object": "chat.completion.chunk",

                "created": created,

                "model": target_model,

                "choices": [
                    {
                        "index": 0,

                        "delta": {},

                        "finish_reason": "stop",
                    }
                ],
            }


            yield (
                "data: "
                + json.dumps(
                    finish_payload,
                    ensure_ascii=False
                )
                + "\n\n"
            )


            yield "data: [DONE]\n\n"


        return StreamingResponse(

            event_stream(),

            media_type="text/event-stream",

            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",

                # 防止某些代理转换 SSE
                "Content-Encoding": "identity",
            },
        )


    # =====================================================
    # NON STREAM
    # =====================================================

    last_error = None


    for attempt in range(retry_count):


        current_client = get_next_client()


        try:


            response = (
                await current_client.aio.models.generate_content(
                    model=target_model,
                    contents=prompt_text,
                    config=config,
                )
            )


            return {

                "id": (
                    "chatcmpl-gemini-"
                    + str(int(time.time() * 1000))
                ),

                "object": "chat.completion",

                "created": int(time.time()),

                "model": target_model,

                "choices": [
                    {
                        "index": 0,

                        "message": {
                            "role": "assistant",
                            "content": response.text or "",
                        },

                        "finish_reason": "stop",
                    }
                ],
            }


        except Exception as e:


            last_error = e


            print(
                "[Gemini Error] "
                f"model={target_model} "
                f"attempt={attempt + 1}/{retry_count} "
                f"error_type={type(e).__name__} "
                f"error={repr(e)}"
            )


            if attempt < retry_count - 1:

                await asyncio.sleep(0.5)


    # =====================================================
    # 所有 KEY 都失败
    # =====================================================

    raise HTTPException(

        status_code=503,

        detail={
            "message": "所有 Gemini API Key 请求均失败",

            "model": target_model,

            "error_type": (
                type(last_error).__name__
                if last_error
                else None
            ),

            "error": (
                str(last_error)
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
