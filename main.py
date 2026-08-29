import os
import time
import json
import itertools
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

from google import genai
from google.genai import types


app = FastAPI(title="Gemini OpenAI Compatible Router")


# =========================================================
# API Keys
# =========================================================

keys_env = os.environ.get("GEMINI_API_KEYS", "")

API_KEYS = [
    k.strip()
    for k in keys_env.split(",")
    if k.strip()
]

CLIENT_POOL = [
    genai.Client(api_key=k)
    for k in API_KEYS
]

client_cycle = itertools.cycle(CLIENT_POOL) if CLIENT_POOL else None


def get_next_client():
    if not client_cycle:
        raise HTTPException(
            status_code=500,
            detail="请在环境变量中设置 GEMINI_API_KEYS"
        )

    return next(client_cycle)


# =========================================================
# Safety Settings
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
# Request Models
# =========================================================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):

    # 必填
    # Chatbox 传什么模型，就原样传给 Gemini
    model: str

    messages: List[ChatMessage]

    stream: Optional[bool] = True

    # 默认高思考
    thinking_level: Optional[str] = "high"


# =========================================================
# Config
# =========================================================

def build_config(req: ChatRequest):

    thinking_level = req.thinking_level

    # 非法值统一回退 high
    if thinking_level not in ("low", "medium", "high", None):
        thinking_level = "high"

    thinking_config = None

    if thinking_level:
        thinking_config = types.ThinkingConfig(
            thinking_level=thinking_level
        )

    return types.GenerateContentConfig(
        safety_settings=GLOBAL_SAFETY_SETTINGS,
        thinking_config=thinking_config,
    )


# =========================================================
# Root
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
# Models Endpoint
# =========================================================
#
# 这里只是为了兼容部分 OpenAI 客户端。
#
# 真正请求时不会对 model 做任何限制或替换。
#
# =========================================================

@app.get("/v1/models")
async def models():

    return {
        "object": "list",
        "data": []
    }


# =========================================================
# Chat Completions
# =========================================================

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):

    if not CLIENT_POOL:
        raise HTTPException(
            status_code=500,
            detail="未配置 GEMINI_API_KEYS",
        )

    # =====================================================
    # MODEL 完全透传
    # =====================================================
    #
    # Chatbox:
    #
    # gemini-3.7-flash
    #        ↓
    # Gemini API:
    # gemini-3.7-flash
    #
    # 不修改
    # 不替换
    # 不使用默认模型
    # 不检查白名单
    #
    # =====================================================

    target_model = req.model


    # =====================================================
    # Messages
    # =====================================================

    prompt_text = "\n".join(
        f"{msg.role}: {msg.content}"
        for msg in req.messages
    )


    config = build_config(req)

    retry_count = len(CLIENT_POOL)


    # =====================================================
    # STREAM
    # =====================================================

    if req.stream:

        async def event_stream():

            last_error = None


            # =================================================
            # 每个 Key 最多尝试一次
            # =================================================

            for attempt in range(retry_count):

                current_client = get_next_client()

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


                        chunk_payload = {
                            "id": "chatcmpl-gemini",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
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
                                chunk_payload,
                                ensure_ascii=False
                            )
                            + "\n\n"
                        )


                    # =================================================
                    # 正常完成
                    # =================================================

                    final_payload = {
                        "id": "chatcmpl-gemini",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
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
                            final_payload,
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
                    # 已经输出过内容
                    # =================================================
                    #
                    # 这种情况下不能换 Key 从头重新生成。
                    #
                    # 否则 Chatbox 会收到重复回答。
                    #
                    # =================================================

                    if started:

                        error_payload = {
                            "id": "chatcmpl-gemini-error",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
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
                                    "finish_reason": "stop",
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

                        yield "data: [DONE]\n\n"

                        return


                    # =================================================
                    # Gemini 还没有输出任何 token
                    # =================================================
                    #
                    # 此时可以安全换下一个 Key。
                    #
                    # =================================================

                    if attempt < retry_count - 1:

                        await asyncio.sleep(0.5)

                        continue


            # =================================================
            # 所有 Key 都失败
            # =================================================

            error_payload = {
                "id": "chatcmpl-gemini-error",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
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
                        "finish_reason": "stop",
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

            yield "data: [DONE]\n\n"


        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
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
                "id": "chatcmpl-gemini",
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
    # 所有 Key 都失败
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
# Run
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
