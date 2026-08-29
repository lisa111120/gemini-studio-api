import os
import time
import json
import itertools
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from google import genai
from google.genai import types
from google.genai.errors import APIError, ServerError

app = FastAPI(title="Gemini 3.7 Flash Dedicated Router")

# 首页健康检查（解决访问根目录 404 的问题）
@app.get("/")
async def root():
    return {"status": "running", "service": "Gemini 3.7 Flash Dedicated Router"}

# 从环境变量读取多个 API Key（逗号分隔）
keys_env = os.environ.get("GEMINI_API_KEYS", "")
API_KEYS = [k.strip() for k in keys_env.split(",") if k.strip()]
key_cycle = itertools.cycle(API_KEYS) if API_KEYS else None

def get_next_client() -> genai.Client:
    if not key_cycle:
        raise HTTPException(status_code=500, detail="请在 Zeabur 环境变量中设置 GEMINI_API_KEYS")
    return genai.Client(api_key=next(key_cycle))

# 全解审核过滤 (BLOCK_NONE)
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

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = "gemini-3.7-flash"
    messages: List[ChatMessage]
    stream: Optional[bool] = True
    thinking_budget: Optional[int] = 4096

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    prompt_text = "\n".join([f"{msg.role}: {msg.content}" for msg in req.messages])
    target_model = "gemini-3.7-flash"
    
    config = types.GenerateContentConfig(
        safety_settings=GLOBAL_SAFETY_SETTINGS,
        thinking_config=types.ThinkingConfig(
            thinking_budget=req.thinking_budget
        ) if req.thinking_budget and req.thinking_budget > 0 else None
    )

    last_err_msg = ""
    retry_count = max(len(API_KEYS), 1)

    for attempt in range(retry_count):
        try:
            client = get_next_client()
            
            if req.stream:
                stream_obj = client.models.generate_content_stream(
                    model=target_model,
                    contents=prompt_text,
                    config=config,
                )
                
                def event_stream():
                    try:
                        for chunk in stream_obj:
                            if chunk.text:
                                # 使用标准的 json.dumps，确保双引号与转义 100% 正确
                                chunk_payload = {
                                    "id": "chatcmpl-gemini",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": target_model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": chunk.text},
                                        "finish_reason": None
                                    }]
                                }
                                yield f"data: {json.dumps(chunk_payload, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as stream_err:
                        err_payload = {
                            "choices": [{"delta": {"content": f"\n\n[网络中断: {str(stream_err)}]"}}]
                        }
                        yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"

                return StreamingResponse(
                    event_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no"
                    }
                )
            else:
                response = client.models.generate_content(
                    model=target_model,
                    contents=prompt_text,
                    config=config,
                )
                return {
                    "id": "chatcmpl-gemini",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": target_model,
                    "choices": [{
                        "message": {"role": "assistant", "content": response.text},
                        "finish_reason": "stop"
                    }]
                }

        except (ServerError, APIError, Exception) as e:
            last_err_msg = str(e)
            print(f"[Warn] 3.7 Flash 遇到重试: {last_err_msg}，正在切下一个 Key...")
            time.sleep(1)
            continue

    raise HTTPException(status_code=503, detail=f"Gemini 3.7 Flash 请求失败: {last_err_msg}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
