import os
import itertools
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from google import genai
from google.genai import types
from google.genai.errors import APIError

app = FastAPI(title="Gemini Multi-Key Router")

# 从环境变量读取多个 API Key（用英文逗号分隔）
keys_env = os.environ.get("GEMINI_API_KEYS", "")
API_KEYS = [k.strip() for k in keys_env.split(",") if k.strip()]

key_cycle = itertools.cycle(API_KEYS) if API_KEYS else None

def get_next_client() -> genai.Client:
    if not key_cycle:
        raise HTTPException(status_code=500, detail="请在 Zeabur 环境变量中设置 GEMINI_API_KEYS")
    key = next(key_cycle)
    return genai.Client(api_key=key)

# 强制完全解除四大审核过滤 (BLOCK_NONE)
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
    thinking_budget: Optional[int] = 8192  # 默认高深度思考

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    prompt_text = "\n".join([f"{msg.role}: {msg.content}" for msg in req.messages])
    
    config = types.GenerateContentConfig(
        safety_settings=GLOBAL_SAFETY_SETTINGS,
        thinking_config=types.ThinkingConfig(
            thinking_budget=req.thinking_budget
        ) if req.thinking_budget and req.thinking_budget > 0 else None
    )

    last_err = None
    for _ in range(len(API_KEYS) if API_KEYS else 1):
        try:
            client = get_next_client()
            if req.stream:
                def event_stream():
                    response_stream = client.models.generate_content_stream(
                        model=req.model,
                        contents=prompt_text,
                        config=config,
                    )
                    for chunk in response_stream:
                        if chunk.text:
                            yield f"data: {{\"choices\": [{{\"delta\": {{\"content\": {repr(chunk.text)}}} }}]}}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(event_stream(), media_type="text/event-stream")
            else:
                response = client.models.generate_content(
                    model=req.model,
                    contents=prompt_text,
                    config=config,
                )
                return {
                    "choices": [{
                        "message": {"role": "assistant", "content": response.text}
                    }]
                }
        except APIError as e:
            print(f"[Warn] Key 触发异常 (Code {e.code})，正在切换下一个 Key...")
            last_err = e
            continue
        except Exception as e:
            last_err = e
            continue

    raise HTTPException(status_code=500, detail=f"请求失败: {str(last_err)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
