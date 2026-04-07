"""
智能文档问答助手 — FastAPI 后端

完全自主实现的 RAG 管道，不依赖 hello-agents 库。
核心模块：parser → chunker → embedder → vector_store → retriever → reranker → llm_client
"""

import os
import shutil
import tempfile
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from assistant import DocumentAssistant

# ────────────────────────────────────────────
# 请求/响应模型
# ────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


class DeleteDocRequest(BaseModel):
    source: str


class MessageResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None


# ────────────────────────────────────────────
# 应用初始化
# ────────────────────────────────────────────

app = FastAPI(title="智能文档问答助手", version="2.0")

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# 全局助手实例
_assistant: Optional[DocumentAssistant] = None


def get_assistant() -> DocumentAssistant:
    global _assistant
    if _assistant is None or not _assistant.initialized:
        raise HTTPException(status_code=400, detail="请先初始化助手")
    return _assistant


# ────────────────────────────────────────────
# 页面路由
# ────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# ────────────────────────────────────────────
# API 路由
# ────────────────────────────────────────────

@app.post("/api/init", response_model=MessageResponse)
async def init_assistant():
    """初始化 RAG 管道"""
    global _assistant
    try:
        _assistant = DocumentAssistant()
        _assistant.initialize()
        return MessageResponse(success=True, message="RAG 管道初始化成功")
    except Exception as e:
        return MessageResponse(success=False, message=f"初始化失败: {str(e)}")


@app.post("/api/upload", response_model=MessageResponse)
async def upload_document(file: UploadFile = File(...)):
    """上传文档到知识库"""
    ast = get_assistant()

    allowed_ext = {".pdf", ".md", ".txt", ".docx", ".csv", ".json"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {ext}")

    # 保存到临时目录
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)
    try:
        with open(tmp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        result = ast.load_document(tmp_path)
        return MessageResponse(
            success=result["success"],
            message=result["message"],
            data={"document": result.get("document"), "chunks": result.get("chunks")},
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/api/chat", response_model=MessageResponse)
async def chat(req: ChatRequest):
    """智能问答"""
    ast = get_assistant()
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    result = ast.ask(req.message)
    return MessageResponse(
        success=True,
        message=result["answer"],
        data={
            "type": "answer",
            "steps": result.get("steps", []),
            "total_time": result.get("total_time"),
        },
    )


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式智能问答 — SSE 实时推送管道步骤"""
    from fastapi.responses import StreamingResponse

    ast = get_assistant()
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    def event_generator():
        for event_data in ast.ask_streaming(req.message):
            yield f"data: {event_data}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/stats", response_model=MessageResponse)
async def get_stats():
    """获取统计信息"""
    ast = get_assistant()
    stats = ast.get_stats()
    return MessageResponse(success=True, message="获取成功", data=stats)


# ────────────────────────────────────────────
# 文档管理 API
# ────────────────────────────────────────────

@app.get("/api/documents", response_model=MessageResponse)
async def list_documents():
    """列出知识库中所有已索引的文档"""
    ast = get_assistant()
    try:
        documents = ast.store.list_documents()
        return MessageResponse(
            success=True,
            message=f"共 {len(documents)} 个文档",
            data={"documents": documents},
        )
    except Exception as e:
        return MessageResponse(success=False, message=f"查询失败: {str(e)}")


@app.post("/api/documents/delete", response_model=MessageResponse)
async def delete_document(req: DeleteDocRequest):
    """从知识库中删除指定文档"""
    ast = get_assistant()
    source = req.source.strip()
    if not source:
        raise HTTPException(status_code=400, detail="文档标识不能为空")

    try:
        count = ast.store.delete_by_source(source)
        if count == 0:
            return MessageResponse(success=False, message=f"未找到文档: {source}")
        return MessageResponse(
            success=True,
            message=f"已删除「{source}」的 {count} 个分块",
        )
    except Exception as e:
        return MessageResponse(success=False, message=f"删除失败: {str(e)}")


# ────────────────────────────────────────────
# 启动
# ────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("  智能文档问答助手 — FastAPI 后端 v2.0")
    print("  核心：自主实现 RAG 管道（无 hello-agents 依赖）")
    print("=" * 60)
    print(f"  前端地址: http://localhost:8000")
    print(f"  API 文档: http://localhost:8000/docs")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
