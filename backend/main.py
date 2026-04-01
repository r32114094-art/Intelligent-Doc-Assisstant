#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能文档问答助手 - FastAPI 后端
提供 RESTful API 供前端调用
"""

import os
import shutil
import tempfile
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from assistant import PDFLearningAssistant

# ────────────────────────────────────────────
# Pydantic 请求/响应模型
# ────────────────────────────────────────────

class InitRequest(BaseModel):
    user_id: str = "web_user"

class ChatRequest(BaseModel):
    message: str
    use_advanced_search: bool = True

class NoteRequest(BaseModel):
    content: str
    concept: Optional[str] = None

class RecallRequest(BaseModel):
    query: str
    limit: int = 5

class MessageResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None

# ────────────────────────────────────────────
# FastAPI 应用
# ────────────────────────────────────────────

app = FastAPI(
    title="智能文档问答助手 API",
    description="基于 HelloAgents 的智能文档问答系统",
    version="1.0.0",
)

# CORS - 允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局助手实例
assistant: Optional[PDFLearningAssistant] = None


def get_assistant() -> PDFLearningAssistant:
    """获取助手实例，未初始化则抛出异常"""
    if assistant is None:
        raise HTTPException(status_code=400, detail="助手尚未初始化，请先调用 /api/init")
    return assistant


# ────────────────────────────────────────────
# API 路由
# ────────────────────────────────────────────

@app.post("/api/init", response_model=MessageResponse)
async def init_assistant(req: InitRequest):
    """初始化助手"""
    global assistant
    user_id = req.user_id or "web_user"
    assistant = PDFLearningAssistant(user_id=user_id)
    return MessageResponse(
        success=True,
        message=f"助手已初始化 (用户: {user_id})",
        data={"user_id": user_id, "session_id": assistant.session_id},
    )


@app.post("/api/upload", response_model=MessageResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """上传 PDF 文件并加载到知识库"""
    ast = get_assistant()

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    # 保存临时文件
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
            data={"document": result.get("document")},
        )
    finally:
        # 清理临时文件
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/api/chat", response_model=MessageResponse)
async def chat(req: ChatRequest):
    """智能问答"""
    ast = get_assistant()

    if not req.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    # 判断是回顾还是问答
    recall_keywords = ["之前", "学过", "回顾", "历史", "记得"]
    if any(kw in req.message for kw in recall_keywords):
        response = ast.recall(req.message)
        resp_type = "recall"
    else:
        response = ast.ask(req.message, use_advanced_search=req.use_advanced_search)
        resp_type = "answer"

    return MessageResponse(
        success=True,
        message=response,
        data={"type": resp_type},
    )


@app.post("/api/note", response_model=MessageResponse)
async def add_note(req: NoteRequest):
    """添加学习笔记"""
    ast = get_assistant()

    if not req.content.strip():
        raise HTTPException(status_code=400, detail="笔记内容不能为空")

    ast.add_note(req.content, req.concept)
    return MessageResponse(
        success=True,
        message=f"笔记已保存: {req.content[:50]}...",
    )


@app.post("/api/recall", response_model=MessageResponse)
async def recall_memory(req: RecallRequest):
    """回顾学习历程"""
    ast = get_assistant()
    result = ast.recall(req.query, req.limit)
    return MessageResponse(success=True, message=result)


@app.get("/api/stats", response_model=MessageResponse)
async def get_stats():
    """获取学习统计"""
    ast = get_assistant()
    stats = ast.get_stats()
    return MessageResponse(success=True, message="获取成功", data=stats)


@app.get("/api/report", response_model=MessageResponse)
async def get_report():
    """生成学习报告"""
    ast = get_assistant()
    report = ast.generate_report(save_to_file=True)
    return MessageResponse(success=True, message="报告已生成", data=report)


# ────────────────────────────────────────────
# 静态文件 & 前端页面
# ────────────────────────────────────────────

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
async def serve_frontend():
    """返回前端首页"""
    return FileResponse(os.path.join(frontend_dir, "index.html"))


# ────────────────────────────────────────────
# 入口
# ────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("  智能文档问答助手 — FastAPI 后端")
    print("=" * 60)
    print("  前端地址: http://localhost:8000")
    print("  API 文档: http://localhost:8000/docs")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
