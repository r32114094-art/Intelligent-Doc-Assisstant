#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能文档问答助手 - 核心业务逻辑层
从原 11_Q&A_Assistant.py 提取，供 FastAPI 后端调用
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv

# 加载 .env（向上查找到项目根目录）
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(env_path)

from hello_agents.tools import MemoryTool, RAGTool


class PDFLearningAssistant:
    """智能文档问答助手"""

    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 初始化工具
        self.memory_tool = MemoryTool(user_id=user_id)
        self.rag_tool = RAGTool(rag_namespace=f"pdf_{user_id}")

        # 学习统计
        self.stats = {
            "session_start": datetime.now(),
            "documents_loaded": 0,
            "questions_asked": 0,
            "concepts_learned": 0,
        }

        # 当前加载的文档
        self.current_document: Optional[str] = None

    def load_document(self, pdf_path: str) -> Dict[str, Any]:
        """加载 PDF 文档到知识库"""
        if not os.path.exists(pdf_path):
            return {"success": False, "message": f"文件不存在: {pdf_path}"}

        start_time = time.time()

        try:
            result = self.rag_tool.run({
                "action": "add_document",
                "file_path": pdf_path,
                "chunk_size": 1000,
                "chunk_overlap": 200,
            })

            process_time = time.time() - start_time
            self.current_document = os.path.basename(pdf_path)
            self.stats["documents_loaded"] += 1

            self.memory_tool.run({
                "action": "add",
                "content": f"加载了文档《{self.current_document}》",
                "memory_type": "episodic",
                "importance": 0.9,
                "event_type": "document_loaded",
                "session_id": self.session_id,
            })

            return {
                "success": True,
                "message": f"加载成功！(耗时: {process_time:.1f}秒)",
                "document": self.current_document,
            }
        except Exception as e:
            return {"success": False, "message": f"加载失败: {str(e)}"}

    def ask(self, question: str, use_advanced_search: bool = True) -> str:
        """向文档提问"""
        if not self.current_document:
            return "⚠️ 请先加载文档！"

        self.memory_tool.run({
            "action": "add",
            "content": f"提问: {question}",
            "memory_type": "working",
            "importance": 0.6,
            "session_id": self.session_id,
        })

        answer = self.rag_tool.run({
            "action": "ask",
            "question": question,
            "limit": 5,
            "enable_advanced_search": use_advanced_search,
            "enable_mqe": use_advanced_search,
            "enable_hyde": use_advanced_search,
        })

        self.memory_tool.run({
            "action": "add",
            "content": f"关于'{question}'的学习",
            "memory_type": "episodic",
            "importance": 0.7,
            "event_type": "qa_interaction",
            "session_id": self.session_id,
        })

        self.stats["questions_asked"] += 1
        return answer

    def add_note(self, content: str, concept: Optional[str] = None):
        """添加学习笔记"""
        self.memory_tool.run({
            "action": "add",
            "content": content,
            "memory_type": "semantic",
            "importance": 0.8,
            "concept": concept or "general",
            "session_id": self.session_id,
        })
        self.stats["concepts_learned"] += 1

    def recall(self, query: str, limit: int = 5) -> str:
        """回顾学习历程"""
        result = self.memory_tool.run({
            "action": "search",
            "query": query,
            "limit": limit,
        })
        return result

    def get_stats(self) -> Dict[str, Any]:
        """获取学习统计"""
        duration = (datetime.now() - self.stats["session_start"]).total_seconds()
        return {
            "会话时长": f"{duration:.0f}秒",
            "加载文档": self.stats["documents_loaded"],
            "提问次数": self.stats["questions_asked"],
            "学习笔记": self.stats["concepts_learned"],
            "当前文档": self.current_document or "未加载",
        }

    def generate_report(self, save_to_file: bool = True) -> Dict[str, Any]:
        """生成学习报告"""
        memory_summary = self.memory_tool.run({"action": "summary", "limit": 10})
        rag_stats = self.rag_tool.run({"action": "stats"})

        duration = (datetime.now() - self.stats["session_start"]).total_seconds()
        report = {
            "session_info": {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "start_time": self.stats["session_start"].isoformat(),
                "duration_seconds": duration,
            },
            "learning_metrics": {
                "documents_loaded": self.stats["documents_loaded"],
                "questions_asked": self.stats["questions_asked"],
                "concepts_learned": self.stats["concepts_learned"],
            },
            "memory_summary": memory_summary,
            "rag_status": rag_stats,
        }

        if save_to_file:
            report_file = f"learning_report_{self.session_id}.json"
            try:
                with open(report_file, "w", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2, default=str)
                report["report_file"] = report_file
            except Exception as e:
                report["save_error"] = str(e)

        return report
