#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 消融实验评估脚本

使用人工标注的测试集（docx/test.md），对比不同 RAG 管道配置的效果。
评估维度：
  - Context Relevance: 检索到的文档是否相关
  - Faithfulness: 回答是否忠于检索内容（不编造）
  - Answer Relevance: 回答是否解决了问题
  - Rejection Accuracy: 对超纲题是否正确拒答

实验组：
  1. Baseline     — 纯向量检索，无 MQE/HyDE/Rerank
  2. +MQE         — 启用多查询扩展
  3. +HyDE        — 启用假设文档嵌入
  4. +Rerank      — 启用 Cross-Encoder 精排
  5. +MQE+HyDE    — 组合
  6. Full Pipeline — 全部启用（MQE + HyDE + Rerank）

用法：
  python eval_rag.py                        # 全量评测（6组×25题）
  python eval_rag.py -c quick -n 5          # 快速验证（基线+全量×前5题）
  python eval_rag.py -c baseline --dry-run  # 仅解析测试集，不实际评测
"""

import os
import re
import sys
import json
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

# 加载环境变量
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(env_path)


# ────────────────────────────────────────────
# 测试数据集解析（适配结构化 test.md）
# ────────────────────────────────────────────

@dataclass
class TestCase:
    """一条评测用例"""
    id: str                        # Q1, Q2, ...
    question: str
    ground_truth: str
    source: str = ""
    difficulty: str = ""           # easy / medium / hard / irrelevant
    expect_rejection: bool = False  # 超纲题应拒绝回答


# 难度映射：## 标题 → difficulty
_SECTION_MAP = {
    "简单问题": "easy",
    "中等问题": "medium",
    "困难问题": "hard",
    "无关问题": "irrelevant",
}


def parse_test_md(path: str) -> List[TestCase]:
    """
    解析结构化 test.md 文件。

    期望格式：
        ## 简单问题
        ### Q1
        - **Q:** ...
        - **A:** ...
        - **S:** ...
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    cases: List[TestCase] = []
    current_difficulty = "medium"

    # 按 ### Q 分割出每个题目块
    # 先找到所有 ## 节标题来确定难度区间
    lines = text.splitlines()
    # 构建行号 → 难度的映射
    section_starts: List[Tuple[int, str]] = []
    for i, line in enumerate(lines):
        stripped = line.strip().lstrip("#").strip()
        for keyword, diff in _SECTION_MAP.items():
            if stripped == keyword:
                section_starts.append((i, diff))
                break

    def get_difficulty_at_line(line_no: int) -> str:
        """根据行号返回所属的难度分类"""
        result = "medium"
        for start_line, diff in section_starts:
            if line_no >= start_line:
                result = diff
        return result

    # 找到所有 ### Qn 的位置
    q_pattern = re.compile(r'^###\s+(Q\d+)\s*$')
    q_positions: List[Tuple[int, str]] = []  # (line_no, id)
    for i, line in enumerate(lines):
        m = q_pattern.match(line.strip())
        if m:
            q_positions.append((i, m.group(1)))

    # 为每个题目提取内容块
    for idx, (start_line, qid) in enumerate(q_positions):
        end_line = q_positions[idx + 1][0] if idx + 1 < len(q_positions) else len(lines)
        block = "\n".join(lines[start_line:end_line])
        difficulty = get_difficulty_at_line(start_line)

        # 提取 Q / A / S 字段
        q_match = re.search(r'-\s*\*\*Q:\*\*\s*(.+)', block)
        a_match = re.search(r'-\s*\*\*A:\*\*\s*(.+)', block)
        s_match = re.search(r'-\s*\*\*S:\*\*\s*(.*)', block)

        if not q_match:
            continue

        question = q_match.group(1).strip()
        answer = a_match.group(1).strip() if a_match else ""
        source = s_match.group(1).strip() if s_match else ""

        is_rejection = difficulty == "irrelevant" or "文档中未提及" in answer

        cases.append(TestCase(
            id=qid,
            question=question,
            ground_truth=answer,
            source=source,
            difficulty=difficulty,
            expect_rejection=is_rejection,
        ))

    return cases


# ────────────────────────────────────────────
# LLM 评分器
# ────────────────────────────────────────────

class LLMJudge:
    """使用 DeepSeek API 做 LLM-as-Judge 评估"""

    def __init__(self, llm_client):
        self.llm = llm_client

    def _call(self, system: str, user: str) -> str:
        try:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            return self.llm.generate(messages, temperature=0.3, max_tokens=1024) or ""
        except Exception as e:
            print(f"    [Judge] LLM 调用失败: {e}")
            return ""

    def _parse_score(self, response: str, metric: str) -> Dict:
        try:
            text = response.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            result = json.loads(text)
            return {
                "metric": metric,
                "score": max(1, min(5, int(result.get("score", 3)))),
                "reason": result.get("reason", ""),
            }
        except Exception:
            nums = re.findall(r"[1-5]", response)
            return {
                "metric": metric,
                "score": int(nums[0]) if nums else 3,
                "reason": response[:100],
            }

    def score_context_relevance(self, question: str, contexts: List[str]) -> Dict:
        ctx = "\n---\n".join(contexts[:5])
        system = """你是公正的评估专家。评估检索到的文档上下文与问题的相关性。
评分（1-5）：5=完全覆盖 4=基本覆盖 3=部分相关 2=大部分无关 1=完全无关
JSON回复：{"score": <1-5>, "reason": "<原因>"}"""
        return self._parse_score(self._call(system, f"问题：{question}\n\n上下文：\n{ctx}"), "context_relevance")

    def score_faithfulness(self, question: str, answer: str, contexts: List[str]) -> Dict:
        ctx = "\n---\n".join(contexts[:5])
        system = """你是公正的评估专家。评估回答是否忠实于上下文（无编造/幻觉）。
评分（1-5）：5=完全忠实 4=基本忠实 3=部分有依据 2=较多无依据 1=大部分编造
JSON回复：{"score": <1-5>, "reason": "<原因>"}"""
        return self._parse_score(self._call(system, f"问题：{question}\n\n回答：{answer}\n\n上下文：\n{ctx}"), "faithfulness")

    def score_answer_relevance(self, question: str, answer: str, ground_truth: str) -> Dict:
        system = """你是公正的评估专家。对比回答与标准答案，评估回答是否正确、完整地解决了问题。
评分（1-5）：5=精准完整 4=基本正确 3=部分正确 2=偏题较多 1=完全错误
JSON回复：{"score": <1-5>, "reason": "<原因>"}"""
        return self._parse_score(
            self._call(system, f"问题：{question}\n\n标准答案：{ground_truth}\n\n系统回答：{answer}"),
            "answer_relevance",
        )

    def score_rejection(self, question: str, answer: str) -> Dict:
        """评分：对超纲题是否正确拒绝回答"""
        system = """你是公正的评估专家。该问题在文档中并无相关内容，系统应当拒绝回答或说明无相关信息。
评估系统是否正确拒答（而非编造答案）。
评分（1-5）：5=明确拒答 4=表示不确定 3=部分回答但承认不确定 2=大部分编造 1=完全编造
JSON回复：{"score": <1-5>, "reason": "<原因>"}"""
        return self._parse_score(self._call(system, f"问题：{question}\n\n系统回答：{answer}"), "rejection_accuracy")


# ────────────────────────────────────────────
# 管道执行器
# ────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """管道配置"""
    name: str
    enable_mqe: bool = False
    enable_hyde: bool = False
    enable_rerank: bool = False
    enable_bm25: bool = True


# V4 消融配置：隔离 BM25 贡献，禁用 Rerank
ABLATION_CONFIGS = [
    PipelineConfig("Dense Only",                 False, False, False, False),   # 纯向量基线
    PipelineConfig("Dense+MQE+HyDE",             True,  True,  False, False),   # 纯向量 + 高级检索
    PipelineConfig("Hybrid (Dense+BM25)",        False, False, False, True),    # 混合检索基线
    PipelineConfig("Hybrid+MQE+HyDE",            True,  True,  False, True),    # 混合检索 + 高级检索
]


def run_single_query(assistant, question: str, config: PipelineConfig) -> Tuple[str, List[str], float]:
    """
    用指定配置执行单次 RAG 查询，返回 (answer, contexts, latency)
    """
    # 临时修改配置
    orig_mqe = assistant.config.enable_mqe
    orig_hyde = assistant.config.enable_hyde
    orig_rerank = assistant.config.enable_rerank
    orig_bm25 = assistant.config.enable_bm25

    assistant.config.enable_mqe = config.enable_mqe
    assistant.config.enable_hyde = config.enable_hyde
    assistant.config.enable_rerank = config.enable_rerank
    assistant.config.enable_bm25 = config.enable_bm25

    t0 = time.time()
    try:
        result = assistant.ask(question)
        answer = result.get("answer", "")

        # 直接从 ask() 返回值中获取 context（确保与 answer 来自同一次检索）
        contexts = result.get("contexts", [])
        latency = time.time() - t0

    except Exception as e:
        answer = f"❌ 执行失败: {e}"
        contexts = []
        latency = time.time() - t0
    finally:
        # 恢复配置
        assistant.config.enable_mqe = orig_mqe
        assistant.config.enable_hyde = orig_hyde
        assistant.config.enable_rerank = orig_rerank
        assistant.config.enable_bm25 = orig_bm25

    return answer, contexts, latency


# ────────────────────────────────────────────
# 主评估流程
# ────────────────────────────────────────────

def run_evaluation(
    assistant,
    test_cases: List[TestCase],
    configs: List[PipelineConfig] = None,
    delay: float = 2.0,
) -> Dict:
    """
    执行完整消融实验

    Args:
        assistant: DocumentAssistant 实例
        test_cases: 评测用例列表
        configs: 管道配置列表（默认 6 组）
        delay: 每次 API 调用间隔（秒）
    """
    if configs is None:
        configs = ABLATION_CONFIGS

    judge = LLMJudge(assistant.llm)
    all_results = {}
    eval_start = time.time()

    total_steps = len(configs) * len(test_cases)
    current_step = 0

    for ci, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"  实验组 {ci+1}/{len(configs)}: {config.name}")
        print(f"  MQE={config.enable_mqe} | HyDE={config.enable_hyde} | BM25={config.enable_bm25} | Rerank={config.enable_rerank}")
        print(f"{'='*60}")

        config_results = []

        for ti, tc in enumerate(test_cases):
            current_step += 1
            print(f"\n  [{current_step}/{total_steps}] [{tc.id}|{tc.difficulty}] {tc.question[:50]}...")

            # 执行查询
            answer, contexts, latency = run_single_query(assistant, tc.question, config)
            print(f"    ⏱ {latency:.1f}s | 检索到 {len(contexts)} 个上下文")

            time.sleep(delay)

            # 评分
            if tc.expect_rejection:
                # 超纲题：评估拒答能力
                rej = judge.score_rejection(tc.question, answer)
                print(f"    🚫 拒答评分: {rej['score']}/5")
                config_results.append({
                    "id": tc.id,
                    "question": tc.question,
                    "difficulty": tc.difficulty,
                    "expect_rejection": True,
                    "answer": answer[:200],
                    "latency": round(latency, 1),
                    "num_contexts": len(contexts),
                    "rejection_accuracy": rej,
                })
            else:
                # 正常题：三维评估
                cr = judge.score_context_relevance(tc.question, contexts) if contexts else {"score": 1, "reason": "无上下文", "metric": "context_relevance"}
                time.sleep(1)
                ff = judge.score_faithfulness(tc.question, answer, contexts) if contexts else {"score": 1, "reason": "无上下文", "metric": "faithfulness"}
                time.sleep(1)
                ar = judge.score_answer_relevance(tc.question, answer, tc.ground_truth)
                print(f"    📚 CR={cr['score']}/5 | 🔒 FF={ff['score']}/5 | 💬 AR={ar['score']}/5")

                config_results.append({
                    "id": tc.id,
                    "question": tc.question,
                    "difficulty": tc.difficulty,
                    "expect_rejection": False,
                    "answer": answer[:200],
                    "latency": round(latency, 1),
                    "num_contexts": len(contexts),
                    "context_relevance": cr,
                    "faithfulness": ff,
                    "answer_relevance": ar,
                })

            time.sleep(delay)

        all_results[config.name] = config_results

    # 汇总报告
    report = build_report(all_results, test_cases, time.time() - eval_start)
    print_report(report)

    # 保存
    report_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n💾 报告已保存: {report_file}")

    return report


def build_report(all_results: Dict, test_cases: List[TestCase], elapsed: float) -> Dict:
    """汇总消融实验报告"""
    summary_table = []

    for config_name, results in all_results.items():
        normal = [r for r in results if not r.get("expect_rejection")]
        rejection = [r for r in results if r.get("expect_rejection")]

        avg_cr = round(sum(r["context_relevance"]["score"] for r in normal) / len(normal), 2) if normal else 0
        avg_ff = round(sum(r["faithfulness"]["score"] for r in normal) / len(normal), 2) if normal else 0
        avg_ar = round(sum(r["answer_relevance"]["score"] for r in normal) / len(normal), 2) if normal else 0
        avg_rej = round(sum(r["rejection_accuracy"]["score"] for r in rejection) / len(rejection), 2) if rejection else 0
        avg_latency = round(sum(r["latency"] for r in results) / len(results), 1) if results else 0
        overall = round((avg_cr + avg_ff + avg_ar) / 3, 2)

        # 按难度分组统计
        by_difficulty = {}
        for diff in ("easy", "medium", "hard"):
            diff_results = [r for r in normal if r["difficulty"] == diff]
            if diff_results:
                by_difficulty[diff] = {
                    "count": len(diff_results),
                    "avg_cr": round(sum(r["context_relevance"]["score"] for r in diff_results) / len(diff_results), 2),
                    "avg_ff": round(sum(r["faithfulness"]["score"] for r in diff_results) / len(diff_results), 2),
                    "avg_ar": round(sum(r["answer_relevance"]["score"] for r in diff_results) / len(diff_results), 2),
                }

        summary_table.append({
            "config": config_name,
            "overall": overall,
            "context_relevance": avg_cr,
            "faithfulness": avg_ff,
            "answer_relevance": avg_ar,
            "rejection_accuracy": avg_rej,
            "avg_latency_s": avg_latency,
            "num_normal": len(normal),
            "num_rejection": len(rejection),
            "by_difficulty": by_difficulty,
        })

    return {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "num_test_cases": len(test_cases),
        "num_configs": len(all_results),
        "summary_table": summary_table,
        "details": {k: v for k, v in all_results.items()},
    }


def print_report(report: Dict):
    """打印对比报告"""
    if not report.get("success"):
        print(f"\n❌ 评估失败")
        return

    print(f"\n{'='*80}")
    print(f"  📊 RAG 消融实验报告")
    print(f"{'='*80}")
    print(f"  评估时间: {report['timestamp']}")
    print(f"  测试题数: {report['num_test_cases']}  |  配置数: {report['num_configs']}  |  耗时: {report['elapsed_seconds']}s")
    print(f"{'='*80}")
    print(f"  {'配置':<25} {'综合':>6} {'CR':>6} {'FF':>6} {'AR':>6} {'拒答':>6} {'延迟':>8}")
    print(f"  {'-'*73}")

    for row in report["summary_table"]:
        print(f"  {row['config']:<25} {row['overall']:>6} {row['context_relevance']:>6} {row['faithfulness']:>6} {row['answer_relevance']:>6} {row['rejection_accuracy']:>6} {row['avg_latency_s']:>7}s")

    # 按难度分组
    print(f"\n  {'─'*73}")
    print(f"  📈 各难度分组得分 (最后一组: {report['summary_table'][-1]['config']})")
    last_config = report["summary_table"][-1]
    for diff, stats in last_config.get("by_difficulty", {}).items():
        label = {"easy": "简单", "medium": "中等", "hard": "困难"}.get(diff, diff)
        print(f"    {label} ({stats['count']}题):  CR={stats['avg_cr']}  FF={stats['avg_ff']}  AR={stats['avg_ar']}")

    # 找最佳配置
    best = max(report["summary_table"], key=lambda x: x["overall"])
    print(f"\n  🏆 最佳配置: {best['config']}  (综合 {best['overall']}/5.00)")
    print(f"{'='*80}")


# ────────────────────────────────────────────
# CLI 入口
# ────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG 消融实验评估")
    parser.add_argument("--test-file", "-t", type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docx", "test.md"),
                        help="评测数据集路径 (默认: docx/test.md)")
    parser.add_argument("--configs", "-c", type=str, default="all",
                        choices=["all", "baseline", "full", "quick"],
                        help="实验组: all=6组, baseline=仅基线, full=仅全量, quick=基线+全量")
    parser.add_argument("--max-questions", "-n", type=int, default=0,
                        help="最多评测多少题 (0=全部)")
    parser.add_argument("--delay", "-d", type=float, default=2.0,
                        help="API 调用间隔秒数 (默认: 2.0)")
    parser.add_argument("--difficulty", type=str, default=None,
                        choices=["easy", "medium", "hard", "irrelevant"],
                        help="只评测指定难度的题目")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅解析测试集并打印，不实际执行评测")
    args = parser.parse_args()

    # 1. 解析测试集
    print(f"📋 加载测试集: {args.test_file}")
    test_cases = parse_test_md(args.test_file)
    print(f"   解析到 {len(test_cases)} 条测试用例:")
    for diff in ("easy", "medium", "hard", "irrelevant"):
        count = sum(1 for tc in test_cases if tc.difficulty == diff)
        if count:
            print(f"   - {diff}: {count} 条")

    # 按难度筛选
    if args.difficulty:
        test_cases = [tc for tc in test_cases if tc.difficulty == args.difficulty]
        print(f"   (筛选 {args.difficulty}，剩余 {len(test_cases)} 条)")

    if args.max_questions > 0:
        test_cases = test_cases[:args.max_questions]
        print(f"   (截取前 {args.max_questions} 条)")

    # dry-run 模式：仅打印解析结果
    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"  🔍 Dry Run — 测试集预览")
        print(f"{'='*60}")
        for tc in test_cases:
            rej_tag = " [拒答]" if tc.expect_rejection else ""
            print(f"  [{tc.id}|{tc.difficulty}{rej_tag}] {tc.question[:60]}")
            print(f"    答案: {tc.ground_truth[:80]}...")
            if tc.source:
                print(f"    来源: {tc.source}")
            print()
        sys.exit(0)

    # 2. 选择实验组
    if args.configs == "baseline":
        configs = [ABLATION_CONFIGS[0]]
    elif args.configs == "full":
        configs = [ABLATION_CONFIGS[-1]]
    elif args.configs == "quick":
        configs = [ABLATION_CONFIGS[0], ABLATION_CONFIGS[-1]]
    else:
        configs = ABLATION_CONFIGS

    # 3. 初始化助手
    print(f"\n🚀 初始化 RAG 管道...")
    from assistant import DocumentAssistant
    assistant = DocumentAssistant()
    assistant.initialize()
    print(f"   ✅ 初始化完成")

    # 4. 执行评估
    report = run_evaluation(assistant, test_cases, configs, delay=args.delay)
