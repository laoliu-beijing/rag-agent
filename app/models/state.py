"""
LangGraph Agent 状态定义

使用 TypedDict 定义 Agent 执行过程中各节点共享的状态结构，
所有节点通过读写 state 对象传递数据。
"""

from typing import Any, Literal, TypedDict

from app.models.schemas import Source


class Chunk(TypedDict):
    """文档 chunk 结构"""

    text: str
    doc_id: str
    page: int
    chunk_type: Literal["text", "table"]
    score: float


class CheckResult(TypedDict):
    """自检节点输出结构"""

    has_evidence: bool
    confidence: float
    hallucination_risk: Literal["low", "medium", "high"]
    out_of_scope: bool
    reason: str


class AgentState(TypedDict):
    """LangGraph Agent 全局状态"""

    question: str
    doc_ids: list[str] | None
    retrieved_chunks: list[Chunk]
    answer: str | None
    sources: list[Source]
    check_result: CheckResult | None
    final_output: dict[str, Any] | None
    error: str | None
