"""
问答 API 路由

提供基于 RAG Agent 的智能问答接口。
"""

from fastapi import APIRouter, HTTPException

from app.core.agent import build_agent
from app.models.schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["query"])

# Agent 单例
agent = build_agent()


@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    提交问题并获取答案

    Agent 执行流程：
    1. 向量检索相关文档片段
    2. 基于检索结果生成答案
    3. 自检答案质量（依据/幻觉检查）
    4. 通过则返回答案和来源，不通过则返回拒答文案
    """
    try:
        result = agent.run(
            question=request.question,
            doc_ids=request.doc_ids,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答处理失败: {str(e)}")
