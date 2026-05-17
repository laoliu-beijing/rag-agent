"""
LangGraph Agent 集成测试

测试完整的检索-生成-自检流程，使用 mock 避免真实 API 调用。
"""

from unittest.mock import MagicMock, patch

import pytest

from app.core.agent import RAGAgent, REJECTION_MESSAGE
from app.models.state import AgentState


class TestRAGAgent:
    """RAGAgent 测试"""

    @patch("app.core.agent.ChromaRetriever")
    @patch("app.core.agent.SelfChecker")
    @patch("app.core.agent.ChatOpenAI")
    def test_successful_qa_flow(self, mock_llm_cls, mock_checker_cls, mock_retriever_cls):
        """测试正常问答流程"""
        # mock 检索器
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"text": "表面粗糙度 Ra 不应大于 3.2μm", "doc_id": "doc-1", "page": 5, "chunk_type": "text", "score": 0.9}
        ]
        mock_retriever_cls.return_value = mock_retriever

        # mock LLM 生成
        mock_response = MagicMock()
        mock_response.content = "键槽表面粗糙度 Ra 不应大于 3.2μm。"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_cls.return_value = mock_llm

        # mock 自检器（通过）
        mock_checker = MagicMock()
        mock_checker.check.return_value = {
            "has_evidence": True,
            "confidence": 0.95,
            "hallucination_risk": "low",
            "out_of_scope": False,
            "reason": "有充分依据",
        }
        mock_checker_cls.return_value = mock_checker

        agent = RAGAgent()
        result = agent.run("表面粗糙度要求是什么？")

        assert "3.2μm" in result.answer
        assert result.has_evidence is True
        assert result.confidence == "high"
        assert len(result.sources) > 0

    @patch("app.core.agent.ChromaRetriever")
    @patch("app.core.agent.SelfChecker")
    @patch("app.core.agent.ChatOpenAI")
    def test_rejection_on_no_evidence(self, mock_llm_cls, mock_checker_cls, mock_retriever_cls):
        """测试无证据时拒答"""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []
        mock_retriever_cls.return_value = mock_retriever

        mock_checker = MagicMock()
        mock_checker.check.return_value = {
            "has_evidence": False,
            "confidence": 0.0,
            "hallucination_risk": "high",
            "out_of_scope": True,
            "reason": "无证据",
        }
        mock_checker_cls.return_value = mock_checker

        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        agent = RAGAgent()
        result = agent.run("这个标准是谁写的？")

        assert result.answer == REJECTION_MESSAGE
        assert result.has_evidence is False
        assert result.confidence == "none"
        assert result.sources == []

    @patch("app.core.agent.ChromaRetriever")
    @patch("app.core.agent.SelfChecker")
    @patch("app.core.agent.ChatOpenAI")
    def test_rejection_on_low_confidence(self, mock_llm_cls, mock_checker_cls, mock_retriever_cls):
        """测试低置信度时拒答"""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"text": "一些模糊的内容", "doc_id": "doc-1", "page": 1, "chunk_type": "text", "score": 0.9}
        ]
        mock_retriever_cls.return_value = mock_retriever

        mock_response = MagicMock()
        mock_response.content = "一个模糊的答案。"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_cls.return_value = mock_llm

        mock_checker = MagicMock()
        mock_checker.check.return_value = {
            "has_evidence": True,
            "confidence": 0.4,  # 低于 0.5 阈值，触发拒答
            "hallucination_risk": "medium",
            "out_of_scope": False,
            "reason": "置信度不足",
        }
        mock_checker_cls.return_value = mock_checker

        agent = RAGAgent()
        result = agent.run("测试问题")

        assert result.answer == REJECTION_MESSAGE
        assert result.has_evidence is False

    def test_decide_output_logic(self):
        """测试决策逻辑"""
        agent = RAGAgent()

        # 自检通过
        state_pass = {
            "check_result": {
                "has_evidence": True,
                "confidence": 0.9,
                "hallucination_risk": "low",
                "out_of_scope": False,
            }
        }
        assert agent._decide_output(state_pass) == "output"

        # 无证据
        state_fail = {
            "check_result": {
                "has_evidence": False,
                "confidence": 0.0,
                "hallucination_risk": "high",
                "out_of_scope": True,
            }
        }
        assert agent._decide_output(state_fail) == "reject"

        # 高风险
        state_risk = {
            "check_result": {
                "has_evidence": True,
                "confidence": 0.8,
                "hallucination_risk": "high",
                "out_of_scope": False,
            }
        }
        assert agent._decide_output(state_risk) == "reject"
