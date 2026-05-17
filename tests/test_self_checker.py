"""
答案自检器单元测试

测试规则层拦截和 LLM 自检逻辑。
"""

from unittest.mock import MagicMock, patch

import pytest

from app.core.self_checker import SelfChecker


class TestSelfChecker:
    """SelfChecker 测试"""

    def test_rule_check_empty_chunks(self):
        """测试空检索结果直接拦截"""
        checker = SelfChecker()
        result = checker.check(
            question="测试问题",
            answer="测试答案",
            retrieved_chunks=[],
        )
        assert result["has_evidence"] is False
        assert result["hallucination_risk"] == "high"
        assert "未检索到" in result["reason"]

    def test_rule_check_empty_answer(self):
        """测试空答案直接拦截"""
        checker = SelfChecker()
        result = checker.check(
            question="测试问题",
            answer="",
            retrieved_chunks=[{"text": "证据", "page": 1}],
        )
        assert result["has_evidence"] is False

    def test_rule_check_vague_answer(self):
        """测试规避性答案直接拦截"""
        checker = SelfChecker()
        result = checker.check(
            question="测试问题",
            answer="我不知道这个问题的答案",
            retrieved_chunks=[{"text": "证据", "page": 1}],
        )
        assert result["has_evidence"] is False
        assert "规避" in result["reason"]

    @patch("app.core.self_checker.ChatOpenAI")
    def test_llm_check_pass(self, mock_llm_cls):
        """测试 LLM 自检通过"""
        mock_response = MagicMock()
        mock_response.content = '{"has_evidence": true, "confidence": 0.9, "hallucination_risk": "low", "out_of_scope": false, "reason": "有充分依据"}'
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_cls.return_value = mock_llm

        checker = SelfChecker()
        result = checker.check(
            question="测试问题",
            answer="基于检索证据生成的合理答案内容",
            retrieved_chunks=[{"text": "证据内容", "page": 1}],
        )

        assert result["has_evidence"] is True
        assert result["confidence"] == 0.9
        assert result["hallucination_risk"] == "low"

    @patch("app.core.self_checker.ChatOpenAI")
    def test_llm_check_json_in_code_block(self, mock_llm_cls):
        """测试从 markdown 代码块解析 JSON"""
        mock_response = MagicMock()
        mock_response.content = '```json\n{"has_evidence": false, "confidence": 0.1, "hallucination_risk": "high", "out_of_scope": true, "reason": "test"}\n```'
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_cls.return_value = mock_llm

        checker = SelfChecker()
        result = checker.check(
            question="测试问题",
            answer="答案",
            retrieved_chunks=[{"text": "证据", "page": 1}],
        )

        assert result["has_evidence"] is False

    @patch("app.core.self_checker.ChatOpenAI")
    def test_llm_check_failure_fallback(self, mock_llm_cls):
        """测试 LLM 调用失败时的保守回退"""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API error")
        mock_llm_cls.return_value = mock_llm

        checker = SelfChecker()
        result = checker.check(
            question="测试问题",
            answer="一段足够长的答案内容用于测试LLM调用失败的保守回退",
            retrieved_chunks=[{"text": "证据", "page": 1}],
        )

        assert result["has_evidence"] is False
        assert result["confidence"] == 0.0
        assert "异常" in result["reason"]
