"""
答案自检器模块

对生成的答案进行质量审核，判断是否有依据、是否存在幻觉、是否超出范围。
采用双层防御机制：规则层硬拦截 + LLM 自检层。
"""

import json
import re

from langchain_openai import ChatOpenAI

from app.config.settings import get_settings
from app.models.state import CheckResult
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 自检 Prompt，要求 LLM 以 JSON 格式输出审核结果
SELF_CHECK_PROMPT_TEMPLATE = """你是答案质量审核员。请严格根据以下信息判断答案质量，不得宽松判断。

【用户问题】
{question}

【检索到的证据】
{evidence}

【生成的答案】
{answer}

请判断并输出 JSON，遵循以下规则：
1. has_evidence: 答案中的主要事实是否能在检索到的证据中找到对应？只要核心观点有依据即可为 true，细节表述差异不影响。
2. confidence: 0-1 之间，依据越充分分数越高。文档中有明确相关内容时，confidence 不应低于 0.6。
3. hallucination_risk: 只有当答案明显编造了证据中完全没有的信息时才为 "high"，正常推理和概括不为幻觉。
4. out_of_scope: 只有当答案完全脱离证据、使用大量外部知识时才为 true。基于证据的合理回答不为越界。
5. reason: 简要说明判断理由。

输出格式（仅输出 JSON，不要其他内容）：
{{"has_evidence": bool, "confidence": float, "hallucination_risk": "low|medium|high", "out_of_scope": bool, "reason": "str"}}
"""


class SelfChecker:
    """
    答案自检器

    提供双层防御：
    1. 规则层：零成本快速拦截明显问题
    2. LLM 层：深度语义审核
    """

    def __init__(self):
        """初始化自检器和大模型客户端"""
        self.settings = get_settings()
        self.llm = ChatOpenAI(
            model=self.settings.LLM_MODEL,
            api_key=self.settings.DASHSCOPE_API_KEY,
            base_url=self.settings.LLM_BASE_URL,
            temperature=0.0,  # 自检需要确定性输出
        )

    def check(
        self,
        question: str,
        answer: str,
        retrieved_chunks: list,
    ) -> CheckResult:
        """
        执行答案自检

        Args:
            question: 用户问题
            answer: 生成的答案
            retrieved_chunks: 检索到的证据片段

        Returns:
            CheckResult: 自检结果
        """
        # 第一层：规则硬拦截
        rule_result = self._rule_check(question, answer, retrieved_chunks)
        if rule_result is not None:
            logger.info(
                "规则层拦截触发",
                has_evidence=rule_result["has_evidence"],
                reason=rule_result["reason"],
            )
            return rule_result

        # 第二层：LLM 自检
        return self._llm_check(question, answer, retrieved_chunks)

    def _rule_check(
        self,
        question: str,
        answer: str,
        retrieved_chunks: list,
    ) -> CheckResult | None:
        """
        规则层硬拦截检查

        无需调用 LLM，零成本快速判断明显问题。

        Returns:
            CheckResult: 如果触发拦截则返回结果，否则返回 None 进入 LLM 层
        """
        # 1. 检索结果为空
        if not retrieved_chunks:
            return CheckResult(
                has_evidence=False,
                confidence=0.0,
                hallucination_risk="high",
                out_of_scope=True,
                reason="未检索到任何相关证据",
            )

        # 2. 答案为空或太短（但"信息不足"是合法回答）
        if not answer or (len(answer.strip()) < 10 and "信息不足" not in answer):
            return CheckResult(
                has_evidence=False,
                confidence=0.0,
                hallucination_risk="high",
                out_of_scope=True,
                reason="答案为空或过短",
            )

        # 3. 答案包含"不知道"、"不清楚"等规避词汇（"信息不足"除外，它是 Prompt 要求的规范回答）
        vague_patterns = ["不知道", "不清楚", "不了解", "无法确定", "没有相关信息"]
        if any(p in answer for p in vague_patterns):
            return CheckResult(
                has_evidence=False,
                confidence=0.0,
                hallucination_risk="high",
                out_of_scope=True,
                reason="答案包含规避性表述",
            )

        # 规则层未触发，进入 LLM 层
        return None

    def _llm_check(
        self,
        question: str,
        answer: str,
        retrieved_chunks: list,
    ) -> CheckResult:
        """
        LLM 深度自检

        调用大模型进行语义层面的质量审核。

        Returns:
            CheckResult: 自检结果
        """
        # 拼接证据文本
        evidence_text = "\n\n".join(
            f"[来源: 第{chunk.get('page', '?')}页] {chunk['text']}"
            for chunk in retrieved_chunks
        )

        prompt = SELF_CHECK_PROMPT_TEMPLATE.format(
            question=question,
            evidence=evidence_text,
            answer=answer,
        )

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            # 从 LLM 输出中提取 JSON
            result = self._parse_json(content)

            logger.info(
                "LLM 自检完成",
                has_evidence=result["has_evidence"],
                confidence=result["confidence"],
                hallucination_risk=result["hallucination_risk"],
            )

            return CheckResult(**result)

        except Exception as e:
            logger.error("LLM 自检失败", error=str(e))
            # 自检失败时保守处理：标记为无证据
            return CheckResult(
                has_evidence=False,
                confidence=0.0,
                hallucination_risk="high",
                out_of_scope=True,
                reason=f"自检过程异常: {str(e)}",
            )

    def _parse_json(self, text: str) -> dict:
        """
        从 LLM 输出中提取 JSON

        Args:
            text: LLM 原始输出文本

        Returns:
            dict: 解析后的 JSON 数据

        Raises:
            ValueError: 无法解析 JSON 时抛出
        """
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试从 markdown 代码块中提取
        pattern = r"```(?:json)?\s*([\s\S]*?)```"
        matches = re.findall(pattern, text)
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass

        # 尝试查找花括号包裹的 JSON
        pattern = r"(\{[\s\S]*\})"
        matches = re.findall(pattern, text)
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"无法从 LLM 输出中解析 JSON: {text[:200]}")
