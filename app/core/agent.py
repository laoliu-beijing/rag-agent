"""
LangGraph Agent 引擎

构建检索-生成-自检状态机，使用 LangGraph 定义节点和边，
实现可控的问答流程和明确的拒答机制。
"""

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from app.config.settings import get_settings
from app.core.retriever import ChromaRetriever
from app.core.self_checker import SelfChecker
from app.models.schemas import QueryResponse, Source
from app.models.state import AgentState
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 答案生成 Prompt，严格要求基于上下文
ANSWER_GENERATION_PROMPT = """你是一个专业的文档问答助手。请根据以下检索到的文档内容回答用户问题。

【重要规则】
1. 你只能基于下面提供的【检索内容】回答，不得使用外部知识。
2. 如果检索内容不足以回答问题，请直接回答"信息不足"。
3. 答案必须准确，引用具体的条款编号或页码。
4. 不要编造、推测或扩展未在检索内容中明确出现的信息。

【用户问题】
{question}

【检索内容】
{context}

请用中文回答。"""

# 拒答固定文案
REJECTION_MESSAGE = (
    "此问题已经超出我的范围，为了对您负责，我不会扩散回答该问题，"
    "我会记录下您的问题，等我有了明确的答案后再回答您，非常抱歉！"
)


class RAGAgent:
    """
    RAG Agent 实现

    使用 LangGraph 构建状态机，节点包括：
    - retrieve: 向量检索
    - generate: 答案生成
    - self_check: 答案自检
    - output: 输出答案
    - reject: 输出拒答
    """

    def __init__(self):
        """初始化 Agent 组件"""
        self.settings = get_settings()
        self.retriever = ChromaRetriever()
        self.self_checker = SelfChecker()
        self.llm = ChatOpenAI(
            model=self.settings.LLM_MODEL,
            api_key=self.settings.DASHSCOPE_API_KEY,
            base_url=self.settings.LLM_BASE_URL,
            temperature=0.1,
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        构建 LangGraph 状态机

        Returns:
            StateGraph: 编译后的状态图
        """
        workflow = StateGraph(AgentState)

        # 注册节点
        workflow.add_node("retrieve", self._node_retrieve)
        workflow.add_node("generate", self._node_generate)
        workflow.add_node("self_check", self._node_self_check)
        workflow.add_node("output", self._node_output)
        workflow.add_node("reject", self._node_reject)

        # 设置入口
        workflow.set_entry_point("retrieve")

        # retrieve -> generate
        workflow.add_edge("retrieve", "generate")

        # generate -> self_check
        workflow.add_edge("generate", "self_check")

        # self_check -> output 或 reject（条件边）
        workflow.add_conditional_edges(
            "self_check",
            self._decide_output,
            {
                "output": "output",
                "reject": "reject",
            },
        )

        # output/reject -> END
        workflow.add_edge("output", END)
        workflow.add_edge("reject", END)

        return workflow.compile()

    def _node_retrieve(self, state: AgentState) -> dict:
        """
        检索节点：执行向量检索

        Args:
            state: 当前 Agent 状态

        Returns:
            dict: 更新后的状态字段
        """
        logger.info(
            "Agent 检索节点开始",
            question=state["question"],
            doc_ids=state.get("doc_ids"),
        )

        try:
            chunks = self.retriever.search(
                query=state["question"],
                doc_ids=state.get("doc_ids"),
                top_k=self.settings.RETRIEVAL_TOP_K,
            )

            logger.info(
                "检索完成",
                chunks_found=len(chunks),
                top_score=chunks[0]["score"] if chunks else 0,
            )

            return {"retrieved_chunks": chunks}

        except Exception as e:
            logger.error("检索节点异常", error=str(e))
            return {"retrieved_chunks": [], "error": f"检索失败: {str(e)}"}

    def _node_generate(self, state: AgentState) -> dict:
        """
        生成节点：调用 LLM 生成答案

        Args:
            state: 当前 Agent 状态

        Returns:
            dict: 更新后的状态字段
        """
        chunks = state["retrieved_chunks"]

        # 如果检索结果为空或相似度过低，跳过生成
        if not chunks:
            logger.warning("检索结果为空，跳过生成")
            return {"answer": "", "sources": []}

        max_score = max(c["score"] for c in chunks)
        if max_score < self.settings.RETRIEVAL_MIN_SCORE:
            logger.warning(
                "最高相似度低于阈值，跳过生成",
                max_score=max_score,
                threshold=self.settings.RETRIEVAL_MIN_SCORE,
            )
            return {"answer": "", "sources": []}

        # 构建上下文：按相似度排序，拼接文本
        context_parts = []
        sources = []

        for chunk in sorted(chunks, key=lambda x: x["score"], reverse=True):
            context_parts.append(
                f"[第{chunk['page']}页] {chunk['text']}"
            )
            sources.append(
                Source(
                    doc_id=chunk["doc_id"],
                    doc_name=chunk["doc_id"],  # 实际使用时从 metadata 获取文件名
                    page=chunk["page"],
                    chunk_text=chunk["text"][:200],  # 截取前 200 字
                    score=chunk["score"],
                )
            )

        context = "\n\n".join(context_parts)
        prompt = ANSWER_GENERATION_PROMPT.format(
            question=state["question"],
            context=context,
        )

        try:
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)

            logger.info(
                "答案生成完成",
                answer_length=len(answer),
                sources_count=len(sources),
            )

            return {"answer": answer, "sources": sources}

        except Exception as e:
            logger.error("生成节点异常", error=str(e))
            return {"answer": "", "sources": [], "error": f"生成失败: {str(e)}"}

    def _node_self_check(self, state: AgentState) -> dict:
        """
        自检节点：对答案进行质量审核

        Args:
            state: 当前 Agent 状态

        Returns:
            dict: 更新后的状态字段
        """
        # 如果前面步骤已出错或答案为空，标记为失败
        if state.get("error") or not state.get("answer"):
            return {
                "check_result": {
                    "has_evidence": False,
                    "confidence": 0.0,
                    "hallucination_risk": "high",
                    "out_of_scope": True,
                    "reason": "检索或生成失败",
                }
            }

        result = self.self_checker.check(
            question=state["question"],
            answer=state["answer"],
            retrieved_chunks=state["retrieved_chunks"],
        )

        return {"check_result": result}

    def _decide_output(self, state: AgentState) -> str:
        """
        决策函数：根据自检结果决定输出或拒答

        Args:
            state: 当前 Agent 状态

        Returns:
            str: "output" 或 "reject"
        """
        check = state.get("check_result")
        if not check:
            return "reject"

        # 任一条件不满足即拒答
        if not check["has_evidence"]:
            return "reject"
        if check["hallucination_risk"] == "high":
            return "reject"
        if check["confidence"] < 0.5:
            return "reject"
        if check["out_of_scope"]:
            return "reject"

        return "output"

    def _node_output(self, state: AgentState) -> dict:
        """
        输出节点：组装最终答案响应

        Args:
            state: 当前 Agent 状态

        Returns:
            dict: 更新后的状态字段
        """
        return {
            "final_output": {
                "answer": state["answer"],
                "sources": [s.model_dump() for s in state["sources"]],
                "confidence": "high" if state["check_result"]["confidence"] > 0.8 else "medium",
                "has_evidence": True,
            }
        }

    def _node_reject(self, state: AgentState) -> dict:
        """
        拒答节点：返回固定拒答文案

        Args:
            state: 当前 Agent 状态

        Returns:
            dict: 更新后的状态字段
        """
        return {
            "final_output": {
                "answer": REJECTION_MESSAGE,
                "sources": [],
                "confidence": "none",
                "has_evidence": False,
            }
        }

    def run(self, question: str, doc_ids: list[str] | None = None) -> QueryResponse:
        """
        运行 Agent 问答流程

        Args:
            question: 用户问题
            doc_ids: 限定检索的文档 ID 列表

        Returns:
            QueryResponse: 标准问答响应
        """
        initial_state: AgentState = {
            "question": question,
            "doc_ids": doc_ids,
            "retrieved_chunks": [],
            "answer": None,
            "sources": [],
            "check_result": None,
            "final_output": None,
            "error": None,
        }

        result = self.graph.invoke(initial_state)
        final = result["final_output"]

        return QueryResponse(
            answer=final["answer"],
            sources=final["sources"],
            confidence=final["confidence"],
            has_evidence=final["has_evidence"],
        )


def build_agent() -> RAGAgent:
    """
    工厂函数：构建 Agent 实例

    Returns:
        RAGAgent: 初始化的 Agent 实例
    """
    return RAGAgent()
