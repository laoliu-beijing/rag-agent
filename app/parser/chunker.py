"""
文档分块器模块

将 docling 解析后的结构化文档切分为适合向量检索的文本块，
不同内容类型采用不同分块策略，确保语义完整性。
"""

from dataclasses import dataclass
from typing import Literal

from app.utils.logger import get_logger

logger = get_logger(__name__)

# 默认分块参数
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100


@dataclass
class ChunkData:
    """
    单个文本块数据结构

    Attributes:
        text: 块文本内容
        doc_id: 所属文档 ID
        page: 来源页码
        chunk_type: 内容类型（text 正文 / table 表格）
        metadata: 额外元数据
    """

    text: str
    doc_id: str
    page: int
    chunk_type: Literal["text", "table"] = "text"
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentChunker:
    """
    文档分块器

    支持以下分块策略：
    - 正文段落：按字符数切分，保留重叠上下文
    - 表格内容：整张表作为一个块，不可拆分
    - 条款编号：保留编号前缀作为上下文
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        """
        初始化分块器

        Args:
            chunk_size: 每个块的最大字符数
            chunk_overlap: 相邻块之间的重叠字符数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, parsed_data: dict) -> list[ChunkData]:
        """
        将解析后的文档数据切分为 chunk 列表

        Args:
            parsed_data: docling 解析后的结构化数据

        Returns:
            list[ChunkData]: 切分后的 chunk 列表
        """
        doc_id = parsed_data["doc_id"]
        chunks = []

        # 处理正文段落
        for page_item in parsed_data.get("pages", []):
            text = page_item.get("text", "").strip()
            page_num = page_item.get("page", 1)
            label = page_item.get("label", "text")

            if not text:
                continue

            # 表格内容单独处理（docling 可能将表格标注为特定 label）
            if "table" in label.lower():
                chunks.append(
                    ChunkData(
                        text=text,
                        doc_id=doc_id,
                        page=page_num,
                        chunk_type="table",
                        metadata={"label": label},
                    )
                )
            else:
                # 正文按大小切分
                text_chunks = self._split_text(text)
                for chunk_text in text_chunks:
                    chunks.append(
                        ChunkData(
                            text=chunk_text,
                            doc_id=doc_id,
                            page=page_num,
                            chunk_type="text",
                            metadata={"label": label},
                        )
                    )

        # 处理独立表格
        for table in parsed_data.get("tables", []):
            page_num = table.get("page", 1)
            table_data = table.get("data", [])
            caption = table.get("caption", "")

            # 将表格数据转为文本表示
            table_text = self._table_to_text(table_data, caption)
            if table_text:
                chunks.append(
                    ChunkData(
                        text=table_text,
                        doc_id=doc_id,
                        page=page_num,
                        chunk_type="table",
                        metadata={"table_caption": caption},
                    )
                )

        logger.info(
            "文档分块完成",
            doc_id=doc_id,
            total_chunks=len(chunks),
            text_chunks=sum(1 for c in chunks if c.chunk_type == "text"),
            table_chunks=sum(1 for c in chunks if c.chunk_type == "table"),
        )

        return chunks

    def _split_text(self, text: str) -> list[str]:
        """
        按字符数切分文本，保留重叠

        Args:
            text: 原始文本

        Returns:
            list[str]: 切分后的文本块列表
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap

        return chunks

    def _table_to_text(self, table_data: list, caption: str = "") -> str:
        """
        将表格数据转换为文本表示

        Args:
            table_data: 表格数据（字典列表）
            caption: 表格标题

        Returns:
            str: 表格的文本表示
        """
        if not table_data:
            return ""

        lines = []
        if caption:
            lines.append(f"表格: {caption}")

        # 提取表头
        headers = list(table_data[0].keys()) if table_data else []
        if headers:
            lines.append(" | ".join(str(h) for h in headers))
            lines.append("-" * 40)

        # 提取数据行
        for row in table_data:
            row_text = " | ".join(str(v) for v in row.values())
            lines.append(row_text)

        return "\n".join(lines)
