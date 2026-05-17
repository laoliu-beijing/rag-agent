"""
PDF 解析器模块

使用 docling 将扫描版或文字版 PDF 解析为结构化 JSON 数据，
支持中文正文、条款编号和表格内容的提取。

当 docling 遇到字体编码问题产生乱码时，自动 fallback 到 PyMuPDF。

注意：docling 会自动处理 OCR，无需额外配置 Tesseract。
"""

import json
import re
import shutil
from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter

# 导入 OCR pipeline 配置
try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
    from docling.document_converter import PdfFormatOption
    _HAS_OCR_OPTS = True
except Exception:
    _HAS_OCR_OPTS = False

from app.config.settings import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 乱码检测阈值：正常字符比例低于该值时触发 fallback
GARBLED_THRESHOLD = 0.5


def _create_docling_converter() -> DocumentConverter:
    """
    创建 docling DocumentConverter，始终启用 OCR 中文识别。

    docling 2.0.0 默认会提取 PDF 文字层，遇到字体编码问题时会乱码。
    这里强制启用 OCR（do_ocr=True），用 EasyOCR 从图片识别文字，
    并设置语言为简体中文+英文，彻底解决中文乱码问题。
    """
    if not _HAS_OCR_OPTS:
        logger.warning("docling OCR 配置模块未导入，使用默认 Converter")
        return DocumentConverter()

    try:
        # EasyOCR 语言代码：ch_sim = 简体中文, en = 英文
        # 使用 CPU（用户环境无 GPU）
        ocr_options = EasyOcrOptions(
            lang=["ch_sim", "en"],
            use_gpu=False,
        )
        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            ocr_options=ocr_options,
        )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        logger.info("docling 已启用 OCR 模式（语言: ch_sim + en, CPU）")
        return converter
    except Exception as e:
        logger.warning(f"配置 docling OCR 失败，使用默认模式: {e}")
        return DocumentConverter()


def _is_garbled(text: str) -> bool:
    """
    检测文本是否为乱码。

    通过统计正常字符（CJK、英文字母、数字、常用标点、空白）的比例来判断。
    比例低于阈值时认为是乱码。
    """
    if not text or len(text.strip()) == 0:
        return True

    # 正常字符：CJK 汉字、英文字母、数字、常用标点、空白
    normal_pattern = re.compile(
        r"[一-鿿　-〿＀-￯"  # CJK + 全角标点
        r"a-zA-Z0-9\s\n\r\t.,;:!?()\[\]{}'\"\-+=/\\@%&*<>|^~$#]"
    )

    normal_count = len(normal_pattern.findall(text))
    total_count = len(text)

    if total_count == 0:
        return True

    ratio = normal_count / total_count
    logger.debug("乱码检测", text_length=total_count, normal_ratio=ratio)
    return ratio < GARBLED_THRESHOLD


def _parse_with_pymupdf(file_path: Path, doc_id: str, filename: str) -> dict:
    """
    使用 PyMuPDF 解析 PDF 作为 docling 的 fallback。

    Args:
        file_path: PDF 文件路径
        doc_id: 文档 ID
        filename: 原始文件名

    Returns:
        dict: 与 docling 输出兼容的结构化数据
    """
    import fitz  # pymupdf

    logger.info("使用 PyMuPDF fallback 解析 PDF", doc_id=doc_id, filename=filename)

    doc = fitz.open(str(file_path))
    pages = []
    tables = []
    markdown_parts = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        # 提取文本
        text = page.get_text("text").strip()
        if text:
            pages.append({
                "text": text,
                "label": "text",
                "page": page_num + 1,
            })
            markdown_parts.append(f"## 第 {page_num + 1} 页\n\n{text}")

        # 尝试提取表格（简单启发式：检测页面中的表格结构）
        try:
            tab = page.find_tables()
            if tab and tab.tables:
                for t in tab.tables:
                    table_data = {
                        "page": page_num + 1,
                        "data": t.extract(),
                        "caption": "",
                    }
                    tables.append(table_data)
        except Exception:
            # 表格提取失败不阻断主流程
            pass

    doc.close()

    # 按段落拆分，模拟 docling 的细粒度文本块
    fine_grained_pages = []
    for p in pages:
        paragraphs = [para.strip() for para in p["text"].split("\n") if para.strip()]
        for para in paragraphs:
            fine_grained_pages.append({
                "text": para,
                "label": "text",
                "page": p["page"],
            })

    return {
        "doc_id": doc_id,
        "filename": filename,
        "metadata": {
            "page_count": len(doc),
            "title": filename,
        },
        "pages": fine_grained_pages,
        "tables": tables,
        "markdown": "\n\n".join(markdown_parts),
    }


class PDFParser:
    """
    PDF 文档解析器

    封装 docling 的文档转换功能，提供：
    1. PDF 到结构化 JSON 的转换
    2. 解析结果的本地持久化
    3. 快速页数检测（不解全文）
    """

    def __init__(self):
        """初始化 docling 文档转换器（始终启用 OCR）"""
        self.converter = _create_docling_converter()
        self.settings = get_settings()

    def parse(self, file_path: Path | str, doc_id: str) -> dict:
        """
        解析 PDF 文件为结构化数据。

        始终使用 docling OCR 模式识别文字，忽略有问题的 PDF 文字层。
        如果 docling OCR 仍然乱码，fallback 到 PyMuPDF。

        Args:
            file_path: PDF 文件路径
            doc_id: 文档唯一标识，用于命名输出文件

        Returns:
            dict: 结构化文档数据，包含 pages、tables、metadata

        Raises:
            PDFParseError: 当所有解析方式都失败时抛出
        """
        file_path = Path(file_path)
        logger.info(
            "开始解析 PDF（OCR 模式）",
            doc_id=doc_id,
            filename=file_path.name,
        )

        # 第 1 步：docling OCR 模式
        try:
            result = self.converter.convert(str(file_path))
            structured_data = self._extract_data(result, doc_id, file_path.name)

            # 检查是否乱码
            all_text = " ".join(p["text"] for p in structured_data.get("pages", []))
            if not _is_garbled(all_text):
                self._save_json(structured_data, doc_id)
                logger.info(
                    "PDF 解析完成",
                    doc_id=doc_id,
                    parser="docling-ocr",
                    page_count=structured_data["metadata"]["page_count"],
                    text_blocks=len(structured_data["pages"]),
                    tables=len(structured_data["tables"]),
                )
                return structured_data

            logger.warning(
                "docling OCR 模式输出乱码，触发 PyMuPDF fallback",
                doc_id=doc_id,
            )
        except Exception as e:
            logger.warning(
                "docling OCR 模式失败，触发 PyMuPDF fallback",
                doc_id=doc_id,
                error=str(e),
            )

        # 第 2 步：PyMuPDF fallback
        try:
            structured_data = _parse_with_pymupdf(file_path, doc_id, file_path.name)
            structured_data["_fallback"] = True
            self._save_json(structured_data, doc_id)
            logger.info(
                "PDF 解析完成（PyMuPDF fallback）",
                doc_id=doc_id,
                parser="pymupdf",
                page_count=structured_data["metadata"]["page_count"],
                text_blocks=len(structured_data["pages"]),
                tables=len(structured_data["tables"]),
            )
            return structured_data
        except Exception as e:
            logger.error(
                "PDF 解析失败",
                doc_id=doc_id,
                error=str(e),
            )
            raise PDFParseError(f"无法解析 PDF 文件: {file_path.name}, 错误: {e}") from e

    def _extract_data(self, result: Any, doc_id: str, filename: str) -> dict:
        """
        从 docling 转换结果中提取结构化数据

        Args:
            result: docling 转换结果
            doc_id: 文档 ID
            filename: 原始文件名

        Returns:
            dict: 包含 pages、tables、metadata 的结构化数据
        """
        # 导出为 markdown 格式以获取文本结构
        md_exporter = result.document.export_to_markdown()

        pages = []
        tables = []

        def _get_page_no(item) -> int:
            """从 docling 元素的 prov 中安全提取页码"""
            prov = getattr(item, "prov", None)
            if prov and len(prov) > 0:
                prov_item = prov[0]
                # docling-core 2.0.0: ProvenanceItem 是 Pydantic model,用属性访问
                if hasattr(prov_item, "page_no"):
                    return prov_item.page_no
                # 旧版兼容:字典
                elif isinstance(prov_item, dict):
                    return prov_item.get("page_no", 1)
            return 1

        # 遍历文档元素，提取文本和表格
        for item in result.document.texts:
            pages.append({
                "text": item.text,
                "label": str(item.label) if hasattr(item, "label") else "text",
                "page": _get_page_no(item),
            })

        # 提取表格
        for table in result.document.tables:
            table_data = {
                "page": _get_page_no(table),
                "data": table.export_to_dataframe().to_dict(orient="records") if hasattr(table, "export_to_dataframe") else [],
                "caption": "",
            }
            tables.append(table_data)

        return {
            "doc_id": doc_id,
            "filename": filename,
            "metadata": {
                "page_count": len(result.pages) if hasattr(result, "pages") else 1,
                "title": result.document.name if hasattr(result.document, "name") else filename,
            },
            "pages": pages,
            "tables": tables,
            "markdown": md_exporter,
        }

    def _save_json(self, data: dict, doc_id: str) -> None:
        """
        将解析结果保存为 JSON 文件

        Args:
            data: 结构化文档数据
            doc_id: 文档 ID
        """
        json_dir = self.settings.json_dir_path
        json_dir.mkdir(parents=True, exist_ok=True)
        json_path = json_dir / f"{doc_id}.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(
            "解析结果已保存",
            doc_id=doc_id,
            json_path=str(json_path),
        )


def parse_pdf(file_path: Path | str, doc_id: str) -> dict:
    """
    解析 PDF 的便捷函数

    Args:
        file_path: PDF 文件路径
        doc_id: 文档唯一标识

    Returns:
        dict: 结构化文档数据
    """
    parser = PDFParser()
    return parser.parse(file_path, doc_id)


class PDFParseError(Exception):
    """PDF 解析异常"""
    pass


def quick_page_count(file_path: Path | str) -> int:
    """
    快速获取 PDF 页数（不解全文）

    使用 PyPDF2 轻量读取 PDF 元数据，用于判断同步/异步处理。

    Args:
        file_path: PDF 文件路径

    Returns:
        int: PDF 总页数，读取失败返回 1
    """
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(file_path))
        return len(reader.pages)
    except Exception as e:
        logger.warning(
            "无法快速读取 PDF 页数，默认返回 1",
            file_path=str(file_path),
            error=str(e),
        )
        return 1
