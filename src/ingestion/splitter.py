"""文本清洗 + 结构化切分（标题层级感知，长块递归兜底）。"""
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config

_REPLACEMENTS = [
    (r"\r\n?", "\n"),
    (r"[ \t\u3000]+\n", "\n"),          # 行尾空白
    (r"\n{3,}", "\n\n"),                # 连续空行
    (r"[\u200b\ufeff\u2028\u2029]", ""),  # 不可见字符
    (r"-{3,}", ""),                     # 分隔线
    (r"\[\d+\]", ""),                   # 文献引用标号 [12]
    (r"(?m)\(\d{1,3}\)\s*$", ""),       # 仅行尾的 (12)：公式编号，保留正文中的括号数字
]
_PRETTY_REPLACEMENTS = [(r"[ \u3000]{2,}", " ")]


def clean_text(text: str) -> str:
    for pat, rep in _REPLACEMENTS:
        text = re.sub(pat, rep, text)
    return text.strip()


def _heading_of(doc: Document) -> str:
    return doc.metadata.get("heading", "")


def split_documents(docs: list[Document]) -> list[Document]:
    """标题层级感知切分：小节整块保留；超长小节用递归切分兜底，块头带上标题路径。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    )
    chunks: list[Document] = []
    for doc in docs:
        text = clean_text(doc.page_content)
        if not text:
            continue
        heading = _heading_of(doc)
        pieces = [doc] if len(text) <= config.CHUNK_SIZE * 1.5 else splitter.split_text(text)
        for i, piece in enumerate(pieces, start=1):
            content = piece.page_content if isinstance(piece, Document) else piece
            chunks.append(Document(
                page_content=content,
                metadata={**doc.metadata, "heading": heading, "chunk_index": i},
            ))
    return chunks
