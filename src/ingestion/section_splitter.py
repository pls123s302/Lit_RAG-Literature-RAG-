"""学术论文结构化切分：跨页组装章节，按 [摘要][引言][方法][实验] 等标签切块。

针对英文（LaTeX 生成）论文优化：编号标题 + 标准章节名 + 全大写行三层识别，
子节继承父节标签，[参考文献] 默认不入库（检索毒药）。"""
import re
from collections import Counter

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config

# ---- 标题 → 标签映射（关键词按优先级排列，含子串匹配） ----
_TAG_KEYWORDS = [
    ("abstract", "摘要"),
    ("introduction", "引言"),
    ("related work", "相关工作"),
    ("background", "相关工作"),
    ("preliminar", "相关工作"),
    ("problem setup", "方法"),   # 问题定义/符号设定归入方法
    ("setup", "方法"),
    ("method", "方法"),          # method / methodology / THE XX METHOD
    ("approach", "方法"),
    ("proposed", "方法"),
    ("model", "方法"),           # Model / Model Architecture
    ("experiment", "实验"),
    ("result", "实验"),
    ("evaluation", "实验"),
    ("ablation", "实验"),
    ("dataset", "实验"),
    ("discussion", "讨论"),
    ("conclusion", "结论"),
    ("summary", "结论"),
    ("acknowledg", "致谢"),
    ("reference", "参考文献"),
    ("bibliograph", "参考文献"),
    ("appendix", "附录"),
    ("supplementary", "附录"),
]

_RE_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$")
_RE_DIRTY = re.compile(r"(?i)(figure\s*\d|table\s*\d|fig\.\s*\d|http|www\.|@|et al\.?|github)")
_RE_PAGE_NUM = re.compile(r"^\d{1,4}$")
_RE_BOLD_LINE = re.compile(r"^\*\*(.+?)\*\*$")  # pymupdf4llm 整行加粗 = 标题强信号


def _map_tag(heading_text: str, is_subsection: bool) -> str:
    low = heading_text.lower()
    for kw, tag in _TAG_KEYWORDS:
        if kw in low:
            return tag
    # 子节无关键词 → 继承父节标签由调用方处理；顶层无关键词 → 正文
    return ""  # "" 表示继承/兜底


def _is_heading(line: str, md_prefix: bool) -> bool:
    """判断一行是否为章节标题。md_prefix: pymupdf4llm 已标记为 # 标题。"""
    line = line.strip()
    if not line or len(line) > 100 or _RE_DIRTY.search(line):
        return False
    if line.startswith(("|", "-", "•")) or line.endswith((",", ";", ":")):
        return False

    # 整行加粗 → 取出内文本参与判定（LaTeX 小节粗体的 markdown 形式）
    inner = line
    m_bold = _RE_BOLD_LINE.match(line)
    if m_bold:
        inner = m_bold.group(1).strip()
    elif line.startswith("*"):
        return False  # 其余以 * 开头的按 bullet 列表排除

    if md_prefix:  # pymupdf4llm 识别的标题直接采信
        return True

    m = _RE_NUMBERED.match(inner)
    if m:
        rest = m.group(2).strip()
        # 编号 + 短文本，标题文本必须以大写字母/数字开头（排除 "42 again. Again..." 正文句），
        # 且不以句号结尾（排除 "3. This shows that..."）
        if (rest and rest[0].isupper()
                and len(rest) <= 90 and not rest.endswith(".") and not rest.endswith(",")):
            return True
        return False

    # 命名标题：Abstract / References / Acknowledgments（含全大写形式）
    named = re.compile(
        r"^(abstract|references|acknowledg\w*|appendix\w*|conclusions?|summary|"
        r"introduction|related\s+work|background|method\w*|approach|experiment\w*|"
        r"results?|evaluation|discussion|bibliography)\s*:?\s*$", re.I)
    if named.match(inner):
        return True

    # 全大写短行（LaTeX 小节常用形式 "3 THE AKT METHOD" 已被编号分支覆盖）
    if inner.isupper() and 2 <= len(inner.split()) <= 12 and sum(c.isdigit() for c in inner) <= 4:
        return True

    return False


def _assemble_sections(page_docs: list[Document]) -> list[dict]:
    """跨页扫描行流，组装为 [{tag, heading, pages, text}] 列表。"""
    # 页眉/页脚剔除：在 ≥3 个页面的首尾两行重复出现的行
    edge_counter = Counter()
    first_lines = {}
    for d in page_docs:
        lines = [l.strip() for l in d.page_content.splitlines() if l.strip()]
        first_lines[d.metadata.get("page")] = lines
        for l in (lines[:2] + lines[-2:]):
            if len(l) < 60:
                edge_counter[l] += 1
    repeated_edges = {l for l, c in edge_counter.items() if c >= 3}

    sections: list[dict] = []
    current = {"tag": "题录", "heading": "", "pages": [], "lines": []}
    seen_abstract = False

    def flush():
        if current["lines"]:
            sections.append(dict(current))

    for d in page_docs:
        page_no = d.metadata.get("page")
        md_lines = d.page_content.splitlines()
        n = len(md_lines)
        for idx, raw in enumerate(md_lines):
            line = raw.strip()
            if not line:
                continue
            # 页眉页脚：纯页码，或重复出现的边缘行，且位于页面前两行/后两行
            if _RE_PAGE_NUM.match(line) and (idx < 2 or idx >= n - 2):
                continue
            if line in repeated_edges and (idx < 2 or idx >= n - 2):
                continue

            md_heading = line.lstrip().startswith("#")
            if _is_heading(line, md_heading):
                flush()
                clean = line.lstrip("#").strip()
                m_bold = _RE_BOLD_LINE.match(clean)
                if m_bold:
                    clean = m_bold.group(1).strip()
                m = _RE_NUMBERED.match(clean)
                tag = _map_tag(clean, is_subsection=bool(m and "." in m.group(1)))
                if not tag:
                    if m and "." in m.group(1) and sections:
                        tag = sections[-1]["tag"]          # 子节继承父节
                    elif seen_abstract:
                        tag = "正文"
                    else:
                        tag = "题录"                        # 摘要前的标题区（题名/作者）
                if tag == "摘要":
                    seen_abstract = True
                current = {"tag": tag, "heading": clean,
                           "pages": [page_no], "lines": []}
                continue

            current["lines"].append(line)
            if page_no not in current["pages"]:
                current["pages"].append(page_no)
    flush()
    return sections


def split_documents_academic(docs: list[Document]) -> list[Document]:
    """切分入口（注册为 chunker: academic_section）。

    docs = parse_pdf 的输出：页文档（无 heading）+ 公式转录文档（heading=公式）。"""
    formula_docs = [d for d in docs if d.metadata.get("heading") == "公式"]
    page_docs = [d for d in docs if d.metadata.get("heading") != "公式"]

    sections = _assemble_sections(page_docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    )

    out: list[Document] = []
    for sec in sections:
        text = clean_section_text("\n".join(sec["lines"]))
        if len(text) < 40:
            continue
        if sec["tag"] == "参考文献" and not config.INDEX_REFERENCES:
            continue  # 参考文献是检索毒药，默认整节丢弃

        head = f"[{sec['tag']}] {sec['heading']}".strip()
        pieces = [text] if len(text) <= config.CHUNK_SIZE * 1.5 else splitter.split_text(text)
        for i, piece in enumerate(pieces, start=1):
            out.append(Document(
                page_content=f"{head}\n{piece}",
                metadata={"source": page_docs[0].metadata.get("source"),
                          "page": sec["pages"][0],
                          "heading": sec["heading"] or sec["tag"],
                          "section": f"[{sec['tag']}]",
                          "chunk_index": i},
            ))

    # 公式转录块原样通过，打 [公式] 标签
    for d in formula_docs:
        out.append(Document(
            page_content=f"[公式] 第{d.metadata.get('page')}页\n{d.page_content}",
            metadata={**d.metadata, "section": "[公式]"},
        ))
    return out


def clean_section_text(text: str) -> str:
    """小节内轻清洗：压空行、去 pymupdf4llm 的转义残留。"""
    text = text.replace("\\(", "").replace("\\)", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
