"""多模态文档解析：PDF（含内嵌图片 OCR）、Markdown、纯文本、图片。

统一输出 Document 列表，metadata 带 source/page/heading。"""
import re
from pathlib import Path

import fitz
from langchain_core.documents import Document
from rapidocr_onnxruntime import RapidOCR

from src import config

_ocr = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = RapidOCR()
    return _ocr


def _ocr_image_bytes(img_bytes: bytes) -> str:
    import io

    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    result, _ = _get_ocr()(np.array(img))
    if not result:
        return ""
    return "\n".join(line[1] for line in result if line[1])


def _process_image_bytes(img_bytes: bytes) -> str:
    """OCR 文字 + VLM 语义描述，拼成图片块的文本内容。"""
    parts = []
    ocr_text = _ocr_image_bytes(img_bytes)
    if ocr_text.strip():
        parts.append(f"[图片文字]\n{ocr_text}")
    from src.vlm import describe_image
    vlm_desc = describe_image(img_bytes)
    if vlm_desc:
        parts.append(f"[图片描述]\n{vlm_desc}")
    return "\n\n".join(parts)


# 数学字体特征：Computer Modern（CMMI/CMSY/CMEX）、AMS（MSAM/MSBM）、Symbol、
# txfonts/newtx（txmi|rtxmi 数学斜体、txsy 数学符号、txex 大符号）等常见 LaTeX 数学字体族
_MATH_FONT_RE = re.compile(
    r"(?i)(cmmi|cmsy|cmex|msam|msbm|symbol|math|mtmi|mtsy"
    r"|txmi|rtxmi|txsy|txex|ntxsy|ptmmi|nimbusmath|cambriamath)")


def _detect_formula_regions(page) -> list:
    """按行统计数学字体占比，识别独立公式行并合并连续行（多行推导）为区域矩形。"""
    try:
        data = page.get_text("dict")
    except Exception:
        return []
    formula_lines = []
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            # PyMuPDF 2.x 的 line 无 text 字段，从 spans 拼接
            line_text = "".join(sp.get("text", "") for sp in spans).strip()
            math_w = total_w = 0.0
            for sp in spans:
                w = max(1.0, sp["bbox"][2] - sp["bbox"][0])
                total_w += w
                if _MATH_FONT_RE.search(sp.get("font", "")):
                    math_w += w
            if total_w > 0 and math_w / total_w >= 0.5 and len(line_text) >= 2:
                formula_lines.append(line["bbox"])

    if not formula_lines:
        return []
    # 按 y 排序合并垂直相邻（间距<18pt）的公式行 → 多行公式一个区域
    formula_lines.sort(key=lambda b: b[1])
    regions, cur = [], list(formula_lines[0])
    for bbox in formula_lines[1:]:
        if bbox[1] - cur[3] < 18:
            cur[3] = max(cur[3], bbox[3])
            cur[0] = min(cur[0], bbox[0])
            cur[2] = max(cur[2], bbox[2])
        else:
            regions.append(cur)
            cur = list(bbox)
    regions.append(cur)
    # 加 padding、过滤碎片
    out = []
    for x0, y0, x1, y1 in regions:
        if y1 - y0 < 6 or x1 - x0 < 15:
            continue
        out.append(fitz.Rect(x0 - 8, y0 - 4, x1 + 8, y1 + 4))
    return out


def _transcribe_page_formulas(page) -> list[str]:
    """把页面上的独立公式区域截图发给 VLM，转写为 LaTeX。返回 LaTeX 列表。"""
    if not config.FORMULA_VLM:
        return []
    from src.vlm import transcribe_latex
    texs = []
    for rect in _detect_formula_regions(page)[: config.FORMULA_MAX_PER_PAGE]:
        try:
            pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(3, 3))  # ~216dpi
            tex = transcribe_latex(pix.tobytes("png"))
            if tex:
                texs.append(tex)
        except Exception:
            continue
    return texs


def parse_pdf(path: Path) -> list[Document]:
    """逐页解析：PyMuPDF4LLM 版面级 markdown 提取 + 公式区域 VLM LaTeX 转录 + 插图 OCR/VLM。
    公式转录独立成块（embedding 不被正文稀释，公式查询可直接命中）。"""
    import pymupdf4llm

    docs = []
    with fitz.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages(), start=1):
            parts = []
            # A: 版面级 markdown（段落/标题/表格结构优于纯 get_text）
            try:
                md = pymupdf4llm.to_markdown(pdf, pages=[page_no - 1])
            except Exception:
                md = page.get_text("text")
            if md and md.strip():
                parts.append(md)

            # B: 独立公式区域 → LaTeX 转录，单独成块
            texs = _transcribe_page_formulas(page)
            if texs:
                docs.append(Document(
                    page_content="[页面公式-LaTeX转录]\n"
                                 + "\n\n".join(f"$$\n{t}\n$$" for t in texs),
                    metadata={"source": path.name, "page": page_no, "heading": "公式"},
                ))

            # C: 内嵌插图 OCR + VLM 语义描述
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                try:
                    pix = fitz.Pixmap(pdf, xref)
                    if pix.n - pix.alpha > 3:  # CMYK 转 RGB
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    if pix.width < 10 or pix.height < 10:  # 分隔线/装饰线条，跳过
                        continue
                    img_text = _process_image_bytes(pix.tobytes("png"))
                    if img_text.strip():
                        parts.append(img_text)
                except Exception:
                    continue

            text = "\n\n".join(p for p in parts if p.strip())
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": path.name, "page": page_no},
                ))
    return docs


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def parse_markdown(path: Path) -> list[Document]:
    """按标题层级切段，保留 heading 路径作为 metadata。"""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    docs, buf, heading_path = [], [], []
    for line in lines:
        m = _MD_HEADING.match(line.strip())
        if m:
            if buf:
                docs.append(_md_doc(buf, heading_path, path))
                buf = []
            level, title = len(m.group(1)), m.group(2).strip()
            heading_path = heading_path[: level - 1] + [title]
        buf.append(line)
    if buf:
        docs.append(_md_doc(buf, heading_path, path))
    return [d for d in docs if d.page_content.strip()]


def _md_doc(buf: list[str], heading_path: list[str], path: Path) -> Document:
    # buf[0] 是标题行本身，标题路径已单独拼入，避免重复
    body = buf[1:] if buf and _MD_HEADING.match(buf[0].strip()) else buf
    return Document(
        page_content="\n".join(f"{'#' * (i + 1)} {h}" for i, h in enumerate(heading_path))
        + "\n" + "".join(body),
        metadata={"source": path.name, "heading": " > ".join(heading_path) or "root"},
    )


def parse_image(path: Path) -> list[Document]:
    text = _process_image_bytes(path.read_bytes())
    if not text.strip():
        return []
    return [Document(page_content=text, metadata={"source": path.name, "page": 1, "modality": "image"})]


def parse_file(path: Path) -> list[Document]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(path)
    if ext in (".md", ".markdown"):
        return parse_markdown(path)
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        return parse_image(path)
    if ext in (".txt", ".text"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [Document(page_content=text, metadata={"source": path.name, "page": 1})]
    raise ValueError(f"不支持的文件类型: {ext}")
