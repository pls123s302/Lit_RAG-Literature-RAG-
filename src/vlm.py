"""VLM 图片语义理解：qwen-vl 系列，走 DashScope OpenAI 兼容接口。

输入图片字节，输出图片的语义描述（图表结构、视觉关系、场景内容等）。
失败时返回空字符串，调用方降级回纯 OCR。"""
import base64
import functools
import logging
import re

from openai import OpenAI

from src import config

logger = logging.getLogger(__name__)

VLM_PROMPT = (
    "请详细描述这张图片的内容，用于知识库检索。要求：\n"
    "1. 若图片以数学公式为主体（公式截图、推导过程等），在描述最开头写上标记 [含公式]；\n"
    "2. 说明图片类型（照片/流程图/架构图/表格/截图等）；\n"
    "3. 描述其中的关键对象、结构和关系（如图表的数值趋势、流程的步骤顺序）；\n"
    "4. 图片中的普通文字不用逐字复述（OCR 已单独处理），但数学公式必须用 LaTeX 转录（如 $\\alpha_t$、$\\sum_i$）；\n"
    "5. 用简洁的中文段落输出，不超过200字。"
)

LATEX_PROMPT = (
    "将图片中的数学公式逐字转写为 LaTeX 源码。要求：\n"
    "1. 只输出 LaTeX 代码本身：不要解释、不要 markdown 代码块标记、不要 $ 定界符；\n"
    "2. 多个公式或公式编号按出现顺序逐行输出，编号用 \\tag{n} 附在行尾；\n"
    "3. 分数用 \\frac、上下标用 ^ 和 _、多行推导用 \\\\ 分行，忠实原排版结构。"
)


def transcribe_latex(img_bytes: bytes) -> str:
    """公式区域图片 → LaTeX 源码。VLM 未启用或失败返回空串。"""
    if not config.VLM_ENABLED:
        return ""
    try:
        b64 = base64.b64encode(img_bytes).decode()
        resp = _client().chat.completions.create(
            model=config.VLM_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": LATEX_PROMPT},
                ],
            }],
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        # 去掉可能的代码块包裹
        text = re.sub(r"^```(?:latex)?\s*|\s*```$", "", text)
        return text.strip()
    except Exception as e:
        logger.warning("公式 LaTeX 转录失败: %s", e)
        return ""


@functools.lru_cache(maxsize=1)
def _client():
    return OpenAI(api_key=config.API_KEY, base_url=config.DASHSCOPE_BASE_URL)


def describe_image(img_bytes: bytes) -> str:
    """返回图片语义描述；VLM 未启用或调用失败时返回空字符串。"""
    if not config.VLM_ENABLED:
        return ""
    try:
        b64 = base64.b64encode(img_bytes).decode()
        resp = _client().chat.completions.create(
            model=config.VLM_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": VLM_PROMPT},
                ],
            }],
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("VLM 描述失败，降级回纯 OCR: %s", e)
        return ""
