"""全局配置。API key 通过环境变量 DASHSCOPE_API_KEY 或 .env 文件提供。"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ---- 模型（阿里通义 DashScope，OpenAI 兼容接口）----
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen-plus")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-v3")
EMBED_DIM = 1024
# VLM 图片语义理解（失败自动降级回纯 OCR）
VLM_ENABLED = os.getenv("VLM_ENABLED", "1") == "1"
VLM_MODEL = os.getenv("VLM_MODEL", "qwen-vl-flash")
# PDF 公式区域检测 + VLM LaTeX 转录（依赖 VLM_ENABLED；每页公式区域限量防API爆炸）
FORMULA_VLM = os.getenv("FORMULA_VLM", "1") == "1"
FORMULA_MAX_PER_PAGE = 8

# ---- 检索参数 ----
CHUNK_SIZE = 512          # 切分目标 token 数（按字符近似）
CHUNK_OVERLAP = 64
DENSE_TOP_K = 30
SPARSE_TOP_K = 30
RRF_K = 60                # Reciprocal Rank Fusion 常数
RERANK_MAX_CANDIDATES = 32  # 送入 LLM 重排的候选上限（关思考后 32×400字 约 2-3s）
RERANK_TOP_K = 5          # 重排后送入生成的块数
MIN_EVIDENCE_SCORE = 2    # LLM 重排后保留的最低相关性分（1-5；公式/符号类文本易被低估，勿设过高）

# ---- 语义缓存 ----
SEMANTIC_CACHE_THRESHOLD = 0.95
SEMANTIC_CACHE_TOP_K = 1

# ---- 引用核查 ----
# LLM 内容级核查（较慢，~3s/次）；正则校验（编号合法性）始终开启
LLM_CITATION_VERIFY = os.getenv("LLM_CITATION_VERIFY", "0") == "1"

# ---- qwen3 混合思考模型 ----
# 工具型调用（路由/改写/重排/核查）固定关闭深度思考（关后快 ~20 倍）
# 最终回答生成是否开启思考：1=更高质量但更慢，0=默认，快
GENERATE_THINKING = os.getenv("GENERATE_THINKING", "0") == "1"

# ---- 文献结构化切分 ----
# [参考文献] 整节是否入库（默认丢弃：乱码引用列表是检索毒药）
INDEX_REFERENCES = os.getenv("INDEX_REFERENCES", "0") == "1"

# UI 构建号：改了 ui/ 下的代码时递增，界面侧边栏会显示，用于确认"重启后的进程跑的是新代码"
UI_BUILD = "2026-11-16.5"

# ---- 路径 ----
DATA_DIR = BASE_DIR / "data"
KB_DIR = DATA_DIR / "knowledge"          # 上传的原始文档
INDEX_DIR = DATA_DIR / "index"           # Chroma 持久化目录
CACHE_DIR = DATA_DIR / "cache"
EVAL_DIR = DATA_DIR / "eval"
TESTSET_PATH = DATA_DIR / "eval" / "testset.json"

for d in (KB_DIR, INDEX_DIR, CACHE_DIR, EVAL_DIR):
    d.mkdir(parents=True, exist_ok=True)
