"""DashScope（OpenAI 兼容）模型的统一封装。"""
import socket
from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src import config

# 部分 Windows 机器对阿里云端点的 IPv6 路由黑洞（DNS 返回 v6 地址优先，
# TCP 握手干等 ~40s 才回退 IPv4）。这里全局改为 IPv4 优先，仅重排顺序不改语义。
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_first_getaddrinfo(*args, **kwargs):
    results = _orig_getaddrinfo(*args, **kwargs)
    return sorted(results, key=lambda r: r[0] == socket.AF_INET6)


socket.getaddrinfo = _ipv4_first_getaddrinfo


@lru_cache(maxsize=4)
def get_chat_llm(temperature: float = 0.2, thinking: bool = False) -> ChatOpenAI:
    kwargs = {}
    low = config.CHAT_MODEL.lower()
    if "qwen3" in low or "deepseek" in low:
        # qwen3 / deepseek 系混合思考模型：工具型调用关掉深度思考（实测快 ~20 倍）
        kwargs["extra_body"] = {"enable_thinking": thinking}
    return ChatOpenAI(
        model=config.CHAT_MODEL,
        api_key=config.API_KEY,
        base_url=config.DASHSCOPE_BASE_URL,
        temperature=temperature,
        timeout=90,        # 单次调用上限，避免瞬时卡顿拖死整批任务
        max_retries=3,     # 网络抖动自动重试
        **kwargs,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.EMBED_MODEL,
        api_key=config.API_KEY,
        base_url=config.DASHSCOPE_BASE_URL,
        # qwen3.7 系列要求 input 为纯字符串，禁用 token 数组编码
        check_embedding_ctx_length=False,
        # DashScope 单请求最多 20 条文本，留余量
        chunk_size=16,
    )
