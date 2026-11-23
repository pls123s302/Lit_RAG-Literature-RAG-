"""可插拔模块注册表（工厂模式）：解析 / 切分 / 向量化 / 查询改写 / 重排 / 检索融合。

每个链路环节注册多个实现，运行时按名字从配置取用，便于效果对比（A/B 实验）。
"""
import functools
from typing import Callable

# name -> factory（惰性创建，避免 import 时初始化重资源）
_REGISTRY: dict[str, dict[str, Callable]] = {}


def register(stage: str, name: str):
    """装饰器：把实现注册到某个环节。factory 是零参可调用对象。"""
    def deco(fn):
        _REGISTRY.setdefault(stage, {})[name] = fn
        return fn
    return deco


@functools.lru_cache(maxsize=None)
def get_plugin(stage: str, name: str):
    """按名字实例化插件（带缓存，同一插件全进程单例）。"""
    if stage not in _REGISTRY or name not in _REGISTRY[stage]:
        raise KeyError(f"未注册的插件: {stage}.{name}，可选: {list(_REGISTRY.get(stage, {}))}")
    return _REGISTRY[stage][name]()


def list_plugins(stage: str | None = None) -> dict:
    if stage:
        return dict(_REGISTRY.get(stage, {}))
    return {s: list(d) for s, d in _REGISTRY.items()}
