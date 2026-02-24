import importlib
from typing import TYPE_CHECKING

from mineru_vl_utils.version import __version__

__lazy_attrs__ = {
    "MinerUClient": (".client", "MinerUClient"),
    "SamplingParams": (".vlm_client.base_client", "SamplingParams"),
}

if TYPE_CHECKING:
    from .client import MinerUClient
    from .vlm_client.base_client import SamplingParams


def __getattr__(name: str):
    if name in __lazy_attrs__:
        module_name, attr_name = __lazy_attrs__[name]
        module = importlib.import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")


__all__ = [
    "MinerUClient",
    "SamplingParams",
    "__version__",
]
