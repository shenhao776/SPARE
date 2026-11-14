from core.datasets.base import PoseAdapter
from core.datasets.fourseasons import FourSeasonsAdapter
from core.datasets.custom import CustomAdapter, CustomWGS84Adapter

_ADAPTERS = {}


def register(adapter_cls):
    _ADAPTERS[adapter_cls.pose_format] = adapter_cls
    return adapter_cls


register(FourSeasonsAdapter)
register(CustomAdapter)
register(CustomWGS84Adapter)


def get_adapter(pose_format):

    if pose_format not in _ADAPTERS:
        raise ValueError(
            f"Unknown pose_format '{pose_format}'. " f"Registered: {sorted(_ADAPTERS)}"
        )
    return _ADAPTERS[pose_format]()


def available_formats():
    return sorted(_ADAPTERS)


__all__ = ["PoseAdapter", "get_adapter", "register", "available_formats"]
