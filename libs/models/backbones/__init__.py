# from mmdet.models.backbones import ResNet
from .resnet import ResNet
from .dla import DLA  # noqa: F401

__all__ = ['ResNet', 'DLA']