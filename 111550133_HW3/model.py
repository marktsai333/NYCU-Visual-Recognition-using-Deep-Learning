"""Mask R-CNN models for cell instance segmentation.

Supports multiple backbone options. All variants are well under the 200M
parameter limit required by the assignment.

References:
  - Mask R-CNN: He et al., ICCV 2017
  - FPN: Lin et al., CVPR 2017
  - ResNet: He et al., CVPR 2016
  - Swin Transformer: Liu et al., ICCV 2021
"""

import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

NUM_CLASSES = 5  # background + class1..class4

# Smaller anchors suit cell-scale objects (10-150 px)
ANCHOR_SIZES = ((8,), (16,), (32,), (64,), (128,))
ANCHOR_RATIOS = ((0.5, 1.0, 2.0),) * 5


def _replace_heads(model, num_classes):
    in_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_box, num_classes)
    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, num_classes)
    return model


def build_resnet50(num_classes=NUM_CLASSES, trainable_layers=3):
    """Mask R-CNN V2 with ResNet50-FPN backbone (~44M params)."""
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights="DEFAULT",
        trainable_backbone_layers=trainable_layers,
    )
    model.rpn.anchor_generator = AnchorGenerator(ANCHOR_SIZES, ANCHOR_RATIOS)
    return _replace_heads(model, num_classes)


def build_resnet101(num_classes=NUM_CLASSES, trainable_layers=3):
    """Mask R-CNN with ResNet101-FPN backbone (~65M params).

    ResNet101 provides a deeper feature hierarchy than ResNet50, which
    typically improves AP by 3-5 points on dense instance segmentation tasks.
    Uses ImageNet-pretrained weights for the backbone (allowed by the spec).
    """
    from torchvision.models.resnet import ResNet101_Weights

    backbone = resnet_fpn_backbone(
        backbone_name="resnet101",
        weights=ResNet101_Weights.IMAGENET1K_V2,
        trainable_layers=trainable_layers,
    )
    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=AnchorGenerator(ANCHOR_SIZES, ANCHOR_RATIOS),
        box_detections_per_img=300,
        min_size=600,
        max_size=800,
    )
    return model


def build_model(
    backbone: str = "resnet101",
    num_classes: int = NUM_CLASSES,
    trainable_layers: int = 3,
):
    """Factory function. backbone: 'resnet50' | 'resnet101'."""
    if backbone == "resnet50":
        return build_resnet50(num_classes, trainable_layers)
    elif backbone == "resnet101":
        return build_resnet101(num_classes, trainable_layers)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
