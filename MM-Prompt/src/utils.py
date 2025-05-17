"""
Utility functions for the MM-Prompt CVQA project.
Includes geometric operations, training metrics, model utilities, and logging functions.
"""

import re
import numpy as np
import torch
import torch.distributed as dist
import collections
import logging

def get_area(pos):
    """
    Calculate the area of bounding boxes.
    
    Args:
        pos: Tensor of shape [B, N, 4] containing bounding box coordinates
             in format (x1, x2, y1, y2)

    Returns:
        Tensor of shape [B, N] containing the areas of each bounding box
    """
    # Calculate height and width from coordinates
    height = pos[:, :, 3] - pos[:, :, 2]
    width = pos[:, :, 1] - pos[:, :, 0]
    area = height * width
    return area

def get_relative_distance(pos):
    """
    Calculate the relative distance between all pairs of bounding boxes.
    
    Args:
        pos: Tensor of shape [B, N, 4] containing bounding box coordinates
             in format (x1, x2, y1, y2)

    Returns:
        Tensor of shape [B, N, N, 4] containing pairwise coordinate differences
    """
    # Create pairwise differences using broadcasting
    relative_distance = pos.unsqueeze(1) - pos.unsqueeze(2)
    return relative_distance


class LossMeter(object):
    """
    Running average calculator for tracking loss values during training.
    Maintains a fixed-length queue of recent values and computes their average.
    """
    def __init__(self, maxlen=100):
        """
        Initialize the loss meter with a maximum queue length.
        
        Args:
            maxlen: Maximum number of values to keep in the running average
        """
        self.vals = collections.deque([], maxlen=maxlen)

    def __len__(self):
        """Return the current number of values in the meter."""
        return len(self.vals)

    def update(self, new_val):
        """
        Add a new value to the meter.
        
        Args:
            new_val: New value to include in the running average
        """
        self.vals.append(new_val)

    @property
    def val(self):
        """
        Calculate the current running average of all values.
        
        Returns:
            Average of all values currently in the meter
        """
        return sum(self.vals) / len(self.vals)

    def __repr__(self):
        """String representation of the current average value."""
        return str(self.val)


def count_parameters(model):
    """
    Count the total number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_state_dict(state_dict_path, loc='cpu'):
    """
    Load a model state dictionary from disk with support for models trained with DataParallel.
    
    Args:
        state_dict_path: Path to the saved state dictionary
        loc: Device to load the state dictionary to (default: 'cpu')
        
    Returns:
        Loaded state dictionary with 'module.' prefix removed if present
    """
    state_dict = torch.load(state_dict_path, map_location=loc)
    # Remove "module." prefix from DataParallel trained models
    original_keys = list(state_dict.keys())
    for key in original_keys:
        if key.startswith("module."):
            new_key = key[len("module."):]
            state_dict[new_key] = state_dict.pop(key)
    return state_dict


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels for specified modules based on name prefixes.
    
    Args:
        level: Target logging level (e.g., logging.ERROR, logging.INFO)
        prefices: List of string prefixes to match logger names 
                 (default: [""] matches all loggers)
                 
    Note:
        Must be called after modules are imported so their loggers are initialized.
        Matching is case-sensitive using module_name.startswith(prefix).
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


def get_iou(anchors, gt_boxes):
    """
    Calculate Intersection over Union (IoU) between sets of bounding boxes.
    
    Args:
        anchors: Tensor of shape (N, 4) containing N bounding boxes
        gt_boxes: Tensor of shape (K, 4) or (4,) containing K or 1 ground truth boxes
        
    Returns:
        Tensor of shape (N, K) containing IoU values for each anchor-gt pair
        
    Note:
        Box format is [x1, y1, x2, y2] where (x1, y1) is the top-left corner
        and (x2, y2) is the bottom-right corner.
    """
    N = anchors.size(0)

    # Handle single gt_box case by reshaping to (1, 4)
    if gt_boxes.size() == (4,):
        gt_boxes = gt_boxes.view(1, 4)
    K = gt_boxes.size(0)

    # Calculate areas of ground truth boxes
    gt_boxes_area = (
        (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) *
        (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
    ).view(1, K)

    # Calculate areas of anchor boxes
    anchors_area = (
        (anchors[:, 2] - anchors[:, 0] + 1) *
        (anchors[:, 3] - anchors[:, 1] + 1)
    ).view(N, 1)

    # Broadcast boxes to shape (N, K, 4) for pairwise calculations
    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    # Calculate intersection width
    iw = (
        torch.min(boxes[:, :, 2], query_boxes[:, :, 2])
        - torch.max(boxes[:, :, 0], query_boxes[:, :, 0])
        + 1
    )
    iw[iw < 0] = 0

    # Calculate intersection height
    ih = (
        torch.min(boxes[:, :, 3], query_boxes[:, :, 3])
        - torch.max(boxes[:, :, 1], query_boxes[:, :, 1])
        + 1
    )
    ih[ih < 0] = 0

    # Calculate IoU = intersection area / union area
    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps


def xywh_to_xyxy(boxes):
    """
    Convert bounding boxes from [x, y, width, height] to [x1, y1, x2, y2] format.
    
    Args:
        boxes: NumPy array of boxes in [x, y, w, h] format
        
    Returns:
        NumPy array of boxes in [x1, y1, x2, y2] format where (x1, y1) is top-left
        and (x2, y2) is bottom-right corner
    """
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))
