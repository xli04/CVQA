"""
Class Activation Map (CAM) visualization module for MM-Prompt CVQA model.

This module provides functions to generate and visualize attention-based heatmaps 
on images, showing which regions the model attends to when answering questions.
These visualizations help understand the model's cross-modal attention mechanisms
and provide interpretability for the VQA task.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import functional as TF
from PIL import Image
import math
from Question_type import All_task, Comp_task, show_results_matrix, evaluate_metric, Category_splits, ImgId_cate_map, QuesId_task_map, random_dic

def normalize_heatmap(attention):
    """
    Normalize attention scores to range [0, 1] for visualization.
    
    Args:
        attention: Tensor containing raw attention scores
        
    Returns:
        Normalized attention tensor with values in range [0, 1]
        
    Note:
        Handles edge cases where attention range is very small to avoid division by zero
    """
    range_value = attention.max() - attention.min()
    if range_value > 1e-8:  # Avoid division by zero
        attention = (attention - attention.min()) / range_value
    else:
        print("Warning: Attention map has negligible range. Setting to zeros.")
        attention = torch.zeros_like(attention)
    return attention


def visualize_region_attention(image_path, boxes, attn_scores, output_path):
    """
    Generate a heatmap visualization by overlaying region-based attention scores on an image.
    
    Args:
        image_path: Path to the original image file
        boxes: Normalized bounding box coordinates [N, 4] as (x1, y1, x2, y2)
        attn_scores: Attention scores for each box region [N]
        output_path: Path to save the visualization output
        
    Returns:
        None (saves visualization to disk)
        
    Note:
        Attention scores are amplified (squared) to enhance visibility of important regions
    """
    # Load and convert image
    image = Image.open(image_path).convert("RGB")
    W, H = image.size

    # Convert boxes & attn_scores to NumPy if needed
    if hasattr(boxes, 'cpu'):
        boxes = boxes.cpu().numpy()
    if hasattr(attn_scores, 'cpu'):
        attn_scores = attn_scores.cpu().numpy()

    # Amplify small attention scores to make them more visible
    attn_scores = attn_scores ** 2

    # Create an empty map for heatmap
    heatmap = np.zeros((H, W), dtype=float)

    # Fill each bounding box area with its corresponding attention value
    for i, box in enumerate(boxes):
        x1 = int(box[0] * W)
        y1 = int(box[1] * H)
        x2 = int(box[2] * W)
        y2 = int(box[3] * H)
        # Ensure coordinates are within image boundaries
        x1, x2 = max(x1, 0), min(x2, W)
        y1, y2 = max(y1, 0), min(y2, H)
        if x2 <= x1 or y2 <= y1:
            continue
        # Multiply by 100 for better visibility
        heatmap[y1:y2, x1:x2] += attn_scores[i] * 100

    # Handle case where heatmap is empty
    if heatmap.sum() == 0:
        print("Warning: Heatmap is empty. Assigning uniform attention.")
        heatmap += 1  # avoid a completely empty map

    # Normalize heatmap values to [0,1]
    heatmap = normalize_heatmap(torch.tensor(heatmap)).numpy()

    # Convert heatmap to a color map using 'jet' colormap
    cmap = plt.get_cmap("jet")
    heatmap_color = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)

    # Blend the heatmap with the original image (50% transparency)
    heatmap_img = Image.fromarray(heatmap_color)
    image_blend = Image.blend(image, heatmap_img, alpha=0.5)

    # Save result
    image_blend.save(output_path)
    print(f"Saved region-attention heatmap to: {output_path}")

def generate_cam_for_test_set(
    tokenizer, question_ids, attention_imgId_boxes_answer, task, test_task, category,
    patch_size=16
):
    """
    Generate Class Activation Maps (CAMs) for all samples in the test set.
    
    Args:
        tokenizer: Tokenizer used for input processing
        question_ids: List of question IDs to process
        attention_imgId_boxes_answer: Dictionary mapping question IDs to attention data
        task: Current task identifier
        test_task: Test task identifier
        category: Dictionary mapping question IDs to category information
        patch_size: Size of image patches (default: 16)
        
    Returns:
        None (saves visualizations to disk)
        
    Note:
        Creates visualizations showing which regions the model attends to for each test sample.
        Handles the extraction of attention weights from various model architectures.
    """
    # Create output directories for visualizations
    cam_dir = "CAM"
    os.makedirs(os.path.join(cam_dir, f"image_{task}"), exist_ok=True)
    os.makedirs(os.path.join(cam_dir, f"text_{task}"), exist_ok=True)

    # Process each question in the test set
    for idx, (ques_id) in enumerate(question_ids):
        # Extract attention and metadata for current question
        information = attention_imgId_boxes_answer[ques_id]
        cross_attn = information["attentions"]
        img_id = information["img_id"]
        ans = information["answer"]
        boxes = information["boxes"]
        ids = information["inp_id"]
        TEXT_LEN = len(ids)  # Length of text tokens
        IMAGE_LEN = cross_attn.shape[-1] - TEXT_LEN  # Length of image tokens
        img_cate = category[ques_id]["img_cate"]
        que_cate = category[ques_id]["que_cate"]
        
        print(f"Debug (generate_cam_for_test_set): Sample {idx+1}/{len(attention_imgId_boxes_answer.keys())}")

        # Skip samples with empty or invalid attention maps
        if cross_attn.numel() == 0:
            print(f"Warning: Empty cross_attention for {img_id}. Skipping.")
            continue

        # Extract encoder sequence length (typically the last dimension)
        enc_seq_len = cross_attn.shape[-1]

        # Calculate indices for image tokens in the sequence
        image_start = TEXT_LEN
        image_end = TEXT_LEN + IMAGE_LEN
        if image_end > enc_seq_len:
            print(f"  Error: image_end={image_end} > enc_seq_len={enc_seq_len}. Skipping.")
            continue

        # Extract attention scores for image tokens only
        cross_attn_image_only = cross_attn[..., image_start:image_end]

        # Average attention across heads and decoder tokens
        if cross_attn_image_only.dim() == 3:
            # [heads, dec_seq, IMAGE_LEN]
            region_attn_torch = cross_attn_image_only.mean(dim=(0, 1))  # => [IMAGE_LEN]
        elif cross_attn_image_only.dim() == 4:
            # [layers, heads, dec_seq, IMAGE_LEN]; use last layer
            region_attn_torch = cross_attn_image_only[-1].mean(dim=(0, 1))  # => [IMAGE_LEN]
        else:
            print(f"  Unexpected cross_attn_image_only shape {cross_attn_image_only.shape}. Skipping.")
            continue

        # Normalize attention scores to [0,1]
        region_attn_torch = normalize_heatmap(region_attn_torch)
        region_attn = region_attn_torch.cpu().numpy()
        
        # Construct file paths for input image and output visualization
        image_path = f"/root/autodl-tmp/original_images/{img_id}.jpg"
        image_save_path = os.path.join(
            cam_dir, 
            f"image_{task}", 
            f"heatmap_{img_id}_{ques_id}_{ans}_{img_cate}_{que_cate}_{task}_{test_task}.png"
        )

        # Generate and save the visualization
        visualize_region_attention(image_path, boxes, region_attn, image_save_path)




