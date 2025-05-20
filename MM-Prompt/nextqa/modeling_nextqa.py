"""
MM-Prompt Model for NextQA: Multi-Modal Prompt Tuning for Video-Based Question Answering

This module implements the NextQA adaptation of the MM-Prompt model architecture,
specialized for video question answering tasks. Key components include:

1. Visual feature extraction using LXMERT for processing video frames
2. Cross-modal prompt pools with modality-specific task adapters
3. Hierarchical recovery mechanisms to balance modality importance
4. T5-based encoder-decoder architecture for answer generation

The NextQA adaptation extends the original MM-Prompt with temporal reasoning
capabilities while maintaining the core cross-modal prompt tuning approach.
"""

from dataclasses import dataclass
import transformers
from transformers.models.t5.modeling_t5 import (
    T5Stack, T5Block, T5LayerNorm, T5LayerSelfAttention, T5LayerFF, T5LayerCrossAttention,
    T5PreTrainedModel, T5ForConditionalGeneration, T5Model
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import copy
from transformers.modeling_outputs import ModelOutput, BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging
from transformers import BeamScorer, BeamSearchScorer, LxmertModel, LxmertConfig

logger = logging.get_logger(__name__)

#####################################################################
# PART 1: CORE COMPONENTS & VISUAL EMBEDDING
#####################################################################

class VisualEmbeddingLXMERT(nn.Module):
    """
    Visual features extraction and processing module using LXMERT.
    
    This class processes visual inputs (video frames) by:
    1. Using pretrained LXMERT to extract deep features from input frames
    2. Projecting these features to match the T5 model dimensions
    3. Adding positional and order embeddings for spatial awareness
    4. Applying normalization if configured
    
    Args:
        config: Model configuration object containing parameters
        obj_order_embedding: Embedding layer for object ordering
    """
    def __init__(self, config, obj_order_embedding):
        super().__init__()
        self.config = config
        self.n_images = config.n_images

        # Initialize LXMERT model for visual feature extraction
        self.lxmert, loading_info = LxmertModel.from_pretrained('unc-nlp/lxmert-base-uncased', output_loading_info=True)
        self.lxmert.requires_grad_(False)  # Freeze LXMERT parameters to avoid finetuning
        print("Initialize LXMERT: ", loading_info)

        # Project LXMERT visual output to T5 hidden dimension
        self.feat_projection = nn.Linear(self.lxmert.config.hidden_size, config.d_model)
        
        # Position encoding: 4 coordinates (x1,y1,x2,y2) plus area
        self.absolute_vis_pos_embedding = nn.Linear(5, config.d_model)  # 4 for bbox + 1 for area

        # Optional order embeddings for objects and images (frames)
        if self.config.use_vis_order_embedding:
            self.obj_order_embedding = obj_order_embedding
            self.img_order_embedding = nn.Embedding(self.n_images, config.d_model)

        # Optional layer normalization
        if self.config.use_vis_layer_norm:
            self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def get_area(self, pos):
        """
        Calculate area from bounding box coordinates.
        
        Args:
            pos: Tensor of shape [B, N, 4] containing bounding box coordinates
                in format (x1, y1, x2, y2)
                
        Returns:
            Tensor of shape [B, N] containing area for each bounding box
        """
        height = pos[:, :, 3] - pos[:, :, 2]
        width = pos[:, :, 1] - pos[:, :, 0]
        area = height * width
        return area

    def forward(self, feats, pos, input_ids, img_order_ids=None, obj_order_ids=None):
        """
        Process visual features through LXMERT and prepare for the T5 model.
        
        Args:
            feats: Visual features tensor of shape [B, N, feat_dim]
            pos: Bounding box positions of shape [B, N, 4]
            input_ids: Text input tensor (used only for batch size)
            img_order_ids: Optional image ordering IDs (for frame ordering)
            obj_order_ids: Optional object ordering IDs
            
        Returns:
            vis_embedding: Visual embeddings ready for the encoder
                of shape [B, N, d_model]
        """
        B, N, _ = feats.size()
        assert pos.size() == (B, N, 4)

        # Process features through LXMERT's vision encoder
        # Use dummy text input since we only need the visual branch
        dummy_input_ids = torch.full(
            (B, 1), self.config.pad_token_id, dtype=torch.long, device=feats.device
        ) 
        lxmert_output = self.lxmert(
            input_ids=dummy_input_ids,  # Text input is not needed
            visual_feats=feats,
            visual_pos=pos,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        ).vision_output

        # Project LXMERT output to T5 dimension
        feat_embedding = self.feat_projection(lxmert_output)

        # Process position information with area
        area = self.get_area(pos).unsqueeze(2)
        pos_with_area = torch.cat([pos, area], dim=2)
        pos_embedding = self.absolute_vis_pos_embedding(pos_with_area)

        device = feats.device

        if self.config.use_vis_order_embedding:
            # Handle image order embedding (which frame in sequence)
            if img_order_ids is None:
                img_order_ids = torch.zeros(N, dtype=torch.long, device=device)
                img_order_ids = img_order_ids.unsqueeze(0)
            img_order_embedding = self.img_order_embedding(img_order_ids)

            # Handle object order embedding (which object in frame)
            if obj_order_ids is None:
                obj_order_ids = torch.arange(N, dtype=torch.long, device=device)
                obj_order_ids = obj_order_ids.unsqueeze(0)
            # Invert the order to match the model's expectations
            obj_order_ids = self.obj_order_embedding.num_embeddings - obj_order_ids - 1
            obj_order_embedding = self.obj_order_embedding(obj_order_ids)

            # Combine all embeddings: features + position + ordering
            vis_embedding = feat_embedding + pos_embedding + img_order_embedding + obj_order_embedding
        else:
            # Just combine features and position if no ordering
            vis_embedding = feat_embedding + pos_embedding

        # Apply layer normalization if configured
        if not self.config.individual_vis_layer_norm:
            if self.config.use_vis_layer_norm:
                vis_embedding = self.layer_norm(vis_embedding)

        return vis_embedding

class VLCouplingFunction(nn.Module):
    """
    Vision-Language coupling function for cross-modal alignment.
    
    Creates explicit conditioning between vision and language prompts
    to enhance cross-modal alignment. Uses linear projections with near-identity
    initialization to allow gradual learning of cross-modal relationships.
    
    Args:
        d_model: Hidden dimension size of the model
    """
    def __init__(self, d_model):
        super().__init__()
        # Linear projections for bidirectional conditioning
        self.q_to_v = nn.Linear(d_model, d_model, bias=False)
        self.v_to_q = nn.Linear(d_model, d_model, bias=False)
        
        # Initialize with near-identity matrices for gentle adaptation
        nn.init.eye_(self.q_to_v.weight)
        nn.init.eye_(self.v_to_q.weight)
        
        # Add small random noise to break symmetry and enable learning
        with torch.no_grad():
            self.q_to_v.weight.add_(torch.randn_like(self.q_to_v.weight) * 0.02)
            self.v_to_q.weight.add_(torch.randn_like(self.v_to_q.weight) * 0.02)
    
    def forward(self, q_prompt, v_prompt):
        """
        Apply bidirectional coupling between text and vision prompts.
        
        Args:
            q_prompt: Language (query) prompt tensor
            v_prompt: Vision prompt tensor
            
        Returns:
            Tuple containing:
                q_conditioned: Language prompt conditioned by vision
                v_conditioned: Vision prompt conditioned by language
        """
        # Condition vision prompt on language (with scaling factor)
        v_conditioned = v_prompt + 0.1 * self.q_to_v(q_prompt)
        
        # Condition language prompt on vision (with scaling factor)
        q_conditioned = q_prompt + 0.1 * self.v_to_q(v_prompt)
        
        return q_conditioned, v_conditioned
    
    def compute_orthogonality_loss(self):
        """
        Compute orthogonality loss for coupling matrices.
        
        Encourages the coupling matrices to remain close to orthogonal
        transformations, which helps prevent overfitting and preserve
        important information during cross-modal transfer.
        
        Returns:
            Combined orthogonality loss for both coupling directions
        """
        # Calculate how far q_to_v is from being orthogonal
        q2v_loss = torch.norm(
            torch.matmul(self.q_to_v.weight, self.q_to_v.weight.t()) - 
            torch.eye(self.q_to_v.weight.size(0), device=self.q_to_v.weight.device),
            p='fro'  # Frobenius norm
        )
        
        # Calculate how far v_to_q is from being orthogonal
        v2q_loss = torch.norm(
            torch.matmul(self.v_to_q.weight, self.v_to_q.weight.t()) - 
            torch.eye(self.v_to_q.weight.size(0), device=self.v_to_q.weight.device),
            p='fro'  # Frobenius norm
        )
        
        return q2v_loss + v2q_loss

#####################################################################
# PART 2: PROMPT REPRESENTATION & SELECTION
#####################################################################

class DecomposedPromptPool(nn.Module):
    """
    Decomposed prompt representation for flexible prompt adaptation.
    
    Maintains a pool of prompt components that can be selected and
    combined based on the input query. This approach allows for more
    flexible prompt composition for different video question types.
    
    Args:
        num_components: Number of prompt components in the pool
        d_model: Hidden dimension size of the model
        pool_name: Optional name for the prompt pool (for logging/debugging)
    """
    def __init__(self, num_components, d_model, pool_name=""):
        super().__init__()
        # Learnable parameters for prompt components
        self.prompt_components = nn.Parameter(torch.randn(num_components, d_model))
        self.component_keys = nn.Parameter(torch.randn(num_components, d_model))
        self.component_attention = nn.Parameter(torch.randn(num_components, d_model))
        self.d_model = d_model
        self.name = pool_name
        self.num_components = num_components
        
        # Layer for fusing multiple components
        self.query_fusion = nn.Linear(d_model * 2, d_model)
        
        # Initialize with small random values for stable training
        nn.init.normal_(self.prompt_components, std=0.02)
        nn.init.normal_(self.component_keys, std=0.02)
        nn.init.normal_(self.component_attention, std=0.02)
        
    def compute_orthogonality_loss(self):
        """
        Calculate orthogonality loss for prompt components.
        
        Encourages diversity among prompt components by penalizing
        redundancy in the component representations.
        
        Returns:
            Orthogonality loss value
        """
        # Calculate Gram matrix (component-to-component similarity)
        gram_matrix = torch.matmul(self.prompt_components, self.prompt_components.transpose(0, 1))
        # Compare with identity matrix (zero off-diagonal = orthogonal components)
        identity = torch.eye(self.prompt_components.size(0), device=self.prompt_components.device)
        # Frobenius norm of difference
        loss = torch.norm(gram_matrix - identity, p='fro')
        return loss
    
    def select_prompts(self, query, top_k=5):
        """
        Select prompts using a single query vector with detached keys.
        
        The selection process works by:
        1. Computing similarity between the query and component keys
        2. Selecting the top-k most similar components
        3. Weighting and combining these components
        
        Args:
            query: Feature vector [B, d_model]
            top_k: Number of components to select
        
        Returns:
            selected_prompts: Weighted combination of prompt components [B, 1, d_model]
            top_indices: Indices of selected components [B, top_k]
            selected_keys: Selected component keys [B, top_k, d_model]
        """
        B = query.size(0)  # Batch size
        
        # Apply attention weights to focus the query
        attended_query = query * self.component_attention.mean(dim=0)
        
        # Calculate similarities with detached keys
        # Detaching prevents gradients from flowing directly to key selection
        detached_keys = self.component_keys.detach()
        
        # Compute similarity with each component key
        similarities = []
        for i in range(self.num_components):
            key = detached_keys[i]
            sim = F.cosine_similarity(attended_query, key.unsqueeze(0), dim=1)
            similarities.append(sim)
        
        # Stack similarities [B, num_components]
        similarities = torch.stack(similarities, dim=1)
        
        # Get top-k components based on similarity
        top_values, top_indices = torch.topk(similarities, k=top_k, dim=1)
        
        # Normalize weights with softmax for weighted combination
        weights = F.softmax(top_values, dim=1)
        
        # Select components based on indices
        selected_prompts = torch.zeros(B, self.d_model, device=query.device)
        selected_keys = []
        
        # Gather and combine components for each batch element
        for b in range(B):
            batch_keys = []
            for i, idx in enumerate(top_indices[b]):
                # Add weighted component to prompt
                selected_prompts[b] += weights[b, i] * self.prompt_components[idx]
                
                # Get the original (non-detached) key for loss calculation
                batch_keys.append(self.component_keys[idx])
            
            # Stack keys for this batch
            selected_keys.append(torch.stack(batch_keys))
        
        # Stack all batch keys [B, top_k, d_model]
        selected_keys = torch.stack(selected_keys)
        
        # Reshape prompts to [B, 1, d_model] for compatibility with model
        return selected_prompts.unsqueeze(1), top_indices, selected_keys


#####################################################################
# PART 3: HIERARCHICAL RECOVERY (ARMOR CORE)
#####################################################################

class EnhancedHierarchicalRecovery(nn.Module):
    """
    Enhanced hierarchical recovery with improved coupling and progressive refinement.
    Avoids in-place operations to prevent autograd issues.
    """
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, prompt_types=["QG", "QE", "VG", "VE"]):
        super().__init__()
        
        self.d_model = d_model
        self.recovery_modules = nn.ModuleDict()
        
        # Create specialized modules for each prompt type
        for prompt_type in prompt_types:
            self.recovery_modules[prompt_type] = SingleHierarchicalRecovery(d_model, nhead, dim_feedforward)
            
        # Cross-modal coupling functions for each level
        self.coarse_coupling = VLCouplingFunction(d_model)
        self.refined_coupling = VLCouplingFunction(d_model)
        self.detail_coupling = VLCouplingFunction(d_model)
        
    def forward(self, masked, context, type):
        """
        Perform hierarchical recovery with cross-modal coupling.
        Avoids in-place operations to prevent autograd issues.
        
        Args:
            masked: Masked prompt to recover [B, L, d_model]
            context: Context information (from other modality) [B, L, d_model]
            type: Type of prompt to recover (QG, QE, VG, or VE)
        """
        # Get the appropriate recovery module
        recovery_module = self.recovery_modules[type]
        
        # Create mask based on zero values (Boolean mask where tokens are masked)
        zero_mask = (masked.abs().sum(dim=-1) < 1e-6)  # [B, L]
        
        # Level 1: Coarse recovery
        coarse_recovered = recovery_module.coarse_recovery(masked)
        
        # Save original for loss calculation
        recovery_module.coarse_output = coarse_recovered.detach()
        
        # Apply coupling if we have both modalities
        if "Q" in type and context is not None:
            # This is Q recovery with V context
            coarse_q, _ = self.coarse_coupling(coarse_recovered, context)
            coarse_recovered = coarse_q
        elif "V" in type and context is not None:
            # This is V recovery with Q context
            _, coarse_v = self.coarse_coupling(context, coarse_recovered)
            coarse_recovered = coarse_v
            
        # Level 2: Context-aware refinement
        context_features, _ = recovery_module.context_attention(
            query=coarse_recovered,
            key=context,
            value=context
        )
        
        # Combine with coarse features
        refined = coarse_recovered + 0.5 * context_features
        refined = recovery_module.refinement(refined)
        
        # Apply coupling at refinement level
        if "Q" in type and context is not None:
            refined_q, _ = self.refined_coupling(refined, context)
            refined = refined_q
        elif "V" in type and context is not None:
            _, refined_v = self.refined_coupling(context, refined)
            refined = refined_v
        
        # Save refined output for loss calculation
        recovery_module.refined_output = refined.detach()
        
        # Level 3: Detail Enhancement (conditional)
        # Calculate gate values to determine which tokens need detail enhancement
        avg_context = context.mean(dim=1, keepdim=True).expand_as(refined)
        gate_input = torch.cat([refined, avg_context], dim=-1)
        gate_values = recovery_module.detail_gate(gate_input)
        
        # Reshape gate values to match zero_mask
        gate_values = gate_values.squeeze(-1)  # [B, L]
        
        # Combine gate and zero mask
        detail_mask = (gate_values > 0.5) & zero_mask  # [B, L]
        
        # Apply detail enhancement only if any token qualifies
        if detail_mask.any():
            detailed = recovery_module.detail_enhancement(refined)
            
            # Apply coupling at detail level
            if "Q" in type and context is not None:
                detailed_q, _ = self.detail_coupling(detailed, context)
                detailed = detailed_q
            elif "V" in type and context is not None:
                _, detailed_v = self.detail_coupling(context, detailed)
                detailed = detailed_v
            
            # Create a new tensor instead of modifying in-place
            # Use where to conditionally select values
            enhanced = torch.where(
                detail_mask.unsqueeze(-1),  # Condition: [B, L, 1]
                detailed,                   # If True: [B, L, d_model]
                refined                     # If False: [B, L, d_model]
            )
            
            # Replace refined with enhanced version
            refined = enhanced
            
        return refined
    
    def compute_coupling_loss(self):
        """Compute orthogonality loss for all coupling functions."""
        coupling_loss = (
            self.coarse_coupling.compute_orthogonality_loss() +
            self.refined_coupling.compute_orthogonality_loss() +
            self.detail_coupling.compute_orthogonality_loss()
        )
        return 0.01 * coupling_loss


class SingleHierarchicalRecovery(nn.Module):
    """
    Single hierarchical recovery module for a specific prompt type.
    """
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        
        # Level 1: Coarse Recovery - lighter weight
        self.coarse_recovery = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, 
                nhead=4,  # Reduced heads
                dim_feedforward=dim_feedforward//2,  # Smaller FFN
                dropout=0.1, 
                batch_first=True
            ),
            num_layers=2  # Fewer layers
        )
        
        # Level 2: Refinement - add context information
        self.context_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=nhead, 
            batch_first=True
        )
        
        self.refinement = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, 
                nhead=nhead,
                dim_feedforward=dim_feedforward, 
                dropout=0.1, 
                batch_first=True
            ),
            num_layers=1
        )
        
        # Gate for determining if detail enhancement is needed
        self.detail_gate = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # Level 3: Detail enhancement
        self.detail_enhancement = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, 
                nhead=nhead,
                dim_feedforward=dim_feedforward*2,  # Larger FFN for details
                dropout=0.1, 
                batch_first=True
            ),
            num_layers=1
        )
        
        # For storing intermediate outputs
        self.coarse_output = None
        self.refined_output = None

#####################################################################
# PART 4: PROMPT FUSION & UTILITY FUNCTIONS
#####################################################################

class PromptFusion(nn.Module):
    def __init__(self, d_model, nheads, lambda_reg=0.2):
        super().__init__()
        # Cross-attention modules for different prompt types
        self.cross_attn_Qe = nn.MultiheadAttention(d_model, nheads, batch_first=True)
        self.cross_attn_Ve = nn.MultiheadAttention(d_model, nheads, batch_first=True)
        self.cross_attn_Qg = nn.MultiheadAttention(d_model, nheads, batch_first=True)
        self.cross_attn_Vg = nn.MultiheadAttention(d_model, nheads, batch_first=True)
        
        # MLP layers for refinement
        self.mlp_Q = nn.Linear(d_model, d_model)
        self.mlp_V = nn.Linear(d_model, d_model)
        
        # Coupling functions for cross-modal integration
        self.coupling_e = VLCouplingFunction(d_model)
        self.coupling_g = VLCouplingFunction(d_model)

        # Save the initial attention weights for regularization
        self.initial_weight_Qe = None
        self.initial_weight_Ve = None
        self.initial_weight_Qg = None
        self.initial_weight_Vg = None
        
        # Regularization strength
        self.lambda_reg = lambda_reg
        self.reg_loss = 0.0

    def forward(self, recovered_Q, selected_V, recovered_V, selected_Q, type):
        """
        Fuse recovered and selected prompts with cross-attention and coupling.
        """
        if type == "E":
            # Apply cross-attention
            attended_Q, _ = self.cross_attn_Qe(
                query=recovered_Q,
                key=selected_V,
                value=selected_V
            )
            
            attended_V, _ = self.cross_attn_Ve(
                query=recovered_V,
                key=selected_Q,
                value=selected_Q
            )
            
            # Apply MLP
            attended_Q = attended_Q + self.mlp_Q(attended_Q)
            attended_V = attended_V + self.mlp_V(attended_V)
            
            # Apply coupling function
            fused_Q, fused_V = self.coupling_e(attended_Q, attended_V)
            
        else:  # General prompts
            # Apply cross-attention
            attended_Q, _ = self.cross_attn_Qg(
                query=recovered_Q,
                key=selected_V,
                value=selected_V
            )
            
            attended_V, _ = self.cross_attn_Vg(
                query=recovered_V,
                key=selected_Q,
                value=selected_Q
            )
            
            # Apply MLP
            attended_Q = attended_Q + self.mlp_Q(attended_Q)
            attended_V = attended_V + self.mlp_V(attended_V)
            
            # Apply coupling function
            fused_Q, fused_V = self.coupling_g(attended_Q, attended_V)
            
        return fused_Q, fused_V
    
    def compute_reg_loss(self):
        """Compute regularization loss to maintain attention stability."""
        if self.initial_weight_Qe is None:
            return 0.0
            
        reg_loss = 0.0
        
        # Compare current learned weight with the initial weight using cosine similarity
        reg_loss += 1 - F.cosine_similarity(
            self.cross_attn_Qe.in_proj_weight.view(-1), 
            self.initial_weight_Qe.view(-1), 
            dim=0
        ).mean()
        
        reg_loss += 1 - F.cosine_similarity(
            self.cross_attn_Ve.in_proj_weight.view(-1), 
            self.initial_weight_Ve.view(-1), 
            dim=0
        ).mean()
        
        reg_loss += 1 - F.cosine_similarity(
            self.cross_attn_Qg.in_proj_weight.view(-1), 
            self.initial_weight_Qg.view(-1), 
            dim=0
        ).mean()
        
        reg_loss += 1 - F.cosine_similarity(
            self.cross_attn_Vg.in_proj_weight.view(-1), 
            self.initial_weight_Vg.view(-1), 
            dim=0
        ).mean()
        
        # Add coupling function regularization
        reg_loss += 0.1 * (
            self.coupling_e.compute_orthogonality_loss() +
            self.coupling_g.compute_orthogonality_loss()
        )
        
        return self.lambda_reg * reg_loss
    
    def update_attention_weight(self):
        """Store current attention weights for regularization."""
        self.initial_weight_Qe = self.cross_attn_Qe.in_proj_weight.clone().detach()
        self.initial_weight_Ve = self.cross_attn_Ve.in_proj_weight.clone().detach() 
        self.initial_weight_Qg = self.cross_attn_Qg.in_proj_weight.clone().detach()
        self.initial_weight_Vg = self.cross_attn_Vg.in_proj_weight.clone().detach()

def random_mask_shared(prompts_Q, prompts_V, mask_ratio=0.05, mask_mode="zeros"):
    """
    Apply shared masking to language and vision prompts.
    Uses non-in-place operations to avoid autograd issues.
    
    Args:
        prompts_Q: Language prompts [B, k, d_model]
        prompts_V: Vision prompts [B, k, d_model]
        mask_ratio: Percentage of tokens to mask
        mask_mode: How to mask tokens ("zeros" replaces with zeros)
    
    Returns:
        masked_Q: Masked language prompts
        masked_V: Masked vision prompts
        mask: Boolean mask indicating which positions are masked
    """
    B, k, d_model = prompts_Q.shape
    device = prompts_Q.device

    # Generate random mask
    rand_mat = torch.rand(B, k, device=device)
    mask = (rand_mat < mask_ratio)  # [B, k]

    if mask_mode == "zeros":
        # Create new tensors instead of modifying in place
        masked_Q = torch.where(
            mask.unsqueeze(-1),       # Condition: [B, k, 1]
            torch.zeros_like(prompts_Q),  # If True: [B, k, d_model]
            prompts_Q                    # If False: [B, k, d_model]
        )
        
        masked_V = torch.where(
            mask.unsqueeze(-1),       # Condition: [B, k, 1]
            torch.zeros_like(prompts_V),  # If True: [B, k, d_model]
            prompts_V                    # If False: [B, k, d_model]
        )
    else:
        # For other modes (if implemented)
        masked_Q = prompts_Q
        masked_V = prompts_V

    return masked_Q, masked_V, mask

#####################################################################
# PART 5: MAIN ENCODER IMPLEMENTATION
#####################################################################

class EnhancedJointEncoder(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super(T5Stack, self).__init__(config)
        self.config = config
        self.embed_tokens = embed_tokens
        self.is_decoder = self.config.is_decoder
        assert self.config.is_decoder is False
        
        num_layers = config.num_layers
        self.visual_embedding = VisualEmbeddingLXMERT(self.config, embed_tokens)

        # Prompt injection layers
        self.G_prompt_layer = [1, 2]
        self.E_prompt_layer = [3, 4, 5]
        self.prompt_layer = [1, 2, 3, 4, 5]

        self.block = nn.ModuleList([
            T5Block(config, has_relative_attention_bias=True if (i == 0 or i in self.prompt_layer) else False)
            for i in range(num_layers)
        ])
        
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.init_weights()
        self.model_parallel = False
        self.device_map = None
        
        print("========== Enhanced ARMOR Joint Encoder ========== ")

        # 1. Decomposed Prompt Pools
        self.GQ_pool = DecomposedPromptPool(num_components=40, d_model=config.d_model, pool_name="GQ")
        self.EQ_pool = DecomposedPromptPool(num_components=60, d_model=config.d_model, pool_name="EQ") 
        self.GV_pool = DecomposedPromptPool(num_components=80, d_model=config.d_model, pool_name="GV")
        self.EV_pool = DecomposedPromptPool(num_components=120, d_model=config.d_model, pool_name="EV")
        
        # 2. Query Projections
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        
        # 3. Cross-Modal Attention (for initial feature enrichment)
        self.q_to_v_attention = nn.MultiheadAttention(
            embed_dim=config.d_model, 
            num_heads=8, 
            batch_first=True
        )
        self.v_to_q_attention = nn.MultiheadAttention(
            embed_dim=config.d_model, 
            num_heads=8, 
            batch_first=True
        )
        
        # 4. Hierarchical Recovery System (core ARMOR mechanism)
        self.recovery = EnhancedHierarchicalRecovery(
            d_model=config.d_model,
            nhead=8,
            dim_feedforward=config.d_model*4
        )
        
        # 5. Prompt Fusion for Modal Integration
        self.prompt_fusion = PromptFusion(config.d_model, 8)
        
        # Loss tracking
        self.loss_memory_Q = None
        self.ortho_loss_weight = 0.01

    def update_attention_weight(self):
        self.prompt_fusion.update_attention_weight()

    def set_input_embeddings(self, new_embeddings):
        """Set input embeddings."""
        self.embed_tokens = new_embeddings
        self.visual_embedding.obj_order_embedding = new_embeddings
    
    def compute_orthogonality_loss(self):
        """Compute orthogonality loss for all component pools."""
        ortho_loss = (
            self.GQ_pool.compute_orthogonality_loss() +
            self.EQ_pool.compute_orthogonality_loss() +
            self.GV_pool.compute_orthogonality_loss() +
            self.EV_pool.compute_orthogonality_loss()
        )
        return self.ortho_loss_weight * ortho_loss
    
    def select_prompts(self, text_query, visual_query, pool_type, top_k=5):
        """
        Select prompts using only the relevant modality.
        Also returns selected keys for loss calculation.
        
        Args:
            text_query: Text features [B, L, d_model] or [B, d_model]
            visual_query: Visual features [B, V_L, d_model] or [B, d_model]
            pool_type: Pool type ("GQ", "EQ", "GV", "EV")
            top_k: Number of components to select
        
        Returns:
            selected_prompts: Selected prompt values
            indices: Indices of selected components
            selected_keys: Selected component keys
        """
        # Ensure we have 2D tensors [B, d_model]
        if text_query is not None and len(text_query.shape) == 3:
            text_query = text_query.mean(dim=1)  # [B, d_model]
        
        if visual_query is not None and len(visual_query.shape) == 3:
            visual_query = visual_query.mean(dim=1)  # [B, d_model]
        
        # Choose the appropriate query based on pool type
        is_text_pool = "Q" in pool_type  # GQ or EQ pools are text pools
        
        if is_text_pool:
            # Use text query for text pools
            query = self.q_proj(text_query)
        else:
            # Use visual query for visual pools
            query = self.v_proj(visual_query)
        
        # Get the appropriate pool
        pool = getattr(self, f"{pool_type}_pool")
        
        # Select prompts using single modality
        selected_prompts, indices, selected_keys = pool.select_prompts(query, top_k)
        
        return selected_prompts, indices, selected_keys
    
    def alignment_loss(self, recovered_Q, selected_V, recovered_V, selected_Q, type):
        """
        Compute alignment loss between recovered and selected prompts.
        """
        # Apply prompt fusion
        E_Q, E_V = self.prompt_fusion(recovered_Q, selected_V, recovered_V, selected_Q, type)
        
        # Average across prompt dimension
        E_Q_avg = E_Q.mean(dim=1)  # [B, d_model]
        E_V_avg = E_V.mean(dim=1)  # [B, d_model]
        
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(E_Q_avg, E_V_avg, dim=-1)  # [B]
        
        # Alignment loss is 1 - cosine similarity
        align_loss = (1.0 - cos_sim).mean()
        
        return align_loss

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        vis_inputs=None,
        vis_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=True,
        output_hidden_states=None,
        return_dict=None,
    ):
        #-------------------------------------------------------------------
        # 1. PREPARE INPUTS AND EMBEDDINGS
        #-------------------------------------------------------------------
        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        B, L = inputs_embeds.size()[:-1]

        # Process visual inputs
        vis_feats = vis_inputs[0]
        boxes = vis_inputs[1]
        img_order_ids = None if len(vis_inputs) < 3 else vis_inputs[2]
        obj_order_ids = None if len(vis_inputs) < 4 else vis_inputs[3]

        vis_embeds = self.visual_embedding(
            vis_feats, boxes, input_ids, img_order_ids, obj_order_ids
        )

        V_L = vis_embeds.size(1)

        # Concatenate text and vision embeddings
        inputs_embeds = torch.cat([inputs_embeds, vis_embeds], dim=1)

        #-------------------------------------------------------------------
        # 2. PREPARE ATTENTION MASKS AND INITIAL HIDDEN STATES
        #-------------------------------------------------------------------
        # Initialize key-value states
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

            # Create attention mask
            if attention_mask is None:
                attention_mask = input_ids.ne(self.config.pad_token_id).to(
                    dtype=inputs_embeds.dtype,
                    device=inputs_embeds.device
                )
                
            # Create vision attention mask
            if vis_attention_mask is None:
                vis_attention_mask = attention_mask.new_ones(B, V_L)

            # Combine masks
            attention_mask = torch.cat([attention_mask, vis_attention_mask], dim=1)
            
            # Make mask broadcastable
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask, 
                (B, L + V_L),
                inputs_embeds.device
            )

        # Prepare head mask
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None

        # Apply dropout to input embeddings
        hidden_states = self.dropout(inputs_embeds)

        # Compute initial position bias
        seq_length = L + V_L
        text_position_bias = self.block[0].layer[0].SelfAttention.compute_bias(L, L)
        num_heads = text_position_bias.size(1)
        position_bias = text_position_bias.new_zeros(1, num_heads, seq_length, seq_length)
        position_bias[:, :, :L, :L] = text_position_bias
        position_bias = position_bias + extended_attention_mask

        #-------------------------------------------------------------------
        # 3. CROSS-MODAL FEATURE ENRICHMENT
        #-------------------------------------------------------------------
        # Initialize loss accumulator
        cumulative_losses = 0.0
        
        # Extract text and visual features
        Q_feature = hidden_states[:, :L, :]
        V_feature = hidden_states[:, L:L+V_L, :]
        
        # Apply cross-modal attention
        Q_query, _ = self.q_to_v_attention(Q_feature, V_feature, V_feature)
        V_query, _ = self.v_to_q_attention(V_feature, Q_feature, Q_feature)
        
        # Add residual connection
        Q_query = Q_query + Q_feature
        V_query = V_query + V_feature

        #-------------------------------------------------------------------
        # 4. PROMPT SELECTION AND SIMILARITY LOSS CALCULATION
        #-------------------------------------------------------------------
        # Select prompts from pools using both modalities
        selected_QG, qg_indices, selected_QG_keys = self.select_prompts(Q_query, V_query, "GQ", top_k=5)
        selected_QE, qe_indices, selected_QE_keys = self.select_prompts(Q_query, V_query, "EQ", top_k=5)
        selected_VG, vg_indices, selected_VG_keys = self.select_prompts(Q_query, V_query, "GV", top_k=5)
        selected_VE, ve_indices, selected_VE_keys = self.select_prompts(Q_query, V_query, "EV", top_k=5)
        
        # Add key similarity losses - this will update only the selected keys
        key_loss_QG = (1-F.cosine_similarity(
            Q_query.mean(dim=1).unsqueeze(1).expand_as(selected_QG_keys), 
            selected_QG_keys, 
            dim=-1
        )).mean()
        
        key_loss_QE = (1-F.cosine_similarity(
            Q_query.mean(dim=1).unsqueeze(1).expand_as(selected_QE_keys), 
            selected_QE_keys, 
            dim=-1
        )).mean()
        
        key_loss_VG = (1-F.cosine_similarity(
            V_query.mean(dim=1).unsqueeze(1).expand_as(selected_VG_keys), 
            selected_VG_keys, 
            dim=-1
        )).mean()
        
        key_loss_VE = (1-F.cosine_similarity(
            V_query.mean(dim=1).unsqueeze(1).expand_as(selected_VE_keys), 
            selected_VE_keys, 
            dim=-1
        )).mean()
        
        # Add similarity losses
        cumulative_losses += 0.05 * (key_loss_QG + key_loss_QE + key_loss_VG + key_loss_VE)
        
        # Add orthogonality loss
        cumulative_losses += self.compute_orthogonality_loss()

        #-------------------------------------------------------------------
        # 5. TRANSFORMER LAYER PROCESSING WITH PROMPT INJECTION
        #-------------------------------------------------------------------
        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            # Store hidden states if needed
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Inject prompts at specified layers
            if i in self.prompt_layer:
                # Select appropriate prompts based on layer
                if i in self.E_prompt_layer:
                    selected_Q = selected_QE
                    selected_V = selected_VE
                    prompt_type = "E"
                else:
                    selected_Q = selected_QG
                    selected_V = selected_VG
                    prompt_type = "G"
                
                #-------------------------------------------------------
                # 5.1 MASKING AND RECOVERY (ARMOR CORE)
                #-------------------------------------------------------
                masked_Q, masked_V, QV_mask = random_mask_shared(selected_Q, selected_V)
                
                if QV_mask.any():
                    # Apply recovery
                    recovered_Q = self.recovery(
                        masked_Q, 
                        selected_V, 
                        f"Q{prompt_type}"
                    )
                    
                    recovered_V = self.recovery(
                        masked_V, 
                        selected_Q, 
                        f"V{prompt_type}"
                    )
                    
                    # Update masked values without in-place operations
                    mask_expanded = QV_mask.unsqueeze(-1)
                    
                    # Create new tensors using torch.where
                    masked_Q = torch.where(
                        mask_expanded,
                        recovered_Q,
                        masked_Q
                    )
                    
                    masked_V = torch.where(
                        mask_expanded,
                        recovered_V,
                        masked_V
                    )
                    
                    # Update masked values with recovered ones
                    masked_Q = masked_Q.clone()
                    masked_V = masked_V.clone()
                    
                    mask_expanded = QV_mask.unsqueeze(-1).expand_as(masked_Q)
                    masked_Q[mask_expanded] = recovered_Q[mask_expanded]
                    masked_V[mask_expanded] = recovered_V[mask_expanded]
                    
                    # Calculate alignment loss
                    align_loss = self.alignment_loss(recovered_Q, selected_V, recovered_V, selected_Q, prompt_type)
                    cumulative_losses += align_loss
                    
                    # Calculate hierarchical recovery losses
                    if self.training:
                        # Get original prompts
                        original_Q = selected_Q.clone()
                        original_V = selected_V.clone()
                        
                        # Calculate losses at each level
                        coarse_loss_Q = F.mse_loss(
                            self.recovery.recovery_modules[f"Q{prompt_type}"].coarse_output[mask_expanded], 
                            original_Q[mask_expanded]
                        )
                        
                        coarse_loss_V = F.mse_loss(
                            self.recovery.recovery_modules[f"V{prompt_type}"].coarse_output[mask_expanded], 
                            original_V[mask_expanded]
                        )
                        
                        # Add hierarchical losses
                        hierarchical_loss = 0.3 * (coarse_loss_Q + coarse_loss_V)
                        cumulative_losses += hierarchical_loss
                
                #-------------------------------------------------------
                # 5.2 PROMPT FUSION AND INJECTION
                #-------------------------------------------------------
                # Concatenate with hidden states
                hidden_states = torch.cat([hidden_states, masked_Q, masked_V], dim=1)
                
                # Update attention mask and position bias
                new_prompts_num = masked_Q.size(1) + masked_V.size(1)
                old_seq_len = attention_mask.size(1)
                
                new_mask = attention_mask.new_ones((B, old_seq_len + new_prompts_num))
                new_mask[:, :old_seq_len] = attention_mask
                attention_mask = new_mask
                
                # Update extended attention mask
                extended_attention_mask = self.get_extended_attention_mask(
                    attention_mask, 
                    (B, attention_mask.size(1)), 
                    hidden_states.device
                )
                
                # Update position bias
                pos_bias_len = hidden_states.size(1)
                
                if layer_module.layer[0].SelfAttention.has_relative_attention_bias:
                    # Compute new position bias
                    new_position_bias = layer_module.layer[0].SelfAttention.compute_bias(
                        pos_bias_len, 
                        pos_bias_len
                    )
                    new_position_bias = new_position_bias + extended_attention_mask
                    position_bias = new_position_bias
                else:
                    # Slice existing position bias and add extended attention mask
                    position_bias = position_bias[..., :pos_bias_len, :pos_bias_len] + extended_attention_mask
            
            #-------------------------------------------------------------------
            # 6. STANDARD TRANSFORMER PROCESSING 
            #-------------------------------------------------------------------
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                encoder_decoder_position_bias=None,
                head_mask=head_mask[i],
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            
            # Update hidden states and position bias
            hidden_states, present_key_value_state = layer_outputs[:2]
            position_bias = layer_outputs[2]
            
            # Store key-value states if using cache
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)
            
            # Store attention outputs if requested
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
        
        #-------------------------------------------------------------------
        # 7. FINALIZATION AND RETURN
        #-------------------------------------------------------------------
        # Apply final layer norm and dropout
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Add prompt fusion regularization loss
        reg_loss = self.prompt_fusion.compute_reg_loss()
        if reg_loss != 0:
            cumulative_losses += 0.1*reg_loss
        
        # Store total loss
        self.loss_memory_Q = cumulative_losses.to(hidden_states.device)
        
        # Return outputs
        if not return_dict:
            return (hidden_states, present_key_value_states, all_hidden_states, all_attentions, all_cross_attentions)
            
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

class VLT5(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super(T5ForConditionalGeneration, self).__init__(config)

        self.config = config

        self.model_dim = config.d_model

        t5_model = T5Model.from_pretrained("t5-base")  # Load pre-trained T5
        self.shared = t5_model.shared
        self.shared.weight.requires_grad = False  # Freezes the embeddings

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        #---- Modified ----#
        # self.encoder = T5Stack(encoder_config, self.shared)
        self.encoder = EnhancedJointEncoder(encoder_config, self.shared)
        #------------------#

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True # decoder
        decoder_config.is_encoder_decoder = False

        self.decoder = T5Stack(decoder_config, self.shared) # decoder

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.prototype_fc1 = nn.Linear(config.d_model, config.d_model)
        self.prototype_fc2 = nn.Linear(config.d_model, config.d_model)
        self.L = 20
        self.V_L = 36


        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def extend_vocab(self, vocab_size):

        new_shared = nn.Embedding(vocab_size, self.config.d_model)
        old_weight = self.shared.weight.data.detach().clone()
        old_vocab_size = old_weight.size(0)
        new_shared.weight.data[:old_vocab_size, :] = old_weight
        self.shared = new_shared

        new_lm_head = nn.Linear(self.config.d_model, vocab_size, bias=False)
        old_weight = self.lm_head.weight.data.detach().clone()
        old_vocab_size = old_weight.size(0)
        new_lm_head.weight.data[:old_vocab_size, :] = old_weight
        self.lm_head = new_lm_head

        self.vis_encoder.visual_embedding.obj_order_embedding = self.shared

        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

        self.lm_head.weight = self.shared.weight

        self.config.vocab_size = vocab_size
        self.encoder.config.vocab_size = vocab_size
        self.vis_encoder.config.vocab_size = vocab_size
        self.decoder.config.vocab_size = vocab_size

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,

        vis_inputs=None,
        vis_attention_mask=None,

        decoder_input_ids=None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        reduce_loss=False,

        return_hidden_state=False,

        **kwargs,
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:

            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,

                vis_inputs=vis_inputs,
                vis_attention_mask=vis_attention_mask,

                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(
                    encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(
                    encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0] # [bs, L+V_L, 768]


        # ==================================================================
        if 'cate_labels' in kwargs:
            cate_labels = kwargs['cate_labels'] #[bs, num_classes]
        if 'ques_labels' in kwargs:
            ques_labels = kwargs['ques_labels'] #[bs, num_classes]

        if 'current_task_id' in kwargs:
            current_task_id = kwargs['current_task_id']

        if 'proto_alpha' in kwargs:
            proto_alpha = kwargs['proto_alpha']
        if 'proto_beta' in kwargs:
            proto_beta = kwargs['proto_beta']


        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        if attention_mask is None: 
            attention_mask = input_ids.ne(self.config.pad_token_id).to(dtype=hidden_states.dtype, device=hidden_states.device) #[bs, L]

        if vis_attention_mask is None:
            B, L = attention_mask.size()
            V_L = hidden_states.size(1) - L
            vis_attention_mask = attention_mask.new_ones(B, V_L)
        encoder_attention_mask = torch.cat([attention_mask, vis_attention_mask], dim=1) #[bs, V_L]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,

            encoder_hidden_states=hidden_states,
            encoder_attention_mask=encoder_attention_mask,

            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # print('decoder_outputs')
        # print(decoder_outputs)

        sequence_output = decoder_outputs[0]

        assert self.config.tie_word_embeddings is True

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        if return_hidden_state:
            return sequence_output

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss(ignore_index=-100)
            # loss = loss_fct(
            #     lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

            if reduce_loss:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
            loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)),
                labels.view(-1))

            # print('loss')
            # print(loss)

        # if not return_dict:
        #     output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        #     return ((loss,) + output) if loss is not None else output

        return VLSeq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            # decoder_attentions=decoder_outputs.attentions,
            # encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=encoder_attention_mask,
            # encoder_attentions=encoder_outputs.attentions,
            # vis_encoder_last_hidden_state=vis_encoder_outputs.last_hidden_state,
            # vis_encoder_hidden_states=vis_encoder_outputs.hidden_states,
            # vis_encoder_attentions=vis_encoder_outputs.attentions,
            # cross_encoder_outputs=cross_encoder_outputs
            loss_memory_Q = None,
            loss_memory_V = None,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None,
        encoder_outputs=None,
        **kwargs):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        output = {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

        if 'vis_attention_mask' in kwargs:
            output['vis_attention_mask'] = kwargs['vis_attention_mask']

        return output

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1,
                                                                expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(
                0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(
                0, expanded_return_idx)

        if model_kwargs.get("vis_attention_mask", None) is not None:
            model_kwargs['vis_attention_mask'] = model_kwargs['vis_attention_mask'].index_select(
                0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx
            )
            model_kwargs["encoder_outputs"] = encoder_outputs

        return input_ids, model_kwargs


@dataclass
class VLSeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Languaged modeling loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
            :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see ``past_key_values`` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

    vis_encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    vis_encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vis_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

    # new_add
    encoder_attention_mask: Optional[Tuple[torch.FloatTensor]] = None
    loss_memory_Q: torch.FloatTensor = None
    loss_memory_V: torch.FloatTensor = None

    # cross_encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None
