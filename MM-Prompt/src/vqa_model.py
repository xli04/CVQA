from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from modeling import VLT5

class VLT5VQA(VLT5):
    """
    T5-based Vision-Language model for Visual Question Answering tasks.
    Extends VLT5 with specific functionality for VQA including cross-modal prompting.
    """
    def __init__(self, config, num_answers=None, label2ans=None):
        """
        Initialize the VQA model.
        
        Args:
            config: Model configuration
            num_answers: Number of possible answers in classification setup
            label2ans: Mapping from label indices to answer strings
        """
        super().__init__(config)

        self.num_answers = num_answers
        self.label2ans = label2ans
        self.bce_loss = nn.BCEWithLogitsLoss()

    def train_step(self, batch, current_task_id, proto_alpha, proto_beta, mem_num_Q=0, total_num_Q=1000, memory=False):
        """
        Perform a single training step.
        
        Args:
            batch: Batch of training data
            current_task_id: ID of the current task being trained
            proto_alpha: Weight for prototype update
            proto_beta: Weight for prototype recovery
            mem_num_Q: Number of questions in memory
            total_num_Q: Total number of questions
            memory: Flag indicating if using memory replay
            
        Returns:
            Dictionary containing loss and intermediate states
        """
        # Move all inputs to the device
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)  # Visual features [batch_size, 36, 2048]
        input_ids = batch['input_ids'].to(device)  # Input token IDs [batch_size, 20]
        vis_pos = batch['boxes'].to(device)        # Visual position info [batch_size, 36, 4]
        lm_labels = batch["target_ids"].to(device) # Target IDs [batch_size, 5]

        # Category and question labels for prompt learning
        cate_labels = batch['cate_labels'].to(device)
        ques_labels = batch['ques_labels'].to(device)

        # Forward pass with all parameters for cross-modal prompt learning
        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            labels=lm_labels,
            cate_labels=cate_labels,
            ques_labels=ques_labels,
            proto_update=True,
            memory=memory,
            current_task_id=current_task_id,
            mem_num_Q=mem_num_Q,
            total_num_Q=total_num_Q,
            proto_alpha=proto_alpha,
            proto_beta=proto_beta,
            return_dict=True
        )
        assert 'loss' in output

        # Apply mask to exclude padding tokens from loss calculation
        lm_mask = (lm_labels != -100).float()
        B, L = lm_labels.size()

        loss = output['loss']  # Raw loss [B*L]

        # Reshape and apply mask to get per-sample loss
        loss = loss.view(B, L) * lm_mask
        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # Average over sequence length
        loss = loss * batch['scores'].to(device=device)  # Weight loss by sample scores
        loss = loss.mean()  # Get final scalar loss
        
        # Prepare result dictionary
        result = {
            'loss': loss
        }
        # Include encoder states for further processing
        result['encoder_hidden_states'] = output['encoder_hidden_states']
        result['BL'] = (B, L)
        result['encoder_attention_mask'] = output['encoder_attention_mask']
        
        # Include memory-specific losses if present
        if 'loss_memory' in output:
            result['loss_memory'] = output['loss_memory']
        if 'loss_memory_new' in output:
            result['loss_memory_new'] = output['loss_memory_new']
        return result

    def update_prompt_fusion_attention(self):
        """
        Update attention weights for prompt fusion.
        This method triggers weight update in the encoder.
        """
        self.encoder.update_attention_weight

    @torch.no_grad()
    def test_step(self, batch, **kwargs):
        """
        Perform a single test/evaluation step.
        
        Args:
            batch: Batch of test data
            **kwargs: Additional arguments for generation
            
        Returns:
            Dictionary containing generated answers and attention matrices
        """
        self.eval()
        device = next(self.parameters()).device
    
        # Move inputs to device
        vis_feats = batch['vis_feats'].to(device)
        vis_pos = batch['boxes'].to(device)
        input_ids = batch['input_ids'].to(device)
    
        # Generate answers with cross-attention tracking
        generate_output = self.generate_with_cross_attention(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            **kwargs
        )
    
        # Extract outputs from generation
        generated_ids = generate_output["sequences"]
        generated_sents = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        cross_attentions = generate_output["cross_attentions"]
        decoder_attentions = generate_output["decoder_attentions"]
    
        # Return results dictionary with predictions and attention maps
        return {
            'token_ids': generated_ids,
            'pred_ans': generated_sents,
            'cross_attention': cross_attentions,
            'input_ids': input_ids,
            'decoder_attentions': decoder_attentions
        }





