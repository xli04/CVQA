"""
Parameter handling module for MM-Prompt CVQA project.
Defines command-line arguments, config settings, and utility functions 
for model configuration and hyperparameter management.
"""

import argparse
import random

import numpy as np
import torch

import pprint
import yaml


def str2bool(v):
    """
    Convert string representation of boolean to actual boolean value.
    
    Args:
        v: String input ('yes', 'true', 't', 'y', '1', 'no', 'false', 'f', 'n', '0')
        
    Returns:
        Boolean value
        
    Raises:
        ArgumentTypeError: If input string doesn't represent a boolean
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def is_interactive():
    """
    Check if code is running in interactive environment (e.g., Jupyter notebook).
    
    Returns:
        Boolean indicating if running in interactive mode
    """
    import __main__ as main
    return not hasattr(main, '__file__')


def get_optimizer(optim, verbose=False):
    """
    Get PyTorch optimizer based on string identifier.
    
    Args:
        optim: String name of optimizer ('rms', 'adam', 'adamw', 'adamax', 'sgd')
        verbose: Whether to print optimizer selection
        
    Returns:
        PyTorch optimizer class
        
    Raises:
        AssertionError: If optimizer name is not recognized
    """
    # Bind the optimizer
    if optim == 'rms':
        if verbose:
            print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        if verbose:
            print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamw':
        if verbose:
            print("Optimizer: Using AdamW")
        # optimizer = torch.optim.AdamW
        optimizer = 'adamw'
    elif optim == 'adamax':
        if verbose:
            print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        if verbose:
            print("Optimizer: SGD")
        optimizer = torch.optim.SGD
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args(parse=True, **optional_kwargs):
    """
    Parse command-line arguments for the MM-Prompt CVQA model.
    
    Args:
        parse: Whether to actually parse args or return parser
        **optional_kwargs: Additional arguments to override defaults
        
    Returns:
        Parsed arguments or argument parser
    """
    parser = argparse.ArgumentParser()

    # Random seed settings
    parser.add_argument('--ifseed', action='store_true', help='Whether to use fixed seed')
    parser.add_argument('--seed', type=int, default=66666, help='Random seed for reproducibility')

    # Data configuration
    parser.add_argument("--train", default='train', help='Train dataset name')
    parser.add_argument("--valid", default='valid', help='Validation dataset name')
    parser.add_argument("--test", default=None, help='Test dataset name')
    parser.add_argument('--test_only', action='store_true', help='Run in evaluation-only mode')

    parser.add_argument('--submit', action='store_true', help='Generate submission for competition')

    # Dataset size limits for quick experiments
    parser.add_argument('--train_topk', type=int, default=-1, help='Use only top-k train examples (-1 for all)')
    parser.add_argument('--valid_topk', type=int, default=-1, help='Use only top-k val examples (-1 for all)')

    # Checkpoint and output settings
    parser.add_argument('--output', type=str, default='snap/test', help='Output directory for checkpoints')
    parser.add_argument('--load', type=str, default=None, help='Path to load pre-trained model')
    parser.add_argument('--from_scratch', action='store_true', help='Train model from scratch')

    # Hardware settings
    parser.add_argument("--multiGPU", action='store_const', default=False, const=True, help='Use multiple GPUs')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    parser.add_argument("--distributed", action='store_true', help='Use distributed training')
    parser.add_argument("--num_workers", default=0, type=int, help='Number of workers for data loading')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')

    # Model architecture
    parser.add_argument('--backbone', type=str, default='t5-base', help='Base model architecture')
    parser.add_argument('--tokenizer', type=str, default=None, help='Tokenizer type (defaults to backbone if not set)')

    # Feature dimensions
    parser.add_argument('--feat_dim', type=float, default=2048, help='Visual feature dimensionality')
    parser.add_argument('--pos_dim', type=float, default=4, help='Position feature dimensionality')

    # Vision-language settings
    parser.add_argument('--use_vision', default=True, type=str2bool, help='Whether to use visual features')
    parser.add_argument('--use_vis_order_embedding', default=True, type=str2bool, help='Use vision order embeddings')
    parser.add_argument('--use_vis_layer_norm', default=True, type=str2bool, help='Apply layer norm to visual features')
    parser.add_argument('--individual_vis_layer_norm', default=True, type=str2bool, help='Individual layer norm per visual feature')
    parser.add_argument('--share_vis_lang_layer_norm', action='store_true', help='Share layer norm between vision and language')

    # Input limits
    parser.add_argument('--n_boxes', type=int, default=36, help='Number of visual features')
    parser.add_argument('--max_n_boxes', type=int, default=36, help='Maximum number of visual features')
    parser.add_argument('--max_text_length', type=int, default=20, help='Maximum text sequence length')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=256, help='Training batch size')
    parser.add_argument('--valid_batch_size', type=int, default=None, help='Validation batch size (defaults to batch_size)')
    parser.add_argument('--optim', default='adamw', help='Optimizer type')
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help='Warmup proportion of training')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay coefficient')
    parser.add_argument('--clip_grad_norm', type=float, default=-1.0, help='Gradient clipping norm (-1 for no clipping)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Steps before param update')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--adam_eps', type=float, default=1e-6, help='Adam epsilon parameter')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='Adam beta1 parameter')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help='Adam beta2 parameter')
    parser.add_argument('--epochs', type=int, default=12, help='Number of training epochs')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')

    parser.add_argument("--losses", default='lm,obj,attr,feat', type=str, help='Comma-separated loss types')

    parser.add_argument('--log_train_accuracy', action='store_true', help='Log training accuracy')

    # Masking parameters
    parser.add_argument('--n_ground', type=int, default=1, help='Number of grounding iterations')
    parser.add_argument("--wordMaskRate", dest='word_mask_rate', default=0.15, type=float, help='Word mask rate')
    parser.add_argument("--objMaskRate", dest='obj_mask_rate',default=0.15, type=float, help='Object mask rate')

    # Inference settings
    parser.add_argument('--num_beams', type=int, default=1, help='Beam size for generation')
    parser.add_argument('--gen_max_length', type=int, default=20, help='Maximum generation length')

    # Dataset filters
    parser.add_argument('--caption_only', action='store_true', help='Use only caption data')
    parser.add_argument('--coco_only', action='store_true', help='Use only COCO data')
    parser.add_argument('--caption_cocoonly', default=True, type=str2bool, help='Use only COCO captions')

    parser.add_argument('--do_lower_case', action='store_true', help='Lowercase all text')
    parser.add_argument('--oscar_tags', action='store_true', help='Use OSCAR tags')

    parser.add_argument('--prefix', type=str, default=None, help='Prefix for prompts')

    # Pretraining options
    parser.add_argument('--ground_upsample', type=int, default=1, help='Grounding upsample ratio')
    parser.add_argument('--ground_weight', type=int, default=1, help='Grounding loss weight')
    parser.add_argument('--itm_cocoonly', default=True, type=str2bool, help='ITM COCO only')
    parser.add_argument('--single_vqa_prefix', action='store_true', help='Use single VQA prompt prefix')

    # COCO Caption
    parser.add_argument('--no_prefix', action='store_true', help='Do not use prompt prefix')

    # VQA specific
    parser.add_argument("--raw_label", action='store_true', help='Use raw label text')
    parser.add_argument("--answer_normalize", action='store_true', help='Normalize answer distributions')
    parser.add_argument("--classifier", action='store_true', help='Use classifier for answers')
    parser.add_argument("--test_answerable", action='store_true', help='Test only answerable questions')

    # RefCOCOg
    parser.add_argument('--RefCOCO_GT', action='store_true', help='Use RefCOCO ground truth')
    parser.add_argument('--RefCOCO_BUTD', action='store_true', help='Use RefCOCO BUTD features')
    parser.add_argument("--shuffle_boxes", action='store_true', help='Shuffle boxes for robustness')
    parser.add_argument('--vis_pointer', type=str2bool, default=False, help='Use visual pointer')

    # Multitask
    parser.add_argument("--multitask_sampling", type=str, default='roundrobin', help='Multitask sampling strategy')
    parser.add_argument("--tasks", type=str, default='', help='Comma-separated task list')

    # Misc
    parser.add_argument('--comment', type=str, default='', help='Comment for experiment tracking')
    parser.add_argument("--dry", action='store_true', help='Dry run (no actual training)')

    # Continual learning / Memory
    parser.add_argument("--memory", type=lambda x: x.lower() == 'true', default=False, help="Enable or disable memory buffer")
    parser.add_argument("--m_size", type=int, default=1000, help="Memory buffer size")

    parser.add_argument("--checkpoint", type=str, default="None", help="Checkpoint task name")
    parser.add_argument("--Q", type=str, default="All_Q_v4", help="Question set version")
    parser.add_argument("--pull_constraint_coeff", type=float, default=1.0, help="Prompt pull constraint coefficient")

    parser.add_argument("--freeze", action='store_true', default=True, help="Freeze backbone parameters")

    # Cross-modal prompt parameters
    parser.add_argument("--lambda_Q", type=float, default=0.01, help="Question prompt loss weight")
    parser.add_argument("--lambda_V", type=float, default=0.1, help="Visual prompt loss weight")
    parser.add_argument("--lambda_Q_new", type=float, default=0, help="New question prompt loss weight")
    parser.add_argument("--lambda_V_new", type=float, default=0, help="New visual prompt loss weight")

    parser.add_argument("--comp_cate", type=str, default='G3', help="Composition category for testing")
    parser.add_argument("--ewc_loss_weight", type=float, default=100.0, help="EWC loss weight for regularization")
    parser.add_argument("--lambda_neighbor", type=float, default=10, help="Neighbor consistency loss weight")
    parser.add_argument("--reg_lambda", type=float, default=10000, help="Regularization lambda")
    parser.add_argument("--now_train", action='store_true', help="Start training immediately")

    # Prototype parameters
    parser.add_argument("--proto_alpha", type=float, default=0.5, help="Prototype alpha weight")
    parser.add_argument("--proto_beta", type=float, default=0.3, help="Prototype beta weight")

    # Parse the arguments.
    if parse:
        args = parser.parse_args()
    # For interative engironmnet (ex. jupyter)
    else:
        args = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(args)
    kwargs.update(optional_kwargs)

    args = Config(**kwargs)

    # Bind optimizer class.
    verbose = False
    args.optimizer = get_optimizer(args.optim, verbose=verbose)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.load(f)

        return Config(**kwargs)


if __name__ == '__main__':
    args = parse_args(True)