"""
Custom tokenizer implementations for the MM-Prompt CVQA model.

This module extends the T5 tokenizers from the transformers library to handle
visual tokens needed for cross-modal processing. It adds <vis_extra_id_X> tokens 
alongside the original <extra_id_X> tokens from T5.
"""

from transformers import T5Tokenizer, T5TokenizerFast, PreTrainedTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizerBase
import re
import sentencepiece as spm

# The special tokens of T5Tokenizer is hard-coded with <extra_id_{}>
# I create another class VLT5Tokenizer extending it to add <vis_extra_id_{}>

class VLT5Tokenizer(T5Tokenizer):
    """
    Extension of T5Tokenizer that adds visual tokens for cross-modal processing.
    
    Adds <vis_extra_id_X> tokens for representing visual features, similar to 
    how T5 uses <extra_id_X> tokens for text.
    """

    # vocab_files_names = VOCAB_FILES_NAMES
    # pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # model_input_names = ["attention_mask"]

    def __init__(
        self,
        vocab_file,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        vis_extra_ids=100,
        additional_special_tokens=None,
        **kwargs
    ):
        """
        Initialize the VLT5 tokenizer.
        
        Args:
            vocab_file: Path to SentencePiece vocabulary file
            eos_token: End of sequence token
            unk_token: Unknown token
            pad_token: Padding token
            extra_ids: Number of extra_id tokens to add (<extra_id_0>, ..., <extra_id_99>)
            vis_extra_ids: Number of visual tokens to add (<vis_extra_id_0>, ..., <vis_extra_id_99>)
            additional_special_tokens: Additional special tokens beyond the default ones
            **kwargs: Additional keyword arguments passed to the parent class
        """
        # Add extra_ids to the special token list
        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = ["<extra_id_{}>".format(i) for i in range(extra_ids)]
        elif extra_ids > 0 and additional_special_tokens is not None:
            # Check that we have the right number of extra_id special tokens
            extra_tokens = len(set(filter(lambda x: bool("extra_id" in x), additional_special_tokens)))
            if extra_tokens != extra_ids:
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are provided to T5Tokenizer. "
                    "In this case the additional_special_tokens must include the extra_ids tokens"
                )

        # Add visual tokens to special tokens list
        if vis_extra_ids > 0:
            additional_special_tokens.extend(["<vis_extra_id_{}>".format(i) for i in range(vis_extra_ids)])

        self.vocab_file = vocab_file
        self._extra_ids = extra_ids
        self._vis_extra_ids = vis_extra_ids

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

        PreTrainedTokenizer.__init__(
            self,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

    @property
    def vocab_size(self):
        """
        Get the size of the vocabulary.
        
        Returns:
            Integer: Size of SentencePiece vocabulary + extra tokens + visual tokens
        """
        return self.sp_model.get_piece_size() + self._extra_ids + self._vis_extra_ids

    def get_vocab(self):
        """
        Get the vocabulary as a dictionary.
        
        Returns:
            Dictionary mapping tokens to their ids
        """
        vocab = {self.convert_ids_to_tokens(
            i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _convert_token_to_id(self, token):
        """
        Convert a token string to its vocabulary id.
        
        Args:
            token: Token string to convert
            
        Returns:
            Integer: ID of the token in the vocabulary
            
        Note:
            Handles both regular tokens and special <extra_id_X> and <vis_extra_id_X> tokens.
        """
        if token.startswith("<extra_id_"):
            match = re.match(r"<extra_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._vis_extra_ids
        elif token.startswith("<vis_extra_id_"):
            match = re.match(r"<vis_extra_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """
        Convert a vocabulary id to its token string.
        
        Args:
            index: ID to convert
            
        Returns:
            String: Token corresponding to the ID
            
        Note:
            Handles regular vocabulary IDs and special token IDs (<extra_id_X> and <vis_extra_id_X>).
        """
        if index < self.sp_model.get_piece_size():
            token = self.sp_model.IdToPiece(index)
        else:
            if index > self.sp_model.get_piece_size() + self._extra_ids - 1:
                token = "<vis_extra_id_{}>".format(self.vocab_size - 1 - index)
            else:
                token = "<extra_id_{}>".format(self.vocab_size - self._vis_extra_ids - 1 - index)
        return token


# Below are for Rust-based Fast Tokenizer

from transformers.convert_slow_tokenizer import SpmConverter
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from typing import Any, Dict, List, Optional, Tuple, Union


class VLT5Converter(SpmConverter):
    """
    Converter class to transform the slow VLT5Tokenizer into a fast Rust-based tokenizer.
    Extends the SpmConverter to handle the additional visual tokens.
    """
    
    def vocab(self, proto):
        """
        Create the vocabulary for the fast tokenizer.
        
        Args:
            proto: SentencePiece model proto
            
        Returns:
            List of (token, score) tuples including special tokens
        """
        vocab = [(piece.piece, piece.score) for piece in proto.pieces]
        num_extra_ids = self.original_tokenizer._extra_ids
        vocab += [("<extra_id_{}>".format(i), 0.0)
                  for i in range(num_extra_ids - 1, -1, -1)]

        num_vis_extra_ids = self.original_tokenizer._vis_extra_ids
        vocab += [("<vis_extra_id_{}>".format(i), 0.0)
                  for i in range(num_vis_extra_ids - 1, -1, -1)]

        return vocab

    def post_processor(self):
        """
        Create a post-processor for the fast tokenizer.
        
        Returns:
            TemplateProcessing: Post-processor for single and paired sequences
        """
        return processors.TemplateProcessing(
            single=["$A", "</s>"],
            pair=["$A", "</s>", "$B", "</s>"],
            special_tokens=[
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


def convert_slow_vlt5tokenizer(vlt5tokenizer):
    """
    Convert a slow VLT5Tokenizer to a fast Rust-based implementation.
    
    Args:
        vlt5tokenizer: Instance of VLT5Tokenizer
        
    Returns:
        Fast tokenizer implementation
    """
    return VLT5Converter(vlt5tokenizer).converted()


class VLT5TokenizerFast(T5TokenizerFast):
    """
    Fast version of VLT5Tokenizer implemented using the Rust-based tokenizers library.
    Inherits from T5TokenizerFast and adds support for visual tokens.
    """
    
    # vocab_files_names = VOCAB_FILES_NAMES
    # pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # model_input_names = ["attention_mask"]
    slow_tokenizer_class = VLT5Tokenizer
    prefix_tokens: List[int] = []

    def __init__(
        self,
        vocab_file,
        tokenizer_file=None,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        vis_extra_ids=100,
        additional_special_tokens=None,
        **kwargs
    ):
        """
        Initialize the fast VLT5 tokenizer.
        
        Args:
            vocab_file: Path to SentencePiece vocabulary file
            tokenizer_file: Path to pre-serialized tokenizer file (optional)
            eos_token: End of sequence token
            unk_token: Unknown token
            pad_token: Padding token
            extra_ids: Number of extra_id tokens to add
            vis_extra_ids: Number of visual tokens to add
            additional_special_tokens: Additional special tokens beyond the default ones
            **kwargs: Additional keyword arguments passed to the parent class
        """
        # Add extra_ids to the special token list
        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = ["<extra_id_{}>".format(i) for i in range(extra_ids)]
        elif extra_ids > 0 and additional_special_tokens is not None:
            # Check that we have the right number of extra_id special tokens
            extra_tokens = len(set(filter(lambda x: bool("extra_id" in x), additional_special_tokens)))
            if extra_tokens != extra_ids:
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are provided to T5Tokenizer. "
                    "In this case the additional_special_tokens must include the extra_ids tokens"
                )

        # Add visual tokens to special tokens list
        if vis_extra_ids > 0:
            additional_special_tokens.extend(["<vis_extra_id_{}>".format(i) for i in range(vis_extra_ids)])

        # Create the slow tokenizer and convert it to fast
        slow_tokenizer = self.slow_tokenizer_class(
            vocab_file,
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            vis_extra_ids=vis_extra_ids,
            # additional_special_tokens=additional_special_tokens,
            **kwargs
        )
        fast_tokenizer = convert_slow_vlt5tokenizer(slow_tokenizer)
        self._tokenizer = fast_tokenizer

        PreTrainedTokenizerBase.__init__(
            self,
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            vis_extra_ids=vis_extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.vocab_file = vocab_file
        self._extra_ids = extra_ids
        self._vis_extra_ids = vis_extra_ids