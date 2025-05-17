"""
Download and cache T5 and BART backbone models for the MM-Prompt VQA project.
These models serve as the base for the multimodal transformers used in the project.
"""

from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer

if __name__ == '__main__':
    print('Downloading checkpoints if not cached')
    
    # Download T5-base model and tokenizer
    print('T5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    
    # Download BART-base model and tokenizer
    print('BART-base')
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    
    print('Done!')

