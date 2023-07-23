
import numpy as np
import torch

from torch.utils.data import Dataset
from tokenizers import Tokenizer
from typing import List


class ConstantLenghtDataset(Dataset):
    def __init__(self, 
                 texts: List[str],
                 tokenizer: Tokenizer,
                 length: int=512,):
        self.texts = texts
        self.length = length
        self.tokenizer = tokenizer
        self.tokenizer.no_padding()

        encoded_text = tokenizer.encode_batch(self.texts)
        tokens_num = [len(s.tokens) for s in encoded_text]
        constant_len_dataset_ids = []
        concat_sentences_ids = []
        sum=0
        
        for idx, num in enumerate(tokens_num):
            if sum > 512:
                constant_len_dataset_ids.append(concat_sentences_ids)
                concat_sentences_ids = []
                sum = 0

            concat_sentences_ids.append(idx)
            sum+=num
        
        np_text = np.array(self.texts)
        new_dataset = []
        for idxs in constant_len_dataset_ids:
            new_dataset.append(' '.join(np_text[idxs].tolist()))

        self.dataset = new_dataset

    def __len__(self,):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    
class TokenizerWrapper():
    def __init__(self,
                 tokenizer,
                 pad_seq_len=512):
        self.tokenizer = tokenizer
        self.pad_seq_len = pad_seq_len
        self.tokenizer.enable_padding(pad_id=2, pad_token="<pad>", length=pad_seq_len)
        self.vocab_size = self.tokenizer.get_vocab_size()

    def __call__(self, input_sentences: List[str], batch=True):
        output = {}
        if batch:
            encoded_input = self.tokenizer.encode_batch(input_sentences)
            ids = torch.tensor([input.ids for input in encoded_input], requires_grad=False)
            attn_masks = torch.tensor([input.attention_mask for input in encoded_input], requires_grad=False)
        else:
            encoded_input = self.tokenizer.encode(input_sentences)
            ids = torch.tensor(encoded_input.ids, requires_grad=False).unsqueeze(0)
            attn_masks = torch.tensor(encoded_input.attention_mask, requires_grad=False).unsqueeze(0)
            
        output['input_ids'] = ids
        output['attn_mask'] = attn_masks

        return output
