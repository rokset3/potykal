import torch
import torch.multiprocessing as mp

from model import GPT
from utils import setup
from state import load_checkpoint, save_checkpoint
from torch.nn.parallel import DistributedDataParallel as DDP
from tokenizers import Tokenizer
from objects import TokenizerWrapper

import os

def example(rank, world_size):

    setup(rank, world_size)
    tokenizer = Tokenizer.from_file("data/tokenizer.json")
    prefix = "<bos> A thou"


    model_config = dict(
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        ffc_hidden_dim=3072,
        max_seq_len=512,
        vocab_size=tokenizer.get_vocab_size()
    )

    parallel_model = DDP(GPT(**model_config).to(rank))
    parallel_model.load_state_dict(torch.load("checkpoints/model.pt", map_location={str(rank): 'cuda:0'})['model']
    )
    parallel_model.eval()

    tokenizerwrapped = TokenizerWrapper(tokenizer, 0)
    batch = tokenizerwrapped(prefix, batch=False)

    
    num_generations = 400
    with torch.cuda.amp.autocast():
        for i in range(num_generations):
            attn_mask = batch['attn_mask']
            curr_num_tokens = batch['input_ids'].shape[-1]
            outputs = parallel_model(batch['input_ids'].cuda(rank), attn_mask.cuda(rank))
            probs = outputs[0, -1].div(0.8).softmax(-1)
            token = torch.multinomial(probs, 1).view([])

            print(tokenizerwrapped.tokenizer.decode([token]), end=' ', flush=True)
            batch = dict(input_ids=outputs[0, -1].argmax(-1).reshape(1, 1),
                         attn_mask=torch.ones(1, curr_num_tokens+1, requires_grad=False).cuda(rank))

        
    
if __name__ == "__main__":

    world_size = 1
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)


