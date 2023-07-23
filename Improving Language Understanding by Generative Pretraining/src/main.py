import os
import sys
from time import time_ns

import numpy as np
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn


from tokenizers import Tokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader


from model import GPT
from state import load_checkpoint, save_checkpoint
from objects import ConstantLenghtDataset, TokenizerWrapper
from utils import (setup, cleanup, train, test, prepare_data)



tokenizer = Tokenizer.from_file("data/tokenizer.json")
model_config = dict(
    num_layers=12,
    num_heads=12,
    hidden_dim=768,
    ffc_hidden_dim=3072,
    max_seq_len=512,
    vocab_size=tokenizer.get_vocab_size()
)

tokenizerwrapped = TokenizerWrapper(tokenizer, pad_seq_len=model_config['max_seq_len'])





train_texts, test_texts = prepare_data()

train_dataset = ConstantLenghtDataset(train_texts, tokenizer, length=model_config['max_seq_len'])
test_dataset = ConstantLenghtDataset(test_texts, tokenizer, length=model_config['max_seq_len'])

NUM_EPOCHS = 200
WORLD_SIZE = 2
BATCH_SIZE = 32
LR = 2e-5
SAVE_INTERVAL=10
SAVE_PATH = "checkpoints/model.pt"




def demo_basic(rank, world_size):
    print(f"Running basic GPT-1 traning on device: {rank}.")
    setup(rank, world_size)
    
    torch.cuda.set_device(rank)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True, sampler=train_sampler)
    
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)

    
    
    model = GPT(**model_config).cuda(rank)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, nesterov=True)
    model = DDP(model, device_ids=[rank])
    state = load_checkpoint(SAVE_PATH, rank, model, optimizer)

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    cudnn.benchmark = True
    
    for epoch in range(NUM_EPOCHS):
        t0 = time_ns()

        train(epoch, model, optimizer, train_dataloader, tokenizerwrapped, scaler)

        t1 = time_ns()
        delta = (t1 - t0) / (10 ** 9)
        print(f"Device {rank} - Train time: {delta} sec")
        
        if rank == 0:
            loss = test(epoch, model, test_dataloader, tokenizerwrapped)
            print(f"Loss: {loss}%")

        if epoch in [int(NUM_EPOCHS * 0.5), int(NUM_EPOCHS * 0.75)]:
            optimizer.param_groups[0]['lr'] /= 10.
            
        if epoch % SAVE_INTERVAL == 0 and rank == 0:
            save_checkpoint(state, SAVE_PATH)

    state.epoch = epoch
    cleanup()
    
    
def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    
    
if __name__ == '__main__':
    run_demo(demo_basic,
             2)
