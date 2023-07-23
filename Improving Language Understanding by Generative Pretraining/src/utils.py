import os

import torch
import torch.distributed as dist
import torch.nn as nn

from tqdm import tqdm

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8080'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dist.barrier()
    
def cleanup():
    dist.destroy_process_group()
    
    
def train(epoch, model, optimizer, train_dataloader, tokenizerwrapped, scaler):
    model.train()
    training_loss = 0
    tokenizerwrapped.tokenizer.enable_padding(pad_id=2, pad_token="<pad>", length=tokenizerwrapped.pad_seq_len)
    for batch_num, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        
        inputs = tokenizerwrapped(batch)
        labels = inputs['input_ids'].cuda(non_blocking=True)
        attn_mask = inputs['attn_mask'].cuda(non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=True):
            logits = model(labels, attn_mask)
        
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))        
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        training_loss += loss.item()

    training_loss /= batch_num
    print(f"Epoch: {epoch}, Training loss: {training_loss}")

def test(epoch, model, test_dataloader, tokenizerwrapped):
    model.eval()
    test_loss = 0
    tokenizerwrapped.tokenizer.enable_padding(pad_id=2, pad_token="<pad>", length=tokenizerwrapped.pad_seq_len)
    with tqdm(total=len(test_dataloader.dataset)) as progress_bar:
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                inputs = tokenizerwrapped(batch)
                labels = inputs['input_ids'].cuda(non_blocking=True)
                attn_mask = inputs['attn_mask'].cuda(non_blocking=True)

                logits = model(labels, attn_mask)
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
        
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                test_loss += loss.item()
                progress_bar.update(labels.size(0))
            
            test_loss /= batch_idx
    
    return test_loss


def prepare_data():
    with open('data/input.txt', 'r') as f:
        input_text = f.readlines()
    input_text = [i for i in input_text if i!='\n']
    train_size = 0.9
    train_ids = int(len(input_text) * train_size)
    train_data = input_text[: train_ids]
    test_data = input_text[train_ids:]
    
    return train_data, test_data
