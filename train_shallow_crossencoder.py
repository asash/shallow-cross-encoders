import os 
from collections import defaultdict, deque
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime
import subprocess
import shlex

from torch.utils.data import DataLoader

from argparse import ArgumentParser
import ir_datasets
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import socket
from subprocess import check_call


parser = ArgumentParser()
parser.add_argument("-t", type=float, default=0.75)
parser.add_argument("--mask", type=float, default=0)
parser.add_argument("--negs", type=int, default=16)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--val-batch-size", type=int, default=32)
parser.add_argument("--backbone-model", type=str, default="prajjwal1/bert-tiny")
parser.add_argument("--max-epochs", type=int, default=1000000)
parser.add_argument("--val-negs", type=int, default=999)
parser.add_argument("--max-batches-per-epoch", type=int, default=600)
parser.add_argument("--val-docs", type=int, default=200)
parser.add_argument("--early-stopping", type=int, default=200)
parser.add_argument("--run-id", type=str, default=None)
parser.add_argument("--output-dir", type=str, default=None)
parser.add_argument("--tensorboard-dir", type=str, default=None)


args = parser.parse_args()

run_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if args.run_id is None:
    run_id  = f"crossencoder-{args.backbone_model.replace('/', '-')}-gbce-{args.negs}-{args.t}-{args.batch_size}-{run_time}-mask-{args.mask:.2f}"
else:
    run_id = args.run_id

if args.output_dir is not None:
    output_dir = args.output_dir
else:
    output_dir = "output/" + run_id

if args.tensorboard_dir is not None:
    tensorboard_dir = args.tensorboard_dir
else:
    tensorboard_dir = "output/" + run_id + "/tensorboard"
    


check_call(["mkdir", "-p", output_dir])
check_call(["mkdir", "-p", tensorboard_dir])
cmd = "tensorboard --logdir " + os.path.abspath(tensorboard_dir) + " --host 0.0.0.0 --port 26006"

summary_writer = SummaryWriter(tensorboard_dir)

alpha = args.negs / 1000.0 
beta = alpha * (args.t * (1 - 1/alpha) + 1/alpha)
print("gbce beta: ", beta)

tokenizer = AutoTokenizer.from_pretrained(args.backbone_model)


rng = np.random.RandomState(31339)

dataset = ir_datasets.load('msmarco-passage/train')
all_queries = {int(q.query_id): q.text.strip() for q in dataset.queries}

#bm25_preindexed = np.memmap("datasets/qid_rel_999_bm25negs_502574x1001.mmap", shape=(502574, 1001), dtype=np.int32)
bm25_preindexed = np.memmap("datasets/qid_rel_999_bm25negs_436299x1001.mmap", shape=(436299, 1001), dtype=np.int32)
pass

#all_qrels = [(q.query_id, q.doc_id) for q in dataset.qrels]
all_row_ids = np.arange(len(bm25_preindexed))
train_rows, val_rows = train_test_split(all_row_ids, test_size=args.val_docs, random_state=31339)
pass

def batch_tokeniser(rows_ids, max_negatives, mask_percent = 0): # batch_size . (max_negativese + 1) x max_len
    batch_texts = []
    for row_id  in rows_ids:
        row = bm25_preindexed[row_id]
        qid = row[0]
        positive = row[1:2]
        negatives = row[2:]
        if max_negatives < len(negatives):
            negatives = rng.choice(negatives, max_negatives, replace=False)
        query_docs = np.concatenate([positive, negatives])
        for docno in query_docs:
            batch_texts.append([all_queries[qid], dataset.docs.lookup(str(docno)).text])
    batch = tokenizer.batch_encode_plus(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512, return_special_tokens_mask=True)
    special_tokens_mask = batch.pop('special_tokens_mask')
    
    if mask_percent > 0:
        assert mask_percent < 1
        # see https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L782C9-L813
        # TODO how to separate query from doc?
        probability_matrix = torch.full(batch['input_ids'].shape, mask_percent)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        batch['input_ids'][masked_indices] = tokenizer.mask_token_id # TODO: check this behaves as expected
      
    return batch


val_dataloader = DataLoader(val_rows, batch_size=args.val_batch_size, #num_workers=1,
                            shuffle=False,
                            collate_fn=lambda x : batch_tokeniser(x, args.val_negs))

val_batches = defaultdict(list) 
print("loading val data...")
for batch in tqdm(val_dataloader):
    for key in batch:
        cur = 0
        while cur < batch[key].shape[0]:
            val_batches[key].append(batch[key][cur:cur+args.val_batch_size])
            cur += args.val_batch_size

val_batches_merged = []
for i in range(len(val_batches["input_ids"])):
    batch = {}
    for key in val_batches:
        batch[key] = val_batches[key][i] 
    val_batches_merged.append(batch)

model = AutoModelForSequenceClassification.from_pretrained(args.backbone_model, num_labels=2).cuda()
device = "cuda:0"

def validate():
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for batch in tqdm(val_batches_merged):
            batch = {k: v.to(device) for k, v in batch.items()}
            class_logits = model(**batch).logits
            logprobs = torch.nn.functional.log_softmax(class_logits, dim=1)
            all_outputs.append(logprobs[:,1])
            pass
    all_outputs = torch.cat(all_outputs).reshape(args.val_docs, args.val_negs+1)
    ranks = torch.argsort(all_outputs, dim=1, descending=True).argsort(dim=1)
    rel_rank = ranks[:,0]
    rel_dcg = 1.0 / torch.log2(rel_rank.float() + 2.0)
    return torch.mean(rel_dcg).item()


train_dataloader = DataLoader(train_rows, batch_size=args.batch_size, 
                              shuffle=True, # num_workers=1, 
                              collate_fn=lambda x: batch_tokeniser(x, args.negs, mask_percent = args.mask))

steps = 0
optimiser = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_hist = deque(maxlen=100)
best_val_ndcg = float("-inf")
best_model_checkpoint = None

evals_without_improvement = 0

tb_process = subprocess.Popen(shlex.split(cmd))
print(f"Access tensorboard monitoring at http://{socket.gethostname()}:26006")

for epoch in range(args.max_epochs):
    model.train()
    epoch_steps = min(args.max_batches_per_epoch, len(train_dataloader))
    train_iteratpr = iter(train_dataloader)
    pbar = tqdm(total=epoch_steps)
    for step_in_batch in range(epoch_steps):
        batch = next(train_iteratpr)
        batch = {k: v.to(device) for k, v in batch.items()}
        class_logits = model(**batch).logits
        num_samples = class_logits.shape[0] // (args.negs+1)
        class_logprobs = torch.nn.functional.log_softmax(class_logits, dim=1).reshape((num_samples,  args.negs+1, 2))
        mean_positive_prob = torch.mean(class_logprobs[:,0,1].exp())
        positive_logprobs = class_logprobs[:,0:1,1]*beta
        negatives_log_one_minus_probs = class_logprobs[:,1:,0]
        negative_probs = class_logprobs[:,1:,1].exp()
        false_positives_rate = torch.mean((negative_probs > 0.9).to(torch.float))
        
        mean_negative_prob = torch.mean(negative_probs)
        all_logprobs = torch.cat([positive_logprobs, negatives_log_one_minus_probs], dim=1)
        loss = -torch.mean(all_logprobs)

        loss.backward()
        total_norm = 0.0 
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
            
        optimiser.step()
        optimiser.zero_grad()
        summary_writer.add_scalar("train/loss", loss.item(), steps)
        summary_writer.add_scalar("train/mean_positive_prob", mean_positive_prob.item(), steps)
        summary_writer.add_scalar("train/mean_negative_prob", mean_negative_prob.item(), steps)
        summary_writer.add_scalar("train/grad_norm", total_norm, steps)
        summary_writer.add_scalar("train/false_positives_rate", false_positives_rate.item(), steps)
        
        loss_hist.append(loss.item())
        pbar.update(1)
        pbar.set_description(f"loss: {np.mean(loss_hist):.6f}")
        steps += 1

    val_ndcg = validate()
    summary_writer.add_scalar("val/ndcg", val_ndcg, steps)
    summary_writer.add_scalar("val/best_ndcg", best_val_ndcg, steps)

    print(f"epoch {epoch} val ndcg: {val_ndcg:.6f}")
    if val_ndcg > best_val_ndcg:
        best_val_ndcg = val_ndcg
        new_best_checkpoint = output_dir  + f"/epoch:{epoch}-val_ndcg:{val_ndcg:.6f}"
        model.save_pretrained(new_best_checkpoint)
        evals_without_improvement = 0 
        #remove old best checkpoint
        if best_model_checkpoint is not None:
            check_call(["rm", "-rf", best_model_checkpoint])
        best_model_checkpoint = new_best_checkpoint
        print(f"new best checkpoint: {new_best_checkpoint}")
    else:
        evals_without_improvement += 1
        if evals_without_improvement > args.early_stopping:
            print("early stopping at epoch ", epoch)
            print("best checkpoint: ", best_model_checkpoint)
            break
        else:
            print("evals without improvement: ", evals_without_improvement)
    summary_writer.add_scalar("val/evals_without_improvement", evals_without_improvement, steps)
    summary_writer.add_scalar("val/epochs_to_stop", args.early_stopping - evals_without_improvement, steps)
tb_process.kill()
