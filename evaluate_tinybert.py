import pyterrier as pt
pt.init()

import torch
import pandas as pd
from pyterrier_pisa import PisaIndex
from transformers import   T5ForConditionalGeneration, T5Tokenizer, GPT2LMHeadModel, GPT2Tokenizer
from ir_measures import nDCG, P, MAP, R, RR 
from matplotlib import pyplot as plt
import seaborn as sns
from cross_decoder import CrossDecoder, get_tokenizer
from prompt_utils import get_doc2query, get_document_prompt, get_query2doc, get_query_prompt
import ir_measures
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer
import numpy as np


index = PisaIndex.from_dataset('msmarco_passage')
retrieve_candidates  = sorted(list(set([round(1.5849 ** i) for i in range(16)]))) 

retrievers = {}

for k in retrieve_candidates:
    retrievers[f"bm25_{k}"] = index.bm25(num_results=k)

#warmup
for retriever in retrievers:
    retrievers[retriever].search("hello world")



from argparse import ArgumentParser

argparser = ArgumentParser()
argparser.add_argument("--dataset", default="DL2019")



args = argparser.parse_args()

models = {"tinybert_t0.75_negs128": ("models/tiny_bert_negs_t_bm25_ignorebad=2023-09-18_02-42-02/tiny_bert-t:0.75-negs:128-bs:8/epoch:992-val_ndcg:0.503679","prajjwal1/bert-tiny"),
         "tinybert_t0.75_negs1": ("models/tiny_bert_negs_t_bm25_ignorebad=2023-09-18_02-42-02/tiny_bert-t:0.75-negs:1-bs:8/epoch:1083-val_ndcg:0.490661","prajjwal1/bert-tiny"),
         "tinybert_t0.0_negs128": ("models/tiny_bert_negs_t_bm25_ignorebad=2023-09-18_02-42-02/tiny_bert-t:0-negs:128-bs:8/epoch:816-val_ndcg:0.497080","prajjwal1/bert-tiny"),
         "tinybert_t0.0_negs1": ("models/tiny_bert_negs_t_bm25_ignorebad=2023-09-18_02-42-02/tiny_bert-t:0-negs:1-bs:8/epoch:607-val_ndcg:0.478958", "prajjwal1/bert-tiny"),
         "minybert_t0.75_negs128": ("models/bert-mini-gbce-128-0.75-2-2023-09-26_10-30-04/epoch:658-val_ndcg:0.510416/","prajjwal1/bert-mini"),
         "smallbert_t0.75_negs128": ("models/bert-small-gbce-128-0.75-1-2023-09-27_10-42-49/epoch:396-val_ndcg:0.512554/", "prajjwal1/bert-small")  
         }

if args.dataset == "DL2019":
    dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')
elif args.dataset == "DL2020":
    dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2020/judged')
elif args.dataset == "dev_small":
    dataset = pt.get_dataset("irds:msmarco-passage/dev/small")
else:
    raise Exception(f"unknown dataset {args.dataset}")

Path(f"./{args.dataset}").mkdir(exist_ok=True)


class ModelsCache(object):
    def __init__(self, device="cuda:0"):
        self.models = {}
        self.t5models = {}
        self.device = device

    def get_model(self, variant, checkpoint):
        if (variant, checkpoint) not in self.models:
            if checkpoint is not None:
                model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(self.device)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(variant).to(self.device)
            model.eval()

            self.models[(variant, checkpoint)] = model
        return self.models[(variant, checkpoint)]

    def get_t5_model(self, t5_model):
        if t5_model not in self.t5models:
            model = T5ForConditionalGeneration.from_pretrained(t5_model).to(self.device)
            model.eval()

            self.t5models[t5_model] = model
        return self.t5models[t5_model]



models_cache = ModelsCache()

class TinyBertPieline(object):
    def __init__(self, retrieve_pipeline, model_variant, checkpoint=None, batch_size = 2):
        self.batch_size = batch_size
        self.checkpoint = checkpoint
        self.model = None
        self.tokenizer = None
        self.model_variant = model_variant
        self.retrieve_pipeline = retrieve_pipeline
        self.ensure_model()
    
    def ensure_model(self): #lazy initialisation, as pyterier can re-use previous results
        if self.model is None:
            self.model = models_cache.get_model(self.model_variant, self.checkpoint) 
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_variant)

    def crossencoder_apply(self, df : pd.DataFrame):
        input_texts = [ [q, d]for (q, d) in  zip(df['query'].values, df['text'].values)]
        result = []
        with torch.no_grad():
            bs = 8 
            cur = 0
            while cur < len(input_texts):
                batch_model_inputs = self.tokenizer.batch_encode_plus(input_texts[cur:cur+bs], return_tensors='pt', padding=True, truncation=True, max_length=512).to("cuda:0")
                logits = self.model(**batch_model_inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                result += list(probs[:,1].detach().cpu().numpy()) 
                cur += bs 
        return result

    def build(self):
        cross_encT = pt.apply.doc_score(self.crossencoder_apply, batch_size=self.batch_size)
        return self.retrieve_pipeline >> pt.text.get_text(dataset, 'text') >> cross_encT


def get_monot5_pipeline(monot5model, retriever_pipeline):
    def get_monot5_prompt(query, doc):
        prompt =  f"Query: {query} Document: {doc} Relevant:"
        return prompt

    tokenizer = T5Tokenizer.from_pretrained(monot5model)
    model = models_cache.get_t5_model(monot5model)
    RELEVANT = 'true'
    IRRELEVANT = 'false'
    OUTPUTS=[RELEVANT, IRRELEVANT]
    OUT_IDS = [tokenizer(t)['input_ids'][0] for t in OUTPUTS]
 
    def _monot5_apply(df : pd.DataFrame):
        prompts = []
        for q, t in zip(df['query'].values, df['text'].values):
            prompts.append(get_monot5_prompt(q, t))
        cur  = 0
        bs = 64 
        probs = []
        with torch.no_grad():
            while cur < len(prompts):
                tensors = tokenizer.batch_encode_plus(prompts[cur:cur+bs], return_tensors = "pt", padding="longest").to("cuda:0")
                model_out = model(input_ids=tensors.input_ids, attention_mask = tensors.attention_mask, decoder_input_ids=torch.full_like(tensors.input_ids[:,:1], model.config.decoder_start_token_id)).logits
                rel_irel_logits =  torch.squeeze(model_out[:, :, OUT_IDS], 1)
                batch_probs = torch.softmax(rel_irel_logits, dim=1)[:,0].detach().cpu().numpy()
                probs += list(batch_probs)
                cur += bs
        return probs

    monot5_applyT = pt.apply.doc_score(_monot5_apply, batch_size=1)
    return retriever_pipeline >> pt.text.get_text(dataset, 'text') >> monot5_applyT


pipelines_kv = {}

for retriever in retrievers:
    pipelines_kv["monot5_10k_" + retriever] = get_monot5_pipeline("castorini/monot5-base-msmarco-10k", retrievers[retriever])
    pipelines_kv["monoBERTLarge" + retriever] = TinyBertPieline(retrievers[retriever], model_variant="castorini/monobert-large-msmarco-finetune-only").build()
    for model  in models:
        checkpoint, variant = models[model]
        pipelines_kv[model + "_" + retriever] = TinyBertPieline(retrievers[retriever], checkpoint=checkpoint, model_variant=variant).build()

ppl_names, pipelines =[], []
for k, v in pipelines_kv.items():
    ppl_names.append(k)
    pipelines.append(v)


print(ppl_names)


def avg_score_k_impl(run, qrels, k):
    return qrels.score[:k].mean()

def avg_score_k(k):
    return ir_measures.define_byquery(lambda run, qrels: avg_score_k_impl(run, qrels, k), name=f"avg_score@{k}")

metrics = [nDCG@10, MAP, P@10, RR@10, "mrt"]
for k in range(1, 101):
    metrics += [P@k, avg_score_k(k)]

topics_filter_rng = np.random.RandomState(42)
max_qids = min(len(dataset.get_topics().qid.unique()), 200)
random_qids = topics_filter_rng.choice(dataset.get_topics().qid.unique(), max_qids, replace=False)
topics = dataset.get_topics()
topics_filtered = topics[topics.qid.apply(lambda x: x in random_qids)]
qrels = dataset.get_qrels()
qrels_filtered = qrels[qrels.qid.apply(lambda x: x in random_qids)]
result = pt.Experiment(
#   [bm25, monot5_100k, monot5_10k, distilroberta_default_full,  distilroberta_gbce_onecycle, distilroberta_gbce_onecycle_calibrated, distilroberta_gbce_onecycle_t075_25Msteps],
    pipelines,
    topics_filtered,
    qrels_filtered, 
    metrics,
   names = ppl_names,
#   names=["BM25", "BM25_monot5_100k", "BM25_monot5_10k",  "BM25_ce_distilroberta_default_full", 
#          "BM25_ce_distilroberta_gbce_full", "BM25_ce_distilroberta_gbce_full_calibrated", "BM25_distilroberta_gbce_onecycle_t075_25Msteps"],
   save_dir = f"./{args.dataset}",
    save_mode="overwrite"

)

interesting_metrics = ["nDCG@10", "AP", "P@10", "RR@10", "mrt"]
result[["name"] + interesting_metrics].to_csv(f"tinybert_vs_monot5_{args.dataset}.csv")

