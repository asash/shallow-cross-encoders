from collections import defaultdict
import pyterrier as pt
pt.init()
from pyterrier_pisa import PisaIndex
index = PisaIndex.from_dataset('msmarco_passage')
bm25 = index.bm25()
from tqdm import tqdm
import pandas as pd
import numpy as np

rng = np.random.RandomState(31339)

import ir_datasets
dataset = ir_datasets.load('msmarco-passage/train')
num_docs = len(dataset.docs)
queries = {q.query_id: q.text for q in dataset.queries}
batch_size = 4096

all_qrels = [(q.query_id, q.doc_id) for q in dataset.qrels]
q_rel = defaultdict(list)

for q, doc in all_qrels:
    q_rel[q].append(doc)

all_queries_with_rels = list(q_rel.keys())
row_size = 1001 # 1 qid, 1 rel, 999 negs

rows = []
for i in tqdm(range(0, len(all_queries_with_rels), batch_size)):
    qids = all_queries_with_rels[i:i+batch_size]
    df_docs = []
    for qid in qids:
        df_docs.append({"qid": qid, "query" : queries[qid]})
    df = bm25.transform(pd.DataFrame(df_docs))
    for qid,qid_df in df.groupby("qid"):
        qid_df_filtered = qid_df[qid_df.docno.apply(lambda x: x not in q_rel[qid])]
        random_rel = rng.choice(q_rel[qid])
        if random_rel not in set(qid_df.docno): #we don't use queries without positives in bm25
            continue
        bm25_docnos = qid_df_filtered.docno.values
        if len(bm25_docnos) > row_size - 2:
            bm25_docnos = rng.choice(bm25_docnos, row_size - 2, replace=False)
        
        result_row = [int(qid), int(random_rel)] + [int(docno) for docno in bm25_docnos]
        if len(result_row) < row_size: #add uniformly sampled negs if not enough in bm25
            extra_docs = rng.choice(num_docs, row_size - len(result_row))
            result_row += list(extra_docs)
        assert(len(result_row) == row_size)
        rows.append(result_row)
        
mmap_file = f"datasets/qid_rel_{row_size-2}_bm25negs_{len(rows)}x{row_size}.mmap"
rows = np.array(rows, dtype=np.int32)
mmap = np.memmap(mmap_file, dtype=np.int32, mode="w+", shape=rows.shape)
mmap[:] = rows[:]
mmap.flush()