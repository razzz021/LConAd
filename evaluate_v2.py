
import torch
from torchmetrics import RetrievalNormalizedDCG, RetrievalMRR, RetrievalMAP, RetrievalPrecision, RetrievalRecall



def calculate_ndcg_topk(pred, label, topk):
    pl = []
    tl = []
    idxl = []
    for idx, (qid, docids) in enumerate(pred.items()):
        for rank, docid in enumerate(docids[:topk]):
            pl.append(len(docids) - rank)
            tl.append(True if docid in label[qid] else False)
            idxl.append(idx)
    pl = torch.tensor(pl, dtype=torch.float)
    tl = torch.tensor(tl, dtype=torch.bool)
    idxl = torch.tensor(idxl, dtype=torch.long)
    ndcg = RetrievalNormalizedDCG()
    return ndcg(pl, tl, idxl)


def calculate_recall_topk(pred, label, topk):
    pl = []
    tl = []
    idxl = []
    for idx, (qid, docids) in enumerate(pred.items()):
        for rank, docid in enumerate(docids[:topk]):
            pl.append(len(docids) - rank)
            tl.append(True if docid in label[qid] else False)
            idxl.append(idx)
    pl = torch.tensor(pl, dtype=torch.float)
    tl = torch.tensor(tl, dtype=torch.bool)
    idxl = torch.tensor(idxl, dtype=torch.long)
    r_k = RetrievalRecall(top_k=2)
    return r_k(pl, tl, idxl)

