from sparse_ranker import TfidfDocRanker, Bm25DocRanker
from utils import read_jsonlines, write_json


CORPUS_FILE="/home/ray/suniRet/data/long_data/fact_candidates.jsonl"
QUERY_FILE="/home/ray/suniRet/data/querys.jsonl"

def main(query_file, corpus_file):
    
    querys = read_jsonlines(query_file)
    corpus = read_jsonlines(corpus_file)
    
    ranker = Bm25DocRanker(tokenizer_type="jieba", ngrams=3, b=0.75, k1=1.6)
    
    corpus_id = [doc['text_id'] for doc in corpus]
    corpus_text = [doc['text'] for doc in corpus]
    querys_id = [doc['text_id'] for doc in querys]
    querys_text = [doc['text'] for doc in querys]
    
    ranker.fit(corpus_text)
    topk = len(corpus_id)
    doc_idxs, scores = ranker.query_batch(querys_text, topk)
    
    
    # results = {}
    # for query_idx, doc_idx in enumerate(doc_idxs):
    #     query_id = querys_id[query_idx]
    #     docid_score = {}
    #     for idx, score in zip(doc_idx, scores[query_idx]):
    #         docid_score[corpus_id[idx]] = float(score)
    #     results[query_id] = docid_score
        
    # return results

    results = {}
    for query_idx, doc_idx in enumerate(doc_idxs):
        query_id = querys_id[query_idx]
        doc_id =  []
        for idx in doc_idx:
            doc_id.append(corpus_id[idx])
        results[query_id] = doc_id
    return results
    
if __name__=='__main__':
    results = main(QUERY_FILE, CORPUS_FILE)
    write_json(results, "results/fact_bm25.json")

