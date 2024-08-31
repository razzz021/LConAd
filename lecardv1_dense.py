from utils import write_json, read_jsonlines
from dense_ranker import DenseRanker, build_model
from collections import defaultdict
import argparse
import torch 


def main():
    parser = argparse.ArgumentParser(description='Process some inputs.')
    parser.add_argument('--corpus', type=str, required=True, help='Path to the corpus file')
    parser.add_argument('--queries', type=str, required=True, help='Path to the queries file')
    parser.add_argument('--output', type=str, required=True, help='Path to the output file')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--out_score', action='store_true', help='Include scores in the output')
    parser.add_argument('--top_k', type=int, default=1000, help='Number of top results to return for each query')
    parser.add_argument('--query_add_prompt', type=bool, default=True, help='add prompt before query')
    
    
    args = parser.parse_args()

    # Read corpus and queries
    corpus_list = read_jsonlines(args.corpus)
    querys_list = read_jsonlines(args.queries)
    
    # Convert lists to dictionaries
    corpus = {item['text_id']: item['text'] for item in corpus_list}
    if args.query_add_prompt:
        querys = {item['text_id']: "案情事实：" + item['text'] for item in querys_list}
        
    else:
        querys = {item['text_id']: item['text'] for item in querys_list}
    
    query_ids = list(querys.keys())
    query_texts = list(querys.values())
    
    # Build and move model to GPU if available
    model = build_model(args.model_name)
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Fit ranker with corpus and model
    ranker = DenseRanker.fit(corpus, model)
    
    # Perform search
    results = ranker.search(query_texts, top_k=args.top_k, return_text=False)
    
    # Prepare predictions
    if args.out_score:
        preds = defaultdict(dict)
    else:
        preds = defaultdict(list)
    
    for idx, query_id in enumerate(query_ids):
        result = results[idx]
        for item in result:
            docid = item['docid']
            score = item['score']
            score = float(score)
            
            if args.out_score:
                preds[query_id][docid] = score
            else:
                preds[query_id].append(docid)
    
    # Write predictions to output file
    write_json(preds, args.output)

if __name__ == "__main__":
    main()