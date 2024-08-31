import json
import random
import argparse
import jsonlines

def ensure_keys_as_strings(d):
    return {str(k): v for k, v in d.items()}

def jsonl_to_dict(ls):
    res = {}
    for l in ls:
        text = l['text']
        text = text.strip("：").strip("，").strip()
        res[str(l['text_id'])] = text
    return res  

def read_jsonlines(file):
    with jsonlines.open(file, 'r') as reader:
        return list(reader)

def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def write_faltten_txt(file, data):
    with open(file, 'w', encoding='utf-8') as f:
        for d in data:
            f.write("\n".join(d) + '\n')
    return 

def generate_triplets_random(candidates, querys, golden_labels):
    triplets = []
    
    all_doc_ids = set(candidates.keys())

    for qid, pos_doc_ids in golden_labels.items():

        if qid not in querys:
            continue 
        
        pos_doc_ids = list(set(pos_doc_ids).intersection(all_doc_ids))  # Ensure pos_doc_ids is a subset of all_doc_ids
        if not pos_doc_ids:
            print(f"Warning: No valid pos_doc_ids for query {qid}.")
            continue
        
        query_text = querys[qid]
        
        if len(query_text) > 950:
            print(f"Warning: Query text too long, length:{len(query_text)}, query {qid}.")
            continue
        
        pos_docs = [candidates[doc_id] for doc_id in pos_doc_ids]

        neg_doc_ids = list(all_doc_ids - set(pos_doc_ids))
        
        for pos_doc in pos_docs:
            neg_doc_id = random.choice(neg_doc_ids)
            neg_doc = candidates[neg_doc_id]
            
            triplets.append((query_text, pos_doc, neg_doc))

    return triplets

def generate_triplets_with_scores(candidates, querys, golden_labels, scores):
    triplets = []

    all_doc_ids = set(candidates.keys())

    for qid, pos_doc_ids in golden_labels.items():
        
        if qid not in querys:
            continue 
        
        pos_doc_ids = list(set(pos_doc_ids).intersection(all_doc_ids))  # Ensure pos_doc_ids is a subset of all_doc_ids
        if not pos_doc_ids:
            print(f"Warning: No valid pos_doc_ids for query {qid}.")
            continue
        
        if qid not in querys:
            continue 
        
        query_text = querys[qid]
        
        if len(query_text) > 950:
            print(f"Warning: Query text too long, length:{len(query_text)}, query {qid}.")
            continue
        
        pos_doc_scores = {doc_id: scores[qid][doc_id] for doc_id in pos_doc_ids}

        all_doc_scores = scores[qid]
        used_neg_docs = set()

        for pos_doc_id, pos_doc_score in pos_doc_scores.items():
            closest_neg_doc_id = None
            closest_neg_doc_score_diff = float('inf')

            for doc_id, score in all_doc_scores.items():
                if doc_id not in pos_doc_ids and doc_id not in used_neg_docs:
                    score_diff = abs(pos_doc_score - score)
                    if score_diff < closest_neg_doc_score_diff:
                        closest_neg_doc_score_diff = score_diff
                        closest_neg_doc_id = doc_id

            if closest_neg_doc_id:
                used_neg_docs.add(closest_neg_doc_id)
                pos_doc_text = candidates[pos_doc_id]
                neg_doc_text = candidates[closest_neg_doc_id]
                triplets.append((query_text, pos_doc_text, neg_doc_text))

    return triplets

def main(cads_file, query_file, golden_labels_file, output_file, scores_file=None):
    candidates = read_jsonlines(cads_file)
    querys = read_jsonlines(query_file)
    golden_labels = read_json(golden_labels_file)

    candidates = jsonl_to_dict(candidates)
    querys = jsonl_to_dict(querys)
    
    for k, q in querys.items():
        querys[k] = "案情事实：" + q
    
    golden_labels = {str(k): [str(i) for i in v] for k, v in golden_labels.items()}

    candidates = ensure_keys_as_strings(candidates)
    querys = ensure_keys_as_strings(querys)

    if scores_file:
        scores = read_json(scores_file)
        scores = {str(k): {str(i): v for i, v in d.items()} for k, d in scores.items()}
        triplets = generate_triplets_with_scores(candidates, querys, golden_labels, scores)
    else:
        triplets = generate_triplets_random(candidates, querys, golden_labels)
    
    write_faltten_txt(output_file, triplets)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate triplets for training.")
    parser.add_argument("--cads_file", type=str, required=True, help="Path to the fact candidates JSONL file")
    parser.add_argument("--query_file", type=str, required=True, help="Path to the query JSONL file")
    parser.add_argument("--golden_labels_file", type=str, required=True, help="Path to the golden labels JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output triplets TXT file")
    parser.add_argument("--scores_file", type=str, required=False, help="Path to the scores JSON file")

    args = parser.parse_args()

    main(args.cads_file, args.query_file, args.golden_labels_file, args.output_file, args.scores_file)
