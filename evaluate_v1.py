import os
import numpy as np
import json
import math
import argparse
import json
from prettytable import PrettyTable

# 


"""
labels format: {qid1: {docid1: score1, docid2:score2,...}, qid2: {docid1: score1, docid2:score2,...}}
preds format: {qid1: [docid1, docid2, ...], qid2: [docid1, docid2, ...]}
"""

def format_and_validate(preds, labels):

    warnings = []

    # Format labels
    formatted_labels = {}
    for qid, docs in labels.items():
        formatted_qid = str(qid)
        formatted_docs = {str(docid): float(score) for docid, score in docs.items() if int(score) > 0}
        formatted_labels[formatted_qid] = formatted_docs

    # Format preds and ensure qid in labels
    formatted_preds = {}
    for qid, docs in preds.items():
        formatted_qid = str(qid)
        if formatted_qid not in formatted_labels:
            warnings.append(f"Warning: Query ID '{formatted_qid}' in preds not found in labels. It has been removed.")
            continue
        formatted_docs = [str(docid) for docid in docs]
        formatted_preds[formatted_qid] = formatted_docs

    return formatted_preds, formatted_labels, warnings

def predsForm2labelsForm(preds):
    new_dict = {}
    for qid in preds:
        new_dict[qid] = {doc_id: 1 for doc_id in preds[qid]}
    return new_dict

def calculate_recall_topk(preds, labels, k):

    recall_sum = 0.0
    num_queries = len(preds)
    
    for query_id in labels:
        relevant_docs = {doc_id for doc_id, score in labels[query_id].items()}
        retrieved_docs = preds.get(query_id, [])

        # Ensure topK is not greater than the number of retrieved documents
        topK_retrieved_docs = set(retrieved_docs[:k])

        true_positives = topK_retrieved_docs.intersection(relevant_docs)
        
        if len(relevant_docs) == 0:
            recall = 0.0
            num_queries -= 1
        # elif len(true_positives) == len(topK_retrieved_docs):
        #     recall = 1
        else:
            recall = len(true_positives) / len(relevant_docs)
        
        recall_sum += recall
    
    average_recall = recall_sum / num_queries
    
    return average_recall

def calculate_ndcg_topk(preds, labels, k):

    def dcg_at_k(scores, k):

        dcg = 0.0
        for i in range(min(k, len(scores))):
            dcg += (2 ** scores[i] - 1) / math.log2(i + 2)
            # dcg += scores[i]/math.log2(i + 2)
        return dcg
    
    ndcg_sum = 0.0
    num_queries = len(labels)

    for query_id in labels:
        relevant_docs = labels[query_id]
        retrieved_docs = preds.get(query_id, [])

        # Get relevance scores of the retrieved documents
        retrieved_scores = [relevant_docs.get(doc_id, 0) for doc_id in retrieved_docs[:k]]
        
        # use rank instead
        # retrieved_scores = [(len(retrieved_scores) - idx) * score for idx, score in enumerate(retrieved_scores)]

        # Calculate DCG for the retrieved documents
        dcg = dcg_at_k(retrieved_scores, k)
        
        # print(retrieved_scores)
        
        # Calculate the ideal DCG for the top K relevant documents
        ideal_retrieved_scores = sorted(relevant_docs.values(), reverse=True)
        
        # use rank instead
        # ideal_retrieved_scores = [(len(retrieved_scores) - idx) * 1 for idx, score in enumerate(retrieved_scores)]
        
        # print(ideal_retrieved_scores)
        
        ideal_dcg = dcg_at_k(ideal_retrieved_scores, k)
        
        # print()
        # Calculate NDCG
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        ndcg_sum += ndcg

    average_ndcg = ndcg_sum / num_queries
    return average_ndcg

def calculate_precision_topk(preds, labels, k):
    recall_sum = 0.0
    num_queries = len(preds)
    
    for query_id in labels:
        relevant_docs = {doc_id for doc_id, score in labels[query_id].items()}
        retrieved_docs = preds.get(query_id, [])

         # Ensure k is not greater than the number of retrieved documents
        topk = min(k, len(retrieved_docs), len(relevant_docs))
        
        # Ensure topK is not greater than the number of retrieved documents
        topK_retrieved_docs = set(retrieved_docs[:topk])

        true_positives = topK_retrieved_docs.intersection(relevant_docs)
        
        if len(relevant_docs) == 0:
            recall = 0.0
            num_queries -= 1
        # elif len(true_positives) == len(topK_retrieved_docs):
        #     recall = 1
        else:
            recall = len(true_positives) / k
        
        recall_sum += recall
    
    average_recall = recall_sum / num_queries
    
    return average_recall

def calculate_map(preds, labels):


    def calculate_average_precision(retrieved_docs, relevant_docs):

        if not relevant_docs:
            return 0.0

        hits = 0
        sum_precisions = 0.0

        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                hits += 1
                precision_at_i = hits / (i + 1)
                sum_precisions += precision_at_i

        avg_precision = sum_precisions / len(relevant_docs)
        return avg_precision

    avg_precisions = []

    for query_id in preds:
        retrieved_docs = preds[query_id]
        relevant_docs = {doc_id for doc_id, score in labels.get(query_id, {}).items()}

        avg_precision = calculate_average_precision(retrieved_docs, relevant_docs)
        avg_precisions.append(avg_precision)

    mean_avg_precision = sum(avg_precisions) / len(avg_precisions) if avg_precisions else 0.0
    return mean_avg_precision

def calculate_mrr_topk(preds, labels, k):

    mrr_sum = 0.0
    num_queries = len(preds)

    for query_id in preds:
        retrieved_docs = preds[query_id][:k]
        relevant_docs = {doc_id for doc_id, score in labels.get(query_id, {}).items()}

        for rank, doc_id in enumerate(retrieved_docs, start=1):
            if doc_id in relevant_docs:
                mrr_sum += 1.0 / rank
                break

    mean_reciprocal_rank = mrr_sum / num_queries
    
    return mean_reciprocal_rank

def show_metric_table(metric_table):
    
    def custom_sort_key(key):
        if '@' not in key:
            return (key, 0)
        metric, value = key.split('@')
        return (metric, int(value))

    sorted_metrics = dict(sorted(metric_table.items(), key=lambda item: custom_sort_key(item[0])))
    

    # Create a PrettyTable object
    table = PrettyTable()
    table.field_names = ["Metric", "Value"]

    # Add rows to the table
    for metric, value in sorted_metrics.items():
        table.add_row([metric, f"{value:.3f}"])
    
    table.align["Metric"] = "l"

    # Print the table to the screen
    print(table)
    
    return sorted_metrics

def main():
    parser = argparse.ArgumentParser(description="Help info:")
    parser.add_argument('--label', type=str, default=None, help='Label file path.')
    parser.add_argument('--pred', type=str, default=None, help='Prediction file path.')
    parser.add_argument('--output', type=str, default=None, help='Output file path.')
    parser.add_argument('--is_golden', type=bool, default=False, help='the labels are golden or dict.')
    
    args = parser.parse_args()
    
    print(f"Reading preds from  : {args.pred}")
    with open(args.pred, 'r', encoding='utf-8') as f:
        preds = json.load(f)
        
    print(f"Reading labels from : {args.label}")
    with open(args.label, 'r', encoding='utf-8') as f:
        labels = json.load(f)

    if args.is_golden:
        labels = predsForm2labelsForm(labels)
    
    preds, labels, _ = format_and_validate(preds, labels)
    
    metric_dict = {}
    
    metric_dict["MAP"] = calculate_map(preds, labels)
    
    for topk in [5, 10, 30]:
        metric_dict[f"NDCG@{topk}"] = calculate_ndcg_topk(preds, labels, topk)
        
        metric_dict[f"MRR@{topk}"] = calculate_mrr_topk(preds, labels, topk)

    for topk in [30,50]:
        metric_dict[f"Recall@{topk}"] = calculate_recall_topk(preds, labels, topk)
    
    for topk in [10]:
        metric_dict[f"Precision@{topk}"] = calculate_precision_topk(preds, labels, topk)
        
    sorted_metrics = show_metric_table(metric_dict)

    file = args.output
    if file is not None:
        with open(file, 'w') as f:
            json.dump(sorted_metrics, f, indent=4)
            print(f"\nMetrics saved to {file}")

if __name__ == "__main__":
    main()
    

    
    

    
    

