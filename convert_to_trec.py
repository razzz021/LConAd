import os
from utils import write_json, read_json

def get_dir_files_dict(root):
    dir_files_dict = {}
    
    for dirpath, dirnames, filenames in os.walk(root):
        # Exclude the root directory itself from the dictionary
        if dirpath != root:
            dirpath = str(dirpath.split("/")[-1])
            filenames = [file.replace(".json","").strip() for file in filenames]
            dir_files_dict[dirpath] = filenames
    
    return dir_files_dict

def sort_docs_by_score(scores, cad_dict):
    updated_cad_dict = {}

    for qid, docids in cad_dict.items():
        # Create a dictionary to store scores for each docid
        doc_scores = {}

        for docid in docids:
            # Get score from scores dictionary, default to 0 if not found
            score = scores.get(qid, {}).get(docid, 0)
            doc_scores[docid] = score

        # Sort docids by score in descending order
        sorted_doc_scores = dict(sorted(doc_scores.items(), key=lambda item: item[1], reverse=True))
        
        updated_cad_dict[qid] = sorted_doc_scores

    return updated_cad_dict

def convert_to_trec_format(updated_cad_dict, run_id):
    trec_lines = []

    for qid, doc_scores in updated_cad_dict.items():
        rank = 1
        for docid, score in doc_scores.items():
            # Format each line in TREC format
            trec_line = f"{qid} Q0 {docid} {rank} {score} {run_id}"
            trec_lines.append(trec_line)
            rank += 1

    return "\n".join(trec_lines)

# Example usage
root_directory = '/home/ray/suniRet/data/LeCaRD/candidates'  # Replace with the path to your root directory
cad_dict = get_dir_files_dict(root_directory)
# print(cad_dict)
# write_json(cad_dict, "query_cand_dict.json")
dir = "/home/ray/suniRet/trec_eval/scores"
for file in os.listdir(dir):
    if file.endswith("json"):
        run_id = file.replace(".json","")
        scores = read_json(os.path.join(dir, file))
        result = sort_docs_by_score(scores, cad_dict)
        trec_lines = convert_to_trec_format(result, run_id)

        with open(f"{dir}/{run_id}.trec", 'w') as f:
            f.write(trec_lines)