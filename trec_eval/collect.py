import os
import csv 
from collections import OrderedDict

def read_scores_from_dir(dir):
    all_scores = {}
    valid_prefixes = ('P_', 'recall_', 'ndcg_')
    valid_suffixes = ('5','10','15','20','30')
    files = os.listdir(dir)
    
    for file in files:
        file_path = os.path.join(dir, file)
        scores_dict = {}
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                # Strip leading/trailing whitespace and split by tab
                parts = line.strip().split('\t')
                
                if len(parts) == 3:
                    key, value_type, value = parts
                    key = key.strip()
                    if value_type == 'all' and key.startswith(valid_prefixes) and key.endswith(valid_suffixes):
                        # Convert numeric values to float
                        try:
                            value = float(value)
                        except ValueError:
                            continue
                        
                        # Store in dictionary
                        key = key.strip(",").strip()
                        key = key.replace("_cut_","@")
                        key = key.replace("_","@")
                        scores_dict[key] = value
                        sorted_scores_dict = OrderedDict(sorted(scores_dict.items(), key=lambda item: item[1], reverse=True))
        # Store in the all_scores dictionary with filename as key
        all_scores[os.path.basename(file_path)] = sorted_scores_dict

    return all_scores


def dict_to_csv(data, output_csv_file):
    # Collect all unique metrics
    all_metrics = set()
    for scores in data.values():
        all_metrics.update(scores.keys())
    all_metrics = sorted(all_metrics)  # Sort for consistent ordering

    # Prepare the rows for the CSV
    rows = []
    for filename, scores in data.items():
        row = [filename]
        for metric in all_metrics:
            row.append(scores.get(metric, ''))  # Use empty string if metric is not present
        rows.append(row)

    # Write to CSV
    with open(output_csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(['file'] + all_metrics)
        # Write the data rows
        writer.writerows(rows)



dir = "/home/ray/suniRet/trec_eval/res"
all_scores = read_scores_from_dir(dir)
dict_to_csv(all_scores, "trec_table.csv")

        