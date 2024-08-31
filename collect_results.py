import prettytable
import os, csv
import json

METRICS_DIR="/home/ray/suniRet/metrics"
# METRICS_DIR="/home/ray/suniRet/sup_metrics"
OUTPUT_FILE="all_results_metrics.csv"

def load_results_table(results_dir):
    files = os.listdir(results_dir)
    results_table = {}
    for file in files:
        if file.endswith(".json"):
            with open(os.path.join(results_dir,file),'r',encoding='utf-8') as f:
                name = file.replace(".json","")
                results_table[name] = json.load(f)
    return results_table

def custom_sort_key(key):
    if '@' not in key:
        return (key, 0)
    metric, value = key.split('@')
    return (metric, int(value))

def show_results_table(results_table):
    # Create a PrettyTable instance
    table = prettytable.PrettyTable()

    # Extract all unique metric names
    metrics = set()
    for metrics_dict in results_table.values():
        metrics.update(metrics_dict.keys())
    
    # Add columns to the table: "Model_Name" and one column for each metric
    
    sorted_metrics = sorted(metrics, key=custom_sort_key)
    table.field_names = ["Model_Name"] + sorted_metrics
    
    # Populate rows with data
    for directory, metrics_dict in results_table.items():
        row = [directory]
        for metric in table.field_names[1:]:  # Skip the first column which is "Directory"
            value = metrics_dict.get(metric, "N/A")
            if isinstance(value, (int, float)):
                value = f"{value:.2f}"  # Format numeric values to 3 decimal places
            row.append(value)
        table.add_row(row)
    
    # Print the table
    print(table)

def show_results_table_v2(results_table):

    
    # Extract all unique metric names
    metrics = set()
    for metrics_dict in results_table.values():
        metrics.update(metrics_dict.keys())
    
    # Sort metrics
    sorted_metrics = sorted(metrics, key=custom_sort_key)
    
    # Create a PrettyTable instance
    table = prettytable.PrettyTable()

    # Define table headers: all directories + "Metric"
    headers = ["Model"] + list(results_table.keys())
        # Wrap headers to handle long column names

    table.field_names = headers
    # Populate rows with data
    for metric in sorted_metrics:
        row = [metric]
        for directory in results_table:
            value = results_table[directory].get(metric, "N/A")
            if isinstance(value, (int, float)):
                value = f"{value:.3f}"  # Format numeric values to 3 decimal places
            row.append(value)
        table.add_row(row)
    
    # Print the table
    print(table)
    
def write_results_to_csv(results_table, csv_filename):
    # Extract all unique metric names

    metrics = set()
    for metrics_dict in results_table.values():
        metrics.update(metrics_dict.keys())
    
    # Open a CSV file for writing
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write the header row
        sorted_metrics = sorted(metrics, key=custom_sort_key)
        header = ["Model_Name"] + sorted_metrics
        writer.writerow(header)

        # Write data rows
        for directory, metrics_dict in results_table.items():
            row = [directory]
            for metric in header[1:]:  # Skip the first column which is "Directory"
                value = metrics_dict.get(metric, "N/A")
                if isinstance(value, (int, float)):
                    value = f"{value:.5f}"  # Format numeric values to 3 decimal places
                row.append(value)
            writer.writerow(row)
            
def main():
    table = load_results_table(METRICS_DIR)
    show_results_table(table)
    write_results_to_csv(table, OUTPUT_FILE)
    
if __name__=="__main__":
    main()