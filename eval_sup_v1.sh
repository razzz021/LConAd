# Directory containing the files to process
REUSLTS_DIR=sup_results
OUTPUT_DIR=sup_metrics
# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through each file in the input directory
for file in "$REUSLTS_DIR"/*; do
  # Get the base name of the file (without directory)
  base_name=$(basename "$file")
  base_name_without_suffix="${base_name%.*}"
  # Construct the output file path
  output_file="$OUTPUT_DIR/${base_name_without_suffix}.json"
  
    python evaluate_v1.py --label /home/ray/suniRet/data/golden_labels.json\
        --pred $file \
        --output $output_file \
        --is_golden false
done

python collect_results.py