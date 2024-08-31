for ckp in 600 1200 1800
do

MODEL_NAME=/home/ray/suniRet/train_output/train_simcse_mixture_small/checkpoint-$ckp
OUTPUT_NAME=results/r-gs-simcse-mix-"$ckp".json
CORPUS=/home/ray/suniRet/data/long_data/reason_candidates.jsonl
QUERYS=/home/ray/suniRet/data/querys.jsonl


python /home/ray/suniRet/lecardv1_dense.py --top_k 500 \
    --corpus $CORPUS \
    --queries $QUERYS \
    --output $OUTPUT_NAME \
    --model_name $MODEL_NAME 


# # Directory containing the files to process
# REUSLTS_DIR=results
# OUTPUT_DIR=metrics
# # Create the output directory if it doesn't exist
# mkdir -p "$OUTPUT_DIR"

# # Loop through each file in the input directory
# for file in "$REUSLTS_DIR"/*; do
#   # Get the base name of the file (without directory)
#   base_name=$(basename "$file")
#   base_name_without_suffix="${base_name%.*}"
#   # Construct the output file path
#   output_file="$OUTPUT_DIR/${base_name_without_suffix}.json"
  
#     python evaluate_v1.py --label /home/ray/suniRet/data/golden_labels.json\
#         --pred $file \
#         --output $output_file \
#         --is_golden true
# done

# python collect_results.py

done