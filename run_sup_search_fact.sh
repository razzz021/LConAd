# MODEL_NAME=/home/ray/suniRet/train_output/sup_random_train_gpt
# OUTPUT_NAME=sup_results/sup-rob-random.json

# MODEL_NAME=/home/ray/suniRet/train_output/sup_hn_train_gpt
# OUTPUT_NAME=sup_results/sup-rob-hn.json

# MODEL_NAME=/home/ray/suniRet/train_output/sup_hn_train_gs_gpt
# OUTPUT_NAME=sup_results/sup-rob-hn-gs.json

# MODEL_NAME=/home/ray/suniRet/train_output/sup_random_train_gs_gpt
# OUTPUT_NAME=sup_results/sup-rob-random-gs.json

MODEL_NAME=/home/ray/suniRet/train_output/train_mixture/checkpoint-1800
OUTPUT_NAME=sup_results/gs.json


# MODEL_NAME=hfl/chinese-roberta-wwm-ext
# OUTPUT_NAME=sup_results/rob.json

# MODEL_NAME=CSHaitao/SAILER_zh
# OUTPUT_NAME=sup_results/SAILER.json

CORPUS=/home/ray/suniRet/data/long_data/fact_candidates.jsonl
QUERYS=/home/ray/suniRet/data/querys.jsonl


python /home/ray/suniRet/lecardv1_dense.py --top_k 500 \
    --corpus $CORPUS \
    --queries $QUERYS \
    --output $OUTPUT_NAME \
    --model_name $MODEL_NAME 