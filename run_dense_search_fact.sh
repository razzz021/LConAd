MODEL_NAME=google-bert/bert-base-chinese
OUTPUT_NAME=results/f-bert-base-chinese.json

# MODEL_NAME=hfl/chinese-roberta-wwm-ext
# OUTPUT_NAME=results/f-roberta-wwm-ext.json

# MODEL_NAME=cyclone/simcse-chinese-roberta-wwm-ext
# OUTPUT_NAME=results/f-simcse-roberta-wwm-ext.json

# MODEL_NAME=hfl/chinese-bert-wwm
# OUTPUT_NAME=results/f-hfl-bert-wwm.json

# MODEL_NAME=BAAI/bge-large-zh-v1.5
# OUTPUT_NAME=results/f-bge-large-zh-v1.5.json

# MODEL_NAME=CSHaitao/SAILER_zh
# OUTPUT_NAME=results/f-SAILER.json

# MODEL_NAME=thunlp/Lawformer
# OUTPUT_NAME=results/f-Lawformer.json

# /home/ray/suniRet/train_output

python /home/ray/suniRet/lecardv1_dense.py --top_k 1000 \
    --corpus /home/ray/suniRet/data/long_data/fact_candidates.jsonl \
    --queries /home/ray/suniRet/data/querys.jsonl \
    --output $OUTPUT_NAME \
    --model_name $MODEL_NAME 

