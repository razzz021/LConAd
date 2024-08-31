# MODEL_NAME=/home/ray/suniRet/train_output/train_simcse_mixture_small/checkpoint-1800
# OUTPUT_NAME=/home/ray/suniRet/trec_eval/scores/scores_my.json

# MODEL_NAME=BAAI/bge-large-zh-v1.5
# OUTPUT_NAME=/home/ray/suniRet/trec_eval/scores/scores_bge.json

# MODEL_NAME=CSHaitao/SAILER_zh
# OUTPUT_NAME=/home/ray/suniRet/trec_eval/scores/scores_SAILER.json

# MODEL_NAME=hfl/chinese-roberta-wwm-ext
# OUTPUT_NAME=/home/ray/suniRet/trec_eval/scores/scores_RoBERTa.json

# MODEL_NAME=google-bert/bert-base-chinese
# OUTPUT_NAME=/home/ray/suniRet/trec_eval/scores/scores_bert.json

MODEL_NAME=thunlp/Lawformer
OUTPUT_NAME=/home/ray/suniRet/trec_eval/scores/scores_Lawformer.json


python /home/ray/suniRet/lecardv1_dense.py --top_k 10709 \
    --corpus /home/ray/suniRet/data/fact_fix_candidates.jsonl \
    --queries /home/ray/suniRet/data/querys.jsonl \
    --output $OUTPUT_NAME \
    --model_name $MODEL_NAME \
    --out_score

    # --corpus /home/ray/suniRet/data/gpt_data/gpt4_fact.jsonl \