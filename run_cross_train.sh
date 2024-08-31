python train.py \
    model_name="hfl/chinese-roberta-wwm-ext" \
    data_path="/home/ray/suniRet/data/cross_ranker.jsonl" \
    batch_size=2 \
    epochs=3 \
    evaluation_steps=2 \
    warmup_steps=100 \
    output_dir="./cross_ranker_output" \
    test_size=0.2 \
    seed=42 \
    learning_rate=2e-5 \
    weight_decay=0.01
