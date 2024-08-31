
python unsup_data_prepare.py single --input /home/ray/suniRet/data/gpt4_fact.jsonl --output /home/ray/suniRet/data/gpt4_fact.txt
python unsup_data_prepare.py single --input /home/ray/suniRet/data/gpt4_reason.jsonl --output /home/ray/suniRet/data/gpt4_reason.txt
python unsup_data_prepare.py combined \
    --input1 /home/ray/suniRet/data/gpt4_fact.jsonl  \
    --input2 /home/ray/suniRet/data/gpt4_reason.jsonl \
    --output /home/ray/suniRet/data/gpt4_fact_reason.txt