python unsup_data_prepare.py single --input /home/ray/suniRet/data/corpus/fact_candidates.jsonl --output /home/ray/suniRet/data/corpus/fact.txt
python unsup_data_prepare.py single --input /home/ray/suniRet/data/corpus/reason_candidates.jsonl --output /home/ray/suniRet/data/corpus/reason.txt
python unsup_data_prepare.py combined \
    --input1 /home/ray/suniRet/data/corpus/fact_candidates.jsonl \
    --input2 /home/ray/suniRet/data/corpus/reason_candidates.jsonl \
    --output /home/ray/suniRet/data/corpus/fact_reason.txt