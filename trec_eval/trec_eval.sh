/home/ray/trec_eval/trec_eval  /home/ray/suniRet/trec_eval/qrel.trec  /home/ray/suniRet/trec_eval/scores//scores_my_shifted.trec -m all_trec | tee res/my
/home/ray/trec_eval/trec_eval  /home/ray/suniRet/trec_eval/qrel.trec  /home/ray/suniRet/trec_eval/scores//scores_SAILER.trec -m all_trec | tee res/sailer
/home/ray/trec_eval/trec_eval  /home/ray/suniRet/trec_eval/qrel.trec  /home/ray/suniRet/trec_eval/scores//scores_bge.trec -m all_trec | tee res/bge
/home/ray/trec_eval/trec_eval  /home/ray/suniRet/trec_eval/qrel.trec  /home/ray/suniRet/trec_eval/scores//scores_RoBERTa.trec -m all_trec | tee res/roberta
/home/ray/trec_eval/trec_eval  /home/ray/suniRet/trec_eval/qrel.trec  /home/ray/suniRet/trec_eval/scores//scores_bert.trec -m all_trec | tee res/bert
/home/ray/trec_eval/trec_eval  /home/ray/suniRet/trec_eval/qrel.trec  /home/ray/suniRet/trec_eval/scores//scores_Lawformer.trec -m all_trec | tee res/lawformer