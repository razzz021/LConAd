from utils import read_jsonlines, write_jsonlines

def merge_two(part, all):
    part = read_jsonlines(part)
    all= read_jsonlines(all)
    
    all_dict = {x['text_id']:x for x in all}
    
    part_dict = {x['text_id']:x for x in part}
    
    new_list = []
    
    for id in all_dict:
        if id in part_dict:
            new_list.append(part_dict[id])
        else:
            new_list.append(all_dict[id])
            
    return new_list

part = "/home/ray/suniRet/data/gpt_data/gpt4_fact.jsonl"
all = "/home/ray/suniRet/data/fact_candidates.jsonl"

new_list = merge_two(part, all)
write_jsonlines(new_list, "/home/ray/suniRet/data/fact_fix_candidates.jsonl")