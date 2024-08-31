import json 
import jsonlines 



def main():
    with open("/home/ray/suniRet/data/golden_labels.json", 'r', encoding='utf8') as f:
        labels = json.load(f)
    
    with jsonlines.open("/home/ray/suniRet/data/fact_candidates.jsonl", 'r') as f:
        lines = list(f)
        cads = [item['text_id'] for item in lines]
        cads = set(cads)
    
    for k, vlist in labels:
        tmp_list = []
        for did in vlist:
            if did in cads:
                tmp_list.append(did)
            labels[k] = tmp_list

    
    with open("tmp_new_gold.json", 'w', encoding='utf-8') as f:
        json.dump(labels, ensure_ascii=False)
        

if __name__ == "__main__":
    main()