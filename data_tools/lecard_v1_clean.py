import os 
import json 
import jsonlines
import subprocess
import re 

from clean_rule import clean_text as truncation

OUTPUT_DIR = r"../data"
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

DIR_PREFIX = "/home/ray/suniRet/data/LeCaRD/"

# CANDIDATE_DIR = DIR_PREFIX + r"/corpus/documents"
CANDIDATE_DIR = DIR_PREFIX + r"/candidates"
TEST_LABEL_FILE = DIR_PREFIX + r"/label/label_top30_dict.json"
TEST_QUERY_FILE = DIR_PREFIX + r"/query/query.jsonl"

CRIME_FILE = r"/home/ray/suniRet/data/crimes.txt"


# FACT_TEMPLATE="案件罪名：{}。案情事实：{}。"
# REASON_TEMPLATE="案件罪名：{}。案情分析：{}。"
FACT_TEMPLATE="案情事实：{}。"
REASON_TEMPLATE="案情分析：{}。"

def crime_extract(text):
    with open(CRIME_FILE,'r', encoding='utf-8') as f:
        crimes = f.readlines()
        crimes = [c.strip() for c in crimes]
        crimes_set = set(crimes)
    
    res = []
    for c in crimes_set:
        if c in text:
            res.append(c)        
    sent = "、".join(res)
            
    return sent

def strip_cn_punc(text):
    text = text.strip()
    chinese_punctuation = r"！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～、。"
    pattern = re.compile(f'^[{chinese_punctuation}]+|[{chinese_punctuation}]+$')

    return pattern.sub('', text) 

def clean_fact(fact):
    fact = strip_cn_punc(fact)
    
    # if '经审理查明' in fact:
    #     fact = fact.split('经审理查明')[1]
    
    # if '上述事实' in fact:
    #     fact = fact.split('上述事实')[0]
    
    # if '公诉机关指控' in fact:
    #     fact = fact.split('公诉机关指控')[1]
        
    fact = strip_cn_punc(fact)
    fact = fact.strip(" ")
    
    # if len(fact) < 100:
    #     return None
    
    # if len(fact) > 512:
    #     return None
    
    return fact

def clean_reason(reason):
    
    reason = strip_cn_punc(reason)

    reason = strip_cn_punc(reason)
    reason = reason.strip("判决如下")
    reason = strip_cn_punc(reason)
    
    reason = strip_cn_punc(reason)
    reason = reason.strip(" ")

    # if len(reason) < 50:
    #     return None 
    
    # if len(reason) > 512:
    #     return None
    
    return reason 
    

def candidate_process(cad):
    try:
        # fact = cad.get("ajjbqk", "")
        # reason = cad.get("cpfxgc","")
        fact = cad['ajjbqk']
        reason = cad['cpfxgc']
        crim = crime_extract(cad['pjjg'])
    except:
        # print(cad['ajId'])
        return None 
    
    fact = clean_fact(fact)
    reason = clean_reason(reason)
    
    if fact and reason:
        return  {"text_id":cad['ajId'], "fact":fact, "reason":reason, 'crim':crim}
    else:
        # print(cad['ajId'])
        return None 
    ## clean reson

def walk_dir_read_all_jsons(directory):
    jsons = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    d = json.load(f)
                    d['ajId'] = str(file.strip().strip('.json'))
                    jsons.append(d)
    return jsons

def read_one_file_per_dir(root_dir):
    results = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if filenames:
            # Choose one file to read (e.g., the first file in the list)
            filenames = [f for f in filenames if f.endswith(".json")]
            if not filenames:
                continue
            
            file_to_read = filenames[0]
            file_path = os.path.join(dirpath, file_to_read)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                d = json.load(f)
                d['ajId'] = str(file_to_read.strip().strip('.json'))
                results.append(d)

    return results



def write_jsonlines(list, file):
    with jsonlines.open(file, mode='w') as writer:
        for item in list:
            writer.write(item)

def main():
    
    candidates_dir = CANDIDATE_DIR
    new_data_dir = OUTPUT_DIR
    os.makedirs(new_data_dir, exist_ok=True)
    
    candidates = walk_dir_read_all_jsons(candidates_dir)
    
    # new_cads = []
    # for c in all_candidates:
    #     if 'cpfxgc' not in c:
    #         continue
    #     new_cads.append({'text_id':c['ajId'], 'text':truncation(c['cpfxgc'])})
    
    # all_candidates = new_cads
    
    # all_candidates = [ {'text_id':c['ajId'], 'text':truncation(c['ajjbqk'])} for c in all_candidates]
    
    facts = []
    reasons = []
    for cad in candidates:
        item =  candidate_process(cad)
        if item is not None:
            # facts.append({"text_id":item['text_id'], "text":FACT_TEMPLATE.format(item['crim'], item['fact'])})
            # reasons.append({"text_id":item['text_id'], "text":REASON_TEMPLATE.format(item['crim'], item['reason'])})
            facts.append({"text_id":item['text_id'], "text":FACT_TEMPLATE.format(item['fact'])})
            reasons.append({"text_id":item['text_id'], "text":REASON_TEMPLATE.format(item['reason'])})
        
    
    write_jsonlines(facts, os.path.join(new_data_dir, 'fact_candidates.jsonl'))
    write_jsonlines(reasons, os.path.join(new_data_dir, 'reason_candidates.jsonl'))
    
    
    # read jsonl file
    with jsonlines.open(TEST_QUERY_FILE) as reader:
        all_queries = list(reader)
    
    # ridx to id, q to contents
    all_queries = [{'text_id':int(q['ridx']), 'text':q['q']} for q in all_queries]
    # write_jsonlines(all_queries, os.path.join(new_data_dir, 'querys.jsonl'))
    write_jsonlines(all_queries, os.path.join(new_data_dir, 'querys.jsonl'))
    
    
    
    golden_labels_original_path = DIR_PREFIX + "/label/golden_labels.json"
    label_dict_original_path = DIR_PREFIX + "/label/label_top30_dict.json"
    
    copy_golden_cmd = f"cp -f {golden_labels_original_path} {new_data_dir}/golden_labels.json" 
    copy_label_cmd = f"cp -f {label_dict_original_path} {new_data_dir}/label_dict.json"
    subprocess.run(copy_golden_cmd, shell=True)
    subprocess.run(copy_label_cmd, shell=True)
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    main() 
