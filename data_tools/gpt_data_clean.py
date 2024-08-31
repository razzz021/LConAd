import jsonlines 
import re
import os 


FACT =  r'\*\*案情事实总结\*\*(.*?)\*\*法院裁判分析\*\*'

REASON =  r'\*\*法院裁判分析\*\*(.*?)\*\*判决结果总结\*\*'

FACT_TEMPLATE="案情事实：{}。"
REASON_TEMPLATE="案情分析：{}。"


def read_jsonlines(file):
    with jsonlines.open(file, 'r') as reader:
        return list(reader)

def write_jsonlines(file, obj):
    with jsonlines.open(file, 'w') as writer:
        writer.write_all(obj)

def get_response(item):
    
    text_id = item['custom_id']
    text = item['response']["body"]["choices"][0]['message']['content']
    
    return text_id, text


def extract_pattern(text, pat):

    match = re.search(pat, text, re.DOTALL)
    
    if match:
        text = match.group(1).strip()
        text = text.replace("\n","。")
        text = text.strip("\n ：,*  。")
        return text 
    else:
        return None


def load_all_jsonl(dir):
    responses = []
    files = os.listdir(dir)
    for file in files:
        if file.endswith(".jsonl"):
            responses.extend(read_jsonlines(os.path.join(dir, file)))
    return responses


def main():
    responses = load_all_jsonl("/home/ray/suniRet/data/gpt_ori_data")
    # responses = read_jsonlines("/home/ray/suniRet/gpt_data/sample_data/batch_bwN2FtNv3K3Z7hfv9Dr5v29h_output.jsonl")
    # print(responses)
    
    fact_cads = []
    reason_cads = []
    for response in responses:
        text_id, text = get_response(response)
        fact = extract_pattern(text, FACT)
        reason = extract_pattern(text, REASON)
        
        if fact and reason:
        # if (fact and len(fact) > 20) & (reason and len(reason) > 20):
            fact_cads.append({'text_id':text_id, "text":FACT_TEMPLATE.format(fact)})
            reason_cads.append({'text_id':text_id, "text":REASON_TEMPLATE.format(reason)})
            
    write_jsonlines("/home/ray/suniRet/data/gpt4_fact.jsonl", fact_cads)
    write_jsonlines("/home/ray/suniRet/data/gpt4_reason.jsonl", reason_cads)

    
    
    
if __name__ == "__main__":
    main()
