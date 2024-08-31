import jsonlines 
import json 
import copy

'"content": "你是一位专业的细致的法官，你正在整理法庭裁判文书。请整理法庭裁判文书，按照以下四个字段进行分段：{“案情事实总结”：对案件的基本事实进行简明扼要的总结，包括当事人情况、争议焦点等， “法院裁判分析”: 对案件中涉及的法律问题进行分析，阐述法院的法律依据、论证过程、对相关法律条款的解释，“判决结果总结”：明确裁判结果，包括法院的判决或裁定，判决理由以及对当事人的具体裁决，“触犯罪名”：指出案件中涉及的具体罪名或法律条款}"'

TEMPLATE={
    "custom_id": None,
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "你是一位专业的细致的法官，请整理法庭裁判文书，按照以下四个字段进行内容抽取总结：“案情事实总结”、“法院裁判分析”、“判决结果总结”、“触犯罪名”"
            },
            {
                "role": "user",
                "content": None,
            }
        ],
    }
}


def write_jsonlines(list, file):
    with jsonlines.open(file, mode='w') as writer:
            writer.write_all(list)
            
def read_jsonlines(file):
    with jsonlines.open(file, 'r') as reader:
        return list(reader)
        

def split_list(lst, num_parts):
    length = len(lst)
    if length == 0:
        return [[] for _ in range(num_parts)]
    
    # Determine the size of each part
    avg_size = length // num_parts
    remainder = length % num_parts

    parts = []
    start = 0

    for i in range(num_parts):
        # If there is a remainder, add one more element to the current part
        end = start + avg_size + (1 if i < remainder else 0)
        parts.append(lst[start:end])
        start = end

    return parts


def test():
    corpus = read_jsonlines("original_corpus.jsonl")
    # corpus = corpus[:2]
    
    batches = []
    for doc in corpus:
        new_dict = copy.deepcopy(TEMPLATE)
        new_dict['custom_id'] = str(doc['text_id'])
        new_dict['body']['messages'][1]['content'] = str(doc['text'])
        
        batches.append(new_dict)
    
    total_num = len(batches)
    print(total_num)
    
    
    stack_batches = split_list(batches, 20)
    
    
    for idx, batch in enumerate(stack_batches):
        write_jsonlines(batch, f"data/batches_{idx}.jsonl")

    
    # write_jsonlines(batches, "lecardv1_batches.jsonl")
    

def main():
    return None 



if __name__=="__main__":
    test()