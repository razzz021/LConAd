import jsonlines
import json 

def write_jsonlines(list, file):
    with jsonlines.open(file, mode='w') as writer:
            writer.write_all(list)
            
def read_jsonlines(file):
    with jsonlines.open(file, 'r') as reader:
        return list(reader)
    
    
def write_json(dict, file):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(dict, f, indent=4, ensure_ascii=False)
        
def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)