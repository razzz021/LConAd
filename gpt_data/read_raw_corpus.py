import os 
import json 
import jsonlines

def write_jsonlines(list, file):
    with jsonlines.open(file, mode='w') as writer:
            writer.write_all(list)
            
def walk_dir_read_all_jsons(directory):
    jsons = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    d = json.load(f)
                    d['ajId'] = file.strip().replace('.json',"").strip()
                    jsons.append(d)
    return jsons

corpus = walk_dir_read_all_jsons("/home/ray/suniRet/data/LeCaRD/candidates")

copurs_ids = [doc['ajId'] for doc in corpus]
print("total number of corpus")
print(len(copurs_ids))
copurs_ids = set(copurs_ids)
print("total number of unduplicate docs")
print(len(copurs_ids))


def deduplicate(list):
    new_list = []
    ids = set()
    for item in list:
        id = item['text_id']
        if id not in ids:
            ids.add(id)
            new_list.append(item)
        else:
            continue
    return new_list



new_docs = []
for doc in corpus:
    if len(doc['ajjbqk']) < 10000:
    # if len(doc['ajjbqk']) < 10000:

        new_docs.append({'text_id':doc['ajId'], 'text':doc['qw']})
        
        
length = 0
num_doc = 0 
new_docs = deduplicate(new_docs)
for doc in new_docs:
    num_doc += 1
    length += len(doc['text'])

print(length * 2 / 1000000 * 0.075)
print(num_doc * 2 * 1000 / 1000000 * 0.300)
print(num_doc)

# from utils import write_jsonlines
write_jsonlines(new_docs,'/home/ray/suniRet/gpt_data/original_corpus.jsonl')