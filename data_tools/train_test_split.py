import jsonlines
import json 

def separate_jsonl(input_file, output_file_train, output_file_test, query_ids):
    query_ids = set(map(str, query_ids))  # 确保所有ID都是字符串
    with jsonlines.open(input_file, 'r') as reader, \
         jsonlines.open(output_file_train, 'w') as writer_train, \
         jsonlines.open(output_file_test, 'w') as writer_test:
        
        for obj in reader:
            text_id_str = str(obj['text_id'])  # 确保text_id是字符串
            if text_id_str in query_ids:
                writer_test.write(obj)
            else:
                writer_train.write(obj)

def separate_json(golden_labels, query_ids):
    with open(golden_labels, 'r', encoding='utf8') as f:
        golden_labels = json.load(f)
        train_golden_labels = {}
        test_golden_labels = {}
        for qid in golden_labels:
            if qid in query_ids:
                test_golden_labels[qid] = golden_labels[qid]
            else:
                train_golden_labels[qid] = golden_labels[qid]
    with open("/home/ray/suniRet/data/train_golden_labels.json", 'w', encoding='utf8') as f:
        json.dump(train_golden_labels, f)
    with open("/home/ray/suniRet/data/test_golden_labels.json", 'w', encoding='utf8') as f:
        json.dump(test_golden_labels, f)
        

# 查询ID列表（测试样本ID）
query_ids = [
    "1978", "3765", "1430", "5504", "5239", "3814", "6816", "6706",
    "6432", "6409", "4852", "4873", "4847", "6094", "6072", "6081",
    "-1071", "9", "11", "12", "21", "22", "27", "28", "29"
]

# 输入文件和输出文件路径
input_file = "/home/ray/suniRet/data/querys.jsonl"  # 更新为实际的JSONL文件路径
output_file_train = "/home/ray/suniRet/data/train_querys.jsonl"
output_file_test = "/home/ray/suniRet/data/test_querys.jsonl"

# 执行分割
separate_jsonl(input_file, output_file_train, output_file_test, query_ids)
separate_json("/home/ray/suniRet/data/golden_labels.json", query_ids)