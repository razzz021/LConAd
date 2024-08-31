from sentence_transformers import SentenceTransformer, models
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import jsonlines
import json 

# 全局变量定义
# MODEL_NAME = "/home/ray/suniRet/train_output/train_simcse_mixture_small/checkpoint-1800"
# MODEL_NAME="hfl/chinese-roberta-wwm-ext"
# MODEL_NAME="CSHaitao/SAILER_zh"
# MODEL_NAME = "cyclone/simcse-chinese-roberta-wwm-ext"
MODEL_NAME="BAAI/bge-large-zh-v1.5"
# MODEL_NAME="hfl/chinese-bert-wwm"
# MODEL_NAME="thunlp/Lawformer"


CORPUS_PATH = "/home/ray/suniRet/data/reason_candidates.jsonl"
QUERY_PATH = "/home/ray/suniRet/data/querys.jsonl"
GOLDEN_LABEL_PATH = "/home/ray/suniRet/data/golden_labels.json"
VISUALIZE = False
OUTPUT_FILE = "tsne_visualization.png"

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
    
def compute_alignment(positive_pairs, model):
    query_texts = [pair['query_text'] for pair in positive_pairs]
    document_texts = [pair['document_text'] for pair in positive_pairs]
    
    # 编码所有的 query_text 和 document_text
    query_embeddings = model.encode(query_texts, convert_to_tensor=True)
    document_embeddings = model.encode(document_texts, convert_to_tensor=True)
    
    # 计算成对距离 (这里计算每个 query_text 和相应 document_text 之间的距离)
    distances = torch.norm(query_embeddings - document_embeddings, p=2, dim=1)

    
    # 计算 Z-score 归一化后的平均对齐度
    mean_alignment = torch.mean(distances).item()
    
    mean_alignment = mean_alignment
    return mean_alignment

# 计算均匀性 (Uniformity)
def uniformity(embeddings, t=2):
    # Compute the pairwise Euclidean distances between embeddings
    pairwise_dists = torch.pdist(embeddings, p=2)
    
    # Apply the Log-Sum-Exp trick for numerical stability
    uniformity_value = torch.mean(torch.exp(-t * pairwise_dists ** 2))
    
    # Take the logarithm of the result to compute the final uniformity score
    uniformity_value = torch.log(uniformity_value)
    
    uniformity_value = uniformity_value
    
    return uniformity_value.item()

# 构建模型函数
def build_model(model_name):
    if "sentence-transformers" in model_name:
        return SentenceTransformer(model_name)
    
    # 创建自定义Transformer模型
    word_embedding_model = models.Transformer(model_name, max_seq_length=512)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode="cls")
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model

# t-SNE 可视化函数，保存为文件
def visualize_embeddings(embeddings, corpus, output_file, perplexity=30, n_iter=300):
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    for i, label in enumerate(corpus):
        x, y = embeddings_2d[i, :]
        plt.scatter(x, y)
        plt.text(x + 0.1, y + 0.1, label, fontsize=9)
    
    plt.title("t-SNE visualization of sentence embeddings")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    
    # 保存图像到文件
    plt.savefig(output_file)
    plt.close()

def get_positive_pairs(corpus, query, golden_label):
    # 创建一个字典用于快速查找文本内容
    corpus_dict = {str(item["text_id"]): item["text"] for item in corpus}
    
    # 生成正向文本对
    positive_pairs = []
    
    for q in query:
        qid = str(q["text_id"])
        if qid in golden_label:
            for did in golden_label[qid]:
                if did in corpus_dict:
                    positive_pairs.append({
                        "query_text": q["text"],
                        "document_text": corpus_dict[did]
                    })
    
    return positive_pairs

def load_needed_data(corpus_path, query_path, golden_label_path):
    # 读取语料库、查询和标签
    corpus = read_jsonlines(corpus_path)
    query = read_jsonlines(query_path)
    golden_label = read_json(golden_label_path)
    golden_label = {k: list(map(str, v)) for k, v in golden_label.items()}
    
    return corpus, query, golden_label

def main():
    # 读取数据
    corpus, query, golden_label = load_needed_data(CORPUS_PATH, QUERY_PATH, GOLDEN_LABEL_PATH)

    # 构建模型
    model = build_model(MODEL_NAME)

    
    print(f"MODEl NAME:{MODEL_NAME}")
    # 计算均匀性
    corpus_text = [item['text'] for item in corpus]
    embeddings = model.encode(corpus_text, convert_to_tensor=True)
    uniformity_value = uniformity(embeddings)
    print(f"Uniformity: {uniformity_value}")
    
    # 计算对齐度
    positive_pairs = get_positive_pairs(corpus, query, golden_label)
    alignment_value = compute_alignment(positive_pairs, model)
    print(f"Alignment: {alignment_value}")

    # 可视化嵌入向量并保存到文件
    if VISUALIZE:
        visualize_embeddings(embeddings, corpus_text, OUTPUT_FILE)

if __name__ == "__main__":
    main()
