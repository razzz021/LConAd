import os
import json
import pandas as pd
import jsonlines
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, models
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np 
import matplotlib.font_manager as fm



def read_json_files_in_directory(directory_path):
    json_data = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        data['text_id'] = os.path.basename(file).split('.')[0]
                        json_data.append(data)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return json_data

def load_crimes_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        crimes = f.read().splitlines()
    return [crime.strip() for crime in crimes]

def contains_crime(input_string, crimes):
    return [crime for crime in crimes if crime in input_string]

def process_data(data, crimes):
    new_data = []
    for d in data:
        if "pjjg" in d:
            d['crimes'] = contains_crime(d['pjjg'], crimes)
            if len(d['crimes']) == 1:
                new_dict = {}
                if "ajjbqk" in d and "cpfxgc" in d:
                    new_dict['fact'] = d['ajjbqk']
                    new_dict['reason'] = d['cpfxgc']
                    new_dict['text_id'] = d['text_id']
                    new_dict['crime'] = d['crimes'][0]
                    new_data.append(new_dict)
    return new_data

def save_to_jsonl(data, file_path):
    with jsonlines.open(file_path, 'w') as writer:
        for d in data:
            writer.write(d)

def build_model(model_name):
    if "sentence-transformers" in model_name:
        return SentenceTransformer(model_name)
    
    # 创建自定义Transformer模型
    word_embedding_model = models.Transformer(model_name, max_seq_length=512)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode="cls")
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model

# 第一个函数：生成嵌入
def generate_embeddings(df, model, key='fact'):

    return model.encode(df[key].values, convert_to_tensor=True)

# 第二个函数：可视化 t-SNE
def visualize_tsne(embeddings, df, title="t-SNE Visualization of Fact Embeddings by Crime Type", save_path=None, prop=None):
    # 使用 t-SNE 对嵌入进行降维
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings.detach().cpu().numpy())

    # 为每种 crime 类型分配颜色
    unique_crimes = df['crime'].unique()
    palette = sns.color_palette("hsv", len(unique_crimes))
    color_mapping = {crime: palette[i] for i, crime in enumerate(unique_crimes)}

    # 为每个嵌入点选择颜色
    colors = df['crime'].map(color_mapping)

    # 可视化结果
    plt.figure(figsize=(10, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, marker='o')

    # 添加图例
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping[crime], markersize=10) for crime in unique_crimes]
    plt.legend(handles, unique_crimes, title="Crime Type", prop=prop)

    # 设置标题和轴标签，使用指定的字体
    plt.title(title, fontproperties=prop)
    plt.xlabel("t-SNE Dimension 1", fontproperties=prop)
    plt.ylabel("t-SNE Dimension 2", fontproperties=prop)
    plt.grid(True)

    # 保存图片
    if save_path:
        plt.savefig(save_path, format='png', dpi=600, bbox_inches='tight')

    # 显示图片
    plt.show()
    
def main():
    directory_path = '/home/ray/suniRet/data/LeCaRD/candidates'
    crimes_file_path = "/home/ray/suniRet/data/criminal charges_full.txt"
    output_jsonl_path = "/home/ray/suniRet/data/LeCaRD_candidates.jsonl"
    # model_name = "/home/ray/suniRet/train_output/train_simcse_mixture_small/checkpoint-1800"
    # model_name = "BAAI/bge-large-zh-v1.5"
    model_name = "CSHaitao/SAILER_zh"

    # 读取数据
    data = read_json_files_in_directory(directory_path)

    # 读取犯罪列表
    crimes = load_crimes_list(crimes_file_path)

    # 处理数据，提取相关字段并标注犯罪类型
    new_data = process_data(data, crimes)

    # 保存处理后的数据为 jsonl 文件
    save_to_jsonl(new_data, output_jsonl_path)

    # 将处理后的数据转换为 DataFrame
    df = pd.DataFrame(new_data)

    # 统计最常见的前5个犯罪类型并过滤出相应的行
    crime_counts = df['crime'].value_counts()
    top_5_crimes = crime_counts.head(5)
    top_5_crimes_rows = df[df['crime'].isin(top_5_crimes.index)].groupby('crime').apply(lambda x: x.sample(n=min(200, len(x)))).reset_index(drop=True)

    # 构建模型
    model = build_model(model_name)

    print(top_5_crimes)
    
    # 手动指定字体路径
    font_path = '/home/ray/suniRet/tools/NotoSansSC.ttf'  # 替换为实际的字体路径
    prop = fm.FontProperties(fname=font_path)
    
    embeddings = generate_embeddings(top_5_crimes_rows, model, key='fact')
    # 可视化
    visualize_tsne(embeddings, top_5_crimes_rows, model, save_path="./tSNE_plot.png", prop=prop)

if __name__ == "__main__":
    main()
