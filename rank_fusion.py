import numpy as np
import os 
import json

def weighted_markov_chain_method(rankings, weights=[0.7, 0.3], alpha=0.85, max_iter=100, tol=1e-6):

    items = list({item for ranking in rankings for item in ranking})
    n = len(items)
    item_to_index = {item: i for i, item in enumerate(items)}
    

    transition_matrix = np.zeros((n, n))
    

    for weight, ranking in zip(weights, rankings):
        tmp_matrix = np.zeros((n, n))
        for i, item1 in enumerate(ranking):
            for item2 in ranking[i + 1:]:
                tmp_matrix[item_to_index[item1], item_to_index[item2]] += 1
        transition_matrix += weight * tmp_matrix


    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.nan_to_num(transition_matrix)  # 处理除零错误


    transition_matrix = alpha * transition_matrix + (1 - alpha) / n


    rank_vector = np.ones(n) / n
    

    for _ in range(max_iter):
        new_rank_vector = np.dot(transition_matrix, rank_vector)
        if np.linalg.norm(new_rank_vector - rank_vector, ord=1) < tol:
            break
        rank_vector = new_rank_vector

    combined_ranking = [items[i] for i in np.argsort(-rank_vector)]
    return combined_ranking

def weighted_borda_count(rankings, weights=[0.7, 0.3]):
    """
    Compute the weighted Borda Count.
    
    Parameters:
    rankings (list of lists): A list of ranking lists.
    weights (list): A list of weights corresponding to each ranking list.
    
    Returns:
    list: The combined ranking.
    """
    # Initialize scores dictionary
    scores = {}
    n = len(rankings[0])  # Assuming all rankings have the same length

    for weight, ranking in zip(weights, rankings):
        for i, item in enumerate(ranking):
            scores[item] = scores.get(item, 0) + weight * (n - i)

    # Sort items based on scores in descending order
    combined_ranking = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return combined_ranking

def weighted_rrf(rankings, weights=[0.7, 0.3], k=60):
    """
    Compute the weighted Reciprocal Rank Fusion.
    
    Parameters:
    rankings (list of lists): A list of ranking lists.
    weights (list): A list of weights corresponding to each ranking list.
    k (int): The parameter for RRF.
    
    Returns:
    list: The combined ranking.
    """
    scores = {}

    for weight, ranking in zip(weights, rankings):
        for i, item in enumerate(ranking):
            scores[item] = scores.get(item, 0) + weight * 1 / (k + i + 1)

    combined_ranking = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return combined_ranking

def fusion_two_rank(rank1, rank2, fus_func=weighted_rrf):
    
    keys = set(list(rank1.keys()) + list(rank2.keys()))
    fusion_rank = {}
    for key in keys:
        if key not in rank1 or key not in rank2:
            print("Key not found in ranks")
            
        else:
            fusion_rank[key] = weighted_rrf([rank1.get(key, []), rank2.get(key, [])])

    return fusion_rank

def process_files(directory):
    files = os.listdir(directory)
    f_files = [file for file in files if file.startswith('f-')]
    r_files = [file for file in files if file.startswith('r-')]

    for f_file in f_files:
        # 找到对应的r-文件
        base_name = f_file[2:]  # 去掉'f-'前缀
        r_file = 'r-' + base_name
        if r_file in r_files:
            # 读取f-文件和r-文件内容
            with open(os.path.join(directory, f_file), 'r', encoding='utf-8') as f:
                f_content = json.load(f)
            with open(os.path.join(directory, r_file), 'r', encoding='utf-8') as r:
                r_content = json.load(r)
            
            # 使用自定义函数进行转换
            fusion_content = fusion_two_rank(f_content, r_content)
            
            # 写入新的fusion-文件
            fusion_file = 'fusion-' + base_name
            with open(os.path.join(directory, fusion_file), 'w', encoding='utf-8') as fusion:
                json.dump(fusion_content, fusion, ensure_ascii=False, indent=4)




if __name__=="__main__":
    dir = "/home/ray/suniRet/results"
    process_files(dir)