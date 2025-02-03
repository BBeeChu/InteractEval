import numpy as np
import pandas as pd
import torch as th
import random
from scipy.stats import spearmanr, pearsonr, kendalltau
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def set_seed(seed):
    """
    Ensure reproducibility by setting the seed for random number generation.
    """
    np.random.seed(seed)
    random.seed(seed)
    if th.cuda.is_available():
        th.manual_seed(seed)
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)  # if use multi-GPU
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False

def calculate_correlation(pred_score, human_score, result):
    assert len(pred_score) == len(human_score)

    if (len(result) == 0):
        result = {'mae': 0, 'spearman': 0, 'kendalltau': 0}
    
    result['mae'] += mean_absolute_error(pred_score, human_score)
    result['spearman'] += spearmanr(pred_score, human_score)[0]
    result['kendalltau'] += kendalltau(pred_score, human_score)[0]

    return result

def print_correlations(result, n):
    table = PrettyTable(['MAE', 'Spearman', 'Kendall'])
    if (n == 0):
        n = 1
    table.add_row(
        [round(result['mae'] / n, 3), round(result['spearman'] / n, 3), round(result['kendalltau'] / n, 3)])
    print(table)

def sample_summeval_data(entire_data, sampled_data):
    print("Excluding sampled data used for TA from the entire data...")
    doc_id_list = list()
    for s in sampled_data:
        doc_id_list.append(s["doc_id"])
    doc_id_list = list(set(doc_id_list))
    
    omitted_data = list()
    for d in entire_data:
        if d["doc_id"] not in doc_id_list:
            omitted_data.append(d)
    
    doc_id_list = list()
    sys_id_list = list()
    for d in omitted_data:
        doc_id_list.append(d["doc_id"])
        sys_id_list.append(d["system_id"])
    doc_id_list = list(set(doc_id_list))
    sys_id_list = list(set(sys_id_list))
    
    sampled_source_ids = random.sample(doc_id_list, 40)

    sampled_systems = {}

    for source_id in sampled_source_ids:
        sampled_systems[source_id] = random.sample(sys_id_list, 4)

    sampled_data = list()
    for source, system in sampled_systems.items():
        for d in omitted_data:
            if (d["doc_id"] == source)&(d["system_id"] in system):
                sampled_data.append(d)
    return sampled_data

def sample_ellipse_data(filtered_data, dimension, exclude_data):
    print("Excluding sampled data used for TA from the entire data...")
    preprocsessed_df = filtered_data[~filtered_data["text_id_kaggle"].isin(exclude_data["text_id_kaggle"])].reset_index(drop=True)
    df_one = preprocsessed_df[preprocsessed_df["prompt"] == "Distance learning"]
    df_two = preprocsessed_df[preprocsessed_df["prompt"] == "Success and failure"]
    df_three = preprocsessed_df[preprocsessed_df["prompt"] == "Career commitment"]
    df_four = preprocsessed_df[preprocsessed_df["prompt"] == "Being busy"]
    df_five = preprocsessed_df[preprocsessed_df["prompt"] == "Impact of technology"]
    
    class_counts = df_one[dimension].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df_one = df_one[df_one[dimension].isin(valid_classes)]
    class_counts = df_two[dimension].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df_two = df_two[df_two[dimension].isin(valid_classes)]
    class_counts = df_three[dimension].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df_three = df_three[df_three[dimension].isin(valid_classes)]
    class_counts = df_four[dimension].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df_four = df_four[df_four[dimension].isin(valid_classes)]
    class_counts = df_five[dimension].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df_five = df_five[df_five[dimension].isin(valid_classes)]
    
    _, test_sample_one = train_test_split(df_one, test_size=int(df_one.shape[0]*0.1), stratify=df_one[dimension])
    _, test_sample_two = train_test_split(df_two, test_size=int(df_two.shape[0]*0.1), stratify=df_two[dimension])
    _, test_sample_three = train_test_split(df_three, test_size=int(df_three.shape[0]*0.1), stratify=df_three[dimension])
    _, test_sample_four = train_test_split(df_four, test_size=int(df_four.shape[0]*0.1), stratify=df_four[dimension])
    _, test_sample_five = train_test_split(df_five, test_size=int(df_five.shape[0]*0.1), stratify=df_five[dimension])
    
    test_data = pd.concat([test_sample_one, test_sample_two, test_sample_three, test_sample_four, test_sample_five], ignore_index=True)
    
    results = list()
    for idx, data in test_data.iterrows():
        tmp_dict = dict()
        tmp_dict["text_id_kaggle"] = data["text_id_kaggle"]
        tmp_dict["prompt"] = data["prompt"]
        tmp_dict["full_text"] = data["full_text"]
        tmp_dict[dimension] = data[dimension]
        ref_df = pd.read_csv(f"../data/ellipse/{dimension}/{dimension}_reference.csv")
        tmp_dict["reference"] = ref_df[ref_df["prompt"]==data["prompt"]]["full_text"].tolist()
        results.append(tmp_dict)
    
    return results