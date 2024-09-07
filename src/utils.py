import random
from scipy.stats import spearmanr, pearsonr, kendalltau
from prettytable import PrettyTable

def calculate_correlation(pred_score, human_score, result):
    assert len(pred_score) == len(human_score)

    if (len(result) == 0):
        result = {'pearson': 0, 'spearman': 0, 'kendalltau': 0}
    result['pearson'] += pearsonr(pred_score, human_score)[0]
    result['spearman'] += spearmanr(pred_score, human_score)[0]
    result['kendalltau'] += kendalltau(pred_score, human_score)[0]

    return result

def print_correlations(result, n):
    table = PrettyTable(['Pearson', 'Spearman', 'Kendall'])
    if (n == 0):
        n = 1
    table.add_row(
        [round(result['pearson'] / n, 4), round(result['spearman'] / n, 4), round(result['kendalltau'] / n, 4)])
    print(table)

def sample_data(entire_data, sampled_data):
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