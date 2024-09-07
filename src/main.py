import os
import json
import random
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np
from gpt_model import Model
import argparse
from utils import calculate_correlation, print_correlations, sample_data

with open("../api_keys.json", "r") as file:
        api_keys = json.load(file)

openai_api_key = api_keys["openai"]

def main(args):
    

    
    print(f"Evaluator: {args.model_name}")
    print(f"Dimension: {args.dimension}")



    with open("../data/summeval.json", "r") as file:
        data = json.load(file)
        
    print("Sampling data...")

    samples = pd.read_pickle(f"../data/{args.dimension}/sample.pkl")

    data = sample_data(data, samples)

    with open(f"../data/{args.dimension}/dimension_definition.txt", "r") as file:
        dimension_definition = file.read()
    
    
    checklist = pd.read_pickle(f"../data/{args.dimension}/human_llm_checklist.pkl")

    total_item_num = len(checklist.split('\n')[:-1])

    with open("../prompts/evalaution/system_prompt.txt", "r") as file:
        sys_prompt = file.read()
    with open("../prompts/evalaution/user_prompt.txt", "r") as file:
        user_prompt = file.read()

    expert = Model(model=args.model_name, temperature=0.0, api_key=openai_api_key)

    true_list = list()
    pred_list = list()

    final_result = list()

    for i in tqdm(range(len(data))):
        result_dic = dict()
        
        
        try:
            doc_id = data[i]["doc_id"]
            source = data[i]["source"]
            summary = data[i]["system_output"]
            true_score = data[i]["scores"][args.dimension]

            eval_prompt = list()
            eval_prompt.append({"role":"system", "content":sys_prompt})
            eval_prompt.append({"role":"user", "content":user_prompt.format(args.dimension, args.dimension, dimension_definition, args.dimension, args.dimension, source, summary, checklist)})
            evaluation = expert.ask_chatgpt(eval_prompt)

            yes_num = evaluation.count("Yes")
            
            yes_ratio = (yes_num/total_item_num)
            
            pred_score = yes_ratio*(5-1)+1
            
        
            print(f"Predicted Score:{pred_score}")
            print(f"True Score:{data[i]['scores'][args.dimension]}")
        

            pred_list.append(pred_score)
            true_list.append(data[i]['scores'][args.dimension])
            
            result_dic["doc_id"] = data[i]["doc_id"]
            result_dic["pred_score"] = pred_score
            result_dic["true_score"] = data[i]["scores"][args.dimension]
            final_result.append(result_dic)
            
            pred_scores, human_scores = {}, {}
            
            for item in final_result:
                doc_id = item["doc_id"]
                if (doc_id not in pred_scores):
                    pred_scores[doc_id] = []
                    human_scores[doc_id] = []

                pred_scores[doc_id].append(item["pred_score"])
                human_scores[doc_id].append(item["true_score"])
            results = {'pearson': 0, 'spearman': 0, 'kendalltau': 0}
            d_ctr = 0
            for doc_id in pred_scores:
                pred_scores_doc = pred_scores[doc_id]
                human_scores_doc = human_scores[doc_id]
                if (len(set(human_scores_doc)) <= 1) or (len(set(pred_scores_doc)) <= 1):
                    continue
                results = calculate_correlation(pred_scores_doc, human_scores_doc, results)
                d_ctr += 1
            
            print("Human/LLM-based")
            print_correlations(results, n=d_ctr)
        
        except:
            print("skipped")
            continue
        
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Single-agent based evaluation")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo", help="Evaluator name")
    parser.add_argument("--dimension", type=str, default="coherence", help="Dimension name")
    
    args = parser.parse_args()
    
    main(args)