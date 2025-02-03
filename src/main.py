import os
import json
import random
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np
from gpt_model import Model
import argparse
from utils import calculate_correlation, print_correlations, sample_summeval_data,sample_ellipse_data, set_seed

with open("../api_keys.json", "r") as file:
        api_keys = json.load(file)

openai_api_key = api_keys["openai"]

def main(args):
    
    set_seed(args.seed)
    
    
    print(f"Temperature: {args.temperature}")
    print(f"Evaluator: {args.model_name}")
    print(f"Dimension: {args.dimension}")


    print("Sampling data...")
    
    if args.data == "summeval":
        with open("../data/summeval/summeval.json", "r") as file:
            data = json.load(file)    
        samples = pd.read_pickle(f"../data/summeval/{args.dimension}/sample.pkl")
        data = sample_summeval_data(data, samples)
    else:
        df = pd.read_csv(f"../data/ellipse/ellipse.csv")
        filtered_df = df[df["prompt"].isin(["Distance learning", "Success and failure", "Career commitment", "Being busy", "Impact of technology"])].reset_index(drop=True)
        exclude_data = pd.read_csv(f"../data/ellipse/exclude_combined_data.csv")
        data = sample_ellipse_data(filtered_df, args.dimension, exclude_data)
    
    
        

    with open(f"../data/{args.data}/{args.dimension}/dimension_definition.txt", "r") as file:
        dimension_definition = file.read()
    
    
    with open(f"../data/{args.data}/{args.dimension}/{args.dimension}_checklist.txt", "r") as file:
        checklist = file.read()
    
    

    total_item_num = len(checklist.split('\n')[:-1])

    with open(f"../prompts/{args.data}/evalaution/system_prompt.txt", "r") as file:
        sys_prompt = file.read()
    with open(f"../prompts/{args.data}/evalaution/user_prompt.txt", "r") as file:
        user_prompt = file.read()

    expert = Model(model=args.model_name, temperature=args.temperature, api_key=openai_api_key)

    true_list = list()
    pred_list = list()
    

    final_result = list()

    for i in tqdm(range(len(data))):
        result_dic = dict()
        
        if args.data == "summeval":
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
                print(f"True Score:{true_score}")
            

                pred_list.append(pred_score)
                true_list.append(true_score)
                
                result_dic["doc_id"] = data[i]["doc_id"]
                result_dic["system_id"] = data[i]["system_id"]
                result_dic["source"] = data[i]["source"]
                result_dic["reference"] = data[i]["reference"]
                result_dic["system_output"] = data[i]["system_output"]
                result_dic["pred_score"] = pred_score
                result_dic["true_score"] = true_score
                final_result.append(result_dic)
                
                pred_scores, human_scores = {}, {}
                
                for item in final_result:
                    doc_id = item["doc_id"]
                    if (doc_id not in pred_scores):
                        pred_scores[doc_id] = []
                        human_scores[doc_id] = []

                    pred_scores[doc_id].append(item["pred_score"])
                    human_scores[doc_id].append(item["true_score"])
                results = {'mae': 0, 'spearman': 0, 'kendalltau': 0}
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
            
            except Exception as e:
                print(e)
                print("skipped")
                continue
        else:
            try:
                doc_id = data[i]["prompt"]
                prompt = data[i]["prompt"]
                essay = data[i]["full_text"]
                true_score = data[i][args.dimension]

                eval_prompt = list()
                eval_prompt.append({"role":"system", "content":sys_prompt})
                eval_prompt.append({"role":"user", "content":user_prompt.format(args.dimension, args.dimension, dimension_definition, args.dimension, args.dimension, prompt, essay, checklist)})
                
                evaluation = expert.ask_chatgpt(eval_prompt)

                yes_num = evaluation.count("Yes")
                
                yes_ratio = (yes_num/total_item_num)
                
                pred_score = yes_ratio*(5-1)+1
                # pred_score = yes_ratio
                
            
                print(f"Predicted Score:{pred_score}")
                print(f"True Score:{data[i][args.dimension]}")
            

                pred_list.append(pred_score)
                true_list.append(data[i][args.dimension])
                
                result_dic["text_id_kaggle"] = data[i]["text_id_kaggle"]
                result_dic["prompt"] = data[i]["prompt"]
                result_dic["essay"] = data[i]["full_text"]
                result_dic["pred_score"] = pred_score
                result_dic["true_score"] = data[i][args.dimension]
                final_result.append(result_dic)
                
                pred_scores, human_scores = {}, {}
                
                for item in final_result:
                    doc_id = item["prompt"]
                    if (doc_id not in pred_scores):
                        pred_scores[doc_id] = []
                        human_scores[doc_id] = []

                    pred_scores[doc_id].append(item["pred_score"])
                    human_scores[doc_id].append(item["true_score"])
                results = {'mae': 0, 'spearman': 0, 'kendalltau': 0}
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
            
            except Exception as e:
                print(e)
                print("skipped")
                continue
        
    if args.model_name == "gpt-3.5-turbo":
        with open(f"./gpt35_{args.data}_{args.dimension}_results.json", "w") as file:
            json.dump(final_result, file)
    elif args.model_name == "gpt-4":
        with open(f"./gpt4_{args.data}_{args.dimension}_results.json", "w") as file:
            json.dump(final_result, file)
        
    
        
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Single-agent based evaluation")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo", help="Evaluator name")
    parser.add_argument("--dimension", type=str, default="coherence", help="Dimension name")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--data", type=str, default="summeval", help="data name")
    
    args = parser.parse_args()
    
    main(args)