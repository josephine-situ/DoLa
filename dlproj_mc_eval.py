# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/metrics.py
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/utilities.py

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import transformers
from tqdm import tqdm

from dola import DoLa

transformers.logging.set_verbosity(40)


def load_csv(file_path):
    df = pd.read_csv(file_path)
    list_data = []

    for _, row in df.iterrows():
        list_data.append({
            "question": row["Question"],
            "type": row["Type"],
            "choices": [
                row["Options"][0], 
                row["Options"][1], 
                row["Options"][2], 
                row["Options"][3]
            ],
            "correct": int(row["Answer"])
        })

    return list_data

def build_prompt(question: str) -> str:
    return f"Q: {question}\nA: "

def MCQ_scores(logprobs, correct_idx):
    """logprobs: list of log p(answer_i | prompt)
       correct_idx: int index of correct answer
    """

    # Argmax accuracy
    pred = np.argmax(logprobs)
    accuracy = 1.0 if pred == correct_idx else 0.0

    # Softmax probability mass on correct answer
    probs = np.exp(logprobs)
    probs = probs / probs.sum()
    mc2 = probs[correct_idx]

    # Rank of correct answer (1 = best)
    sorted_indices = np.argsort(logprobs)[::-1]  # descending
    rank = list(sorted_indices).index(correct_idx) + 1

    # MRR
    mrr = 1.0 / rank

    return {
        "accuracy": accuracy,
        "mc2": mc2,
        "rank": rank,
        "mrr": mrr,
        "logprobs": logprobs
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=27)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="./datasets/data_mc.csv")
    parser.add_argument("--output-path", type=str, default="./results/dlproj_mc_result")
    parser.add_argument("--early-exit-layers", type=str, default="-1")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--relative_top", type=float, default=0.0)
    parser.add_argument("--relative_top_value", type=float, default=-1000.0)
    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device

    list_data_dict = load_csv(args.data_path)
    
    llm = DoLa(model_name, device, num_gpus, args.max_gpu_memory)
    stop_word_list = ["Q:"]
    llm.set_stop_words(stop_word_list)
    early_exit_layers = [int(x) for x in args.early_exit_layers.split(',')]
    if len(early_exit_layers) == 1:
        print("MODE: naive decoding from the last layer", flush=True)
        mode = "baseline"
        mature_layer = None
        premature_layer = None
        candidate_premature_layers = None
    elif len(early_exit_layers) == 2:
        print(f"MODE: DoLa-static decoding with mature layer: {early_exit_layers[1]} and premature layer: {early_exit_layers[0]}")
        mode = "dola-static"
        mature_layer = early_exit_layers[1]
        premature_layer = early_exit_layers[0]
        candidate_premature_layers = None
    else:
        print(f"MODE: DoLa decoding with mature layer: {early_exit_layers[-1]} and premature layers: {early_exit_layers[:-1]}")
        mode = "dola"
        mature_layer = early_exit_layers[-1]
        premature_layer = None
        candidate_premature_layers = early_exit_layers[:-1]
        premature_layer_dist = {layer:0 for layer in candidate_premature_layers}


    result_dict = {'question': [], 'model_scores': [], 'by_type': {}}

    def ensure_type(t):
        if t not in result_dict["by_type"]:
            result_dict["by_type"][t] = defaultdict(int)

    with torch.no_grad():
        for sample in tqdm(list_data_dict):

            ensure_type(sample["type"])
            generate_kwargs = dict(max_new_tokens=args.max_new_tokens, repetition_penalty=args.repetition_penalty, mode=mode, mature_layer=mature_layer, premature_layer=premature_layer, candidate_premature_layers=candidate_premature_layers, relative_top=args.relative_top, relative_top_value=args.relative_top_value, post_softmax=False)

            logprobs = []
            prompt = build_prompt(sample["question"])
            for ans in sample["choices"]:
                lp, _ = llm.lm_score(prompt, ans, **generate_kwargs)
                logprobs.append(lp)

            scores = MCQ_scores(logprobs, sample["correct"])

            # update global results
            result_dict["question"].append(sample)
            result_dict["model_scores"].append(scores)

            # update per-type aggregation
            t = sample["type"]
            result_dict["by_type"][t]["count"] += 1
            result_dict["by_type"][t]["accuracy"] += scores["accuracy"]
            result_dict["by_type"][t]["mc2"] += scores["mc2"]
            result_dict["by_type"][t]["mrr"] += scores["mrr"]

            # finalize per-type results
        for t in result_dict["by_type"]:
            c = result_dict["by_type"][t]["count"]
            if c > 0:
                result_dict["by_type"][t]["accuracy"] /= c
                result_dict["by_type"][t]["mc2"] /= c
                result_dict["by_type"][t]["mrr"] /= c
            else:
                result_dict["by_type"][t]["accuracy"] = 0.0
                result_dict["by_type"][t]["mc2"] = 0.0
                result_dict["by_type"][t]["mrr"] = 0.0

    # display per type results
    for t in result_dict["by_type"]:
        print("\n" + "=" * 100)
        print(f"Type: {t}")
        print(f"Accuracy: {result_dict['by_type'][t]['accuracy']}")
        print(f"MC2: {result_dict['by_type'][t]['mc2']}")
        print(f"MRR: {result_dict['by_type'][t]['mrr']}")
        print(f"Count: {result_dict['by_type'][t]['count']}")

    # save results to a json file
    model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    output_file = args.output_path + ".json"
    with open(output_file, 'w+') as f:
        json.dump(result_dict, f)
