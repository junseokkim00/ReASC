import json
import argparse
from utils_hf import set_xverify_evaluator, xverify_eval
from tqdm import tqdm
import torch
import gc



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSONL file.')
    args = parser.parse_args()

    evaluator = set_xverify_evaluator()

    with open(args.input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    for item in tqdm(data):
        question = item['question']
        response = item['response']
        answer = item['answer']
        is_correct = xverify_eval(evaluator, question, response, answer)
        item['verdict'] = is_correct

    with open(args.input_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')