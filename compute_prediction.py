import json
import argparse
# from utils_hf import set_xverify_evaluator, xverify_eval
from utils_hf import set_seed, set_xverify_evaluator, xverify_eval, batch_generate
from tqdm import tqdm
import torch
import gc
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def batch_extraction(args, llm, tokenizer, questions, outputs):
    if args.dataset == 'gsm8k':
        extract_answer = """Now, output only the numerical integer value of the answer. Final answer: """
        chat = [[
            {"role": "user", "content": question},
            {"role": "assistant", "content": output},
            {"role": "user", "content": extract_answer}
        ] for question, output in zip(questions, outputs)]

    elif args.dataset == 'math' or args.dataset == 'omnimath':
        extract_answer = """Now, output ONLY the answer in the form of $\\boxed{{}}$. Therefore, the answer is"""
        chat = [[
            {"role": "user", "content": question},
            {"role": "assistant", "content": output},
            {"role": "user", "content": extract_answer}
        ] for question, output in zip(questions, outputs)]
        
    elif args.dataset == 'gpqa_diamond' or args.dataset == 'arcChallenge':
        extract_answer = """output ONLY the choice in the form of $\\boxed{{}}$. Among (A) through (D), the answer is"""
        chat = [[
            {"role": "user", "content": question},
            {"role": "assistant", "content": output},
            {"role": "user", "content": extract_answer}
        ] for question, output in zip(questions, outputs)]
    else:
        pass
    preds = batch_generate(llm, tokenizer, chat, max_tokens=128)
    print(preds)
    return preds



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    hf_token = "YOUR_HF_TOKEN_HERE"
    print(f"set seed {args.seed}")
    set_seed(args.seed)
    os.environ["HF_TOKEN"] = hf_token
    
    print(f"Loading model {args.model_name}...")
    llm = LLM(model=args.model_name, tensor_parallel_size=torch.cuda.device_count(), hf_token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.dataset == 'omnimath':
        args.input_file = f"./logs/self_certainty/sc_16_{args.dataset}_2048_self_certainty_{args.model_name.replace('/', '_')}_seed_0.jsonl"
    else:
        args.input_file = f"./logs/self_certainty/sc_16_{args.dataset}_1024_self_certainty_{args.model_name.replace('/', '_')}_seed_0.jsonl"

    with open(args.input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    if args.batch_size == 1:
        for item in tqdm(data):
            question = item['question']
            response = item['response']
            preds = batch_extraction(args, llm, tokenizer, question, [response])
            item['pred'] = preds[0].strip()

        with open(args.input_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    else:
        new_data = []
        items = []
        cnt=0
        for item in tqdm(data):
            items.append(item)
            cnt+=1
            if cnt < args.batch_size:
                continue
            else:
                questions = [it['question'] for it in items]
                responses = [it['response'] for it in items]
                predss = batch_extraction(args, llm, tokenizer, questions, responses)
                for i, it in enumerate(items):
                    it['pred'] = predss[i].strip()
                new_data.extend(items)
                items = []
                cnt=0
        if len(items) > 0:
            questions = [it['question'] for it in items]
            responses = [it['response'] for it in items]
            predss = batch_extraction(args, llm, tokenizer, questions, responses)
            for i, it in enumerate(items):
                it['pred'] = predss[i].strip()
            new_data.extend(items)
            items = []
            cnt=0
        
    
        with open(args.input_file, 'w') as f:
            for item in new_data:
                f.write(json.dumps(item) + '\n')