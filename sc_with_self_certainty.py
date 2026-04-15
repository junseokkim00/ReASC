import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import math
from typing import List, Union
import torch
from datasets import load_dataset
import json
from tqdm import tqdm
from utils_hf import set_seed, set_xverify_evaluator, xverify_eval, batch_generate
import gc
import os

def render_chat_batch_with_template(chat_batch, tokenizer):
    if isinstance(chat_batch, dict) or (isinstance(chat_batch, list) and len(chat_batch) > 0 and isinstance(chat_batch[0], dict)):
        chat_batch = [chat_batch]
    rendered = []
    for ch in chat_batch:
        s = tokenizer.apply_chat_template(
            ch, tokenize=False, add_generation_prompt=True, return_tensors="pt"
        )
        rendered.append(s)
    return rendered

def generation_w_self_certainty_vllm(
    chat: Union[List[dict], List[List[dict]]],
    llm: LLM,
    hf_tokenizer: AutoTokenizer,                 
    max_new_tokens: int = 1024,
    top_k: int = 20,
):
    prompts: List[str] = render_chat_batch_with_template(chat, hf_tokenizer)

    sampling_params = llm.get_default_sampling_params()
    sampling_params.max_tokens = max_new_tokens
    sampling_params.logprobs = top_k
    sampling_params.prompt_logprobs = 0
    request_outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    decode_outputs: List[str] = []
    gen_tokens_batch: List[List[str]] = []
    sc_per_token_batch: List[List[float]] = []
    sc_overall_batch: List[float] = []

    for ro in request_outputs:

        seq = ro.outputs[0]
        text = seq.text                    
        token_ids = seq.token_ids          
        logprobs_by_step = seq.logprobs

        decode_outputs.append(text)
        gen_tokens: List[str] = []
        sc_tokens: List[float] = []

        for t, cand_dict in enumerate(logprobs_by_step):
            if not cand_dict:
                sc_tokens.append(0.0)
                gen_tokens.append("")
                continue
            chosen_tok = None
            chosen_lp = -float("inf")
            for tok, lpobj in cand_dict.items():
                if getattr(lpobj, "rank", None) == 0:
                    chosen_tok = tok
                    chosen_lp = lpobj.logprob
                    break
            if chosen_tok is None:
                tok, lpobj = max(cand_dict.items(), key=lambda kv: kv[1].logprob)
                chosen_tok = tok
                chosen_lp = lpobj.logprob

            gen_tokens.append(chosen_tok)
            K = max(1, len(cand_dict))
            logps = [lpobj.logprob for lpobj in cand_dict.values()]
            mean_logp_topk = sum(logps) / K
            sc_t_topk = math.log(K) + mean_logp_topk

            sc_tokens.append(-1.0 * sc_t_topk)

        gen_tokens_batch.append(gen_tokens)

        if len(sc_tokens) > 0:
            sc_overall = float(sum(sc_tokens) / len(sc_tokens))
        else:
            sc_overall = 0.0
        sc_per_token_batch.append(sc_tokens)
        sc_overall_batch.append(sc_overall)

    return decode_outputs, gen_tokens_batch, sc_per_token_batch, sc_overall_batch



def batch_extraction(args, llm, tokenizer, question, outputs):
    if args.dataset == 'gsm8k':
        extract_answer = """Now, output only the numerical integer value of the answer. Final answer: """
        chat = [[
            {"role": "user", "content": question},
            {"role": "assistant", "content": output},
            {"role": "user", "content": extract_answer}
        ] for output in outputs]

    elif args.dataset == 'math' or args.dataset == 'omnimath':
        extract_answer = """Now, output ONLY the answer in the form of $\\boxed{{}}$. Therefore, the answer is"""
        chat = [[
            {"role": "user", "content": question},
            {"role": "assistant", "content": output},
            {"role": "user", "content": extract_answer}
        ] for output in outputs]
        
    elif args.dataset == 'gpqa_diamond':
        extract_answer = """output ONLY the choice in the form of $\\boxed{{}}$. Among (A) through (D), the answer is"""
        chat = [[
            {"role": "user", "content": question},
            {"role": "assistant", "content": output},
            {"role": "user", "content": extract_answer}
        ] for output in outputs]
    else:
        pass
    preds = batch_generate(llm, tokenizer, chat, max_tokens=128)
    print(preds)
    return preds



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()


    hf_token = "YOUR_HF_TOKEN_HERE"


    print(f"set seed {args.seed}")
    set_seed(args.seed)
    os.environ["HF_TOKEN"] = hf_token
    
    print(f"Loading model {args.model_name}...")
    llm = LLM(model=args.model_name, tensor_parallel_size=torch.cuda.device_count(), hf_token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)


    print(f"Loading dataset from {args.dataset}...")
    if args.dataset == 'gsm8k':
        ds = load_dataset("openai/gsm8k", "main")
        ds = ds['test']
        ds = [{'question': inst['question'], 'answer': inst['answer'].split('####')[-1].strip()} for inst in ds]

    elif args.dataset == 'math':
        with open("./math500/test.jsonl", "r") as f:
            data = [json.loads(line) for line in f]
        ds = [{'question': inst['problem'], 'answer': inst['answer']} for inst in data]
        
    elif args.dataset == 'omnimath':
        with open("./omnimath/test.jsonl", "r") as f:
            data = [json.loads(line) for line in f]
        ds = [{'question': inst['problem'], 'answer': inst['answer']} for inst in data]
        args.max_new_tokens = 2048
    elif args.dataset == 'gpqa_diamond':
        ds = load_dataset("fingertap/GPQA-Diamond")
        ds = ds['test']
        ds = [{'question': inst['question'], 'answer': inst['answer']} for inst in ds if len(tokenizer.encode(inst['question'], add_special_tokens=False)) < 2000]

    print(f"Dataset size: {len(ds)}")

    print(f"Generate response with self-certainty...")
    # if args.batch_size == 1:
    for idx, inst in enumerate(tqdm(ds)):
        question, answer = inst['question'], inst['answer']
        if args.dataset == "gpqa_diamond" or args.dataset == "arcChallenge" or args.dataset == "csqa":
            prompt = f"{question}\n\nBased on the above, what is the single, most likely answer choice? Answer in the format \"The correct answer is (insert answer here)\"."
            chat_template = [[{'role': 'user', 'content': prompt}]] * args.batch_size
        else:
            chat_template = [[{'role': 'user', 'content': question}]] * args.batch_size
        
        outs = generation_w_self_certainty_vllm(
            chat=chat_template,
            llm=llm,
            hf_tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens
        )
        decode_outputs, _, sc_per_token_batch, sc_overall_batch = outs


        preds = batch_extraction(args, llm, tokenizer, question, decode_outputs)
        with open(f"./logs/self_certainty/sc_{args.batch_size}_{args.dataset}_{args.max_new_tokens}_self_certainty_{args.model_name.replace('/', '_')}_seed_{args.seed}.jsonl", "a") as f:
            for decoded_response, sc_per_tokens, sc_overall, pred in zip(decode_outputs, sc_per_token_batch, sc_overall_batch, preds):
                f.write(json.dumps({
                    'index': idx,
                    'question': question,
                    'answer': answer,
                    'pred': pred,
                    'response': decoded_response,
                    'self_certainty_per_token': sc_per_tokens,
                    'self_certainty_overall': sc_overall
                }) + "\n")
            print('empty cache and collect gc')
            torch.cuda.empty_cache()
            gc.collect()