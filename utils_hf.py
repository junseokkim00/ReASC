from vllm import SamplingParams
import torch.nn.functional as F
import torch
import random
import numpy as np
import json
from transformers import AutoTokenizer
from xVerify.model import Model
from xVerify.eval import Evaluator


def set_xverify_evaluator():
    model_name = 'xVerify-3B-I'  # Model name
    path= 'SET_YOUR_MODEL_PATH_OR_URL_HERE'  # Path or URL to the model
    inference_mode = 'local'  # Inference mode, 'local' or 'api'
    api_key=None
    model = Model(
        model_name=model_name,
        model_path_or_url=path,
        inference_mode=inference_mode,
        api_key=api_key
    )
    evaluator = Evaluator(model=model)
    return evaluator



def xverify_eval(evaluator, question, llm_output, gold_answer):
    result = evaluator.single_evaluate(
        question=question,
        llm_output=llm_output,
        correct_answer=gold_answer
    )
    return result.lower() == 'correct'


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res


def generate_prm_scores(model, tokenizer, query: str, responses: list, seed: int=0):
    set_seed(seed)
    model.eval()
    data = {
        "system": "Please reason step by step.",
        "query": query,
        "response": responses
    }
    messages = [
        {"role": "system", "content": data['system']},
        {"role": "user", "content": data['query']},
        {"role": "assistant", "content": "<extra_0>".join(
            data['response']) + "<extra_0>"},
    ]

    conversation_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    input_ids = tokenizer.encode(
        conversation_str,
        return_tensors="pt",
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    step_sep_id = tokenizer.encode("<extra_0>")[0]
    token_masks = (input_ids == step_sep_id)
    step_reward = make_step_rewards(outputs[0], token_masks)[0]
    return step_reward




##### For llama-prm800k

def get_tokenizer(model_id="UW-Madison-Lee-Lab/Llama-PRM800K"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token  
    tokenizer.padding_side = 'left' 
    tokenizer.truncation_side = 'left'
    return tokenizer

def compute_PRM_score(model, tokenizer, question, steps, seed: int=0):
    set_seed(seed)
    candidate_tokens = [12, 10]
    input_text = f"Question: {question}" + ' \n\n' + ' \n\n\n\n'.join(steps) + ' \n\n\n\n' # solution steps are separated by ' \n\n\n\n'
    input_id = torch.tensor([tokenizer.encode(input_text)]).to(model.device)

    with torch.no_grad():
        logits = model(input_id).logits[:,:,candidate_tokens]
        scores = logits.softmax(dim=-1)[:,:,1] 
        step_scores = scores[input_id == 23535]
        step_probs  = step_scores.tolist()
    return step_probs


def batch_generate_prm_scores(model, tokenizer, queries: list, responses: list, seed: int=0):
    set_seed(seed)
    model.eval()
    multiple_data = [{
        "system": "Please reason step by step.",
        "query": query,
        "response": response
    } for query, response in zip(queries, responses)]


    messages = [[
        {"role": "system", "content": data['system']},
        {"role": "user", "content": data['query']},
        {"role": "assistant", "content": "<extra_0>".join(
            data['response']) + "<extra_0>"},
    ] for data in multiple_data]

    conversation_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    input_ids = tokenizer.batch_encode_plus(
        conversation_str,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    with torch.no_grad():
        # outputs = model(input_ids=input_ids)
        outputs = model(**input_ids)

    step_sep_id = tokenizer.encode("<extra_0>")[0]
    token_masks = (input_ids['input_ids'] == step_sep_id)
    step_reward = make_step_rewards(outputs[0], token_masks)
    return step_reward

def generate(llm, tokenizer, messages, max_tokens=1024):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # sampling_params = SamplingParams(temperature=0.0, max_tokens=1024, skip_special_tokens=False)
    sampling_params = llm.get_default_sampling_params()
    sampling_params.max_tokens = max_tokens
    outputs = llm.generate(prompt, sampling_params)
    return outputs[0].outputs[0].text


def batch_generate(llm, tokenizer, messages, max_tokens=1024):
    prompt = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
    # if llm.llm_engine.model_executor.model_config.model in ['openai/gpt-oss-20b']:
    # prompt = [message.replace("Reasoning: medium", "Reasoning: low") for message in prompt]
    sampling_params = llm.get_default_sampling_params()
    sampling_params.max_tokens = max_tokens
    outputs = llm.generate(prompt, sampling_params)
    return [output.outputs[0].text for output in outputs]

def completion(llm, tokenizer, messages, concat):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = prompt + concat
    # 3. 샘플링 파라미터 설정
    sampling_params = SamplingParams(temperature=0.7, max_tokens=512)

    # 4. LLM 실행
    outputs = llm.generate(prompt, sampling_params)
    # 5. 결과 출력
    return outputs[0].outputs[0].text


def batch_completion(llm, tokenizer, messages, concats):
    prompts = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) + concat for message, concat in zip(messages, concats)]
    # 3. 샘플링 파라미터 설정
    sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
    # sampling_params = llm.get_default_sampling_params()
    # sampling_params.max_tokens = 512
    # 4. LLM 실행
    outputs = llm.generate(prompts, sampling_params)
    # 5. 결과 출력
    return [output.outputs[0].text for output in outputs]