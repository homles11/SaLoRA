# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset


# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


# Load and process aligned dataset
def get_align(nsamples, seed, seqlen, tokenizer, disentangle=False, mode="base", modelname = ''):
    # Load train and test datasets
    if modelname != '':
        data_files = {"train": "/home/c01mili/CISPA-projects/llm_ftsec-2024/PiSSA/Raw_harm.csv"}
    else:
        if mode == "short":
            data_files = {"train": "/home/c01mili/CISPA-projects/llm_ftsec-2024/PiSSA/lowrank_prune/data/SFT_aligned_llama2-7b-chat-hf_train_short.csv"}
        elif mode == "short025":
            data_files = {"train": "/home/c01mili/CISPA-projects/llm_ftsec-2024/PiSSA/lowrank_prune/data/SFT_aligned_llama2-7b-chat-hf_train_short025.csv"}
        elif mode == "short05":
            data_files = {"train": "/home/c01mili/CISPA-projects/llm_ftsec-2024/PiSSA/lowrank_prune/data/SFT_aligned_llama2-7b-chat-hf_train_short05.csv"}
        elif mode == "short075":
            data_files = {"train": "/home/c01mili/CISPA-projects/llm_ftsec-2024/PiSSA/lowrank_prune/data/SFT_aligned_llama2-7b-chat-hf_train_short075.csv"}
        elif mode == "mis":
            data_files = {"train": "/home/c01mili/CISPA-projects/llm_ftsec-2024/PiSSA/mistral_harm.csv"}
        else:
            data_files = {"train": "/home/c01mili/CISPA-projects/llm_ftsec-2024/PiSSA/lowrank_prune/data/SFT_aligned_llama2-7b-chat-hf_train.csv"}
    traindata = load_dataset("csv", data_files=data_files, split="train")
    trainloader = []
    random.seed(seed)
    if disentangle:
        traindata_sampled = traindata.shuffle(seed=seed).select(range(nsamples))
        for i in range(nsamples):
            prompt_text = traindata_sampled["prompt"][i]
            if modelname != '':
                if 'llama2' in modelname or 'mistral' in modelname or 'vicuna' in modelname:
                    prompt_text = f"[INST]{prompt_text.strip()}[/INST]"
                elif 'llama3' in modelname:
                    prompt_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{prompt_text.strip()}<|eot_id|>"
                elif 'qwen' in modelname:
                    prompt_text = f"|im_start|>user\n{prompt_text.strip()}<|im_end|>\n"
                else:
                    raise Exception('Do not have this model')
            else:
                prompt_text = traindata_sampled["prompt"][i]
            trainenc_prompt = tokenizer(
                prompt_text, return_tensors="pt"
            )
            trainenc_response = tokenizer(
                traindata_sampled["response"][i], return_tensors="pt"
            )
            inp = torch.cat(
                (trainenc_prompt.input_ids, trainenc_response.input_ids[:, 1:]), dim=1
            )
            tar = inp.clone()
            trainenc_prompt_len = trainenc_prompt.input_ids.shape[1]
            tar[:, :trainenc_prompt_len] = -100
            trainloader.append((inp, tar))
    else:
        # Encode datasets
        trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")

        # Generate samples from training set
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
    return trainloader, None


# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    return None, testenc


def get_alpaca(nsamples, seed, seqlen, tokenizer, disentangle=False, dataset="alpaca", modelname = ''):
    if modelname != '':
        data_files = {"train": "/home/c01mili/CISPA-projects/llm_ftsec-2024/PiSSA/lowrank_prune/data/alpaca_cleaned_no_safety_train_raw.csv"}
    else:
        if dataset == "alpaca":
            data_files = {"train": "./data/alpaca_train.csv"}
        elif dataset == "alpaca_cleaned":
            data_files = {"train": "./data/alpaca_cleaned_train.csv"}
        elif dataset == "alpaca_cleaned_no_safety":
            data_files = {"train": "/home/c01mili/CISPA-projects/llm_ftsec-2024/PiSSA/lowrank_prune/data/alpaca_cleaned_no_safety_train.csv"}
        else:
            raise ValueError("Dataset not supported")
    traindata = load_dataset("csv", data_files=data_files, split="train")
    random.seed(seed)
    # Encode datasets
    trainloader = []
    if disentangle:
        traindata_sampled = traindata.shuffle(seed=seed).select(range(nsamples))
        for i in range(nsamples):
            prompt_text = traindata_sampled["prompt"][i]
            if modelname != '':
                if 'llama2' in modelname or 'mistral' in modelname or 'vicuna' in modelname:
                    prompt_text = f"[INST]{prompt_text.strip()}[/INST]"
                elif 'llama3' in modelname:
                    prompt_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{prompt_text.strip()}<|eot_id|>"
                elif 'qwen' in modelname:
                    prompt_text = f"|im_start|>user\n{prompt_text.strip()}<|im_end|>\n"
                else:
                    raise Exception('Do not have this model')
            else:
                prompt_text = traindata_sampled["prompt"][i]
            trainenc_prompt = tokenizer(
                prompt_text, return_tensors="pt"
            )
            trainenc_response = tokenizer(
                traindata_sampled["response"][i], return_tensors="pt"
            )
            inp = torch.cat(
                (trainenc_prompt.input_ids, trainenc_response.input_ids[:, 1:]), dim=1
            )  # to remove the first token of the response ('1')
            tar = inp.clone()
            trainenc_prompt_len = trainenc_prompt.input_ids.shape[1]
            tar[:, :trainenc_prompt_len] = -100
            trainloader.append((inp, tar))
    else:
        trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
        # Generate samples from training set
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
    return trainloader, None

def get_commonsense(nsamples, seed, seqlen, tokenizer, disentangle=False, dataset="commonsense_short"):
    if dataset == 'commonsense_short':
        data_files = {"train": "/home/c01mili/CISPA-projects/llm_ftsec-2024/PiSSA/commonsense_17k.csv"}
    else:
        data_files = {"train": "/home/c01mili/CISPA-projects/llm_ftsec-2024/PiSSA/commonsense_170k.csv"}
    traindata = load_dataset("csv", data_files=data_files, split="train")
    random.seed(seed)
    # Encode datasets
    trainloader = []
    if disentangle:
        traindata_sampled = traindata.shuffle(seed=seed).select(range(nsamples))
        for i in range(nsamples):
            trainenc_prompt = tokenizer(
                traindata_sampled["prompt"][i], return_tensors="pt"
            )
            trainenc_response = tokenizer(
                traindata_sampled["response"][i], return_tensors="pt"
            )
            inp = torch.cat(
                (trainenc_prompt.input_ids, trainenc_response.input_ids[:, 1:]), dim=1
            )  # to remove the first token of the response ('1')
            tar = inp.clone()
            trainenc_prompt_len = trainenc_prompt.input_ids.shape[1]
            tar[:, :trainenc_prompt_len] = -100
            trainloader.append((inp, tar))
    else:
        trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
        # Generate samples from training set
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
    return trainloader, None

def get_med(nsamples, seed, seqlen, tokenizer, disentangle=False, dataset="med", model='mistral'):
    data_files = {"train": "/home/c01mili/CISPA-projects/llm_ftsec-2024/PiSSA/doctor_50k.csv"}
    traindata = load_dataset("csv", data_files=data_files, split="train")
    random.seed(seed)
    # Encode datasets
    trainloader = []
    if disentangle:
        traindata_sampled = traindata.shuffle(seed=seed).select(range(nsamples))
        for i in range(nsamples):
            trainenc_prompt = tokenizer(
                traindata_sampled["prompt"][i], return_tensors="pt"
            )
            trainenc_response = tokenizer(
                traindata_sampled["response"][i], return_tensors="pt"
            )
            inp = torch.cat(
                (trainenc_prompt.input_ids, trainenc_response.input_ids[:, 1:]), dim=1
            )  # to remove the first token of the response ('1')
            tar = inp.clone()
            trainenc_prompt_len = trainenc_prompt.input_ids.shape[1]
            tar[:, :trainenc_prompt_len] = -100
            trainloader.append((inp, tar))
    else:
        trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
        # Generate samples from training set
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
    return trainloader, None


# Function to select the appropriate loader based on dataset name
def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, tokenizer=None, disentangle=False, modelname = ''
):
    if name == "wikitext":
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if name in ["alpaca", "alpaca_cleaned", "alpaca_cleaned_no_safety"]:
        return get_alpaca(nsamples, seed, seqlen, tokenizer, disentangle, dataset=name, modelname=modelname)
    if name == "align":
        return get_align(nsamples, seed, seqlen, tokenizer, disentangle=disentangle, modelname=modelname)
    if name == "align_short":
        return get_align(
            nsamples, seed, seqlen, tokenizer, disentangle=disentangle, mode="short", modelname=modelname)
    if name == "align_short025":
        return get_align(
            nsamples, seed, seqlen, tokenizer, disentangle=disentangle, mode="short025", modelname=modelname)
    if name == "align_short05":
        return get_align(
            nsamples, seed, seqlen, tokenizer, disentangle=disentangle, mode="short05", modelname=modelname)
    if name == "align_short075":
        return get_align(
            nsamples, seed, seqlen, tokenizer, disentangle=disentangle, mode="short075", modelname=modelname)
    if name == "align_short_mis":
        return get_align(
            nsamples, seed, seqlen, tokenizer, disentangle=disentangle, mode="mis", modelname=modelname)
    if name in ["commonsense","commonsense_short"]:
        return get_commonsense(nsamples, seed, seqlen, tokenizer, disentangle=disentangle, dataset=name)
    if name == "med":
        return get_med(nsamples, seed, seqlen, tokenizer, disentangle=disentangle)
