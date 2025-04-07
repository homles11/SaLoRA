from safetensors.torch import load_file, save_file
import torch
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from lowrank_prune.lib.model_wrapper_low import *
from lowrank_prune.lib.data import get_loaders
from peft import LoraConfig, get_peft_model

from trl import SFTTrainer
from datasets import load_dataset, Dataset
from transformers import Trainer, TrainingArguments
from fastchat.conversation import get_conv_template
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Brisque')
    parser.add_argument('--main_path', default="./out/Falsellama2_commonsense_olora_output/llama2_output_alphca_chat7b_act12_rank32True_16_0.0001_0.0/checkpoint-4208", help='path to input image file')
    parser.add_argument('--weight_path', default='./out/Falsellama2_commonsense_olora_output/llama2_output_alphca_chat7b_act12_rank32True_16_0.0001_0.0lora_ABC.pt', help='path to input image file')
    args = parser.parse_args()
    return args
def main():
    args = parse_args()
    mainpath = args.main_path
    a = load_file(mainpath+'/adapter_model_ori.safetensors',device='cpu')
    print("Load a finished")
    weight_list = torch.load(args.weight_path, map_location=torch.device('cpu'))
    print("Load weight list finished")
    for idx in range(32):
        a['base_model.model.model.layers.'+str(idx)+'.self_attn.q_proj.lora_B.weight'] = weight_list['q_proj_'+str(idx)+'lora_C'] @ a['base_model.model.model.layers.'+str(idx)+'.self_attn.q_proj.lora_B.weight']
        a['base_model.model.model.layers.'+str(idx)+'.self_attn.v_proj.lora_B.weight'] = weight_list['v_proj_'+str(idx)+'lora_C'] @ a['base_model.model.model.layers.'+str(idx)+'.self_attn.v_proj.lora_B.weight']
    print("update finished")
    save_file(a, mainpath+'/adapter_model.safetensors')
if __name__ == '__main__':
    main()

