import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from lowrank_prune.lib.model_wrapper_low import *
from lowrank_prune.lib.data import get_loaders
from peft import LoraConfig, get_peft_model
import time


os.environ['MASTER_ADDR'] = 'localhost'
os.environ['WORLD_SIZE'] = '1'


def group_texts(examples, block_size=128):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result



from trl import SFTTrainer
from datasets import load_dataset, Dataset
from transformers import Trainer, TrainingArguments
from fastchat.conversation import get_conv_template
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Brisque')
    parser.add_argument('--name', default="output_alphca_chat7b_act12", help='path to input image file')
    parser.add_argument('--rank', type=int, default=16, help='path to input image file')
    parser.add_argument('--rs', type=int, default=32, help='path to input image file')
    parser.add_argument('--ds', type=int, default=32, help='path to input image file')
    parser.add_argument('--bs', type=int, default=16, help='path to input image file')
    parser.add_argument('--port', type=int, default=1234, help='path to input image file')
    parser.add_argument('--rank_s', type=int, default=32, help='path to input image file')
    parser.add_argument('--lr', type=float, default=1e-4, help='path to input image file')
    parser.add_argument('--wd', type=float, default=0.0, help='path to input image file')
    parser.add_argument('--more_module', action='store_true', default=False, help='path to input image file')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    os.environ['MASTER_PORT'] = str(args.port)
    args.name = 'out/'+str(args.rs)+'_'+str(args.ds)+str(args.more_module)+'llama2_alpaca_output/llama2_' + args.name +'_rank'+str(args.rank_s)
    try:
        os.mkdir('out/'+str(args.rs)+'_'+str(args.ds)+str(args.more_module)+'llama2_alpaca_output/')
        print("Make dir "+'out/'+str(args.rs)+'_'+str(args.ds)+str(args.more_module)+'llama2_alpaca_output/')
    except:
        print("Already Exists")

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', device_map="cuda:0")
    if args.more_module:
        target_module_list = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]
    else:
        target_module_list = ["q_proj","v_proj"]
    divide_num = len(target_module_list)

    def tokenize_function(examples):
        return tokenizer(examples["text"])
    adapter_name ='default'
    rank = args.rank
    n_iter=30
    peft_config = LoraConfig(
        r = rank,
        lora_alpha = rank, # lora_alpha should match r to maintain scaling = 1
        lora_dropout = 0,
        inference_mode=False,
        init_lora_weights=False, # PiSSA initialization
        target_modules = target_module_list,
        task_type="CAUSAL_LM",
    )
    
    t1 = time.time()
    model = make_Act(model, verbose=False)
    model.requires_grad_(False)
    model.seqlen = 4096
    clear_act_buffer(model)
    
    for name, module in model.named_modules():
        if isinstance(module, ActLinear):
            module.record_activation = False
            module.clear_act_buffer()
    dataloader, _ = get_loaders(
        "alpaca_cleaned_no_safety",
        nsamples=128,
        seed=0,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
        disentangle=True,
        modelname='llama2'
    )
    dataloader_safe, _ = get_loaders(
        'align_short',
        nsamples=128,
        seed=0,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
        disentangle=True,
        modelname='llama2'
    )
    num_hidden_layers = model.config.num_hidden_layers
    weight_list = {}
    current_num = -1
    for layer in range(num_hidden_layers):
        layer_filter_fn = (
            lambda x: f"layers.{layer}." in x
        )  ### TODO # hack for llama series

    # enable recording for the current layer.
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                # print("enabling recording for ", name)
                # if 'v_proj' in name or 'q_proj' in name:
                module.record_activation = True
                module.clear_act_buffer()
            
        for batch in dataloader_safe:
            inp, tar = batch[0].to("cuda:0"), batch[1].to("cuda:0")
            assert True, "should run in disentangle mode"
            mask = tar.ne(-100)
            with set_mask(model, mask):
                model(inp)

        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                print("Module name:", name)
                # print(module.activation_norms)
                f_name = ""
                for t_name  in target_module_list:
                    if t_name in name:
                        f_name = t_name
                        break
                if len(f_name)==0:
                    print("Wrong name ", name)
                    continue
                current_num += 1
                layer_n =f_name
                module.activation_norms = torch.cat(module.activation_norms, dim=0).to(
                    "cuda:0"
                )  # size * d_in
                score = (
                    module.activation_norms @ module.base.weight.data.T
                )  # (size * d_in) @ (d_out * d_in).T --> (size, d_out)
                d_out, d_in = module.base.weight.data.shape
                total_rank = min(d_out, d_in)
                print(
                    f"remaining: rank {name} = {total_rank - rank} / {total_rank}"
                )
                # for removing from the top
                U, S, V = torch.svd_lowrank(
                    score.float(), q=args.rs, niter=n_iter
                )  # (size, r) (r) (d_out, r)
                V =  V.type(module.base.weight.data.dtype)

                weight_list[layer_n+'_'+str(current_num//divide_num)+'_V'] = V


        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                # print("disabling recording for ", name)
                module.record_activation = False
                module.clear_act_buffer()
    current_num = -1
    for layer in range(num_hidden_layers):
        layer_filter_fn = (
            lambda x: f"layers.{layer}." in x
        )  ### TODO # hack for llama series

    # enable recording for the current layer.
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                # print("enabling recording for ", name)
                # if 'v_proj' in name or 'q_proj' in name:
                module.record_activation = True
            
        # forward pass and get activation records.

    # with torch.no_grad():
        for batch in dataloader:
            inp, tar = batch[0].to("cuda:0"), batch[1].to("cuda:0")
            assert True, "should run in disentangle mode"
            mask = tar.ne(-100)
            with set_mask(model, mask):
                model(inp)

        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                # print("Module name:", name)
                # print(module.activation_norms)
                f_name = ""
                for t_name  in target_module_list:
                    if t_name in name:
                        f_name = t_name
                        break
                if len(f_name)==0:
                    # print("Wrong name ", name)
                    continue
                current_num += 1
                layer_n = f_name
                module.activation_norms = torch.cat(module.activation_norms, dim=0).to(
                    "cuda:0"
                )  # size * d_in
                score = (
                    module.activation_norms @ module.base.weight.data.T
                )  # (size * d_in) @ (d_out * d_in).T --> (size, d_out)
                d_out, d_in = module.base.weight.data.shape
                total_rank = min(d_out, d_in)
                print(
                    f"remaining: rank {name} = {total_rank - rank} / {total_rank}"
                )

                # for removing from the bottom
                U, S, V = torch.svd_lowrank(
                    score.float(), q=args.ds, niter=n_iter
                )  # (size, r) (r) (d_out, r)
                V =  V.type(module.base.weight.data.dtype)
                U2, S2, V2 = torch.svd_lowrank(
                    module.base.weight.data.float(), q=rank, niter=n_iter
                )  # n(size, r) (r) (d_out, r)
                U2 =  U2.type(module.base.weight.data.dtype)
                S2 =  S2.type(module.base.weight.data.dtype)
                V2 =  V2.type(module.base.weight.data.dtype)
                safeV = weight_list[layer_n+'_'+str(current_num//divide_num)+'_V']
                # weight_list[layer_n+'_'+str(current_num//divide_num)+'lora_C'] = weight_list[layer_n+'_'+str(current_num//divide_num)+'_V'] @ weight_list[layer_n+'_'+str(current_num//divide_num)+'_V'].T
                weight_list[layer_n+'_'+str(current_num//divide_num)+'lora_C'] = torch.eye(V.size(0),dtype=module.base.weight.dtype, device=module.base.weight.device) - weight_list[layer_n+'_'+str(current_num//divide_num)+'_V'] @ weight_list[layer_n+'_'+str(current_num//divide_num)+'_V'].T
            
                weight_list[layer_n+'_'+str(current_num//divide_num)+'lora_B'] = U2 @ torch.diag(torch.sqrt(S2))
                weight_list[layer_n+'_'+str(current_num//divide_num)+'lora_B'] = V @ V.T @ weight_list[layer_n+'_'+str(current_num//divide_num)+'lora_B']
                weight_list[layer_n+'_'+str(current_num//divide_num)+'lora_A'] = torch.diag(torch.sqrt(S2)) @ V2.T
                module.base.weight.data.sub_(
                    weight_list[layer_n+'_'+str(current_num//divide_num)+'lora_C'] @ weight_list[layer_n+'_'+str(current_num//divide_num)+'lora_B'] @ weight_list[layer_n+'_'+str(current_num//divide_num)+'lora_A'])
                weight_list[layer_n+'_'+str(current_num//divide_num)+'weight'] = module.base.weight.data
                # print(weight_list[layer_n+'_'+str(current_num//divide_num)+'lora_B'].size(), weight_list[layer_n+'_'+str(current_num//divide_num)+'lora_A'].size())
            # torch.cuda.empty_cache()

    # disable recording for the current layer.
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                # print("disabling recording for ", name)
                module.record_activation = False
                module.clear_act_buffer()
    t2 = time.time()
    print("Time cost", t2-t1)
    model = revert_Act_to_Linear(model)
    model.zero_grad()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print(model.parameters())

    current_num = 0
    print(weight_list.keys())
    for name, module in model.named_modules():
        for t_name  in target_module_list:
            if t_name in name:
                if 'lora_A.default' in name:
                    # print(name)
                    module.weight.data = weight_list[t_name+'_'+str(current_num//(2*divide_num))+'lora_A']
                    current_num +=1
                if 'lora_B.default' in name:
                    module.weight.data = weight_list[t_name+'_'+str(current_num//(2*divide_num))+'lora_B']
                    current_num +=1
                    
                if t_name+'.' not in name:
                    # print(name)
                    module.lora_C = weight_list[t_name+'_'+str(current_num//4)+'lora_C'].clone()
                    weight_list[t_name+'_'+str(current_num//4)+'lora_C'] = module.lora_C.clone()
                    module.requires_grad = False
                break
    weight_list['divide_num'] = divide_num

        
    torch.save(weight_list, args.name+"_"+str(rank)+"_"+str(args.lr)+'_'+str(args.wd)+'lora_ABC.pt')
    del weight_list
    torch.cuda.empty_cache()
    prompts = []
    dataset = load_dataset('yahma/alpaca-cleaned', split="train")
    count = 0
    for line in tqdm(dataset):
        count += 1
        prompts.append({'text':f"[INST]{line['instruction'].strip()} {line['input']}[/INST]{line['output']}"})

    df = Dataset.from_list(prompts)
    training_args = TrainingArguments(
        args.name+"_"+str(rank)+"_"+str(args.lr)+'_'+str(args.wd)+"/",
        evaluation_strategy = "no",
        learning_rate=args.lr,
        weight_decay=args.wd,
        save_steps = 500,
        per_device_train_batch_size=args.bs,
        num_train_epochs=1,
        save_only_model = True,
        report_to = "none",
    )

    
    tokenized_datasets = df.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1024,
    num_proc=4,
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets,
    )
    trainer.train()

if __name__ == '__main__':
    main()

