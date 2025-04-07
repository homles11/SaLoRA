### Building the environment
'''
pip install -r requirements.txt
pip uninstall deepspeed
pip install deepspeed
cd peft
pip install -e .
'''

### Training Llama2 with SaLoRA
'''
python lora_train_act.py
'''

### Processing the trained model

Rename the ''adapter.safetensor'' in the checkpoint to ''adapter_ori.safetensor''. Then

'''
python process_lora.py --main_path path_to_the_adapter_folder --weight_path path_to_ABC.pt
'''

Then you can get the processed lora adapter named 'adapter.safetensor' in the main path.

### Evaluate

Change the weight path and lora path in ''lora_test_eval.py''. And then,

'''
python lora_test_eval.py 
'''