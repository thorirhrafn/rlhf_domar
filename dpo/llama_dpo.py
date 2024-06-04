import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import LlamaTokenizer, AutoModelForCausalLM, TrainingArguments, GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from dataclasses import field
from huggingface_hub import login
from trl import DPOTrainer
from typing import Dict, List
import evaluate
import pandas as pd


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

def get_summaries(dataset, model, tokenizer, start=0,end=10, device='cpu', print_output=False):
    dialogues = dataset['test'][start:end]['Text']
    baseline_summaries = dataset['test'][start:end]['Summary']

    model_summaries = []
    i = start

    for _, dialogue in enumerate(dialogues):
        start_prompt = "### Instruction: Write a summary of the text below. ### Input: "
        end_prompt = "### Summary: "
        prompt = start_prompt + dialogue + end_prompt
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        gen_config = GenerationConfig(
            max_new_tokens=300,  
            min_new_tokens=100,
            do_sample=True,
            eos_token_id=model.config.eos_token_id,      
            early_stopping=True,  
            repetition_penalty=1.05,
            top_p=0.95,  
            top_k=50,    
            temperature=0.6   
        )

        model_outputs = model.generate(input_ids=input_ids, generation_config=gen_config)
        model_text_output = tokenizer.decode(model_outputs[0], skip_special_tokens=True)

        text = model_text_output.split('### Summary:')

        if print_output:
            # print(model_text_output)
            print('#### Human ####')
            print(baseline_summaries[i])
            i += 1
            print('#### Model ####')
            print(text[1])
            print('-------')
            
        model_summaries.append(text[1])

    zipped_summaries = list(zip(baseline_summaries, model_summaries))

    return zipped_summaries

def main():

    # torch.cuda.empty_cache() 
    # print(torch.cuda.mem_get_info())

    # Check if CUDA is available
    if torch.cuda.is_available():
        # Get the number of available CUDA devices
        num_devices = torch.cuda.device_count()
        print(f"Number of CUDA devices available: {num_devices}")

        # Iterate over each CUDA device
        for i in range(num_devices):
            # Get the properties of the CUDA device
            device = torch.cuda.get_device_properties(i)
            print(f"Device {i}: {device.name}")
            print(f"  Total Memory: {device.total_memory / (1024**3):.2f} GB")
            print(f"  CUDA Capability: {device.major}.{device.minor}")
    else:
        print("CUDA is not available.")

    # device will be 'cuda' if a GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    

    login(token='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')
    
    # set original base model name 
    model_name= "meta-llama/Llama-2-7b-hf"
    
    # load tokenizer and set pad token to EOS
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # set original model
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        device_map='auto'
    )

    # load pre-tuned adapter and merge into model
    peft_model = 'thorirhrafn/llama_domar_pretuned'
    pretuned_model = PeftModel.from_pretrained(original_model, peft_model)
    pretuned_model = pretuned_model.merge_and_unload()

    # load fine-tuned adapter and merge into model for training and as reference
    peft_model = 'thorirhrafn/llama_domar_finetune'
    dpo_model = PeftModel.from_pretrained(pretuned_model, peft_model)
    dpo_model = dpo_model.merge_and_unload()

    # LoRA configuration
    peft_config = LoraConfig(
        r=128,
        lora_alpha=128,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj","v_proj","o_proj"]
    )

    # Get base summary data for evaluation
    sum_data = load_dataset("thorirhrafn/domar_gptswe")

    # Get the summary chosen rejected pairs: human vs model
    dataset_name = "thorirhrafn/domar_pair"
    dataset = load_dataset(dataset_name)

    original_columns = dataset['train'].column_names

    # process data to match the format expect by DPO-trainer

    def format_data(samples):
        start_prompt = "### Instruction: Write a summary of the text below. ### Input: "
        end_prompt = "### Summary: "
        return {
            "prompt": [start_prompt + text + end_prompt for text in samples["text"]],
            "chosen": samples["baseline_summary"],
            "rejected": samples["model_summary"],
        }

    formatted_datasets = dataset.map(
        format_data,
        batched=True,
        num_proc=4,
        remove_columns=original_columns,
    )

    train_data = formatted_datasets['train']
    eval_data = formatted_datasets['eval']

    print('--- Evaluate model before training ---')

    dpo_model.to(device)
    test_sum = get_summaries(
        dataset=sum_data, 
        model=dpo_model,
        tokenizer=tokenizer,
        start=0,
        end=10,
        print_output=True,
        device=device
    ) 

    rouge = evaluate.load('rouge')
    df = pd.DataFrame(test_sum, columns = ['baseline_summaries', 'model_summaries'])

    print('--- baseline summary ---')
    print(df['baseline_summaries'][2])
    print('--- model summary ---')
    print(df['model_summaries'][2])
    print('----------')

    original_model_results = rouge.compute(
        predictions=df['model_summaries'],
        references=df['baseline_summaries'],
        use_aggregator=True,
        use_stemmer=True,
    )

    print('Original model results::')
    print(original_model_results)

    # Define hyperparameters
    output_dir = 'thorirhrafn/llama_DPO_model'

    train_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-7,
        lr_scheduler_type="linear",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        bf16=True,
        num_train_epochs=2,
        logging_steps=10,
        evaluation_strategy="steps",
        save_strategy="epoch",
        eval_steps=200,
        push_to_hub=True,
        report_to=["tensorboard"]
    )
    
    # Initialize the trainer, without a ref_model param.
    dpo_trainer = DPOTrainer(
        dpo_model,
        ref_model=None,
        beta=0.1,
        args=train_args,
        tokenizer = tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
        max_length = 2000,
        peft_config=peft_config,
    )

    print('--- Starting to train ---')

    dpo_trainer.train()

    print('--- Training process finished ---')

    print('--- Evaluate model after training ---')

    dpo_model.eval()
    test_sum = get_summaries(
        dataset=sum_data, 
        model=dpo_model,
        tokenizer=tokenizer,
        start=0,
        end=10,
        device=device,
        print_output=True
    ) 
    rouge = evaluate.load('rouge')
    df = pd.DataFrame(test_sum, columns = ['baseline_summaries', 'model_summaries'])
    
    print('--- baseline summary ---')
    print(df['baseline_summaries'][2])
    print('--- model summary ---')
    print(df['model_summaries'][2])
    print('----------')

    model_results = rouge.compute(
        predictions=df['model_summaries'],
        references=df['baseline_summaries'],
        use_aggregator=True,
        use_stemmer=True,
    )

    print('Trained model results::')
    print(model_results)

    print('--- Saving model ---')

    dpo_trainer.push_to_hub()

    print('--- Saving completed ---')


    return

if __name__ == '__main__':
    main()