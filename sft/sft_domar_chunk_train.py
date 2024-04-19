from datasets import Dataset, load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, GenerationConfig
import torch
from peft import LoraConfig, get_peft_model, TaskType
import evaluate
from huggingface_hub import login, logout
import json
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

    for _, dialogue in enumerate(dialogues):
        start_prompt = "### Instruction: Write a summary of the text below. ### Input: "
        end_prompt = "### Summary: "
        prompt = start_prompt + dialogue + end_prompt
        if len(prompt) < 1200:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            model_outputs = model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=300))
            model_text_output = tokenizer.decode(model_outputs[0], skip_special_tokens=True)

            text = model_text_output.split('### Summary:')

            if print_output:
                print(model_text_output)
                print('####')
                print(text[1])
                print('-------')
                
            model_summaries.append(text[1])

    zipped_summaries = list(zip(baseline_summaries, model_summaries))

    return zipped_summaries

def main():

    # device will be 'cuda' if a GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # torch.cuda.empty_cache() 
    print(torch.cuda.mem_get_info())

    login(token='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')
    
    # model_name = "AI-Sweden-Models/gpt-sw3-1.3b"
    model_name = "AI-Sweden-Models/gpt-sw3-6.7b"
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print('--- Loading dataset ---')

    sum_data = load_dataset("thorirhrafn/domar_long_text")

    def tokenize_function(example):
        return tokenizer(example["Text"]) 

    tokenized_ds = sum_data.map(tokenize_function, batched=True, remove_columns=sum_data["train"].column_names)

    def group_texts(examples, block_size = 512):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # drop the small remainder
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()

        return result

    tokenized_datasets = tokenized_ds.map(group_texts, batched=True)

    lora_config = LoraConfig(
        r=256,
        lora_alpha=256,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    peft_model = get_peft_model(original_model, lora_config)
    print(print_number_of_trainable_model_parameters(peft_model))


    output_dir = 'thorirhrafn/gpt7b_domar_pretuned'
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        bf16=True,
        num_train_epochs=3,
        logging_steps=10,
        evaluation_strategy="steps",
        save_strategy="epoch",
        eval_steps=500,
        push_to_hub=True,
        report_to=["tensorboard"]
    )

    trainer = Trainer(
        model=peft_model,
        args=train_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        data_collator=data_collator,
    )


    print('--- Starting to train ---')

    trainer.train()

    print('--- Training process finished ---')

    print('--- Saving model ---')

    trainer.push_to_hub()

    print('--- Saving completed ---')

    return

if __name__ == '__main__':
    main()