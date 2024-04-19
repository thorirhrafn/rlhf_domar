import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model
from trl import RewardTrainer, RewardConfig
from dataclasses import field
from huggingface_hub import login


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

def main():

    # device will be 'cuda' if a GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # torch.cuda.empty_cache() 
    print(torch.cuda.mem_get_info())

    login(token='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')
    
    model_name = "AI-Sweden-Models/gpt-sw3-1.3b"
    # model_name = "AI-Sweden-Models/gpt-sw3-6.7b"
    original_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        # device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Get the summary chosen rejected pairs
    dataset_name = "thorirhrafn/domar_pair"
    dataset = load_dataset(dataset_name)

    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        lora_dropout=0.1,
        bias="none",
        inference_mode=False,
        task_type=TaskType.SEQ_CLS
    )

    reward_model = get_peft_model(original_model, lora_config)
    print(print_number_of_trainable_model_parameters(reward_model))

    
    def preprocess(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for text, chosen, rejected in zip(examples["text"], examples["baseline_summary"], examples["model_summary"]):
            tokenized_chosen = tokenizer(text + "\nSummary: " + chosen)
            tokenized_rejected = tokenizer(text + "\nSummary: " + rejected)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples
        
    formatted_datasets = dataset.map(
        preprocess,
        batched=True,
        num_proc=4
    )

    max_length = 2000

    formatted_datasets = formatted_datasets.filter(
        lambda x: len(x["input_ids_chosen"]) <= max_length
        and len(x["input_ids_rejected"]) <= max_length
    )

    train_data = formatted_datasets['train']
    eval_data = formatted_datasets['eval']

    print('### CHOSEN ###')
    print(tokenizer.decode(train_data['input_ids_chosen'][10]))
    print('### REJECTED ###')
    print(tokenizer.decode(train_data['input_ids_rejected'][10]))

    # Define hyperparameters
    output_dir = 'thorirhrafn/gpt1B_reward_model'

    training_args = RewardConfig(
        output_dir=output_dir,
        learning_rate=3e-5,
        max_length = 2000,
        remove_unused_columns=False,
        lr_scheduler_type="linear",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        bf16=True,
        num_train_epochs=1,
        logging_steps=10,
        evaluation_strategy="steps",
        save_strategy="epoch",
        eval_steps=20,
        push_to_hub=True,
        report_to=["tensorboard"]
    )

    # Define the Trainer
    trainer = RewardTrainer(
            model=reward_model,
            args=training_args,
            tokenizer = tokenizer,
            train_dataset=train_data,
            eval_dataset=eval_data
    )

    print('--- Starting to train ---')

    trainer.train()

    print('--- Saving model ---')

    trainer.push_to_hub()

    print('--- Saving completed ---')

    return

if __name__ == '__main__':
    main()