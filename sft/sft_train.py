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

def load_json_data(path):
    # load  dataset
    with open(path, encoding='utf-8') as file:
        icesum = json.load(file)

    title_list = []
    text_list = []
    summary_list = []

    for entry in icesum:
        data = icesum[entry]
        title_list.append(data['title'])
        text_list.append(data['text'])
        summary_list.append(data['summary'])

    datadict = {"Title": title_list, "Text": text_list, "Summary": summary_list}
    dataset = Dataset.from_dict(datadict)

    train_idx = int(0.9 * len(dataset))
    idx = list(range(len(dataset)))

    # Create training split
    ds_split = DatasetDict({
        'train': dataset.select(idx[:train_idx]),
        'eval': dataset.select(idx[train_idx: train_idx + 50]),
        'test': dataset.select(idx[train_idx + 50:]),
    })

    return ds_split


def get_summaries(dataset, model, tokenizer, start=0,end=10, device='cpu', print_output=False):
    dialogues = dataset['test'][start:end]['Text']
    baseline_summaries = dataset['test'][start:end]['Summary']

    model_summaries = []

    for _, dialogue in enumerate(dialogues):
        start_prompt = "### Instruction: Write a summary of the text below. ### Input: "
        end_prompt = "### Summary: "
        prompt = start_prompt + dialogue + end_prompt
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
    
    model_name = "AI-Sweden-Models/gpt-sw3-1.3b"
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # model_name= "meta-llama/Llama-2-7b-hf"
    # original_model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.bfloat16,
    #     use_cache=False,
    # )
    # tokenizer = LlamaTokenizer.from_pretrained(model_name)
    # tokenizer.pad_token = tokenizer.eos_token

    print('--- Loading dataset ---')

    sum_data = load_json_data('icesum.json')

    def tokenize_function(example):
        start_prompt = "### Instruction: Write a summary of the text below. ### Input: "
        prompt = [start_prompt + dialogue + "### Summary: " + summary
                for dialogue, summary in zip(example["Text"], example["Summary"])]
        example['input_ids'] = tokenizer(prompt, max_length=1600, padding=True, truncation=True, return_tensors="pt").input_ids
        example['labels'] = example['input_ids']

        return example

    tokenized_datasets = sum_data.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(['Title', 'Text', 'Summary'])

    lora_config = LoraConfig(
        r=256,
        lora_alpha=256,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    peft_model = get_peft_model(original_model, lora_config)
    print(print_number_of_trainable_model_parameters(peft_model))

    print('--- Evaluate model before training ---')

    test_sum = get_summaries(
        dataset=sum_data, 
        model=original_model,
        tokenizer=tokenizer 
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


    output_dir = 'thorirhrafn/gpt_icesum'
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        bf16=True,
        num_train_epochs=5,
        logging_steps=5,
        evaluation_strategy="steps",
        save_strategy="epoch",
        eval_steps=50,
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

    print('--- Evaluate model after training ---')

    peft_model.eval()
    test_sum = get_summaries(
        dataset=sum_data, 
        model=peft_model,
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

    # trainer.push_to_hub()

    print('--- Saving completed ---')

    return

if __name__ == '__main__':
    main()