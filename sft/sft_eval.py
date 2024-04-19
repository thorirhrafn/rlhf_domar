from datasets import Dataset, load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, GenerationConfig
import torch
from peft import LoraConfig, PeftModel, TaskType
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

    datadict = {"title": title_list, "text": text_list, "summary": summary_list}
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


def get_summaries(dataset, model, tokenizer, start=0,end=10):
    dialogues = dataset['test'][start:end]['text']
    baseline_summaries = dataset['test'][start:end]['summary']

    model_summaries = []

    for _, dialogue in enumerate(dialogues):
        prompt = f"""
        Summarize the following conversation.

        {dialogue}

        Summary: """
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        model_outputs = model.generate(input_ids=input_ids.to("cuda"), generation_config=GenerationConfig(max_new_tokens=256))
        model_text_output = tokenizer.decode(model_outputs[0], skip_special_tokens=True)
        model_summaries.append(model_text_output)

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

    print('--- Loading dataset ---')

    sum_data = load_json_data('icesum.json')

    peft_model = 'thorirhrafn/gpt_icesum'
    model = PeftModel.from_pretrained(original_model, peft_model)
    model = model.merge_and_unload()

    print('--- Evaluate Model ---')

    model = model.to(device)
    model.eval()

    test_sum = get_summaries(
        dataset=sum_data, 
        model=model,
        tokenizer=tokenizer 
    ) 
    rouge = evaluate.load('rouge')
    df = pd.DataFrame(test_sum, columns = ['baseline_summaries', 'model_summaries'])

    model_results = rouge.compute(
        predictions=df['model_summaries'],
        references=df['baseline_summaries'],
        use_aggregator=True,
        use_stemmer=True,
    )

    print('Trained model results::')
    print(model_results)

    return

if __name__ == '__main__':
    main()