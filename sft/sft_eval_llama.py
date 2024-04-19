from datasets import Dataset, load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, LlamaTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, GenerationConfig
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

    # device will be 'cuda' if a GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # torch.cuda.empty_cache() 
    print(torch.cuda.mem_get_info())

    login(token='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')

    model_name= "meta-llama/Llama-2-7b-hf"
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        device_map="auto"
    )
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print('--- Loading dataset ---')

    sum_data = load_dataset("thorirhrafn/domar_gptswe")

    pretrain_model = 'thorirhrafn/llama_domar_pretuned'
    model = PeftModel.from_pretrained(original_model, pretrain_model)
    model = model.merge_and_unload()


    sft_model = 'thorirhrafn/llama_domar_finetune'
    eval_model = PeftModel.from_pretrained(model, sft_model)
    eval_model = eval_model.merge_and_unload()

    print('--- Evaluate Model ---')

    eval_model = eval_model.to(device)
    eval_model.eval()

    test_sum = get_summaries(
        dataset=sum_data, 
        model=eval_model,
        tokenizer=tokenizer,
        start=0,
        end=20,
        device=device,
        print_output=False
    ) 

    for sum in test_sum:
        print('### HUMAN ###')
        print(sum[0])
        print('### MODEL ###')
        print(sum[1])
        print('----------')

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