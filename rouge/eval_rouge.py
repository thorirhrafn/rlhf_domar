from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from peft import PeftModel
import evaluate
from huggingface_hub import login
import pandas as pd


def get_summaries(dataset, model, tokenizer, start=0,end=10, device='cpu', print_output=False):
    dialogues = dataset['test'][start:end]['Text']
    baseline_summaries = dataset['test'][start:end]['Summary']

    gen_config = GenerationConfig(
            max_new_tokens=250,
            min_new_tokens=100,
            do_sample=True,
            eos_token_id=model.config.eos_token_id,
            pad_token_id=model.config.eos_token_id,
            repetition_penalty=1.05,
            top_p=0.95,
            top_k=50,
            temperature=0.6
        )

    model_summaries = []
    human_summaries = []

    for idx , dialogue in enumerate(dialogues):
        start_prompt = "### Instruction: Write a summary of the text below. ### Input: "
        end_prompt = "### Summary: "
        prompt = start_prompt + dialogue + end_prompt
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        # print(f'Tokenized input length: {input_ids.shape[1]}')
        if input_ids.shape[1] < 1801:
          model_outputs = model.generate(input_ids=input_ids, generation_config=gen_config)
          model_text_output = tokenizer.decode(model_outputs[0], skip_special_tokens=True)

          text = model_text_output.split('### Summary:')

          if print_output:
              # print(model_text_output)
              print(f'#### {idx} ####')
              print(text[1])
              # print(f'#### HUMAN ####')
              # print(baseline_summaries[idx])
              print('-------')

          model_summaries.append(text[1])
          human_summaries.append(baseline_summaries[idx])

    zipped_summaries = list(zip(human_summaries, model_summaries))

    return zipped_summaries


def main():

    # device will be 'cuda' if a GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # torch.cuda.empty_cache() 
    print(torch.cuda.mem_get_info())

    login(token='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')

    model_name= "AI-Sweden-Models/gpt-sw3-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    sft_model_name = "thorirhrafn/gpt1B_domar_merged"
    sft_model = AutoModelForCausalLM.from_pretrained(
        sft_model_name, 
        torch_dtype=torch.bfloat16,
        use_cache=False,
        device_map="auto"
    )
    

    print('--- Loading dataset ---')

    sum_data = load_dataset("thorirhrafn/domar_gptswe")

    rlhf_model = 'thorirhrafn/GPT1B_domar_RLHF_e20'
    eval_model = PeftModel.from_pretrained(sft_model, rlhf_model)
    eval_model = eval_model.merge_and_unload()

    print('--- Evaluate Model ---')

    eval_model = eval_model.to(device)
    eval_model.eval()

    test_sum = get_summaries(
        dataset=sum_data, 
        model=eval_model,
        tokenizer=tokenizer,
        start=0,
        end=300,
        device=device,
        print_output=False
    ) 

    for idx, sum in enumerate(test_sum):
        print(f'### HUMAN {idx} ###')
        print(sum[0])
        print(f'### MODEL {idx} ###')
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