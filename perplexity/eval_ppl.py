from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer
from datasets import load_dataset
import torch
import random
from peft import PeftModel
from huggingface_hub import login
from tqdm import tqdm


def main():

    # device will be 'cuda' if a GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # torch.cuda.empty_cache() 
    print(torch.cuda.mem_get_info())

    login(token='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')

    model_name= "meta-llama/Llama-2-7b-hf"
    # model_name = "AI-Sweden-Models/gpt-sw3-6.7b"
    # model_name = "AI-Sweden-Models/gpt-sw3-1.3b"
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    pretrained_model = 'thorirhrafn/llama_domar_pretuned'
    model = PeftModel.from_pretrained(original_model, pretrained_model)
    model = model.merge_and_unload()


    # sft_model = 'thorirhrafn/icellama_domar_finetune_v3'
    sft_model = 'thorirhrafn/llama_domar_finetune'
    eval_model = PeftModel.from_pretrained(model, sft_model)
    eval_model = eval_model.merge_and_unload()

    print('--- Process Test Data ---')

    test = load_dataset('thorirhrafn/domar_long_text', split="test")

    # randomly select 300 rows to use for evaluation
    num = 300
    sampled_indices = random.sample(range(len(test)), num)
    test = test.select(sampled_indices)
    encodings = tokenizer("\n\n".join(test["Text"]), return_tensors="pt")

    print('--- Evaluate Model ---')

    # model = original_model.to(device)
    eval_model = eval_model.to(device)
    eval_model.eval()

    # max_length = 2048 # GPT-SW3 context window
    max_length = 4096   # Llama xontext window
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = eval_model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    print(f'\nMean perplexity from domar long test set: {ppl}')

    return

if __name__ == '__main__':
    main()