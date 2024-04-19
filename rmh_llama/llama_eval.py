from transformers import AutoModelForCausalLM, LlamaTokenizer
import torch
from peft import PeftModel
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

    model_name= "meta-llama/Llama-2-7b-hf"
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    peft_model = 'thorirhrafn/ice_llama_v3'
    model = PeftModel.from_pretrained(original_model, peft_model)

    print('--- Evaluate Model ---')

    model = model.to(device)
    model.eval()
    inputs = tokenizer("Trú á andsetningar og særingar lifir enn góðu lífi þar sem andatrú á sterkar rætur, en mjög hafi dregið úr henni á Vesturlöndum. Þó sé hún ekki alveg óþekkt.", return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=256)
        print('')
        print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])

    return

if __name__ == '__main__':
    main()