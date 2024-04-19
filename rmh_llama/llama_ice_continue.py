from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, LlamaTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from huggingface_hub import login, logout
import math

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


def print_cuda_memory():
    if torch.cuda.is_available():
        print("CUDA is available. Found {} GPU(s).".format(torch.cuda.device_count()))

        for i in range(torch.cuda.device_count()):
            gpu = torch.cuda.get_device_name(i)
            print("GPU {}: {}".format(i, gpu))

            # Get the GPU memory summary
            memory_info = torch.cuda.mem_get_info(i)
            print("  Memory (free, total):\n{}".format(memory_info))

            print("\n")
    else:
        print("CUDA is not available on this machine.")


def main():

    # device will be 'cuda' if a GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # torch.cuda.empty_cache() 
    # print(torch.cuda.mem_get_info())
    print_cuda_memory()

    login(token='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')

    model_name= "meta-llama/Llama-2-7b-hf"
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        device_map="auto"
    )
    base_model.enable_input_require_grads()
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    peft_model = 'thorirhrafn/ice_llama_v3'
    model = PeftModel.from_pretrained(
        base_model, 
        peft_model, 
        is_trainable=True,
        device_map="auto"
    )

    print('--- Loading dataset ---')

    rmh_data = load_dataset("thorirhrafn/rmh_subset_medium4")

    def preprocess_function(examples):
        return tokenizer(["".join(x) for x in examples['text']])

    print('--- Processing data ---')

    tokenized_rmh = rmh_data.map(
        preprocess_function,
        batched=True,
        remove_columns=rmh_data["train"].column_names,
    )

    
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

    rmh_dataset = tokenized_rmh.map(group_texts, batched=True, num_proc=4)

    print(print_number_of_trainable_model_parameters(model))

    output_dir = 'thorirhrafn/ice_llama2_v4'
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-5,
        lr_scheduler_type="linear",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=8,
        bf16=True,
        num_train_epochs=2,
        logging_steps=25,
        evaluation_strategy="steps",
        save_strategy="epoch",
        eval_steps=1000,
        push_to_hub=True,
        report_to=["tensorboard"]
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=rmh_dataset["train"],
        eval_dataset=rmh_dataset["eval"],
        data_collator=data_collator,
    )


    print('--- Starting to train ---')

    trainer.train()

    print('--- Training process finished ---')

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    print('--- Saving model ---')

    trainer.push_to_hub()

    print('--- Saving completed ---')

    return

if __name__ == '__main__':
    main()