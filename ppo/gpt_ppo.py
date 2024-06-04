from accelerate import Accelerator
from tqdm import tqdm
import torch
from torch import optim
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from dataclasses import field
from huggingface_hub import login
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl import create_reference_model
from trl.core import LengthSampler
import evaluate
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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

    # torch.cuda.empty_cache() 
    print(torch.cuda.mem_get_info())

    login(token='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')
    
    # set original base model name 
    model_name = "AI-Sweden-Models/gpt-sw3-1.3b"
    # model_name = "AI-Sweden-Models/gpt-sw3-6.7b"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Get the summary chosen rejected pairs: human vs model
    dataset_name = "thorirhrafn/domar_pair"
    dataset = load_dataset(dataset_name)
    

    input_min_text_length = 1200 # 800 # 500
    input_max_text_length = 1600 # 1000 # 600
    input_size = LengthSampler(input_min_text_length, input_max_text_length)


    # preprocess data to fit the form#at expected by the PPO-trainer
    def tokenize(sample):
        start_prompt = "### Instruction: Write a summary of the text below. ### Input: "
        end_prompt = " ### Summary: "
        prompt = start_prompt + sample["text"] + end_prompt

        sample["input_ids"] = tokenizer.encode(prompt)[: input_size()]
        # This must be called "query", which is a requirement of our PPO library.
        sample["query"] = tokenizer.decode(sample["input_ids"])

        return sample

    # Tokenize each dialogue.
    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format(type="torch")
    
    train_data = dataset['train']

    # Define hyperparameters
    output_dir = 'thorirhrafn/gpt1B_SFT_RLHF_model'

    def collator(data):
        return {key: [d[key] for d in data] for key in data[0]}

    # Load fine-tuned model for RLHF-training with the PPO algorithm

   # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_model = "thorirhrafn/gpt1B_domar_merged"
    device_map = {"": Accelerator().local_process_index}
    print(f'Device Map: {device_map}')    
    # set original model
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        sft_model,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        device_map='auto',
        peft_config=lora_config
    )

    ref_model = None
   

    config = PPOConfig(
        model_name=model_name,
        learning_rate=2e-5,
        batch_size=32,
        mini_batch_size=4,
        gradient_accumulation_steps=8,
        log_with='tensorboard',
        project_kwargs={"logging_dir": './output_RLHF'}
    )
    

    ppo_trainer = PPOTrainer(
        config=config,
        model=ppo_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=train_data,
        data_collator=collator,
    )


    #  get trained reward adapter model for sequence classification
    original_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        # device_map='auto'
    )

    peft_model = 'thorirhrafn/gpt1B_reward_model3'
    reward_model = PeftModel.from_pretrained(original_model, peft_model)
    reward_model = reward_model.merge_and_unload()
    reward_model = reward_model.to(device)
    reward_model.eval()

    print(f'Num processes: {ppo_trainer.accelerator.num_processes}')
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

    reward_pipe = pipeline(
        "text-classification", 
        model=reward_model, 
        tokenizer=tokenizer,
        return_token_type_ids=False,
        device=device
    )

    
    BATCH = 32
    
    reward_kwargs = {
        "top_k": None,
        "function_to_apply": "none",
        "batch_size": BATCH,
        "truncation": True,
    }

    # test reward model before training
    print('### Testing Reward Pipeline ###')

    prompt_list = []
    chosen_list = []
    rejected_list = []

    for i in range(BATCH):
        prompt_list.append(dataset['test'][i]['text'])
        chosen_list.append(dataset['test'][i]['baseline_summary'])
        rejected_list.append(dataset['test'][i]['model_summary'])

    # get chosen and rejected responses to calculate gain and bias
    texts = [q + r for q, r in zip(prompt_list, rejected_list)]
    for q, r in zip(prompt_list, chosen_list):
        texts.append(q + r)
    pipe_outputs = reward_pipe(texts, **reward_kwargs)
    
    # calculate mean and std to normalize model rewards so that mean is 0 and std is 1
    model_rewards = [torch.tensor(output[0]["score"]) for output in pipe_outputs]
    r_mean = np.mean(model_rewards)
    r_std = np.std(model_rewards)

    r_gain = 1 / r_std
    r_bias = -1 * r_gain * r_mean

    print(f'Reward Gain: {r_gain}')
    print(f'Reward Bias: {r_bias}')


    # Compute reward score using the pipeline
    texts = [q + r for q, r in zip(prompt_list, chosen_list)]
    pipe_outputs = reward_pipe(texts, **reward_kwargs)
    chosen_unweighted = [torch.tensor(output[0]["score"]) for output in pipe_outputs]
    chosen_rewards = [torch.tensor(output[0]["score"] * r_gain + r_bias) for output in pipe_outputs]

    print('-----------')
    print('Chosen rewards:')
    print(chosen_rewards)
    print(f'Unweighted Chosen Reward: {sum(chosen_unweighted)}')
    print(f'Total Chosen Reward: {sum(chosen_rewards)}')
    print(f'Average Chosen Reward: {sum(chosen_rewards) / len(chosen_rewards)}')


    # Compute reward score using the pipeline
    texts = [q + r for q, r in zip(prompt_list, rejected_list)]
    pipe_outputs = reward_pipe(texts, **reward_kwargs)
    rejected_unweighted = [torch.tensor(output[0]["score"]) for output in pipe_outputs]
    rejected_rewards = [torch.tensor(output[0]["score"] * r_gain + r_bias) for output in pipe_outputs]

    print('-----------')
    print('Rejected rewards:')
    print(rejected_rewards)
    print(f'Unweighted Rejected Reward: {sum(rejected_unweighted)}')
    print(f'Total Rejected Reward: {sum(rejected_rewards)}')
    print(f'Average Rejected Reward: {sum(rejected_rewards) / len(rejected_rewards)}')
    print('-----------')

    # Get base summary data for evaluation
    sum_data = load_dataset("thorirhrafn/domar_gptswe")

    # load and process training data for RLHF
    # Get the summary chosen rejected pairs: human vs model
    dataset_name = "thorirhrafn/domar_pair"
    dataset = load_dataset(dataset_name)


    # ppo_model.to(device)

    print('--- Evaluate model before training ---')

    test_sum = get_summaries(
        dataset=sum_data, 
        model=ppo_model,
        tokenizer=tokenizer,
        start=0,
        end=30,
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

    original_model_results = rouge.compute(
        predictions=df['model_summaries'],
        references=df['baseline_summaries'],
        use_aggregator=True,
        use_stemmer=True,
    )

    print('Original model results::')
    print(original_model_results)


    # Setup training loop to run reinforcement learning

    output_min_length = 50
    output_max_length = 250
    output_length_sampler = LengthSampler(output_min_length, output_max_length)


    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    steps = []
    reward = []
    kl = []
    print('--- Starting to train ---')
    
    epochs = 10
    n = 0

    for epoch in range(epochs):
        print(f'#### EPOCH {epoch} ####')
        for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):

            query_tensors = batch["input_ids"]

            # Get responses from the queries to the model
            response_tensors = []
            for query in query_tensors:
                gen_len = output_length_sampler()
                generation_kwargs["max_new_tokens"] = gen_len
                response = ppo_trainer.generate(query, **generation_kwargs)
                response_tensors.append(response.squeeze()[-gen_len:])
            batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

            # Compute reward score using the pipeline
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            pipe_outputs = reward_pipe(texts, **reward_kwargs)
            reward_tensors = [torch.tensor(output[0]["score"] * r_gain + r_bias) for output in pipe_outputs]


            #### Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
            ppo_trainer.log_stats(stats, batch, reward_tensors)

            steps.append(n)
            reward.append(stats["ppo/returns/mean"])
            kl.append(stats["objective/kl"])
            n += 1

            print(f'\nEpoch: {epoch}')
            print(f'Step: {step}')
            print(f'objective/kl: {stats["objective/kl"]}')
            print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
            print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
            print('-'.join('' for x in range(100)))

            # if n == 150:
            #     break



    print('--- Training process finished ---')

    # Plotting the data
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(steps, reward)
    plt.title('PPO Training')
    plt.ylabel('Mean Reward')
    plt.subplot(2, 1, 2)
    plt.plot(steps, kl)
    plt.xlabel('steps')
    plt.ylabel('KL Value')

    # Save the plot as a PNG file
    plt.savefig('ppo_reward_norm2_e10.png')

    print('--- Evaluate model after training ---')

    ppo_model.eval()
    test_sum = get_summaries(
        dataset=sum_data, 
        model=ppo_model,
        tokenizer=tokenizer,
        start=0,
        end=300,
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

    #### Save model
    ppo_trainer.save_pretrained(output_dir)
    ppo_model.push_to_hub("GPT1B_domar_RLHF")

    print('--- Saving completed ---')



    return

if __name__ == '__main__':
    main()