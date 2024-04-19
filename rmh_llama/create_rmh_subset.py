from datasets import DatasetDict, load_dataset
from huggingface_hub import login, logout

def main():

    login(token='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')

    # load rmh dataset
    print('Loading rmh dataset....\n')
    rmh_ds = load_dataset("stoddur/rmh")

    # Calculate the index size for the new subset (5%)
    new_train_size = int(0.05 * len(rmh_ds['train']))

    # Want to train on the 15-20% part of the rmh
    start_index = new_train_size*3

    # Create subset
    rmh_subset_medium = DatasetDict({
        'train': rmh_ds['train'].select(range(start_index, start_index + new_train_size)),
        'test': rmh_ds['train'].select(range(start_index + new_train_size, start_index + new_train_size + 2000)),
        'eval': rmh_ds['train'].select(range(start_index + new_train_size + 2000, start_index + new_train_size + 4000))
    })

    print(f'Train size: {len(rmh_subset_medium["train"])}')
    print(f'Eval size: {len(rmh_subset_medium["eval"])}')
    print(f'Test size: {len(rmh_subset_medium["test"])}')

    # push to huggingface hub
    rmh_subset_medium.push_to_hub("thorirhrafn/rmh_subset_medium4")

    logout()

    print('\nData processing finished')

    return


if __name__ == '__main__':
    main()