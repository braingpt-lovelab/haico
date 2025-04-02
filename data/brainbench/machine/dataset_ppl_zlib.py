import os
import zlib
import torch
import numpy as np
from datasets import load_dataset

from utils import model_utils


def main(llm, dataset):
    model, tokenizer = model_utils.load_model_and_tokenizer(llm)

    PPL = []
    ZLIB = []
    for i, text in enumerate(dataset):
        print(f"Processing {i}th sample,\nlength: {len(text)},\nsnip: {text[:100]}")
        with torch.no_grad():
            prompt = tokenizer(
                text, return_tensors='pt', truncation=True
            ).to("cuda")
            if "token_type_ids" in prompt:
                prompt.pop("token_type_ids")
            output = model(input_ids=prompt["input_ids"], labels=prompt["input_ids"])
            nll = output.loss.item()
            PPL.append(np.exp(nll))
            ZLIB.append(len(zlib.compress(bytes(text, 'utf-8'))))

    PPL = np.array(PPL)
    ZLIB = np.array(ZLIB)
    np.save(f"{results_dir}/PPL.npy", PPL)
    np.save(f"{results_dir}/ZLIB.npy", ZLIB)


if __name__ == "__main__":
    seed = 42
    n_samples = 1000
    dataset_source = "j_of_neuro"
    llms = [
        "gpt2",
         "finetune_gpt2",
        "gpt2_scratch_neuro_tokenizer"
    ]

    data_sources = {

    }

    if dataset_source == "refinedweb":
        dataset = load_dataset("data", data_files="train-00000-of-05534-b8fc5348cbe605a5.parquet")
        dataset = dataset.shuffle(seed=seed)["train"][:n_samples]["content"]
        
    elif 'biorxiv' in dataset_source:
        dataset = load_dataset("data", data_files=f"{dataset_source}.json")
        dataset = dataset.shuffle(seed=seed)["train"][:n_samples]["text"]
    
    elif 'arxiv' in dataset_source:
        dataset = load_dataset("data", data_files=f"{dataset_source}.json")
        dataset = dataset.shuffle(seed=seed)["train"][:n_samples]["text"]
    
    elif dataset_source == "gettysburg_address":
        dataset = load_dataset("data", data_files="gettysburg_address.json")
        dataset = dataset["train"][:n_samples]["text"]

    elif dataset_source == "j_of_neuro":
        dataset = load_dataset("data", data_files="j_of_neuro_abstracts_2017_2022.json")
        dataset = dataset.shuffle(seed=seed)["train"][:n_samples]["text"]

    for llm in llms:
        results_dir = f"model_results/{llm.replace('/', '--')}/{dataset_source}"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        main(llm, dataset)
