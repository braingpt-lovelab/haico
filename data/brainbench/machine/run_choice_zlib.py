import os
import argparse

import zlib
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F 

from utils import data_utils
from utils import model_utils
from utils import general_utils


@general_utils.timer
def main(llm, abstracts_fpath):
    np.random.seed(42)

    # # Load model, tokenizer
    # model, tokenizer = model_utils.load_model_and_tokenizer(llm)

    # Load dataset
    df = pd.read_csv(abstracts_fpath)
    prompt_template = data_utils.read_prompt_template(llm)

    ZLIB_A_and_B = []
    labels = []
    for abstract_index, abstract in enumerate(df["combined_abstract"]):
        original_abstract, incorrect_abstract = data_utils.extract_abstract_pair(abstract)

        # Randomly shuffle to determine which abstract is A and which is B,
        # keep a record of the correct choice, which is used to determine
        # later if the model's choice is correct
        if np.random.rand() > 0.5:
            original_abstract, incorrect_abstract = incorrect_abstract, original_abstract
            choice_true = "B"
        else:
            choice_true = "A"

        # choices is [prompt_A, prompt_B]
        # where each prompt is the question + one of the abstracts as option.
        choices = data_utils.prepare_prompt_multiple_choice_harness(
            original_abstract, incorrect_abstract, prompt_template, 
        )

        print(
            f"-"*70 + "\n",
            f"*** Abstract index: {abstract_index} ***",
        )

        # Forward each prompt to get nll and convert to ppl
        zlib_entropy = []
        for choice_index, prompt in enumerate(choices):

            zlib_entropy.append(
                len(zlib.compress(bytes(prompt, 'utf-8')))
            )

        ZLIB_A_and_B.append(zlib_entropy)
        labels.append(0 if choice_true == "A" else 1)

    ZLIB_A_and_B = np.array(ZLIB_A_and_B)
    labels = np.array(labels)
    np.save(f"{results_dir}/ZLIB_A_and_B.npy", ZLIB_A_and_B)
    np.save(f"{results_dir}/labels.npy", labels)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_human_abstract", type=str, default="True")

    if parser.parse_args().use_human_abstract == "True":
        use_human_abstract = True
    else:
        use_human_abstract = False

    llms = [
        "gpt2",
        "finetune_gpt2",
        "gpt2_scratch_neuro_tokenizer"
    ]

    for llm in llms:
        if use_human_abstract:
            type_of_abstract = 'human_abstracts'
            abstracts_fpath = "testcases/BrainBench_Human_v0.1.csv"
        else:
            type_of_abstract = 'llm_abstracts'
            abstracts_fpath = "testcases/BrainBench_GPT-4_v0.1.csv"
        results_dir = f"model_results/{llm.replace('/', '--')}/{type_of_abstract}"

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        main(llm, abstracts_fpath)
