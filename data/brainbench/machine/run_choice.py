import os
import argparse

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F 

from utils import data_utils
from utils import model_utils
from utils import general_utils


def forward_pass(model, tokenizer, choices):
    """
    Args:
        - choices (list): list of strings, where each string is a prompt

    Perform a single forward pass over a testcase 
    (i.e., a prompt with choices) and computes perplexities
    for each choice.
    """
    # Forward pass to get nll and convert to ppl
    ppl = []
    for choice_index, prompt in enumerate(choices):
        with torch.no_grad():
            prompt = tokenizer(prompt, return_tensors='pt').to("cuda")
            if "token_type_ids" in prompt:
                prompt.pop("token_type_ids")
            output = model(
                input_ids=prompt["input_ids"], 
                labels=prompt["input_ids"]
            )

            # logits of the prompt tokens
            logits = output.logits
            labels = prompt["input_ids"]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            vocab_size = shift_logits.size(-1)
            shift_logits = shift_logits.view(-1, vocab_size)
            shift_labels = shift_labels.view(-1)

            log_probs = F.log_softmax(shift_logits, dim=-1)
            true_log_probs = log_probs.gather(dim=1, index=shift_labels.view(-1, 1)).squeeze()

            # Cast to float32 and compute nll and ppl
            true_log_probs = true_log_probs.float()
            nll = -true_log_probs.mean()
            ppl.append(np.exp(nll.item()))
    return ppl


@general_utils.timer
def main(llm, abstracts_fpath):
    np.random.seed(42)

    # Load model, tokenizer
    model, tokenizer = model_utils.load_model_and_tokenizer(llm)

    # Load dataset
    df = pd.read_csv(abstracts_fpath)
    prompt_template = data_utils.read_prompt_template(llm)

    PPL_A_and_B = []
    true_labels = []
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
        ppl = forward_pass(model, tokenizer, choices)
        PPL_A_and_B.append(ppl)
        true_labels.append(0 if choice_true == "A" else 1)

    PPL_A_and_B = np.array(PPL_A_and_B)
    true_labels = np.array(true_labels)

    # Compute accuracy
    tie_indices = []
    pred_labels = np.ones(PPL_A_and_B.shape[0], dtype=np.int32)
    for i, (ppl_A, ppl_B) in enumerate(PPL_A_and_B):
        if ppl_A < ppl_B:
            pred_labels[i] = 0
        elif ppl_A > ppl_B:
            pred_labels[i] = 1
        else:
            pred_labels[i] = -1
            tie_indices.append(i)
    
    print(f"Number of ties: {len(tie_indices)}")

    # Accuracy after removing ties
    acc = np.sum(pred_labels == true_labels) / (PPL_A_and_B.shape[0])
    print(f"Accuracy: {acc}")

    np.save(f"{results_dir}/PPL_A_and_B.npy", PPL_A_and_B)
    np.save(f"{results_dir}/labels.npy", true_labels)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_human_abstract", type=str, default="True")

    if parser.parse_args().use_human_abstract == "True":
        use_human_abstract = True
    else:
        use_human_abstract = False

    llms = [
        # "lora_r16_a32_finetune_mistral_7b_v01",
        # "lora_r32_a64_finetune_mistral_7b_v01",
        # "TinyLlama/TinyLlama_v1.1"
        # "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        # "microsoft/Phi-3-mini-4k-instruct",
        # "microsoft/Phi-3-mini-128k-instruct",
        # "lora_r256_a512_finetune_phi3_mini_4k_instruct",
        # "full_finetune_mistral_7b_v01",
        # "lora_r256_a512_finetune_mistral_7b_v01",
        # "mistralai/Mistral-7B-v0.1"
        "gpt2-medium_scratch_neuro_tokenizer",
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
    
