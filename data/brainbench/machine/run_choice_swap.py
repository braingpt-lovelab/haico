import os
import argparse
import json
import nltk
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F 

from utils import data_utils
from utils import model_utils
from utils import general_utils


def swap(abstract_index, testcase, subfield):
    """
    Given an original and incorrect abstracts (target), swap in result sentences from another original or 
    incorrect abstract from the same subfield (source).

    Number of swaps depends on the number of result sentences in the target.

    For a target with multiple result sentences, each swap-in result sentence must come from a different abstract.
    The swap-in result sentences cannot be from the target. 

    Source abstracts are from `data/swap_{type_of_abstract}.json`, with the data structure:
    {
        "subfield": {
            "abstract_id": {
                "original": ["result_sentence_1", "result_sentence_2", ...],
                "incorrect": ["result_sentence_1", "result_sentence_2", ...]
            }
        }
    }

    Args:
        - testcase (str): combined abstract (with [[, ]])
    """
    with open(swap_source_fpath, "r") as f:
        swap_source = json.load(f)

    swap_source_subfield = swap_source[subfield].copy()
    # Remove the target abstract
    swap_source_subfield.pop(f"{abstract_index}")
    print(f"Target abstract_id: {abstract_index}")
    
    original_collection = []
    incorrect_collection = []
    sentences = nltk.sent_tokenize(testcase)
    n_swaps = 0
    for sentence_idx, sentence in enumerate(sentences):
        a_successful_swap = False
        # For a result sentence, 
        if '[[' in sentence and ']]' in sentence:
            print(f"sentence {sentence_idx} | A result sentence found, swapping..")
            while not a_successful_swap:
                random_abstract_id = np.random.choice(list(swap_source_subfield.keys()))
                assert random_abstract_id != abstract_index, "Cannot swap in result sentence from the target abstract."
                num_result_sentences = len(swap_source_subfield[random_abstract_id]["original"])
                print(f"random_abstract_id: {random_abstract_id}, num_result_sentences: {num_result_sentences}")
                if num_result_sentences > 0:  # has reserve left
                    random_result_sentence_idx = np.random.randint(num_result_sentences)
                    # Retrieve the swap-in result sentence and remove from the source
                    original_result_sentence = swap_source_subfield[random_abstract_id]["original"].pop(random_result_sentence_idx)
                    incorrect_result_sentence = swap_source_subfield[random_abstract_id]["incorrect"].pop(random_result_sentence_idx)
                    original_collection.append(original_result_sentence)
                    incorrect_collection.append(incorrect_result_sentence)
                    
                    a_successful_swap = True
                    n_swaps += 1
                else:
                    swap_source_subfield.pop(f"{random_abstract_id}")
                    if len(swap_source_subfield) == 0:
                        raise ValueError("No more result sentences to swap in. Should not happen.")
        else:
            print(f"sentence {sentence_idx} | Not a result sentence, appending..")
            original_collection.append(sentence)
            incorrect_collection.append(sentence)
    
    print(f"Number of swaps: {n_swaps}")
    print("-" * 70)
    return " ".join(original_collection), " ".join(incorrect_collection)
            


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
def main(llm, abstracts_fpath, seed):
    nltk.download('punkt')
    np.random.seed(seed)

    # Load model, tokenizer
    model, tokenizer = model_utils.load_model_and_tokenizer(llm)

    # Load dataset
    df = pd.read_csv(abstracts_fpath)
    prompt_template = data_utils.read_prompt_template(llm)

    PPL_A_and_B = []
    true_labels = []
    for abstract_index, row in df.iterrows():
        subfield = row["journal_section"]
        testcase = row["combined_abstract"]
        original_abstract, incorrect_abstract = swap(abstract_index, testcase, subfield)

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_human_abstract", type=str, default="True")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.use_human_abstract == "True":
        use_human_abstract = True
    else:
        use_human_abstract = False

    llms = [
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf",
        "meta-llama/Llama-2-70b-hf",
        "tiiuae/falcon-40b",
        "tiiuae/falcon-40b-instruct",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Llama-2-70b-chat-hf",
        "facebook/galactica-6.7b",
        "facebook/galactica-30b",
        "facebook/galactica-120b",
        "mistralai/Mistral-7B-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "tiiuae/falcon-180B",
        "tiiuae/falcon-180B-chat"
    ]

    for llm in llms:
        if use_human_abstract:
            type_of_abstract = 'human_abstracts'
            abstracts_fpath = "testcases/BrainBench_Human_v0.1.csv"
            swap_source_fpath = f"data/swap_{type_of_abstract}.json"
        else:
            type_of_abstract = 'llm_abstracts'
            abstracts_fpath = "testcases/BrainBench_GPT-4_v0.1.csv"
            swap_source_fpath = f"data/swap_{type_of_abstract}.json"
        results_dir = f"model_results/{llm.replace('/', '--')}/{type_of_abstract}/swap_seed{args.seed}"

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        main(llm, abstracts_fpath, args.seed)
    
