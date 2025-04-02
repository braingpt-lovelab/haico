import os
import json
import argparse
import pandas as pd
import numpy as np

from utils import data_utils

"""
Use `testcases/BrainBench_Human_v0.1.csv` or `testcases/BrainBench_GPT-4_v0.1.csv` to prepare the swap testcases.

Final output format:
{
    "subfield": {
        "abstract_id": {
            "original": ["result_sentence_1", "result_sentence_2", ...],
            "incorrect": ["result_sentence_1", "result_sentence_2", ...]
        }
    }
}

Implementation:
    Iterate through rows in csv, and for each row, extract the abstract_id, and the abstract.
    For each abstract, locate result sentences (ones contain [[, ]]), parse out the correct and incorrect versions,
    and individually append to corresponding lists.
"""    


def main():
    testcases_swap = {}

    df = pd.read_csv(abstracts_fpath)
    for abstract_id, row in df.iterrows():
        subfield = row["journal_section"]
        testcase = row["combined_abstract"]
        original_result_sentences, incorrect_result_sentences = \
            data_utils.extract_abstract_pair_isolated_sentences(testcase)
        
        if subfield not in testcases_swap:
            testcases_swap[subfield] = {}
        
        testcases_swap[subfield][abstract_id] = {
            "original": original_result_sentences,
            "incorrect": incorrect_result_sentences
        }
    
    with open(f"data/swap_{type_of_abstract}.json", "w") as f:
        json.dump(testcases_swap, f, indent=4)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_human_abstract", type=str, default="True")

    if parser.parse_args().use_human_abstract == "True":
        use_human_abstract = True
    else:
        use_human_abstract = False

    if use_human_abstract:
        type_of_abstract = 'human_abstracts'
        abstracts_fpath = "testcases/BrainBench_Human_v0.1.csv"
    else:
        type_of_abstract = 'llm_abstracts'
        abstracts_fpath = "testcases/BrainBench_GPT-4_v0.1.csv"

    main()