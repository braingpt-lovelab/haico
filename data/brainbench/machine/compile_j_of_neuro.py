import os 
import json 

"""
Get Journal of neuroscience (j_of_neuro) data from the train split portion of 
https://huggingface.co/datasets/BrainGPT/train_valid_split_pmc_neuroscience_2002-2022_filtered_subset/tree/main

Locally, dataset is at /datadrive1/train_valid_split/train
"""

def iterate_directory(train_directory):
    """
    Iterate the train directory and track down 
    all Journal of Neuroscience abstracts
    """
    pub_years = []
    j_of_neuro_files = []
    for root, dirs, files in os.walk(train_directory):
        for file in files:
            if file.startswith("Journal_of_Neuroscience_abstract") \
                and "jneurosci" in file.lower():
                j_of_neuro_files.append(file)

                # Get publication year from fname, year info is by the end of .json
                pub_year = file.split(".")[0][-4:]
                pub_years.append(pub_year)
    
    earliest = min(pub_years)
    latest = max(pub_years)
    print(f"Earliest publication year: {earliest}, latest publication year: {latest}")
    return j_of_neuro_files


def compile_files(fnames):
    """
    Compile files into desired format for models to computing ppl and zlib.

    json = [
        {
            "text": "abstract content"
        },
        {
            "text": "abstract content"
        },
        ...]
    """
    j_of_neuro_files = []
    for fname in fnames:
        fpath = os.path.join(train_directory, fname)
        with open(fpath, 'r') as f:
            data = json.load(f)
            # the source data is already {"text": "abstract content}
            j_of_neuro_files.append(data)  
    
    with open(datafile_path, 'w', encoding='utf-8') as f:
        json.dump(j_of_neuro_files, f, ensure_ascii=False, indent=2)
    print(f"Saved all abstracts to {datafile_path}")


def main():
    j_of_neuro_files = iterate_directory(train_directory)
    compile_files(j_of_neuro_files)


if __name__ == "__main__":
    train_directory = "/datadrive1/train_valid_split/train"
    start_date = '2017'
    end_date = '2022'
    if not os.path.exists("data"):
        os.makedirs("data")
    datafile_path = f"data/j_of_neuro_abstracts_{start_date}_{end_date}.json"

    main()