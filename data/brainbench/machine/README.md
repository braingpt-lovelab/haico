# brainbench_experiments

### To work with the repo locally:
```
git clone git@github.com:braingpt-lovelab/brainbench_experiments.git --recursive
```
### To create conda environment:
```
conda env create -f conda_envs/environment.yml
```

### Reproducing experiments from scratch:
1. Run inference with all LLMs on BrainBench test cases: `python run_choice.py --use_human_abstract <True|False>`.
2. Run inference with all LLMs on BrainBench test cases under the **without context** condition: `python run_choice_iso.py --use_human_abstract <True|False>`.
3. Run memorization analysis:
    * First compile data from biorxiv and arxiv: `python compile_biorxiv.py` and `python compile_arxiv.py`. For now, need to manually adjust year range inside the scripts.
    * Run inference to obtain zlib entropy of BrainBench test cases: `python run_choice_zlib.py`.
    * Run inference to obtain zlib entropy and perplexity of all compiled data from biorxiv, arxiv and the Gettysburg Address: `python dataset_ppl_zlib.py`. For now, the choice of which data source to run needs to be set manually inside the script.

### Results and plotting
* All analyses results (pre-plotting) are saved in `model_results` grouped by LLM names and further organized by experiment type such as inference without context.
* For obtaining figures in the paper, please refer to the dedicated repo for plotting here: https://github.com/braingpt-lovelab/brainbench_results.

### Attribution
```
@misc{luo2024large,
      title={Large language models surpass human experts in predicting neuroscience results}, 
      author={Xiaoliang Luo and Akilles Rechardt and Guangzhi Sun and Kevin K. Nejad and Felipe Yáñez and Bati Yilmaz and Kangjoo Lee and Alexandra O. Cohen and Valentina Borghesani and Anton Pashkov and Daniele Marinazzo and Jonathan Nicholas and Alessandro Salatiello and Ilia Sucholutsky and Pasquale Minervini and Sepehr Razavi and Roberta Rocca and Elkhan Yusifov and Tereza Okalova and Nianlong Gu and Martin Ferianc and Mikail Khona and Kaustubh R. Patil and Pui-Shee Lee and Rui Mata and Nicholas E. Myers and Jennifer K Bizley and Sebastian Musslick and Isil Poyraz Bilgin and Guiomar Niso and Justin M. Ales and Michael Gaebler and N Apurva Ratan Murty and Chloe M. Hall and Jessica Dafflon and Sherry Dongqi Bao and Bradley C. Love},
      year={2024},
      eprint={2403.03230},
      archivePrefix={arXiv},
      primaryClass={q-bio.NC}
}
```
