import json
import transformers
import torch


def load_model_and_tokenizer(model_fpath, tokenizer_only=False):
    if tokenizer_only:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_fpath,
        )
        return tokenizer
    
    load_in_8bit = False
    torch_dtype = torch.float16


    if model_fpath in [
            "full_finetune_mistral_7b_v01"
        ]:
        model_fpath = f"/home/ken/projects/full_finetuning/exp/{model_fpath}/checkpoint.0"
        print("Loading model from", model_fpath)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_fpath,
            load_in_8bit=load_in_8bit,
            device_map='auto',
            torch_dtype=torch.bfloat16
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-v0.1"
        )
    
    elif "lora_" in model_fpath:
        import peft  # global import will default to using all GPUs.

        model_config = f"/home/ken/projects/full_finetuning/exp/{model_fpath}/model_config.json"
        model_fpath = f"/home/ken/projects/full_finetuning/exp/{model_fpath}/checkpoint.1"
        # Load the base model
        with open(model_config, "r") as f:
            base_model_fpath = json.load(f)["model_path"]

        model = transformers.AutoModelForCausalLM.from_pretrained(
            base_model_fpath,
            load_in_8bit=False,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        # Load the PEFT model
        peft_model_fpath = model_fpath
        print("Loading PEFT model from", peft_model_fpath)
        model = peft.PeftModel.from_pretrained(model, peft_model_fpath)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model_fpath,
        )
        

    # Load model trained from scratch from local checkpoint
    elif model_fpath in [
            "gpt2_scratch",
            "finetune_gpt2",
            "gpt2_scratch_neuro_tokenizer",
            "gpt2-medium_scratch_neuro_tokenizer",
            "gpt2-large_scratch_neuro_tokenizer"
        ]:
        model_fpath = f"/home/ken/projects/matching_experts/model_training/exp/{model_fpath}/checkpoint.4"
        print("Loading GPT2 model from", model_fpath)
        model = transformers.GPT2LMHeadModel.from_pretrained(
            model_fpath,
            load_in_8bit=load_in_8bit,
            device_map='auto',
            trust_remote_code=True,
            torch_dtype=torch_dtype
        )

        tokenizer = transformers.GPT2Tokenizer.from_pretrained(
            model_fpath,    
        )
    
    # Load model untrained (config only)
    elif model_fpath == "gpt2_init":
        print("Loading GPT2 model untrained")
        from transformers import AutoConfig, AutoModelForCausalLM
        model_config = AutoConfig.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_config(model_config).to('cuda')
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(
            "gpt2",
        )

    # Load pretrained model
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_fpath,
            load_in_8bit=load_in_8bit,
            device_map='auto',
            trust_remote_code=True,
            torch_dtype=torch_dtype
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_fpath,
        )

    return model, tokenizer
