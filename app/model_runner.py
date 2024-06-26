from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
    pipeline,
)
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from app_state import SingletonState
import torch
from utils import DEVICE


def getConfigFromModel(model, tokenizer, temperature):
    generation_config = GenerationConfig.from_pretrained(model)
    generation_config.max_new_tokens = 1024
    generation_config.temperature = temperature
    generation_config.top_p = 0.85
    generation_config.do_sample = True
    generation_config.repetition_penalty = 1.1
    generation_config.pad_token_id = tokenizer.eos_token_id
    return generation_config


def model_runner(model_name, quant, device, temperature):
    app_state = SingletonState()
    model_args = dict()
    app_state.data["llm_obj"] = dict()

    if quant == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model_args.update({"quantization_config": bnb_config})
    elif quant == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_args.update({"quantization_config": bnb_config})

    if device == "gpu" and "cuda" in DEVICE:
        model_args.update({"device_map": DEVICE})
    else:
        model_args.update({"device_map": "auto"})

    app_state.data["llm_obj"]["tokenizer"] = AutoTokenizer.from_pretrained(model_name)
    # total_steps = list(model.parameters())
    
    app_state.data["llm_obj"]["open_llm"] = HuggingFacePipeline(
        pipeline=pipeline(
            "text-generation",
            model=AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True, torch_dtype="auto", **model_args
            ),
            tokenizer=app_state.data["llm_obj"]["tokenizer"],
            generation_config=getConfigFromModel(
                model_name, app_state.data["llm_obj"]["tokenizer"], temperature
            ),
            return_full_text=False
        )
    )
