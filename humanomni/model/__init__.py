# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

import torch
from transformers import PretrainedConfig, AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

from .projector import load_mm_projector
from .humanomni_model import HumanOmniQwen2ForCausalLM, HumanOmniQwen2Config



VLLMs = {
    "HumanOmni_qwen2": HumanOmniQwen2ForCausalLM,
}

VLLMConfigs = {
    "HumanOmni_qwen2": HumanOmniQwen2Config,
}


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, config_kwargs={}, **kwargs):
    if 'token' in kwargs:
        token = kwargs['token']
    else:
        token = None
    
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        # NOTE: High-version Transformers will report: """ValueError: You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing `quantization_config` argument at the same time."""
        # kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    config = AutoConfig.from_pretrained(model_path, **config_kwargs)

    # judge model type
    model_type = config.model_type

    # judge pretrain/finetune
    try:
        is_pretraining = config.tune_mm_mlp_adapter
    except:
        is_pretraining = False

        # NOTE: SFT model loading
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, token=token)
    model = HumanOmniQwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=config, **kwargs)
    processor = None

    if "HumanOmni" in model_type:
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        # NOTE: HuanOmni adopts the same processor for processing image and video.

        processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    if getattr(model.config, "mm_audio_tower", None):
        audio_tower = model.get_audio_tower()
        if not audio_tower.is_loaded:
            audio_tower.load_model()
        audio_tower.to(device=device, dtype=torch.float16)

        audio_processor = audio_tower.audio_processor
        return tokenizer, model, processor, context_len, audio_processor
    else:
        return tokenizer, model, processor, context_len, None
    # return tokenizer, model, processor, context_len
