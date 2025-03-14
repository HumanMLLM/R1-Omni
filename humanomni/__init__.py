import os
import copy
import warnings
import shutil
from functools import partial

import torch

from .model import load_pretrained_model
from .mm_utils import process_image, process_video, process_audio,tokenizer_multimodal_token, get_model_name_from_path, KeywordsStoppingCriteria,process_image_npary
from .constants import NUM_FRAMES, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, MODAL_INDEX_MAP, DEFAULT_AUDIO_TOKEN
import transformers

def model_init(model_path=None, config_kwargs={}, **kwargs):
    # with_face = kwargs.get('with_face', False)
    model_path = "HumanOmni_7B" if model_path is None else model_path
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, processor, context_len, audio_processor = load_pretrained_model(model_path, None, model_name, config_kwargs=config_kwargs, **kwargs)

    if tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    num_frames = model.config.num_frames if hasattr(model.config, "num_frames") else NUM_FRAMES
    if "qwen2vit" in model_path:
        from .mm_utils import process_image_qwen, process_video_qwen
        processor = {
            'image': partial(process_image_qwen, processor=processor, aspect_ratio=None),
            'video': partial(process_video_qwen, processor=processor, aspect_ratio=None, num_frames=num_frames),
        } 
    else:
        processor = {
                'image': partial(process_image, processor=processor, aspect_ratio=None),
                'video': partial(process_video, processor=processor, aspect_ratio=None, num_frames=num_frames),
                'face': partial(process_image_npary, processor=processor, aspect_ratio=None),
                'audio': partial(process_audio, processor=audio_processor),
            }
    return model, processor, tokenizer


def mm_infer(image_or_video, instruct, model, tokenizer, audio=None, modal='video', question=None, bert_tokeni=None, **kwargs):
    """inference api of HumanOmni for video understanding.

    Args:
        model: HumanOmni model.
        image_or_video (torch.Tensor): image tensor (1, C, H, W) / video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
        modal (str): inference modality.
    Returns:
        str: response of the model.
    """
    question_prompt = None
    if question is not None:
        question = [question]
        question_prompt = bert_tokeni(question, return_tensors='pt', padding=True, truncation=True,add_special_tokens=True)
        question_prompt = {key: value.to('cuda') for key, value in question_prompt.items()}

    if modal == 'image':
        modal_token = DEFAULT_IMAGE_TOKEN
    elif modal == 'video':
        modal_token = DEFAULT_VIDEO_TOKEN
    elif modal == 'audio':
        modal_token = DEFAULT_AUDIO_TOKEN
    elif modal == 'video_audio':
        modal_token = DEFAULT_VIDEO_TOKEN + '\n' +DEFAULT_AUDIO_TOKEN
    elif modal == 'text':
        modal_token = ''
    else:
        raise ValueError(f"Unsupported modal: {modal}")


    # 1. vision preprocess (load & transform image or video).

    if modal == 'text' or modal == 'audio':
        tensor = [(torch.zeros(32, 3, 384, 384).cuda().half(), "video")]
    else:
        if "video" in modal:
            vi_modal = "video"
        else:
            vi_modal = "image"

        if isinstance(image_or_video, transformers.image_processing_base.BatchFeature):
            # 处理 BatchFeature 中的所有 tensor
            processed_data = transformers.image_processing_base.BatchFeature({
                'pixel_values_videos': image_or_video['pixel_values_videos'][0].half().cuda(),
                'video_grid_thw': image_or_video['video_grid_thw'][0].cuda()
            })
        else:
            # 处理普通 tensor
            processed_data = image_or_video.half().cuda()
        tensor = [(processed_data, vi_modal)]

    
    if audio is not None:
        audio = audio.half().cuda()

    # 2. text preprocess (tag process & generate prompt).
    if isinstance(instruct, str):
        message = [{'role': 'user', 'content': modal_token + '\n' + instruct}]
    elif isinstance(instruct, list):
        message = copy.deepcopy(instruct)
        message[0]['content'] = modal_token + '\n' + message[0]['content']
    else:
        raise ValueError(f"Unsupported type of instruct: {type(instruct)}")


    
    if model.config.model_type in ['HumanOmni', 'HumanOmni_mistral', 'HumanOmni_mixtral']:
        system_message = [
            {'role': 'system', 'content': (
            """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
            """\n"""
            """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
            }
        ]
    else:
        system_message = []

    message = system_message + message
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    # add modal warpper tokken
    if model.config.mm_use_x_start_end:
        prompt = prompt.replace("<video>", "<vi_start><video><vi_end>").replace("<image>", "<im_start><image><im_end>").replace("<audio>", "<au_start><audio><au_end>")

    input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors='pt').unsqueeze(0).long().cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    # 3. generate response according to visual signals and prompts. 
    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
    top_p = kwargs.get('top_p', 0.9)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)


    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            images=tensor,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
            prompts=question_prompt,
            audios=audio
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs
