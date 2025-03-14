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
import time
import os
from abc import ABC, abstractmethod
import math
import re
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from .projector import load_mm_projector, build_vision_projector, build_audio_projector
from .encoder import build_vision_tower, build_audio_tower
from ..constants import IGNORE_INDEX, NUM_FRAMES, MODAL_INDEX_MAP, MODAL_INDEX_REMAP
from transformers import BertModel, BertTokenizer

class SFDynamicCompressor(nn.Module):
    def __init__(self, model_args, vision_tower):
        super().__init__()

        self.out_channels = vision_tower.hidden_size
        self.mid_channel = 256

        self.vlm_query_projector = nn.Linear(self.out_channels, self.mid_channel)
        self.vlm_key_projector = nn.Linear(self.out_channels, self.mid_channel)

    def downsample(self, x):
        return F.avg_pool2d(x, 2, 2)
    
    def downsample_4(self, x):
        return F.avg_pool2d(x, 4, 4)
        
    def forward(self, image_features, image_size=None):
        if image_size is None:
            W = int(math.sqrt(image_features.shape[1]))
            H = int(W)
        else:
            H, W = image_size
        image_features = einops.rearrange(image_features, 't (r w) h -> t r w h', r = H)
        T, H, W, C = image_features.shape
        image_features = image_features.unsqueeze(0)
        B = 1
        fast_feature = F.avg_pool2d(image_features.permute(0, 1, 4, 2, 3).view(B*T, C, H, W), 2, 2) # B * T, C, H // 2, W //2 
        fast_feature = fast_feature.view(B*T, C, -1)
        fast_feature = fast_feature.permute(0, 2, 1).view(B, T, -1, C).view(B, -1, C)
        
        index = torch.arange(1, T, 4)
        if len(index) == 0:
            index = torch.tensor([0])
        slow_feature = image_features[:, index, :, :, :].view(B, -1, C)

        final_feature = torch.cat([fast_feature, slow_feature], dim=1)  
        return final_feature


class HumanOmniMetaModel:

    def __init__(self, config):
        super(HumanOmniMetaModel, self).__init__(config)
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)


        # Comment out this part of the code during training to avoid repeated initialization.
        num_branches = 3
        bert_model = "/mnt/data/jiaxing.zjx/code/R1-V-Qwen/R1-V/bert-base-uncased"
        self.bert_model =  BertModel.from_pretrained(bert_model)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
        modules = [nn.Linear(self.bert_model.config.hidden_size, 3584)]
        modules.append(nn.GELU())
        modules.append(nn.Linear(3584, num_branches))
        self.bert_gate = nn.Sequential(*modules)
        self.bert_softmax = nn.Softmax(dim=1)
       # self.feature_compressor = SFDynamicCompressor(config, self.vision_tower)
        #####
        
        if hasattr(config, "mm_audio_tower"):
            self.audio_tower = build_audio_tower(config, delay_load=True)
            self.config.audio_hidden_size = getattr(self.audio_tower, "hidden_size", 1280)
            self.audio_projector = build_audio_projector(config, vision_cfg=self.audio_tower.config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_audio_tower(self):
        audio_tower = getattr(self, "audio_tower", None)
        return audio_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if model_args.audio_tower:
            self.initialize_audio_modules(model_args, fsdp)

        if pretrain_mm_mlp_adapter is not None:
            if os.path.exists(pretrain_mm_mlp_adapter):
                is_local = True
                if os.path.isdir(pretrain_mm_mlp_adapter):
                    mm_projector_weights = load_mm_projector(pretrain_mm_mlp_adapter)
                else:
                    mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            else:
                # Support loading projector weights from remote HuggingFace model hub
                is_local = False
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.replace('mm_projector.bin', '')
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.strip('/').strip('\\').strip()
                mm_projector_weights = load_mm_projector(pretrain_mm_mlp_adapter)

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            # self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            # set strict=False to avoid missing key error regarding bert.embeddings.position_ids
          
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=True)

   
     #  self.feature_compressor = SFDynamicCompressor(model_args, vision_tower)
        num_branches = 3
        bert_model = "bert-base-uncased"
        self.bert_model =  BertModel.from_pretrained(bert_model)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
        # self.bert_gate = nn.Linear(self.bert_model.config.hidden_size, num_branches)
        modules = [nn.Linear(self.bert_model.config.hidden_size, 3584)]
        modules.append(nn.GELU())
        modules.append(nn.Linear(3584, num_branches))
        self.bert_gate = nn.Sequential(*modules)
        self.bert_softmax = nn.Softmax(dim=1)


    def initialize_audio_modules(self, model_args, fsdp=None):
        audio_tower = model_args.audio_tower
        pretrain_audio_mlp_adapter = model_args.pretrain_audio_mlp_adapter
        self.config.mm_audio_tower = audio_tower
        self.config.mm_audio_projector_type = getattr(model_args, "mm_audio_projector_type", "mlp2x_gelu")
        if self.get_audio_tower() is None:
            audio_tower = build_audio_tower(model_args)
            
            if fsdp is not None and len(fsdp) > 0:
                self.audio_tower = [audio_tower]
            else:
                self.audio_tower = audio_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                audio_tower = self.audio_tower[0]
            else:
                audio_tower = self.audio_tower
            audio_tower.load_model()
        self.config.audio_hidden_size = getattr(audio_tower, "hidden_size", 1280)
        if getattr(self, "audio_projector", None) is None:
            self.audio_projector = build_audio_projector(self.config, vision_cfg=audio_tower.config)
        else:
            # In case it is frozen by LoRA
            for p in self.audio_projector.parameters():
                p.requires_grad = True
        def get_w(weights, keyword):
            return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
        if pretrain_audio_mlp_adapter is not None:
            audio_projector_weights = torch.load(pretrain_audio_mlp_adapter, map_location="cpu")
        #     # import pdb; pdb.set_trace()
            incompatible_keys = self.audio_projector.load_state_dict(get_w(audio_projector_weights, "audio_projector"))
            print(f"load audio projector: {incompatible_keys}")
        num_trainable_parameters = sum(p.numel() for p in self.audio_projector.parameters() if p.requires_grad) / 1e6
        print(f"Number of trainable parameters in audio projector: {num_trainable_parameters}M")


class HumanOmniMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def num_frames(self):
        if hasattr(self.config, 'num_frames'):
            return self.config.num_frames
        else:
            return NUM_FRAMES

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_audio_tower(self):
        return self.get_model().get_audio_tower()

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()

        height, weight = image_feature.shape[2:]
        scaled_shape = [math.ceil(height / stride), math.ceil(weight / stride)]
        image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def encode_images_or_videos(self, images, device=None,prompts=None):

        num_frames = self.config.num_frames if hasattr(self.config, 'num_frames') else NUM_FRAMES
        current_device = torch.cuda.current_device()
    
        data_batch = []
        video_idx_in_batch = []
        # for i, (data, modal) in enumerate(images):
        #     data = data
        #     video_idx_in_batch.append(i)
        #     data_batch.append(data)

        for i, image in enumerate(images):
            if isinstance(image, (list, tuple)):
                # 如果image是列表或元组，且长度至少为1
                if len(image) >= 1:
                    data = image[0]  # 获取data部分
                    # 可选：如果有modal部分，则获取它；否则忽略
                    modal = image[1] if len(image) > 1 else None
                else:
                    print(f"Warning: Element at index {i} is a list/tuple but does not contain any elements and will be skipped.")
                    continue
            elif isinstance(image, torch.Tensor):
                # 如果image是张量，则直接作为data
                data = image
                modal = 'video'  # 或者根据需要设置其他默认值
            else:
                print(f"Warning: Element at index {i} is of unsupported type {type(image)} and will be skipped.")
                continue
            video_idx_in_batch.append(i)
            # 将data添加到data_batch
            data_batch.append(data)

            
        batch_size = len(data_batch)
        split_sizes = [image.shape[0] for image in data_batch]
        frames = torch.cat([image for image in data_batch], dim=0)
        # ddd
        frames_features = self.get_model().get_vision_tower()(frames)
        video_features = einops.rearrange(frames_features, '(b t) n h -> b t n h', b = batch_size)
        body_features = video_features       
        face_features = frames_features         
        video_features, body_features, face_features = self.get_model().mm_projector(video_features, body_features, face_features)
        face_features = einops.rearrange(face_features, '(b t) n h -> b t n h', b = batch_size)
        
        inputs_bert = prompts
        # Get BERT features
        outputs_bert = self.get_model().bert_model(**inputs_bert)
        last_hidden_state_bert = outputs_bert.last_hidden_state
        # Use [CLS] token representation
        cls_token_embedding_bert = last_hidden_state_bert[:, 0, :]
        # Calculate branch probabilities
        logits = self.get_model().bert_gate(cls_token_embedding_bert)
        branch_probs = self.get_model().bert_softmax(logits)

        image_features = []
        for idx, image_feat in enumerate(face_features):
            if idx in video_idx_in_batch:
                image_features.append(self.get_2dPool(image_feat))
            else:
                image_features.append(image_feat)
        face_features = image_features

        new_image_features = []
        for image_idx, face_feature in enumerate(face_features):
            video_feature = video_features[image_idx]
            body_feature = body_features[image_idx]
            if image_idx in video_idx_in_batch:  # video operations
                face_feature = face_feature.flatten(0, 1)
                image_feature = video_feature * branch_probs[image_idx][0] + body_feature * branch_probs[image_idx][1] + face_feature * branch_probs[image_idx][2]
                ###如果有slow fast分支，取消注释
                # image_feature = einops.rearrange(image_feature, '(t n) h -> t n h', t = num_frames)
                # image_feature = self.get_model().feature_compressor(image_feature)
                new_image_features.append(image_feature)      

        return new_image_features

      
    def encode_audios(self, audios):
        audio_features = self.get_model().get_audio_tower()(audios).permute(0, 2, 1).contiguous() #b, t, c -> b, c, t   # torch.Size([1, 1280, 1500])
        audio_features = torch.nn.functional.avg_pool1d(audio_features, kernel_size=3, stride=3).permute(0, 2, 1).contiguous() # torch.Size([1, 1280, 500])
        audio_features = self.get_model().audio_projector(audio_features)
        return audio_features


    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, prompts=None,audios=None
    ):

        if audios is not None:
            if len(audios.shape) == 4 and audios.shape[1] == 1:
                audios = audios.squeeze(1)  # 移除第一维
        vision_tower = self.get_vision_tower()
        audio_tower = self.get_audio_tower()
        # NOTE: text-only situation
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, attention_mask, past_key_values, None, labels
        device_ = input_ids.device
        mm_features = self.encode_images_or_videos(images ,device_,prompts)

        if audios is not None and audio_tower is not None:
            audio_features = self.encode_audios(audios)
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_mm_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_multimodals = sum((cur_input_ids == mm_token_idx).sum() for mm_token_idx in MODAL_INDEX_MAP.values())
            # pure text input
            if num_multimodals == 0:
                half_len = cur_input_ids.shape[0] // 2
                cur_mm_features = mm_features[cur_mm_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_mm_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_mm_idx += 1 
                continue

            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            mm_token_indices = torch.where(sum([cur_input_ids == mm_token_idx for mm_token_idx in MODAL_INDEX_MAP.values()]))[0]
            while mm_token_indices.numel() > 0:
                mm_token_start = mm_token_indices[0]
                cur_modal = MODAL_INDEX_REMAP[cur_input_ids[mm_token_start].item()]
                if cur_modal in ["<image>", "<video>"]:
                    cur_mm_idx += 1
                    cur_mm_features = mm_features[batch_idx]
                    if len(cur_mm_features.size())==3:
                        cur_mm_features=cur_mm_features.flatten(0,1)
                elif cur_modal in  ["<audio>"] and audio_tower is not None:
                    cur_mm_features = audio_features[batch_idx]

                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:mm_token_start])) 
                cur_new_input_embeds.append(cur_mm_features)
                if labels is not None:
                    cur_new_labels.append(cur_labels[:mm_token_start])
                    cur_new_labels.append(torch.full((cur_mm_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_labels = cur_labels[mm_token_start+1:]

                cur_input_ids = cur_input_ids[mm_token_start+1:] 
                mm_token_indices = torch.where(sum([cur_input_ids == mm_token_idx for mm_token_idx in MODAL_INDEX_MAP.values()]))[0]

            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        # padding
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]
        return None, attention_mask, past_key_values, new_input_embeds, new_labels
