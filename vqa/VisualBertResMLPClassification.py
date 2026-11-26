'''
Description     : VisualBertResMLP Classification Model
Author          : Lalithkumar Seenivasan, Mobarakol Islam, Adithya Krishna, Hongliang Ren
Paper           : Surgical-VQA: Visual Question Answering in Surgical Scenes Using Transformers
Lab             : MMLAB, National University of Singapore
Acknowledgement : Code adopted from the official implementation of VisualBertModel from 
                  huggingface/transformers (https://github.com/huggingface/transformers.git) and modified.
'''

import torch
from torch import nn
from torchvision import models
from transformers import VisualBertConfig
from vqa.VisualBertResMLP import VisualBertResMLPModel
import torch.nn.functional as F

class Adapter(nn.Module):
    def __init__(self):
        super(Adapter, self).__init__()
        # Step 1: 改变特征维度，从 256 到 512
        self.feature_reduction = nn.Linear(768, 512)
        
        # Step 2: 降低空间维度，从 196 到 25
        self.spatial_reduction = nn.Linear(49, 25)
        
        # 可选：激活函数和归一化层，增加非线性和平滑性
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(512)

    def forward(self, x):
        # x: (B, 196, 256)

        # Step 1: (B, 196, 256) -> (B, 196, 512)
        x = self.feature_reduction(x)
        
        # Step 2: 激活函数和归一化层
        x = self.activation(x)
        x = self.layer_norm(x)

        # Step 3: 转置并进行空间维度降维
        x = x.transpose(1, 2)  # (B, 196, 512) -> (B, 512, 196)
        x = self.spatial_reduction(x)  # (B, 512, 196) -> (B, 512, 25)

        # Step 4: 转置回去
        x = x.transpose(1, 2)  # (B, 512, 25) -> (B, 25, 512)
        
        return x

'''
VisualBertResMLP Classification Model
'''
class VisualBertResMLPClassification(nn.Module):
    def __init__(self, vocab_size, layers, n_heads, num_class = 10, token_size = 26):
        super(VisualBertResMLPClassification, self).__init__()
        VBconfig = VisualBertConfig(vocab_size= vocab_size, visual_embedding_dim = 512, num_hidden_layers = layers, num_attention_heads = n_heads, hidden_size = 2048)
        self.VisualBertResMLPEncoder = VisualBertResMLPModel(VBconfig, token_size = token_size)
        self.classifier = nn.Linear(VBconfig.hidden_size, num_class)

        self.adapter = Adapter()
        

    def forward(self, inputs, visual_embeds):
        # prepare visual embedding
        
        # visual_embeds = self.ViT_seg(visual_embeds)
        visual_embeds = self.adapter(visual_embeds.view(-1,49,768))
        
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long).to(visual_embeds.device)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float).to(visual_embeds.device)

        # append visual features to text
        inputs.update({
                        "visual_embeds": visual_embeds,
                        "visual_token_type_ids": visual_token_type_ids,
                        "visual_attention_mask": visual_attention_mask,
                        "output_attentions": True
                        })
                        
        inputs['input_ids'] = inputs['input_ids'].to(visual_embeds.device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(visual_embeds.device)
        inputs['attention_mask'] = inputs['attention_mask'].to(visual_embeds.device)
        inputs['visual_token_type_ids'] = inputs['visual_token_type_ids'].to(visual_embeds.device)
        inputs['visual_attention_mask'] = inputs['visual_attention_mask'].to(visual_embeds.device)

        # Encoder output
        outputs = self.VisualBertResMLPEncoder(**inputs)
        
        # Classification layer
        outputs = self.classifier(outputs['pooler_output'])
        return outputs