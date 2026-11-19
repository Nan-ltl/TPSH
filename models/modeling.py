# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random
from unittest.mock import patch

import math

from os.path import join as pjoin

import ml_collections
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from transformers.models.flava.modeling_flava import FlavaSelfOutput

import models.configs as configs


from transformers import ViTModel, CLIPProcessor, CLIPModel, CLIPTextModelWithProjection
import torchvision.transforms as transforms




logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}



class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # [bs, 12, num_patch]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # [bs, 196]
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        patch_size = _pair(config.patches["size"])

        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                   out_channels=config.hidden_size,
                                   kernel_size=patch_size,
                                   stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))



def min_max_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    x_min = x.min(dim=dim, keepdim=True).values
    x_max = x.max(dim=dim, keepdim=True).values
    x_norm = (x - x_min) / (x_max - x_min + eps)
    return x_norm
class Score_Module(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.Tensor([0.0]), requires_grad=True,)
    def forward(self, tokens, attn_weights):
        embed_patches = tokens[:, 1:]

        embed_patches_norm = F.normalize(embed_patches, dim=-1)
        score_mlp_norm = self.mlp(embed_patches_norm).squeeze(-1)  # [bs, 196]

        embed_glo_norm = F.normalize(embed_patches.mean(dim=1, keepdim=True), dim=-1)
        img_spatial_self_attention_score = (embed_glo_norm * embed_patches_norm).sum(dim=-1)  # [bs, 196]

        img_spatial_self_attention_score_norm = min_max_normalize(img_spatial_self_attention_score, dim=1)
        score_self_attn = torch.mean(attn_weights[:, :, 0, 1:197], dim=1)  # [bs, 196,]
        score_self_attn_norm = min_max_normalize(score_self_attn, dim=1)
        scores = 0.2 * score_mlp_norm + 0.4 * img_spatial_self_attention_score_norm + 0.4 * score_self_attn_norm
        return scores


class Patch_Selector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.img_mask = Score_Module(config, dim=config.img_mask_dim)
        self.img_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.img_norm_new = LayerNorm(config.hidden_size, eps=1e-6)
        self.text_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.MSE = nn.MSELoss()
        self.i2t = i2t(config.hidden_size, 768, 768, config)
        self.proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.num_classes),
            nn.Sigmoid(),
        )
        self.tau = 1.0
    def forward(self, tokens, att_weights, text_emb=None, retrieval=False, labels=None):
        tokens = self.img_norm(tokens)
        scores = self.img_mask(tokens, att_weights)
        num_patches = torch.full((tokens.shape[0],), 196).cuda()
        if not retrieval:
            loss_i2t_align, score = self.i2t(tokens, num_patches, text_emb, retrieval, labels)

        logits = torch.stack([scores, 1.0 - scores], dim=-1)  # [bs, 196, 2]
        y = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)
        mask = y[:, :, 0]  # [bs, 196]
        cls_tokens = tokens[:, 0]
        max_len = 0
        mean_num_patch = 0
        # select patch
        for line in mask:
            sum = (line == 1).sum()
            mean_num_patch += sum
            if sum > max_len:
                max_len = sum
        mean_num_patch = mean_num_patch / tokens.shape[0]
        if max_len < self.config.least_chosen_patches:
            max_len = self.config.least_chosen_patches

        for i in range(tokens.shape[0]):
            indicate = torch.where(mask[i] == 1)[0]
            if indicate.shape[0] < self.config.least_chosen_patches:
                indicate = torch.argsort(scores[i], descending=True)[:self.config.least_chosen_patches]
                indicate = indicate[torch.argsort(indicate)]
            select_token = tokens[i, 1:][indicate]
            padding_token = torch.zeros(max_len - select_token.shape[0], self.config.hidden_size).cuda()
            new_token = torch.cat([select_token, padding_token], dim=0).unsqueeze(0)
            if i == 0:
                patch_tokens = new_token
            else:
                patch_tokens = torch.cat([patch_tokens, new_token], dim=0)
        new_tokens = torch.cat([cls_tokens.unsqueeze(1), patch_tokens], dim=1)

        if retrieval:
            return new_tokens, None, None
        else:
            loss_kl = F.kl_div(F.log_softmax(scores,dim=-1), F.softmax(score, dim=-1), reduction='batchmean')
            loss = loss_i2t_align + loss_kl * 0.1
            return new_tokens, loss, mean_num_patch
    def load_from(self, weights, n_block):
        return


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        for _ in range(config.transformer["num_layers"] - 1):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))

        self.select_layer_number = config.select_layer_number
        self.patch_selector = nn.ModuleList()

        for _ in range(self.select_layer_number):
            self.patch_selector.append(Patch_Selector(config))
        self.part_layer = Block(config)
        self.part_norm = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_states, text_emb, retrieval=False, labels=None):
        loss_list = []
        for i in range(len(self.layer)):
            hidden_states, weights = self.layer[i](hidden_states)
            if (i + 1) > (len(self.layer) - self.select_layer_number):
                hidden_states, loss, mean_num_patch = self.patch_selector[i - len(self.layer) + self.select_layer_number](hidden_states, weights, text_emb, retrieval, labels)
                loss_list.append(loss)
        part_states, part_weights = self.part_layer(hidden_states)
        part_encoded = self.part_norm(part_states)

        return part_encoded, loss_list, mean_num_patch

class Transformer(nn.Module):
    def  __init__(self, config, img_size):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config)
    def forward(self, input_ids, text_emb=None, retrieval=False, labels=None):
        embedding_output = self.embeddings(input_ids)
        part_encoded, loss_list, mean_num_patch = self.encoder(embedding_output, text_emb, retrieval, labels)
        loss_list = loss_list
        return part_encoded, loss_list, mean_num_patch

class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843):
        super(VisionTransformer, self).__init__()
        self.config = config
        self.hash_bit = self.config.hash_bit
        self.num_classes = num_classes
        self.classifier = self.config.classifier
        self.transformer = Transformer(self.config, img_size)
        self.load_from(np.load(config.pretrained_dir))
        self.hash_layer = nn.Linear(config.hidden_size, config.hash_bit)
        self.Myloss = MyHashloss(config)

        self.clip_model = CLIPTextModelWithProjection.from_pretrained(config.clip_path)
        for parameters in self.clip_model.parameters():
            parameters.requires_grad = False
        self.processor = CLIPProcessor.from_pretrained(config.clip_path)
        self.text_layer = nn.Linear(768, config.hidden_size)


    def forward(self, x, labels=None, sentence = None, retrieval: bool=False):
        if retrieval:
            part_tokens, _, _ = self.transformer(x, None, retrieval, None)  # [bs, 13, 768]
            cls_token = part_tokens[:, 0, :]
            u_hash = self.hash_layer(cls_token)  # [bs, bit]
            return u_hash
        else:
            inputs = self.processor(text=sentence, return_tensors="pt", max_length=77, padding='max_length', truncation=True)
            inputs.data["input_ids"] = inputs.data["input_ids"].cuda()
            inputs.data["attention_mask"] = inputs.data["attention_mask"].cuda()
            text_embeddings_word = self.clip_model(**inputs).last_hidden_state
            # text_embeddings_cls = self.clip_model(**inputs).text_embeds
            # text_embeddings_cls = self.text_layer(text_embeddings_cls)


            part_tokens, loss_list, mean_num_patch = self.transformer(x, text_embeddings_word, retrieval, labels)  # [bs, 13, 768]
            cls_token = part_tokens[:, 0, :]

            loss_select = 0
            for loss in loss_list:
                loss_select += loss

            u_hash = self.hash_layer(cls_token)

            loss_dict = {}

            loss_total = 0
            loss_cross_entropy, loss_qua_reg= self.Myloss(u_hash, labels)
            loss_total += loss_cross_entropy
            loss_total += loss_qua_reg
            loss_dict['loss_cross_entropy'] = loss_cross_entropy
            loss_dict['loss_qua_reg'] = loss_qua_reg
            loss_total += loss_select
            loss_dict['loss_select'] = loss_select
            return loss_total, loss_dict, mean_num_patch


    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.part_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.part_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                if bname.startswith('part') == False:
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)

CONFIGS = {
    'ViT-S_16':configs.get_s16_config(),
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'testing': configs.get_testing(),
}

class MyHashloss(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_qua = nn.MSELoss()
        self.hash_cls = nn.Sequential(nn.Linear(self.config.hash_bit, self.config.num_classes))

    def forward(self, u, y):
        y_hat_hash = self.hash_cls(u)
        L_h = self.criterion(y_hat_hash, y)
        L_g = self.criterion_qua(u.abs(), torch.ones(u.shape).cuda())
        return L_h, L_g


class i2t(nn.Module):
    def __init__(self, embed_dim_image, embed_dim_text,
                 embed_dim_hidden, config):
        super(i2t, self).__init__()
        self.config = config
        self.embed_dim_image = embed_dim_image
        self.embed_dim_text = embed_dim_text
        self.embed_dim_hidden = embed_dim_hidden
        self.fc_image = nn.Linear(embed_dim_image, embed_dim_hidden)
        self.fc_text = nn.Linear(embed_dim_text, embed_dim_hidden)
        self.margin = 0.2
        self.max_violation = True
        self.fc_cls_txt = nn.Linear(embed_dim_hidden, config.num_classes)
        self.ln = nn.LayerNorm(embed_dim_hidden)

    def forward(self, embed_image, num_patches, embed_text, retrieval, labels):
        bs = embed_image.shape[0]

        embed_text_glo = torch.mean(embed_text, dim=1)  # [64, 768]
        embed_text_glo = self.fc_text(embed_text_glo)
        embed_text_glo_norm = F.normalize(embed_text_glo, dim=1)  # [bs, embed_dim]

        sim_mat = []
        score_i2t = []
        for i in range(bs):
            num_patches_i = num_patches[i]
            embed_image_i = embed_image[i]
            embed_patch_i = embed_image_i[1:num_patches_i+1]
            embed_patch_i = self.fc_image(embed_patch_i)  # [196, 768]
            embed_patch_i_norm = F.normalize(embed_patch_i, dim=1)  # [196, 768]
            sim_patch2word = torch.einsum('nd,bd->bn', embed_patch_i_norm, embed_text_glo_norm)  # [64, 196]
            score_i2t.append(sim_patch2word[i])
            sim_i2t = sim_patch2word.mean(dim=-1)  # [64]
            sim_mat.append(sim_i2t)

        sim_mat = torch.stack(sim_mat, dim=0)  # [bs, bs]
        score_i2t = torch.stack(score_i2t, dim=0)  # [bs, 196]
        if retrieval:
            return None

        p_text = self.fc_cls_txt(embed_text_glo_norm)
        loss_cls_txt = F.cross_entropy(p_text, labels)

        diagonal = sim_mat.diag().view(bs, 1)
        d1 = diagonal.expand_as(sim_mat)
        cost_s = (self.margin + sim_mat - d1).clamp(min=0)
        labels = F.one_hot(labels, num_classes=self.config.num_classes)
        mask = torch.matmul(labels.float(), labels.T.float()) > 0
        cost_s = cost_s.masked_fill_(mask, 0)
        if self.max_violation:
            cost_s, idx_s = cost_s.max(1)
        loss = cost_s.mean() + loss_cls_txt
        return loss * 0.1, score_i2t
class ViT(nn.Module):
    def __init__(self, config, hash_bit, ratio, num_classes):
        super().__init__()
        self.pm = timm.create_model('vit_base_patch16_224', checkpoint_path=config.pretrained_dir)
        self.hash_layer = nn.Linear(768,hash_bit)
    def forward(self, x, labels=None, sentence = None, retrieval: bool=False):
        x = self.pm.patch_embed(x)
        cls_token = self.pm.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pm.pos_drop(x + self.pm.pos_embed)
        x = self.pm.blocks(x)
        x = self.pm.norm(x)
        return self.hash_layer(x)

if __name__ == "__main__" :
    x = torch.Tensor([[2,3],[2,3],[2,3]])
    print(x)
    indicate = torch.where(x == 2)
    print(indicate)

