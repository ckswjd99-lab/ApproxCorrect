import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2, numpy as np

from detectron2.structures import Instances
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.backbone.vit import Block, window_partition, window_unpartition, get_abs_pos

from typing import List, Dict, Tuple

from .counter.base import ExtendedModule
from .counter.counting import CountedAdd, CountedMatmul, AddDecomposedRelPos
from .utils import register_cache, cuda_timer
from .subops import create_dmask

class MaskedRCNN_ViT_FPN_AppCorr(ExtendedModule):
    def __init__(self, model, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
):
        super().__init__()

        # Base model
        self.model: GeneralizedRCNN = model.to(device)
        self.device = device

        # Counting module
        self.add = CountedAdd()
        self.matmul = CountedMatmul()
        for block in self.model.backbone.net.blocks:
            block.attn.rel_pos_module = AddDecomposedRelPos()

        # AppCorr settings
        self.approx_level = 1
        self.prate_attn = 0
        self.dmask_thres = 0.3

        # Debug settings
        self.verbose = False
        self.debug_time = False
        self.num_inferences = 0
        self.eta_approx = {
            "vit": 0.0,
            "attention": 0.0,
            "encoder": 0.0,
        }
        self.eta_correct = {
            "vit": 0.0,
            "attention_global": 0.0,
            "attention_windowed": 0.0,
            "encoder": 0.0,
        }
        self.eta_etc = {
            "postprocess": 0.0,
        }
        self.dindice_alive = 0
        self.last_cache_size = 0
    
    def set_approx_level(self, level: int):
        self.approx_level = level

    def set_prate_attn(self, prate: float):
        self.prate_attn = prate

    def set_dmask_thres(self, thres: float):
        self.dmask_thres = thres

    def set_verbose(self, verbose: bool):
        self.verbose = verbose

    def set_debug_time(self, debug: bool):
        self.debug_time = debug
    
    def avg_timecount(self):
        avg_approx = {key: val / max(1, self.num_inferences) for key, val in self.eta_approx.items()}
        avg_correct = {key: val / max(1, self.num_inferences) for key, val in self.eta_correct.items()}
        avg_etc = {key: val / max(1, self.num_inferences) for key, val in self.eta_etc.items()}
        return avg_approx, avg_correct, avg_etc
    
    def avg_dindice_alive(self):
        return self.dindice_alive / max(1, self.num_inferences)
    
    def reset_timecount(self):
        self.num_inferences = 0
        self.dindice_alive = 0.0
        for key in self.eta_approx:
            self.eta_approx[key] = 0.0
        for key in self.eta_correct:
            self.eta_correct[key] = 0.0

    def forward(self, inputs: List[Dict[str, torch.Tensor]]) -> List[Dict[str, Instances]]:
        if len(inputs) != 1:
            raise NotImplementedError("Only batch size 1 is supported in AppCorr inference.")
        
        # Image pyramid
        image_pyramid = []
        image_original = inputs[0]["image"].cpu().numpy().transpose(1, 2, 0)  # HWC, numpy
        for l in range(self.approx_level + 1):
            downsampled = image_original
            for _ in range(l):
                downsampled = cv2.pyrDown(downsampled)
            for _ in range(l):
                downsampled = cv2.pyrUp(downsampled)
            image_pyramid.append(torch.from_numpy(downsampled.transpose(2, 0, 1)).to(self.device))  # CHW, tensor

        # Preprocess
        image_pyramid = [
            self.model.preprocess_image([{
                "file_name": inputs[0]["file_name"], 
                "height": inputs[0]["height"],
                "width": inputs[0]["width"],
                "image_id": inputs[0]["image_id"],
                "image": image,
            }]) for image in image_pyramid
        ]

        # Approximate inference with lowest resolution
        images = image_pyramid[-1]

        x, cache_features = self.approx_vit(images.tensor)

        # Correct inference
        for l in range(self.approx_level-1, -1, -1):
            diff_l = image_pyramid[l].tensor - image_pyramid[l+1].tensor
            x, cache_features = self.correct_vit(diff_l, cache_features)
        
        detections = self.postprocess(x, inputs, images)

        # Update time count
        self.num_inferences += 1
        
        self.last_cache_size = 0
        for fname, fvec in cache_features.items():
            if isinstance(fvec, torch.Tensor):
                self.last_cache_size += fvec.numel() * fvec.element_size()
        
        return detections
    
    @cuda_timer("eta_approx", "vit")
    def approx_vit(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        # Initilization
        new_cache_features = {}
        vit = self.model.backbone.net
        
        # Patch embedding
        x = vit.patch_embed(x)
        if vit.pos_embed is not None:
            ape = get_abs_pos(
                vit.pos_embed, vit.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )
            x = self.add(x, ape)

        # Encoder blocks
        register_cache(f"vit_input", new_cache_features, x)
        for bidx, block in enumerate(vit.blocks):
            x, new_cache_features = self.approx_encoder(block, x, new_cache_features, self.prate_attn, f"bidx{bidx}")

        return x, new_cache_features
    
    @cuda_timer("eta_correct", "vit")
    def correct_vit(
        self, dinput: torch.Tensor, cache_features: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        vit = self.model.backbone.net

        # create mask
        dmask, dindice = create_dmask(dinput, threshold=self.dmask_thres)
        self.dindice_alive += dindice.size(0) / dmask.numel()

        # Patch embedding
        dx = F.conv2d(dinput, vit.patch_embed.proj.weight, None, stride=vit.patch_embed.proj.stride, padding=vit.patch_embed.proj.padding)
        dx = dx.permute(0, 2, 3, 1)  # BCHW -> BHWC

        # Encoder blocks
        x = cache_features[f"vit_input"]
        x = register_cache(f"vit_input", cache_features, x+dx)
        for bidx, block in enumerate(vit.blocks):
            x, cache_features = self.correct_encoder(block, x, cache_features, self.prate_attn, dindice, f"bidx{bidx}")

        return x, cache_features

    @cuda_timer("eta_approx", "encoder")
    def approx_encoder(
        self,
        block: Block,
        x: torch.Tensor,
        cache_features: Dict[str, torch.Tensor],
        prate_attn: float,
        prefix: str
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        shortcut = x
        x = block.norm1(x)
        
        # Window partition
        if block.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, block.window_size)

        # Attention
        x, cache_features = self.approx_attention(block.attn, x, cache_features, prate_attn, prefix)
        
        # Reverse window partition
        if block.window_size > 0:
            x = window_unpartition(x, block.window_size, pad_hw, (H, W))

        # Residual connection
        x = self.add(shortcut, block.drop_path(x))
        
        # Norm & MLP
        x, cache_features = self.approx_mlp(block, x, cache_features, prefix)
    
        return x, cache_features

    @cuda_timer("eta_correct", "encoder")
    def correct_encoder(
        self,
        block: Block,
        x_new: torch.Tensor,
        cache_features: Dict[str, torch.Tensor],
        prate_attn: float,
        dindice: torch.Tensor,
        prefix: str
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        shortcut = x_new
        x_new = block.norm1(x_new)
        
        # Window partition
        if block.window_size > 0:
            H, W = x_new.shape[1], x_new.shape[2]
            x_new, pad_hw = window_partition(x_new, block.window_size)

        # Attention
        attn_func = self.correct_attention_global if block.window_size == 0 else self.correct_attention_windowed
        x_new, cache_features = attn_func(
            block.attn, 
            x_new, 
            cache_features,
            prate_attn,
            dindice,
            prefix
        )
        
        # Reverse window partition
        if block.window_size > 0:
            x_new = window_unpartition(x_new, block.window_size, pad_hw, (H, W))

        # Residual connection
        x_new = self.add(shortcut, block.drop_path(x_new))
        
        # Norm & MLP
        x_new, cache_features = self.correct_mlp(block, x_new, cache_features, dindice, prefix)
    
        return x_new, cache_features

    @cuda_timer("eta_approx", "attention")
    def approx_attention(
        self,
        attn: nn.Module,
        x: torch.Tensor,
        cache_features: Dict[str, torch.Tensor],
        prate_attn: float,
        prefix: str
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        B, H, W, _ = x.shape
        
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = attn.qkv(x).reshape(B, H * W, 3, attn.num_heads, -1).permute(2, 0, 3, 1, 4)
        
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * attn.num_heads, H * W, -1).unbind(0)

        qkT = self.matmul((q * attn.scale), k.transpose(-2, -1))

        if attn.use_rel_pos:
            attn_score = attn.rel_pos_module(qkT, q, attn.rel_pos_h, attn.rel_pos_w, (H, W), (H, W))

        attn_prob = attn_score.softmax(dim=-1)
        register_cache(f"{prefix}_attn_prob", cache_features, attn_prob)

        x = self.matmul(attn_prob, v).view(B, attn.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)        
        x = attn.proj(x)
        register_cache(f"{prefix}_attn_proj", cache_features, x)

        return x, cache_features

    @cuda_timer("eta_correct", "attention_windowed")
    def correct_attention_windowed(
        self,
        attn: nn.Module,
        x: torch.Tensor,
        cache_features: Dict[str, torch.Tensor],
        prate_attn: float,
        dindice: torch.Tensor,
        prefix: str
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        B, H, W, _ = x.shape
        
        v_dim = attn.qkv.out_features // 3
        C = v_dim // attn.num_heads

        v_weight = attn.qkv.weight[2*v_dim: , :]
        v_bias = attn.qkv.bias[2*v_dim: ] if attn.qkv.bias is not None else None

        v = F.linear(x, v_weight, v_bias)
        v = v.reshape(B, H * W, attn.num_heads, C).transpose(1, 2).reshape(B * attn.num_heads, H * W, C)

        attn_prob = cache_features[f"{prefix}_attn_prob"]
        attn_out = self.matmul(attn_prob, v).view(B, attn.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        attn_proj = attn.proj(attn_out)

        return attn_proj, cache_features
    
    @cuda_timer("eta_correct", "attention_global")
    def correct_attention_global(
        self,
        attn: nn.Module,
        x: torch.Tensor,
        cache_features: Dict[str, torch.Tensor],
        prate_attn: float,
        dindice: torch.Tensor,
        prefix: str
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        dindice = dindice[:, 0]  # flatten index
        num_alive = dindice.size(0)

        B, H, W, _ = x.shape
        
        v_dim = attn.qkv.out_features // 3
        C = v_dim // attn.num_heads

        v_weight = attn.qkv.weight[2*v_dim: , :]
        v_bias = attn.qkv.bias[2*v_dim: ] if attn.qkv.bias is not None else None

        v = F.linear(x, v_weight, v_bias)
        v = v.reshape(B, H * W, attn.num_heads, C).transpose(1, 2).reshape(B * attn.num_heads, H * W, C)

        # attn_prob = cache_features[f"{prefix}_attn_prob"]
        # attn_out = self.matmul(attn_prob, v).view(B, attn.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        # attn_proj = attn.proj(attn_out)

        attn_prob = cache_features[f"{prefix}_attn_prob"]

        attn_prob_sampled = attn_prob[:, dindice]  # TODO: do it during approximation for efficiency

        attn_out_sampled = self.matmul(attn_prob_sampled, v)
        attn_out_sampled = attn_out_sampled.permute(1, 0, 2).reshape(B, num_alive, -1)
        attn_proj_sampled = attn.proj(attn_out_sampled)

        attn_proj_old = cache_features[f"{prefix}_attn_proj"]
        attn_proj_old = attn_proj_old.reshape(B, H * W, -1)
        attn_proj_old[:, dindice] = attn_proj_sampled

        attn_proj = attn_proj_old.reshape(B, H, W, -1)

        return attn_proj, cache_features

    @cuda_timer("eta_approx", "ffn")
    def approx_mlp(
        self,
        block: Block,
        x: torch.Tensor,
        cache_features: Dict[str, torch.Tensor],
        prefix: str
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        shortcut = x
        x = block.norm2(x)

        mlp_output = block.mlp.fc1(x)
        mlp_output = block.mlp.act(mlp_output)
        mlp_output = block.mlp.fc2(mlp_output)
        register_cache(f"{prefix}_mlp_input", cache_features, x)
        register_cache(f"{prefix}_mlp_output", cache_features, mlp_output)

        x = self.add(shortcut, block.drop_path(mlp_output))

        if block.use_residual_block:
            x = block.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return x, cache_features
    
    @cuda_timer("eta_correct", "ffn")
    def correct_mlp(
        self,
        block: Block,
        x_new: torch.Tensor,
        cache_features: Dict[str, torch.Tensor],
        dindice: torch.Tensor,
        prefix: str
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        shortcut = x_new
        mlp_input_new = block.norm2(x_new)
        mlp_input_old = cache_features[f"{prefix}_mlp_input"]

        mlp_input_flat = mlp_input_new.view(-1, mlp_input_new.size(-1))
        mlp_input_sampled = F.embedding(dindice, mlp_input_flat)
        mlp_output_sampled = F.linear(mlp_input_sampled, block.mlp.fc1.weight, block.mlp.fc1.bias)
        mlp_output_sampled = block.mlp.act(mlp_output_sampled)
        mlp_output_sampled = F.linear(mlp_output_sampled, block.mlp.fc2.weight, block.mlp.fc2.bias)

        mlp_output_old = cache_features[f"{prefix}_mlp_output"]
        mlp_output_old.view(-1, mlp_output_old.size(-1))[dindice] = mlp_output_sampled
        mlp_output_new = mlp_output_old

        x_new = self.add(shortcut, block.drop_path(mlp_output_new))

        if block.use_residual_block:
            x_new = block.residual(x_new.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return x_new, cache_features

    @cuda_timer("eta_etc", "postprocess")
    def postprocess(self, x: torch.Tensor, input, images):
        backbone = self.model.backbone
        net = backbone.net

        bottom_up_features = {net._out_features[0]: x.permute(0, 3, 1, 2)}

        features = bottom_up_features[backbone.in_feature]  # (1, 768, 64, 64)
        results = []
        for stage in backbone.stages:
            results.append(stage(features))

        if backbone.top_block is not None:
            if backbone.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[backbone.top_block.in_feature]
            else:
                top_block_in_feature = results[backbone._out_features.index(backbone.top_block.in_feature)]
            results.extend(backbone.top_block(top_block_in_feature))
        assert len(backbone._out_features) == len(results)
        features = {f: res for f, res in zip(backbone._out_features, results)}
        
        # RPN
        proposals, _ = self.model.proposal_generator(images, features, None)
        results, _ = self.model.roi_heads(images, features, proposals, None)
        detections = self.model._postprocess(results, input, images.image_sizes)

        return detections
