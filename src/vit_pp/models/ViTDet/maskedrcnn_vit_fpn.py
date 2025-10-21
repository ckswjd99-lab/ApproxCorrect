import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
import time

from typing import List, Optional, Dict, Tuple, Union

from .modeling.backbone.utils import get_abs_pos, window_partition, window_unpartition, AddDecomposedRelPos

from .eventful_transformer.base import ExtendedModule
from .eventful_transformer.counting import CountedAdd, CountedMatmul

from .differ_utils import diff_convolution

from .modeling.backbone.vit import SimpleFeaturePyramid
from .modeling.meta_arch import GeneralizedRCNN

from .structures import ImageList


def register_cache(fname: str, cache: Dict[str, torch.Tensor], tensor: torch.Tensor):
    cache[fname] = tensor.clone()

    return cache

class MaskedRCNN_ViT_FPN_Contexted(ExtendedModule):
    def __init__(self, device="cuda:0", dataset_name="coco"):
        super().__init__()
        self.idx = 0

        self.device = device

        self.embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1

        # backbone
        self.backbone: SimpleFeaturePyramid = nn.Identity().to(self.device)

        # model
        self.base_model: GeneralizedRCNN = nn.Identity().to(self.device)

        # counting module
        self.add = CountedAdd()
        self.matmul = CountedMatmul()
        self.add_decomposed_rel_pos = AddDecomposedRelPos()
    
    def forward(self, image_ndarray: np.ndarray, *args, **kwargs):
        image_tensor = torch.tensor(image_ndarray, dtype=torch.uint8, device=self.device).permute(2, 0, 1)
        input = [{"image": image_tensor, "height": image_tensor.shape[-2], "width": image_tensor.shape[-1]}]
        images = self.base_model.preprocess_image(input)
        
        # detections = self.base_model(input)
        backbone = self.base_model.backbone
        net = backbone.net

        features = backbone(images.tensor)
        proposals, _ = self.base_model.proposal_generator(images, features, None)
        results, _ = self.base_model.roi_heads(images, features, proposals, None)
        detections = self.base_model._postprocess(results, input, images.image_sizes)
        

        predictions = detections[0]
        boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
        labels = predictions["instances"].pred_classes.cpu().numpy()
        scores = predictions["instances"].scores.cpu().numpy()
        segments = predictions["instances"].pred_masks.cpu().numpy()

        return boxes, segments, labels, scores

    def approx(self, image_ndarray: np.ndarray, *args, **kwargs):
        debug_time = kwargs.get("debug_time", False)

        latency_preproc = []
        latency_attn = []
        latency_ffn = []

        new_cache_feature = {}

        # image_ndarray: (H, W, C)
        image_tensor = torch.tensor(image_ndarray, dtype=torch.float32, device=self.device).permute(2, 0, 1)
        input = [{"image": image_tensor, "height": image_tensor.shape[-2], "width": image_tensor.shape[-1]}]
        images = self.base_model.preprocess_image(input)

        new_cache_feature["input"] = input
        new_cache_feature["images"] = images

        # ViT Preprocess
        ts_preproc_start = time.time()

        backbone = self.base_model.backbone
        net = backbone.net
        
        x = net.patch_embed(images.tensor)
        ape = get_abs_pos(
            net.pos_embed, net.pretrain_use_cls_token, (x.shape[1], x.shape[2])
        )
        if net.pos_embed is not None:
            x = x + ape
        latency_preproc.append(time.time() - ts_preproc_start)

        # EncoderBlocks
        for bidx, block in enumerate(net.blocks):
            register_cache(f"bidx{bidx}_input", new_cache_feature, x)

            shortcut = x
            x_attn = block.norm1(x)

            # Window partition
            ts_attn_start = time.time()
            if block.window_size > 0:
                H, W = x_attn.shape[1], x_attn.shape[2]
                x_attn, pad_hw = window_partition(x_attn, block.window_size)

            # Attention
            B_attn, H_attn, W_attn, _ = x_attn.shape
            qkv = block.attn.qkv(x_attn).reshape(B_attn, H_attn * W_attn, 3, block.attn.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.reshape(3, B_attn * block.attn.num_heads, H_attn * W_attn, -1).unbind(0)
            
            register_cache(f"bidx{bidx}_attn_q", new_cache_feature, q)
            register_cache(f"bidx{bidx}_attn_k", new_cache_feature, k)
            register_cache(f"bidx{bidx}_attn_v", new_cache_feature, v)

            attn_score = (q * block.attn.scale) @ k.transpose(-2, -1)
            register_cache(f"bidx{bidx}_attn_qkT", new_cache_feature, attn_score)
            if block.attn.use_rel_pos:
                attn_score = block.attn.rel_pos_module(attn_score, q, block.attn.rel_pos_h, block.attn.rel_pos_w, (H_attn, W_attn), (H_attn, W_attn))
            register_cache(f"bidx{bidx}_attn_qkT_relpos", new_cache_feature, attn_score)
            
            attn_score = attn_score.softmax(dim=-1)

            x_attn = (attn_score @ v)
            register_cache(f"bidx{bidx}_attn_output", new_cache_feature, x_attn)

            x_attn = x_attn.view(B_attn, block.attn.num_heads, H_attn, W_attn, -1).permute(0, 2, 3, 1, 4).reshape(B_attn, H_attn, W_attn, -1)
            x_attn = block.attn.proj(x_attn)
            
            # Reverse window partition
            if block.window_size > 0:
                x_attn = window_unpartition(x_attn, block.window_size, pad_hw, (H, W))

            latency_attn.append(time.time() - ts_attn_start)

            # FFN
            ts_ffn_start = time.time()

            x_ffn = x_attn
            x_ffn = shortcut + block.drop_path(x_ffn)

            register_cache(f"bidx{bidx}_ffn_input", new_cache_feature, x_ffn)

            shortcut2 = x_ffn
            x_ffn = block.norm2(x_ffn)
            x_ffn = block.mlp(x_ffn)
            register_cache(f"bidx{bidx}_ffn_output", new_cache_feature, x_ffn)
            x_ffn = shortcut2 + block.drop_path(x_ffn)

            if block.use_residual_block:
                x_ffn = block.residual(x_ffn.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

            x = x_ffn
            latency_ffn.append(time.time() - ts_ffn_start)
        
        return x, new_cache_feature
        
    def correct(
        self,
        diff_ndarray: np.ndarray | None,
        cache_features: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        diff_tensor = torch.tensor(diff_ndarray, dtype=torch.float32).permute(2, 0, 1).to(self.device)
        diff = [
            {"image": diff_tensor, "height": diff_tensor.shape[-2], "width": diff_tensor.shape[-1]},
        ]
        
        # preprocess
        diffs = [self.base_model._move_to_current_device(x["image"]) for x in diff]
        diffs = [x / self.base_model.pixel_std for x in diffs]
        diffs = ImageList.from_tensors(
            diffs,
            self.base_model.backbone.size_divisibility,
            padding_constraints=self.base_model.backbone.padding_constraints,
        )

        # inference: backbone
        backbone = self.base_model.backbone
        net = backbone.net

        # PREPROCESS: correction
        dx = diff_convolution(net.patch_embed, diffs.tensor)
        dx = dx.permute(0, 2, 3, 1)  # B C H W -> B H W C
        
        # EncoderBlocks
        for bidx, block in enumerate(net.blocks):
            x_binput = cache_features[f"bidx{bidx}_input"]
            
            d_shortcut = dx

            dx_norm = block.norm1(x_binput+dx) - block.norm1(x_binput)
            
            dx = dx_norm

            # Window partition
            if block.window_size > 0:
                H, W = dx.shape[1], dx.shape[2]
                dx, pad_hw = window_partition(dx, block.window_size)

            # Attention
            # > qkv with shape (3, B, nHead, H * W, C)
            # > q, k, v with shape (B * nHead, H * W, C)
            dx_attn = dx

            B_attn, H_attn, W_attn, _ = dx_attn.shape

            q = cache_features[f"bidx{bidx}_attn_q"]
            k = cache_features[f"bidx{bidx}_attn_k"]
            v = cache_features[f"bidx{bidx}_attn_v"]
            qkT = cache_features[f"bidx{bidx}_attn_qkT"]

            dqkv = F.linear(dx_attn, block.attn.qkv.weight).reshape(B_attn, H_attn * W_attn, 3, block.attn.num_heads, -1).permute(2, 0, 3, 1, 4)
            dq, dk, dv = dqkv.reshape(3, B_attn * block.attn.num_heads, H_attn * W_attn, -1).unbind(0)

            d_qkT = ((q + dq) * block.attn.scale) @ (k + dk).transpose(-2, -1) - qkT
            if block.attn.use_rel_pos:
                d_qkT = block.attn.rel_pos_module(
                    d_qkT,
                    dq,
                    block.attn.rel_pos_h,
                    block.attn.rel_pos_w,
                    (H_attn, W_attn),
                    (H_attn, W_attn)
                )
            
            qkT_relpos = cache_features[f"bidx{bidx}_attn_qkT_relpos"]

            d_softmax = (qkT_relpos + d_qkT).softmax(-1) - qkT_relpos.softmax(-1)

            # > SPARSE D_SOFTMAX
            if block.window_size == 0:
                prune_rate = 0.6

                threshold_row = torch.quantile(qkT.sum(dim=-1), prune_rate, dim=-1)
                alive_mask_row = (qkT.sum(dim=-1) >= threshold_row[:, None]).float()

                threshold_col = torch.quantile(qkT.sum(dim=-2), prune_rate, dim=-1)
                alive_mask_col = (qkT.sum(dim=-2) >= threshold_col[:, None]).float()

                alive_mask = (alive_mask_row[:, :, None] * alive_mask_col[:, None, :])
                d_softmax = d_softmax * alive_mask
                
                print(f"[DEBUG] Block {bidx} Attention d_softmax sparsity: {alive_mask.sum().item() / alive_mask.numel():%}")

            attn_output = cache_features[f"bidx{bidx}_attn_output"]
            d_attn_output = (qkT_relpos.softmax(-1) + d_softmax) @ (v + dv) - attn_output

            # DEBUGGED: Correct till here

            d_attn_output = d_attn_output.view(B_attn, block.attn.num_heads, H_attn, W_attn, -1).permute(0, 2, 3, 1, 4).reshape(B_attn, H_attn, W_attn, -1)
            d_proj = F.linear(d_attn_output, block.attn.proj.weight)

            dx = d_proj
            
            # Reverse window partition
            if block.window_size > 0:
                dx = window_unpartition(dx, block.window_size, pad_hw, (H, W))

            # FFN Block
            dx = d_shortcut + block.drop_path(dx)

            x = cache_features[f"bidx{bidx}_ffn_input"]

            dx_ffn = dx
            ffn_output = cache_features[f"bidx{bidx}_ffn_output"]
            dx_ffn = dx_ffn + block.drop_path(block.mlp(block.norm2(x + dx)) - ffn_output)
            dx = dx_ffn

            if block.use_residual_block:
                print(f"[WARN] Residual block is not supported in contexted inference.")
                dx = block.residual(dx.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return dx, cache_features

    def postprocess(self, x: torch.Tensor, cache_features: Dict[str, torch.Tensor]):
        backbone = self.base_model.backbone
        net = backbone.net
        
        input = cache_features["input"]
        images = cache_features["images"]

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
        proposals, _ = self.base_model.proposal_generator(images, features, None)
        results, _ = self.base_model.roi_heads(images, features, proposals, None)
        detections = self.base_model._postprocess(results, input, images.image_sizes)

        predictions = detections[0]
        boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
        labels = predictions["instances"].pred_classes.cpu().numpy()
        scores = predictions["instances"].scores.cpu().numpy()
        segments = predictions["instances"].pred_masks.cpu().numpy()

        return boxes, segments, labels, scores