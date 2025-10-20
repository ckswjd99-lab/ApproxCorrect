import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
import time

from typing import List, Optional, Dict, Tuple, Union

from .modeling.backbone.utils import get_abs_pos, window_partition, window_unpartition, add_decomposed_rel_pos, partial_mlp_inference, AddDecomposedRelPos

from .eventful_transformer.base import ExtendedModule
from .eventful_transformer.counting import CountedAdd, CountedMatmul


from .structures import ImageList
from .layers import ShapeSpec
from .layers.wrappers import move_device_like, shapes_to_tensor
from .differ_utils import diff_convolution, diff_layernorm, diff_mlp_gelu

from ..proc_image import (
    calculate_multi_iou, calculate_iou, visualize_detection, refine_images,
    graph_iou, graph_recompute
)

class MaskedRCNN_ViT_FPN_Contexted(ExtendedModule):
    def __init__(self, device="cuda:0", dataset_name="coco"):
        super().__init__()
        self.idx = 0

        self.device = device

        self.embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1

        # backbone
        self.backbone = nn.Identity().to(self.device)

        # model
        self.base_model = nn.Identity().to(self.device)

        # counting module
        self.add = CountedAdd()
        self.matmul = CountedMatmul()
        self.add_decomposed_rel_pos = AddDecomposedRelPos()

    def forward(self, image_ndarray: np.ndarray, *args, **kwargs):
        latency_preproc = []
        latency_attn = []
        latency_ffn = []
        latency_postproc = []

        # image_ndarray: (H, W, C)
        image_tensor = torch.tensor(image_ndarray, dtype=torch.uint8, device=self.device).permute(2, 0, 1)
        input = [{"image": image_tensor, "height": image_tensor.shape[-2], "width": image_tensor.shape[-1]}]
        
        images = [self.base_model._move_to_current_device(x["image"]) for x in input]
        images = [(x - self.base_model.pixel_mean) / self.base_model.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.base_model.backbone.size_divisibility,
            padding_constraints={"size_divisibility": self.base_model.backbone.size_divisibility, "padding_constraints": image_ndarray.shape[0]},
        )

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

        print(x.shape)

        # EncoderBlocks
        for block in net.blocks:
            shortcut = x
            x_attn = block.norm1(x)

            # Window partition
            ts_attn_start = time.time()
            if block.window_size > 0:
                H, W = x_attn.shape[1], x_attn.shape[2]
                x_attn, pad_hw = window_partition(x_attn, block.window_size)

            # Attention
            x_attn = block.attn(x_attn)
            
            # Reverse window partition
            if block.window_size > 0:
                x_attn = window_unpartition(x_attn, block.window_size, pad_hw, (H, W))

            latency_attn.append(time.time() - ts_attn_start)

            # FFN
            ts_ffn_start = time.time()

            x_ffn = x_attn
            x_ffn = shortcut + block.drop_path(x_ffn)
            x_ffn = x_ffn + block.drop_path(block.mlp(block.norm2(x_ffn)))

            if block.use_residual_block:
                x_ffn = block.residual(x_ffn.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

            x = x_ffn
            latency_ffn.append(time.time() - ts_ffn_start)
        
        ts_postproc_start = time.time()
        # FPN
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

        latency_postproc.append(time.time() - ts_postproc_start)

        print(f"[INFO] Latency Preprocess: {np.sum(latency_preproc)*1000:.2f} ms")
        print(f"[INFO] Latency Attention: {np.sum(latency_attn)*1000:.2f} ms")
        print(f"[INFO] Latency FFN: {np.sum(latency_ffn)*1000:.2f} ms")
        print(f"[INFO] Latency Postprocess: {np.sum(latency_postproc)*1000:.2f} ms")

        return boxes, labels, scores

    def forward_correcting(
            self, 
            image_ndarray: np.ndarray,
            diff_ndarray: np.ndarray | None = None,
            cache_features: Dict[str, torch.Tensor] = {},
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, torch.Tensor]]:
        # image_ndarray: (H, W, C)
        
        if diff_ndarray is None:
            diff_ndarray = np.zeros_like(image_ndarray)
        new_cache_feature = {}
        
        # convert to tensor
        image_tensor = torch.tensor(image_ndarray, dtype=torch.uint8).permute(2, 0, 1).to(self.device)
        diff_tensor = torch.tensor(diff_ndarray, dtype=torch.float32).permute(2, 0, 1).to(self.device)
        input = [
            {"image": image_tensor, "height": image_tensor.shape[-2], "width": image_tensor.shape[-1]},
        ]
        diff = [
            {"image": diff_tensor, "height": diff_tensor.shape[-2], "width": diff_tensor.shape[-1]},
        ]
        
        # preprocess
        images = [self.base_model._move_to_current_device(x["image"]) for x in input]
        images = [(x - self.base_model.pixel_mean) / self.base_model.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.base_model.backbone.size_divisibility,
            padding_constraints={"size_divisibility": self.base_model.backbone.size_divisibility, "padding_constraints": image_ndarray.shape[0]},
        )

        diffs = [self.base_model._move_to_current_device(x["image"]) for x in diff]
        diffs = [(x) / self.base_model.pixel_std for x in diffs]
        diffs = ImageList.from_tensors(
            diffs,
            self.base_model.backbone.size_divisibility,
            padding_constraints={"size_divisibility": self.base_model.backbone.size_divisibility, "padding_constraints": image_ndarray.shape[0]},
        )

        # inference: backbone
        backbone = self.base_model.backbone
        net = backbone.net

        # PREPROCESS: inference
        x = net.patch_embed(images.tensor)

        # PREPROCESS: correction
        dx = diff_convolution(net.patch_embed, diffs.tensor)
        dx = dx.permute(0, 2, 3, 1)  # B C H W -> B H W C

        # DEBUG: correctness check
        x2 = net.patch_embed(diffs.tensor + images.tensor)
        error = torch.abs((x + dx) - x2).mean()
        print(f"[DEBUG] PatchEmbed error: {error.item()}")

        ape = get_abs_pos(
            net.pos_embed, net.pretrain_use_cls_token, (x.shape[1], x.shape[2])
        )
        if net.pos_embed is not None:
            x = x + ape
        
        x_init = x
        dx_init = dx
        
        # > x: Tensor(1, 64, 64, 768)
        # > dx: Tensor(1, 64, 64, 768)

        # EncoderBlocks
        for bidx, block in enumerate(net.blocks):
            # DEBUG
            print(f"[DEBUG] Processing Block {bidx}: dx {dx.norm():.4f} / x {x.norm():.4f} ({dx.norm()/x.norm():%})")

            shortcut = x
            d_shortcut = dx

            x_norm = block.norm1(x)
            # dx_norm = diff_layernorm(block.norm1, x, dx)
            dx_norm = block.norm1(x+dx) - x_norm

            # DEBUG: correctness check
            x2 = block.norm1(x + dx)
            error = ((x_norm + dx_norm) - x2).norm() / x2.norm()
            print(f"[DEBUG] Block {bidx} LayerNorm error: {error.item():%}")
            
            x = x_norm
            dx = dx_norm

            # Window partition
            if block.window_size > 0:
                H, W = x.shape[1], x.shape[2]
                x, pad_hw = window_partition(x, block.window_size)
                dx, _ = window_partition(dx, block.window_size)

            # Attention
            # > qkv with shape (3, B, nHead, H * W, C)
            # > q, k, v with shape (B * nHead, H * W, C)
            x_attn = x
            dx_attn = dx

            B_attn, H_attn, W_attn, _ = x_attn.shape

            qkv = block.attn.qkv(x_attn).reshape(B_attn, H_attn * W_attn, 3, block.attn.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.reshape(3, B_attn * block.attn.num_heads, H_attn * W_attn, -1).unbind(0)

            dqkv = F.linear(dx_attn, block.attn.qkv.weight).reshape(B_attn, H_attn * W_attn, 3, block.attn.num_heads, -1).permute(2, 0, 3, 1, 4)
            dq, dk, dv = dqkv.reshape(3, B_attn * block.attn.num_heads, H_attn * W_attn, -1).unbind(0)

            # DEBUG: correctness check
            x2 = block.attn.qkv(x + dx).reshape(B_attn, H_attn * W_attn, 3, block.attn.num_heads, -1).permute(2, 0, 3, 1, 4)
            error = ((qkv + dqkv) - x2).norm() / x2.norm()
            print(f"[DEBUG] Block {bidx} QKV error: {error.item():%}")

            attn_score_raw = (q * block.attn.scale) @ k.transpose(-2, -1)

            q_dk = (q * block.attn.scale) @ dk.transpose(-2, -1)
            dq_k = (dq * block.attn.scale) @ k.transpose(-2, -1)
            # d_attn_score = q_dk + dq_k
            d_attn_score = q_dk + dq_k + (dq * block.attn.scale) @ dk.transpose(-2, -1)    # Critical for low error!

            if block.attn.use_rel_pos:
                attn_score_raw = block.attn.rel_pos_module(attn_score_raw, q, block.attn.rel_pos_h, block.attn.rel_pos_w, (H_attn, W_attn), (H_attn, W_attn))
                d_attn_score += block.attn.rel_pos_module(torch.zeros_like(d_attn_score), dq, block.attn.rel_pos_h, block.attn.rel_pos_w, (H_attn, W_attn), (H_attn, W_attn))

            attn_score = attn_score_raw.softmax(dim=-1)

            sum_term = (d_attn_score * attn_score).sum(dim=-1, keepdim=True)
            # d_softmax = attn_score * (d_attn_score - sum_term)
            d_softmax = (attn_score_raw + d_attn_score).softmax(dim=-1) - attn_score    # Critical for low error!

            # SPARSE D_SOFTMAX
            if block.window_size == 0:
                prune_rate = 0.998
                # threshold = torch.quantile(attn_score.abs(), prune_rate, dim=-1)
                # d_softmax = d_softmax * (attn_score.abs() >= threshold[:, :, None])

                threshold_row = torch.quantile(attn_score_raw.sum(dim=-1), prune_rate, dim=-1)
                alive_mask_row = (attn_score_raw.sum(dim=-1) >= threshold_row[:, None]).float()

                threshold_col = torch.quantile(attn_score_raw.sum(dim=-2), prune_rate, dim=-1)
                alive_mask_col = (attn_score_raw.sum(dim=-2) >= threshold_col[:, None]).float()

                alive_mask = (alive_mask_row[:, :, None] + alive_mask_col[:, None, :]).clamp(max=1.0)
                d_softmax = d_softmax * alive_mask
                
                print(f"[DEBUG] Block {bidx} Attention d_softmax sparsity: {alive_mask.sum().item() / alive_mask.numel():%}")

            x_attn = (attn_score @ v).view(B_attn, block.attn.num_heads, H_attn, W_attn, -1).permute(0, 2, 3, 1, 4).reshape(B_attn, H_attn, W_attn, -1)
            # dx_attn = (attn_score @ dv) + (d_softmax @ v)
            dx_attn = (attn_score + d_softmax) @ (v + dv) - (attn_score @ v)    # Critical for low error!
            dx_attn = dx_attn.view(B_attn, block.attn.num_heads, H_attn, W_attn, -1).permute(0, 2, 3, 1, 4).reshape(B_attn, H_attn, W_attn, -1)

            x_attn = block.attn.proj(x_attn)
            dx_attn = F.linear(dx_attn, block.attn.proj.weight)

            # DEBUG: correctness check
            x2 = block.attn(x + dx)
            error = ((x_attn + dx_attn) - x2).norm() / x2.norm()
            print(f"[DEBUG] Block {bidx} Attention error: {error.item():%}")

            x = x_attn
            dx = dx_attn
            
            # Reverse window partition
            if block.window_size > 0:
                x = window_unpartition(x, block.window_size, pad_hw, (H, W))
                dx = window_unpartition(dx, block.window_size, pad_hw, (H, W))

            x = shortcut + block.drop_path(x)
            dx = d_shortcut + block.drop_path(dx)

            x_ffn = x; dx_ffn = dx

            x_ffn = x_ffn + block.drop_path(block.mlp(block.norm2(x_ffn)))
            dx_ffn = dx_ffn + block.drop_path(diff_mlp_gelu(block.mlp, block.norm2(x), block.norm2(x+dx) - block.norm2(x)))

            # DEBUG: correctness check
            x2 = (x + dx) + block.drop_path(block.mlp(block.norm2(x + dx)))
            error = ((x_ffn + dx_ffn) - x2).norm() / x2.norm()
            print(f"[DEBUG] Block {bidx} FFN error: {error.item():%}")


            x = x_ffn; dx = dx_ffn

            if block.use_residual_block:
                print(f"[WARN] Residual block is not supported in contexted inference.")
                x = block.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                dx = block.residual(dx.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

            # DEBUG: correctness check
            x2 = block(shortcut + d_shortcut)
            error = ((x + dx) - x2).norm() / x2.norm()
            print(f"[DEBUG] Block {bidx} Full error: {error.item():%}")

        x += dx  # final output

        # DEBUG: final correctness check
        x2 = x_init + dx_init
        for block in net.blocks:
            x2 = block(x2)
        error = ((x) - x2).norm() / x2.norm()
        print(f"[DEBUG] Backbone final error: {error.item():%}")

        # > FPN
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

        # inference: RPN

        # > proposal_generator
        proposals, _ = self.base_model.proposal_generator(images, features, None)
        results, _ = self.base_model.roi_heads(images, features, proposals, None)

        # postprocess
        detections = self.base_model._postprocess(results, input, images.image_sizes)

        predictions = detections[0]
        boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
        labels = predictions["instances"].pred_classes.cpu().numpy()
        scores = predictions["instances"].scores.cpu().numpy()
        pred_masks = predictions["instances"].pred_masks.cpu().numpy()

        return (boxes, labels, scores), new_cache_feature, pred_masks
    
    