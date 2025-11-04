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

    def approx(
        self, 
        image_ndarray: np.ndarray,
        prate_attn: float,
        *args, **kwargs
    ):
        debug_time = kwargs.get("debug_time", False)

        latency_preproc = []
        latency_attn = []
        latency_ffn = []

        latency_norm = []
        latency_gen_qkv = []
        latency_self_attn = []
        latency_proj = []

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

            ts_norm_start = time.time()
            x_attn = block.norm1(x)
            latency_norm.append(time.time() - ts_norm_start)

            # Window partition
            ts_attn_start = time.time()
            if block.window_size > 0:
                H, W = x_attn.shape[1], x_attn.shape[2]
                x_attn, pad_hw = window_partition(x_attn, block.window_size)

            # Attention
            B_attn, H_attn, W_attn, _ = x_attn.shape

            ts_gen_qkv_start = time.time()
            qkv = block.attn.qkv(x_attn).reshape(B_attn, H_attn * W_attn, 3, block.attn.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.reshape(3, B_attn * block.attn.num_heads, H_attn * W_attn, -1).unbind(0)
            latency_gen_qkv.append(time.time() - ts_gen_qkv_start)

            ts_self_attn_start = time.time()

            attn_score = (q * block.attn.scale) @ k.transpose(-2, -1)

            if block.attn.use_rel_pos:
                attn_score = block.attn.rel_pos_module(attn_score, q, block.attn.rel_pos_h, block.attn.rel_pos_w, (H_attn, W_attn), (H_attn, W_attn))
            attn_score = attn_score.softmax(dim=-1)

            if block.window_size == 0:
                attn_score_pruned = attn_score.reshape(1, block.attn.num_heads, H_attn*W_attn, H_attn*W_attn)
    
                attn_sum_over_rows = attn_score_pruned.sum(dim=-2)

                threshold_col = torch.quantile(attn_sum_over_rows, prate_attn, dim=-1)

                alive_mask_col_flat = (attn_sum_over_rows >= threshold_col[:, :, None]).view(-1)
                alive_indices_col = torch.nonzero(alive_mask_col_flat, as_tuple=False).squeeze(1)

                attn_score_pruned_permuted_flat = attn_score_pruned.permute(0, 1, 3, 2).reshape(1, block.attn.num_heads * H_attn * W_attn, H_attn * W_attn)

                attn_score_pruned_indexed = attn_score_pruned_permuted_flat[:, alive_indices_col]

                register_cache(f"bidx{bidx}_attn_score_pruned", new_cache_feature, attn_score_pruned_indexed)
                register_cache(f"bidx{bidx}_attn_score_alive_idx", new_cache_feature, alive_indices_col)

            x_attn = (attn_score @ v)
            latency_self_attn.append(time.time() - ts_self_attn_start)

            register_cache(f"bidx{bidx}_attn_output", new_cache_feature, x_attn)

            ts_proj_start = time.time()
            x_attn = x_attn.view(B_attn, block.attn.num_heads, H_attn, W_attn, -1).permute(0, 2, 3, 1, 4).reshape(B_attn, H_attn, W_attn, -1)
            x_attn = block.attn.proj(x_attn)
            latency_proj.append(time.time() - ts_proj_start)
            
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
        
        if debug_time:
            print(f"[DEBUG] Approximation Time Stats (ms):")
            print(f"  Preprocess: {np.sum(latency_preproc)*1000:.2f}")
            print(f"  Attention : {np.sum(latency_attn)*1000:.2f}")
            print(f"   - Norm     : {np.sum(latency_norm)*1000:.2f}")
            print(f"   - Gen QKV : {np.sum(latency_gen_qkv)*1000:.2f}")
            print(f"   - Self Attn: {np.sum(latency_self_attn)*1000:.2f}")
            print(f"   - Proj     : {np.sum(latency_proj)*1000:.2f}")
            print(f"  FFN       : {np.sum(latency_ffn)*1000:.2f}")
        
        return x, new_cache_feature
        
    def correct(
        self,
        diff_ndarray: np.ndarray | None,
        cache_features: Dict[str, torch.Tensor],
        prate_attn: float = 0.0,
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
            register_cache(f"bidx{bidx}_input", cache_features, x_binput + dx)
            
            d_shortcut = dx

            x_norm_new = block.norm1(x_binput+dx)
            dx_norm = x_norm_new - block.norm1(x_binput)
            
            dx = dx_norm

            # Window partition
            if block.window_size > 0:
                H, W = dx.shape[1], dx.shape[2]
                dx, pad_hw = window_partition(dx, block.window_size)
                x_norm_new, _ = window_partition(x_norm_new, block.window_size)

            # Attention
            # > qkv with shape (3, B, nHead, H * W, C)
            # > q, k, v with shape (B * nHead, H * W, C)
            dx_attn = dx

            B_attn, H_attn, W_attn, _ = dx_attn.shape

            # > SPARSE D_SOFTMAX
            attn_output = cache_features[f"bidx{bidx}_attn_output"]
            
            if block.window_size == 0:
                # print(dx_attn.shape, block.attn.qkv.weight.shape)   # torch.Size([1, 64, 64, 768]) torch.Size([2304, 768])
                dim_v = block.attn.qkv.weight.shape[0] // 3

                dv = F.linear(
                    dx_attn, block.attn.qkv.weight.reshape(3, dim_v, -1)[-1]
                ).reshape(
                    B_attn, H_attn * W_attn, 1, block.attn.num_heads, -1
                ).permute(2, 0, 3, 1, 4)
                dv = dv.reshape(B_attn * block.attn.num_heads, H_attn * W_attn, -1)

                attn_score_pruned_indexed = cache_features[f"bidx{bidx}_attn_score_pruned"]
                alive_indices_col = cache_features[f"bidx{bidx}_attn_score_alive_idx"]

                num_alive_cols = alive_indices_col.shape[0]
                attn_score_pruned_indexed = attn_score_pruned_indexed.reshape(
                    1, 
                    block.attn.num_heads, 
                    num_alive_cols // block.attn.num_heads, 
                    H_attn * W_attn
                ).permute(0, 1, 3, 2)

                dv_indexed = dv.reshape(-1, dv.shape[-1])[alive_indices_col].reshape(block.attn.num_heads, -1, dv.shape[-1])
                
                d_attn_output = attn_score_pruned_indexed @ dv_indexed
            else:
                qkv = block.attn.qkv(x_norm_new).reshape(B_attn, H_attn * W_attn, 3, block.attn.num_heads, -1).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.reshape(3, B_attn * block.attn.num_heads, H_attn * W_attn, -1).unbind(0)

                qkT_updated = (q @ k.transpose(-2, -1)) * block.attn.scale
                if block.attn.use_rel_pos:
                    qkT_updated = block.attn.rel_pos_module(
                        qkT_updated,
                        q,
                        block.attn.rel_pos_h,
                        block.attn.rel_pos_w,
                        (H_attn, W_attn),
                        (H_attn, W_attn)
                    )
                
                d_attn_output = qkT_updated.softmax(-1) @ v - attn_output
            register_cache(f"bidx{bidx}_attn_output", cache_features, attn_output + d_attn_output)

            d_attn_output = d_attn_output.view(B_attn, block.attn.num_heads, H_attn, W_attn, -1).permute(0, 2, 3, 1, 4).reshape(B_attn, H_attn, W_attn, -1)
            d_proj = F.linear(d_attn_output, block.attn.proj.weight)

            dx = d_proj
            
            # Reverse window partition
            if block.window_size > 0:
                dx = window_unpartition(dx, block.window_size, pad_hw, (H, W))

            # FFN Block
            dx = d_shortcut + block.drop_path(dx)

            x = cache_features[f"bidx{bidx}_ffn_input"]
            # Keep FFN input cache in sync for subsequent corrections
            register_cache(f"bidx{bidx}_ffn_input", cache_features, x + dx)

            ffn_output = cache_features[f"bidx{bidx}_ffn_output"]
            delta_mlp = block.mlp(block.norm2(x + dx)) - ffn_output
            # Update FFN output cache (pre-residual MLP output)
            register_cache(f"bidx{bidx}_ffn_output", cache_features, ffn_output + delta_mlp)

            dx = dx + block.drop_path(delta_mlp)

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