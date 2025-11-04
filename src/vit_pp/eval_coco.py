import torch
import numpy as np
import cv2
import argparse
import time

from models.ViTDet.maskedrcnn_vit_b_fpn import MaskedRCNN_ViT_B_FPN_Contexted
from models.constants import COCO_LABELS_LIST, COCO_COLORS_ARRAY

import fiftyone as fo
import fiftyone.zoo as foz

VERBOSE = True

def approx_and_correct(
    model: MaskedRCNN_ViT_B_FPN_Contexted, 
    input_image: np.ndarray,
    level: int = 1,
    prate_attn: float = 0.0,
    approx_only: bool = False,
):
    # Snap h, w to be divisible by 8
    h, w, _ = input_image.shape
    new_h = (h + 7) // 8 * 8
    new_w = (w + 7) // 8 * 8
    input_image = cv2.resize(input_image, (new_w, new_h))  # (new_h, new_w, 3)

    image_pyramid = []
    for l in range(level+1):
        downsampled = input_image.astype(np.float32)
        for _ in range(l):
            downsampled = cv2.pyrDown(downsampled)
        for _ in range(l):
            downsampled = cv2.pyrUp(downsampled)
        image_pyramid.append(downsampled)
    
    diff_pyramid = [None]
    for l in range(1, level + 1):
        diff = (image_pyramid[l - 1] - image_pyramid[l]).astype(np.float32)
        diff_pyramid.append(diff)
    
    ts_approx_start = time.time()
    x, cache_features = model.approx(image_pyramid[level], prate_attn=prate_attn)
    ts_approx_end = time.time()
    
    ts_correct_start = time.time()
    if not approx_only:
        for l in range(level, 0, -1):
            dx, cache_features = model.correct(diff_pyramid[l], cache_features, prate_attn=prate_attn)
            x = x + dx
    ts_correct_end = time.time()

    (boxes, segments, labels, scores) = model.postprocess(x, cache_features)

    detections = []
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        rel_box = [x1 / new_w, y1 / new_h, (x2 - x1) / new_w, (y2 - y1) / new_h]

        detections.append(fo.Detection(
            label=COCO_LABELS_LIST[label],
            bounding_box=rel_box,
            confidence=float(score)
        ))

    if False:
        cache_size = sum(cache.numel() for cache in cache_features.values() if isinstance(cache, torch.Tensor)) * 4
        print(f"Cache size: {cache_size/1e6:.4f} MB")

        for cname, cvalue in cache_features.items():
            if isinstance(cvalue, torch.Tensor):
                print(f"{cname}: {cvalue.numel() * 4 / 1e6 :.4f} MB")

        print(f"Latency")
        print(f" > Approximate: {ts_approx_end - ts_approx_start}")
        print(f" > Correct: {ts_correct_end - ts_correct_start}")
    
    return detections
    

@torch.no_grad()
def main():
    # PREPARE DATASET
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        max_samples=50,
        shuffle=False,
        progress=VERBOSE,
    )
    print(dataset)

    # PREPARE MODEL
    model = MaskedRCNN_ViT_B_FPN_Contexted()
    model.load_weight("weights/model_final_61ccd1.pkl")
    model.eval()
    model.to("cuda")


    # EVALUATE: Baseline
    level=0
    pred_field = f"predictions_level{level}_baseline"
    with fo.ProgressBar(quiet=not VERBOSE) as pb:
        for sample in pb(dataset):
            # Load image
            image_path = sample.filepath
            image = cv2.imread(image_path)

            # Inference
            detections = approx_and_correct(model, image, level=level, approx_only=True)

            # 변환된 Detections 리스트를 샘플의 새 필드에 저장
            sample[pred_field] = fo.Detections(detections=detections)
            sample.save() # 변경 사항 저장
    
        results = dataset.evaluate_detections(
            pred_field,
            gt_field="ground_truth",
            eval_key=f"eval_map_level{level}_baseline",
            compute_mAP=True,
            progress=VERBOSE,
        )

        map_score = results.mAP()
        print(f"Level: {level} Baseline => mAP: {map_score}")

    # EVALUATE: ApproxCorrect
    for level in [1, 2]:
        for prune_rate in [0.0, 0.5, 0.7, 0.9, 0.95, 1.0]:
            pred_field = f"predictions_level{level}_prate{int(prune_rate*100)}"
            with fo.ProgressBar(quiet=not VERBOSE) as pb:
                for sample in pb(dataset):
                    # Load image
                    image_path = sample.filepath
                    image = cv2.imread(image_path)

                    # Inference
                    detections = approx_and_correct(model, image, level=level, prate_attn=prune_rate)

                    # 변환된 Detections 리스트를 샘플의 새 필드에 저장
                    sample[pred_field] = fo.Detections(detections=detections)
                    sample.save() # 변경 사항 저장
    
                results = dataset.evaluate_detections(
                    pred_field,
                    gt_field="ground_truth",
                    eval_key=f"eval_map_level{level}_prate{int(prune_rate*100)}",
                    compute_mAP=True,
                    progress=VERBOSE,
                )

                map_score = results.mAP()
                print(f"Level: {level}, Prune Rate: {prune_rate} => mAP: {map_score}")

        pred_field = f"predictions_level{level}_approx_only"
        with fo.ProgressBar(quiet=not VERBOSE) as pb:
            for sample in pb(dataset):
                image_path = sample.filepath
                image = cv2.imread(image_path)

                detections = approx_and_correct(model, image, level=level, approx_only=True)

                sample[pred_field] = fo.Detections(detections=detections)
                sample.save()
            
            results = dataset.evaluate_detections(
                pred_field,
                gt_field="ground_truth",
                eval_key=f"eval_map_level{level}_approx_only",
                compute_mAP=True,
                progress=VERBOSE,
            )
            map_score = results.mAP()
            print(f"Level: {level}, Approx Only => mAP: {map_score}")

    # print(results.report())

    # session = fo.launch_app(
    #     dataset,
    #     auto=False
    # )
    # print("Press Ctrl+C to stop the server.")
    # session.wait()

    # COMPUTE METRICS


if __name__ == "__main__":
    # arguments: 


    main()