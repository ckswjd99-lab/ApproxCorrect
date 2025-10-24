import torch
import numpy as np
import cv2

from models.ViTDet.maskedrcnn_vit_b_fpn import MaskedRCNN_ViT_B_FPN_Contexted
from models.constants import COCO_LABELS_LIST, COCO_COLORS_ARRAY

INPUT_IMAGE_PATH = "coco_example.jpg"


@torch.no_grad()
def main():
    model = MaskedRCNN_ViT_B_FPN_Contexted()
    model.load_weight("weights/model_final_61ccd1.pkl")
    model.eval()
    model.to("cuda")

    # load image
    input_image = cv2.imread(INPUT_IMAGE_PATH)  # (H, W, 3)
    
    h, w, _ = input_image.shape
    # round and pad to make divisible by 8
    new_h = (h + 7) // 8 * 8
    new_w = (w + 7) // 8 * 8
    input_image = cv2.resize(input_image, (new_w, new_h))  # (new_h, new_w, 3)

    input_0 = input_image.astype(np.float32)
    input_1 = cv2.pyrDown(input_0)      # 1/2 resolution
    input_2 = cv2.pyrDown(input_1)      # 1/4 resolution
    input_3 = cv2.pyrDown(input_2)      # 1/8 resolution
    
    input_1 = cv2.pyrUp(input_1)                        # full resolution
    input_2 = cv2.pyrUp(cv2.pyrUp(input_2))             # full resolution
    input_3 = cv2.pyrUp(cv2.pyrUp(cv2.pyrUp(input_3)))  # full resolution

    diff_0 = (input_0 - input_1).astype(np.float32)     # full resolution
    diff_1 = (input_1 - input_2).astype(np.float32)     # full resolution
    diff_2 = (input_2 - input_3).astype(np.float32)     # full resolution

    # x3, cache_features = model.approx(input_3)
    x2, cache_features = model.approx(input_1)
    # dx1, cache_features = model.correct(diff_1, cache_features)
    dx0, cache_features = model.correct(diff_0, cache_features, prate_attn=0.9)
    x = x2 + dx0

    (boxes, segments, labels, scores) = model.postprocess(x, cache_features)
    
    cache_size = sum([v.numel() for vname, v in cache_features.items() if isinstance(v, torch.Tensor)]) * 4 / (1024 **2)
    print(f"Cache size: {cache_size:.2f} MB")
    for vname, v in cache_features.items():
        if isinstance(v, torch.Tensor):
            print(f"  {vname:30s}: {v.shape} {v.numel() * 4 / (1024 **2):.2f} MB")

    for _ in range(10):
        _ = model.approx(np.random.randn(1024, 1024, 3), debug_time=True)

    # make rectangles and labels on the input image
    for box, label, score in zip(boxes, labels, scores):
        if score < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box)
        color = (COCO_COLORS_ARRAY[label] * 255).astype(int).tolist()
        cv2.rectangle(input_0, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            input_0,
            f"{COCO_LABELS_LIST[label]}: {score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        # print(f"{COCO_LABELS_LIST[label]:14s}: {score:.2f}")

    # save
    cv2.imwrite("output.jpg", input_0)

if __name__ == "__main__":
    main()