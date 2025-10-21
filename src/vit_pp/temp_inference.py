import torch
import numpy as np
import cv2

from models.ViTDet.maskedrcnn_vit_b_fpn import MaskedRCNN_ViT_B_FPN_Contexted
from models.constants import COCO_LABELS_LIST, COCO_COLORS_ARRAY

INPUT_IMAGE_PATH = "coco_example_2.jpg"


@torch.no_grad()
def main():
    model = MaskedRCNN_ViT_B_FPN_Contexted()
    model.load_weight("weights/model_final_61ccd1.pkl")
    model.eval()
    model.to("cuda")

    # load cat.jpg image and resize to (512, 512)
    input_image = cv2.imread(INPUT_IMAGE_PATH)  # (H, W, 3)
    
    h, w, _ = input_image.shape
    # round and pad to make divisible by 8
    new_h = (h + 7) // 8 * 8
    new_w = (w + 7) // 8 * 8
    input_image = cv2.resize(input_image, (new_w, new_h))  # (new_h, new_w, 3)

    input_0 = input_image.astype(np.float32)
    input_1 = cv2.pyrDown(input_0)  # (256, 256, 3)
    input_2 = cv2.pyrDown(input_1)  # (128, 128, 3)
    input_3 = cv2.pyrDown(input_2)  # (64, 64, 3)
    
    input_1 = cv2.pyrUp(input_1)  # (512, 512, 3)
    input_2 = cv2.pyrUp(cv2.pyrUp(input_2))  # (512, 512, 3)
    input_3 = cv2.pyrUp(cv2.pyrUp(cv2.pyrUp(input_3)))  # (512, 512, 3)

    diff_0 = (input_0 - input_1).astype(np.float32)  # (512, 512, 3)
    diff_1 = (input_1 - input_2).astype(np.float32)  # (512, 512, 3)
    diff_2 = (input_2 - input_3).astype(np.float32)  # (512, 512, 3)

    x, cache_features = model.approx(input_2)

    dx, cache_features = model.correct(diff_1+diff_0, cache_features)
    x = x + dx

    (boxes, segments, labels, scores) = model.postprocess(x, cache_features)

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
        print(f"{COCO_LABELS_LIST[label]:14s}: {score:.2f}")

    # save 
    cv2.imwrite("output.jpg", input_0)

if __name__ == "__main__":
    main()