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

    # load cat.jpg image and resize to (512, 512)
    input_image = cv2.imread(INPUT_IMAGE_PATH)  # (H, W, 3)
    
    h, w, _ = input_image.shape
    # new_w = int(w * (512 / h))
    # input_image = cv2.resize(input_image, (new_w, 512))
    # input_image = input_image[:512, new_w-512:, :]  # (512, 512, 3)

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

    diff_rate_0 = np.linalg.norm(diff_0) / np.linalg.norm(input_0)
    diff_rate_1 = np.linalg.norm(diff_1) / np.linalg.norm(input_1)
    diff_rate_2 = np.linalg.norm(diff_2) / np.linalg.norm(input_2)
    # print(f"Diff rate 0: {diff_rate_0:.6%}")
    # print(f"Diff rate 1: {diff_rate_1:.6%}")
    # print(f"Diff rate 2: {diff_rate_2:.6%}")

    # save input3, diff2, diff1, diff0 as images
    # cv2.imwrite("input_3.jpg", input_3)
    # cv2.imwrite("diff_2.jpg", ((diff_2 - diff_2.min()) / (diff_2.max() - diff_2.min()) * 255).astype(np.uint8))
    # cv2.imwrite("diff_1.jpg", ((diff_1 - diff_1.min()) / (diff_1.max() - diff_1.min()) * 255).astype(np.uint8))
    # cv2.imwrite("diff_0.jpg", ((diff_0 - diff_0.min()) / (diff_0.max() - diff_0.min()) * 255).astype(np.uint8))


    # (boxes, labels, scores), _, _ = model.forward_correcting(input_0)
    # (boxes, labels, scores), _, _ = model.forward_correcting(input_1, diff_0)
    (boxes, labels, scores), _, _ = model.forward_correcting(input_2, diff_1+diff_0)

    # make rectangles and labels on the input image
    for box, label, score in zip(boxes, labels, scores):
        if score < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box)
        color = (COCO_COLORS_ARRAY[label] * 255).astype(int).tolist()
        cv2.rectangle(input_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            input_image,
            f"{COCO_LABELS_LIST[label]}: {score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        print(f"Detected {COCO_LABELS_LIST[label]} with confidence {score:.2f}")

    # save 
    cv2.imwrite("output.jpg", input_image)

if __name__ == "__main__":
    main()