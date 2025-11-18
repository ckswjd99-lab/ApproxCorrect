import torch
import torch.nn.functional as F

from typing import Tuple, Dict


def create_dmask(dinput: torch.Tensor, threshold=0.3) -> Tuple[torch.Tensor, torch.Tensor]:
    dinput = dinput.abs().mean(dim=1, keepdim=True)  # B, 1, H, W
    # dinput = F.max_pool2d(dinput, kernel_size=16, stride=16, padding=0)

    # calculate energy in each patch (L2 norm)
    dinput_sq = dinput.pow(2)
    dinput_avg_sq = F.avg_pool2d(dinput_sq, kernel_size=16, stride=16, padding=0)
    dinput = dinput_avg_sq * (16 * 16)

    pixel_std_avg = 57
    pixel_thres = 10
    
    # threshold = pixel_thres / pixel_std_avg  # normalized threshold

    dmask = (dinput > threshold).float()  # B, 1, H/16, W/16
    dindice = dmask.view(-1).nonzero(as_tuple=False)

    return dmask, dindice