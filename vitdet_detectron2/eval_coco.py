from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import default_argument_parser, default_setup

from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format

import torch

from appcorr import MaskedRCNN_ViT_FPN_AppCorr


def do_test(cfg, model):
    func_inference = MaskedRCNN_ViT_FPN_AppCorr(model)
    func_inference.to(model.device)
    func_inference.eval()

    # AppCorr settings
    APPROX_LEVEL = 1
    PRATE_ATTN = 0.0
    DEBUG_TIME = False

    func_inference.set_approx_level(APPROX_LEVEL)
    func_inference.set_prate_attn(PRATE_ATTN)
    func_inference.set_debug_time(DEBUG_TIME)

    ret = inference_on_dataset(
        func_inference,
        # model,
        instantiate(cfg.dataloader.test),
        instantiate(cfg.dataloader.evaluator),
    )
    print_csv_format(ret)
    return ret

@torch.no_grad()
def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model.eval()
    model = create_ddp_model(model)
    DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)