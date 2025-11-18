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
    PRATE_ATTN = 0.5
    DMASK_THRES = 0.3
    
    VERBOSE = False
    DEBUG_TIME = True

    func_inference.set_approx_level(APPROX_LEVEL)
    func_inference.set_prate_attn(PRATE_ATTN)
    func_inference.set_dmask_thres(DMASK_THRES)
    
    func_inference.set_verbose(VERBOSE)
    func_inference.set_debug_time(DEBUG_TIME)

    func_inference.reset_timecount()

    ret = inference_on_dataset(
        func_inference,
        instantiate(cfg.dataloader.test),
        instantiate(cfg.dataloader.evaluator),
    )
    print_csv_format(ret)

    eta_approx, eta_correct, eta_etc = func_inference.avg_timecount()
    print("Average Approx Time (ms):", eta_approx)
    print("Average Correct Time (ms):", eta_correct)
    print("Average Etc Time (ms):", eta_etc)

    avg_dindice_alive = func_inference.avg_dindice_alive()
    print("Average Dinidce Alive Rate:", avg_dindice_alive)

    last_cache_size = func_inference.last_cache_size
    print("Last Cache Size:", last_cache_size)

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