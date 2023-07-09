
import argparse
from collections import defaultdict
import os
from pathlib import Path
import sys
import wandb

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_setup, launch
from fastreid.engine.train_loop import HookBase
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.events import get_event_storage


sys.path.append('.')


class WandbHook(HookBase):

    def __init__(self, eval_period, cfg):
        self._period = eval_period
        self._last_write = -1

        wandb.run or wandb.init(project='fastreid')

        wandb.config.update(cfg)

    def log_step(self):
        storage = get_event_storage()
        to_save = defaultdict(dict) # By default, any elements of to_save are empty dictionaries

        for k, (v, iter) in storage.latest().items():
            if iter <= self._last_write:
                continue
            to_save[iter][k] = v
        
        if len(to_save):
            all_iters = sorted(to_save.keys())
            self._last_write = max(all_iters)
        
        for itr, scalars_per_iter in to_save.items():
            scalars_per_iter["iteration"] = itr
            wandb.run.log(scalars_per_iter)

    def after_epoch(self):
        next_epoch = self.trainer.epoch + 1 # From HookBase
        if self._period > 0 and next_epoch % self._period == 0:
            self.log_step()

    def after_train(self):
        self.log_step()


def increment_path(path, exist_ok=False, sep='', mkdir=False):

    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.OUTPUT_DIR = increment_path(cfg.OUTPUT_DIR, mkdir=True)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = DefaultTrainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = DefaultTrainer.test(cfg, model)
        return res

    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    trainer.register_hooks(WandbHook(trainer.cfg.TEST.EVAL_PERIOD, trainer.cfg))
    return trainer.train()


def default_argument_parser():
    """
    Create a parser with some common arguments used by fastreid users.
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="fastreid Training")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    
    return parser

def improve_parser(parser):
    parser.add_argument("--OUTPUT_DIR", type=str, default='runs/apparence/train', help="sdfgh")

    parser.add_argument("--INPUT.SIZE_TRAIN", type=str, default='(64, 64)', help="sdfgh")
    parser.add_argument("--INPUT.SIZE_TEST", type=str, default='(64, 64)', help="sdfgh")
    parser.add_argument("--INPUT.AFFINE.ENABLED", action="store_true", help="sdfgh")
    parser.add_argument("--INPUT.AUGMIX.ENABLED", action="store_true", help="sdfgh")
    parser.add_argument("--INPUT.AUGMIX.PROB", type=float, default=0.0, help="sdfgh")
    parser.add_argument("--INPUT.AUTOAUG.ENABLED", action="store_true", help="sdfgh")
    parser.add_argument("--INPUT.AUTOAUG.PROB", type=float, default=0.0, help="sdfgh")
    parser.add_argument("--INPUT.CJ.ENABLED", action="store_true", help="sdfgh")
    parser.add_argument("--INPUT.CJ.PROB", type=float, default=0.0, help="sdfgh")
    parser.add_argument("--INPUT.CJ.BRIGHTNESS", type=float, default=0.0, help="sdfgh")
    parser.add_argument("--INPUT.CJ.CONTRAST", type=float, default=0.0, help="sdfgh")
    parser.add_argument("--INPUT.CJ.HUE", type=float, default=0.0, help="sdfgh")
    parser.add_argument("--INPUT.CJ.SATURATION", type=float, default=0.0, help="sdfgh")
    parser.add_argument("--INPUT.CROP.ENABLED", action="store_true", help="sdfgh")
    parser.add_argument("--INPUT.CROP.RATIO", type=str, default="(0., 1.)", help="sdfgh")
    parser.add_argument("--INPUT.CROP.SCALE", type=str, default="(0., 1.)", help="sdfgh")
    parser.add_argument("--INPUT.CROP.SIZE", type=str, default='(64, 64)', help="sdfgh")
    parser.add_argument("--INPUT.FLIP.ENABLED", action="store_true", help="sdfgh")
    parser.add_argument("--INPUT.FLIP.PROB", type=float, default=0.0, help="sdfgh")
    parser.add_argument("--INPUT.PADDING.ENABLED", action="store_true", help="sdfgh")
    parser.add_argument("--INPUT.PADDING.MODE", type=str, default="constant", help="sdfgh")
    parser.add_argument("--INPUT.PADDING.SIZE", type=int, default=0, help="sdfgh")
    parser.add_argument("--INPUT.REA.ENABLED", action="store_true", help="sdfgh")
    parser.add_argument("--INPUT.REA.PROB", type=float, default=0.0, help="sdfgh")
    parser.add_argument("--INPUT.REA.VALUE", type=str, default="(0.0, 0.0, 0.0)", help="sdfgh")
    parser.add_argument("--INPUT.RPT.ENABLED", action="store_true", help="sdfgh")
    parser.add_argument("--INPUT.RPT.PROB", type=float, default=0.0, help="sdfgh")

    parser.add_argument("--MODEL.LOSSES.CE.ALPHA", type=float, default=0.2, help="sdfgh")
    parser.add_argument("--MODEL.LOSSES.CE.EPSILON", type=float, default=0.1, help="sdfgh")
    parser.add_argument("--MODEL.LOSSES.CE.SCALE", type=float, default=1.0, help="sdfgh")
    parser.add_argument("--MODEL.LOSSES.TRI.HARD_MINING", action="store_true", help="sdfgh")
    parser.add_argument("--MODEL.LOSSES.TRI.NORM_FEAT", action="store_true", help="sdfgh")
    parser.add_argument("--MODEL.LOSSES.TRI.MARGIN", type=float, default=0.3, help="sdfgh")
    parser.add_argument("--MODEL.LOSSES.TRI.SCALE", type=float, default=1.0, help="sdfgh")
    parser.add_argument("--MODEL.PIXEL_MEAN", type=str, default="(123.675, 116.28, 103.53)", help="sdfgh")
    parser.add_argument("--MODEL.PIXEL_STD", type=str, default="(58.395, 57.12, 57.375)", help="sdfgh")
    parser.add_argument("--MODEL.QUEUE_SIZE", type=int, default=8192, help="sdfgh")

    parser.add_argument("--SOLVER.AMP.ENABLED", action="store_true", help="sdfgh")
    parser.add_argument("--SOLVER.BASE_LR", type=float, default=0.001, help="sdfgh")
    parser.add_argument("--SOLVER.BIAS_LR_FACTOR", type=float, default=1.0, help="sdfgh")
    parser.add_argument("--SOLVER.CLIP_GRADIENTS.ENABLED", action="store_true", help="sdfgh")
    parser.add_argument("--SOLVER.CLIP_GRADIENTS.CLIP_TYPE", type=str, default="norm", help="sdfgh")
    parser.add_argument("--SOLVER.CLIP_GRADIENTS.CLIP_VALUE", type=float, default=0.0, help="sdfgh")
    parser.add_argument("--SOLVER.CLIP_GRADIENTS.NORM_TYPE", type=float, default=2.0, help="sdfgh")
    parser.add_argument("--SOLVER.DELAY_EPOCHS", type=int, default=0, help="sdfgh")
    parser.add_argument("--SOLVER.FREEZE_ITERS", type=int, default=0, help="sdfgh")
    parser.add_argument("--SOLVER.GAMMA", type=float, default=0.1, help="sdfgh")
    parser.add_argument("--SOLVER.HEADS_LR_FACTOR", type=float, default=1.0, help="sdfgh")
    parser.add_argument("--SOLVER.IMS_PER_BATCH", type=int, default=64, help="sdfgh")
    parser.add_argument("--SOLVER.MOMENTUM", type=float, default=0.9, help="sdfgh")
    parser.add_argument("--SOLVER.NESTEROV", action="store_true", help="sdfgh")
    parser.add_argument("--SOLVER.OPT", type=str, default="Adam", help="sdfgh")
    parser.add_argument("--SOLVER.SCHED", type=str, default="MultiStepLR", help="sdfgh")
    parser.add_argument("--SOLVER.STEPS", type=str, default="(40, 90)", help="sdfgh")
    parser.add_argument("--SOLVER.WARMUP_FACTOR", type=float, default=0.1, help="sdfgh")
    parser.add_argument("--SOLVER.WARMUP_ITERS", type=int, default=2000, help="sdfgh")
    parser.add_argument("--SOLVER.WARMUP_METHOD", type=str, default="linear", help="sdfgh")
    parser.add_argument("--SOLVER.WEIGHT_DECAY", type=float, default=0.0005, help="sdfgh")
    parser.add_argument("--SOLVER.WEIGHT_DECAY_BIAS", type=float, default=0.0005, help="sdfgh")
    parser.add_argument("--SOLVER.WEIGHT_DECAY_NORM", type=float, default=0.0005, help="sdfgh")

    parser.add_argument("--TEST.FLIP.ENABLED", action="store_true", help="sdfgh")
    parser.add_argument("--TEST.ROC.ENABLED", action="store_true", help="sdfgh")
    parser.add_argument("--TEST.IMS_PER_BATCH", type=int, default=128, help="sdfgh")

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser


if __name__ == "__main__":
    args = improve_parser(default_argument_parser()).parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
