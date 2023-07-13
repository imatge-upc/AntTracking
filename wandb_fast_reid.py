
# TODO: recover previous comits where a Hook was used and add https://docs.wandb.ai/guides/track/log/distributed-training into it

import argparse
import os
from pathlib import Path
import sys

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer


sys.path.append('.')

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

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    
    return parser


FLOAT_OPTS_LIST = [
    'PROB',
    'BRIGHTNESS',
    'CONTRAST',
    'HUE',
    'SATURATION',
    'ALPHA',
    'EPSILON',
    'MODEL.LOSSES.CE.SCALE',
    'MODEL.LOSSES.TRI.SCALE',
    'MARGIN',
    'BIAS_LR_FACTOR',
    'CLIP_VALUE',
    'NORM_TYPE',
    'GAMMA',
    'HEADS_LR_FACTOR',
    'MOMENTUM',
    'WARMUP_FACTOR',
    'WEIGHT_DECAY',
    'WEIGHT_DECAY_BIAS',
    'WEIGHT_DECAY_NORM'
]


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    
    aux = [x.split("=") for x in args.opts]
    args.opts = []
    for x in aux:
        if x[0] == "OUTPUT_DIR":
            x[1] = str(increment_path(x[1], mkdir=True))
        
        if any([k in x[0] for k in FLOAT_OPTS_LIST]):
            x[1] = str(float(x[1]))

        args.opts += x

    print("\n\nCommand Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
