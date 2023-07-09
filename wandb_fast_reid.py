
from collections import defaultdict
import os
from pathlib import Path
import sys
import wandb

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
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


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
