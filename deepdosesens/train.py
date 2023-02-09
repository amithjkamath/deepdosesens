# -*- encoding: utf-8 -*-
import os
import sys

if os.path.abspath("..") not in sys.path:
    sys.path.insert(0, os.path.abspath(".."))

import argparse

from data.dataloader_DLDP_C3D import get_loader
from training.network_trainer import NetworkTrainer
from model.C3D.model import Model
from model.C3D.online_evaluation import online_evaluation
from model.C3D.loss import Loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", type=int, default=2, help="batch size for training (default: 2)"
    )
    parser.add_argument(
        "--list_GPU_ids",
        nargs="+",
        type=int,
        default=[0],
        help="list_GPU_ids for training (default: [0])",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=80000,
        help="training iterations(default: 80000)",
    )
    args = parser.parse_args()

    #  Start training
    trainer = NetworkTrainer()
    trainer.setting.project_name = "C3D"
    trainer.setting.output_dir = (
        "/home/akamath/Documents/deep-planner/models/dldp-concave-1"
    )
    list_GPU_ids = args.list_GPU_ids

    trainer.setting.network = Model(
        in_ch=15,
        out_ch=1,
        list_ch_A=[-1, 16, 32, 64, 128, 256],
        list_ch_B=[-1, 32, 64, 128, 256, 512],
    )

    trainer.setting.max_iter = args.max_iter

    """
    # For the OpenKBP dataset, these are the entries.
    list_eval_dirs = [
        "/home/akamath/Documents/deep-planner/data/processed-kbp/pt_" + str(i)
        for i in range(161, 201)
    ]
    data_paths = {
        "train": [
            "/home/akamath/Documents/deep-planner/data/processed-kbp/pt_" + str(i)
            for i in range(1, 160)
        ],
        "val": list_eval_dirs,
    }
    """
    list_eval_dirs = [
        "/home/akamath/Documents/deep-planner/data/processed-dldp/DLDP_"
        + str(i).zfill(3)
        for i in range(62, 80)
        if i not in [63, 65, 67, 77]  # missing data
    ]
    list_eval_dirs += [
        "/home/akamath/Documents/deep-planner/data/processed-dldp/DLDP_"
        + str(i).zfill(3)
        for i in range(108, 109)
    ]

    list_train_dirs = [
        "/home/akamath/Documents/deep-planner/data/processed-dldp/DLDP_"
        + str(i).zfill(3)
        for i in range(1, 62)
        if i != 40  # missing data
    ]
    list_train_dirs += [
        "/home/akamath/Documents/deep-planner/data/processed-dldp/DLDP_"
        + str(i).zfill(3)
        for i in range(101, 107)
    ]

    data_paths = {
        "train": list_train_dirs,
        "val": list_eval_dirs,
    }

    trainer.setting.train_loader, trainer.setting.val_loader = get_loader(
        data_paths,
        train_bs=args.batch_size,
        val_bs=1,
        train_num_samples_per_epoch=args.batch_size * 500,  # 500 iterations per epoch
        val_num_samples_per_epoch=1,
        num_works=4,
    )

    trainer.setting.eps_train_loss = 0.01
    trainer.setting.lr_scheduler_update_on_iter = True
    trainer.setting.loss_function = Loss()
    trainer.setting.online_evaluation_function_dirs = list_eval_dirs
    trainer.setting.online_evaluation_function_val = online_evaluation

    trainer.set_optimizer(
        optimizer_type="Adam", args={"lr": 1e-3, "weight_decay": 1e-4}
    )

    trainer.set_lr_scheduler(
        lr_scheduler_type="cosine",
        args={"T_max": args.max_iter, "eta_min": 1e-7, "last_epoch": -1},
    )

    if not os.path.exists(trainer.setting.output_dir):
        os.mkdir(trainer.setting.output_dir)
    trainer.set_GPU_device(list_GPU_ids)
    trainer.run()

    trainer.print_log_to_file("# Done !\n", "a")
