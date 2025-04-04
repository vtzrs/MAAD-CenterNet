# ------------------------------------------------------------------------------
# Modified by Vasileios Tzouras, 2024
# ------------------------------------------------------------------------------

import os.path
import random

import numpy as np
import torch
from loguru import logger
from maad_centernet.config import BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        # ------------------------ General Config ------------------------- #
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(
            "."
        )[0]
        self.seed = 111
        self.output_dir = "./results"
        self.debug = False
        self.gpus = [0]

        # ------------------------ General Config ------------------------- #

        self.debug = True
        self.center_thresh = 0.1
        self.down_ratio = 4

        # ------------------------ Model Config ----------------------- #
        self.pretrained_weights = (
            "imagenet"  # Either "imagenet" or a path to pretrained weights
        )
        self.head_complexity = {"hm": 0, "kp": 0, "off": 0}
        self.head_conv = 64
        self.up_dc = False

        self.init_bias = None

        # ------------------------ Training Config ----------------------- #
        self.arch = "msraresnet_50"
        self.batch_size = 32
        self.num_iterations = 6000
        self.lr = 0.001
        # Make on iteration level, for now on epoch level
        self.eval_interval = 50
        self.save_interval = 50
        self.print_interval = 4
        self.center_loss = "focal"  # CenterPoint: focal | mse
        self.regression_loss = "l1"  # CenterPoint: sl1 | l1 | l2
        self.vein_loss = False  # If true, for keypoint loss, we use a specifically designed vein loss.
        self.loss_weights = {
            "hm": 1.0,
            "kp": 0.1,
            "off": 1.0,
            "lamda_da": 1.0,
        }
        self.kp_weights = {
            "dist": 1.0,
            "angle": 1.0,
        }
        self.optimizer = "adam"
        self.weight_decay = 0
        self.warmup_epochs = 0
        self.warmup_lr = self.lr
        self.scheduler = "constant"  # constant, multistep, cos, warmcos
        self.multi_step_milestones = [90, 120]
        self.gamma = 0.1
        # ------------------------ Dataset Config ----------------------- #
        self.classes = ["leaf_blade"]
        self.num_classes = len(self.classes)
        self.data_num_workers = 4
        self.input_size = (256, 256)

        self.source_data_folder = "data/processed/RumexLeaves/iNaturalist"
        self.target_data_folder = "data/processed/RumexLeaves/RoboRumex"
        self.source_train_split_list = "random_train.txt"
        self.source_val_split_list = "random_val.txt"
        self.target_train_split_list = "random_train.txt"
        self.target_val_split_list = "random_val.txt"

        self.norm_target = False
        self.target_mode_conf = {
            "do_kp": True,
            "do_kphm": False,
            "do_obb": False,
            "do_da": False,
            "angle_mode": "sincos",  # 360
            "box_mode": "tlbr",  # wh
            "kphm_mode": "point",  # "segment"
            "cp_i": 3,
            "normalize": True,
        }
        self.cp_i = 2

        self.anno_file_id_inat = "annotations_oriented_bb"
        self.anno_file_roborumex = "../../annotations_oriented_bb"

        # ------------------------ Augmentations ----------------------- #
        self.p_flip = 0.0
        self.p_scale_shift = 0.0
        self.p_rotate = 0.0
        self.p_color_jitter = 0.0
        self.p_rgb_shift = 0.0
        self.rgb_shift_values = [30, 50, 30]
        self.jitter_values = [0.2, 0.2, 0.2, 0.2]
        self.p_blur = 0.0
        self.p_random_brightness_contrast = 0.0
        self.p_normalize = 1.0
        self.ds_mean = (0.47139463, 0.50289506, 0.360179)  # Format: (R, G, B)
        self.ds_std = (0.29554695, 0.2789082, 0.30842647)  # Format: (R, G, B)

    def get_device(self):
        device = torch.device("cuda" if self.gpus[0] >= 0 else "cpu")
        return device

    def config(self):
        self.down_ratio = 4  # output stride

    def get_model(self):
        from maad_centernet.models.bb_model import BBModel

        self.heads = {"hm": 1, "off": 2}
        if self.target_mode_conf["do_kp"]:
            mult = 2
            if self.target_mode_conf["angle_mode"] == "sincos":
                mult += 1
            self.heads["kp"] = mult * 8
        if self.target_mode_conf["do_kphm"]:
            dim = 8
            if self.target_mode_conf["kphm_mode"] == "segment":
                dim = 5
            self.heads["kphm"] = dim
        if self.target_mode_conf["do_obb"]:
            ddim = 3
            if self.target_mode_conf["box_mode"] == "tlbr":
                ddim += 2
            if self.target_mode_conf["angle_mode"] == "sincos":
                ddim += 1
            self.heads["obb"] = ddim
        self.num_layers = (
            int(self.arch[self.arch.find("_") + 1 :])
            if "_" in self.arch
            else 0
        )
        model = BBModel(self)
        model.train()
        return model

    def split_to_img_list(self, split_list, domain="source"):
        if self.target_mode_conf["do_da"]:
            if domain == "source":
                img_list = []
                with open(
                    f"{self.source_data_folder}/{split_list}",
                    "r+",
                ) as f:
                    img_list = [
                        line.replace("\n", "") for line in f.readlines()
                    ]
                return img_list
            elif domain == "target":
                img_list = []
                with open(
                    f"{self.target_data_folder}/{split_list}", "r+"
                ) as f:
                    img_list = [
                        line.replace("\n", "") for line in f.readlines()
                    ]
                return img_list
        else:
            if domain == "source":
                img_list = []
                with open(
                    f"{self.source_data_folder}/{split_list}",
                    "r+",
                ) as f:
                    img_list = [
                        line.replace("\n", "") for line in f.readlines()
                    ]
                return img_list

    def get_data_loader(self, split):

        from maad_centernet.data import (
            RumexLeavesDataset,
            TrainTransform,
            ValTransform,
        )

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(self.seed)

        shuffle = False
        if split == "train":
            transform = TrainTransform(
                max_labels=50,
                image_size=self.input_size,
                p_flip=self.p_flip,
                p_scale_shift=self.p_scale_shift,
                p_rotate=self.p_rotate,
                p_color_jitter=self.p_color_jitter,
                p_rgb_shift=self.p_rgb_shift,
                rgb_shift_values=self.rgb_shift_values,
                jitter_values=self.jitter_values,
                p_blur=self.p_blur,
                p_random_brightness_contrast=self.p_random_brightness_contrast,
                p_normalize=self.p_normalize,
                ds_mean=self.ds_mean,
                ds_std=self.ds_std,
            )

            if self.target_mode_conf["do_da"]:
                source_img_list = self.split_to_img_list(
                    self.source_train_split_list, domain="source"
                )
                target_img_list = self.split_to_img_list(
                    self.target_train_split_list, domain="target"
                )
            else:
                source_img_list = self.split_to_img_list(
                    self.source_train_split_list, domain="source"
                )
            shuffle = True

        elif split == "val":
            transform = ValTransform(
                image_size=self.input_size,
                p_normalize=self.p_normalize,
                ds_mean=self.ds_mean,
                ds_std=self.ds_std,
            )

            source_img_list = self.split_to_img_list(
                self.source_val_split_list, domain="source"
            )

        if self.target_mode_conf["do_da"] and split != "val":
            shuffle = True
            source_dataset = RumexLeavesDataset(
                self.source_data_folder,
                source_img_list,
                self.classes,
                preproc=transform,
                norm_target=self.norm_target,
                target_mode_conf=self.target_mode_conf,
                annotation_file_rel_to_img=self.anno_file_id_inat,
                cp_i=self.cp_i,
                domain_type="source",
            )
            target_dataset = RumexLeavesDataset(
                self.target_data_folder,
                target_img_list,
                self.classes,
                preproc=transform,
                norm_target=self.norm_target,
                target_mode_conf=self.target_mode_conf,
                annotation_file_rel_to_img=self.anno_file_roborumex,
                cp_i=self.cp_i,
                domain_type="target",
            )

            source_loader = torch.utils.data.DataLoader(
                source_dataset,
                batch_size=self.batch_size // 2,  # Half batch size for source
                shuffle=shuffle,
                num_workers=self.data_num_workers,
                pin_memory=True,
                drop_last=False,
                worker_init_fn=seed_worker,
            )

            target_loader = torch.utils.data.DataLoader(
                target_dataset,
                batch_size=self.batch_size // 2,  # Half batch size for target
                shuffle=shuffle,
                num_workers=self.data_num_workers,
                pin_memory=True,
                drop_last=False,
                worker_init_fn=seed_worker,
            )

            return source_loader, target_loader

        elif self.target_mode_conf["do_da"] == True and split == "val":
            dataset = RumexLeavesDataset(
                self.source_data_folder,
                source_img_list,
                self.classes,
                preproc=transform,
                norm_target=self.norm_target,
                target_mode_conf=self.target_mode_conf,
                annotation_file_rel_to_img=self.anno_file_id_inat,
                cp_i=self.cp_i,
                domain_type="source",
            )

            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size // 2,
                shuffle=shuffle,
                num_workers=self.data_num_workers,
                pin_memory=True,
                drop_last=False,
                worker_init_fn=seed_worker,
                generator=g,
            )
            return data_loader
        elif (
            self.target_mode_conf["do_da"] == False
            and split == "val"
            or split == "train"
        ):
            dataset = RumexLeavesDataset(
                self.source_data_folder,
                source_img_list,
                self.classes,
                preproc=transform,
                norm_target=self.norm_target,
                target_mode_conf=self.target_mode_conf,
                annotation_file_rel_to_img=self.anno_file_id_inat,
                cp_i=self.cp_i,
                domain_type="source",
            )

            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.data_num_workers,
                pin_memory=True,
                drop_last=False,
                worker_init_fn=seed_worker,
                generator=g,
            )
            return data_loader

    def get_optimizer(self, model):
        if self.optimizer not in ["adam"]:
            logger.warning(
                f"Optimizer {self.optimizer} not supported, using adam instead"
            )
            self.optimizer = "adam"
        if self.optimizer == "adam":
            optimizer = torch.optim.RAdam(
                model.parameters(),
                self.lr,
                weight_decay=self.weight_decay,
            )
        return optimizer

    def get_lr_scheduler(self, iters_per_epoch, num_epochs):
        from maad_centernet.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            self.lr,
            iters_per_epoch,
            num_epochs,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            milestones=self.multi_step_milestones,
            gamma=self.gamma,
        )
        return scheduler

    def get_evaluator(self, args):
        from maad_centernet.core.evaluator import Evaluator

        evaluator = Evaluator(self, args)

        return evaluator

    def get_trainer(self, args, trial=None):
        from maad_centernet.core.trainer import Trainer

        trainer = Trainer(self, args, trial)
        return trainer

    def eval(self, model, evaluator):
        return evaluator.evaluate(model)
