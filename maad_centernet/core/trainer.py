#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ------------------------------------------------------------------------------
# Copyright (c) Megvii, Inc. and its affiliates.
# Code originally from: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/core/trainer.py
# ------------------------------------------------------------------------------
# Modified by Ronja Güldenring
# ------------------------------------------------------------------------------
# Modified by Vasileios Tzouras, 2024
# ------------------------------------------------------------------------------

import argparse
import math
import os
import shutil
import time

import numpy as np
import optuna
import torch
import torch.nn as nn
from loguru import logger
from maad_centernet.config import Exp
from maad_centernet.models.discri_low_feat import DiscriminatorLowFeat
from maad_centernet.models.discriminator import Discriminator
from maad_centernet.models.grl import GradientReversalLayer
from maad_centernet.utils import (
    MeterBuffer,
    WandbLogger,
    get_model_info,
    gpu_mem_usage,
    load_ckpt,
    save_checkpoint,
    setup_logger,
)
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """
    Core class to train a model.
    ...

    Attributes
    ----------
    exp : Exp
        Experiment configuration
    args :
        Additional arguments from the command line
    """

    def __init__(self, exp: Exp, args: argparse.Namespace, trial) -> None:
        """
        Sets up training configuration.
        """
        self.exp = exp
        self.args = args
        self.trial = trial

        # Training related attributes
        self.num_iterations = exp.num_iterations
        self.device = exp.get_device()

        # Data/dataloader related attributes
        self.input_size = exp.input_size
        self.best_aps = {
            "best_average_ap50": 0,
            "best_mAP50": 0,
            "best_OKS50": 0,
            "best_OKS95": 0,
        }

        # Metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        os.makedirs(self.file_name, exist_ok=True)
        if self.exp.debug:
            self.debug_folder = os.path.join(self.file_name, "debug")
            os.makedirs(self.debug_folder, exist_ok=True)

        setup_logger(
            self.file_name,
            filename="train_log.txt",
            mode="a",
        )

    def train(self) -> None:
        """
        Train function

        Returns
        -------
        None
        """
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def before_train(self) -> None:
        """
        Initializes/logs parameters relevant for training, such as optimizer, model, data loader, lr scheduler etc.

        Returns
        -------
        None
        """
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))
        shutil.copy(self.args.exp_file, self.file_name)

        # Model related init
        model = self.exp.get_model()
        logger.info(
            "Model Summary: {}".format(
                get_model_info(model, self.exp.input_size)
            )
        )
        model.to(self.device)

        # Solver related init
        self.optimizer = self.exp.get_optimizer(model)

        # Discriminator related init
        self.D_model = Discriminator(in_channels=64).to(self.device)
        self.optimizer_discr = torch.optim.Adam(
            self.D_model.parameters(), lr=0.0001, weight_decay=1e-5
        )

        self.D_low_feat_model = DiscriminatorLowFeat(in_channels=256).to(
            self.device
        )
        self.optimizer_discr_low_feat = torch.optim.RAdam(
            self.D_low_feat_model.parameters(), lr=0.0001, weight_decay=1e-5
        )

        # Value of epoch will be set in `resume_train`
        self.model = self.resume_train(model)
        self.epoch = 0

        if self.exp.target_mode_conf["do_da"]:
            self.source_train_loader, self.target_train_loader = (
                self.exp.get_data_loader("train")
            )
        else:
            self.source_train_loader = self.exp.get_data_loader("train")

        # max_iter means iters per epoch
        if self.exp.target_mode_conf["do_da"]:
            self.max_iter = min(
                len(self.source_train_loader), len(self.target_train_loader)
            )
            self.max_epoch = math.ceil(
                self.num_iterations // self.max_iter + 1
            )
            print("max_epoch", self.max_epoch)
        else:
            self.max_iter = len(self.source_train_loader)
            self.max_epoch = math.ceil(
                self.num_iterations // self.max_iter + 1
            )

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.max_iter, self.max_epoch
        )

        self.evaluator = self.exp.get_evaluator(self.args)
        # Tensorboard and Wandb loggers
        if self.args.logger == "tensorboard":
            self.tblogger = SummaryWriter(
                os.path.join(self.file_name, "tensorboard")
            )
        elif self.args.logger == "wandb":
            self.wandb_logger = WandbLogger.initialize_wandb_logger(
                self.args, self.exp, self.evaluator.val_loader.dataset
            )
        else:
            raise ValueError("logger must be either 'tensorboard' or 'wandb'")

        logger.info("Training start...")
        logger.info("\n{}".format(model))

    def before_epoch(self) -> None:
        """
        Initializes/Logs parameters relevant for training an epoch, such as progress in epoch.

        Returns
        -------
        None
        """
        logger.info("---> start train epoch{}".format(self.epoch + 1))

    def train_in_epoch(self) -> None:
        """
        Trains the model for one epoch.

        Returns
        -------
        None
        """
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def after_epoch(self) -> None:
        """
        Wraps up the training of one epoch, such as saving checkpoints, and evaluating the model.

        Returns
        -------
        None
        """
        if (self.epoch + 1) % self.exp.eval_interval == 0:
            update_best_ckpt = self.evaluate_and_save_model()
            self.save_ckpt(
                ckpt_name="latest", update_best_ckpt=update_best_ckpt
            )

    def before_iter(self) -> None:
        """
        Initializes parameters relevant for iterating over the training data.

        Returns
        -------
        None
        """
        self.model.train()
        self.D_model.train()
        self.D_low_feat_model.train()

    def train_in_iter(self) -> None:
        """
        Iterating over the training data.

        Returns
        -------
        None
        """
        if self.exp.target_mode_conf["do_da"]:
            for iter, (source_batch, target_batch) in enumerate(
                zip(self.source_train_loader, self.target_train_loader)
            ):
                self.before_iter()
                self.train_one_iter(iter, source_batch, target_batch)
                self.after_iter()
        else:
            for iter, source_batch in enumerate(self.source_train_loader):
                self.before_iter()
                self.train_one_iter(iter, source_batch)
                self.after_iter()

    def set_requires_grad(self, model, requires_grad=True):
        for param in model.parameters():
            param.requires_grad = requires_grad

    def train_one_iter(
        self, iter: int, source_batch: dict, target_batch: dict = None
    ) -> None:
        """
        Training the model for one iteration, i.e. forward, backward, update learning rate etc.

        Returns
        -------
        None
        """
        self.iter = iter
        total_iter = self.progress_in_iter
        iter_start_time = time.time()

        # Calculate p for GRL
        p = float(self.epoch / self.max_epoch)
        p = min(max(p, 0.0), 1.0)
        lambda_p = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

        # ----------------------------------------------------------------------
        if self.exp.target_mode_conf["do_da"]:
            # Source domain
            targets_source = {
                k: source_batch[k]
                for k in source_batch.keys()
                if k != "img" and k != "meta"
            }
            for k in targets_source:
                targets_source[k] = targets_source[k].to(self.device)
            imgs_source = source_batch["img"].to(self.device)

            # Target domain
            targets_target = {
                k: target_batch[k]
                for k in target_batch.keys()
                if k != "img" and k != "meta"
            }
            for k in targets_target:
                targets_target[k] = targets_target[k].to(self.device)
            imgs_target = target_batch["img"].to(self.device)
            data_end_time = time.time()

            # ------------------------------------------------------------------
            # Forward pass of the detection model
            (
                _,
                loss_stats_source,
                x_source_feat,
                x_target_feat,
                x_source_low,
                x_target_low,
            ) = self.model(
                imgs_source,
                imgs_target,
                targets_source,
                lambda_p,
                domain_type="source",
            )
            loss_det = loss_stats_source["loss"]

            # ------------------------------------------------------------------
            # High-Features Discriminator related parts

            source_reverse_feature = GradientReversalLayer.apply(
                x_source_feat, lambda_p
            )
            D_source_out = self.D_model(source_reverse_feature)

            target_reverse_feature = GradientReversalLayer.apply(
                x_target_feat, lambda_p
            )
            D_target_out = self.D_model(target_reverse_feature)

            source_loss = torch.zeros_like(D_source_out).to(
                D_source_out.device
            )
            target_loss = torch.zeros_like(D_target_out).to(
                D_target_out.device
            )

            source_label = torch.zeros_like(D_source_out).to(
                D_source_out.device
            )
            source_loss = F.binary_cross_entropy_with_logits(
                D_source_out, source_label
            )
            target_label = torch.ones_like(D_target_out).to(
                D_target_out.device
            )
            target_loss = F.binary_cross_entropy_with_logits(
                D_target_out, target_label
            )
            loss_da_high = source_loss + target_loss
            loss_stats_source["loss_da_high"] = loss_da_high

            # ------------------------------------------------------------------
            # Low-Features Discriminator related parts

            source_reverse_low = GradientReversalLayer.apply(
                x_source_low, lambda_p
            )
            target_reverse_low = GradientReversalLayer.apply(
                x_target_low, lambda_p
            )
            D_feat_source = self.D_low_feat_model(source_reverse_low)
            D_feat_target = self.D_low_feat_model(target_reverse_low)

            source_feat_loss = torch.mean(D_feat_source**2)
            target_feat_loss = torch.mean((1 - D_feat_target) ** 2)

            loss_da_low = source_feat_loss + target_feat_loss
            loss_stats_source["loss_da_low"] = loss_da_low
            # # ------------------------------------------------------------------
            loss_total = loss_det + 0.001 * loss_da_high + 0.0001 * loss_da_low
            loss_stats_source["loss_total"] = loss_total

            # Backward
            self.optimizer.zero_grad()
            self.optimizer_discr.zero_grad()
            self.optimizer_discr_low_feat.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=25
            )
            torch.nn.utils.clip_grad_norm_(
                self.D_model.parameters(), max_norm=25
            )
            torch.nn.utils.clip_grad_norm_(
                self.D_low_feat_model.parameters(), max_norm=25
            )
            self.optimizer.step()
            self.optimizer_discr.step()
            self.optimizer_discr_low_feat.step()

        else:
            # Source domain only - Forward
            targets = {
                k: source_batch[k]
                for k in source_batch.keys()
                if k != "img" and k != "meta"
            }
            for k in targets:
                targets[k] = targets[k].to(self.device)
            imgs = source_batch["img"].to(self.device)
            data_end_time = time.time()

            _, loss_stats_source = self.model(imgs, targets)
            self.optimizer.zero_grad()
            loss = loss_stats_source["loss"]
            loss.backward()
            self.optimizer.step()
        # ----------------------------------------------------------------------

        # Update learning rate
        lr = self.lr_scheduler.update_lr(total_iter)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        for param_group in self.optimizer_discr.param_groups:
            param_group["lr"] = lr

        for param_group in self.optimizer_discr_low_feat.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()

        # Logging
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **loss_stats_source,
        )

    def after_iter(self) -> None:
        """
        Wraps up the training of one iteration, such as logging stats.

        Returns
        -------
        None
        """
        # if (self.iter + 1) % self.exp.print_interval == 0:
        # max_iter_per_epoch = self.max_iter // self.max_epoch

        if (self.iter + 1) == (self.max_iter):
            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                [
                    "{}: {:.3f}".format(k, v.latest)
                    for k, v in loss_meter.items()
                ]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (
                    ", size: ({:d},{:d})".format(
                        self.input_size[0], self.input_size[1]
                    )
                )
            )

            if self.args.logger == "tensorboard":
                self.tblogger.add_scalar(
                    "lr", self.meter["lr"].latest, self.progress_in_iter
                )
                for k, v in loss_meter.items():
                    self.tblogger.add_scalar(
                        f"train/{k}", v.latest, self.progress_in_iter
                    )
            if self.args.logger == "wandb":
                metrics = {
                    "train/" + k: v.latest for k, v in loss_meter.items()
                }
                metrics.update({"train/lr": self.meter["lr"].latest})
                self.wandb_logger.log_metrics(
                    metrics, step=self.progress_in_iter
                )

            self.meter.clear_meters()

    def after_train(self) -> None:
        """
        Wraps up the training of the experiment, such as logging the best AP.

        Returns
        -------
        None
        """
        logger.info("Training of experiment is done and the best APs of:")
        for k, v in self.best_aps.items():
            logger.info(f"{k}: {v:.3f}")

        if self.args.logger == "wandb":
            self.wandb_logger.finish()

    @property
    def progress_in_iter(self) -> int:
        """
        Compute the progress in iteration.
        """
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Resume the training from the latest checkpoint or a specfic checkpoint.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be trained.

        Returns
        -------
        model : torch.nn.Module
            The model with loaded checkpoint weights.
        """
        # Resume training
        if self.args.resume:
            ckpt_file = ""
            if self.args.pretrained_weights is None:
                ckpt_file = os.path.join(
                    self.file_name, "latest" + "_ckpt.pth"
                )
            else:
                ckpt_file = self.args.ckpt

            if os.path.exists(ckpt_file):
                ckpt = torch.load(ckpt_file, map_location=self.device)
                model = load_ckpt(model, ckpt["model"])
                self.optimizer.load_state_dict(ckpt["optimizer"])
                self.best_aps = ckpt.pop(
                    "best_aps",
                    {
                        "best_average_ap50": 0,
                        "best_mAP50": 0,
                        "best_OKS50": 0,
                        "best_OKS95": 0,
                    },
                )
                start_epoch = (
                    self.args.start_epoch - 1
                    if self.args.start_epoch is not None
                    else ckpt["start_epoch"]
                )
                self.start_epoch = start_epoch
                logger.info(
                    "Loaded checkpoint '{}' (epoch {})".format(
                        ckpt_file, self.start_epoch
                    )
                )  # noqa
        # Loading pretrained weights
        else:
            ckpt_file = ""
            if self.args.pretrained_weights is not None:
                ckpt_file = self.args.pretrained_weights
            elif os.path.exists(self.exp.pretrained_weights):
                ckpt_file = self.exp.pretrained_weights
            if ckpt_file:
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
                logger.info(
                    "Loaded pretrained weights '{}'".format(ckpt_file)
                )  # noqa
        # resume training
        self.start_epoch = 0

        return model

    def evaluate_and_save_model(self) -> None:
        """
        Evaluate the model on the validation set and save the latest and best checkpoint.

        Returns
        -------
        None
        """
        (performance, loss_meter), bb_detections, kp_detections = (
            self.exp.eval(self.model, self.evaluator)
        )

        best_average_ap50 = (
            performance["mAP"]["ap50"] + performance["OKS"]["kp_all"]["ap50"]
        ) / 2

        if self.trial:
            self.trial.report(best_average_ap50, self.epoch)

            # Handle pruning based on the intermediate value
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        loss_str = ", ".join(
            ["{}: {:.3f}".format(k, v.latest) for k, v in loss_meter.items()]
        )

        update_best_ckpt = {
            "best_average_ap50": best_average_ap50
            > self.best_aps["best_average_ap50"],
            "best_mAP50": performance["mAP"]["ap50"]
            > self.best_aps["best_mAP50"],
            "best_OKS50": performance["OKS"]["kp_all"]["ap50"]
            > self.best_aps["best_OKS50"],
            "best_OKS95": performance["OKS"]["kp_all"]["ap50_95"]
            > self.best_aps["best_OKS95"],
        }

        self.best_aps = {
            "best_average_ap50": max(
                self.best_aps["best_average_ap50"], best_average_ap50
            ),
            "best_mAP50": max(
                self.best_aps["best_mAP50"], performance["mAP"]["ap50"]
            ),
            "best_OKS50": max(
                self.best_aps["best_OKS50"],
                performance["OKS"]["kp_all"]["ap50"],
            ),
            "best_OKS95": max(
                self.best_aps["best_OKS95"],
                performance["OKS"]["kp_all"]["ap50_95"],
            ),
        }

        if self.args.logger == "tensorboard":
            for k, v in loss_meter.items():
                self.tblogger.add_scalar(
                    f"train/{k}", v.latest, self.progress_in_iter
                )
            self.tblogger.add_scalar(
                "val/COCOAP50",
                performance["mAP"]["ap50"],
                self.progress_in_iter,
            )
            self.tblogger.add_scalar(
                "val/COCOAP50_95",
                performance["mAP"]["ap50_95"],
                self.progress_in_iter,
            )
            self.tblogger.add_scalar(
                "val/OKS50",
                performance["OKS"]["kp_all"]["ap50"],
                self.progress_in_iter,
            )
            self.tblogger.add_scalar(
                "val/OKS50_95",
                performance["OKS"]["kp_all"]["ap50_95"],
                self.progress_in_iter,
            )
        if self.args.logger == "wandb":
            metrics = {
                "val/loss/" + k: v.latest for k, v in loss_meter.items()
            }
            metrics["val/mAP/50"] = performance["mAP"]["ap50"]
            metrics["val/mAP/50_95"] = performance["mAP"]["ap50_95"]
            metrics["val/OKS_orig/50"] = performance["OKS"]["orig"]["ap50"]
            metrics["val/OKS_orig/50_95"] = performance["OKS"]["orig"][
                "ap50_95"
            ]
            metrics["val/OKS_all/50"] = performance["OKS"]["kp_all"]["ap50"]
            metrics["val/OKS_all/50_95"] = performance["OKS"]["kp_all"][
                "ap50_95"
            ]
            metrics["val/OKS_stem/50"] = performance["OKS"]["kp_stem"]["ap50"]
            metrics["val/OKS_stem/50_95"] = performance["OKS"]["kp_stem"][
                "ap50_95"
            ]
            metrics["val/OKS_vein/50"] = performance["OKS"]["kp_vein"]["ap50"]
            metrics["val/OKS_vein/50_95"] = performance["OKS"]["kp_vein"][
                "ap50_95"
            ]
            metrics["val/OKS_true/50"] = performance["OKS"]["kp_true"]["ap50"]
            metrics["val/OKS_true/50_95"] = performance["OKS"]["kp_true"][
                "ap50_95"
            ]
            metrics["val/OKS_inbetween/50"] = performance["OKS"][
                "kp_inbetween"
            ]["ap50"]
            metrics["val/OKS_inbetween/50_95"] = performance["OKS"][
                "kp_inbetween"
            ]["ap50_95"]
            metrics["train/epoch"] = self.epoch + 1

            self.wandb_logger.log_metrics(metrics, step=self.progress_in_iter)
            if self.exp.debug:
                pass
                # self.wandb_logger.log_images(predictions)
        logger.info(f"Evaluation after epoch {self.epoch + 1}: " + loss_str)
        summary = (
            f'Average Precision (AP)\t\t@[ IoU=0.50:0.95\t] = {performance["mAP"]["ap50_95"]:.2f}'
            + f'\nAverage Precision (AP)\t\t@[ IoU=0.50\t\t] = {performance["mAP"]["ap50"]:.2f}'
        )
        summary += "\n"
        summary += (
            f"\nObject Keypoint Similarity (OKS) @[ IoU=0.50:0.95\t] = {performance['OKS']['kp_all']['ap50_95']:.2f}"
            + f"\nObject Keypoint Similarity (OKS) @[ IoU=0.5\t\t] = {performance['OKS']['kp_all']['ap50']:.2f}"
        )
        logger.info("\n" + summary)
        return update_best_ckpt

    def save_ckpt(
        self, ckpt_name: str, update_best_ckpt: dict, ap: float = 0.0
    ) -> None:
        """
        Save the checkpoint.

        Parameters
        ----------
        ckpt_name : str
            The name of the checkpoint.
        update_best_ckpt : bool
            Whether to update the best checkpoint.
        ap : float
            The current AP.

        Returns
        -------
        None
        """
        save_model = self.model
        logger.info("Save weights to {}".format(self.file_name))
        ckpt_state = {
            "start_epoch": self.epoch + 1,
            "model": save_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_aps": self.best_aps,
            "curr_ap": ap,
        }
        save_checkpoint(
            ckpt_state,
            self.file_name,
            ckpt_name,
        )

        for k, v in update_best_ckpt.items():
            if v:
                save_checkpoint(
                    ckpt_state,
                    self.file_name,
                    k,
                )

        if self.args.logger == "wandb":
            self.wandb_logger.save_checkpoint(
                self.file_name,
                ckpt_name,
                update_best_ckpt,
                metadata={
                    "epoch": self.epoch + 1,
                    "optimizer": self.optimizer.state_dict(),
                    "best_aps": self.best_aps,
                    "curr_ap": ap,
                },
            )
