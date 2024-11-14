import os
import torch
import torch.nn.functional as F
import numpy as np
import torchio as tio
import time
import matplotlib.pyplot as plt

from augmentation import random_affine_augment
import loss_ops as loss_ops

from script_utils import aggregate_dicts

def run_test(test_loader, model, args):
    """Evaluate the model on the test set.

    Args:
        test_loader: Dataloader which returns pair of TorchIO subjects per iteration
        model: Registration model
        args: Other script arguments
    """
    start_time = time.time()

    model.eval()
    res = []

    max_random_params = args.max_random_affine_augment_params

    for step_idx, subject in enumerate(test_loader):
        if step_idx == args.steps_per_epoch:
            break

        # Get images and segmentations from TorchIO subject
        img = subject["img"][tio.DATA]

        # Move to device
        img = img.float().to(args.device)

        with torch.set_grad_enabled(False):
            model_out = model(img)
            metrics = {}

            # Compute loss
            if args.label_name == "age":
                label = subject[args.label_name].float().to(args.device)
                pred = model_out["pred_out"]
                metrics["test/age_mse"] = loss_ops.MSELoss()(pred, label)
                metrics["test/age_mae"] = torch.nn.L1Loss()(pred, label)
                loss = metrics["test/age_mae"]
            elif args.label_name == "sex":
                label = subject[args.label_name].float().to(args.device)
                pred = model_out["pred_out"]
                metrics["test/sex_xe"] = F.cross_entropy(pred, label)
                metrics["test/sex_acc"] = loss_ops.acc(
                    pred.argmax(dim=1), label.argmax(dim=1)
                )
                loss = metrics["test/sex_xe"]
            elif args.label_name == "group":
                label = subject[args.label_name].float().to(args.device)
                pred = model_out["pred_out"]
                metrics["test/group_xe"] = F.cross_entropy(pred, label)
                metrics["test/group_acc"] = loss_ops.acc(
                    pred.argmax(dim=1), label.argmax(dim=1)
                )
                loss = metrics["test/group_xe"]
            metrics["test/loss"] = loss

        end_time = time.time()
        metrics["test/epoch_time"] = end_time - start_time

        metrics = {
            k: torch.as_tensor(v).detach().cpu().numpy().item()
            for k, v in metrics.items()
        }
        res.append(metrics)

    return aggregate_dicts(res)