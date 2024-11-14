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

def run_train(train_loader, model, optimizer, args):
    """Train for one epoch.

    Args:
        train_loader: Dataloader which returns pair of TorchIO subjects per iteration
        model: Model
        optimizer: Pytorch optimizer
        args: Other script arguments
    """
    start_time = time.time()

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    model.train()

    res = []

    max_random_params = args.max_random_affine_augment_params

    for step_idx, subject in enumerate(train_loader):
        if step_idx == args.steps_per_epoch:
            break

        # Get images and segmentations from TorchIO subject
        img = subject["img"][tio.DATA]

        # Move to device
        img = img.float().to(args.device)

        # Explicitly augment moving image
        scale_augment = 1

        img = random_affine_augment(
            img,
            max_random_params=max_random_params,
            scale_params=scale_augment,
        )

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            with torch.amp.autocast(
                device_type="cuda", enabled=args.use_amp, dtype=torch.float16
            ):
                model_out = model(img)

                # Compute metrics
                metrics = {}
                metrics["train/scale_augment"] = scale_augment

                # Compute loss
                if args.label_name == "age":
                    label = subject[args.label_name].float().to(args.device)
                    pred = model_out["pred_out"]
                    metrics["train/age_mse"] = loss_ops.MSELoss()(pred, label)
                    metrics["train/age_mae"] = torch.nn.L1Loss()(pred, label)
                    loss = metrics["train/age_mae"]
                elif args.label_name == "sex":
                    label = subject[args.label_name].float().to(args.device)
                    pred = model_out["pred_out"]
                    metrics["train/sex_xe"] = F.cross_entropy(pred, label)
                    metrics["train/sex_acc"] = loss_ops.acc(
                        pred.argmax(dim=1), label.argmax(dim=1)
                    )
                    loss = metrics["train/sex_xe"]
                elif args.label_name == "group":
                    label = subject[args.label_name].float().to(args.device)
                    pred = model_out["pred_out"]
                    metrics["train/group_xe"] = F.cross_entropy(pred, label)
                    metrics["train/group_acc"] = loss_ops.acc(
                        pred.argmax(dim=1), label.argmax(dim=1)
                    )
                    loss = metrics["train/group_xe"]
                metrics["train/loss"] = loss

        # Perform backward pass
        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        end_time = time.time()
        metrics["train/epoch_time"] = end_time - start_time

        # Convert metrics to numpy
        metrics = {
            k: torch.as_tensor(v).detach().cpu().numpy().item()
            for k, v in metrics.items()
        }
        res.append(metrics)

        if args.visualize and step_idx == 0:
            plt.imshow(img[0, 0, img.shape[2] // 2].cpu().detach().numpy())
            plt.title(f"GT: {label}")
            plt.savefig(
                os.path.join(args.model_img_dir, f"img_{args.curr_epoch}.png")
            )
            plt.show()
            plt.close()

    return aggregate_dicts(res)


def run_val(val_loader, model, args):
    """Train for one epoch.

    Args:
        train_loader: Dataloader which returns pair of TorchIO subjects per iteration
        model: Model
        args: Other script arguments
    """
    start_time = time.time()

    model.eval()

    res = []

    for step_idx, subject in enumerate(val_loader):
        if step_idx == args.steps_per_epoch:
            break

        # Get images from TorchIO subject
        img = subject["img"][tio.DATA]

        # Move to device
        img = img.float().to(args.device)

        with torch.set_grad_enabled(False):
            model_out = model(
                img,
            )

            # Compute metrics
            metrics = {}

            # Compute loss
            if args.label_name == "age":
                label = subject[args.label_name].float().to(args.device)
                pred = model_out["pred_out"]
                metrics["val/age_mse"] = loss_ops.MSELoss()(pred, label)
                metrics["val/age_mae"] = torch.nn.L1Loss()(pred, label)
                loss = metrics["val/age_mae"]
            elif args.label_name == "sex":
                label = subject[args.label_name].float().to(args.device)
                pred = model_out["pred_out"]
                metrics["val/sex_xe"] = F.cross_entropy(pred, label)
                metrics["val/sex_acc"] = loss_ops.acc(
                    pred.argmax(dim=1), label.argmax(dim=1)
                )
                loss = metrics["val/sex_xe"]
            elif args.label_name == "group":
                label = subject[args.label_name].float().to(args.device)
                pred = model_out["pred_out"]
                metrics["val/group_xe"] = F.cross_entropy(pred, label)
                metrics["val/group_acc"] = loss_ops.acc(
                    pred.argmax(dim=1), label.argmax(dim=1)
                )
                loss = metrics["val/group_xe"]
            metrics["val/loss"] = loss

        end_time = time.time()
        metrics["val/epoch_time"] = end_time - start_time

        # Convert metrics to numpy
        metrics = {
            k: torch.as_tensor(v).detach().cpu().numpy().item()
            for k, v in metrics.items()
        }
        res.append(metrics)

    return aggregate_dicts(res)