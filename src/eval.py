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

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import roc_curve, auc

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
    all_labels = []
    all_preds = []

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

                # Store predictions and labels for AUC computation
                label = subject[args.label_name].long().to(args.device)

                # Convert one-hot encoded labels to class indices if necessary
                if label.ndim > 1 and label.size(1) > 1:
                    label = label.argmax(dim=1)

                # Ensure labels are 1D tensors
                label = label.view(-1)

                # Append labels and predictions
                all_labels.append(label.cpu().numpy())
                all_preds.append(torch.softmax(pred, dim=1)[:, 1].cpu().numpy())

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

    # Generate AUC plot for binary classification task
    if args.label_name in ["sex"]:
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        
        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Receiver Operating Characteristic ({args.label_name.capitalize()})")
        plt.legend(loc="lower right")
        plt.grid()

        # Save the plot to a file
        output_path = args.save_dir + "/" + "sex_roc_curve.png"
        # Ensure the directory exists
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    return aggregate_dicts(res)
