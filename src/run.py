import sys

import json
import os
import random
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torchio as tio

import wandb
from dataset import (
    adni_dataset,
)
import utils
from model import FinetuneModel
from unet3d.model import UNet3D
import script_utils
from script_utils import initialize_wandb, ParseKwargs
from train import run_train, run_val
from eval import run_test

def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--job_name",
        type=str,
        default="keymorph",
        help="Name of job",
    )
    parser.add_argument(
        "--run_mode",
        required=True,
        choices=["train", "eval"],
        help="Run mode",
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        required=True,
        help="Training dataset to use",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to data",
    )    
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="Num workers")
    parser.add_argument("--lr", type=float, default=3e-6, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=2000, help="Training Epochs")
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=10000000000,
        help="Number of gradient steps per epoch",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./output/",
        help="Path to the folder where outputs are saved",
    )
    parser.add_argument(
        "--pretrained_load_path",
        type=str,
        default=None,
        help="Load checkpoint at .h5 path",
    )
    parser.add_argument("--dim", type=int, default=3)    
    parser.add_argument(
        "--label_name",
        type=str,
        required=True,
        choices=["age", "sex", "group", "seg"],
        help="Type of SSL",
    )
    parser.add_argument(
        "--num_levels_for_unet",
        type=int,
        default=4,
        help="Number of levels for unet",
    )    
    parser.add_argument(
        "--max_random_affine_augment_params",
        nargs="*",
        type=float,
        default=(0.0, 0.0, 0.0, 0.0),
        help="Maximum of affine augmentations during training",
    )
    parser.add_argument("--use_amp", action="store_true", help="Use AMP")
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize images and points"
    )
    parser.add_argument("--use_wandb", action="store_true", help="Use Wandb")
    parser.add_argument(
        "--wandb_api_key_path",
        type=str,
        help="Path to Weights & Biases API Key. If use_wandb is set to True and this argument is not specified, user will be prompted to authenticate.",
    )
    parser.add_argument(
        "--wandb_kwargs",
        nargs="*",
        action=ParseKwargs,
        default={},
        help="keyword arguments for wandb.init() passed as key1=value1 key2=value2",
    )
    parser.add_argument(
        "--seed", type=int, default=23, help="Random seed use to sort the training data"
    )    

    args = parser.parse_args()
    return args


def create_dirs(args):
    arg_dict = vars(deepcopy(args))

    # Path to save outputs
    arguments = (
        args.job_name
        + "_data"
        + str(args.train_dataset)
        + "_batch"
        + str(args.batch_size)
        + "_lr"
        + str(args.lr)
    )

    args.model_dir = Path(args.save_dir) / arguments
    if not os.path.exists(args.model_dir):
        print("Creating directory: {}".format(args.model_dir))
        os.makedirs(args.model_dir)

    if args.run_mode == "eval":
        args.model_eval_dir = args.model_dir / "eval"
        if not os.path.exists(args.model_eval_dir):
            os.makedirs(args.model_eval_dir)

    else:
        args.model_ckpt_dir = args.model_dir / "checkpoints"
        if not os.path.exists(args.model_ckpt_dir):
            os.makedirs(args.model_ckpt_dir)
        args.model_img_dir = args.model_dir / "train_img"
        if not os.path.exists(args.model_img_dir):
            os.makedirs(args.model_img_dir)

    # Write arguments to json
    with open(os.path.join(args.model_dir, "args.json"), "w") as outfile:
        json.dump(arg_dict, outfile, sort_keys=True, indent=4)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def get_data(args):
    if args.train_dataset == "adni":
        transform = tio.Compose(
            [
                tio.ToCanonical(),
                tio.CropOrPad((160, 192, 192), padding_mode=0, include=("img",)),
                tio.Lambda(utils.rescale_intensity, include=("img",)),
            ]
        )
        dataset = adni_dataset.ADNIDataset(args.data_path)
    else:
        raise ValueError('Invalid dataset "{}"'.format(args.train_dataset))

    return {
        "transform": transform,
        "train": dataset.get_train_loader(
            args.batch_size, args.num_workers, transform=transform
        ),
        "val": dataset.get_val_loader(
            args.batch_size, args.num_workers, transform=transform
        ),
        "eval": dataset.get_test_loader(
            args.batch_size, args.num_workers, transform=transform
        ),
        "reference_subject": dataset.get_reference_subject(transform),
    }

def get_encoder(args):
    network = UNet3D(
        1,
        1,
        final_sigmoid=False,
        f_maps=32,  # Used by nnUNet
        layer_order="gcr",
        num_groups=8,
        num_levels=args.num_levels_for_unet,
        is_segmentation=False,
        conv_padding=1,
    )
    network.decoders = torch.nn.ModuleList([])
    network.final_conv = torch.nn.Identity()
    encoder = network

    return encoder

def get_model(args):
    encoder = get_encoder(args)
    
    # Main prediction network
    if args.label_name == "age":
        output_dim = 1
    elif args.label_name == "sex":
        output_dim = 2
    elif args.label_name == "group":
        output_dim = 3
    else:
        raise ValueError
    
    model = FinetuneModel(
        encoder,
        output_dim=output_dim,
        dim=args.dim,
    )
    model.to(args.device)
    script_utils.summary(model)
    
    return model

def main():
    args = parse_args()

    # Create run directories
    create_dirs(args)

    # Select GPU
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        print("WARNING! No GPU available, using the CPU instead...")
        args.device = torch.device("cpu")
    print("Number of GPUs: {}".format(torch.cuda.device_count()))

    # Set seed
    set_seed(args)

    # Data
    loaders = get_data(args)

    # Model
    model = get_model(args)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Load pretrained model
    print(f"Loading pretrained checkpoint from {args.pretrained_load_path}")
    ckpt_state, model = script_utils.load_checkpoint_pretrained_encoder(
        args.pretrained_load_path,
        model,
        device=args.device,
    )

    if args.use_wandb:
        initialize_wandb(args)

    if args.run_mode == "eval":
        # Evaluating
        print("Evaluating model...")

        test_epoch_stats = run_test(
                loaders["eval"],
                model,
                args,
            )

        for metric_name, metric in test_epoch_stats.items():
            print(f"[Test Stat] {metric_name}: {metric:.5f}")

    else:
        # Training
        print("Training model...")

        for epoch in range(1, args.epochs + 1):
            args.curr_epoch = epoch
            print(f"\nEpoch {epoch}/{args.epochs}")
            train_epoch_stats = run_train(
                loaders["train"],
                model,
                optimizer,
                args,
            )

            for metric_name, metric in train_epoch_stats.items():
                print(f"[Train Stat] {metric_name}: {metric:.5f}")

            val_epoch_stats = run_val(
                loaders["val"],
                model,
                args,
            )

            for metric_name, metric in val_epoch_stats.items():
                print(f"[Val Stat] {metric_name}: {metric:.5f}")

            epoch_stats = {**train_epoch_stats, **val_epoch_stats}
            wandb.log(epoch_stats)

            # Save model
            state = {
                "epoch": epoch,
                "args": args,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(
                state,
                os.path.join(
                    args.model_ckpt_dir,
                    "epoch{}_trained_model.pth.tar".format(epoch),
                ),
            )


if __name__ == "__main__":
    main()