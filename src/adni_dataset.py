import os
import torchio as tio
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import itertools
from glob import glob
import pickle
from collections import defaultdict
import random

def adnigroup2int(group):
    mapping = {"EMCI": 1, "CN": 0, "MCI": 1, "LMCI": 1, "AD": 2}
    return torch.tensor(mapping[group]).long()


def adnisex2int(sex):
    mapping = {"M": 0, "F": 1}
    return torch.tensor(mapping[sex]).long()

def create_subjects_from_metadata(root_path, metadata_csv_path):
    # Load metadata from CSV
    metadata_df = pd.read_csv(metadata_csv_path)

    # Dictionary to hold torchio.Subject objects per subject_id
    subject_dict = defaultdict(list)

    # First, check if subject_dict.pkl already exists
    subject_dict_path = "/scratch/users/deantran/rssl_ppmi/adni_train_scripts/adni_subject_list.pkl"
    if os.path.exists(subject_dict_path):
        print(f"Found {subject_dict_path}!")
        with open(subject_dict_path, "rb") as f:
            subject_dict = pickle.load(f)
    else:
        # Iterate over the metadata DataFrame
        print(f"Didn't find {subject_dict_path}. Creating subjects...")
        for i, row in metadata_df.iterrows():
            print(f"{i + 1}/{len(metadata_df)}")
            if row["Group"] not in ["EMCI", "CN", "MCI", "LMCI", "AD"]:
                continue
            image_data_id = row["Image Data ID"]
            subject_id = row["Subject"]

            # Construct the expected directory structure based on the given format
            subject_dir = os.path.join(
                root_path, subject_id, "*", "*", image_data_id, "*.npy"
            )
            glob_paths = glob(subject_dir)

            # Walk through the directory structure to find the .npy file
            for path in glob_paths:
                # Create torchio.Subject with metadata
                subject = tio.Subject(
                    img=tio.ScalarImage(path, reader=numpy_reader),
                    image_data_id=image_data_id,
                    subject=subject_id,
                    group=row["Group"],
                    sex=row["Sex"],
                    age=row["Age"],
                    visit=row["Visit"],
                    modality=row["Modality"],
                    description=row["Description"],
                    acquisition_date=row["Acq Date"],
                    file_format=row["Format"],
                )
                subject_dict[subject_id].append(subject)

        with open(subject_dict_path, "wb") as f:
            pickle.dump(subject_dict, f)

    print(f"Total unique subjects: {len(subject_dict)}")

    with open(subject_dict_path, "rb") as f:
        subject_dict = pickle.load(f)

    # List to hold all group labels
    groups = []

    # Iterate over all subjects and their associated images
    for subject_list in subject_dict.values():
        for subject in subject_list:
            # Access the 'group' attribute of each torchio.Subject
            group = subject["group"]
            groups.append(group)

    # Compute the frequency of each group label
    from collections import Counter

    group_counts = Counter(groups)

    # Display the frequencies
    print("Group Frequencies Across All Images:")
    for group_label, count in group_counts.items():
        print(f"{group_label}: {count}")

    # Now split the subjects into train, val, test
    subject_ids = list(subject_dict.keys())
    random.shuffle(subject_ids)

    total_subjects = len(subject_ids)
    n_train = int(0.80 * total_subjects)
    n_val = int(0.05 * total_subjects)
    n_test = total_subjects - n_train - n_val  # Ensure the sum equals total_subjects

    train_ids = subject_ids[:n_train]
    val_ids = subject_ids[n_train : n_train + n_val]
    test_ids = subject_ids[n_train + n_val :]

    train_subjects = []
    val_subjects = []
    test_subjects = []

    for sid in train_ids:
        train_subjects.extend(subject_dict[sid])

    for sid in val_ids:
        val_subjects.extend(subject_dict[sid])

    for sid in test_ids:
        test_subjects.extend(subject_dict[sid])

    print(f"Total train images: {len(train_subjects)}")
    print(f"Total validation images: {len(val_subjects)}")
    print(f"Total test images: {len(test_subjects)}")

    return train_subjects, val_subjects, test_subjects


def numpy_reader(path):
    data = np.load(path).astype(np.float32)
    affine = np.eye(4)
    return data, affine


class ADNIDataset:
    def __init__(self, batch_size=8, num_workers=4, shuffle=True):
        """
        Initialize the ADNI dataset class.

        Parameters:
        - subjects: list of torchio.Subject objects
        - batch_size: number of samples per batch
        - num_workers: number of subprocesses to use for data loading
        - shuffle: whether to shuffle the dataset before splitting
        """
        root_path = "/scratch/groups/eadeli/data/stru/t1/adni"
        metadata_csv_path = "/oak/stanford/groups/kpohl/data/stru/t1/metadata/adni/preprocessed_ADNI_mri_2_14_2024.csv"
        self.train_subjects, self.val_subjects, self.test_subjects = (
            create_subjects_from_metadata(root_path, metadata_csv_path)
        )

        # Preprocess the metadata
        print("Preprocessing...")
        for subject in self.train_subjects + self.val_subjects + self.test_subjects:
            subject["group"] = F.one_hot(adnigroup2int(subject["group"]), 3)
            subject["sex"] = F.one_hot(adnisex2int(subject["sex"]), 2)
            subject["age"] = torch.tensor(subject["age"])[..., None]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.seg_available = False

    def get_reference_subject(self, transform=None):
        return {"0000": self.train_subjects[0]}

    def get_train_loader(self, batch_size, num_workers, transform=None):
        """
        Get the DataLoader for the training set.

        Returns:
        - train_loader: DataLoader for the training set
        """
        train_dataset = tio.SubjectsDataset(
            self.train_subjects, transform=transform
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=self.shuffle,
        )
        return train_loader

    def get_val_loader(self, batch_size, num_workers, transform=None):
        """
        Get the DataLoader for the validation set.

        Returns:
        - val_loader: DataLoader for the validation set
        """
        val_dataset = tio.SubjectsDataset(self.val_subjects, transform=transform)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
        return val_loader

    def get_test_loader(self, batch_size, num_workers, transform=None):
        """
        Get the DataLoader for the test set.

        Returns:
        - test_loader: DataLoader for the test set
        """
        test_dataset = tio.SubjectsDataset(self.test_subjects, transform=transform)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
        return test_loader