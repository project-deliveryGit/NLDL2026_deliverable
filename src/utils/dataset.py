import os
import cv2
import numpy as np
import torch
from PIL import Image
from collections import Counter
from torchvision import transforms
from torch.utils.data import Dataset, Subset, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit


class PhantomTripletDataset(Dataset):
    def __init__(self, root, case_folders, config):
        self.samples, self.groups = [], []
        self.img_size = config["img_size"]
        self.root = root
        self.use_augmentation = config.get("use_augmentation", False)

        if self.use_augmentation:
            self.img_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=config["aug_horizontal_flip"]),
                transforms.RandomRotation(config["aug_rotation"]),
                transforms.ColorJitter(brightness=config["aug_brightness"], contrast=config["aug_contrast"]),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ])

        for case in case_folders:
            img_dir = os.path.join(root, case, "preprocessed")
            mask_dir = os.path.join(root, case, "masks_edited")
            if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
                continue
            imgs = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
            if len(imgs) < 3:
                continue
            for i in range(1, len(imgs) - 1):
                self.samples.append((case, imgs[i]))
                self.groups.append(case)

    def __len__(self):
        return len(self.samples)

    def _load_img(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load: {path}")
        return cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

    def _load_mask(self, path):
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise ValueError(f"Failed to load: {path}")
        return cv2.resize(m, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

    def __getitem__(self, idx):
        case, center_name = self.samples[idx]
        img_dir = os.path.join(self.root, case, "preprocessed")
        mask_dir = os.path.join(self.root, case, "masks_edited")

        img_files = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
        cidx = img_files.index(center_name)

        im1 = self._load_img(os.path.join(img_dir, img_files[cidx - 1]))
        im2 = self._load_img(os.path.join(img_dir, center_name))
        im3 = self._load_img(os.path.join(img_dir, img_files[cidx + 1]))
        triplet = np.stack([im1, im2, im3], axis=-1)

        mask_name = center_name.replace("frame_", "mask_")
        mask = self._load_mask(os.path.join(mask_dir, mask_name)).astype(np.float32) / 255.0

        if self.use_augmentation:
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            triplet_t = self.img_transform(Image.fromarray(triplet))
            torch.manual_seed(seed)
            mask_t = transforms.Compose([
                transforms.RandomHorizontalFlip(p=self.img_transform.transforms[0].p),
                transforms.RandomRotation(self.img_transform.transforms[1].degrees),
                transforms.ToTensor(),
            ])(Image.fromarray((mask * 255).astype(np.uint8)))
        else:
            triplet_t = self.img_transform(Image.fromarray(triplet))
            mask_t = torch.from_numpy(mask).unsqueeze(0)

        return triplet_t, mask_t, self.groups[idx]


def make_case_balanced_sampler(dataset):
    case_ids = [dataset.dataset.groups[i] for i in dataset.indices] if isinstance(dataset, Subset) else dataset.groups
    counts = Counter(case_ids)
    weights = torch.DoubleTensor([1.0 / counts[c] for c in case_ids])
    return WeightedRandomSampler(weights, num_samples=len(case_ids), replacement=True)


def split_dataset(dataset, val_split, split_seed):
    indices = np.arange(len(dataset))
    case_to_int = {c: i for i, c in enumerate(sorted(set(dataset.groups)))}
    groups_int = np.array([case_to_int[c] for c in dataset.groups], dtype=int)
    gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=split_seed)
    (train_idx, val_idx), = gss.split(indices, groups=groups_int)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)