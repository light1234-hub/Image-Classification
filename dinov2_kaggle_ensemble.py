"""DINOv2-L multi-resolution image classifier for kaggle.
"""

from __future__ import annotations

import copy
import gc
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms

SEED = 2026
ROOT = Path(os.getenv("DATA_ROOT", "/content"))
TRAIN_DIR = ROOT / "FIT5215_Dataset"
TEST_DIR = ROOT / "test_set" / "test_dataset"
SUBMISSION_TEMPLATE = ROOT / "test_set" / "my_solution.csv"
FINAL_OUTPUT = ROOT / "submission_238_90p0_PREV_10p0.csv"

EXPECTED_TEST_IMAGES = 11_681
VALIDATION_FRACTION = 0.15
NUM_TRAIN_VIEWS = 4
BACKBONE_BATCH_SIZE = 4
HEAD_BATCH_SIZE = 512
EPOCHS = 35
RESOLUTIONS = (238, 252, 266)
PATCH_COUNTS = {238: 17 * 17, 252: 18 * 18, 266: 19 * 19}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASS_NAME_CORRECTIONS = {"butterfiles": "butterflies"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class KaggleTestDataset(Dataset):
    """Return a transformed image and the numeric ID in its filename."""

    def __init__(self, directory: Path, transform):
        self.transform = transform
        self.paths = sorted(
            directory.glob("*_animal.jpeg"),
            key=lambda path: int(path.name.split("_", 1)[0]),
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image_id = int(path.name.split("_", 1)[0])
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), image_id


def make_transforms(resolution: int):
    # Downsample to 64x64 FIRST to match the test-image information content.
    # Upscaling afterward changes the ViT token grid, not the image detail.
    train_transform = transforms.Compose([
        transforms.Resize((64, 64), antialias=True),
        transforms.RandomCrop(64, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10),
        transforms.Resize((resolution, resolution), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((64, 64), antialias=True),
        transforms.Resize((resolution, resolution), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


def cls_mean_patch_features(features_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    cls = features_dict["x_norm_clstoken"]      # [batch, 1024]
    patch_mean = features_dict["x_norm_patchtokens"].mean(dim=1)  # [batch, 1024]
    return torch.cat((cls, patch_mean), dim=1)  # [batch, 2048]


@torch.inference_mode()
def extract_features(backbone: nn.Module, loader: DataLoader, use_amp: bool):
    feature_batches, target_batches = [], []
    start = time.time()
    total = len(loader)
    for number, (images, targets) in enumerate(loader, 1):
        images = images.to(DEVICE, dtype=torch.float32, non_blocking=True)
        if use_amp and DEVICE.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                output = backbone.forward_features(images)
                features = cls_mean_patch_features(output)
        else:
            output = backbone.forward_features(images)
            features = cls_mean_patch_features(output)
        feature_batches.append(features.float().cpu())
        target_batches.append(targets.long().cpu())
        if number % max(1, total // 10) == 0 or number == total:
            print(f"  {number}/{total} batches ({(time.time() - start)/60:.1f} min)")
    return torch.cat(feature_batches), torch.cat(target_batches)


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_backbone() -> nn.Module:
    # Requires internet on the first run to download the pretrained weights.
    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vitl14",
        pretrained=True,
        trust_repo=True,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model.to(DEVICE).eval()


def cache_features_for_resolution(
    resolution: int,
    backbone: nn.Module,
    train_indices: list[int],
    validation_indices: list[int],
    class_names: list[str],
):
    train_transform, eval_transform = make_transforms(resolution)
    val_dataset = Subset(datasets.ImageFolder(TRAIN_DIR, transform=eval_transform), validation_indices)
    val_loader = DataLoader(
        val_dataset, batch_size=BACKBONE_BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    test_dataset = KaggleTestDataset(TEST_DIR, eval_transform)
    if len(test_dataset) != EXPECTED_TEST_IMAGES:
        raise ValueError(f"Expected {EXPECTED_TEST_IMAGES} test images, found {len(test_dataset)}")
    test_loader = DataLoader(
        test_dataset, batch_size=BACKBONE_BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # Ensure token count for DINOv2 ViT-L/14 is as expected.
    sample, _ = next(iter(val_loader))
    with torch.inference_mode():
        n_patches = backbone.forward_features(sample[:1].to(DEVICE))["x_norm_patchtokens"].shape[1]
    assert n_patches == PATCH_COUNTS[resolution], (resolution, n_patches)

    all_features, all_labels = [], []
    for view in range(NUM_TRAIN_VIEWS):
        print(f"Resolution {resolution}: augmented training view {view + 1}/{NUM_TRAIN_VIEWS}")
        train_dataset = Subset(
            datasets.ImageFolder(TRAIN_DIR, transform=train_transform), train_indices
        )
        train_loader = DataLoader(
            train_dataset, batch_size=BACKBONE_BATCH_SIZE, shuffle=False,
            num_workers=2, pin_memory=True,
        )
        features, labels = extract_features(backbone, train_loader, use_amp=True)
        all_features.append(features)
        all_labels.append(labels)
        del train_loader, train_dataset, features, labels
        free_memory()

    train_features, train_labels = torch.cat(all_features), torch.cat(all_labels)
    print(f"Resolution {resolution}: validation FP32 features")
    val_features, val_labels = extract_features(backbone, val_loader, use_amp=False)
    print(f"Resolution {resolution}: test FP32 features")
    test_features, test_ids = extract_features(backbone, test_loader, use_amp=False)

    cache_path = ROOT / f"dinov2_vitl14_{resolution}_cls_meanpatch_features.pt"
    torch.save({
        "train_features": train_features,
        "train_labels": train_labels,
        "validation_features": val_features,
        "validation_labels": val_labels,
        "test_features": test_features,
        "test_ids": test_ids,
        "class_names": class_names,
        "feature_dimension": 2048,
        "input_resolution": resolution,
        "patch_tokens": n_patches,
        "num_train_views": NUM_TRAIN_VIEWS,
    }, cache_path)
    print("Saved feature cache:", cache_path)
    del train_features, train_labels, val_features, val_labels, test_features, test_ids
    free_memory()


def train_head(resolution: int, num_classes: int) -> None:
    seed_everything(SEED)
    cache_path = ROOT / f"dinov2_vitl14_{resolution}_cls_meanpatch_features.pt"
    checkpoint_path = ROOT / f"dinov2_vitl14_{resolution}_cls_meanpatch_classifier.pt"
    logits_path = ROOT / f"dinov2_vitl14_{resolution}_cls_meanpatch_test_logits.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)

    x_train = cache["train_features"].float()
    y_train = cache["train_labels"].long()
    x_val = cache["validation_features"].float()
    y_val = cache["validation_labels"].long()
    x_test = cache["test_features"].float()
    test_ids = cache["test_ids"].long()
    class_names = cache["class_names"]

    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=HEAD_BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val), batch_size=1024, shuffle=False
    )
    classifier = nn.Sequential(nn.Dropout(0.40), nn.Linear(2048, num_classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_accuracy, best_loss, best_epoch, best_state = 0.0, float("inf"), 0, None
    for epoch in range(1, EPOCHS + 1):
        classifier.train()
        train_correct = train_total = 0
        train_loss_sum = 0.0
        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(labels)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += len(labels)

        classifier.eval()
        val_correct = val_total = 0
        val_loss_sum = 0.0
        with torch.inference_mode():
            for features, labels in val_loader:
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                logits = classifier(features)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * len(labels)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += len(labels)

        val_accuracy = val_correct / val_total
        val_loss = val_loss_sum / val_total
        if val_accuracy > best_accuracy or (val_accuracy == best_accuracy and val_loss < best_loss):
            best_accuracy, best_loss, best_epoch = val_accuracy, val_loss, epoch
            best_state = copy.deepcopy(classifier.state_dict())
        scheduler.step()
        print(
            f"{resolution} | epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_loss_sum/train_total:.4f} | "
            f"train acc {100*train_correct/train_total:.2f}% | "
            f"val loss {val_loss:.4f} | val acc {100*val_accuracy:.4f}%"
        )

    if best_state is None:
        raise RuntimeError("No training checkpoint was selected.")
    classifier.load_state_dict(best_state)
    torch.save({
        "classifier_state_dict": best_state,
        "class_names": class_names,
        "feature_dimension": 2048,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "best_validation_loss": best_loss,
        "input_resolution": resolution,
    }, checkpoint_path)

    classifier.eval()
    test_logits = []
    with torch.inference_mode():
        for start in range(0, len(x_test), 1024):
            test_logits.append(classifier(x_test[start:start + 1024].to(DEVICE)).float().cpu())
    torch.save({
        "test_ids": test_ids,
        "test_logits": torch.cat(test_logits),
        "class_names": class_names,
        "resolution": resolution,
    }, logits_path)
    print(f"{resolution}: best epoch {best_epoch}, val acc {best_accuracy*100:.4f}%")
    del cache, x_train, y_train, x_val, y_val, x_test, classifier, train_loader, val_loader
    free_memory()


def make_submission(expected_class_names: list[str]) -> Path:
    logits_by_resolution = {}
    for resolution in RESOLUTIONS:
        path = ROOT / f"dinov2_vitl14_{resolution}_cls_meanpatch_test_logits.pt"
        logits_by_resolution[resolution] = torch.load(path, map_location="cpu", weights_only=True)

    ids = logits_by_resolution[238]["test_ids"].long()
    for resolution, data in logits_by_resolution.items():
        if not torch.equal(ids, data["test_ids"].long()):
            raise RuntimeError(f"Test IDs do not match for {resolution}.")
        if data["class_names"] != expected_class_names:
            raise RuntimeError(f"Class names/order differ for {resolution}.")

    probs = {
        resolution: torch.softmax(data["test_logits"].float(), dim=1)
        for resolution, data in logits_by_resolution.items()
    }
    # Winning leaderboard ensemble: 90% of 238 + 10% of previous blend.
    previous_probs = 0.975 * probs[252] + 0.025 * probs[266]
    final_probs = 0.90 * probs[238] + 0.10 * previous_probs
    final_labels = [expected_class_names[i] for i in final_probs.argmax(dim=1).tolist()]
    predictions = pd.DataFrame({"ID": ids.tolist(), "Label": final_labels})
    template = pd.read_csv(SUBMISSION_TEMPLATE)
    submission = template[["ID"]].merge(
        predictions, on="ID", how="left", validate="one_to_one"
    )
    if len(submission) != EXPECTED_TEST_IMAGES:
        raise RuntimeError("Unexpected number of submission rows.")
    if submission["ID"].duplicated().any() or submission["Label"].isna().any():
        raise RuntimeError("Duplicate IDs or missing labels in submission.")
    submission.to_csv(FINAL_OUTPUT, index=False)
    print("Final submission:", FINAL_OUTPUT)
    print("Weights: 238=90.00%, 252=9.75%, 266=0.25%")
    return FINAL_OUTPUT


def main() -> None:
    seed_everything(SEED)
    for path in (TRAIN_DIR, TEST_DIR, SUBMISSION_TEMPLATE):
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}. Extract the dataset archives first.")

    print("Device:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    base_dataset = datasets.ImageFolder(TRAIN_DIR)
    class_names = [CLASS_NAME_CORRECTIONS.get(n, n) for n in base_dataset.classes]
    if len(class_names) != 20:
        raise ValueError(f"Expected 20 classes; found {len(class_names)}")
    all_indices = np.arange(len(base_dataset))
    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=VALIDATION_FRACTION,
        random_state=SEED,
        stratify=np.asarray(base_dataset.targets),
    )
    train_idx, val_idx = train_idx.tolist(), val_idx.tolist()
    print("Classes:", class_names)
    print("Train/validation:", len(train_idx), len(val_idx))

    # Freeze DINOv2 and extract features. Reloading between resolutions is
    # intentional: each resolution gets an independent seeded feature run.
    for resolution in RESOLUTIONS:
        seed_everything(SEED)
        backbone = load_backbone()
        cache_features_for_resolution(
            resolution, backbone, train_idx, val_idx, class_names
        )
        del backbone
        free_memory()

    for resolution in RESOLUTIONS:
        train_head(resolution, num_classes=len(class_names))
    make_submission(class_names)


if __name__ == "__main__":
    main()
