"""Predicteur image -> 14 pathologies CheXpert (DenseNet-121, multi-label).

Entraine sur le split train, selectionne sur val (AUC macro), evalue sur test.
Sert a fournir a MCCAG des concepts predits depuis l IMAGE (pas le texte),
ce qui elimine la fuite. Sauvegarde le checkpoint + les probabilites par etude
pour les 3 splits.

Usage:
    python scripts/train_concept_predictor.py --epochs 30
    python scripts/train_concept_predictor.py --predict-only --checkpoint checkpoints/concept_predictor.pt
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "processed" / "metadata_r2gen.csv"
TARGETS = ROOT / "data" / "processed" / "concept_labels"
IMAGES = ROOT / "dataset" / "NLMCXR_png"
CKPT_DIR = ROOT / "checkpoints"
OUT_DIR = ROOT / "data" / "processed" / "concept_preds"

CONDITIONS = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
              'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
              'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
              'Support Devices', 'No Finding']

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


class CXRDataset(Dataset):
    def __init__(self, split, train_aug=False):
        meta = pd.read_csv(META)
        meta = meta[meta["split"] == split][["uid", "image"]].reset_index(drop=True)
        tgt = pd.read_csv(TARGETS / f"{split}_concept_targets.csv")
        self.df = meta.merge(tgt, on="uid", how="inner").reset_index(drop=True)
        self.train_aug = train_aug
        import torchvision.transforms as T
        if train_aug:
            self.tf = T.Compose([
                T.Resize((256, 256)),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
                T.ToTensor(),
                T.Normalize(IMG_MEAN, IMG_STD),
            ])
        else:
            self.tf = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(IMG_MEAN, IMG_STD),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(IMAGES / row["image"]).convert("RGB")
        x = self.tf(img)
        y = torch.tensor([row[c] for c in CONDITIONS], dtype=torch.float32)
        return {"image": x, "labels": y, "uid": row["uid"]}


class ConceptPredictor(nn.Module):
    def __init__(self, n_classes=14, pretrained=True):
        super().__init__()
        import torchvision.models as models
        try:
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.densenet121(weights=weights)
        except AttributeError:
            self.backbone = models.densenet121(pretrained=pretrained)
        in_feat = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_feat, n_classes)

    def forward(self, x):
        return self.backbone(x)


def compute_pos_weight(split="train"):
    tgt = pd.read_csv(TARGETS / f"{split}_concept_targets.csv")
    pos = tgt[CONDITIONS].sum().values.astype(float)
    n = len(tgt)
    neg = n - pos
    # pos_weight = neg/pos, clippe pour eviter des poids explosifs sur classes ultra-rares
    pw = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
    pw = np.clip(pw, 1.0, 50.0)
    return torch.tensor(pw, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device):
    from sklearn.metrics import roc_auc_score, f1_score
    model.eval()
    all_p, all_y = [], []
    for b in loader:
        logits = model(b["image"].to(device))
        all_p.append(torch.sigmoid(logits).cpu().numpy())
        all_y.append(b["labels"].numpy())
    P = np.concatenate(all_p)
    Y = np.concatenate(all_y)
    aucs, f1s = [], []
    for j in range(len(CONDITIONS)):
        if Y[:, j].sum() > 0 and Y[:, j].sum() < len(Y):
            try:
                aucs.append(roc_auc_score(Y[:, j], P[:, j]))
            except ValueError:
                aucs.append(float("nan"))
        else:
            aucs.append(float("nan"))
        f1s.append(f1_score(Y[:, j], (P[:, j] > 0.5).astype(int), zero_division=0))
    return P, Y, np.array(aucs), np.array(f1s)


def save_predictions(model, device):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import torchvision.transforms as T
    for split in ["train", "val", "test"]:
        ds = CXRDataset(split, train_aug=False)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        P, Y, _, _ = evaluate(model, loader, device)
        out = pd.DataFrame(P, columns=CONDITIONS)
        out.insert(0, "uid", [ds.df.iloc[i]["uid"] for i in range(len(ds.df))])
        out.to_csv(OUT_DIR / f"{split}_concept_probs.csv", index=False)
        print(f"  {split}: probabilites ecrites ({len(out)} etudes)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--predict-only", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    model = ConceptPredictor(pretrained=True).to(device)

    if args.predict_only:
        ckpt = args.checkpoint or str(CKPT_DIR / "concept_predictor.pt")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Charge {ckpt}, generation des probabilites...")
        save_predictions(model, device)
        return

    train_ds = CXRDataset("train", train_aug=True)
    val_ds = CXRDataset("val", train_aug=False)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    pos_weight = compute_pos_weight("train").to(device)
    print("pos_weight par pathologie:", [f"{w:.1f}" for w in pos_weight.cpu().numpy()])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_auc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for b in train_loader:
            optimizer.zero_grad()
            logits = model(b["image"].to(device))
            loss = criterion(logits, b["labels"].to(device))
            loss.backward()
            optimizer.step()
            total += loss.item()
        _, _, aucs, f1s = evaluate(model, val_loader, device)
        macro_auc = np.nanmean(aucs)
        print(f"Epoch {epoch:2d} | loss {total/len(train_loader):.4f} | val AUC macro {macro_auc:.4f}")
        if macro_auc > best_auc:
            best_auc = macro_auc
            torch.save(model.state_dict(), CKPT_DIR / "concept_predictor.pt")
            print(f"   -> nouveau meilleur (AUC {best_auc:.4f}), checkpoint sauve")

    print(f"\nMeilleur AUC macro (val): {best_auc:.4f}")

    # evaluation finale sur test avec le meilleur checkpoint
    model.load_state_dict(torch.load(CKPT_DIR / "concept_predictor.pt", map_location=device))
    test_ds = CXRDataset("test", train_aug=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    _, _, aucs, f1s = evaluate(model, test_loader, device)
    print("\n=== Performance TEST par pathologie ===")
    print(f"{'Pathologie':<28}{'AUC':>8}{'F1':>8}")
    for j, c in enumerate(CONDITIONS):
        auc_s = f"{aucs[j]:.3f}" if not np.isnan(aucs[j]) else "  -  "
        print(f"{c:<28}{auc_s:>8}{f1s[j]:>8.3f}")
    print(f"{'MACRO':<28}{np.nanmean(aucs):>8.3f}{np.mean(f1s):>8.3f}")

    print("\nGeneration des probabilites pour les 3 splits...")
    save_predictions(model, device)
    print("Termine.")


if __name__ == "__main__":
    main()
