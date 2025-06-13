import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from config import get_config
from models import build_model
from models.vmambaDense import CrossEntropyLabelSmooth, TripletLoss
from utils.logger import create_logger
from timm.utils import AverageMeter

import torch, numpy as np, faiss
import torch.nn.functional as F


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def validate_classification(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ce_fn, tri_fn = criterion_list  # 전역 선언된 [CrossEntropyLabelSmooth(), TripletLoss()]
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits, features = model(imgs)

            # features는 train 때처럼 L2 정규화
            features = F.normalize(features, p=2, dim=1)
            loss_ce  = ce_fn(logits, labels)
            loss_tri = tri_fn(features, labels)
            loss     = loss_ce + loss_tri
            
            bs = imgs.size(0)
            total_loss += loss.item() * bs
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total  += bs
    return total_loss/total, correct/total

def extract_features(model, loader, device):
    model.eval()
    feats, pids = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            _, feat = model(imgs)
            feats.append(feat.cpu().numpy())
            pids.extend(labels.numpy())
    return np.vstack(feats), np.array(pids)

def validate_reid(model, gall_loader, query_loader, device):
    # 1) feature 추출
    q_feats, q_pids = extract_features(model, query_loader, device)
    g_feats, g_pids = extract_features(model, gall_loader,  device)
    # 2) cosine distance 행렬
    q_norm = q_feats / np.linalg.norm(q_feats, axis=1, keepdims=True)
    g_norm = g_feats / np.linalg.norm(g_feats, axis=1, keepdims=True)
    distmat = 1 - np.dot(q_norm, g_norm.T)
    # 3) mAP, CMC 계산
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, None])
    all_cmc, all_AP = [], []
    for valid in matches:
        if not valid.any(): continue
        cmc = valid.cumsum(); cmc[cmc>1]=1
        all_cmc.append(cmc)
        precision = valid.cumsum() / (np.arange(len(valid)) + 1)
        all_AP.append((precision * valid).sum() / valid.sum())
    cmc = np.mean(all_cmc, axis=0)
    mAP = np.mean(all_AP)
    return cmc, mAP




# ------------------------- Argparse -------------------------
parser = argparse.ArgumentParser(description='Train Person ReID with VMamba')
#parser.add_argument('--cfg', type=str, default='./configs/vssm/vmambav0_tiny_224.yaml', help='config file')
parser.add_argument('--cfg', type=str, default='./configs/vssm/vmambav2v_tiny_224.yaml', help='config file')
parser.add_argument('--opts', default=None, nargs='+', help="Modify config options like MODEL.NAME newname")
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

# ------------------------- Fix Seed -------------------------
def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_seed(args.seed)

# ------------------------- Config & Logger -------------------------
config = get_config(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(config.OUTPUT, exist_ok=True)

logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")
logger.info(f"Training config:\n{config}")

# ------------------------- Dataset -------------------------
transform = transforms.Compose([
    transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.RandomErasing(
    p=0.8,
    scale=(0.06, 0.7),   # 마스크 영역 비율 범위
    ratio=(0.5, 1.5),     # 가로/세로 비율 범위
    value='random'
    ),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(config.DATA.TRAIN_PATH, transform)
train_loader = DataLoader(train_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=True, num_workers=config.DATA.NUM_WORKERS,drop_last=True)

# --- 검증용 Classification 데이터셋 & loader
val_cls_ds    = datasets.ImageFolder(config.DATA.VAL_PATH, transform)
val_cls_loader= DataLoader(val_cls_ds,   batch_size=config.DATA.BATCH_SIZE,
                           shuffle=False, num_workers=config.DATA.NUM_WORKERS)

# --- ReID 검증용 gallery / query loader
def make_reid_loader(root, subdir):
    ds = datasets.ImageFolder(os.path.join(root, subdir), transform)
    return DataLoader(ds, batch_size=config.DATA.BATCH_SIZE,
                      shuffle=False, num_workers=config.DATA.NUM_WORKERS)

val_gall_loader = make_reid_loader(config.DATA.TEST_PATH, 'gallery')
val_query_loader= make_reid_loader(config.DATA.TEST_PATH, 'query')


num_classes = len(train_dataset.classes)
config.defrost()
config.MODEL.NUM_CLASSES = num_classes
config.MODEL.PRETRAINED = ""


config.MODEL.FUSION_TOKEN = 48
config.MODEL.MERGE_STRATEGY = "upper"
config.MODEL.MERGE_LAYER = 6

config.freeze()
logger.info(f"Number of classes (person IDs): {num_classes}")

# ------------------------- Model -------------------------
model = build_model(config).to(device)
#model.set_skip_mode(False)
total_blocks = sum(len(layer.blocks) for layer in model.layers)
print(f" Total number of VSSBlocks in VMamba: {total_blocks}")


# ------------------------- Loss, Optimizer, Scaler -------------------------
criterion_list = [CrossEntropyLabelSmooth(), TripletLoss(margin=0.5)]
optimizer = optim.AdamW(
    model.parameters(),
    lr=config.TRAIN.BASE_LR,
    weight_decay=config.TRAIN.WEIGHT_DECAY
)
scaler = GradScaler(enabled=config.AMP_ENABLE)


# ------------------------- 지표 기록용 리스트 선언 -------------------------
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []
val_maps                 = []


# ------------------------- Best mAP 추적용 변수 -------------------------
best_map   = 0.0
best_epoch = 0

# ------------------------- Training Loop -------------------------
accumulation_steps = 4 # 가상 batch size = 8 x 4 = 32
logger.info("Start training")

for epoch in range(config.TRAIN.EPOCHS):
    model.train()
    loss_meter = AverageMeter()
    correct, total = 0, 0

    optimizer.zero_grad()

    for step, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)

        with autocast(enabled=config.AMP_ENABLE):
            logits, features = model(images)
            features = nn.functional.normalize(features, p=2, dim=1)
            loss_ce = criterion_list[0](logits, targets)
            loss_tri = criterion_list[1](features, targets)
            loss = loss_ce + loss_tri
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loss_meter.update((loss_ce + loss_tri).item(), targets.size(0))
        _, preds = torch.max(logits, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        if step % config.PRINT_FREQ == 0:
            logger.info(
                f"Epoch [{epoch}] Step [{step}/{len(train_loader)}] "
                f"Loss: {loss_meter.val:.4f} Avg: {loss_meter.avg:.4f}"
            )

    train_loss = loss_meter.avg          # AverageMeter 에 의해 이미 float
    train_acc  = correct / total         # Python float
    logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4%}")
    
        # ----------------- 여기에 검증 호출 -----------------
    # 1) Classification 관점
    val_loss, val_acc = validate_classification(model, val_cls_loader, device)
    logger.info(f"[Epoch {epoch}] Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4%}")

    # 2) ReID 관점
    cmc, val_mAP = validate_reid(model, val_gall_loader, val_query_loader, device)
    logger.info(f"[Epoch {epoch}] Val mAP: {val_mAP:.4%}, Rank-1: {cmc[0]:.4%}, Rank-5: {cmc[4]:.4%}")
    # --------------------------------------------------
    
    # ------------------ Best mAP 기록만 하기 ------------------
    if val_mAP > best_map:
        best_map   = val_mAP
        best_epoch = epoch
        logger.info(f"=> New best mAP {best_map:.4f} at epoch {best_epoch}")
    # ---------------------------------------------------------
    
    #  지표 기록
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    val_maps.append(val_mAP)


# ------------------------- 학습 종료 후 요약 출력 -------------------------
logger.info(f"Training finished. Best mAP {best_map:.4f} was at epoch {best_epoch}")   

# ------------------------- Save Model -------------------------
save_path = os.path.join(config.OUTPUT, f"{config.MODEL.NAME}.pth")
torch.save(model.state_dict(), save_path)
logger.info(f"Model saved at {save_path}")

import matplotlib.pyplot as plt


# 3) plotting: 한 번만 실행
epochs = list(range(1, len(train_losses)+1))

plt.figure()
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses,   label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, train_accs, label='Train Acc')
plt.plot(epochs, val_accs,   label='Val Acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, val_maps, label='Val mAP')
plt.xlabel('Epoch'); plt.ylabel('mAP')
plt.title('mAP Curve')
plt.legend()
plt.show()
