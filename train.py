import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import get_config
from models import build_model
#from models.vmamba import CrossEntropyLabelSmooth, TripletLoss
from models.vmambaDense import CrossEntropyLabelSmooth, TripletLoss
from utils.logger import create_logger
from timm.utils import AverageMeter

# ------------------------- Argparse -------------------------
parser = argparse.ArgumentParser(description='Train Person ReID with VMamba')
parser.add_argument('--cfg', type=str, default='./configs/vssm/vmambav0_tiny_224.yaml', help='config file')
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
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset =  datasets.ImageFolder(config.DATA.TRAIN_PATH, transform)
train_loader = DataLoader(train_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=True, num_workers=config.DATA.NUM_WORKERS)

num_classes = len(train_dataset.classes)
config.defrost()
config.MODEL.NUM_CLASSES = num_classes
config.MODEL.PRETRAINED = ""
config.freeze()
logger.info(f"Number of classes (person IDs): {num_classes}")

# ------------------------- Model -------------------------
model = build_model(config).to(device)
model.set_skip_mode(True)
total_blocks = sum(len(layer.blocks) for layer in model.layers)
print(f" Total number of VSSBlocks in VMamba: {total_blocks}")
# ------------------------- Loss & Optimizer -------------------------
criterion_list = [CrossEntropyLabelSmooth(), TripletLoss(margin=0.5)]
optimizer = optim.AdamW(
    model.parameters(),
    lr=config.TRAIN.BASE_LR,
    weight_decay=config.TRAIN.WEIGHT_DECAY
)

# ------------------------- Training Function -------------------------
def train_one_epoch(epoch, model, loader, optimizer, criterions, config, logger):
    model.train()
    loss_meter = AverageMeter()
    correct, total = 0, 0

    for idx, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            logits, features = model(images)  # logits: classification / features: embedding
            features = nn.functional.normalize(features, p=2, dim=1)

            loss_ce = criterions[0](logits, targets)     # CrossEntropy
            loss_tri = criterions[1](features, targets)  # Triplet
            loss = loss_ce + loss_tri

        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), targets.size(0))
        _, preds = torch.max(logits, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        if idx % config.PRINT_FREQ == 0:
            logger.info(
                f"Epoch [{epoch}] Step [{idx}/{len(loader)}] "
                f"Loss: {loss_meter.val:.4f} Avg: {loss_meter.avg:.4f}"
            )

    acc = correct / total
    return {'train_loss': loss_meter.avg, 'train_accuracy': acc}

# ------------------------- Training Loop -------------------------
logger.info("Start training")
for epoch in range(config.TRAIN.EPOCHS):
    metrics = train_one_epoch(epoch, model, train_loader, optimizer, criterion_list, config, logger)
    logger.info(f"Epoch {epoch}: Loss: {metrics['train_loss']:.4f}, Accuracy: {metrics['train_accuracy']:.4f}")

# ------------------------- Save Model -------------------------
save_path = os.path.join(config.OUTPUT, f"{config.MODEL.NAME}.pth")
torch.save(model.state_dict(), save_path)
logger.info(f"Model saved at {save_path}")
