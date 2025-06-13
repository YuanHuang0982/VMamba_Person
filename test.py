import os
import argparse
import random
import numpy as np
import faiss
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils import *
from metrics import *

from config import get_config
from models import build_model

# ------------------------- Argument Parsing & Setup -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Test Person ReID Model with VMamba')
    parser.add_argument('--seed', default=42)
    #parser.add_argument('--model_path', type=str, default="./output/personskip15batch32/vssm_tiny/baseline/vssm_tiny.pth")
    #parser.add_argument('--model_path', type=str, default="./output/personTokenMerge01/vssm_tiny/baseline/vssm_tiny.pth")
    #parser.add_argument('--model_path', type=str, default="./output/personBase/vssm_base/baseline/vssm_base.pth")
    parser.add_argument('--model_path', type=str, default="./output/oM486val6tvterase3/vssm1_tiny_0230s/baseline/vssm1_tiny_0230s.pth")
    #parser.add_argument('--test_data', type=str, default="./data/val")
    #parser.add_argument('--test_data', type=str, default="./occluded_reid1/val")
    #parser.add_argument('--test_data', type=str, default="./P_ETHZ_MarketStyle/val")
    parser.add_argument('--test_data', type=str, default="./occluded_split1re/test")
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--save_preds', type=str, default="./output/bbbb")
    return parser.parse_args()

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# ------------------------- Feature Extraction -------------------------
def extract_feature(model, dataloaders):
    imgs = torch.FloatTensor()
    features = torch.FloatTensor()
    for data in tqdm(dataloaders):
        img, _ = data
        img_copy = img.clone()
        imgs = torch.cat((imgs, img_copy), 0)
        img = img.to(device)
        with torch.no_grad():
            output = model(img)
            if isinstance(output, tuple):
                output = output[1]
            output = F.normalize(output, p=2, dim=1)
            features = torch.cat((features, output.cpu()), 0)
    return features, imgs

def get_id(img_path):
    camera_id = []
    labels = []
    for path, label in img_path:
        cam_id = int(path.split("/")[-1].split("_")[0])
        labels.append(int(label))
        camera_id.append(cam_id)
    return camera_id, labels

def search(query, k=10):
    encoded_query = query.unsqueeze(dim=0).numpy()
    top_k = index.search(encoded_query, k)
    gallery_imgs_idxs = top_k[1][0].copy()
    top_k[1][0] = np.take(gallery_label, indices=top_k[1][0])
    return top_k, gallery_imgs_idxs

# ------------------------- Visualize -------------------------
def visualize(query_img, gallery_imgs, gallery_idxs, label, gallery_labels, save_path):
    mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
    t = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize(size=(128, 48))
    ])
    plt.figure(figsize=(16., 6.))
    plt.subplot(1, 11, 1)
    img_tensor = query_img.clone()
    for i in range(3):
        img_tensor[i] = (img_tensor[i] * std[i]) + mean[i]
    x = t(img_tensor)
    x = np.array(x)
    plt.xticks([])
    plt.yticks([])
    plt.title("Query")
    plt.imshow(x)

    for j in range(10):
        img_tensor = gallery_imgs[gallery_idxs[j]].clone()
        for i in range(3):
            img_tensor[i] = (img_tensor[i] * std[i]) + mean[i]
        x = t(img_tensor)
        x = np.array(x)
        plt.subplot(1, 11, j + 2)
        plt.title("True" if gallery_labels[j] == label else "False")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    plt.close()

# ------------------------- Main Execution -------------------------
args = parse_args()
fix_seed(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 8
model_path = args.model_path
data_dir = args.test_data

# Data Transform
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Datasets and Dataloaders
image_datasets = {
    'query': datasets.ImageFolder(os.path.join(data_dir, 'query'), transform),
    'gallery': datasets.ImageFolder(os.path.join(data_dir, 'gallery'), transform)
}
query_loader = DataLoader(image_datasets['query'], batch_size=batch_size, shuffle=False)
gallery_loader = DataLoader(image_datasets['gallery'], batch_size=batch_size, shuffle=False)

# Load model
class Args:
    #cfg = './configs/vssm/vmambav0_tiny_224.yaml'
    cfg = './configs/vssm/vmambav2v_tiny_224.yaml'
    opts = None

args_cfg = Args()
config = get_config(args_cfg)
model = build_model(config).to(device)

ckpt = torch.load(model_path, map_location=device)
state_dict = ckpt['model'] if 'model' in ckpt else ckpt
model.load_state_dict(state_dict, strict=False)
model.eval()

# Extract features
query_feature, query_imgs = extract_feature(model, query_loader)
gallery_feature, gallery_imgs = extract_feature(model, gallery_loader)

# Get labels
gallery_cam, gallery_label = get_id(image_datasets['gallery'].imgs)
query_cam, query_label = get_id(image_datasets['query'].imgs)
gallery_label = np.array(gallery_label)
query_label = np.array(query_label)

# Build FAISS index
feature_dim = query_feature.shape[1]
index = faiss.IndexFlatIP(feature_dim)
index.add(gallery_feature.numpy())

# Make directories for visualizations
if args.visualize:
    os.makedirs(args.save_preds, exist_ok=True)
    os.makedirs(os.path.join(args.save_preds, "correct"), exist_ok=True)
    os.makedirs(os.path.join(args.save_preds, "incorrect"), exist_ok=True)

# Evaluation
rank1_score = 0
rank5_score = 0
ap_score = 0
count = 0

for query, label in zip(query_feature, query_label):
    query_img = query_imgs[count]
    output, gallery_idxs = search(query, k=10)

    r1 = rank1(label, output)
    rank1_score += r1
    rank5_score += rank5(label, output)
    ap_score += calc_map(label, output)

    #args.visualize = True
    if args.visualize:
        if r1:
            save_path = os.path.join(args.save_preds, "correct", f"{count:03d}.png")
            visualize(query_img, gallery_imgs, gallery_idxs, label, output[1][0], save_path)
        elif not r1:
            save_path = os.path.join(args.save_preds, "incorrect", f"{count:03d}.png")
            visualize(query_img, gallery_imgs, gallery_idxs, label, output[1][0], save_path)

    count += 1

# Final Results
print("Correct: {}, Total: {}, Incorrect: {}".format(rank1_score, count, count - rank1_score))
print("Rank1: %.3f, Rank5: %.3f, mAP: %.3f" % (
    rank1_score / count,
    rank5_score / count,
    ap_score / count
))
