import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torchvision.transforms as transforms

from config import get_config
from models import build_model

# -------------------- 설정 -------------------- #
class Args:
    cfg = './configs/vssm/vmambav2v_tiny_224.yaml'
    opts = []

args_cfg = Args()
config = get_config(args_cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- 모델 로드 -------------------- #
model = build_model(config).to(device)
ckpt = torch.load("./output/personStiny_Occluded_REID_MarketStyle/vssm_tiny/baseline/vssm_tiny.pth", map_location=device)
state_dict = ckpt['model'] if 'model' in ckpt else ckpt
model.load_state_dict(state_dict, strict=False)
model.eval()

# -------------------- 이미지 전처리 -------------------- #
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

transform_eval = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

invTrans = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
])

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def generate_visualization(original_image, feature_map):
    feature_map = torch.nn.functional.interpolate(
        feature_map.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
        size=(224, 224),
        mode='bilinear',
        align_corners=False
    )
    feature_map = feature_map.squeeze().cpu().numpy()
    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-5)

    image_np = original_image.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-5)
    feature_map = 1.0 - feature_map

    return show_cam_on_image(image_np, feature_map)

# -------------------- Hook 등록 -------------------- #
features = {}

def hook_fn(module, input, output):
    # output: (B, H, W, C) 또는 (B, N, C)
    features['feat'] = output.detach()

# 마지막 SS2D 모듈에 hook 등록
for module in reversed(list(model.modules())):
    if module.__class__.__name__ == 'SS2D':
        module.register_forward_hook(hook_fn)
        print("[INFO] Hook registered on:", module)
        break

# -------------------- 실행 -------------------- #
@torch.no_grad()
def generate_feature_activation_map(model, image_tensor):
    features.clear()
    model(image_tensor)
    if 'feat' not in features:
        raise ValueError("히트맵 생성에 필요한 피처를 추출하지 못했습니다.")
    
    feat = features['feat']  # (B, H, W, C) or (B, N, C)
    
    # (B, H, W, C) → (B, C, H, W)
    if feat.dim() == 4:
        feat = feat.permute(0, 3, 1, 2)
    # (B, N, C) → (B, C, H, W)
    elif feat.dim() == 3:
        B, N, C = feat.shape
        H = W = int(N ** 0.5)
        feat = feat[:, :H * W, :].permute(0, 2, 1).reshape(B, C, H, W)
    else:
        raise ValueError(f"지원하지 않는 feature shape: {feat.shape}")

    # 방법1: L2 norm (or 방법2: feat.abs().mean(dim=1) 도 가능)
    feat = torch.norm(feat, dim=1)  # (B, H, W)

    return feat[0]  # 첫 번째 이미지에 대한 heatmap 반환


# -------------------- 이미지 로드 및 시각화 -------------------- #
image_path = './Occluded_REID_MarketStyle/val/query/165/165_02.jpg'
raw_image = Image.open(image_path).convert('RGB')
raw_image = raw_image.resize((224, 224))  # 시각화를 위해 원본도 리사이즈
image_tensor = transform_eval(raw_image).unsqueeze(0).to(device)

# 히트맵 생성
activation_map = generate_feature_activation_map(model, image_tensor)
vis = generate_visualization(invTrans(image_tensor.squeeze()), activation_map)

# 출력
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(raw_image)
plt.title("Original Image (224x224)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(vis)
plt.title("Heat Map (L2 from SS2D)")
plt.axis('off')

plt.tight_layout()
plt.show()
