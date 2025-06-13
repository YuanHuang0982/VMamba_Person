import torch
import os
import argparse
from fvcore.nn import FlopCountAnalysis
from models.vmambaDense import VSSM  # 정확한 경로에 맞게 조정 필요
from config import get_config
from models import build_model

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

'''
# 모델 구성 설정 (예시는 기본 설정)
model = VSSM(
    patch_size=4,
    in_chans=3,
    num_classes=160,
    imgsize=224,
    forward_type="v05_noz",  # 사용 중인 타입으로 설정
    ssm_ratio=1.0,
    ssm_d_state=1,
    ssm_dt_rank="auto",
    patchembed_version="v2",  # 실제 사용한 버전으로 바꾸세요
    downsample_version="v3",  # 실제 사용한 버전으로 바꾸세요
).eval()
'''
def parse_args():
    parser = argparse.ArgumentParser(description='Test Person ReID Model with VMamba')
    parser.add_argument('--model_path', type=str, default="./output/oM486b1/vssm1_tiny_0230s/baseline/vssm1_tiny_0230s.pth")
    return parser.parse_args()

args = parse_args()
model_path = args.model_path

# Load model
class Args:
    #cfg = './configs/vssm/vmambav0_tiny_224.yaml'
    cfg = './configs/vssm/vmambav2v_tiny_224.yaml'
    opts = None

args_cfg = Args()
config = get_config(args_cfg)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model(config).to(device)
ckpt = torch.load(model_path, map_location=device)
state_dict = ckpt['model'] if 'model' in ckpt else ckpt
model.load_state_dict(state_dict, strict=False)
model.eval()

# 입력 텐서 설정 (배치 크기 1, 채널 3, 224x224)
input_tensor = torch.randn(1, 3, 224, 224)

# FLOPs 분석 전에 모델과 입력 텐서를 모두 GPU로 올려야 합니다
model = model.to("cuda")  # 모델 GPU로 이동
input_tensor = input_tensor.to("cuda")  # 입력도 GPU로 이동


# FLOPs 분석
flops = FlopCountAnalysis(model, input_tensor)
print(f"총 FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
print(f"총 파라미터 수: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
