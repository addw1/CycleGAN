import torch
import torch.nn as nn
from torchvision.models import inception_v3
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
import numpy as np
from scipy.linalg import sqrtm


class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self):
        super(InceptionV3FeatureExtractor, self).__init__()
        inception = inception_v3(pretrained=True, aux_logits=False, transform_input=False)
        # 使用Inception v3的平均池化层前的特征
        self.feature_extractor = nn.Sequential(
            *list(inception.children())[:-1],  # 保留到最后的平均池化层
            nn.Flatten()  # 展平为 (batch_size, feature_dim)
        )
        self.feature_extractor.eval()

    def forward(self, x):
        with torch.no_grad():
            print(f"Input shape to feature_extractor: {x.shape}")  # 调试信息
            x = self.feature_extractor(x)
            print(f"Output shape from feature_extractor: {x.shape}")  # 调试信息
            return x.view(x.size(0), -1)  # 展平为 (batch_size, feature_dim)


def calculate_fid(real_features, fake_features):
    # 计算均值和协方差
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    # 计算 Fréchet 距离公式
    diff = mu_real - mu_fake
    cov_mean, _ = sqrtm(sigma_real @ sigma_fake, disp=False)

    # 处理可能的数值问题
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * cov_mean)
    return fid

def preprocess_images(images, device):
    images = F.resize(images, [299, 299])
    images = F.normalize(images, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 将范围调整到[-1, 1]
    return images.to(device)


def compute_fid(real_loader, fake_loader, device='cuda'):
    # 初始化Inception特征提取器
    feature_extractor = InceptionV3FeatureExtractor().to(device)

    real_features = []
    fake_features = []

    # 提取真实图像的特征
    for real_images in real_loader:
        real_images = preprocess_images(real_images, device)
        features = feature_extractor(real_images)
        real_features.append(features.cpu().numpy())

    # 提取生成图像的特征
    for fake_images in fake_loader:
        fake_images = preprocess_images(fake_images, device)
        features = feature_extractor(fake_images)
        fake_features.append(features.cpu().numpy())

    # 拼接所有特征
    real_features = np.concatenate(real_features, axis=0)
    fake_features = np.concatenate(fake_features, axis=0)

    # 计算FID
    fid_score = calculate_fid(real_features, fake_features)
    return fid_score


from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = [
            os.path.join(root, f) for f in os.listdir(root) if f.endswith(('jpg', 'png'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")  # 确保图像为 RGB 格式
        if self.transform:
            img = self.transform(img)
        return img


# 假设real_loader和fake_loader是两个DataLoader，分别加载真实和生成的图像
real_dataset = CustomImageDataset('./output/A_final', transform=ToTensor())
fake_dataset = CustomImageDataset('./downloads/testA', transform=ToTensor())
real_loader = DataLoader(real_dataset, batch_size=16, shuffle=False)
fake_loader = DataLoader(fake_dataset, batch_size=16, shuffle=False)

for batch in real_loader:
    print(batch.shape)  # 确保形状为 [Batch Size, 3, Height, Width]
    break

device = 'cuda' if torch.cuda.is_available() else 'cpu'
fid = compute_fid(real_loader, fake_loader, device=device)
print(f"FID score: {fid}")
