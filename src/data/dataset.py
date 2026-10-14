# File: src/data/dataset.py
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
import glob
from src.data.fda import apply_fft_amplitude_swap

class DRDataset(Dataset):
    def __init__(self, csv_path: str = None, df: pd.DataFrame = None, img_dir: str = "", 
                 transform=None, is_training: bool = False, style_pool_dir: str = None,
                 fda_target_pool: list = None, fda_prob: float = 0.0, fda_beta: float = 0.03):
        
        # Accept either a CSV path or a direct Pandas DataFrame (for 5-Fold CV splits)
        if df is not None:
            self.df = df.reset_index(drop=True)
        else:
            self.df = pd.read_csv(csv_path)
            
        self.img_dir = img_dir
        self.transform = transform
        self.is_training = is_training
        
        # MixStyle Pool
        self.style_pool_dir = style_pool_dir
        self.style_images = []
        if self.is_training and self.style_pool_dir is not None:
            self.style_images = glob.glob(os.path.join(self.style_pool_dir, "*.*"))
            
        # FDA Pool
        self.fda_target_pool = fda_target_pool
        self.fda_prob = fda_prob
        self.fda_beta = fda_beta

    def __len__(self):
        return len(self.df)

    def _crop_fundus(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            margin = 5
            x, y = max(0, x - margin), max(0, y - margin)
            w, h = min(img.shape[1] - x, w + 2 * margin), min(img.shape[0] - y, h + 2 * margin)
            img = img[y:y+h, x:x+w]
        return img

    def _apply_clahe(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)

    def _apply_ben_graham(self, img: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=10)
        return cv2.addWeighted(img, 4, blur, -4, 128)

    def _process_image(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = self._crop_fundus(img)
        img = self._apply_clahe(img)
        img = self._apply_ben_graham(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row['id_code'])
        if not img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_name += '.png'
            
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)

        # Pre-FDA (Added safety check: len(self.fda_target_pool) > 0)
        if self.is_training and self.fda_target_pool is not None and len(self.fda_target_pool) > 0 and random.random() < self.fda_prob:
            tgt_img_path = random.choice(self.fda_target_pool)
            tgt_img = cv2.imread(tgt_img_path)
            if tgt_img is not None:
                img = apply_fft_amplitude_swap(img, tgt_img, beta=self.fda_beta)

        img = self._crop_fundus(img)
        img = self._apply_clahe(img)
        img = self._apply_ben_graham(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        label = int(row['diagnosis'])

        # MixStyle Target
        if self.is_training and len(self.style_images) > 0:
            style_path = random.choice(self.style_images)
            try:
                style_img = self._process_image(style_path)
                if self.transform:
                    style_img = self.transform(image=style_img)['image']
                return img, label, style_img
            except Exception:
                return img, label, img

        return img, label