import os
import cv2
import numpy as np
from numpy.random import RandomState
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torch.nn

import augmentation

def load_dataset(root):
    imgs_path = []
    labels_path = []

    for r, d, f in os.walk(root):
        for file in f:
            if file.lower().endswith((".png", ".jpg", ".bmp")):
                imgs_path.append(os.path.join(r, file).replace(os.sep, '/'))
            elif file.lower().endswith(".txt"):
                labels_path.append(os.path.join(r, file).replace(os.sep, '/'))
            
    return imgs_path, labels_path

class YOLODataset(Dataset):
    def __init__(self,
                 path,
                 img_w=416,
                 img_h=416,
                 use_augmentation=True):

        self.imgs_path, self.labels_path = load_dataset(path)
        self.labels = [np.loadtxt(label_path,
                                  dtype=np.float32,
                                  delimiter=' ').reshape(-1, 5) for label_path in self.labels_path]

        self.img_w = img_w
        self.img_h = img_h

        assert len(self.imgs_path) == len(self.labels_path), "영상의 갯수와 라벨 파일의 갯수가 서로 맞지 않습니다."

        self.use_augmentation = use_augmentation

    def __getitem__(self, idx):
        assert Path(self.imgs_path[idx]).stem == Path(self.labels_path[idx]).stem, "영상과 어노테이션 파일의 짝이 맞지 않습니다."

        img = cv2.imread(self.imgs_path[idx], cv2.IMREAD_COLOR)

        label = self.labels[idx].copy()
        np.random.shuffle(label)
        
        bboxes_class = label[:, 0].astype(np.long).reshape(-1, 1)
        bboxes_xywh = label[:, 1:].reshape(-1, 4)

        img, bboxes_xywh, bboxes_class, original_img_shape, non_padded_img_shape, padded_lt = augmentation.LetterBoxResize(img, (self.img_w, self.img_h), bboxes_xywh, bboxes_class)
        if self.use_augmentation:
            img, bboxes_xywh = augmentation.HorFlip(img, bboxes_xywh)
            bboxes_xyxy = augmentation.xywh2xyxy(bboxes_xywh)
            
            img, bboxes_xyxy, bboxes_class = augmentation.RandomCrop(img, bboxes_xyxy, bboxes_class)
            img, bboxes_xyxy, bboxes_class = augmentation.RandomTranslation(img, bboxes_xyxy, bboxes_class)
            img, bboxes_xyxy, bboxes_class = augmentation.RandomScale(img, bboxes_xyxy, bboxes_class)

            img = augmentation.ColorJittering(img)

            bboxes_xywh = augmentation.xyxy2xywh(bboxes_xyxy)

            # for visualization
            # augmentation.drawBBox(img, bboxes_xyxy)
            # cv2.imshow("imgdasdad", img)
            # cv2.waitKey(0)

        bboxes_class = torch.from_numpy(bboxes_class)
        bboxes_xywh = torch.from_numpy(bboxes_xywh)

        bboxes = torch.cat([bboxes_class, bboxes_xywh], dim=-1)

        img = img[..., ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.tensor(img, dtype=torch.float32)/255.

        data = {}
        data["img"] = img
        data["bboxes"] = bboxes
        data["idx"] = idx
        data["original_img_shape"] = original_img_shape
        data["non_padded_img_shape"] = non_padded_img_shape
        data["padded_lt"] = padded_lt

        return data

    def __len__(self):
        return len(self.imgs_path)


def yolo_collate(batch_data):

    batch_img = []
    batch_bboxes = []
    batch_idx = []
    batch_original_img_shape = []
    batch_non_padded_img_shape = []
    batch_padded_lt = []

    for data in batch_data:
        batch_img.append(data["img"])
        batch_bboxes.append(data["bboxes"])
        batch_idx.append(data["idx"])
        batch_original_img_shape.append(data["original_img_shape"])
        batch_non_padded_img_shape.append(data["non_padded_img_shape"])
        batch_padded_lt.append(data["padded_lt"])

    batch_img = torch.stack(batch_img, 0)
    
    batch_original_img_shape = np.stack(batch_original_img_shape, 0)
    batch_non_padded_img_shape = np.stack(batch_non_padded_img_shape, 0)
    batch_padded_lt = np.stack(batch_padded_lt, 0)

    batch_data = {}
    batch_data["img"] = batch_img
    batch_data["bboxes"] = batch_bboxes
    batch_data["idx"] = batch_idx
    batch_data["original_img_shape"] = batch_original_img_shape
    batch_data["non_padded_img_shape"] = batch_non_padded_img_shape
    batch_data["padded_lt"] = batch_padded_lt

    return batch_data
