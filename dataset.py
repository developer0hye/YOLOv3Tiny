import os
import cv2
import numpy as np
from numpy.random import RandomState
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torch.nn
import torchvision.transforms as transforms

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

def read_annotation_file(path):
    with open(path, 'r') as label:
        objects_information = []
        for line in label:
            line = line.split()
            if len(line) == 5:  # 0: class, 1:x, 2:y, 3:w, 4:h
                object_information = []
                for data in line:
                    object_information.append(float(data))
                objects_information.append(object_information)
        objects_information = np.asarray(objects_information).astype(np.float32)
        return objects_information


class YOLODataset(Dataset):
    def __init__(self,
                 path,
                 img_w=416,
                 img_h=416,
                 seed=21,
                 use_augmentation=True):

        self.imgs_path, self.labels_path = load_dataset(path)
        self.labels = [np.loadtxt(label_path,
                                  dtype=np.float32,
                                  delimiter=' ').reshape(-1, 5)  for label_path in self.labels_path]

        self.img_w = img_w
        self.img_h = img_h

        assert len(self.imgs_path) == len(self.labels_path), "영상의 갯수와 라벨 파일의 갯수가 서로 맞지 않습니다."

        self.use_augmentation = use_augmentation
        self.prng = RandomState(seed)

    def __getitem__(self, idx):
        assert Path(self.imgs_path[idx]).stem == Path(self.labels_path[idx]).stem, "영상과 어노테이션 파일의 짝이 맞지 않습니다."

        img = cv2.imread(self.imgs_path[idx], cv2.IMREAD_COLOR)

        label = self.labels[idx].copy()
        np.random.shuffle(label)

        bboxes_class = label[:, 0].astype(np.long).reshape(-1, 1)
        bboxes_xywh = label[:, 1:].reshape(-1, 4)

        if self.use_augmentation:
            bboxes_xyxy = augmentation.xywh2xyxy(bboxes_xywh)

            img, bboxes_xyxy, bboxes_class = augmentation.RandomCropPreserveBBoxes(img, bboxes_xyxy, bboxes_class)

            bboxes_xywh = augmentation.xyxy2xywh(bboxes_xyxy)
            img, bboxes_xywh, bboxes_class = augmentation.LetterBoxResize(img, (self.img_w, self.img_h), bboxes_xywh, bboxes_class)
            img, bboxes_xywh = augmentation.HorFlip(img, bboxes_xywh)
            bboxes_xyxy = augmentation.xywh2xyxy(bboxes_xywh)

            img, bboxes_xyxy, bboxes_class = augmentation.RandomTranslation(img, bboxes_xyxy, bboxes_class)
            img, bboxes_xyxy, bboxes_class = augmentation.RandomScale(img, bboxes_xyxy, bboxes_class)
            img = augmentation.ColorJittering(img)
            img = augmentation.RandomErasePatches(img, bboxes_xyxy)

            bboxes_xywh = augmentation.xyxy2xywh(bboxes_xyxy)
        else:
            img, bboxes_xywh, bboxes_class = augmentation.LetterBoxResize(img, (self.img_w, self.img_h), bboxes_xywh, bboxes_class)
        
        bboxes_xyxy = augmentation.xywh2xyxy(bboxes_xywh)

        # for visualization
        # augmentation.drawBBox(img, bboxes_xyxy)
        # cv2.imshow("imgdasdad", img)
        # cv2.waitKey(0)

        bboxes_class = torch.from_numpy(bboxes_class)

        bboxes_xywh[:, [0, 2]] *= self.img_w
        bboxes_xywh[:, [1, 3]] *= self.img_h

        bboxes_xywh = torch.from_numpy(bboxes_xywh)

        bboxes = torch.cat([bboxes_class, bboxes_xywh], dim=-1)

        img = img[..., ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.tensor(img, dtype=torch.float32)/255.

        return img, bboxes, idx

    def __len__(self):
        return len(self.imgs_path)


def yolo_collate(batch_data):
    batch_img = []
    batch_bboxes = []
    batch_idx = []

    for img, bboxes, idx in batch_data:
        batch_img.append(img)
        batch_bboxes.append(bboxes)
        batch_idx.append(idx)

    batch_img = torch.stack(batch_img, 0)
    return batch_img, batch_bboxes, batch_idx

if __name__ == '__main__':
    training_set = YOLODataset(path="test_example_v2",
                                    img_size=(416, 416),
                                    seed=10, 
                                    use_augmentation=True)
    
    for data in training_set:
        pass