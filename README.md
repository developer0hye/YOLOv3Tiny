# YOLOv3Tiny

Re-implementation PyTorch based YOLOv3Tiny

I tried to reproduce Darknet based YOLOv3Tiny using PyTorch.

# Performance

I used the ImageNet pretrained backbone network. 

Refer to [this project](https://github.com/developer0hye/PyTorch-ImageNet) to know training rules for training the pretrained network.


|Training Set|Test Set|mAP(mean Average Precision)|Weights|
|---|---|---|---|
|VOC 07+12 train/val|VOC 07 test|58.86|[download](https://drive.google.com/file/d/1NMFs2LjipSaFg9tUzll8T-ltOQLGVnT6/view?usp=sharing)|

# Prepare Dataset

Use [this script](https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py
) to prepare VOC dataset.


If you finished to download and convert labeling format, put the dataset folder into project folder.

```
├─backbone_weights
├─figures
├─test_example
├─VOCdataset
│  ├─test
│  └─train
| ...
| train.py
| test.py
```

# Training Command
```
python train.py
```

# Evaluation Command
```
python test.py --weights your_weight.pth
```
