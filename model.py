import json
import os

import torch
import torchvision.ops as ops
from torch import nn

import numpy as np

from backbone import YOLOv3TinyBackbone, ConvBnLeakyReLU

def darknet_conv_initializer(conv_layer):
    out_channels, in_channels, h, w = conv_layer.weight.shape
    scale = np.sqrt(2./(w*h*in_channels))
    nn.init.uniform_(conv_layer.weight, a=-scale, b=scale)

class YOLO(nn.Module):
    def __init__(self,
                 num_classes,
                 in_features,
                 anchor_box,
                 num_samples_per_class):

        super(YOLO, self).__init__()

        self.encode = nn.Conv2d(in_features, 4 + 1 + num_classes, 1)

        pi_objectness = 0.01# desired mean objectness score when model is initialized.
        total_num_samples = np.sum(num_samples_per_class)

        darknet_conv_initializer(self.encode)
        nn.init.constant_(self.encode.bias[0], 0)
        nn.init.constant_(self.encode.bias[1], 0)
        nn.init.constant_(self.encode.bias[2], 0)
        nn.init.constant_(self.encode.bias[3], 0)
        nn.init.constant_(self.encode.bias[4], -torch.log(torch.tensor((1.-pi_objectness)/pi_objectness)))
        
        for i, num_samples in enumerate(num_samples_per_class):
            pi_class = num_samples/total_num_samples
            pi_class = np.clip(pi_class, 0.01, 0.99)#for numerical stability
            nn.init.constant_(self.encode.bias[5 + i], -torch.log(torch.tensor((1.-pi_class)/pi_class)))

        self.anchor_box = anchor_box

    def decode(self, x, input_img_w, input_img_h):
        with torch.no_grad():
            x = x.clone()
            grid_h, grid_w = x.shape[2:]
            grid_y, grid_x = torch.meshgrid([torch.arange(grid_h), torch.arange(grid_w)])
            grid_xy = torch.stack([grid_x, grid_y])
            grid_xy = grid_xy.unsqueeze(0).to(x.device).type(x.dtype)

            anchors_wh = self.anchor_box.view(1, 2, 1, 1).to(x.device).type(x.dtype)

            # inference position
            x[:, [0, 1]] = grid_xy + torch.sigmoid(x[:, [0, 1]])  # range: (0, feat_w), (0, feat_h)


            x[:, 0] = (x[:, 0] / grid_w) * input_img_w
            x[:, 1] = (x[:, 1] / grid_h) * input_img_h
            
            
            # inference size

            # clip for numerical stability
            x[:, 2] = torch.clamp(x[:, 2], max=3.) # exp(3.) = 20.0855
            x[:, 3] = torch.clamp(x[:, 3], max=3.) # exp(3.) = 20.0855

            x[:, [2, 3]] = anchors_wh * torch.exp(x[:, [2, 3]]).type(x.dtype)  # range: (0, input_img_w), (0, input_img_h), to apply amp, we need to conver exp return type
            
            # inference objectness and class score
            x[:, 4] = torch.sigmoid(x[:, 4])
            x[:, 5:] = torch.sigmoid(x[:, 5:])
            class_prob, class_idx = torch.max(x[:, 5:], dim=1)

            x[:, 4] = x[:, 4] * class_prob # confidence!
            x[:, 5] = class_idx
            
            x = x[:, :6]

            return x

    def forward(self, x, input_img_w, input_img_h):
        encoded_bboxes = self.encode(x)
        decoded_bboxes = self.decode(encoded_bboxes, input_img_w, input_img_h)

        #flatten
        flatten_encoded_bboxes = encoded_bboxes.flatten(start_dim=2).transpose(1, 2)#[batch, 4 + 1 + num_classes, h*w] -> [batch, h*w, 4 + 1 + num_classes]
        flatten_decoded_bboxes = decoded_bboxes.flatten(start_dim=2).transpose(1, 2)#[batch, 4 + 1 + num_classes, h*w] -> [batch, h*w, 4 + 1 + num_classes]

        return flatten_encoded_bboxes, flatten_decoded_bboxes, encoded_bboxes.shape[1:], self.anchor_box

def load_model_json(model_json_file="yolov3tiny_voc.json"):
    with open(model_json_file, "r") as f:
        model_json = json.load(f)

        num_classes = model_json["num_classes"]

        labels_path = []
        for r, d, f in os.walk(os.path.join(model_json["dataset_root"],"train")):
            for file in f:
                if file.lower().endswith(".txt"):
                    labels_path.append(os.path.join(r, file).replace(os.sep, '/'))

        num_samples_per_class = np.zeros(num_classes)

        classes = [np.loadtxt(label_path,
                    dtype=np.float32,
                    delimiter=' ').reshape(-1, 5)[:, 0]  for label_path in labels_path]
        classes = np.concatenate(classes, axis=0)
        classes = classes.astype(int)
        uniques, counts = np.unique(classes, return_counts=True)

        for unique, count in zip(uniques, counts):
            num_samples_per_class[unique] = count

        anchor_boxes_mask = [] #append
        anchor_boxes = [] #extend

        anchor_boxes_mask.append(model_json['stride 16']['mask'])
        anchor_boxes.extend(model_json['stride 16']['anchors'])

        anchor_boxes_mask.append(model_json['stride 32']['mask'])
        anchor_boxes.extend(model_json['stride 32']['anchors'])

        return num_classes, num_samples_per_class, anchor_boxes_mask, anchor_boxes

class YOLOv3Tiny(nn.Module):
    def __init__(self,
                 model_json_file="yolov3tiny_voc.json",
                 backbone_weight_path="backbone_weights/tiny_best_top1acc58.97.pth"):
        super(YOLOv3Tiny, self).__init__()

        self.backbone = YOLOv3TinyBackbone()

        num_classes, num_samples_per_class, anchor_boxes_mask, anchor_boxes = load_model_json(model_json_file)
        
        self.num_classes = num_classes
        self.anchor_boxes_mask = torch.tensor(anchor_boxes_mask, dtype=torch.long)
        self.anchor_boxes = torch.tensor(anchor_boxes, dtype=torch.float32)
        
        self.yolo_layers = nn.ModuleList([])

        yolo_layers_in_features = [256, 512]

        for yolo_layer_in_features, mask in zip(yolo_layers_in_features, self.anchor_boxes_mask):
            for anchor_box in self.anchor_boxes[mask]:
                self.yolo_layers.append(YOLO(num_classes=self.num_classes,
                                             in_features=yolo_layer_in_features,
                                             anchor_box=anchor_box,
                                             num_samples_per_class=num_samples_per_class))

        self.neck_s32 = ConvBnLeakyReLU(1024, 256, 1)
        self.neck_s16 = nn.Identity()
        
        self.head_s32 = ConvBnLeakyReLU(256, 512, 3, 1)
        self.head_s16 = ConvBnLeakyReLU(384, 256, 3, 1)

        self.up_s32 = nn.Sequential(nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True),
                                    ConvBnLeakyReLU(256, 128, 1))
        
        #yolo_layer need different weight initialization method.
        for block in nn.ModuleList([self.neck_s32, self.head_s32, self.head_s16, self.up_s32]):
            for m in block.modules():
                if isinstance(m, nn.Conv2d):
                    darknet_conv_initializer(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        if os.path.exists(backbone_weight_path):
            model_state_dict = torch.load(backbone_weight_path)
            self.backbone.load_state_dict(model_state_dict)
            print("from pretrained!!!")
        
    def extract_features(self, x):
        feature_pyramid = self.backbone.extract_featrues(x)

        f32 = feature_pyramid["stride 32"]
        f32_neck = self.neck_s32(f32)
        f32 = self.head_s32(f32_neck)
        
        f16 = feature_pyramid["stride 16"]
        f16 = torch.cat([self.up_s32(f32_neck), f16], dim=1)
        f16_neck = self.neck_s16(f16)
        f16 = self.head_s16(f16_neck)

        return f16, f32

    def forward(self, x):
        batch_size, _, input_img_h, input_img_w = x.shape
        
        # backbone
        f16, f32 = self.extract_features(x)
        
        all_encoded_bboxes = []
        all_decoded_bboxes = []
        all_yolo_layers_output_shape = []
        anchor_boxes = []
        
        for f, mask in zip([f16, f32], self.anchor_boxes_mask):
            for idx in mask:
                encoded_bboxes, decoded_bboxes, yolo_layer_output_shape, anchor_box = self.yolo_layers[idx](f, input_img_w, input_img_h)

                all_encoded_bboxes.append(encoded_bboxes)
                all_decoded_bboxes.append(decoded_bboxes)
                all_yolo_layers_output_shape.append(yolo_layer_output_shape)
                anchor_boxes.append(anchor_box)
        
        all_encoded_bboxes = torch.cat(all_encoded_bboxes, dim=1)#[B, #num_bboxes, 4 + 1 + num_classes]
        all_decoded_bboxes = torch.cat(all_decoded_bboxes, dim=1)#[B, #num_bboxes, 4 + 1 + num_classes]

        output = {}
        output["batch_size"] = batch_size
        output["input_img_w"] = input_img_w
        output["input_img_h"] = input_img_h
        output["num_classes"] = self.num_classes
        output["device"] = x.device

        output["encoded_bboxes"] = all_encoded_bboxes
        output["decoded_bboxes"] = all_decoded_bboxes
        output["yolo_layers_output_shape"] = all_yolo_layers_output_shape
        output["anchor_boxes"] = anchor_boxes
        
        return output

def bboxes_filtering(output, batch_padded_lt,  batch_non_padded_img_shape, batch_original_img_shape, conf_thresh=0.25, iou_thresh=0.35, max_dets=100):
    batch_filtered_decoded_bboxes = []
    num_classes = output["num_classes"]
    for decoded_bboxes, padded_lt, non_padded_img_shape, original_img_shape in zip(output["decoded_bboxes"].cpu(), batch_padded_lt,  batch_non_padded_img_shape, batch_original_img_shape):
        filtered_decoded_bboxes = {"position": [], "confidence": [], "class": [], "num_detected_bboxes": 0}

        bboxes_confidence = decoded_bboxes[:, 4]
        confidence_mask = bboxes_confidence > conf_thresh

        decoded_bboxes = decoded_bboxes[confidence_mask]
        decoded_bboxes = decoded_bboxes[torch.argsort(decoded_bboxes[:, 4], descending=True)[:max_dets]]#confidence 순으로 나열했을때 상위 100개만 뽑음
        
        decoded_bboxes_xywh = decoded_bboxes[:, :4]
        decoded_bboxes_xywh = decoded_bboxes_xywh.type(torch.float32)

        decoded_bboxes_xywh[:, 0] -= padded_lt[0]
        decoded_bboxes_xywh[:, 1] -= padded_lt[1]

        decoded_bboxes_xywh[:, [0, 2]] /= non_padded_img_shape[0]
        decoded_bboxes_xywh[:, [1, 3]] /= non_padded_img_shape[1]

        decoded_bboxes_xywh[:, [0, 2]] *= original_img_shape[0]
        decoded_bboxes_xywh[:, [1, 3]] *= original_img_shape[1]

        decoded_bboxes_xyxy = ops.box_convert(decoded_bboxes_xywh, 'cxcywh', 'xyxy')
        
        #+0.3 mAP
        decoded_bboxes_xyxy[:, [0, 2]] = torch.clamp(decoded_bboxes_xyxy[:, [0, 2]], 0, original_img_shape[0])
        decoded_bboxes_xyxy[:, [1, 3]] = torch.clamp(decoded_bboxes_xyxy[:, [1, 3]], 0, original_img_shape[1])
        

        decoded_bboxes_confidence = decoded_bboxes[:, 4]
        decoded_bboxes_class = decoded_bboxes[:, 5]

        for idx_class in range(num_classes):
            class_mask = decoded_bboxes_class == idx_class

            if torch.count_nonzero(class_mask) == 0:
                continue
            
            nms_filtered_inds = ops.nms(decoded_bboxes_xyxy[class_mask], decoded_bboxes_confidence[class_mask], iou_thresh)

            filtered_decoded_bboxes["position"].append(decoded_bboxes_xywh[class_mask][nms_filtered_inds])
            filtered_decoded_bboxes["confidence"].append(decoded_bboxes_confidence[class_mask][nms_filtered_inds])
            filtered_decoded_bboxes["class"].append(decoded_bboxes_class[class_mask][nms_filtered_inds])

            filtered_decoded_bboxes["num_detected_bboxes"] += len(nms_filtered_inds)

        if filtered_decoded_bboxes["num_detected_bboxes"] > 0:
            filtered_decoded_bboxes["position"] = torch.cat(filtered_decoded_bboxes["position"], axis=0).numpy()
            filtered_decoded_bboxes["confidence"] = torch.cat(filtered_decoded_bboxes["confidence"], axis=0).numpy()
            filtered_decoded_bboxes["class"] = torch.cat(filtered_decoded_bboxes["class"], axis=0).numpy()
            
        batch_filtered_decoded_bboxes.append(filtered_decoded_bboxes)

    return batch_filtered_decoded_bboxes

def compute_iou(bboxes1, bboxes2, bbox_format):
    bboxes1 = bboxes1.type(torch.float32)
    bboxes2 = bboxes2.type(torch.float32)

    eps=1e-5 #to avoid divde by zero exception
    if bbox_format == "wh":
        assert bboxes1.shape[1] == 2
        assert bboxes2.shape[1] == 2
        
        overapped_w = torch.min(bboxes1[:, 0], bboxes2[:, 0])
        overapped_h = torch.min(bboxes1[:, 1], bboxes2[:, 1])
        overapped_area = overapped_w * overapped_h
        
        union_area = (bboxes1[:, 0] * bboxes1[:, 1]) + (bboxes2[:, 0] * bboxes2[:, 1]) - overapped_area

        iou = overapped_area / (union_area + eps)
        
        #It seems that margin is needed to consider floating point error"
        assert torch.min(iou) >= 0., "range of iou is [0, 1]"
        assert torch.max(iou) <= 1., "range of iou is [0, 1]"

        return iou
    elif bbox_format == "cxcywh":
        assert bboxes1.shape[1] == 4
        assert bboxes2.shape[1] == 4

        cx1, cy1, w1, h1 = bboxes1[:, 0], bboxes1[:, 1], bboxes1[:, 2], bboxes1[:, 3]
        cx2, cy2, w2, h2 = bboxes2[:, 0], bboxes2[:, 1], bboxes2[:, 2], bboxes2[:, 3]
        
        x11 = cx1 - (w1 / 2)
        y11 = cy1 - (h1 / 2)
        x12 = cx1 + (w1 / 2)
        y12 = cy1 + (h1 / 2)

        x21 = cx2 - (w2 /2)
        y21 = cy2 - (h2 /2)
        x22 = cx2 + (w2 /2)
        y22 = cy2 + (h2 /2)

    inter_x1 = torch.max(x11, x21)
    inter_y1 = torch.max(y11, y21)
    inter_x2 = torch.min(x12, x22)
    inter_y2 = torch.min(y12, y22)
    
    overapped_area = torch.clamp(inter_x2 - inter_x1 + 1, 0) * torch.clamp(inter_y2 - inter_y1  + 1, 0)
    union_area = (x12 - x11  + 1) * (y12 - y11  + 1) + (x22 - x21  + 1) * (y22 - y21  + 1) - overapped_area
    
    assert torch.count_nonzero(torch.isinf(overapped_area)) == 0
    assert torch.count_nonzero(torch.isinf(union_area)) == 0

    iou = overapped_area / (union_area + eps)

    assert torch.min(iou) >= 0., f"torch.min(iou) =  is {torch.min(iou)} range of iou is [0, 1]"
    assert torch.max(iou) <= 1., f"torch.max(iou) =  is {torch.max(iou)} range of iou is [0, 1]"

    return iou

def yololoss(batch_pred, batch_target, ignore_thresh=0.7):
    
    batch_size = batch_pred["batch_size"]
    device = batch_pred["device"]
    
    input_img_w, input_img_h = batch_pred["input_img_w"], batch_pred["input_img_h"]

    anchor_boxes = batch_pred["anchor_boxes"]
    anchor_boxes = torch.stack(anchor_boxes, dim=0) # shape: [n, 2]

    xy_loss = nn.BCEWithLogitsLoss(reduction='none')
    wh_loss = nn.MSELoss(reduction='none')
    objectness_loss = nn.BCEWithLogitsLoss(reduction='none')
    class_loss = nn.BCEWithLogitsLoss(reduction='none')

    loss_x = []
    loss_y = []
    loss_w = []
    loss_h = []
    loss_foreground_objectness = []
    loss_background_objectness = []
    loss_class_prob = []

    num_target_bboxes = 0

    for idx_batch in range(batch_size):

        target_encoded_bboxes = []
        scale_weight = []
        
        for yolo_layer_output_shape in batch_pred["yolo_layers_output_shape"]:
            target_encoded_bboxes.append(torch.zeros(yolo_layer_output_shape).to(device))
            scale_weight.append(torch.zeros(1, yolo_layer_output_shape[1], yolo_layer_output_shape[2]).to(device))

        target_bboxes = batch_target[idx_batch]
        target_bboxes[:, [1, 3]] *= input_img_w
        target_bboxes[:, [2, 4]] *= input_img_h

        for target_bbox in target_bboxes:
        
            target_bbox = target_bbox.clone().view(1, 5) # to compute iou with vectorization

            # 1. anchor box matching
            iou = compute_iou(anchor_boxes, target_bbox[:, 3:], bbox_format="wh")
            
            inds_sorted_iou = torch.argsort(iou, descending=True)
            idx_mathced_anchor_box = inds_sorted_iou[0]

            matched_target_enocded_bboxes = target_encoded_bboxes[idx_mathced_anchor_box]
            matched_scale_weight = scale_weight[idx_mathced_anchor_box]

            # 2. make ground truth
            anchor_w, anchor_h = anchor_boxes[idx_mathced_anchor_box]
            grid_h, grid_w = matched_target_enocded_bboxes.shape[1:]

            target_x, target_y, target_w, target_h = target_bbox[0, 1:]
            
            if target_w < 2:
                continue
            
            if target_h < 2:
                continue

            grid_x = (target_x / input_img_w) * grid_w
            grid_y = (target_y / input_img_h) * grid_h
            
            #tl: top left
            grid_tl_x = int(torch.floor(grid_x))
            grid_tl_y = int(torch.floor(grid_y))

            offset_x = grid_x - grid_tl_x
            offset_y = grid_y - grid_tl_y

            assert offset_x >= 0. and offset_x <= 1.
            assert offset_y >= 0. and offset_y <= 1.

            assert target_w > 0.
            assert target_h > 0.

            assert anchor_w > 0.
            assert anchor_h > 0.

            power_w = torch.log(target_w/anchor_w)
            power_h = torch.log(target_h/anchor_h)

            c = int(target_bbox[0, 0])

            if matched_target_enocded_bboxes[4, grid_tl_y, grid_tl_x] == 1.:#already marked, pass
                continue
            
            num_target_bboxes += 1

            matched_scale_weight[0, grid_tl_y, grid_tl_x] = 2. - ((target_w * target_h)/(input_img_w * input_img_h))
            matched_target_enocded_bboxes[0, grid_tl_y, grid_tl_x] = offset_x
            matched_target_enocded_bboxes[1, grid_tl_y, grid_tl_x] = offset_y
            matched_target_enocded_bboxes[2, grid_tl_y, grid_tl_x] = power_w
            matched_target_enocded_bboxes[3, grid_tl_y, grid_tl_x] = power_h
            matched_target_enocded_bboxes[4, grid_tl_y, grid_tl_x] = 1
            matched_target_enocded_bboxes[5 + c, grid_tl_y, grid_tl_x] = 1

        # flatten features
        flatten_pred_encoded_bboxes = batch_pred["encoded_bboxes"][idx_batch]
        flatten_pred_decoded_bboxes = batch_pred["decoded_bboxes"][idx_batch]
        
        flatten_target_encoded_bboxes = []
        flatten_scale_weight = []

        for idx_anchor_boxes in range(len(anchor_boxes)):
            flatten_target_encoded_bboxes.append(target_encoded_bboxes[idx_anchor_boxes].flatten(start_dim=1).T)
            flatten_scale_weight.append(scale_weight[idx_anchor_boxes].flatten(start_dim=1).T)

        flatten_target_encoded_bboxes = torch.cat(flatten_target_encoded_bboxes, dim=0)
        flatten_scale_weight = torch.cat(flatten_scale_weight, dim=0)

        assert flatten_pred_encoded_bboxes.shape == flatten_target_encoded_bboxes.shape

        foreground_mask = flatten_target_encoded_bboxes[:, 4] == 1
        background_mask = flatten_target_encoded_bboxes[:, 4] == 0

        with torch.no_grad():
            target_bboxes = target_bboxes.to(device)
            for target_bbox in target_bboxes:
                target_bbox = target_bbox.view(1, 5)
                target_class = target_bbox[0, 0]

                iou = compute_iou(flatten_pred_decoded_bboxes[:, :4] ,target_bbox[:, 1:], bbox_format="cxcywh")
                background_mask[(iou > ignore_thresh) & (flatten_pred_decoded_bboxes[:, 5] == target_class)] = 0.# ignore
        
        if torch.count_nonzero(foreground_mask) > 0:
            #localization
            loss_x.append(flatten_scale_weight[foreground_mask, 0] * xy_loss(flatten_pred_encoded_bboxes[foreground_mask, 0], flatten_target_encoded_bboxes[foreground_mask, 0]))
            loss_y.append(flatten_scale_weight[foreground_mask, 0] * xy_loss(flatten_pred_encoded_bboxes[foreground_mask, 1], flatten_target_encoded_bboxes[foreground_mask, 1]))
            loss_w.append(flatten_scale_weight[foreground_mask, 0] * wh_loss(flatten_pred_encoded_bboxes[foreground_mask, 2], flatten_target_encoded_bboxes[foreground_mask, 2]))
            loss_h.append(flatten_scale_weight[foreground_mask, 0] * wh_loss(flatten_pred_encoded_bboxes[foreground_mask, 3], flatten_target_encoded_bboxes[foreground_mask, 3]))
            loss_foreground_objectness.append(objectness_loss(flatten_pred_encoded_bboxes[foreground_mask, 4], flatten_target_encoded_bboxes[foreground_mask, 4]))
            loss_class_prob.append(class_loss(flatten_pred_encoded_bboxes[foreground_mask, 5:], flatten_target_encoded_bboxes[foreground_mask, 5:]))

        loss_background_objectness.append(objectness_loss(flatten_pred_encoded_bboxes[background_mask, 4], flatten_target_encoded_bboxes[background_mask, 4]))
    
    loss_x = torch.sum(torch.cat(loss_x))/batch_size
    loss_y = torch.sum(torch.cat(loss_y))/batch_size
    loss_w = torch.sum(torch.cat(loss_w))/batch_size
    loss_h = torch.sum(torch.cat(loss_h))/batch_size

    loss_foreground_objectness = torch.cat(loss_foreground_objectness)
    loss_background_objectness = torch.cat(loss_background_objectness)

    loss_objectness = (torch.sum(loss_foreground_objectness)+torch.sum(loss_background_objectness))/batch_size
    loss_class_prob = torch.sum(torch.cat(loss_class_prob))/batch_size

    # print("********")
    # print(loss_x)
    # print(loss_y)
    # print(loss_w)
    # print(loss_h)
    # print(loss_objectness)
    # print(loss_class_prob)
    # exit()

    # anchor_boxes 수에 따라 background 샘플 수가 많아지고 이에따라 class imbalance 문제 심해짐, 이거를 anchor_boxes수로 나누어서 anchor boxes 수에 어느정도 independent하게끔 loss를 설계
    # batch_size 도 위와 같은 이유로 나누어줌
    # 입력 이미지의 Scale 또한 background 샘플 수에 영향을 미치는 factor인데... 이거는 어떻게 고려해주는 게 좋을까...흠... 기준이 될만한 factor가 없네
    # 뭐랄까... 설명하기 어려움, positive sample들을 target_bboxes수로 나누어주면 batch_size, 이미지내 positive sample 수에 어느정도 independent해지듯이...
    # negative sample들도 그럼 negative sample 수로 나눠주면 되지 않냐고 생각하게 되는데 실제로 mean 해서 backpropagation 시키면 FP(배경을 물체로 detection) 케이스 수가 기하급수적 증가됨
    # 학습의 성패는 loss_background_objectness 를 얼마나 잘 컨트롤 해주냐에 따라 갈림

    loss = loss_x + loss_y + loss_w + loss_h + loss_objectness + loss_class_prob
    
    return loss