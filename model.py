import json

import torch
from torch import nn

import numpy as np

from backbone import darknet_tiny, Conv_BN_LeakyReLU


class YOLO(nn.Module):
    def __init__(self,
                 num_classes,
                 in_features,
                 anchor_box):
        super(YOLO, self).__init__()

        self.encode = nn.Conv2d(in_features, 4 + 1 + num_classes, 1)
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

        anchor_boxes_mask = [] #append
        anchor_boxes = [] #extend

        anchor_boxes_mask.append(model_json['stride 16']['mask'])
        anchor_boxes.extend(model_json['stride 16']['anchors'])

        anchor_boxes_mask.append(model_json['stride 32']['mask'])
        anchor_boxes.extend(model_json['stride 32']['anchors'])

        return num_classes, anchor_boxes_mask, anchor_boxes

class YOLOv3Tiny(nn.Module):
    def __init__(self,
                 model_json_file="yolov3tiny_voc.json",
                 backbone_weight_path=None):
        super(YOLOv3Tiny, self).__init__()

        num_classes, anchor_boxes_mask, anchor_boxes = load_model_json(model_json_file)
        
        self.num_classes = num_classes
        self.anchor_boxes_mask = torch.tensor(anchor_boxes_mask, dtype=torch.long)
        self.anchor_boxes = torch.tensor(anchor_boxes, dtype=torch.float32)
        
        self.backbone = darknet_tiny(backbone_weight_path)
        self.yolo_layers = nn.ModuleList([])

        yolo_layers_in_features = [256, 512]

        for yolo_layer_in_features, mask in zip(yolo_layers_in_features, self.anchor_boxes_mask):
            for anchor_box in self.anchor_boxes[mask]:
                self.yolo_layers.append(YOLO(num_classes=self.num_classes,
                                             in_features=yolo_layer_in_features,
                                             anchor_box=anchor_box))

        self.neck_s32 = Conv_BN_LeakyReLU(1024, 256, 1)
        self.neck_s16 = nn.Identity()
        
        self.head_s32 = Conv_BN_LeakyReLU(256, 512, 3, 1)
        self.head_s16 = Conv_BN_LeakyReLU(384, 256, 3, 1)

        self.up_s32 = nn.Sequential(nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True),
                                    Conv_BN_LeakyReLU(256, 128, 1))
        
    def extract_features(self, x):
        f16, f32 = self.backbone(x)

        f32_neck = self.neck_s32(f32)
        f32 = self.head_s32(f32_neck)
        
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
        output["model_input_shape"] = (input_img_h, input_img_w)
        output["num_classes"] = self.num_classes
        output["device"] = x.device

        output["encoded_bboxes"] = all_encoded_bboxes# len(all_encoded_bboxes) = 6
        output["decoded_bboxes"] = all_decoded_bboxes# len(all_decoded_bboxes) = 32
        output["yolo_layers_output_shape"] = all_yolo_layers_output_shape
        output["anchor_boxes"] = anchor_boxes
        
        return output


def nms(dets, scores, nms_thresh=0.45):
    """"Pure Python NMS baseline."""
    x1 = dets[:, 0]  # xmin
    y1 = dets[:, 1]  # ymin
    x2 = dets[:, 2]  # xmax
    y2 = dets[:, 3]  # ymax

    areas = (x2 - x1) * (y2 - y1)  # the size of bbox
    order = scores.argsort()[::-1]  # sort bounding boxes by decreasing order

    keep = []  # store the final bounding boxes
    while order.size > 0:
        i = order[0]  # the index of the bbox with highest confidence
        keep.append(i)  # save it to keep
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-28, xx2 - xx1)
        h = np.maximum(1e-28, yy2 - yy1)
        inter = w * h

        # Cross Area / (bbox + particular area - Cross Area)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # reserve all the boundingbox whose ovr less than thresh
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]

    return keep


def bboxes_filtering(output, conf_thresh=0.25, max_dets=100):
    filtered_batch_multi_scale_bboxes = []

    num_classes = output["num_classes"]

    for single_multi_scale_bboxes in output["decoded_bboxes"]:
        filtered_single_multi_scale_bboxes = {}

        confidence = single_multi_scale_bboxes[:, 4]
        is_postive = confidence > conf_thresh

        position = single_multi_scale_bboxes[is_postive, :4]
        confidence, class_idx = confidence[is_postive], single_multi_scale_bboxes[is_postive, 5]

        position = position.cpu().numpy()
        confidence = confidence.cpu().numpy()
        class_idx = class_idx.cpu().numpy()

        sorted_inds_by_conf = np.argsort(confidence)[::-1][:max_dets]

        position = position[sorted_inds_by_conf]
        confidence = confidence[sorted_inds_by_conf]
        class_idx = class_idx[sorted_inds_by_conf]

        def xywh2xyxy(box_xywh):
            box_xyxy = box_xywh.copy()
            box_xyxy[..., 0] = box_xywh[..., 0] - box_xywh[..., 2] / 2.
            box_xyxy[..., 1] = box_xywh[..., 1] - box_xywh[..., 3] / 2.
            box_xyxy[..., 2] = box_xywh[..., 0] + box_xywh[..., 2] / 2.
            box_xyxy[..., 3] = box_xywh[..., 1] + box_xywh[..., 3] / 2.
            return box_xyxy

        # NMS
        keep = np.zeros(len(position), dtype=np.int)
        for i in range(num_classes):
            inds = np.where(class_idx == i)[0]
            if len(inds) == 0:
                continue

            c_bboxes = position[inds]
            c_scores = confidence[inds]
            c_keep = nms(xywh2xyxy(c_bboxes), c_scores, 0.45)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        position = position[keep]
        confidence = confidence[keep]
        class_idx = class_idx[keep]

        filtered_single_multi_scale_bboxes["position"] = position
        filtered_single_multi_scale_bboxes["confidence"] = confidence
        filtered_single_multi_scale_bboxes["class"] = class_idx

        filtered_batch_multi_scale_bboxes.append(filtered_single_multi_scale_bboxes)

    return filtered_batch_multi_scale_bboxes

def compute_iou(bboxes1, bboxes2, bbox_format):
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
        
        x11 = cx1 - w1 / 2
        y11 = cy1 - h1 / 2
        x12 = cx1 + w1 / 2 
        y12 = cy1 + h1 / 2

        x21 = cx2 - w2 /2
        y21 = cy2 - h2 /2
        x22 = cx2 + w2 /2
        y22 = cy2 + h2 /2

    inter_x1 = torch.max(x11, x21)
    inter_y1 = torch.max(y11, y21)
    inter_x2 = torch.min(x12, x22)
    inter_y2 = torch.min(y12, y22)
    
    overapped_area = torch.clamp(inter_x2 - inter_x1, 0) * torch.clamp(inter_y2 - inter_y1, 0) 
    union_area = (x12 - x11) * (y12 - y11) + (x22 - x21) * (y22 - y21) - overapped_area

    iou = overapped_area / (union_area + eps)
    assert torch.min(iou) >= 0., f"torch.min(iou) =  is {torch.min(iou)} range of iou is [0, 1]"
    assert torch.max(iou) <= 1., f"torch.max(iou) =  is {torch.max(iou)} range of iou is [0, 1]"
    return iou

def yololoss(pred, target, ignore_thresh=0.7):

    batch_size = pred["batch_size"]
    device = pred["device"]
    
    input_img_h, input_img_w = pred["model_input_shape"]

    anchor_boxes = pred["anchor_boxes"]
    anchor_boxes = torch.stack(anchor_boxes, dim=0) # shape: [n, 2]

    mse_loss = nn.MSELoss(reduction='none')
    bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction='none')

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
        
        for yolo_layer_output_shape in pred["yolo_layers_output_shape"]:
            target_encoded_bboxes.append(torch.zeros(yolo_layer_output_shape).to(device))

        target_bboxes = target[idx_batch]
        target_bboxes[:, [1, 3]] *= input_img_w
        target_bboxes[:, [2, 4]] *= input_img_h

        for target_bbox in target_bboxes:
        
            target_bbox = target_bbox.clone().view(1, 5) # to compute iou with vectorization
            
            # 1. anchor box matching
            iou = compute_iou(anchor_boxes, target_bbox[:, 3:], bbox_format="wh")
            
            inds_sorted_iou = torch.argsort(iou, descending=True)
            idx_mathced_anchor_box = inds_sorted_iou[0]

            matched_target_enocded_bboxes = target_encoded_bboxes[idx_mathced_anchor_box]

            # 2. make ground truth
            anchor_w, anchor_h = anchor_boxes[idx_mathced_anchor_box]
            grid_h, grid_w = matched_target_enocded_bboxes.shape[1:]

            target_x, target_y, target_w, target_h = target_bbox[0, 1:]
            
            grid_x = (target_x / input_img_w) * grid_w
            grid_y = (target_y / input_img_h) * grid_h
            
            #tl: top left
            grid_tl_x = int(torch.floor(grid_x))
            grid_tl_y = int(torch.floor(grid_y))

            offset_x = grid_x - grid_tl_x
            offset_y = grid_y - grid_tl_y

            assert offset_x >= 0. and offset_x <= 1.
            assert offset_y >= 0. and offset_y <= 1.

            assert target_w >= 0.
            assert target_h >= 0.

            assert anchor_w >= 0.
            assert anchor_h >= 0.

            power_w = torch.log(target_w/anchor_w)
            power_h = torch.log(target_h/anchor_h)

            c = int(target_bbox[0, 0])

            if matched_target_enocded_bboxes[4, grid_tl_y, grid_tl_x] == 1.:#already marked, pass
                continue

            num_target_bboxes += 1

            matched_target_enocded_bboxes[0, grid_tl_y, grid_tl_x] = offset_x
            matched_target_enocded_bboxes[1, grid_tl_y, grid_tl_x] = offset_y
            matched_target_enocded_bboxes[2, grid_tl_y, grid_tl_x] = power_w
            matched_target_enocded_bboxes[3, grid_tl_y, grid_tl_x] = power_h
            matched_target_enocded_bboxes[4, grid_tl_y, grid_tl_x] = 1
            matched_target_enocded_bboxes[5 + c, grid_tl_y, grid_tl_x] = 1
            
        # flatten features
        flatten_pred_encoded_bboxes = pred["encoded_bboxes"][idx_batch]
        flatten_pred_decoded_bboxes = pred["decoded_bboxes"][idx_batch]

        flatten_target_encoded_bboxes = []

        for idx_anchor_boxes in range(len(anchor_boxes)):
            flatten_target_encoded_bboxes.append(target_encoded_bboxes[idx_anchor_boxes].flatten(start_dim=1).T)

        flatten_target_encoded_bboxes = torch.cat(flatten_target_encoded_bboxes, dim=0)

        assert flatten_pred_encoded_bboxes.shape == flatten_target_encoded_bboxes.shape

        foreground_mask = flatten_target_encoded_bboxes[:, 4] == 1
        background_mask = flatten_target_encoded_bboxes[:, 4] == 0

        #localization
        loss_x.append(bce_with_logits_loss(flatten_pred_encoded_bboxes[foreground_mask][:, 0], flatten_target_encoded_bboxes[foreground_mask][:, 0]))
        loss_y.append(bce_with_logits_loss(flatten_pred_encoded_bboxes[foreground_mask][:, 1], flatten_target_encoded_bboxes[foreground_mask][:, 1]))
        loss_w.append(mse_loss(flatten_pred_encoded_bboxes[foreground_mask][:, 2], flatten_target_encoded_bboxes[foreground_mask][:, 2]))
        loss_h.append(mse_loss(flatten_pred_encoded_bboxes[foreground_mask][:, 3], flatten_target_encoded_bboxes[foreground_mask][:, 3]))

        #classification(foreground or background)
        loss_foreground_objectness.append(bce_with_logits_loss(flatten_pred_encoded_bboxes[foreground_mask][:, 4], flatten_target_encoded_bboxes[foreground_mask][:, 4]))
        
        # print("bf ignore: ", torch.sum(background_mask))
        for target_bbox in target_bboxes.to(device):
            target_bbox = target_bbox.view(1, 5)
            target_class = target_bbox[0, 0]

            iou = compute_iou(flatten_pred_decoded_bboxes[:, :4] ,target_bbox[:, 1:], bbox_format="cxcywh")
            background_mask[(iou > ignore_thresh) & (flatten_pred_decoded_bboxes[:, 5] == target_class)] = 0.# ignore


        loss_background_objectness.append(bce_with_logits_loss(flatten_pred_encoded_bboxes[background_mask][:, 4], flatten_target_encoded_bboxes[background_mask][:, 4]))
        loss_class_prob.append(bce_with_logits_loss(flatten_pred_encoded_bboxes[foreground_mask][:, 5:], flatten_target_encoded_bboxes[foreground_mask][:, 5:]))
        
    loss_x = torch.sum(torch.cat(loss_x))/num_target_bboxes
    loss_y = torch.sum(torch.cat(loss_y))/num_target_bboxes
    loss_w = torch.sum(torch.cat(loss_w))/num_target_bboxes
    loss_h = torch.sum(torch.cat(loss_h))/num_target_bboxes
    loss_foreground_objectness = torch.sum(torch.cat(loss_foreground_objectness))/num_target_bboxes
    loss_background_objectness = torch.sum(torch.cat(loss_background_objectness))/len(anchor_boxes)/batch_size
    loss_class_prob = torch.sum(torch.cat(loss_class_prob))/num_target_bboxes

    # anchor_boxes 수에 따라 background 샘플 수가 많아지고 이에따라 class imbalance 문제 심해짐, 이거를 anchor_boxes수로 나누어서 anchor boxes 수에 어느정도 independent하게끔 설계
    # batch_size 도 위와 같은 이유로 나누어줌
    # 입력 이미지의 Scale 또한 background 샘플 수에 영향을 미치는 factor인데... 이거는 어떻게 고려해주는 게 좋을까...흠... 기준이 될만한 factor가 없네
    # 뭐랄까... 설명하기 어려움, positive sample들을 target_bboxes수로 나누어주면 batch_size, 이미지내 positive sample 수에 어느정도 independent해지듯이...
    # negative sample들도 그럼 negative sample 수로 나눠주면 되지 않냐고 생각하게 되는데 실제로 mean 해서 backpropagation 시키면 FP(배경을 물체로 detection) 케이스 수가 기하급수적 증가됨
    # 학습의 성패는 loss_background_objectness 를 얼마나 잘 컨트롤 해주냐에 따라 갈림

    loss = loss_x + loss_y + loss_w + loss_h + loss_foreground_objectness + loss_background_objectness + loss_class_prob

    return loss

if __name__ == '__main__':

    # import cv2

    # bbox1 = torch.tensor([170.2334 , 154.0884 ,  90.60755, 112.94735]).reshape(1, 4)
    # # bbox2 = torch.tensor([201.4782 , 169.17638,  90.23747, 100.1186 ]).reshape(1, 4)
    # bbox2 = torch.tensor([201.4782 , 154.0884,  90.60755, 112.94735 ]).reshape(1, 4)
    # bboxes = [bbox1, bbox2]
    
    # img_draw = cv2.imread("test_example/000017.jpg")

    # import augmentation
    # img_draw = augmentation.letter_box_resize(img_draw, dsize=(416, 416))
    

    # for idx in range(len(bboxes)):
    #     bbox = bboxes[idx].view(-1)
    #     # bbox[[0, 2]] *= img_w
    #     # bbox[[1, 3]] *= img_h
    #     bbox = bbox.numpy().astype(np.int32)

    #     l = int(bbox[0] - bbox[2] / 2)
    #     r = int(bbox[0] + bbox[2] / 2)

    #     t = int(bbox[1] - bbox[3] / 2)
    #     b = int(bbox[1] + bbox[3] / 2)

    #     # print(c)
    #     cv2.rectangle(img=img_draw, pt1=(l, t), pt2=(r, b), color=(0, 255, 0))
    # print(compute_iou(bbox1, bbox2, bbox_format='cxcywh'))
    # cv2.imshow('img', img_draw)
    # cv2.waitKey(0)

    # exit()

    torch.set_printoptions(precision=3, sci_mode=False)

    model = YOLOv3Tiny(backbone_weight_path="backbone_weights/darknet_light_90_58.99.pth").cuda()

    import cv2
    import torchvision.transforms as transforms
    import tools

    import dataset
    import torch.utils.data as data
    import augmentation

    train_dataset = dataset.YOLODataset(path="test_example_v2", use_augmentation=True)
    data_loader = data.DataLoader(train_dataset, 32,
                                  num_workers=8,
                                  shuffle=True,
                                  collate_fn=dataset.yolo_collate,
                                  pin_memory=True,
                                  drop_last=False)

    epochs = 300
    lr = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(epochs):
        print("epoch: ", epoch)
        model.train()
        
        
        for img, target, inds in data_loader:
            img = img.cuda()
            i = 0
            while True:
                optimizer.zero_grad()

                pred = model(img)

                loss = yololoss(pred, target)
                loss.backward()
                
                optimizer.step()
                break
                # i += 1
                # if i > 10:
                #    break

        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr*0.5*(1. + np.cos(((epoch / epochs))*np.pi))


        model.eval()
        with torch.no_grad():
            img = cv2.imread("test_example/000017.jpg")
            #img = cv2.resize(img, (416, 416))
            img = augmentation.LetterBoxResize(img, (416, 416))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)
            img /= 255.
            img = torch.from_numpy(img).permute(2, 0, 1).contiguous()

            img = img.unsqueeze(0)

            batch_multi_scale_bboxes = model(img.cuda())

            filtered_batch_multi_scale_bboxes = bboxes_filtering(batch_multi_scale_bboxes)
            filtered_single_multi_scale_bboxes = filtered_batch_multi_scale_bboxes[0]
            print(filtered_single_multi_scale_bboxes)
            img_draw = cv2.imread("test_example/000017.jpg")
            import augmentation
            img_draw = augmentation.LetterBoxResize(img_draw, (416, 416))
            img_h, img_w = img_draw.shape[:2]

            for idx in range(len(filtered_single_multi_scale_bboxes['position'])):
                c = filtered_single_multi_scale_bboxes["class"][idx]
                bbox = filtered_single_multi_scale_bboxes['position'][idx]
                # bbox[[0, 2]] *= img_w
                # bbox[[1, 3]] *= img_h
                bbox = bbox.astype(np.int32)

                l = int(bbox[0] - bbox[2] / 2)
                r = int(bbox[0] + bbox[2] / 2)

                t = int(bbox[1] - bbox[3] / 2)
                b = int(bbox[1] + bbox[3] / 2)

                # print(c)
                cv2.rectangle(img=img_draw, pt1=(l, t), pt2=(r, b), color=(0, 255, 0))

            cv2.imshow('img', img_draw)
            cv2.waitKey(30)
            cv2.imwrite(str(epoch) + ".jpg", img_draw)
    
        import random
        random_scale_factor = random.randint(10, 19)

        train_dataset = dataset.YOLODataset(path="test_example_v2", use_augmentation=False, img_w=32*random_scale_factor, img_h=32*random_scale_factor)
        batch_size = 32 if random_scale_factor <= 13 else 32//2
        data_loader = data.DataLoader(train_dataset, batch_size,
                                    num_workers=8,
                                    shuffle=True,
                                    collate_fn=dataset.yolo_collate,
                                    pin_memory=True,
                                    drop_last=False)
