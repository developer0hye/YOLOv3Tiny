import torch
import numpy as np
import copy

class LRScheduler(object):
    def __init__(self,
                 optimizer,
                 warmup_iter,
                 total_iter,
                 target_lr):

        self.optimizer = optimizer
        self.warmup_iter = warmup_iter
        self.total_iter = total_iter
        self.target_lr = target_lr

    def warmup_lr(self, cur_iter):
        warmup_lr = self.target_lr * float(cur_iter) / float(self.warmup_iter)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr

    def cosine_decay_lr(self, cur_iter):

        Tcur = cur_iter - self.warmup_iter
        Tmax = self.total_iter - self.warmup_iter

        warmup_lr = 0.5 * self.target_lr * (1. + np.cos((Tcur / Tmax) * np.pi))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr

    def step(self, cur_iter):
        if cur_iter <= self.warmup_iter:
            self.warmup_lr(cur_iter)
        else:
            self.cosine_decay_lr(cur_iter)


def xywh2xyxy(box_xywh):
    box_xyxy = box_xywh.clone()
    box_xyxy[..., 0] = box_xywh[..., 0] - box_xywh[..., 2] / 2.
    box_xyxy[..., 1] = box_xywh[..., 1] - box_xywh[..., 3] / 2.
    box_xyxy[..., 2] = box_xywh[..., 0] + box_xywh[..., 2] / 2.
    box_xyxy[..., 3] = box_xywh[..., 1] + box_xywh[..., 3] / 2.
    return box_xyxy


def xyxy2xywh(box_xyxy):
    box_xywh = box_xyxy.clone()
    box_xywh[..., 0] = (box_xyxy[..., 0] + box_xyxy[..., 2]) / 2.
    box_xywh[..., 1] = (box_xyxy[..., 1] + box_xyxy[..., 3]) / 2.
    box_xywh[..., 2] = box_xyxy[..., 2] - box_xyxy[..., 0]
    box_xywh[..., 3] = box_xyxy[..., 3] - box_xyxy[..., 1]
    return box_xywh


def iou_xyxy(boxA_xyxy, boxB_xyxy):
    # determine the (x, y)-coordinates of the intersection rectangle
    x11, y11, x12, y12 = torch.split(boxA_xyxy, 1, dim=1)
    x21, y21, x22, y22 = torch.split(boxB_xyxy, 1, dim=1)

    xA = torch.max(x11, x21.T)
    yA = torch.max(y11, y21.T)
    xB = torch.min(x12, x22.T)
    yB = torch.min(y12, y22.T)

    interArea = (xB - xA).clamp(0) * (yB - yA).clamp(0)
    boxAArea = (x12 - x11) * (y12 - y11)
    boxBArea = (x22 - x21) * (y22 - y21)
    unionArea = (boxAArea + boxBArea.T - interArea)
    iou = interArea / (unionArea + 1e-6)

    # return the intersection over union value
    return iou


def iou_xywh(boxA_xywh, boxB_xywh):
    boxA_xyxy = xywh2xyxy(boxA_xywh)
    boxB_xyxy = xywh2xyxy(boxB_xywh)

    # return the intersection over union value
    return iou_xyxy(boxA_xyxy, boxB_xyxy)


def iou_np(bboxes1, bboxes2):
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou


def whiou(box1_wh, box2_wh):
    # determine the (x, y)-coordinates of the intersection rectangle
    eps = 1e-6

    w1, h1 = torch.split(box1_wh, 1, dim=1)
    w2, h2 = torch.split(box2_wh, 1, dim=1)

    innerW = torch.min(w1, w2.T).clamp(0)
    innerH = torch.min(h1, h2.T).clamp(0)

    interArea = innerW * innerH
    box1Area = w1 * h1
    box2Area = w2 * h2
    iou = interArea / (box1Area + box2Area.T - interArea + eps)

    # return the intersection over union value
    return iou

def build_target_tensor(model,
                        batch_pred_bboxes,
                        batch_target_bboxes,
                        input_size):
    
    #input_size (w, h)
    batch_pred_bboxes = batch_pred_bboxes.cpu()
    w, h = input_size
    o = (4 + 1 + model.num_classes)

    batch_size = len(batch_target_bboxes)

    for idx_batch in range(batch_size):
        

        break



    batch_target_tensor = []

    for _ in range(batch_size):
        single_target_tensor = []
        for idx, stride in enumerate(model.strides):
            for _ in range(len(model.anchors_mask[idx])):
                single_target_tensor.append(torch.zeros((h // stride, w // stride, o), dtype=torch.float32))
        batch_target_tensor.append(single_target_tensor)

    for idx_batch in range(batch_size):
        single_target_bboxes = []
        for single_target_bbox in batch_target_bboxes[idx_batch]:
            single_target_bboxes.append(single_target_bbox[1:])

            c = int(torch.round(single_target_bbox[0]))

            bbox_xy = single_target_bbox[1:3].clone().view(1, 2)
            bbox_wh = single_target_bbox[3:].clone().view(1, 2)

            bbox_wh[0, 0] *= w
            bbox_wh[0, 1] *= h

            iou = whiou(bbox_wh, model.anchors_wh)
            iou = iou.view(-1)

            sorted_iou_inds = torch.argsort(iou, descending=True)

            idx_yolo_layer = sorted_iou_inds[0]

            grid_h, grid_w = batch_target_tensor[idx_batch][idx_yolo_layer].shape[:2]

            grid_tx = bbox_xy[0, 0] * grid_w
            grid_ty = bbox_xy[0, 1] * grid_h

            idx_grid_tx = int(torch.floor(grid_tx))
            idx_grid_ty = int(torch.floor(grid_ty))

            if batch_target_tensor[idx_batch][idx_yolo_layer][idx_grid_ty, idx_grid_tx, 4] == 1.:
                continue

            tx = grid_tx - torch.floor(grid_tx)
            ty = grid_ty - torch.floor(grid_ty)

            tw = torch.log(bbox_wh[0, 0] / model.anchors_wh[idx_yolo_layer, 0])
            th = torch.log(bbox_wh[0, 1] / model.anchors_wh[idx_yolo_layer, 1])

            batch_target_tensor[idx_batch][idx_yolo_layer][idx_grid_ty, idx_grid_tx, [0, 1, 2, 3]] = torch.tensor([tx, ty, tw, th])
            batch_target_tensor[idx_batch][idx_yolo_layer][idx_grid_ty, idx_grid_tx, 4] = 1.0
            batch_target_tensor[idx_batch][idx_yolo_layer][idx_grid_ty, idx_grid_tx, 5 + c] = 1.0
            
            # for sorted_iou_idx in sorted_iou_inds[1:]:
            #     if iou[sorted_iou_idx] < 0.5:
            #         break
                
            #     idx_yolo_layer = sorted_iou_idx

            #     grid_h, grid_w = batch_target_tensor[idx_batch][idx_yolo_layer].shape[:2]

            #     grid_tx = bbox_xy[0, 0] * grid_w
            #     grid_ty = bbox_xy[0, 1] * grid_h

            #     idx_grid_tx = int(torch.floor(grid_tx))
            #     idx_grid_ty = int(torch.floor(grid_ty))

            #     if batch_target_tensor[idx_batch][idx_yolo_layer][idx_grid_ty, idx_grid_tx, 4] == 1.:
            #         continue

            #     tx = grid_tx - torch.floor(grid_tx)
            #     ty = grid_ty - torch.floor(grid_ty)

            #     tw = torch.log(bbox_wh[0, 0] / model.anchors_wh[idx_yolo_layer, 0])
            #     th = torch.log(bbox_wh[0, 1] / model.anchors_wh[idx_yolo_layer, 1])

            #     batch_target_tensor[idx_batch][idx_yolo_layer][idx_grid_ty, idx_grid_tx, [0, 1, 2, 3]] = torch.tensor([tx, ty, tw, th])
            #     batch_target_tensor[idx_batch][idx_yolo_layer][idx_grid_ty, idx_grid_tx, 4] = 1.0
            #     batch_target_tensor[idx_batch][idx_yolo_layer][idx_grid_ty, idx_grid_tx, 5 + c] = 1.0
            

        single_target_bboxes = torch.stack(single_target_bboxes)
        iou = iou_xywh(batch_pred_bboxes[idx_batch, :, :4], single_target_bboxes)
        iou, _ = torch.max(iou, dim=-1)

        single_target_tensor = []
        for idx_yolo_layer in range(len(batch_target_tensor[idx_batch])):
            single_target_tensor.append(batch_target_tensor[idx_batch][idx_yolo_layer].view(-1, o))
        single_target_tensor = torch.cat(single_target_tensor)
        single_target_tensor = single_target_tensor.unsqueeze(0)

        single_target_tensor[..., 4] = iou
        batch_target_tensor[idx_batch] = single_target_tensor

    batch_target_tensor = torch.cat(batch_target_tensor, dim=0)

    return batch_target_tensor
