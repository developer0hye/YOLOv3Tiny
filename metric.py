import numpy as np
np.set_printoptions(suppress=True)

def compute_iou(bboxes1, bboxes2, bbox_format='cxcywh'):
    if bbox_format == 'cxcywh':
        # if bboxes are composed of center point xy and wh/2.
        cx1, cy1, w1, h1 = np.split(bboxes1, 4, axis=1)
        cx2, cy2, w2, h2 = np.split(bboxes2, 4, axis=1)

        x11 = cx1 - w1 / 2
        y11 = cy1 - h1 / 2
        x12 = cx1 + w1 / 2 
        y12 = cy1 + h1 / 2

        x21 = cx2 - w2 /2
        y21 = cy2 - h2 /2
        x22 = cx2 + w2 /2
        y22 = cy2 + h2 /2
    elif bbox_format == 'tlxtlywh':
        # if bboxes are composed of top left point xy and wh.
        tlx1, tly1, w1, h1 = np.split(bboxes1, 4, axis=1)
        tlx2, tly2, w2, h2 = np.split(bboxes2, 4, axis=1)
        
        x11 = tlx1
        y11 = tly1
        x12 = tlx1 + w1 
        y12 = tly1 + h1

        x21 = tlx2
        y21 = tly2
        x22 = tlx2 + w2
        y22 = tly2 + h2
    else:
        # if bboxes are composed of top left point xy and bottom right xy.
        x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
        x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)

    # determine the (x, y)-coordinates of the intersection rectangle
    inter_x1 = np.maximum(x11, x21.T)
    inter_y1 = np.maximum(y11, y21.T)
    inter_x2 = np.minimum(x12, x22.T)
    inter_y2 = np.minimum(y12, y22.T)

    # # compute the area of intersection rectangle
    inter_area = np.maximum(inter_x2 - inter_x1 + 1 , 1e-6) * np.maximum(inter_y2 - inter_y1 + 1 , 1e-6)

    # # compute the area of both the prediction and ground-truth rectangles
    bboxes1_area = (x12 - x11 + 1 ) * (y12 - y11 + 1)
    bboxes2_area = (x22 - x21 + 1 ) * (y22 - y21 + 1)

    # compute the area of intersection rectangle
    # inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)

    # # compute the area of both the prediction and ground-truth rectangles
    # bboxes1_area = (x12 - x11) * (y12 - y11)
    # bboxes2_area = (x22 - x21) * (y22 - y21)
    
    iou = inter_area / ((bboxes1_area + bboxes2_area.T - inter_area) + 1e-5)
    assert np.min(iou) >= 0
    return iou

def measure_tpfp(pred_bboxes, gt_bboxes, iou_threshold=0.5, bbox_format='cxcywh'):
    '''
    https://github.com/rafaelpadilla/Object-Detection-Metrics
    위 프로젝트의 https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/pascalvoc.py
    코드 및 voc 평가방법 참고
    
    pred_bboxes: shape:[num_pred_bboxes, 6], class, x, y, w, h, confidence score 
    gt_bboxes: shape:[num_gt_bboxes, 5], class, x, y, w, h
    iou_threshold: 예측된 바운딩 박스를 tp/fp로 분류하기위한 iou thereshold
    
    return class_tp_fp_score
    class_tp_fp_score: shape:[num_pred_bboxes, 4], class, tp, fp, confidence score
    예측된 바운딩 박스별로 class tp/fp, confidence score 값을 기록한 table을 return 함
    '''
    
    pred_bboxes = pred_bboxes[np.argsort(pred_bboxes[:, 5])[::-1]]
#   print(pred_bboxes)
#   exit()
  
    iou_per_box = np.zeros((len(pred_bboxes), 1))
    tp_per_box = np.zeros((len(pred_bboxes), 1))

    for c in np.unique(gt_bboxes[:, 0]):

        gt_mask = gt_bboxes[:, 0] == c
        pred_mask = pred_bboxes[:, 0] == c

        filtered_gt_bboxes = gt_bboxes[gt_mask] # (M, )
        filtered_pred_bboxes = pred_bboxes[pred_mask] # (N, )

        if len(filtered_gt_bboxes) == 0 or len(filtered_pred_bboxes) == 0:
            continue

        filtered_iou_per_box = iou_per_box[pred_mask]
        filtered_tp_per_box = tp_per_box[pred_mask]
        
        iou_matrix = compute_iou(filtered_pred_bboxes[:, 1:5], filtered_gt_bboxes[:, 1:5], bbox_format) # (N, M)
        #for iou in iou_matrix:
        #print("np.sum(iou_matrix): ", np.sum(iou_matrix))

        # TP 여부를 판단할때, 어떤 한 Predicted Box가 두 Ground Truth Box에 대한 검출 결과로 중복적으로 매칭될 수 있다.
        # 이 경우 다른 Predicted Box가 두 Ground Truth Box중 하나의 Box에 대한 검출 결과로 매칭되고 TP가 될 가능성이 생긴다. 
        # 다만 VOC 는 이런 경우를 신경쓰지않음...

        for i in range(len(filtered_gt_bboxes)):
            iou = iou_matrix[:, i] # filtered_pred_bboxes와 i번째 filtered_gt_bbox간 IoU
            matched = False
            for j in range(len(iou)):
                if filtered_tp_per_box[j] == 1.: # already matched
                    continue
                
                if iou[j] >= iou_threshold:
                    if not matched:
                        matched = True
                        filtered_iou_per_box[j] = iou[j]
                        filtered_tp_per_box[j] = 1.
        
        iou_per_box[pred_mask] = filtered_iou_per_box
        tp_per_box[pred_mask] = filtered_tp_per_box

    fp_per_box = 1 - tp_per_box
    
    # print(pred_bboxes.shape)
    # print(np.sum(tp_per_box) + np.sum(fp_per_box))
    
    
    assert np.sum(tp_per_box) <= len(gt_bboxes), "Your code is wrong. The number of TP cases cannot exceed the number of ground truth boxes."
    assert np.sum(tp_per_box) + np.sum(fp_per_box) <= len(pred_bboxes), "Your code is wrong"

    # print(gt_bboxes)
    # print(fp_per_box)
    # print("--------------------")
    # exit()
    class_tp_fp_score = np.concatenate((pred_bboxes[:, 0, None], 
                                        tp_per_box, 
                                        fp_per_box, 
                                        pred_bboxes[:, 5, None]), 
                                    axis=1)#(N, 3)

    sorted_inds_by_score = np.argsort(pred_bboxes[:, 5])[::-1]#내림차순  

    class_tp_fp_score = class_tp_fp_score[sorted_inds_by_score]
    return class_tp_fp_score
def compute_ap(tp_fp_score, gt_bboxes):
    tp_fp_score = tp_fp_score[np.argsort(tp_fp_score[:, -1])[::-1]]
    num_all_gt_bboxes = len(gt_bboxes)

    accumulated_tp = np.cumsum(tp_fp_score[:, 0]).reshape(-1, 1)
    accumulated_fp = np.cumsum(tp_fp_score[:, 1]).reshape(-1, 1)

    precision = accumulated_tp/(accumulated_tp + accumulated_fp)
    recall = accumulated_tp/num_all_gt_bboxes
    
    # print(precision)
    # import matplotlib.pyplot as plt
    # plt.plot(recall, precision)
    # plt.show()
    
    #return CalculateAveragePrecision(recall, precision)[:3]
    interpolated_precision = np.zeros_like(precision)
    for i in range(len(precision)):
        interpolated_precision[i] = np.max(precision[i:])

    recall = list(recall)
    interpolated_precision = list(interpolated_precision)

    recall.insert(0, 0)
    interpolated_precision.insert(0, interpolated_precision[0])

    ap = 0.
    for i in range(1, len(recall)):
        ap = ap + interpolated_precision[i] * (recall[i] - recall[i - 1]) 

    return ap, precision, recall

def compute_map(class_tp_fp_score, gt_bboxes, num_classes):
  
  class_tp_fp_score = np.concatenate(class_tp_fp_score, axis=0)
  class_tp_fp_score = class_tp_fp_score[np.argsort(class_tp_fp_score[:, 3])[::-1]]
  gt_bboxes = np.concatenate(gt_bboxes, axis=0)

  ap_per_class = np.zeros(num_classes)

  for c in range(num_classes):
      
    if np.count_nonzero(class_tp_fp_score[:, 0] == c) == 0:
      continue
    
    if np.count_nonzero(gt_bboxes[:, 0] == c) == 0:
      continue

    ap, _, _ = compute_ap(class_tp_fp_score[class_tp_fp_score[:, 0] == c, 1:], gt_bboxes[gt_bboxes[:, 0] == c, 1:])
    ap_per_class[c] = ap

  print(ap_per_class)
  mean_ap = np.mean(ap_per_class)
  return mean_ap

if __name__ == '__main__':
    num_classes = 1
    
    pred_bboxes_batch = []
    gt_bboxes_batch =[]

    #xywh
    pred_bboxes1 = np.array([
[ 0 , 5.0 , 67.0 , 31.0 , 48.0 , 0.88 ],
[ 0 , 119.0 , 111.0 , 40.0 , 67.0 , 0.7 ],
[ 0 , 124.0 , 9.0 , 49.0 , 67.0 , 0.8 ]
])
    pred_bboxes2 = np.array([
[ 0 , 64.0 , 111.0 , 64.0 , 58.0 , 0.71 ],
[ 0 , 26.0 , 140.0 , 60.0 , 47.0 , 0.54 ],
[ 0 , 19.0 , 18.0 , 43.0 , 35.0 , 0.74 ]
])
                            
    pred_bboxes3 = np.array([
[ 0 , 109.0 , 15.0 , 77.0 , 39.0 , 0.18 ],
[ 0 , 86.0 , 63.0 , 46.0 , 45.0 , 0.67 ],
[ 0 , 160.0 , 62.0 , 36.0 , 53.0 , 0.38 ],
[ 0 , 105.0 , 131.0 , 47.0 , 47.0 , 0.91 ],
[ 0 , 18.0 , 148.0 , 40.0 , 44.0 , 0.44 ]
])
    
    pred_bboxes4 = np.array([
[ 0 , 83.0 , 28.0 , 28.0 , 26.0 , 0.35 ],
[ 0 , 28.0 , 68.0 , 42.0 , 67.0 , 0.78 ],
[ 0 , 87.0 , 89.0 , 25.0 , 39.0 , 0.45 ],#망한 박스
[ 0 , 10.0 , 155.0 , 60.0 , 26.0 , 0.14 ]
])
    
    pred_bboxes5 = np.array([
[ 0 , 50.0 , 38.0 , 28.0 , 46.0 , 0.62 ],
[ 0 , 95.0 , 11.0 , 53.0 , 28.0 , 0.44 ],
[ 0 , 29.0 , 131.0 , 72.0 , 29.0 , 0.95 ],
[ 0 , 29.0 , 163.0 , 72.0 , 29.0 , 0.23 ]
])
    
    pred_bboxes6 = np.array([
[ 0 , 43.0 , 48.0 , 74.0 , 38.0 , 0.45 ],
[ 0 , 17.0 , 155.0 , 29.0 , 35.0 , 0.84 ],
[ 0 , 95.0 , 110.0 , 25.0 , 42.0 , 0.43 ]
])
    
    pred_bboxes7 = np.array([
[ 0 , 16.0 , 20.0 , 101.0 , 88.0 , 0.48 ],
[ 0 , 33.0 , 116.0 , 37.0 , 49.0 , 0.95 ]
])
    
    gt_bboxes1 = np.array([
[ 0 , 25.0 , 16.0 , 38.0 , 56.0 ],
[ 0 , 129.0 , 123.0 , 41.0 , 62.0 ]
])
    gt_bboxes2 = np.array([
[ 0 , 123.0 , 11.0 , 43.0 , 55.0 ],
[ 0 , 38.0 , 132.0 , 59.0 , 45.0 ]
])
    gt_bboxes3 = np.array([
[ 0 , 16.0 , 14.0 , 35.0 , 48.0 ],
[ 0 , 123.0 , 30.0 , 49.0 , 44.0 ],
[ 0 , 99.0 , 139.0 , 47.0 , 47.0 ]
])
    gt_bboxes4 = np.array([
[ 0 , 53.0 , 42.0 , 40.0 , 52.0 ],
[ 0 , 154.0 , 43.0 , 31.0 , 34.0 ]
])
    gt_bboxes5 = np.array([
[ 0 , 59.0 , 31.0 , 44.0 , 51.0 ],
[ 0 , 48.0 , 128.0 , 34.0 , 52.0 ]
])
    gt_bboxes6 = np.array([
[ 0 , 36.0 , 89.0 , 52.0 , 76.0 ],
[ 0 , 62.0 , 58.0 , 44.0 , 67.0 ]
])
    gt_bboxes7 = np.array([
[ 0 , 28.0 , 31.0 , 55.0 , 63.0 ],
[ 0 , 58.0 , 67.0 , 50.0 , 58.0 ]
])
    
    pred_bboxes_batch.append(pred_bboxes1)
    pred_bboxes_batch.append(pred_bboxes2)
    pred_bboxes_batch.append(pred_bboxes3)
    pred_bboxes_batch.append(pred_bboxes4)
    pred_bboxes_batch.append(pred_bboxes5)
    pred_bboxes_batch.append(pred_bboxes6)
    pred_bboxes_batch.append(pred_bboxes7)
    
    gt_bboxes_batch.append(gt_bboxes1)
    gt_bboxes_batch.append(gt_bboxes2)
    gt_bboxes_batch.append(gt_bboxes3)
    gt_bboxes_batch.append(gt_bboxes4)
    gt_bboxes_batch.append(gt_bboxes5)
    gt_bboxes_batch.append(gt_bboxes6)
    gt_bboxes_batch.append(gt_bboxes7)
    
    class_tp_fp_score_batch = []
    for pred_bboxes_per_image, gt_bboxes_per_image in zip(pred_bboxes_batch, gt_bboxes_batch):
        class_tp_fp_score = measure_tpfp(pred_bboxes_per_image, gt_bboxes_per_image, 0.5, bbox_format='tlxtlywh')
        class_tp_fp_score_batch.append(class_tp_fp_score)
    
    class_tp_fp_score_batch = np.concatenate(class_tp_fp_score_batch, axis=0)
    class_tp_fp_score_batch = class_tp_fp_score_batch[np.argsort(class_tp_fp_score_batch[:, 3])[::-1]]
    gt_bboxes_batch = np.concatenate(gt_bboxes_batch, axis=0)
    
    print(class_tp_fp_score_batch)
    mean_ap = compute_map(class_tp_fp_score_batch, gt_bboxes_batch, num_classes)
    print(mean_ap)
