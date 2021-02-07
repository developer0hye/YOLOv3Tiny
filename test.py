import os
import torch
import numpy as np
import cv2
import argparse
from model import *
import metric
import augmentation


def test(opt, model, device):
    if not os.path.exists(opt.save_folder):
        os.mkdir(opt.save_folder)

    import dataset
    test_dataset = dataset.YOLODataset(path=opt.dataset_root,
                                    img_w=opt.img_w,
                                    img_h=opt.img_h,
                                    use_augmentation=False)
    
    data_loader = torch.utils.data.DataLoader(test_dataset,
                                              opt.batch_size,
                                              num_workers=opt.num_workers,
                                              shuffle=False,
                                              collate_fn=dataset.yolo_collate,
                                              pin_memory=True,
                                              drop_last=False)

    print("----------------------------------------Object Detection--------------------------------------------")
    print("Let's test OD network !")

    # loss counters
    print("----------------------------------------------------------")
    print('Loading the dataset...')
    print('Testing on:', opt.dataset_root)
    print('The dataset size:', len(test_dataset))
    print("----------------------------------------------------------")

    gt_bboxes_batch = []
    class_tp_fp_score_batch = []

    # start test
    model.eval()
    with torch.no_grad():
        for batch_data in data_loader:
            
            batch_img = batch_data["img"]
            batch_bboxes = batch_data["bboxes"]
            batch_idx = batch_data["idx"]
            batch_original_img_shape = batch_data["original_img_shape"]
            batch_non_padded_img_shape = batch_data["non_padded_img_shape"]
            batch_padded_lt = batch_data["padded_lt"]            
            
            batch_img = batch_img.to(device)

            # forward
            batch_output = model(batch_img.cuda())
            batch_filtered_decoded_bboxes = bboxes_filtering(batch_output, 
                                                            batch_padded_lt, 
                                                            batch_non_padded_img_shape, 
                                                            batch_original_img_shape, 
                                                            conf_thresh=1e-2,
                                                            iou_thresh=0.45)
            
            for pred_bboxes, target_bboxes, index, padded_lt, non_padded_img_shape, original_img_shape in zip(batch_filtered_decoded_bboxes, 
                                                                                                            batch_bboxes, 
                                                                                                            batch_idx, 
                                                                                                            batch_padded_lt, 
                                                                                                            batch_non_padded_img_shape,
                                                                                                            batch_original_img_shape):
                target_bboxes = target_bboxes.cpu().numpy()

                target_bboxes[:, [1, 3]] *= opt.img_w
                target_bboxes[:, [2, 4]] *= opt.img_h
                
                target_bboxes[:, 1] -= padded_lt[0]
                target_bboxes[:, 2] -= padded_lt[1]

                target_bboxes[:, [1, 3]] /= non_padded_img_shape[0]
                target_bboxes[:, [2, 4]] /= non_padded_img_shape[1]

                target_bboxes[:, [1, 3]] *= original_img_shape[0]
                target_bboxes[:, [2, 4]] *= original_img_shape[1]

                gt_bboxes_batch.append(target_bboxes)

                if pred_bboxes["num_detected_bboxes"] > 0:
                    pred_bboxes = np.concatenate([pred_bboxes["class"].reshape(-1, 1), 
                    pred_bboxes["position"].reshape(-1, 4),
                    pred_bboxes["confidence"].reshape(-1, 1)], axis=1)

                    class_tp_fp_score = metric.measure_tpfp(pred_bboxes, target_bboxes, 0.5, bbox_format='cxcywh')
                    class_tp_fp_score_batch.append(class_tp_fp_score)

                    # img = cv2.imread(test_dataset.imgs_path[index])
                    # #img = augmentation.LetterBoxResize(img,dsize=(opt.img_w, opt.img_h))

                    # for pred_bbox in pred_bboxes:

                    #     pred_bbox = pred_bbox.astype(np.int32)

                    #     l = int(pred_bbox[1] - pred_bbox[3] / 2)
                    #     r = int(pred_bbox[1] + pred_bbox[3] / 2)

                    #     t = int(pred_bbox[2] - pred_bbox[4] / 2)
                    #     b = int(pred_bbox[2] + pred_bbox[4] / 2)
                    #     print(pred_bbox)
                    #     cv2.rectangle(img=img, pt1=(l, t), pt2=(r, b), color=(0, 255, 0))

                    # cv2.imshow('img', img)
                    # cv2.waitKey(0)

        mean_ap = metric.compute_map(class_tp_fp_score_batch, gt_bboxes_batch, num_classes=model.num_classes)
        print(mean_ap)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO-v3 tiny Detection')

    parser.add_argument('--batch-size', default=32, type=int,
                        help='Batch size for testing')
    parser.add_argument('--img-w', default=416, type=int)
    parser.add_argument('--img-h', default=416, type=int)
    parser.add_argument('--model-json-file', default='yolov3tiny_voc.json', type=str)
    parser.add_argument('--weights', type=str, default=None,
                        help='load weights to resume training')
    parser.add_argument('--dataset-root', default="VOCdataset/test",
                        help='Location of dataset directory')
    parser.add_argument('--save-folder', default="predicted_results",
                        help='Location of output directory')
    parser.add_argument('--num-workers', default=8, type=int,
                        help='Number of workers used in dataloading')

    opt = parser.parse_args()


    torch.backends.cudnn.deterministic = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLOv3Tiny(model_json_file=opt.model_json_file)

    if opt.weights is not None:
        chkpt = torch.load(opt.weights, map_location=device)
        model.load_state_dict(chkpt['model_state_dict'], strict=False)
    
    model = model.to(device)    
    test(opt, model, device)