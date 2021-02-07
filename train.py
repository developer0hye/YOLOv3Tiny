import os
import time
import random
import argparse

import torch
import torch.optim as optim
import torch.utils.data as torchdata
import torch.nn.functional as F

from torch.cuda.amp import *
#from torch.utils.tensorboard import SummaryWriter
import numpy as np

#custom packages
from model import *
import dataset


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

def warmup(model, warmup_epoch, device, opt):
    scaler = GradScaler()

    optimizer = optim.SGD(model.parameters(),
                        lr=0.,
                        momentum=opt.momentum,
                        weight_decay=0.)

    training_set = dataset.YOLODataset(path=opt.dataset_root,
                                    img_w=opt.img_w,
                                    img_h=opt.img_h,
                                    use_augmentation=True)

    training_set_loader = torchdata.DataLoader(training_set, opt.batch_size,
                                  num_workers=opt.num_workers,
                                  shuffle=True,
                                  collate_fn=dataset.yolo_collate,
                                  pin_memory=True,
                                  drop_last=True)

    torch.cuda.synchronize()
    t1 = time.time()

    warmup_iteration = (len(training_set) // opt.batch_size) * warmup_epoch
    iteration = 0
    for epoch in range(warmup_epoch):
        for i, batch_data in enumerate(training_set_loader):
            batch_img = batch_data["img"]
            batch_target = batch_data["bboxes"]
            batch_valid = batch_data["valid"]

            iteration += 1
            warmup_lr = opt.lr * float(iteration) / warmup_iteration
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                batch_img = batch_img.to(device)
                pred = model(batch_img)
                loss = yololoss(pred, batch_target, batch_valid)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if i % 100 == 0:
                torch.cuda.synchronize()
                t2 = time.time()

                print('loss: ', loss)
                print(f'Epoch[{epoch + 1} / {warmup_epoch}]' + ' || iter[%d / %d] ' % (
                i, total_iteration) + \
                    ' || Loss: %.4f ||' % (loss.item()) + ' || lr: %.8f ||' % (
                    optimizer.param_groups[0]['lr']) , end=' ')
                print("time per 100 iter(sec): ", t2 - t1)
                print("batch_img_size: ", batch_img.shape)
                t1 = time.time()

   
def train(model, optimizer, scaler, data_loader, device, epoch, total_iteration, opt):
    model.train()

    torch.cuda.synchronize()
    t1 = time.time()

    img_w = opt.img_w
    img_h = opt.img_h

    for i, batch_data in enumerate(data_loader):
        batch_img = batch_data["img"]
        batch_target = batch_data["bboxes"]
        batch_valid = batch_data["valid"]

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            batch_img = batch_img.to(device)
            batch_img = F.interpolate(batch_img, 
                                size=(img_h, img_w),
                                mode='bilinear',
                                align_corners=True)
            pred = model(batch_img)
            loss = yololoss(pred, batch_target, batch_valid)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if i % 10 == 0:
            torch.cuda.synchronize()
            t2 = time.time()
            
            print('loss: ', loss)
            print('Epoch[%d / %d]' % (epoch + 1, opt.total_epoch) + ' || iter[%d / %d] ' % (
            i, total_iteration) + \
                ' || Loss: %.4f ||' % (loss.item()) + ' || lr: %.8f ||' % (
                optimizer.param_groups[0]['lr']) , end=' ')
            print("time per 100 iter(sec): ", t2 - t1)
            print("batch_img_size: ", batch_img.shape)
            t1 = time.time()
        
        if i % 10 == 0: #for multiscale
            random_scale_factor = random.randint(opt.min_random_scale_factor, opt.max_random_scale_factor)
            img_w = opt.img_w + 32 * random_scale_factor
            img_h = opt.img_h + 32 * random_scale_factor

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='YOLO-v3 tiny Detection')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='Batch size for training')

    parser.add_argument('--img-w', default=416, type=int)
    parser.add_argument('--img-h', default=416, type=int)

    parser.add_argument('--min-random-scale-factor', default=-3, type=int)
    parser.add_argument('--max-random-scale-factor', default=6, type=int)

    parser.add_argument('--model-json-file', default='yolov3tiny_voc.json', type=str)
    parser.add_argument('--num-classes', default=20, type=int)
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--weights', type=str, default="", help='load weights to resume training')
    parser.add_argument('--total-epoch', type=int, default=300,
                        help='total_epoch')
    parser.add_argument('--dataset-root', default="VOCdataset/train",
                        help='Location of dataset directory')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight-decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--num-workers', default=8, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--gpu-ind', default=0, type=int,
                        help='To choose your gpu.')
    parser.add_argument('--save-folder', default='./weights', type=str,
                        help='where you save weights')
    parser.add_argument('--backbone-weights', default='backbone_weights/tiny_best_top1acc59.38.pth', type=str,
                        help='where is the backbone weights?')
    parser.add_argument('--seed', default=21, type=int)

    opt = parser.parse_args()
    setup_seed(opt.seed)
    
    if not os.path.exists(opt.save_folder):
        os.mkdir(opt.save_folder)
    
    training_set = dataset.YOLODataset(path=opt.dataset_root,
                                    img_w=opt.img_w + 32 * opt.max_random_scale_factor,
                                    img_h=opt.img_h + 32 * opt.max_random_scale_factor,
                                    use_augmentation=True)

    num_training_set_images = len(training_set)
    print("#Training set images: ", num_training_set_images)
    assert num_training_set_images > 0, "cannot load dataset, check root dir"

    training_set_loader = torchdata.DataLoader(training_set, opt.batch_size,
                                  num_workers=opt.num_workers,
                                  shuffle=True,
                                  collate_fn=dataset.yolo_collate,
                                  pin_memory=True,
                                  drop_last=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = YOLOv3Tiny(model_json_file=opt.model_json_file,
                       backbone_weight_path=opt.backbone_weights)            
    model = model.to(device)
    
    scaler = GradScaler()
    
    optimizer = optim.SGD(model.parameters(),
                        lr=opt.lr,
                        momentum=opt.momentum,
                        weight_decay=opt.weight_decay)
    
    iterations_per_epoch = num_training_set_images // opt.batch_size 
    total_iteration = iterations_per_epoch * opt.total_epoch

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.total_epoch)

    start_epoch = 0

    if len(opt.weights) == 0:
        warmup(model, warmup_epoch=5, device=device, opt=opt)
    else:
        checkpoint = torch.load(opt.weights)

        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        #load mAP
    
    for epoch in range(start_epoch, opt.total_epoch):

        train(model=model,
              optimizer=optimizer,
              scaler=scaler,
              data_loader=training_set_loader,
              device=device,
              epoch=epoch,
              total_iteration=total_iteration,
              opt=opt)
        
        scheduler.step()
        
        #mAP = validation()
        #best_mAP = max(best_mAP, mAP)
        
        checkpoint = {
        'epoch': epoch,# zero indexing
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'scaler_state_dict' : scaler.state_dict(),
        'scheduler_state_dict' : scheduler.state_dict(),
        'mAP': 0.,
        'best_mAP': 0.
        }
        
        torch.save(checkpoint, os.path.join(opt.save_folder, 'epoch' + str(epoch + 1) + '.pth'))
