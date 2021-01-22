import cv2
import numpy as np
import random

def xywh2xyxy(box_xywh):
    box_xyxy = box_xywh.copy()
    box_xyxy[:, 0] = box_xywh[:, 0] - box_xywh[:, 2] / 2.
    box_xyxy[:, 1] = box_xywh[:, 1] - box_xywh[:, 3] / 2.
    box_xyxy[:, 2] = box_xywh[:, 0] + box_xywh[:, 2] / 2.
    box_xyxy[:, 3] = box_xywh[:, 1] + box_xywh[:, 3] / 2.
    box_xyxy = np.clip(box_xyxy, 0., 1.)
    return box_xyxy


def xyxy2xywh(box_xyxy):
    box_xywh = box_xyxy.copy()
    box_xywh[:, 0] = (box_xyxy[:, 0] + box_xyxy[:, 2]) / 2.
    box_xywh[:, 1] = (box_xyxy[:, 1] + box_xyxy[:, 3]) / 2.
    box_xywh[:, 2] = box_xyxy[:, 2] - box_xyxy[:, 0]
    box_xywh[:, 3] = box_xyxy[:, 3] - box_xyxy[:, 1]
    box_xywh = np.clip(box_xywh, 0., 1.)
    return box_xywh

def calcArea(bboxes_xyxy):
    w = (bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0])
    h = (bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1])
    return w * h

def LetterBoxResize(img, dsize, bboxes=None, class_ids=None):
    
    original_height, original_width = img.shape[:2]
    target_width, target_height = dsize

    ratio = min(
        float(target_width) / original_width,
        float(target_height) / original_height)
    resized_height, resized_width = [
        round(original_height * ratio),
        round(original_width * ratio)
    ]

    img = cv2.resize(img, dsize=(resized_width, resized_height))

    pad_left = (target_width - resized_width) // 2
    pad_right = target_width - resized_width - pad_left
    pad_top = (target_height - resized_height) // 2
    pad_bottom = target_height - resized_height - pad_top

    # padding
    img = cv2.copyMakeBorder(img,
                             pad_top,
                             pad_bottom,
                             pad_left,
                             pad_right,
                             cv2.BORDER_CONSTANT,
                             value=(127, 127, 127))

    try:
        if img.shape[0] != target_height and img.shape[1] != target_width:  # 둘 중 하나는 같아야 함
            raise Exception('Letter box resizing method has problem.')
    except Exception as e:
        print('Exception: ', e)
        exit(1)

    if class_ids is not None and bboxes is not None:
        # padding으로 인한 객체 translation 보상
        bboxes[:, [0, 2]] *= resized_width
        bboxes[:, [1, 3]] *= resized_height

        bboxes[:, 0] += pad_left
        bboxes[:, 1] += pad_top

        bboxes[:, [0, 2]] /= target_width
        bboxes[:, [1, 3]] /= target_height

        return img, bboxes, class_ids
    return img
    
def PhotometricNoise(
             prng,
             img_bgr, #type must be float
             h_delta=18.,
             s_gain=0.5,
             brightness_delta=32,
             ):
    if prng.randint(2):
        if prng.randint(2):
            brightness_delta = prng.uniform(-brightness_delta, brightness_delta)
            img_bgr += brightness_delta
            img_bgr = np.clip(img_bgr, 0., 255.)

        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # h[0, 359], s[0, 1.0], v[0, 255.]
        h_delta = prng.uniform(-h_delta, h_delta)
        s_gain = prng.uniform(1. - s_gain, 1. + s_gain)

        img_hsv[..., 0] = np.clip(img_hsv[..., 0] + h_delta, 0., 359.)
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] * s_gain, 0., 1.0)

        img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        return img_bgr
    return img_bgr

def ColorJittering(img, delta_h=15, scale_s=.5, scale_v=.5):

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(float)
    
    img_hsv[..., 0] += random.randint(-delta_h, delta_h)
    img_hsv[..., 0] = np.clip(img_hsv[..., 0], 0, 179)

    img_hsv[..., 1] *= random.uniform(1. - scale_s, 1. + scale_s)
    img_hsv[..., 1] = np.clip(img_hsv[..., 1], 0, 255)

    img_hsv[..., 2] *= random.uniform(1. - scale_v, 1. + scale_v)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2], 0, 255)

    img_hsv = img_hsv.astype(np.uint8)
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    return img

def HorFlip(img, bboxes_xywh, p=0.5):
    if random.random() < p:
        img = cv2.flip(img, 1)#1이 호리즌탈 방향 반전
        bboxes_xywh[:, 0] = 1. - bboxes_xywh[:, 0]
        return img, bboxes_xywh
    return img, bboxes_xywh

def RandomTranslation(img, bboxes_xyxy, classes, p=1.0):
    if random.random() < p:
        height, width = img.shape[0:2]

        l_bboxes = round(width * np.min(bboxes_xyxy[:, 0]))
        r_bboxes = width-round(width * np.max(bboxes_xyxy[:, 2]))

        t_bboxes = round(np.min(height * bboxes_xyxy[:, 1]))
        b_bboxes = height-round(height * np.max(bboxes_xyxy[:, 3]))

        tx = random.randint(-l_bboxes, r_bboxes)
        ty = random.randint(-t_bboxes, b_bboxes)

        # translation matrix
        tm = np.float32([[1, 0, tx],
                         [0, 1, ty]])  # [1, 0, tx], [1, 0, ty]

        img = cv2.warpAffine(img, tm, (width, height), borderValue=(127, 127, 127))

        bboxes_xyxy[:, [0, 2]] += (tx / width)
        bboxes_xyxy[:, [1, 3]] += (ty / height)
        bboxes_xyxy = np.clip(bboxes_xyxy, 0., 1.)

        return img, bboxes_xyxy, classes
    return img, bboxes_xyxy, classes

def RandomScale(img, bboxes_xyxy, classes, p=1.0):

    if random.random() < p:
        img_h, img_w = img.shape[:2]
        
        min_bbox_w = np.min(bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]) * img_w
        min_bbox_h = np.min(bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]) * img_h

        min_scale = 1.0
        if min_bbox_w > 32 and min_bbox_h > 32: 
            # 최소 크기가 32보다 크면 크기를 줄이지 않음, 더 줄이면 눈으로도 식별하기 힘들어짐
            min_scale = np.maximum(32/min_bbox_w, 32/min_bbox_h) #줄어들 수 있는 최소 크기를 32으로 한정

        max_bbox_w = np.max(bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]) * img_w
        max_bbox_h = np.max(bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]) * img_h

        max_scale = np.minimum(img_w/max_bbox_w, img_h/max_bbox_h)

        cx = img_w//2
        cy = img_h//2

        for _ in range(10):#maximum trial    
            random_scale = random.uniform(min_scale, max_scale)

            #센터 기준으로 확대 혹은 축소
            tx = cx - random_scale * cx
            ty = cy - random_scale * cy

            min_bbox_x = round(img_w * np.min(bboxes_xyxy[:, 0])) * random_scale + tx
            max_bbox_x = round(img_w * np.max(bboxes_xyxy[:, 2])) * random_scale + tx

            min_bbox_y = round(img_h * np.min(bboxes_xyxy[:, 1])) * random_scale + ty
            max_bbox_y = round(img_h * np.max(bboxes_xyxy[:, 3])) * random_scale + ty

            if min_bbox_x < 0 or max_bbox_x >= img_w:
                continue

            if min_bbox_y < 0 or max_bbox_y >= img_h:
                continue
 
            # # scale matrix
            sm = np.float32([[random_scale, 0, tx],
                            [0, random_scale, ty]])  # [1, 0, tx], [1, 0, ty]

            img = cv2.warpAffine(img, sm, (img_w, img_h), borderValue=(127, 127, 127))

            bboxes_xyxy *= random_scale
            bboxes_xyxy[:, [0, 2]] += (tx / img_w)
            bboxes_xyxy[:, [1, 3]] += (ty / img_h)
            bboxes_xyxy = np.clip(bboxes_xyxy, 0., 1.)

            return img, bboxes_xyxy, classes

    return img, bboxes_xyxy, classes

def RandomErasePatches(img, bboxes_xyxy, occlusion_ratio=0.25, p=1.0):
    if random.random() < p:
        img_h, img_w = img.shape[:2]
        for bbox_xyxy in bboxes_xyxy.copy():
            l, t, r, b = bbox_xyxy

            l = int(np.round(l * img_w))
            t = int(np.round(t * img_h))
            r = int(np.round(r * img_w))
            b = int(np.round(b * img_h))

            erased_patch_w = int(np.round((r - l) * occlusion_ratio))
            erased_patch_h = int(np.round((b - t) * occlusion_ratio))

            if erased_patch_w == 0 or erased_patch_h == 0:
                continue
            
            erased_patch_l = random.randint(l, r - erased_patch_w)
            erased_patch_t = random.randint(t, b - erased_patch_h)
            erased_patch_r = erased_patch_l + erased_patch_w
            erased_patch_b = erased_patch_t + erased_patch_h

            img[erased_patch_t:erased_patch_b, erased_patch_l:erased_patch_r] = 127
            
        return img
    return img

def RandomCrop(img, bboxes_xyxy, classes, p=1.0):
    if random.random() < p:
        height, width = img.shape[0:2]

        min_width = width // 4
        min_height = height // 4

        while (min_width <= width and min_height <= height):
            clipped_w = random.randint(min_width, width)
            clipped_h = random.randint(min_height, height)

            l = random.randint(0, width - clipped_w + 1)
            t = random.randint(0, height - clipped_h + 1)
            r = l + clipped_w
            b = t + clipped_h

            w_bboxes = np.round(width * (bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0])).astype(np.int32)
            h_bboxes = np.round(height * (bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1])).astype(np.int32)

            l_clipped_bboxes = np.round(np.clip(width * bboxes_xyxy[:, 0], a_min=l, a_max=r)).astype(np.int32)
            t_clipped_bboxes = np.round(np.clip(height * bboxes_xyxy[:, 1], a_min=t, a_max=b)).astype(np.int32)
            r_clipped_bboxes = np.round(np.clip(width * bboxes_xyxy[:, 2], a_min=l, a_max=r)).astype(np.int32)
            b_clipped_bboxes = np.round(np.clip(height * bboxes_xyxy[:, 3], a_min=t, a_max=b)).astype(np.int32)

            w_clipped_bboxes = r_clipped_bboxes - l_clipped_bboxes
            h_clipped_bboxes = b_clipped_bboxes - t_clipped_bboxes

            inner_bboxes = ((w_clipped_bboxes * h_clipped_bboxes) > 0)

            #0인건 일단 제거
            if not(True in inner_bboxes):
                min_width = clipped_w + 1
                min_height = clipped_h + 1
                continue

            w_bboxes = w_bboxes[inner_bboxes]
            h_bboxes = h_bboxes[inner_bboxes]

            w_clipped_bboxes = w_clipped_bboxes[inner_bboxes]
            h_clipped_bboxes = h_clipped_bboxes[inner_bboxes]

            if np.min(((w_clipped_bboxes * h_clipped_bboxes) / (w_bboxes * h_bboxes))) < 0.75:
                min_width = clipped_w + 1
                min_height = clipped_h + 1
                continue

            l_clipped_bboxes = ((l_clipped_bboxes[inner_bboxes, np.newaxis]-l)/clipped_w)
            t_clipped_bboxes = ((t_clipped_bboxes[inner_bboxes, np.newaxis]-t)/clipped_h)
            r_clipped_bboxes = ((r_clipped_bboxes[inner_bboxes, np.newaxis]-l)/clipped_w)
            b_clipped_bboxes = ((b_clipped_bboxes[inner_bboxes, np.newaxis]-t)/clipped_h)

            classes = classes[inner_bboxes]

            augmented_bboxes_xyxy = np.concatenate([l_clipped_bboxes, t_clipped_bboxes,
                                                    r_clipped_bboxes, b_clipped_bboxes], axis=-1).astype(np.float32)

            augmented_bboxes_xyxy = np.clip(augmented_bboxes_xyxy, a_min=0., a_max=1.0)
            augmented_img = img[t:b, l:r]
            return augmented_img, augmented_bboxes_xyxy, classes
    return img, bboxes_xyxy, classes

def RandomCropPreserveBBoxes(img, bboxes_xyxy, classes, p=1.0):
    if random.random() < p:
        height, width = img.shape[0:2]

        outer_l_bboxes = int(round(width * np.min(bboxes_xyxy[:, 0])))
        outer_r_bboxes = int(round(width * np.max(bboxes_xyxy[:, 2])))

        outer_t_bboxes = int(round(np.min(height * bboxes_xyxy[:, 1])))
        outer_b_bboxes = int(round(height * np.max(bboxes_xyxy[:, 3])))

        l = random.randint(0, outer_l_bboxes)
        t = random.randint(0, outer_t_bboxes)
        r = random.randint(outer_r_bboxes, width)
        b = random.randint(outer_b_bboxes, height)

        cropped_img = img[t:b, l:r]

        cropped_bboxes = bboxes_xyxy.copy()
        cropped_bboxes[:, [0, 2]] *= width
        cropped_bboxes[:, [1, 3]] *= height

        cropped_bboxes[:, [0, 2]] -= l
        cropped_bboxes[:, [1, 3]] -= t

        cropped_bboxes[:, [0, 2]] /= (r-l)
        cropped_bboxes[:, [1, 3]] /= (b-t)

        return cropped_img, cropped_bboxes, classes
    return img, bboxes_xyxy, classes

def drawBBox(img, bboxes_xyxy):
    h, w = img.shape[:2]

    bboxes_xyxy[:, [0, 2]] *= w
    bboxes_xyxy[:, [1, 3]] *= h

    for bbox_xyxy in bboxes_xyxy:
        print(bbox_xyxy)
        cv2.rectangle(img,
                      (int(bbox_xyxy[0]), int(bbox_xyxy[1])),
                      (int(bbox_xyxy[2]), int(bbox_xyxy[3])),
                      (0, 255, 0),2)

if __name__ == '__main__':
    from numpy.random import RandomState
    prng = RandomState(21)

    while(True):
        img = cv2.imread("test_example_v2/000021.jpg", cv2.IMREAD_COLOR)

        import dataset
        label = dataset.read_annotation_file("test_example_v2/000021.txt")
        classes, bboxes_xywh = label[:, 0:1], label[:, 1:]

        bboxes_xyxy = xywh2xyxy(bboxes_xywh)

        img, bboxes_xyxy, classes = RandomCropPreserveBBoxes(img, bboxes_xyxy, classes)
        
        bboxes_xywh = xyxy2xywh(bboxes_xyxy)
        #img = cv2.resize(img, (416, 416))
        img, bboxes_xywh, classes = LetterBoxResize(img, (608, 608), bboxes_xywh, classes)
        img, bboxes_xywh = HorFlip(img, bboxes_xywh)
        bboxes_xyxy = xywh2xyxy(bboxes_xywh)

        img, bboxes_xyxy, classes = RandomTranslation(img, bboxes_xyxy, classes)
        img, bboxes_xyxy, classes = RandomScale(img, bboxes_xyxy, classes)
        img = RandomErasePatches(img, bboxes_xyxy)
        img = ColorJittering(img)
        

        if len(bboxes_xyxy) != len(classes):
            print("bbox랑 class 수랑 일치하지 않다. augmentation 과정에서 실수가 있는 게 분명해")

        drawBBox(img, bboxes_xyxy)
        cv2.imshow("img", img)
        ch = cv2.waitKey(0)

        if ch == 27:
            break
