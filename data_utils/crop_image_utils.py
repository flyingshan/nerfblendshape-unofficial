from centerface import CenterFace
import cv2
import scipy.io as sio
import os
import numpy as np
import time
import random

def resize_image(frame, resize_w):
    """ 由于视频一般是自拍, h > w, 所以resize的时候将窄边w缩放到resize_w, 长边h等比例缩放到对应大小"""
    if resize_w > 0:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (resize_w, int(h/w*resize_w)), interpolation=cv2.INTER_CUBIC)
    return frame, frame.shape[:2]

def detect_face(frames_dir, resize_w=-1, avg_num=20):
    """ 给定图像目录，检测人脸，返回所有检测结果"""
    landmarks = True
    centerface = CenterFace(landmarks=landmarks)
    images = [os.path.join(frames_dir, i) for i in os.listdir(frames_dir) if i.endswith('jpg') or i.endswith('png')]
    random.shuffle(images)
    images = images[:avg_num]
    dets = []
    for image in images:
        frame = cv2.imread(image)
        frame, frame_hw = resize_image(frame, resize_w)
        h, w = frame_hw
        begin = time.time()
        det, _ = centerface(frame, h, w, threshold=0.35)
        # DEBUG:
        # boxes, score = det[0][:4], det[0][4]
        # cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
        # cv2.imwrite("test.jpg", frame)
        end = time.time()
        dets.append(det[0]) # 一张图片应该只有一个人脸
    return dets, frame_hw

# dets[0] + dets
def get_crop_coord(dets, H, W, hw_crop):
    """ 给定所有人脸的检测结果，返回平均位置的左上和右下角坐标"""
    dets = np.array(dets).mean(0).astype(np.int32).tolist()
    
    y_center = (dets[0] + dets[2]) / 2
    x_center = (dets[1] + dets[3]) / 2
    print(f"center: {x_center}, {y_center}")

    x_lt = max(0, int(x_center) - int(H / 2))
    y_lt = max(0, int(y_center) - int(W / 2))

    # 如果另一边越界，那么裁剪后短边设为和原始短边一样（避免最后剪出来大小小于预设的hw_crop） TODO: 还有更复杂的逻辑
    if x_lt + H - 1 > hw_crop[0] - 1:
        x_rb = hw_crop[0] - 1
        x_lt = 0
    else:
        x_rb = x_lt + H - 1
    if y_lt + W - 1 > hw_crop[1] - 1:
        y_rb = hw_crop[1] - 1
        y_lt = 0
    else:
        y_rb = y_lt + W - 1
    # x_rb = min( x_lt + H - 1 , hw_crop[0] - 1)
    # y_rb = min(y_lt + W - 1 , hw_crop[1] - 1)

    # print(x_rb - x_lt + 1, y_rb - y_lt + 1, H, W)
    # print(f"head bbox: {x_lt} {y_lt} {x_rb} {y_rb}")

    # 虽说加了边界条件，不过还是先不支持自适应尺寸，以免出现未知bug
    assert x_rb - x_lt + 1 == H and y_rb - y_lt + 1 == W

    print(f"head bbox: {x_lt} {y_lt} {x_rb} {y_rb}")

    return x_lt, y_lt, x_rb, y_rb

def crop_images(frames_dir, target_dir, bbox, pick_ratio=1, resize_w=-1):
    """ 从frames_dir的图片按照bbox裁剪，写入到target_dir, 每隔pick_ratio挑一张"""
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    images_name = [i for i in os.listdir(frames_dir) if i.endswith('jpg') or i.endswith('png')]
    images_dir  = [os.path.join(frames_dir, i) for i in images_name]
    images_dir = sorted(images_dir, key=lambda x: int(os.path.basename(x).split('.')[0]))
    x_lt, y_lt, x_rb, y_rb = bbox
    prefix = 0
    suffix = '.jpg'
    for i in range(0, len(images_dir), pick_ratio):
        image = images_dir[i]
        frame = cv2.imread(image)
        frame, _ = resize_image(frame, resize_w)
        frame = frame[x_lt:x_rb+1, y_lt:y_rb+1]
        cv2.imwrite(os.path.join(target_dir, str(prefix) + suffix), frame)
        prefix += 1

def detect_and_crop(frames_dir, target_dir, resize_W, target_H, target_W, pick_ratio):
    """
    将原始视频帧短边缩放到对应大小(长边等比例缩放)
    然后定位人脸，并以其中心切片对应大小图片
    最后按一定的跳过比例（帧率调整）取出图片，保存到对应位置
    """
    dets, hw_crop = detect_face(frames_dir, resize_w = resize_W)
    x_lt, y_lt, x_rb, y_rb = get_crop_coord(dets, H=target_H, W=target_W, hw_crop=hw_crop)
    bbox = [x_lt, y_lt, x_rb, y_rb]
    crop_images(frames_dir=frames_dir, target_dir=target_dir, bbox=bbox, pick_ratio=pick_ratio, resize_w=resize_W)

def main():
    frames_dir = '/mnt/home/my-blendshape-nerf/data_utils/deprecated'
    target_dir = '/mnt/home/my-blendshape-nerf/data_utils/deprecated'
    resize_W = 1024
    target_H = 1024
    target_W = 1024
    pick_ratio = 2

    detect_and_crop(frames_dir, target_dir, resize_W, target_H, target_W, pick_ratio)

if __name__ == '__main__':
    main()