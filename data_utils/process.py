import cv2
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import face_alignment
import argparse
import glob
import torch
import tqdm
import imageio

import torch.nn.functional as F
import numpy as np

from crop_image_utils import detect_and_crop
from skimage import io
from sklearn.neighbors import NearestNeighbors

# iphone 需要改的参数:
# extract_audio sample_rate
# asr.py sample_rate, fps
# extract_images 无需改变fps
# crop_images pick_ratio == 1
# 

FFMPEG_PATH = '/mnt/home/ffmpeg-git-20220910-amd64-static/ffmpeg'

def extract_audio(vid_path, out_path, sample_rate=19200):
    
    print(f'[INFO] ===== extract audio from {vid_path} to {out_path} =====')
    cmd = f'{FFMPEG_PATH} -i {vid_path} -f wav -ar {sample_rate} {out_path}'
    os.system(cmd)
    print(f'[INFO] ===== extracted audio =====')


def extract_audio_features(path, mode, fps, sample_rate):

    print(f'[INFO] ===== extract audio labels for {path} =====')
    if mode == 'wav2vec':
        # TODO: 采样率作为asr的输入参数，现在还需要到asr.py中手动修改采样率
        cmd = f'python data_utils/asr.py --wav {path} --save_feats --fps {fps*2} --sample_rate {sample_rate}'
    # else: # deepspeech # worse quality and slower
    #     cmd = f'python data_utils/deepspeech_features/extract_ds_features.py --input {path}'
    os.system(cmd)
    print(f'[INFO] ===== extracted audio labels =====')

# TODO: iphone_vid超参集成
def extract_images(path, out_path, fps=60, iphone_vid=False):
    print(f'[INFO] ===== extract images from {path} to {out_path} =====')
    if iphone_vid:
        convert_name = os.path.splitext(os.path.basename(path))[0] + '_' + str(fps) + 'fps.mp4'
        convert_path = os.path.join(os.path.dirname(path), convert_name)
        print(f'[INFO] convert video\' fps to {fps}..., using latest ffmpeg linux build.') 
        cmd = f'{FFMPEG_PATH} -i {path} -filter:v fps=fps={fps} {convert_path}'
        os.system(cmd)
        print(f'[INFO] extract images from {convert_path} with fps={fps}...')
        cmd = f'{FFMPEG_PATH} -i {convert_path} -vf fps={fps} -qmin 1 -q:v 1 -start_number 0 {os.path.join(out_path, "%d.jpg")}'
        os.system(cmd)
    else:
        cmd = f'{FFMPEG_PATH} -i {path} -vf fps={fps} -qmin 1 -q:v 1 -start_number 0 {os.path.join(out_path, "%d.jpg")}'
        os.system(cmd)
    print(f'[INFO] ===== extracted images =====')

# TODO: pick_ratio超参集成
def crop_images(path, out_path, resize_W=512, target_H=512, target_W=512, pick_ratio=1):
    print(f'[INFO] ===== resize and crop images from {path} to {out_path} =====')
    detect_and_crop(path, out_path, resize_W, target_H, target_W, pick_ratio)
    print(f'[INFO] ===== cropped images =====')


def extract_semantics(ori_imgs_dir, parsing_dir):

    print(f'[INFO] ===== extract semantics from {ori_imgs_dir} to {parsing_dir} =====')
    cmd = f'python data_utils/face_parsing/test.py --respath={parsing_dir} --imgpath={ori_imgs_dir}'
    os.system(cmd)
    print(f'[INFO] ===== extracted semantics =====')


def extract_background(base_dir, ori_imgs_dir):

    print(f'[INFO] ===== extract background image from {ori_imgs_dir} =====')

    from sklearn.neighbors import NearestNeighbors

    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    # only use 1/50 image_paths  TODO: 时间上这个可以优化
    image_paths = image_paths[::50]
    # read one image to get H/W
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED) # [H, W, 3]
    h, w = tmp_image.shape[:2]

    # nearest neighbors
    all_xys = np.mgrid[0:h, 0:w].reshape(2, -1).transpose()
    distss = []
    for image_path in tqdm.tqdm(image_paths):
        parse_img = cv2.imread(image_path.replace('ori_imgs', 'parsing').replace('.jpg', '.png'))
        bg = (parse_img[..., 0] == 255) & (parse_img[..., 1] == 255) & (parse_img[..., 2] == 255)
        fg_xys = np.stack(np.nonzero(~bg)).transpose(1, 0)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
        dists, _ = nbrs.kneighbors(all_xys)
        distss.append(dists)

    distss = np.stack(distss)
    max_dist = np.max(distss, 0)
    max_id = np.argmax(distss, 0)

    bc_pixs = max_dist > 5
    bc_pixs_id = np.nonzero(bc_pixs)
    bc_ids = max_id[bc_pixs]

    imgs = []
    num_pixs = distss.shape[1]
    for image_path in image_paths:
        img = cv2.imread(image_path)
        imgs.append(img)
    imgs = np.stack(imgs).reshape(-1, num_pixs, 3)

    bc_img = np.zeros((h*w, 3), dtype=np.uint8)
    bc_img[bc_pixs_id, :] = imgs[bc_ids, bc_pixs_id, :]
    bc_img = bc_img.reshape(h, w, 3)

    max_dist = max_dist.reshape(h, w)
    bc_pixs = max_dist > 5
    bg_xys = np.stack(np.nonzero(~bc_pixs)).transpose()
    fg_xys = np.stack(np.nonzero(bc_pixs)).transpose()
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
    distances, indices = nbrs.kneighbors(bg_xys)
    bg_fg_xys = fg_xys[indices[:, 0]]
    bc_img[bg_xys[:, 0], bg_xys[:, 1], :] = bc_img[bg_fg_xys[:, 0], bg_fg_xys[:, 1], :]

    cv2.imwrite(os.path.join(base_dir, 'bc.jpg'), bc_img)

    print(f'[INFO] ===== extracted background image =====')

def extract_torso_and_gt(base_dir, ori_imgs_dir, white_bg=True):

    print(f'[INFO] ===== extract torso and gt images for {base_dir} =====')

    from scipy.ndimage import binary_erosion, binary_dilation

    # load bg
    bg_image = cv2.imread(os.path.join(base_dir, 'bc.jpg'), cv2.IMREAD_UNCHANGED)
    
    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))

    for image_path in tqdm.tqdm(image_paths):
        # read ori image
        ori_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # [H, W, 3]

        # read semantics
        seg = cv2.imread(image_path.replace('ori_imgs', 'parsing').replace('.jpg', '.png'))
        head_part = (seg[..., 0] == 255) & (seg[..., 1] == 0) & (seg[..., 2] == 0)
        neck_part = (seg[..., 0] == 0) & (seg[..., 1] == 255) & (seg[..., 2] == 0)
        torso_part = (seg[..., 0] == 0) & (seg[..., 1] == 0) & (seg[..., 2] == 255)
        bg_part = (seg[..., 0] == 255) & (seg[..., 1] == 255) & (seg[..., 2] == 255)

        # get gt image
        gt_image = ori_image.copy()
        if white_bg:
            gt_image[bg_part] = 255
        else:
            gt_image[bg_part] = bg_image[bg_part]
        cv2.imwrite(image_path.replace('ori_imgs', 'gt_imgs'), gt_image)

        # get torso image
        torso_image = gt_image.copy() # rgb
        torso_image[head_part] = bg_image[head_part]
        torso_alpha = 255 * np.ones((gt_image.shape[0], gt_image.shape[1], 1), dtype=np.uint8) # alpha
        
        # torso part "vertical" in-painting...
        L = 8 + 1
        torso_coords = np.stack(np.nonzero(torso_part), axis=-1) # [M, 2]
        # lexsort: sort 2D coords first by y then by x, 
        # ref: https://stackoverflow.com/questions/2706605/sorting-a-2d-numpy-array-by-multiple-axes
        inds = np.lexsort((torso_coords[:, 0], torso_coords[:, 1]))
        torso_coords = torso_coords[inds]
        # choose the top pixel for each column
        u, uid, ucnt = np.unique(torso_coords[:, 1], return_index=True, return_counts=True)
        top_torso_coords = torso_coords[uid] # [m, 2]
        # only keep top-is-head pixels
        top_torso_coords_up = top_torso_coords.copy() - np.array([1, 0])
        mask = head_part[tuple(top_torso_coords_up.T)] 
        if mask.any():
            top_torso_coords = top_torso_coords[mask]
            # get the color
            top_torso_colors = gt_image[tuple(top_torso_coords.T)] # [m, 3]
            # construct inpaint coords (vertically up, or minus in x)
            inpaint_torso_coords = top_torso_coords[None].repeat(L, 0) # [L, m, 2]
            inpaint_offsets = np.stack([-np.arange(L), np.zeros(L, dtype=np.int32)], axis=-1)[:, None] # [L, 1, 2]
            inpaint_torso_coords += inpaint_offsets
            inpaint_torso_coords = inpaint_torso_coords.reshape(-1, 2) # [Lm, 2]
            inpaint_torso_colors = top_torso_colors[None].repeat(L, 0) # [L, m, 3]
            darken_scaler = 0.98 ** np.arange(L).reshape(L, 1, 1) # [L, 1, 1]
            inpaint_torso_colors = (inpaint_torso_colors * darken_scaler).reshape(-1, 3) # [Lm, 3]
            # set color
            torso_image[tuple(inpaint_torso_coords.T)] = inpaint_torso_colors

            inpaint_torso_mask = np.zeros_like(torso_image[..., 0]).astype(bool)
            inpaint_torso_mask[tuple(inpaint_torso_coords.T)] = True
        else:
            inpaint_torso_mask = None
            

        # neck part "vertical" in-painting...
        push_down = 4
        L = 48 + push_down + 1

        neck_part = binary_dilation(neck_part, structure=np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool), iterations=3)

        neck_coords = np.stack(np.nonzero(neck_part), axis=-1) # [M, 2]
        # lexsort: sort 2D coords first by y then by x, 
        # ref: https://stackoverflow.com/questions/2706605/sorting-a-2d-numpy-array-by-multiple-axes
        inds = np.lexsort((neck_coords[:, 0], neck_coords[:, 1]))
        neck_coords = neck_coords[inds]
        # choose the top pixel for each column
        u, uid, ucnt = np.unique(neck_coords[:, 1], return_index=True, return_counts=True)
        top_neck_coords = neck_coords[uid] # [m, 2]
        # only keep top-is-head pixels
        top_neck_coords_up = top_neck_coords.copy() - np.array([1, 0])
        mask = head_part[tuple(top_neck_coords_up.T)] 
        
        top_neck_coords = top_neck_coords[mask]
        # push these top down for 4 pixels to make the neck inpainting more natural...
        offset_down = np.minimum(ucnt[mask] - 1, push_down)
        top_neck_coords += np.stack([offset_down, np.zeros_like(offset_down)], axis=-1)
        # get the color
        top_neck_colors = gt_image[tuple(top_neck_coords.T)] # [m, 3]
        # construct inpaint coords (vertically up, or minus in x)
        inpaint_neck_coords = top_neck_coords[None].repeat(L, 0) # [L, m, 2]
        inpaint_offsets = np.stack([-np.arange(L), np.zeros(L, dtype=np.int32)], axis=-1)[:, None] # [L, 1, 2]
        inpaint_neck_coords += inpaint_offsets
        inpaint_neck_coords = inpaint_neck_coords.reshape(-1, 2) # [Lm, 2]
        inpaint_neck_colors = top_neck_colors[None].repeat(L, 0) # [L, m, 3]
        darken_scaler = 0.98 ** np.arange(L).reshape(L, 1, 1) # [L, 1, 1]
        inpaint_neck_colors = (inpaint_neck_colors * darken_scaler).reshape(-1, 3) # [Lm, 3]
        # set color
        torso_image[tuple(inpaint_neck_coords.T)] = inpaint_neck_colors

        # apply blurring to the inpaint area to avoid vertical-line artifects...
        inpaint_mask = np.zeros_like(torso_image[..., 0]).astype(bool)
        inpaint_mask[tuple(inpaint_neck_coords.T)] = True

        blur_img = torso_image.copy()
        blur_img = cv2.GaussianBlur(blur_img, (5, 5), cv2.BORDER_DEFAULT)

        torso_image[inpaint_mask] = blur_img[inpaint_mask]

        # set mask
        mask = (neck_part | torso_part | inpaint_mask)
        if inpaint_torso_mask is not None:
            mask = mask | inpaint_torso_mask
        torso_image[~mask] = 0
        torso_alpha[~mask] = 0

        cv2.imwrite(image_path.replace('ori_imgs', 'torso_imgs').replace('.jpg', '.png'), np.concatenate([torso_image, torso_alpha], axis=-1))

    print(f'[INFO] ===== extracted torso and gt images =====')


def extract_landmarks(ori_imgs_dir):

    print(f'[INFO] ===== extract face landmarks from {ori_imgs_dir} =====')

    import face_alignment
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    for image_path in tqdm.tqdm(image_paths):
        input = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # [H, W, 3]
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        preds = fa.get_landmarks(input)
        if len(preds) > 0:
            lands = preds[0].reshape(-1, 2)[:,:2]
            np.savetxt(image_path.replace('jpg', 'lms'), lands, '%f')
    del fa
    print(f'[INFO] ===== extracted face landmarks =====')

def face_tracking(ori_imgs_dir):

    print(f'[INFO] ===== perform face tracking =====')

    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    
    # read one image to get H/W
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED) # [H, W, 3]
    h, w = tmp_image.shape[:2]

    cmd = f'python data_utils/face_tracking/face_tracker.py --path={ori_imgs_dir} --img_h={h} --img_w={w} --frame_num={len(image_paths)}'

    os.system(cmd)

    print(f'[INFO] ===== finished face tracking =====')


def save_transforms(base_dir, ori_imgs_dir):
    print(f'[INFO] ===== save transforms =====')

    import torch

    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    # print(ori_imgs_dir)
    # read one image to get H/W
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED) # [H, W, 3]
    h, w = tmp_image.shape[:2]
    params_dict = torch.load(os.path.join(base_dir, 'track_params.pt'))
    focal_len = params_dict['focal']
    euler_angle = params_dict['euler']
    trans = params_dict['trans'] / 10.0

    #trans = params_dict['trans']
    def euler2rot(euler_angle):
        batch_size = euler_angle.shape[0]
        theta = euler_angle[:, 0].reshape(-1, 1, 1)
        phi = euler_angle[:, 1].reshape(-1, 1, 1)
        psi = euler_angle[:, 2].reshape(-1, 1, 1)
        one = torch.ones((batch_size, 1, 1), dtype=torch.float32, device=euler_angle.device)
        zero = torch.zeros((batch_size, 1, 1), dtype=torch.float32, device=euler_angle.device)
        rot_x = torch.cat((
            torch.cat((one, zero, zero), 1),
            torch.cat((zero, theta.cos(), theta.sin()), 1),
            torch.cat((zero, -theta.sin(), theta.cos()), 1),
        ), 2)
        rot_y = torch.cat((
            torch.cat((phi.cos(), zero, -phi.sin()), 1),
            torch.cat((zero, one, zero), 1),
            torch.cat((phi.sin(), zero, phi.cos()), 1),
        ), 2)
        rot_z = torch.cat((
            torch.cat((psi.cos(), -psi.sin(), zero), 1),
            torch.cat((psi.sin(), psi.cos(), zero), 1),
            torch.cat((zero, zero, one), 1)
        ), 2)
        return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))
    # 一半是训练集，一般是验证集
    valid_num = euler_angle.shape[0]
    train_val_split = int(valid_num*0.5)
    train_ids = torch.arange(0, train_val_split)
    val_ids = torch.arange(train_val_split, valid_num)
    all_ids = torch.arange(0, valid_num)
    rot = euler2rot(euler_angle)
    rot_inv = rot.permute(0, 2, 1)

    trans_inv = -torch.bmm(rot_inv, trans.unsqueeze(2))
    pose = torch.eye(4, dtype=torch.float32)
    save_ids = ['train', 'val']
    train_val_ids = [train_ids, val_ids]
    mean_z = -float(torch.mean(trans[:, 2]).item())

    # 生成所有的图像到一个json文件（方便渲染出长视频）
    transform_dict = dict()
    transform_dict['focal_len'] = float(focal_len[0])
    transform_dict['cx'] = float(w/2.0)
    transform_dict['cy'] = float(h/2.0)
    transform_dict['frames'] = []
    ids = all_ids
    # save_id = save_ids[i]
    for i in ids:
        i = i.item()
        frame_dict = dict()
        frame_dict['img_id'] = i
        frame_dict['aud_id'] = i
        pose[:3, :3] = rot_inv[i]
        pose[:3, 3] = trans_inv[i, :, 0]
        frame_dict['transform_matrix'] = pose.numpy().tolist()

        frame_dict['trans'] = trans[i].numpy().tolist()

        lms = np.loadtxt(os.path.join(
            ori_imgs_dir, str(i) + '.lms'))
        # lms = np.loadtxt(os.path.join(
        #     ori_imgs_dir, str(i) + '_lm2d.txt')).reshape(-1, 2)
        min_x, max_x = np.min(lms, 0)[0], np.max(lms, 0)[0]
        import pdb
        cx = int((min_x+max_x)/2.0)
        cy = int(lms[27, 1])
        h_w = int((max_x-cx)*1.5)
        h_h = int((lms[8, 1]-cy)*1.15)
        rect_x = cx - h_w
        rect_y = cy - h_h
        if rect_x < 0:
            rect_x = 0
        if rect_y < 0:
            rect_y = 0
        rect_w = min(w-1-rect_x, 2*h_w)
        rect_h = min(h-1-rect_y, 2*h_h)
        rect = np.array((rect_x, rect_y, rect_w, rect_h), dtype=np.int32)
        frame_dict['face_rect'] = rect.tolist()

        min_x, max_x = np.min(lms, 0)[0], np.max(lms, 0)[0]
        cx = int((min_x+max_x)/2.0)
        cy = int(lms[66, 1])
        h_w = int((lms[54, 0]-cx)*1.2)
        h_h = int((lms[57, 1]-cy)*1.2)
        rect_x = cx - h_w
        rect_y = cy - h_h
        if rect_x < 0:
            rect_x = 0
        if rect_y < 0:
            rect_y = 0
        rect_w = min(w-1-rect_x, 2*h_w)
        rect_h = min(h-1-rect_y, 2*h_h)
        rect = np.array((rect_x, rect_y, rect_w, rect_h), dtype=np.int32)
        frame_dict['lip_rect'] = rect.tolist()
        transform_dict['frames'].append(frame_dict)

    with open(os.path.join(base_dir, 'transforms_all' + '.json'), 'w') as fp:
        json.dump(transform_dict, fp, indent=2, separators=(',', ': '))


    # 保存训练验证集
    for i in range(2):
        transform_dict = dict()
        transform_dict['focal_len'] = float(focal_len[0])
        transform_dict['cx'] = float(w/2.0)
        transform_dict['cy'] = float(h/2.0)
        transform_dict['frames'] = []
        ids = train_val_ids[i]
        save_id = save_ids[i]
        for i in ids:
            i = i.item()
            frame_dict = dict()
            frame_dict['img_id'] = i
            frame_dict['aud_id'] = i
            pose[:3, :3] = rot_inv[i]
            pose[:3, 3] = trans_inv[i, :, 0]
            frame_dict['transform_matrix'] = pose.numpy().tolist()
            frame_dict['trans'] = trans[i].numpy().tolist()
            lms = np.loadtxt(os.path.join(
                ori_imgs_dir, str(i) + '.lms'))
            # lms = np.loadtxt(os.path.join(
            #     ori_imgs_dir, str(i) + '_lm2d.txt')).reshape(-1, 2)
            min_x, max_x = np.min(lms, 0)[0], np.max(lms, 0)[0]
            import pdb
            cx = int((min_x+max_x)/2.0)
            cy = int(lms[27, 1])
            h_w = int((max_x-cx)*1.5)
            h_h = int((lms[8, 1]-cy)*1.15)
            rect_x = cx - h_w
            rect_y = cy - h_h
            if rect_x < 0:
                rect_x = 0
            if rect_y < 0:
                rect_y = 0
            rect_w = min(w-1-rect_x, 2*h_w)
            rect_h = min(h-1-rect_y, 2*h_h)
            rect = np.array((rect_x, rect_y, rect_w, rect_h), dtype=np.int32)
            frame_dict['face_rect'] = rect.tolist()

            min_x, max_x = np.min(lms, 0)[0], np.max(lms, 0)[0]
            cx = int((min_x+max_x)/2.0)
            cy = int(lms[66, 1])
            h_w = int((lms[54, 0]-cx)*1.2)
            h_h = int((lms[57, 1]-cy)*1.2)
            rect_x = cx - h_w
            rect_y = cy - h_h
            if rect_x < 0:
                rect_x = 0
            if rect_y < 0:
                rect_y = 0
            rect_w = min(w-1-rect_x, 2*h_w)
            rect_h = min(h-1-rect_y, 2*h_h)
            rect = np.array((rect_x, rect_y, rect_w, rect_h), dtype=np.int32)
            frame_dict['lip_rect'] = rect.tolist()

            transform_dict['frames'].append(frame_dict)
        with open(os.path.join(base_dir, 'transforms_' + save_id + '.json'), 'w') as fp:
            json.dump(transform_dict, fp, indent=2, separators=(',', ': '))


    print(f'[INFO] ===== finished saving transforms =====')


def extract_head_gt(base_dir, ori_imgs_dir, parsing_dir, head_imgs_dir, white_bg=True):

    print(f'[INFO] ===== extract head images for {base_dir} =====')
    
    if not os.path.exists(head_imgs_dir):
        os.mkdir(head_imgs_dir)
    bc_img = cv2.imread(os.path.join(base_dir, 'bc.jpg'))
    valid_img_ids = range(len(os.listdir(parsing_dir)))
    for i in valid_img_ids:
        parsing_img = cv2.imread(os.path.join(parsing_dir, str(i) + '.png'))
        head_part = (parsing_img[:, :, 0] == 255) & (
            parsing_img[:, :, 1] == 0) & (parsing_img[:, :, 2] == 0)
        bc_part = (parsing_img[:, :, 0] == 255) & (
            parsing_img[:, :, 1] == 255) & (parsing_img[:, :, 2] == 255)
        img = cv2.imread(os.path.join(ori_imgs_dir, str(i) + '.jpg'))
        if white_bg:
            img[~head_part] = 255 
        else:
            img[~head_part] = bc_img[~head_part]
        cv2.imwrite(os.path.join(head_imgs_dir, str(i) + '.jpg'), img)

    print(f'[INFO] ===== finished extracting head images =====')


def extract_head_masks(parsing_dir, head_masks_dir, only_head=True):

    print(f'[INFO] ===== extract head masks for {parsing_dir} =====')

    images_dir = parsing_dir
    temp_masks_path = [i for i in os.listdir(images_dir) if i.endswith('.png')]
    for m in temp_masks_path:
        tmp_mask = cv2.imread(os.path.join(parsing_dir, m))
        mask1 = tmp_mask[:, :, 0] == 255
        mask2 = tmp_mask[:, :, 1] == 255
        mask3 = tmp_mask[:, :, 2] == 255
        mask4 = tmp_mask[:, :, 1] == 0
        mask5 = tmp_mask[:, :, 2] == 0
        mask_white = mask1 & mask2 & mask3
        mask_blue = mask1 & mask4 & mask5
        if only_head:
            tmp_mask[mask_blue] = 255
            tmp_mask[~mask_blue] = 0
            cv2.imwrite(os.path.join(head_masks_dir, m), tmp_mask)
        else:
            tmp_mask[mask_white] = 0
            tmp_mask[~mask_white] = 255
            cv2.imwrite(os.path.join(head_masks_dir, m), tmp_mask)
    
    print(f'[INFO] ===== finished extracting head masks =====')


# def extract_blendshape(path, out_path, add_mean=True):
#     print(f'[INFO] ===== extract blendshape without smoothing...=====')
#     cmd = f'python data_utils/bs_solver/inference.py --img_path {path} --save_path {out_path}'
#     os.system(cmd)
#     # save min/max expr blendshape for trainset
#     expr_temp = np.load(os.path.join(out_path, 'expr.npy'))
#     expr_temp = expr_temp[:-500, :]
#     expr_min = expr_temp.min(axis=0)
#     expr_max = expr_temp.max(axis=0)
#     expr_mean = expr_temp.mean(axis=0)
#     if add_mean:
#         expr_min = np.concatenate([np.array([1.0]), expr_min])
#         expr_max = np.concatenate([np.array([1.0]), expr_max])
#         expr_mean = np.concatenate([np.array([1.0]), expr_mean])
#     np.save(os.path.join(out_path, 'expr_min.npy'), expr_min)
#     np.save(os.path.join(out_path, 'expr_max.npy'), expr_max)
#     np.save(os.path.join(out_path, 'expr_mean.npy'), expr_mean)

#     print(f'[INFO] ===== extract blendshape with smoothing...=====')
#     cmd = f'python data_utils/bs_solver/inference.py --img_path {path} --save_path {out_path} --use_smooth'
#     os.system(cmd)

def extract_gt_test_video(path, out_path, num_frames_test=800, fps=30):
    print(f'[INFO] ===== extract gt test video for {path} with test frames {num_frames_test}...=====')
    imgs_path = [os.path.join(path, i) for i in os.listdir(path) if i.endswith('.jpg') or i.endswith('.png')]
    imgs_path = sorted(imgs_path, key=lambda x: int(os.path.basename(x).split('.')[0]))
    imgs_path = imgs_path[-num_frames_test:]
    frames_output = []
    for i in tqdm.tqdm(range(len(imgs_path))):
        frame_path = imgs_path[i]
        frame_in = cv2.imread(frame_path)
        frames_output.append(cv2.cvtColor((frame_in), cv2.COLOR_BGR2RGB))
    frames_output = np.stack(frames_output, axis=0)
    imageio.mimwrite(os.path.join(out_path, 'gt.mp4'), frames_output, fps=fps, quality=8, macro_block_size=1) # 质量有点低
    print(f'[INFO] ===== Done extract gt test video for {path} with test frames {num_frames_test}.=====')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('id', type=str, help="video name, should be put to data/vids")
    parser.add_argument('--task', type=int, default=-1, help="-1 means all")
    parser.add_argument('--sample_rate', type=int, default=19200, help="sample rate for audio")
    parser.add_argument('--fps', type=int, default=30, help="fps for input video")
    parser.add_argument('--num_test', type=int, default=800, help="number of test frames(for generating gt video)")
    parser.add_argument('--asr', type=str, default='wav2vec', help="wav2vec or deepspeech")

    opt = parser.parse_args()

    # temporarily only support the following settings
    assert opt.fps == 25 or opt.fps == 30
    if opt.fps == 25:
        assert opt.sample_rate == 16000
    elif opt.fps == 30:
        assert opt.sample_rate == 19200

    # 后缀无关的获取文件名
    vid_file = [i for i in os.listdir(os.path.join('data', 'vids')) if i.split('.')[0] == opt.id][0]

    vid_path = os.path.join('data', 'vids', vid_file)
    base_dir = os.path.join('data', opt.id)
    wav_path = os.path.join(base_dir, 'aud.wav')
    ori_imgs_dir = os.path.join(base_dir, 'ori_imgs')
    frames_dir = os.path.join(base_dir, 'frames')
    parsing_dir = os.path.join(base_dir, 'parsing')
    gt_imgs_dir = os.path.join(base_dir, 'gt_imgs')
    torso_imgs_dir = os.path.join(base_dir, 'torso_imgs')
    head_imgs_dir = os.path.join(base_dir, 'head_imgs')
    head_masks_dir = os.path.join(base_dir, 'head_masks')

    os.makedirs(ori_imgs_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(parsing_dir, exist_ok=True)
    os.makedirs(gt_imgs_dir, exist_ok=True)
    os.makedirs(torso_imgs_dir, exist_ok=True)
    os.makedirs(head_imgs_dir, exist_ok=True)
    os.makedirs(head_masks_dir, exist_ok=True)

    # extract audio
    if opt.task == -1 or opt.task == 1:
        extract_audio(vid_path, wav_path, sample_rate=opt.sample_rate)

    # extract audio features
    if opt.task == -1 or opt.task == 2:
        extract_audio_features(wav_path, mode=opt.asr, fps=opt.fps, sample_rate=opt.sample_rate)

    # extract images
    if opt.task == -1 or opt.task == 3:
        extract_images(vid_path, frames_dir, opt.fps)

    # crop images
    if opt.task == -1 or opt.task == 4:
        crop_images(frames_dir, ori_imgs_dir, resize_W=512, target_H=512, target_W=512, pick_ratio=1)

    # face parsing
    if opt.task == -1 or opt.task == 5:
        extract_semantics(ori_imgs_dir, parsing_dir)

    # extract bg
    if opt.task == -1 or opt.task == 6:
        extract_background(base_dir, ori_imgs_dir)

    # extract torso images and gt_images
    if opt.task == -1 or opt.task == 7:
        extract_torso_and_gt(base_dir, ori_imgs_dir)

    # extract head images for training
    if opt.task == -1 or opt.task == 8:
        extract_head_gt(base_dir, ori_imgs_dir, parsing_dir, head_imgs_dir)

    # extract_head_masks
    if opt.task == -1 or opt.task == 9:
        extract_head_masks(parsing_dir, head_masks_dir)

    # extract face landmarks
    if opt.task == -1 or opt.task == 10:
        extract_landmarks(ori_imgs_dir)

    # face tracking
    if opt.task == -1 or opt.task == 11:
        face_tracking(ori_imgs_dir)

    # save transforms.json
    if opt.task == -1 or opt.task == 12:
        save_transforms(base_dir, ori_imgs_dir)
    
    # extract blendshape
#     if opt.task == -1 or opt.task == 13:
#         extract_blendshape(gt_imgs_dir, base_dir)
    
    # extract testset video
    if opt.task == -1 or opt.task == 14:
        extract_gt_test_video(gt_imgs_dir, base_dir, opt.num_test, opt.fps)


    
