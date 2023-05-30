import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
import torch
import argparse

from nerf.provider import NeRFDataset
from nerf.gui import NeRFGUI
from nerf.utils import *

from functools import partial
from loss import huber_loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=300000, help="training iters")
    parser.add_argument('--train_epochs', type=int, default=40, help="training iters") # 多看看40，目前20已足够
    parser.add_argument('--lr', type=float, default=5e-4, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest') # latest scratch best 测试的时候要选好
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step") #
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)") # 1024
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")

    # 非cudaray模式才生效的参数
    parser.add_argument('--num_steps', type=int, default=128, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")

    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")
    parser.add_argument('--eval_interval_epoch', type=int, default=1, help="evaluate every X epoches")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', type=int, default=0, help="0 means load data from disk on-the-fly, 1 means preload to CPU, 2 means GPU.")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=4, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/256, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.05, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    # my options
    parser.add_argument('--mask_all', action='store_true', help="use mask to mask the head and the torso.")
    parser.add_argument('--mask_head', action='store_true', help="use mask to mask only the head")
    parser.add_argument('--use_expr', action='store_true', help="use expression network")
    parser.add_argument('--use_bc', action='store_true', help="to train a replaceable background")

    parser.add_argument('--test_fps', type=int, default=25, help="fps to generate test videos")
    parser.add_argument('--network', type=str, default='hashgrid', help="experimental expression network structure")

    # latent code
    parser.add_argument('--use_latent_code', action='store_true', help="test using latent code")
    parser.add_argument('--loss_latent_lambda', type=float, default=0.001, help="latent loss lambda")

    # mask loss 
    parser.add_argument('--use_mask_loss', action='store_true', help="test adding mask loss")
    parser.add_argument('--loss_mask_lambda', type=float, default=1, help="mask loss lambda") # 0.0001
    parser.add_argument('--mloss_duration_epoch', type=int, default=3, help="how many epochs do we enable mask loss training")


    # TODO: lip pips loss
    parser.add_argument('--use_patch_loss', action='store_true', help="to sample patches of rays to do pips loss on importance areas")
    parser.add_argument('--loss_patch_lambda', type=float, default=0.001, help="latent loss lambda")
    parser.add_argument('--expr_dim', type=int, default=55, help="expression vector dim") # 0.0001
    parser.add_argument('--add_mean', type=bool, default=False, help="flag for adding mean shape face")
    parser.add_argument('--smooth_path', action='store_true', help="brute-force smooth camera pose trajectory with a window size")
    parser.add_argument('--smooth_path_window', type=int, default=7, help="smoothing window size")

    # test in fix pose
    parser.add_argument('--fix_angle', action='store_true', help="test in fix pose")

    # airkit blendshape
    parser.add_argument('--airkit', action='store_true', help="use airkit blendshape")
    parser.add_argument('--white_bg', action='store_true', help="train with white bg")

    # super resolution
    parser.add_argument('--use_sr', action='store_true', help="first downsample nerf rendering , then use super-resolution after nerf rendering")
    parser.add_argument('--sr_patch_size', type=int, default=64, help="training patch size for super-resolution")
    parser.add_argument('--downscale', type=int, default=1, help="downscale 4x for super-resolution, 1x for no sr")
    parser.add_argument('--sr_path', type=str, default='pretrained/RealESRGAN_x4plus.pth', help="pretrained/trained sr net path")

    parser.add_argument('--loss_sr_patch_lambda', type=float, default=0.001, help="latent loss lambda")
    parser.add_argument('--loss_sr_photo_lambda', type=float, default=1, help="latent loss lambda")
    parser.add_argument('--loss_photo_lambda', type=float, default=1, help="photo loss for nerf")
    parser.add_argument('--loss_amb_lambda', type=float, default=0.1, help="ambient loss lambda")
    parser.add_argument('--nerf_pretrained_epoch', type=int, default=15, help="training epoch for nerf")

    # torso
    parser.add_argument('--torso_shrink', type=float, default=0.8, help="shrink bg coords to allow more flexibility in deform")
    parser.add_argument('--density_thresh_torso', type=float, default=0.01, help="threshold for density grid to be occupied (alpha)")
    parser.add_argument('--torso', action='store_true', help="fix head and train torso")
    parser.add_argument('--torso_head_aware', action='store_true', help="train torso with head aware pixel condition")
    parser.add_argument('--ind_dim_head', type=int, default=32, help="individual code dim, 0 to turn off")
    parser.add_argument('--ind_dim_torso', type=int, default=8, help="individual code dim, 0 to turn off")
    parser.add_argument('--head_ckpt', type=str, default='', help="head model")


    # audio2expr
    parser.add_argument('--nerf_expr_ckpt', type=str, default='', help="checkpoint file path for trained expression-driven nerf")
    parser.add_argument('--latent_code_path', type=str, default='', help="latent code dir")

    # multi-modal
    parser.add_argument('--eval_test_modal', type=str, default='audio', help="audio feature path")
    parser.add_argument('--test_audio', action='store_true', help="test trained model with new audio")
    parser.add_argument('--test_expr', action='store_true', help="test trained model with new expression coefficient")
    parser.add_argument('--aud', type=str, default='', help="audio feature path")
    parser.add_argument('--audio_feature_dim', type=int, default=109, help="audio feature dim from wav2vec asr model")

    # # debug
    parser.add_argument('--smooth_expr', action='store_true', help="smooth 3dmm coef for testing(or training)")
    parser.add_argument('--test_num', type=int, default=800, help="test frame num")

    # TODO: downscale参数和sr_ratio参数 在utils.Trainer.test_gui函数中有一定意义上的耦合, 后续需要解决


    # Done: blendshape network

    opt = parser.parse_args()

    if True:
        opt.cuda_ray = True
        opt.use_expr = True
        opt.use_bc = True
        opt.add_mean = True
        opt.airkit = True
        opt.white_bg = True 
        opt.use_latent_code = True
        opt.use_mask_loss = True
        if opt.latent_code_path == '':
            opt.latent_code_path = os.path.join(opt.workspace, 'checkpoints')
        if opt.test:
            opt.smooth_expr = True

        
    if opt.add_mean:
        opt.expr_dim = opt.expr_dim + 1
    else:
        opt.expr_dim = opt.expr_dim

    #
    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = 2
    

    from nerf.network_blend4_noamb import NeRFNetwork

    print(opt)
    
    seed_everything(opt.seed)
    criterion = torch.nn.L1Loss(reduction='none')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeRFNetwork(
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
        use_latent_code=opt.use_latent_code,
        latent_code_dir=opt.latent_code_path,
        expr_dim=opt.expr_dim,
        opt=opt,
    )
    
    print(model)

    # manually load state dict for head
    if opt.head_ckpt != '': # opt.torso and 

        model_dict = torch.load(opt.head_ckpt, map_location='cpu')['model']
        missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)

        if len(missing_keys) > 0:
            print(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"[WARN] unexpected keys: {unexpected_keys}")   

        # freeze these keys
        for k, v in model.named_parameters():
            if k in model_dict:
                # print(f'[INFO] freeze {k}, {v.shape}')
                v.requires_grad = False


    if opt.nerf_expr_ckpt != '': 
        """需要是一个新的workspace才能起效"""
        model_dict = torch.load(opt.nerf_expr_ckpt, map_location='cpu')['model']
        missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)

        if len(missing_keys) > 0:
            print(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"[WARN] unexpected keys: {unexpected_keys}")   

        # freeze these keys
        for k, v in model.named_parameters():
            if k in model_dict:
                # print(f'[INFO] freeze {k}, {v.shape}')
                v.requires_grad = False

    if opt.test:
        opt.smooth_path = True
        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)
        test_loader = NeRFDataset(opt, device=device, type='test', downscale=opt.downscale).dataloader()
        if test_loader.has_gt:
            trainer.evaluate(test_loader) # blender has gt, so evaluate it.

        trainer.test(test_loader, write_video=True, test_fps=opt.test_fps) # test and save video
    else:
        train_dataset = NeRFDataset(opt, device=device, type='train', downscale=opt.downscale)
        train_loader = train_dataset.dataloader(collate='nerf')
        train_loader_sr = train_dataset.dataloader(collate='sr') if opt.use_sr else None

        # training torso, update density grid
        model.poses = train_loader._data.poses

        optimizer = lambda model : torch.optim.Adam(model.get_params(opt.lr, train_loader, device), betas=(0.9, 0.99), eps=1e-15) # 添加latent code作为可训练参数
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))
        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval_epoch)
        val_dataset = NeRFDataset(opt, device=device, type='val', downscale=opt.downscale)
        valid_loader = val_dataset.dataloader(collate='nerf')
        # max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        max_epoch = opt.train_epochs
        print("total train epoch: ", max_epoch)
        trainer.train(train_loader, train_loader_sr, valid_loader, max_epoch)
        # also test
        opt.smooth_path = True

        test_loader = NeRFDataset(opt, device=device, type='test', downscale=opt.downscale).dataloader()
        if test_loader.has_gt:
            trainer.evaluate(test_loader) # blender has gt, so evaluate it.

        model.save_latent_codes(os.path.join(opt.workspace, 'checkpoints'))
        trainer.test(test_loader, write_video=True, test_fps=opt.test_fps) # test and save video
        