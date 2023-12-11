import cv2
import os.path as osp
import logging
import argparse

import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

import lpips
loss_fn_alex = lpips.LPIPS(net='alex').cuda()
import torch
import numpy as np

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, default='./options/test/LOLv2_real.yml', help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)


def main():
    save_imgs = True
    model = create_model(opt)
    save_folder = './results/{}'.format(opt['name'])
    GT_folder = osp.join(save_folder, 'images/GT')
    output_folder = osp.join(save_folder, 'images/output')
    output_folder_s1 = osp.join(save_folder, 'images/output_s1')
    input_folder = osp.join(save_folder, 'images/input')
    util.mkdirs(save_folder)
    util.mkdirs(GT_folder)
    util.mkdirs(output_folder)
    util.mkdirs(output_folder_s1)
    util.mkdirs(input_folder)

    print('mkdir finish')

    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')


    for phase, dataset_opt in opt['datasets'].items():
        val_set = create_dataset(dataset_opt)
        val_loader = create_dataloader(val_set, dataset_opt, opt, None)

        pbar = util.ProgressBar(len(val_loader))
        psnr_rlt = {}  # with border and center frames
        psnr_rlt_avg = {}
        psnr_total_avg = 0.

        ssim_rlt = {}  # with border and center frames
        ssim_rlt_avg = {}
        ssim_total_avg = 0.

        lpips_rlt = {}  # with border and center frames
        lpips_rlt_avg = {}
        lpips_total_avg = 0.

        for val_data in val_loader:
            folder = val_data['folder'][0]
            idx_d = val_data['idx']
            if psnr_rlt.get(folder, None) is None:
                psnr_rlt[folder] = []

            if ssim_rlt.get(folder, None) is None:
                ssim_rlt[folder] = []

            if lpips_rlt.get(folder, None) is None:
                lpips_rlt[folder] = []
            model.feed_data(val_data)

            model.test()
            visuals = model.get_current_visuals()
            rlt_img = util.tensor2img(visuals['rlt'])  # uint8
            rlt_s1_img = util.tensor2img(visuals['rlt_s1'])  # uint8
            gt_img = util.tensor2img(visuals['GT'])  # uint8

            input_img = util.tensor2img(visuals['LQ'])
            if save_imgs:
                try:
                    tag = '{}.{}'.format(val_data['folder'], idx_d[0].replace('/', '-'))
                    print(osp.join(output_folder, '{}.png'.format(tag)))
                    cv2.imwrite(osp.join(output_folder, '{}.png'.format(tag)), rlt_img)
                    cv2.imwrite(osp.join(GT_folder, '{}.png'.format(tag)), gt_img)
                    cv2.imwrite(osp.join(output_folder_s1, '{}.png'.format(tag)), rlt_s1_img)

                    cv2.imwrite(osp.join(input_folder, '{}.png'.format(tag)), input_img)

                except Exception as e:
                    print(e)
                    import ipdb; ipdb.set_trace()

            # calculate PSNR
            # psnr = util.calculate_psnr(rlt_img, gt_img)
            psnr = peak_signal_noise_ratio(rlt_img, gt_img)
            psnr_rlt[folder].append(psnr)

            # ssim = util.calculate_ssim(rlt_img, gt_img)
            ssim = structural_similarity(rlt_img, gt_img, multichannel=True)
            # ssim = 0
            ssim_rlt[folder].append(ssim)

            img, gt = rlt_img, gt_img
            img = torch.from_numpy(np.float32(img))
            gt = torch.from_numpy(np.float32(gt))
            img = img.permute(2, 0, 1).unsqueeze(0).cuda()
            gt = gt.permute(2, 0, 1).unsqueeze(0).cuda()
            lpips_alex = loss_fn_alex(img, gt)
            lpips_alex = lpips_alex.detach().cpu().numpy().squeeze()
            lpips_rlt[folder].append(lpips_alex)

            pbar.update('Test {} - {}'.format(folder, idx_d))
        for k, v in psnr_rlt.items():
            psnr_rlt_avg[k] = sum(v) / len(v)
            psnr_total_avg += psnr_rlt_avg[k]

        for k, v in ssim_rlt.items():
            ssim_rlt_avg[k] = sum(v) / len(v)
            ssim_total_avg += ssim_rlt_avg[k]

        for k, v in lpips_rlt.items():
            lpips_rlt_avg[k] = sum(v) / len(v)
            lpips_total_avg += lpips_rlt_avg[k]

        psnr_total_avg /= len(psnr_rlt)
        ssim_total_avg /= len(ssim_rlt)
        lpips_total_avg /= len(lpips_rlt)
        log_s = '# Validation # PSNR: {:.4e}:'.format(psnr_total_avg)
        for k, v in psnr_rlt_avg.items():
            log_s += ' {}: {:.4e}'.format(k, v)
        logger.info(log_s)

        log_s = '# Validation # SSIM: {:.4e}:'.format(ssim_total_avg)
        for k, v in ssim_rlt_avg.items():
            log_s += ' {}: {:.4e}'.format(k, v)
        logger.info(log_s)

        log_s = '# Validation # LPIPS: {:.4e}:'.format(ssim_total_avg)
        for k, v in lpips_rlt_avg.items():
            log_s += ' {}: {:.4e}'.format(k, v)
        logger.info(log_s)

        psnr_all = 0
        psnr_count = 0
        for k, v in psnr_rlt.items():
            psnr_all += sum(v)
            psnr_count += len(v)
        psnr_all = psnr_all * 1.0 / psnr_count
        print(psnr_all)

        ssim_all = 0
        ssim_count = 0
        for k, v in ssim_rlt.items():
            ssim_all += sum(v)
            ssim_count += len(v)
        ssim_all = ssim_all * 1.0 / ssim_count
        print(ssim_all)

        lpips_all = 0
        lpips_count = 0
        for k, v in lpips_rlt.items():
            lpips_all += sum(v)
            lpips_count += len(v)
        lpips_all = lpips_all * 1.0 / lpips_count
        print(lpips_all)


if __name__ == '__main__':
    main()
