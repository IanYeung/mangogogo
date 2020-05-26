import os
import os.path as osp
import glob
import logging
import math
import numpy as np
import cv2
import torch
import pickle

import utils.util as util
import data.util as data_util
import models.archs.EDVR_arch as EDVR_arch


def main(gpu_id, start_id, step):
    #################
    # configurations
    #################
    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    data_mode = 'MGTV'

    flip_test = False
    model_path = '../experiments/pretrained_models/70000_G.pth'  # TODO: change path

    N_in = 7  # use N_in images to restore one HR image
    model = EDVR_arch.EDVR(128, N_in, 8, 5, 20, predeblur=False, HR_in=True, w_TSA=False)

    test_dataset_folder = '/home/xiyang/Datasets/MGTV/test_damage_A_frames'  # TODO: change path

    #### scene information
    scene_index_path = '../keys/scene_index_test.pkl'  # TODO: change path
    scene_dict = pickle.load(open(scene_index_path, 'rb'))

    #### evaluation
    padding = 'replicate'  # temporal padding mode
    save_imgs = True
    save_folder = '/home/xiyang/Datasets/MGTV/test_damage_A_results'  # TODO: change path
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    #### log info
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip test: {}'.format(flip_test))

    #### set up the models
    print('Loading model from {}'.format(model_path))
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    subfolder_name_l = []
    subfolder_l = sorted(glob.glob(osp.join(test_dataset_folder, '*')))
    seq_id = start_id
    for subfolder in subfolder_l[start_id::step]:
        subfolder_name = osp.basename(subfolder)
        subfolder_name_l.append(subfolder_name)
        save_subfolder = osp.join(save_folder, subfolder_name)
        img_path_l = sorted(glob.glob(osp.join(subfolder, '*')))

        max_idx = len(img_path_l)
        if save_imgs:
            util.mkdirs(save_subfolder)

        # read LQ images
        imgs_LQ = data_util.read_img_seq(subfolder)

        # process each image
        for img_idx, img_path in enumerate(img_path_l):
            img_name = osp.splitext(osp.basename(img_path))[0]
            select_idx = data_util.index_generation_with_scene_list(img_idx, max_idx, N_in,
                                                                    scene_dict[subfolder_name],
                                                                    padding=padding)
            imgs_in = imgs_LQ.index_select(0, torch.LongTensor(select_idx)).to(device)
            imgs_in = imgs_in.unsqueeze(0)

            # # inference once by using padding
            frames = torch.zeros(1, N_in, 3, 1088, 1920).to(device)
            frames[:, :, 0, :, :] =  16. / 255.
            frames[:, :, 1, :, :] = 128. / 255.
            frames[:, :, 2, :, :] = 128. / 255.
            frames[:, :, :, 4:-4, :] = imgs_in
            if flip_test:
                output = util.flipx4_forward(model, frames)
            else:
                output = util.single_forward(model, frames)
            output = output[:, :, 4:-4, :]
            output = util.tensor2img(output)
            if save_imgs:
                cv2.imwrite(osp.join(save_subfolder, '{}.png'.format(img_name)), output)

            # # inference twice and average over overlap part
            # output_YUV = torch.zeros(1, 3, 1080, 1920)
            #
            # h = 1072
            # w = 1920
            # if flip_test:
            #     output_YUV1 = util.flipx4_forward(model, imgs_in[:, :, :, :h, :w])
            #     output_YUV2 = util.flipx4_forward(model, imgs_in[:, :, :, 1080 - h:, :w])
            # else:
            #     output_YUV1 = util.single_forward(model, imgs_in[:, :, :, :h, :w])
            #     output_YUV2 = util.single_forward(model, imgs_in[:, :, :, 1080 - h:, :w])
            #
            # output_YUV[:, :, :1072, :] = output_YUV1
            # output_YUV[:, :, -1072:, :] = output_YUV2
            #
            # h_overlap = h * 2 - 1080
            # h_start = 1080 - h
            #
            # for i in range(h_overlap):
            #     output_YUV[:, :, (h_start + i):(h_start + i + 1), :] = \
            #         output_YUV1[:, :, (h_start + i):(h_start + i + 1), :] * (h_overlap - i) / h_overlap + \
            #         output_YUV2[:, :, i:(i + 1), :] * i / h_overlap
            #
            # output = util.tensor2img(output_YUV)
            # if save_imgs:
            #     cv2.imwrite(osp.join(save_subfolder, '{}.png'.format(img_name)), output)

        seq_id += step


if __name__ == '__main__':
    # testing with single gpu:
    main(gpu_id='0', start_id=0, step=1)

    # manually switch gpu and use 4 gpus in testing:
    # main(gpu_id='0', start_id=0, step=4)
    # main(gpu_id='1', start_id=1, step=4)
    # main(gpu_id='2', start_id=2, step=4)
    # main(gpu_id='3', start_id=3, step=4)
