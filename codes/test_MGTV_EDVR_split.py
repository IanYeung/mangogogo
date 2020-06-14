import os
import os.path as osp
import glob
import logging
import math
import numpy as np
import cv2
import ffmpeg
import imageio
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

    flip_test = True
    model_path = '../experiments/pretrained_models/RRDBEDVR_400000_split.pth'  # TODO: change path

    N_in = 7  # use N_in images to restore one HR image
    model = EDVR_arch.EDVR_YUV420(128, N_in, 8, 5, 15, predeblur=False, HR_in=True, w_TSA=False)

    test_dataset_folder = '/data/test_damage_B_y4m'  # TODO: change path

    #### scene information
    scene_index_path = '../keys/scene_index_test_B.pkl'  # TODO: change path
    scene_dict = pickle.load(open(scene_index_path, 'rb'))

    #### evaluation
    padding = 'replicate'  # temporal padding mode
    save_imgs = True
    save_folder = '/data/test_damage_B_iter_400000_YUV420'  # TODO: change path
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

    video_path_l = sorted(glob.glob(osp.join(test_dataset_folder, '*')))
    seq_id = start_id
    for video_path in video_path_l[start_id::step]:
        video_name = osp.basename(video_path).split('.')[0]
        logger.info('Processing: {}'.format(osp.basename(video_path)))
        output_path = osp.join(save_folder, 'mg_refine_{}.y4m'.format(video_name.split('_')[2]))

        video = imageio.get_reader(video_path, format='ffmpeg', mode='I', dtype=np.uint8)
        fps = video.get_meta_data()['fps']
        w, h = video.get_meta_data()['size']
        video.close()

        # read the whole sequence into buffer
        reader = (
            ffmpeg.input(video_path)
                  .output('pipe:', format='rawvideo', pix_fmt='yuv420p')
                  .run_async(pipe_stdout=True)
        )
        frame_buffer = []
        while True:
            in_bytes_Y = reader.stdout.read(w * h)
            in_bytes_U = reader.stdout.read(w // 2 * h // 2)
            in_bytes_V = reader.stdout.read(w // 2 * h // 2)
            if not in_bytes_Y:
                print('Finish reading video')
                break
            Y = (np.frombuffer(in_bytes_Y, np.uint8).reshape([h, w]))
            U = (np.frombuffer(in_bytes_U, np.uint8).reshape([h // 2, w // 2]))
            V = (np.frombuffer(in_bytes_V, np.uint8).reshape([h // 2, w // 2]))
            YUV = np.zeros((h, w, 3), dtype=np.uint8)
            # Y channel
            YUV[:, :, 0] = Y
            # U channel
            YUV[0::2, 0::2, 1] = U
            YUV[0::2, 1::2, 1] = U
            YUV[1::2, 0::2, 1] = U
            YUV[1::2, 1::2, 1] = U
            # V channel
            YUV[0::2, 0::2, 2] = V
            YUV[0::2, 1::2, 2] = V
            YUV[1::2, 0::2, 2] = V
            YUV[1::2, 1::2, 2] = V
            YUV = YUV / 255.
            frame_buffer.append(YUV)
        imgs_LQ = np.stack(frame_buffer, axis=0)
        imgs_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs_LQ, (0, 3, 1, 2)))).float()

        # process each image
        writer = (
            ffmpeg.input('pipe:', format='rawvideo', pix_fmt='yuv420p', s='{}x{}'.format(w, h), r=fps)
                  .output(output_path)
                  .overwrite_output()
                  .run_async(pipe_stdin=True)
        )

        for img_idx, _ in enumerate(range(imgs_LQ.size(0))):

            select_idx = data_util.index_generation_with_scene_list(img_idx, imgs_LQ.size(0), N_in,
                                                                    scene_dict[video_name],
                                                                    padding=padding)
            imgs_in = imgs_LQ.index_select(0, torch.LongTensor(select_idx)).to(device)
            imgs_in = imgs_in.unsqueeze(0)

            # # inference once by using padding
            frames = torch.zeros(1, N_in, 3, 1088, 1920).to(device)
            frames[:, :, 0, :, :] =  16. / 255.
            frames[:, :, 1, :, :] = 128. / 255.
            frames[:, :, 2, :, :] = 128. / 255.
            frames[:, :, :, 4:-4, :] = imgs_in
            frames_Y = frames[:, :, 0, :, :].unsqueeze(2)
            frames_UV = frames[:, :, 1:, 0::2, 0::2]
            if flip_test:
                # output_Y, output_UV = util.flipx4_forward_split(model, frames_Y, frames_UV)
                output_Y, output_UV = util.flipx2_forward_split(model, frames_Y, frames_UV)
            else:
                output_Y, output_UV = util.single_forward_split(model, frames_Y, frames_UV)
            output_Y = output_Y[:, :, 4:-4, :]
            output_UV = output_UV[:, :, 2:-2, :]
            output_Y = util.tensor2img(output_Y, reverse_channel=False)
            output_UV = util.tensor2img(output_UV, reverse_channel=False)
            writer.stdin.write(output_Y.tobytes())
            writer.stdin.write(output_UV[:, :, 0].tobytes())
            writer.stdin.write(output_UV[:, :, 1].tobytes())

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
            # output = output.astype(np.float32)
            # Y = output[:, :, 0]
            # U = (output[0::2, 0::2, 1] + output[0::2, 1::2, 1] + output[1::2, 0::2, 1] + output[1::2, 1::2, 1]) / 4
            # V = (output[0::2, 0::2, 2] + output[0::2, 1::2, 2] + output[1::2, 0::2, 2] + output[1::2, 1::2, 2]) / 4
            # Y, U, V = Y.astype(np.uint8), U.astype(np.uint8), V.astype(np.uint8)
            # writer.stdin.write(Y.tobytes())
            # writer.stdin.write(U.tobytes())
            # writer.stdin.write(V.tobytes())

        seq_id += step

        reader.stdout.close()
        writer.stdin.close()
        writer.wait()


if __name__ == '__main__':
    # testing with single gpu:
    main(gpu_id='0', start_id=0, step=1)

    # manually switch gpu and use 4 gpus in testing:
    # main(gpu_id='0', start_id=0, step=4)
    # main(gpu_id='1', start_id=1, step=4)
    # main(gpu_id='2', start_id=2, step=4)
    # main(gpu_id='3', start_id=3, step=4)