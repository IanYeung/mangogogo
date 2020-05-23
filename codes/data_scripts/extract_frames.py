import os
import os.path as osp
import sys
import glob
import numpy as np
import cv2
import imageio
import ffmpeg
import utils.util as util
import matplotlib.pyplot as plt


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))


def extract_frames_from_single_video(video_path, out_path, mode='png'):

    assert mode == 'npy' or mode == 'npz' or mode == 'png'
    util.mkdir(out_path)

    video = imageio.get_reader(video_path, format='ffmpeg', mode='I', dtype=np.uint8)
    fps = video.get_meta_data()['fps']
    w, h = video.get_meta_data()['size']
    video.close()
    print('fps: {}'.format(fps))
    print('h: {}'.format(h))
    print('w: {}'.format(w))

    process = (
        ffmpeg
            .input(video_path)
            .output('pipe:', format='rawvideo', pix_fmt='yuv420p')
            .run_async(pipe_stdout=True)
    )

    index = 0

    while True:
        in_bytes_Y = process.stdout.read(w * h)
        in_bytes_U = process.stdout.read(w // 2 * h // 2)
        in_bytes_V = process.stdout.read(w // 2 * h // 2)
        if not in_bytes_Y:
            break
        image_Y = (
            np
                .frombuffer(in_bytes_Y, np.uint8)
                .reshape([h, w])
        )
        image_U = (
            np
                .frombuffer(in_bytes_U, np.uint8)
                .reshape([h // 2, w // 2])
        )
        image_V = (
            np
                .frombuffer(in_bytes_V, np.uint8)
                .reshape([h // 2, w // 2])
        )
        UMatrix = np.zeros([h // 2, w], dtype=np.uint8)
        VMatrix = np.zeros([h // 2, w], dtype=np.uint8)
        UMatrix[:, 0::2] = image_U
        UMatrix[:, 1::2] = image_U
        VMatrix[:, 0::2] = image_V
        VMatrix[:, 1::2] = image_V
        YUV = np.zeros([h, w, 3], dtype=np.uint8)
        YUV[:, :, 0] = image_Y
        YUV[0::2, :, 1] = UMatrix
        YUV[1::2, :, 1] = UMatrix
        YUV[0::2, :, 2] = VMatrix
        YUV[1::2, :, 2] = VMatrix

        index += 1
        if mode == 'png':
            cv2.imwrite(osp.join(out_path, '{:03d}.png'.format(index)), YUV)
        elif mode == 'npy':
            np.save(osp.join(out_path, '{:03d}.npy'.format(index)), YUV)
        elif mode == 'npz':
            np.savez_compressed(osp.join(out_path, '{:03d}'.format(index)), img=YUV)


if __name__ == '__main__':

    # root
    src_root = '/home/xiyang/Downloads/VideoEnhance/test_damage_A'
    dst_root = '/home/xiyang/Downloads/VideoEnhance/test_damage_A_frames'

    # single videos
    video_path = '/home/xiyang/Downloads/VideoEnhance/test_damage_A/mg_test_0800_damage.y4m'
    out_path = osp.join(dst_root, osp.basename(video_path).split('.')[0])
    extract_frames_from_single_video(video_path, out_path)

    # multiple videos
    # video_paths = sorted(glob.glob(osp.join(src_root, '*.y4m')))
    # for video_path in video_paths:
    #     out_path = osp.join(dst_root, osp.basename(video_path).split('.')[0])
    #     extract_frames_from_single_video(video_path, out_path)
