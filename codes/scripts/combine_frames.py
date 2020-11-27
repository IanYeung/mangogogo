import os
import os.path as osp
import sys
import glob
import cv2
import ffmpeg
import numpy as np

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import utils.util as util


def combine_frames_to_single_video(frame_path, video_path):
    pass


if __name__ == '__main__':
    # root
    src_root = '/home/xiyang/Downloads/VideoEnhance/test_damage_A_frames'
    dst_root = '/home/xiyang/Downloads/VideoEnhance/test_damage_A_videos'
    util.mkdir(dst_root)

    h, w = 1080, 1920
    fps = 25

    video_list = sorted(glob.glob(osp.join(src_root, '*')))
    for video_path in video_list:
        video_name = osp.basename(video_path)
        frame_list = sorted(glob.glob(osp.join(video_path, '*.png')))
        output_path = osp.join(dst_root, '{}.y4m'.format(video_name))

        writer = (ffmpeg
                  .input('pipe:', format='rawvideo', pix_fmt='yuv420p', s='{}x{}'.format(w, h), r=fps)
                  .output(output_path)
                  .overwrite_output()
                  .run_async(pipe_stdin=True)
                  )

        for frame_path in frame_list:
            frame_name = osp.basename(frame_path)
            data = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            data = data.astype(np.float)
            Y = data[:, :, 0]
            U = (data[0::2, 0::2, 1] + data[0::2, 1::2, 1] + data[1::2, 0::2, 1] + data[1::2, 1::2, 1]) / 4
            V = (data[0::2, 0::2, 2] + data[0::2, 1::2, 2] + data[1::2, 0::2, 2] + data[1::2, 1::2, 2]) / 4
            Y, U, V = Y.astype(np.uint8), U.astype(np.uint8), V.astype(np.uint8)
            writer.stdin.write(Y.tobytes())
            writer.stdin.write(U.tobytes())
            writer.stdin.write(V.tobytes())

        writer.stdin.close()
        writer.wait()
