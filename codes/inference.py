import argparse
import os
import cv2
import glob
import time
import numpy as np
import subprocess

import os.path as path
from enhancer import SingleFrameEnhancer, MultiFrameEnhancer
import logger
import json

import ffmpeg

import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
from scripts.scene_detect import MySceneManager


parser = argparse.ArgumentParser(description='Video Restoration')
parser.add_argument('--input_video', default='/home/xiyang/Downloads/test/lq/yuntingke.mp4')
parser.add_argument('--save_path', default='/home/xiyang/Downloads/test/hq')
parser.add_argument('--model_arch', default='EDVR')
parser.add_argument('--weight_path', default='../experiments/pretrained_models/EDVR_TSA_200000.pth')
parser.add_argument('--bitrate', type=str, default='10M')
parser.add_argument('--nframes', type=int, default=7)
parser.add_argument('--frame_buf_len', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--log_path', type=str, default='../log/inference.log')
opt = parser.parse_args()

log = logger.Logger(filename=opt.log_path, level='debug')
log.logger.info(opt)


def image_list_demo(opt):
    enhancer = SingleFrameEnhancer(opt.model_arch, opt.weight_path, device_id=opt.device_id)
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
        print('mkdir [{:s}] ...'.format(opt.save_path))  
    else:
        print('Folder [{:s}] already exists!'.format(opt.save_path))

    t1 = time.time()
    enhancer.forward_on_folder(opt.input_video, opt.save_path)
    t2 = time.time()
    print('Cost time: {:.2f}s'.format(t2-t1))


def video2video(opt):
    cap = cv2.VideoCapture(opt.input_video)
    assert cap.isOpened(), \
        '[{}] is a illegal input!'.format(opt.input_video)

    # get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # set output bitrate
    # # HD: 4Mbps，4K:15Mbps，8k:30Mbps
    # if width * 2 <= 1920:
    #     bitrate = '4000k'
    # elif width * 2 <= 3840:
    #     bitrate = '15000k'
    # else:
    #     bitrate = '30000k'
    bitrate = opt.bitrate
    fps = '%.02f' % fps
    video_name = path.basename(opt.input_video).split('.')[0]
    os.makedirs(opt.save_path, exist_ok=True)
    save_file = path.join(opt.save_path, '{}_srx2.mp4'.format(video_name))

    # use python-ffmpeg
    writer = (ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width*2, height*2), r=fps)
                .output(save_file, vcodec='libx264', pix_fmt='yuv420p', video_bitrate=bitrate, r=fps)
                .overwrite_output()
                .run_async(pipe_stdin=True)
                )

    # init model
    enhancer = SingleFrameEnhancer(opt.model_arch, opt.weight_path, device_id=opt.device_id)

    isend = False
    imgs = []
    k = 0
    t_sr = 0
    t1 = time.time()
    while True:
        k += 1
        ret, frame = cap.read()
        # video end
        if not ret or type(frame) is not np.ndarray:
            print('Finish reading video')
            if len(imgs) == 0:
                break
            isend = True
        else:
            imgs.append(frame)
            if len(imgs) < opt.batch_size:
                continue

        t_start = time.time()
        outs = enhancer.forward_batch(imgs)
        t_end = time.time()
        t_sr += t_end - t_start
        for i in range(len(imgs)):
            out = outs[i]
            writer.stdin.write(out.astype(np.uint8).tobytes())

        imgs = []
        if isend:
            break

    writer.stdin.close()
    writer.wait()
    cap.release()

    t2 = time.time()
    log.logger.info('============= Elapsed time =============')
    log.logger.info('>> Cost time: {:.2f}s'.format(t2 - t1))
    log.logger.info('>> Avg. time of processing: {}ms/frame'.format(int((t2-t1)/k*1000)))
    log.logger.info('>> Avg. time of inference : {}ms/frame'.format(int(t_sr/k*1000)))
    log.logger.info('=========================================\n')
 

## decode a video use cv2 and encode output images to a video use opencv
def video_demo(opt):
    cap = cv2.VideoCapture(opt.input_video)
    assert cap.isOpened(), \
        '[{}] is a illegal input!'.format(opt.input_video)

    # get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bitrate = opt.bitrate
    fps = '%.02f' % fps
    video_name = path.basename(opt.input_video).split('.')[0]
    os.makedirs(opt.save_path, exist_ok=True)
    save_file = path.join(opt.save_path, '{}_srx2.mp4'.format(video_name))
    tmp_file = path.join(opt.save_path, '{}_srx2_tmp.mp4'.format(video_name))

    ## get video codec info by ffprobe
    _, audio_params = get_video_info(opt.input_video)

    # use python-ffmpeg
    reader = (ffmpeg
                .input(opt.input_video)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run_async(pipe_stdout=True)
    )
    writer = (ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width*2, height*2), r=fps)
                .output(tmp_file, vcodec='libx264', pix_fmt='yuv420p', video_bitrate=bitrate, r=fps)
                .overwrite_output()
                .run_async(pipe_stdin=True)
    )

    # init model
    enhancer = SingleFrameEnhancer(opt.model_arch, opt.weight_path, device_id=opt.device_id)
    
    k = 0
    t_sr = 0
    t1 = time.time()
    while(True):
        k += 1
        in_bytes = reader.stdout.read(width * height * 3)
        if not in_bytes:
            print('Finish reading video')
            break
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])

        # inference on a frame
        t_start = time.time()
        out = enhancer.forward_single(frame)
        t_end = time.time()
        t_sr += t_end - t_start

        # write output to target video
        writer.stdin.write(out.tobytes())
    
    reader.stdout.close()
    writer.stdin.close()
    writer.wait()
    cap.release()
    
    t2 = time.time()

    # concat audio and video
    if audio_params is not None:
        log.logger.info('Concat video and audio')
        input_ffmpeg = ffmpeg.input(opt.input_video)
        audio = input_ffmpeg['a']
        output_ffmpeg = ffmpeg.input(tmp_file)
        video = output_ffmpeg['v']

        if 'bit_rate' in audio_params.keys():
            a_bitrate = audio_params['bit_rate']
        else:
            a_bitrate = '128k'

        # rewrite video with audio
        mov = (ffmpeg
                .output(video, audio, save_file, vcodec='copy', audio_bitrate=a_bitrate, acodec='aac')
                )
        mov.overwrite_output().run()
        os.system('rm -f {}'.format(tmp_file))

    else:
        os.system('mv {} {}'.format(tmp_file, save_file))

    t3 = time.time()
    log.logger.info('============= Elapsed time =============')
    log.logger.info('>> Cost time: {:.2f}s'.format(t2 - t1))
    log.logger.info('>> Avg. time of processing: {}ms/frame'.format(int((t2-t1)/k*1000)))
    log.logger.info('>> Avg. time of inference : {}ms/frame'.format(int(t_sr/k*1000)))
    log.logger.info('>> Ext. time of concating audio : {:.2f}s'.format(t3-t2))
    log.logger.info('=========================================\n')


# video forward with multi-frame mode
def video_sequence_demo(opt):
    cap = cv2.VideoCapture(opt.input_video)
    assert cap.isOpened(), \
        '[{}] is a illegal input!'.format(opt.input_video)

    # get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bitrate = opt.bitrate
    fps = '%.02f' % fps
    video_name = path.basename(opt.input_video).split('.')[0]
    os.makedirs(opt.save_path, exist_ok=True)
    save_file = path.join(opt.save_path, '{}.mp4'.format(video_name))
    tmp_file = path.join(opt.save_path, '{}_tmp.mp4'.format(video_name))

    cap.release()

    ## get video codec info by ffprobe
    _, audio_params = get_video_info(opt.input_video)

    # use python-ffmpeg
    reader = (ffmpeg
                .input(opt.input_video)
                .output('pipe:', format='rawvideo', pix_fmt='yuv420p')
                .run_async(pipe_stdout=True))
    writer = (ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='yuv420p', s='{}x{}'.format(w, h), r=fps)
                .output(tmp_file, vcodec='libx264', pix_fmt='yuv420p', video_bitrate=bitrate, r=fps)
                .overwrite_output()
                .run_async(pipe_stdin=True))

    # init model
    enhancer = MultiFrameEnhancer(opt.model_arch, opt.weight_path, nframes=opt.nframes, device_id=opt.device_id)

    k = 0
    frame_buf = []
    buf_count = 0

    t_sr = 0
    t1 = time.time()
    while True:
        k += 1
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
        # frame = YUV / 255.
        frame = YUV
        frame_buf.append(frame)
        enhancer.sequence_input_pool(frame)

        if len(frame_buf) < opt.frame_buf_len:
            continue
        else:
            ## shot det
            scene_list = shot_det(frame_buf)

            # inference on a seqnence
            for frame_idx in range(opt.frame_buf_len):
                t_start = time.time()
                select_idx = index_generation_with_scene_list(frame_idx, opt.frame_buf_len,
                                            opt.nframes, scene_list, padding='replicate')
                # img_l = []
                # for idx in select_idx:
                #     img_l.append(frame_buf[idx])
                # out = enhancer.forward_sequence(img_l)
                out = enhancer.forward_sequence(select_idx)
                t_end = time.time()
                t_sr += t_end - t_start 

                # write output to target video
                out = out.astype(np.float32)
                Y = out[:, :, 0]
                U = (out[0::2, 0::2, 1] + out[0::2, 1::2, 1] + out[1::2, 0::2, 1] + out[1::2, 1::2, 1]) / 4
                V = (out[0::2, 0::2, 2] + out[0::2, 1::2, 2] + out[1::2, 0::2, 2] + out[1::2, 1::2, 2]) / 4
                Y, U, V = Y.astype(np.uint8), U.astype(np.uint8), V.astype(np.uint8)
                writer.stdin.write(Y.tobytes())
                writer.stdin.write(U.tobytes())
                writer.stdin.write(V.tobytes())

            # clear buf
            buf_count += 1
            frame_buf.clear()
            enhancer.clear_input_cache()

    if len(frame_buf) > 0:
        ## shot det
        scene_list = shot_det(frame_buf)

        # inference on a seqnence
        for frame_idx in range(len(frame_buf)):
            t_start = time.time()
            select_idx = index_generation_with_scene_list(frame_idx, len(frame_buf),
                                        opt.nframes, scene_list, padding='replicate')
            # img_l = []
            # for idx in select_idx:
            #     img_l.append(frame_buf[idx])
            # out = enhancer.forward_sequence(img_l)
            out = enhancer.forward_sequence(select_idx)
            t_end = time.time()
            t_sr += t_end - t_start

            # write output to target video
            out = out.astype(np.float32)
            Y = out[:, :, 0]
            U = (out[0::2, 0::2, 1] + out[0::2, 1::2, 1] + out[1::2, 0::2, 1] + out[1::2, 1::2, 1]) / 4
            V = (out[0::2, 0::2, 2] + out[0::2, 1::2, 2] + out[1::2, 0::2, 2] + out[1::2, 1::2, 2]) / 4
            Y, U, V = Y.astype(np.uint8), U.astype(np.uint8), V.astype(np.uint8)
            writer.stdin.write(Y.tobytes())
            writer.stdin.write(U.tobytes())
            writer.stdin.write(V.tobytes())

        # clear buf
        buf_count += 1
        frame_buf.clear()
        enhancer.clear_input_cache()

    reader.stdout.close()
    writer.stdin.close()
    writer.wait()
    
    t2 = time.time()

    # concat audio and video
    if audio_params is not None:
        log.logger.info('Concat video and audio')
        input_ffmpeg = ffmpeg.input(opt.input_video)
        audio = input_ffmpeg['a']
        output_ffmpeg = ffmpeg.input(tmp_file)
        video = output_ffmpeg['v']

        if 'bit_rate' in audio_params.keys():
            a_bitrate = audio_params['bit_rate']
        else:
            a_bitrate = '128k'

        # rewrite video with audio
        mov = (ffmpeg
                .output(video, audio, save_file, vcodec='copy', audio_bitrate=a_bitrate, acodec='aac'))
        mov.overwrite_output().run()
        os.system('rm -f {}'.format(tmp_file))

    else:
        os.system('mv {} {}'.format(tmp_file, save_file))

    t3 = time.time()
    log.logger.info('============= Elapsed time =============')
    log.logger.info('>> Cost time: {:.2f}s'.format(t2 - t1))
    log.logger.info('>> Avg. time of processing: {}ms/frame'.format(int((t2-t1)/k*1000)))
    log.logger.info('>> Avg. time of inference : {}ms/frame'.format(int(t_sr/k*1000)))
    log.logger.info('>> Ext. time of concating audio : {:.2f}s'.format(t3-t2))
    log.logger.info('=========================================\n')


# scene detect
def shot_det(frame_buf):
    a = [0]
    scene_manager = MySceneManager(input_mode='images')
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(frame_buf, step=1)
    res = scene_manager._get_cutting_list()
    a = a + res
    a.append(len(frame_buf))
    return a


# get bitrate of video
def bitrate_of_video(video_path):
    command = 'ffprobe -v error -show_entries format=bit_rate \
                -of default=noprint_wrappers=1:nokey=1 {}'.format(video_path)
    status, bitrate = subprocess.getstatusoutput(command)
    if status == 0:
        return float(bitrate)
    else:
        return 0.0


def _encode_video_use_ffmpeg(src_path, dst_path, bitrate, fps=25):
    command = 'ffmpeg -r {} -f image2 -i {} -vcodec h264 -vf fps={} -b:v {}k -an {} -y &>/dev/null'\
        .format(fps, src_path, fps, bitrate, dst_path)
    print('doing... '+command)
    os.system(command)


def get_video_info(video_path):
    video_params = None
    audio_params = None

    command = 'ffprobe -v quiet -print_format json -show_format -show_streams {}'.format(video_path)
    status, video_info = subprocess.getstatusoutput(command)
    
    if status != 0:
        return [video_params, audio_params]

    try:
        video_info = json.loads(video_info)
    except:
        return [video_params, audio_params]

    for s in video_info['streams']:
        if s['codec_type'] == 'video':
            video_params = s
        elif s['codec_type'] == 'audio':
            audio_params = s
        else:
            pass

    return [video_params, audio_params]


def index_generation_with_scene_list(crt_i, max_n, N, scene_list, padding='replicate'):
    """Generate an index list for reading N frames from a sequence of images (with a scene list)
    Args:
        crt_i (int): current center index
        max_n (int): max number of the sequence of images (calculated from 1)
        N (int): reading N frames
        scene_list (list): scene list indicating the start of each scene, example: [0, 10, 51, 100]
        padding (str): padding mode, one of replicate
            Example: crt_i = 0, N = 5
            replicate: [0, 0, 0, 1, 2]

    Returns:
        return_l (list [int]): a list of indexes
    """
    assert max_n == scene_list[-1]
    n_pad = N // 2
    return_l = []

    num_scene = len(scene_list) - 1
    for i in range(num_scene):
        if (crt_i >= scene_list[i]) and (crt_i <= scene_list[i + 1] - 1):
            for j in range(crt_i - n_pad, crt_i + n_pad + 1):
                if j < scene_list[i]:
                    if padding == 'replicate':
                        add_idx = scene_list[i]
                    else:
                        raise ValueError('Wrong padding mode')
                elif j > (scene_list[i + 1] - 1):
                    if padding == 'replicate':
                        add_idx = scene_list[i + 1] - 1
                    else:
                        raise ValueError('Wrong padding mode')
                else:
                    add_idx = j
                return_l.append(add_idx)
    return return_l


if __name__ == '__main__':
    # invoke image list processor if input_video is a folder
    if path.isdir(opt.input_video):
        print('Input video is a folder, use img list processor.')
        image_list_demo(opt)
    # invoke video processor if input_video is a video file
    else:
        print('Input video is a file, use video processor.')
        if opt.nframes > 1:
            print('process on multi-frames mode')
            video_sequence_demo(opt)
        else:
            print('process on single-frames mode')
            video_demo(opt)



