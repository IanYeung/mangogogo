from __future__ import print_function
import math
import numpy as np

import cv2
import requests
from PIL import Image
from io import BytesIO
from scenedetect.platform import tqdm

# PySceneDetect Library Imports
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.scene_manager import SceneManager

try:
    from shotflow.util_function.tool import image_url_to_array

    USE_SHOTFLOW_READER = True
except ImportError:
    USE_SHOTFLOW_READER = False

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


class MySceneManager(SceneManager):

    def __init__(self, input_mode='video', *args, **kwargs):
        super(MySceneManager, self).__init__(*args, **kwargs)

        self._is_video = True if input_mode == 'video' else False

    def _pil_loader(self, img_url):
        # open path as file to avoid ResourceWarning
        if 'http' in img_url:
            response = requests.get(img_url, timeout=2)
            img_file = BytesIO(response.content)
        else:
            img_file = img_url

        img = Image.open(img_file).resize((64, 64))
        img = img.convert('RGB')
        return np.asarray(img)

    def _cv2_loader(self, img_url):
        # open path as file to avoid ResourceWarning
        if 'http' in img_url:
            resp = urllib.request.urlopen(img_url, timeout=2)
            img_file = np.asarray(bytearray(resp.read()), dtype="uint8")
            img = cv2.imdecode(img_file, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(img_url, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError('Fail to read {}'.format(img_url))
        img = cv2.resize(img, (64, 64))

        return img

    def _shotflow_loader(self, img_url):
        img = image_url_to_array(img_url)
        if img is None:
            raise ValueError('Fail to read {}'.format(img_url))
        img = cv2.resize(img, (64, 64))
        return img

    def _loader(self, img_url):
        if USE_SHOTFLOW_READER:
            return self._shotflow_loader(img_url)
        else:
            # return self._pil_loader(img_url)
            return self._cv2_loader(img_url)

    def detect_scenes(self, inputs, step=1, end_time=None, show_progress=True):
        frame_skip = step - 1
        if frame_skip > 0 and self._stats_manager is not None:
            raise ValueError('Step must be 1 when using a StatsManager.')

        start_frame, curr_frame, end_frame = 0, 0, None

        if self._is_video:
            total_frames = math.trunc(inputs.get(cv2.CAP_PROP_FRAME_COUNT))

            start_time = inputs.get(cv2.CAP_PROP_POS_FRAMES)
            if isinstance(start_time, FrameTimecode):
                start_frame = start_time.get_frames()
            elif start_time is not None:
                start_frame = int(start_time)

            if isinstance(end_time, FrameTimecode):
                end_frame = end_time.get_frames()
            elif end_time is not None:
                end_frame = int(end_time)

            if end_frame is not None:
                total_frames = end_frame
        else:
            start_time = None
            total_frames = len(inputs)
            end_frame = total_frames

        self._start_frame = start_frame
        curr_frame = start_frame

        if start_frame is not None and not isinstance(start_time, FrameTimecode):
            total_frames -= start_frame

        if total_frames < 0:
            total_frames = 0

        progress_bar = None
        if tqdm and show_progress:
            progress_bar = tqdm(total=total_frames, unit='frames')
        try:

            while True:
                if end_frame is not None and curr_frame >= end_frame:
                    break

                if (self._is_processing_required(self._num_frames + start_frame)
                        or self._is_processing_required(self._num_frames + start_frame + 1)):
                    try:
                        if self._is_video:
                            ret_val, frame_im = inputs.read()
                        else:
                            ret_val, frame_im = True, self._loader(inputs[curr_frame])
                    except:
                        curr_frame += 1
                        self._num_frames += 1
                        continue
                else:
                    ret_val = inputs.grab() if self._is_video else False
                    frame_im = None

                if not ret_val:
                    break

                self._process_frame(self._num_frames + start_frame, frame_im)

                curr_frame += 1
                self._num_frames += 1

                if progress_bar:
                    progress_bar.update(1)

                if frame_skip > 0:
                    for _ in range(frame_skip):
                        is_skip = inputs.grab() if self._is_video else not curr_frame >= end_frame
                        if not is_skip:
                            break
                        curr_frame += 1
                        self._num_frames += 1
                        if progress_bar:
                            progress_bar.update(1)

            self._post_process(curr_frame)

            num_frames = curr_frame - start_frame

        finally:

            if progress_bar:
                progress_bar.close()

        return num_frames


if __name__ == '__main__':
    import scenedetect
    from scenedetect.video_manager import VideoManager
    from scenedetect.frame_timecode import FrameTimecode
    from scenedetect.stats_manager import StatsManager
    from scenedetect.detectors import ContentDetector

    ####################################################
    # # 1.video input test
    # scene_manager = MySceneManager(input_mode='video')
    #
    # video_manager = VideoManager(['/home/xiyang/Downloads/VideoEnhance/train_ref/mg_train_0000_ref.y4m'])
    # video_manager.set_downscale_factor(1)
    # video_manager.start()
    #
    # # Add ContentDetector algorithm (constructor takes detector options like threshold).
    # scene_manager.add_detector(ContentDetector())
    # base_timecode = video_manager.get_base_timecode()
    #
    # scene_manager.detect_scenes(video_manager, step=1)
    # print(scene_manager._get_cutting_list())

    ####################################################
    # # 2.images inputs
    # scene_manager = MySceneManager(input_mode='images')
    #
    # scene_manager.add_detector(ContentDetector())
    #
    # images = []
    # scene_manager.detect_scenes(images, step=1)
    # print(scene_manager._get_cutting_list())

    # # Choice 2: python interface of scenedetect library
    # import scenedetect
    # from scenedetect.video_manager import VideoManager
    # from scenedetect.frame_timecode import FrameTimecode
    # from scenedetect.stats_manager import StatsManager
    # from scenedetect.detectors import ContentDetector
    # from data_scripts.scene_detect import MySceneManager

    import glob
    import pickle
    import os.path as osp

    root = '/home/xiyang/Datasets/MGTV/test_damage_B_y4m'
    file_paths = sorted(glob.glob(osp.join(root, '*.y4m')))

    scene_dict = dict()
    for file_path in file_paths:
        file_name = osp.basename(file_path).split('.')[0]
        video_manager = VideoManager([file_path])
        stats_manager = StatsManager()
        scene_manager = MySceneManager(input_mode='video')
        # Add ContentDetector algorithm (constructor takes detector options like threshold).
        scene_manager.add_detector(ContentDetector())
        base_timecode = video_manager.get_base_timecode()

        try:
            video_manager.set_downscale_factor(1)
            video_manager.start()
            scene_manager.detect_scenes(video_manager, step=1)
            scene_list = scene_manager.get_scene_list(base_timecode)
            video_scene_list = []
            for i, scene in enumerate(scene_list):
                video_scene_list.append(scene[0].get_frames())
                if i + 1 == len(scene_list):
                    video_scene_list.append(scene[1].get_frames())
            scene_dict[file_name] = video_scene_list
            print('{}.y4m: {}'.format(file_name, video_scene_list))
        finally:
            video_manager.release()

    print(scene_dict)

    save_dict = True
    save_path = '../../keys/scene_index_test_B.pkl'
    if save_dict:
        with open(save_path, 'wb') as f:
            pickle.dump(scene_dict, f)
