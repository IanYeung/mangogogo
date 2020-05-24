import os
import os.path as osp
import sys
import glob
import pickle
import random
import pandas as pd
from collections import Counter

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import data.util as util  # noqa: E402


def save_keys(save_path, root):
    path_list, _ = util.get_image_paths('img', root)
    keys_list = []
    for path in path_list:
        name_list = path.split('/')
        keys_list.append(name_list[-2] + '_' + name_list[-1].split('.')[0])
    with open(save_path, 'wb') as f:
        pickle.dump({'keys': keys_list}, f)


def load_keys(save_path):
    with open(save_path, 'rb') as f:
        keys_list = pickle.load(f)['keys']
    return keys_list


def train_valid_split(root, num_train_seq, num_valid_seq,
                      save_train_list=False, save_valid_list=False,
                      save_path_train_list=None, save_path_valid_list=None):
    """split the sequences for training and validation"""
    seq_list = sorted(glob.glob(osp.join(root, '*')))
    seq_list = [osp.basename(seq) for seq in seq_list]
    num_total_seq = len(seq_list)
    assert num_train_seq + num_valid_seq <= num_total_seq
    print('Total number of sequences: ', num_total_seq)
    print('Number of sequences for training: ', num_train_seq)
    print('Number of sequences for validation: ', num_valid_seq)
    # shuffle the sequence list
    random.seed(1024)
    random.shuffle(seq_list)
    train_seq = seq_list[:num_train_seq]
    valid_seq = seq_list[-num_valid_seq:]
    if save_train_list:
        with open(save_path_train_list, 'wb') as f:
            pickle.dump(train_seq, f)
    if save_valid_list:
        with open(save_path_valid_list, 'wb') as f:
            pickle.dump(valid_seq, f)
    return train_seq, valid_seq


def scene_index(root, verbose=False, save_dict=False, save_path=None):
    """get the scene change index for all the sequences"""
    scene_count = Counter()
    csv_path_list = sorted(glob.glob(root))
    start_dict = {}

    for csv_path in csv_path_list:
        seq_name = osp.basename(csv_path).split('-')[0]
        csv = pd.read_csv(csv_path, skiprows=[0])
        start = []
        end = 0
        for i in range(len(csv)):
            start.append(csv.loc[i, 'Start Frame'])
            end = i
        scene_count[len(start)] += 1
        start.append(csv.loc[end, 'End Frame'])
        start_dict[seq_name] = start
        if verbose:
            print(csv_path, len(start))

    if save_dict:
        with open(save_path, 'wb') as f:
            pickle.dump(start_dict, f)

    return start_dict, scene_count


if __name__ == '__main__':
    
    root = '/data/yangxi/MGTV/GT_frames'
    
    
    #### save all keys
    save_keys('../../keys/all_keys.pkl', root)
    
    
    #### split train sequences and valid sequences
    train_valid_split(root, num_train_seq=780, num_valid_seq=20,
                      save_train_list=True, save_valid_list=True,
                      save_path_train_list='../../keys/train_seqs.pkl', 
                      save_path_valid_list='../../keys/valid_seqs.pkl')
    
    
    #### get scene information
    # Choice 1: command line interface of scenedetect library
    # # Please use scene_detect.sh to get the scene change information first
    # root = '/home/xiyang/Downloads/VideoEnhance/train_ref_scene_detect_thres35/*'
    # start_dict, scene_count = scene_index(root, verbose=True, save_dict=False, save_path=None)

    # Choice 2: python interface of scenedetect library
    import scenedetect
    from scenedetect.video_manager import VideoManager
    from scenedetect.frame_timecode import FrameTimecode
    from scenedetect.stats_manager import StatsManager
    from scenedetect.detectors import ContentDetector
    from data_scripts.scene_detect import MySceneManager

    root = '/data/yangxi/MGTV/GT'
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

    save_dict = False
    save_path = '../../keys/scene_index.pkl'
    if save_dict:
        with open(save_path, 'wb') as f:
            pickle.dump(scene_dict, f)
