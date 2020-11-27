import os
import os.path as osp
import sys
import glob

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import utils.util as util  # noqa: E402


if __name__ == '__main__':

    root = '/data'

    # # download
    # link = 'link.txt'
    # util.mkdir(root)
    # with open(link, 'r') as f:
    #     lines = f.readlines()
    # for line in lines:
    #     os.system("wget -P {} {}".format(root, line))
    #
    # # unzip
    # paths = sorted(glob.glob(os.path.join(root, '*.zip')))
    # for path in paths:
    #     command = 'unzip {} -d {}'.format(path, root)
    #     print(command)
    #     os.system(command)
    
    # move
    util.mkdir(os.path.join(root, 'LQ'))
    util.mkdir(os.path.join(root, 'GT'))

    # lq_name = [
    #     'train_damage_part1',
    #     'train_damage_part2',
    #     'train_damage_part3',
    #     'train_damage_part4',
    #     'train_damage_part5',
    #     'train_damage_part6',
    #     'val_damage_part1',
    #     'val_damage_part2'
    # ]
    lq_name = [
        'train_damage',
        'val_damage'
    ]

    for lq in lq_name:
        seq_paths = sorted(glob.glob(os.path.join(root, lq, '*.y4m')))
        for seq_path in seq_paths:
            dst_path = os.path.join(root, 'LQ')
            command = 'mv {} {}'.format(seq_path, dst_path)
            print(command)
            os.system(command)

    # gt_name = [
    #     'train_ref_part1',
    #     'train_ref_part2',
    #     'train_ref_part3',
    #     'train_ref_part4',
    #     'train_ref_part5',
    #     'train_ref_part6',
    #     'val_ref_part1',
    #     'val_ref_part2'
    # ]
    gt_name = [
        'train_ref',
        'val_ref'
    ]

    for gt in gt_name:
        seq_paths = sorted(glob.glob(os.path.join(root, gt, '*.y4m')))
        for seq_path in seq_paths:
            dst_path = os.path.join(root, 'GT')
            command = 'mv {} {}'.format(seq_path, dst_path)
            print(command)
            os.system(command)
    
    # rename
    src_paths = sorted(glob.glob(os.path.join(root, 'LQ', '*.y4m')))
    for src_path in src_paths:
        name = os.path.basename(src_path).split('.')[0].split('_')[2]
        dst_path = os.path.join(root, 'LQ', '{}.y4m'.format(name))
        command = 'mv {} {}'.format(src_path, dst_path)
        print(command)
        os.system(command)
    
    src_paths = sorted(glob.glob(os.path.join(root, 'GT', '*.y4m')))
    for src_path in src_paths:
        name = os.path.basename(src_path).split('.')[0].split('_')[2]
        dst_path = os.path.join(root, 'GT', '{}.y4m'.format(name))
        command = 'mv {} {}'.format(src_path, dst_path)
        print(command)
        os.system(command)
