import os
import glob


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    # download
    root = '/data/yangxi/MGTV'
    link = '/data/yangxi/MGTV/link.txt'

    with open(link, 'r') as f:
        lines = f.readlines()
    for line in lines:
        os.system("wget {}".format(line))

    # unzip
    paths = sorted(glob.glob(os.path.join(root, '*.zip')))
    for path in paths:
        command = 'unzip {}'.format(path)
        print(command)
        os.system(command)
    
    # move
    mkdir(os.path.join(root, 'LQ'))
    mkdir(os.path.join(root, 'GT'))

    lq_name = [
        'train_damage_part1', 
        'train_damage_part2', 
        'train_damage_part3',
        'train_damage_part4', 
        'train_damage_part5', 
        'train_damage_part6',
        'val_damage_part1',
        'val_damage_part2'
    ]
    
    for lq in lq_name:
        seq_paths = sorted(glob.glob(os.path.join(root, lq, '*.y4m')))
        for seq_path in seq_paths:
            dst_path = os.path.join(root, 'LQ')
            command = 'mv {} {}'.format(seq_path, dst_path)
            print(command)
            os.system(command)

    gt_name = [
        'train_ref_part1', 
        'train_ref_part2', 
        'train_ref_part3',
        'train_ref_part4', 
        'train_ref_part5', 
        'train_ref_part6',
        'val_ref_part1',
        'val_ref_part2'
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
    
    
