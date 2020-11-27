import os.path as path
import glob

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from collections import OrderedDict

import numpy as np
import cv2
import utils.util as util
from models.archs.EDVR_arch import EDVR as EDVR
from models.archs.SRResNet_arch import MSRResNet as MSRResNet


class SingleFrameEnhancer(object):

    def __init__(self, model_arch, weight_path, device_id=0):
        super(SingleFrameEnhancer, self).__init__()

        self.device_id = device_id

        if model_arch == 'MSSResNet':
            self.net = MSRResNet(in_nc=3, out_nc=3, nf=64, nb=16, upscale=2)
        else:
            raise Exception("Unknown model_arch({}), please select from [MSSResNet]".format(model_arch))

        self._load_network(weight_path)

        self.net = self.net.cuda(device=device_id)
        self.net.eval()

    def _load_network(self, weight_path):
        print('loading weights from {}...'.format(weight_path))

        if isinstance(self.net, nn.DataParallel) or isinstance(self.net, DistributedDataParallel):
            self.net = self.net.module
        load_net = torch.load(weight_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        self.net.load_state_dict(load_net_clean, strict=True)

    def forward_on_folder(self, input_folder, save_folder):
        """
            do inference on a list of image of the input_folder,
            and save the results in the save_folder
            Args:
                input_folder (str): a folder in which input images are stored
                save_folder  (str): a folder to save output images
        """
        util.mkdir(save_folder)
        img_list = sorted(glob.glob(path.join(input_folder, '*.png')))
        print('processing images from {}...'.format(input_folder))

        for img_path in img_list:
            print('processing: {}'.format(img_path))
            img = cv2.imread(img_path)
            img_name = path.basename(img_path)

            sr_img = self.forward_single(img)

            # save img
            cv2.imwrite(path.join(save_folder, img_name), sr_img[:, :, [2, 1, 0]])

        print('sr processing done!')

    def forward_single(self, img):
        """
            do inference on a single input image
            Args:
                img (np.uint8): input image with BGR format
            Returns:
                out (np.uint8): output image with RGB format
        """
        with torch.no_grad():
            input = torch.from_numpy(img)
            input = input.cuda(device=self.device_id)
            input = input.unsqueeze(0).permute(0, 3, 1, 2)[:, [2, 1, 0], :, :]  # BGR2RGB
            input = input.float()
            input = input / 255.

            out = self.net(input)
            out = out * 255.
            out = out.to(torch.uint8).squeeze()
            out = out.permute(1, 2, 0)
            out = out.cpu().numpy()

        return out

    def forward_batch(self, imgs):
        with torch.no_grad():
            imgs = np.stack(imgs)
            input = torch.from_numpy(imgs)
            input = input.cuda(device=self.device_id)
            input = input.permute(0, 3, 1, 2)[:, [2, 1, 0], :, :]
            input = input.float()
            input = input / 255.

            out = self.net(input)
            out = out * 255.0
            out = out.to(torch.uint8)
            out = out.permute(0, 2, 3, 1)
            img_np = out.cpu().numpy()

        return img_np

    def convert_onnx(self, save_onnx_file):
        # Convert
        self.net.eval()
        input = torch.ones(1, 3, 1080, 1920)
        torch.onnx.export(self.net.cpu(), input, save_onnx_file, verbose=True)


class MultiFrameEnhancer(object):
    def __init__(self, model_arch, weight_path, nframes=5, device_id=0):
        super(MultiFrameEnhancer, self).__init__()
        self.nframes = nframes
        self.device_id = device_id

        if model_arch == 'EDVR':
            self.net = EDVR(128, self.nframes, 8, 5, 20, predeblur=False, HR_in=True, w_TSA=True)

        else:
            raise Exception("Unknown model_arch({}), please select from [EDVR]".format(model_arch))

        self._load_network(weight_path)

        self.net = self.net.cuda(device=device_id)
        self.net.eval()

        # for cache multiframe input
        self.input_cache = []
        self.input_pool_len = 0

    def _load_network(self, weight_path):
        print('loading weights from {}...'.format(weight_path))

        if isinstance(self.net, nn.DataParallel) or isinstance(self.net, DistributedDataParallel):
            self.net = self.net.module
        load_net = torch.load(weight_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        self.net.load_state_dict(load_net_clean, strict=True)

    def clear_input_cache(self):
        self.input_cache.clear()
        self.input_pool_len = 0

    def get_pool_length(self):
        return self.input_pool_len

    def sequence_input_pool(self, img):
        """
            store input image as gpu tensor buffer for accelerating
            use before invoke FUNC. [forward_sequence]
            Args:
                img (np.uint8): input image with RGB HWC format
        """
        with torch.no_grad():
            tensor = torch.from_numpy(img)
            tensor = tensor.cuda(device=self.device_id)
            tensor = tensor.permute(2, 0, 1)  # HWC2CHW
            tensor = tensor.float()
            tensor = tensor / 255.
            self.input_cache.append(tensor)

        self.input_pool_len += 1

    def forward_sequence(self, idxs):
        """
            do inference on a single input image
            Args:
                idxs (list<int>): input image idx
            Returns:
                out (np.uint8): output image with RGB format
        """
        with torch.no_grad():
            input = torch.stack([self.input_cache[v] for v in idxs], dim=0)  # TCHW
            input = input.unsqueeze(0).cuda(device=self.device_id)  # NTCHW
            input[input < (16. / 255.)] = 16. / 255.
            # frames = torch.zeros(1, self.nframes, 3, 1088, 1920).cuda(device=self.device_id)
            # frames[:, :, 0, :, :] =  16. / 255.
            # frames[:, :, 1, :, :] = 128. / 255.
            # frames[:, :, 2, :, :] = 128. / 255.
            # frames[:, :, :, 4:-4, :] = input

            out = self.net(input)
            out = out * 255.
            out = out.to(torch.uint8).squeeze()
            out = out.permute(1, 2, 0)
            out = out.cpu().numpy()

            # out = out[4:-4, :, :]

        return out


if __name__ == '__main__':
    pass