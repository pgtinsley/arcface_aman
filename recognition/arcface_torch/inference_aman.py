import argparse

import cv2
import numpy as np
import torch

from backbones import get_model

import glob
import pickle

@torch.no_grad()
# def inference(weight, name, img):
def inference(net, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
#     net = get_model(name, fp16=False)
#     net.load_state_dict(torch.load(weight))
#     net.eval()
    feat = net(img).numpy()
#     print(feat)
    return feat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r100', help='backbone network')
    parser.add_argument('--weight', type=str, default='../../model_zoo/ms1mv3_arcface_r100_fp16/backbone.pth')
    parser.add_argument('--images', type=str, default=None, help='path to image directory (png and jpeg supported)')
    args = parser.parse_args()
    
    net = get_model(args.network, fp16=False)
    net.load_state_dict(torch.load(args.weight))
    net.eval()

    img_fnames_png  = glob.glob(args.images+'*.png')
    img_fnames_jpeg = glob.glob(args.images+'*.jpeg')
    
    img_fnames_all  = img_fnames_png + img_fnames_jpeg

    feats = {}
    for img_fname in img_fnames_all:        
        feats[img_fname] = inference(net, img_fname)
    
    with open('./feats_dict.pkl','wb') as f:
        pickle.dump(feats, f)

#     inference(args.weight, args.network, args.img)
