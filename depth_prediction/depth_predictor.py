import cv2
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm

from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import load_ckpt

def parse_args():
    parser = argparse.ArgumentParser(description='Configs for LeReS')
    parser.add_argument('--load_ckpt', default='./res101.pth', help='Checkpoint path to load')
    parser.add_argument('--backbone', default='resnext101', help='Checkpoint name to load')
    
    parser.add_argument('--input_path', default='./images/', help='Path to input images')
    parser.add_argument('--output_path', default='./outputs/', help='Path to output depth images')
    
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)

    args = parser.parse_args()
    return args


scale_torch_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225))
])

def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        img = scale_torch_transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img


if __name__ == '__main__':
    args = parse_args()

    # create depth model
    depth_model = RelDepthModel(backbone=args.backbone)
    depth_model.eval()

    # load checkpoint
    load_ckpt(args, depth_model, None, None)
    depth_model.cuda()

    # get img names
    image_dir = args.input_path
    imgs_list = os.listdir(image_dir)
    imgs_path = [os.path.join(image_dir, name) for name in imgs_list if name.lower().endswith((".png", ".jpeg", ".jpg"))]
    
    # creating output directory if not exists
    image_dir_out = args.output_path
    os.makedirs(image_dir_out, exist_ok=True)

    for i, v in enumerate(tqdm(imgs_path)):
        if args.verbose:
            print('processing (% 4d)-th image... %s' % (i, v))
        rgb = cv2.cvtColor(cv2.imread(v), cv2.COLOR_BGR2RGB)
        A_resize = cv2.resize(rgb, (448, 448))

        img_torch = scale_torch(A_resize)[None, :, :, :]
        pred_depth = depth_model.inference(img_torch).cpu().numpy().squeeze()
        pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))

        img_name = v.split('/')[-1]
        img_name_no_ext = os.path.splitext(img_name)[0]
        
        # save depth
        plt.imsave(os.path.join(image_dir_out, img_name_no_ext + '.png'), pred_depth_ori/pred_depth_ori.max(), cmap='gray')
