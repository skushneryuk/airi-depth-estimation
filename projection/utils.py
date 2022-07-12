import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def parse_calibration(path):
    with open(path) as f:
        calib = f.readlines()
        
    res = dict()    
    for line in calib:
        xs = line.strip('\n').split()
        name = xs[0][:-1]
        if name == "calib_time":
            continue
        res[name] = np.matrix([float(x) for x in xs[1:]])
    
    return res

def extract_velo_to_cam2_transform(path, cam_num=2):
    cam_to_cam = parse_calibration(os.path.join(path, "calib_cam_to_cam.txt"))
    P = cam_to_cam['P_rect_0' + str(cam_num)].reshape(3, 4)
    R0_rect = cam_to_cam['R_rect_00'].reshape(3, 3)
    
    # Add a 1 in bottom-right, reshape to 4 x 4
    R0_rect = np.insert(R0_rect, 3, values=[0,0,0], axis=0)
    R0_rect = np.insert(R0_rect, 3, values=[0,0,0,1], axis=1)
    
    velo_to_cam = parse_calibration(os.path.join(path, "calib_velo_to_cam.txt"))
    # matrix (R | T)
    Tr_velo_to_cam = np.hstack((velo_to_cam['R'].reshape(3, 3), velo_to_cam['T'].reshape(-1, 1)))
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam, 3, values=[0, 0, 0, 1], axis=0)
    
    return {
        "P": P,
        "R0_rect": R0_rect,
        "Tr_velo_to_cam": Tr_velo_to_cam
    } 

def get_cam(path):
    cam = np.fromfile(path, dtype=np.float32).reshape((-1,4))
    cam = cam[:, 0:3] # lidar xyz (front, left, up)
    cam = np.insert(cam, 3, 1, axis=1).T
    cam = np.delete(cam, np.where(cam[0, :] < 0), axis=1)
    return cam

def transfrom_cam(cam, calib):
    cam = calib['P'] * calib['R0_rect'] * calib['Tr_velo_to_cam'] * cam
    cam = np.delete(cam, np.where(cam[2, :] < 0)[1], axis=1)
    return cam

def get_img(path):
    img = mpimg.imread(path)
    if len(img.shape) < 3:
        img = np.repeat(img.reshape(img.shape[0], img.shape[1], 1), 3, -1)
    return img

def project_cam(cam, IMG_W, IMG_H):
    cam[:2] /= cam[2,:]  # cam[2]
    # get u,v,z
    u, v, z = cam

    # filter point out of canvas
    u_out = np.logical_or(u < 0, u >= IMG_W - 0.5)
    v_out = np.logical_or(v < 0, v >= IMG_H - 0.5)
    outlier = np.logical_or(u_out, v_out)
    cam = np.delete(cam, np.where(outlier), axis=1)
    return cam
    
def draw_projection(img, cam, save_path=None, show=False):
    # generate color map from depth
    IMG_H, IMG_W, _ = img.shape
    u, v, z = project_cam(cam, IMG_W, IMG_H)

    plt.figure(figsize=(12,5), dpi=96, tight_layout=True)
    plt.imshow(img)
    plt.scatter([u], [v], c=[z], cmap='rainbow_r', alpha=0.5, s=2)
    plt.axis('off')
    plt.margins(x=0)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()