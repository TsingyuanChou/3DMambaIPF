import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

def save_depth_map_to(proj_depth, name, path):
    proj_depth = proj_depth.detach().cpu().numpy()
    if not os.path.exists(path):
        os.makedirs(path)
    # print(proj_depth.shape) [views, height, width, 1]
    plt.figure(figsize=(12, 7))
    for v in range(proj_depth.shape[0]):
        plt.subplot(4, 8, v+1)
        plt.imshow(proj_depth[v])
        plt.axis('off')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=None, hspace=None)
    plt.savefig(os.path.join(path, f'depthmap-{name}.jpg'))
    plt.close()

def save_depth_map_cv2(proj_depth, name, path):
    proj_depth = proj_depth.detach().cpu().numpy()
    if not os.path.exists(path):
        os.makedirs(path)
    row = 4
    col = 8
    small_pic_size = (64, 64)

    big_picture = np.full((int(row * small_pic_size[0]), int(col * small_pic_size[1]), 1), 0, dtype=np.uint8)

    for epoch in range(proj_depth.shape[0]):
        pic = np.asarray(proj_depth[epoch])
        pic = ((pic - 0.0) / 10.0) * 255
        pic = cv2.resize(pic, small_pic_size)
        pic = np.expand_dims(pic, -1)
        block_x = int(small_pic_size[0] * (epoch // col))
        block_y = int(small_pic_size[1] * (epoch % col))
        # print(f'{block_x}-{block_x + small_pic_size[0]}, {block_y}-{block_y + small_pic_size[1]}')
        
        big_picture[block_x:block_x + small_pic_size[0], block_y:block_y + small_pic_size[1], :] = pic

    cv2.imwrite(os.path.join(path, f'depthmap-cv2-{name}.jpg'), big_picture)