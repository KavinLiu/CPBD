# coding=utf-8

from PIL import Image
import numpy as np
from skimage import feature
import math
from scipy.ndimage import correlate


def block_process(A, block):
    block_contrast = np.zeros((A.shape[0]/block[0], A.shape[1]/block[1]), dtype=np.int32)
    flatten_contrast = list()
    for i in range(0, A.shape[0], block[0]):
        for j in range(0, A.shape[1], block[1]):
            block_view = A[i:i+block[0], j:j+block[1]]
            block_view = np.max(block_view) - np.min(block_view)
            flatten_contrast.append(block_view)
    block_contrast = np.array(flatten_contrast).reshape(block_contrast.shape)
    return block_contrast


def cpbd_compute(image):
    if isinstance(image, str):
        image = Image.open(image)
        image = image.convert('L')
    img = np.array(image, dtype=np.float32)
    m, n = img.shape

    threshold = 0.002
    beta = 3.6
    rb = 64
    rc = 64
    max_blk_row_idx = int(m/rb)
    max_blk_col_idx = int(n/rc)
    widthjnb = np.array([np.append(5 * np.ones((1, 51)), 3*np.ones((1, 205)))])
    total_num_edges = 0
    hist_pblur = np.zeros(101, dtype=np.float64)
    input_image_canny_edge = feature.canny(img)

    input_image_sobel_edge = matlab_sobel_edge(img)
    width = marziliano_method(input_image_sobel_edge, img)
    # print width
    for i in range(1, max_blk_row_idx+1):
        for j in range(1, max_blk_col_idx+1):
            rows = slice(rb*(i-1), rb*i)
            cols = slice(rc*(j-1), rc*j)
            decision = get_edge_blk_decision(input_image_canny_edge[rows, cols], threshold)
            if decision == 1:
                local_width = width[rows, cols]
                local_width = local_width[np.nonzero(local_width)]
                blk_contrast = block_process(img[rows, cols], [rb, rc]) + 1
                blk_jnb = widthjnb[0, int(blk_contrast)-1]
                prob_blur_detection = 1 - math.e ** (-np.power(np.abs(np.true_divide(local_width, blk_jnb)), beta))
                for k in range(1, local_width.size+1):
                    temp_index = int(round(prob_blur_detection[k-1] * 100)) + 1
                    hist_pblur[temp_index-1] = hist_pblur[temp_index-1] + 1
                    total_num_edges = total_num_edges + 1
    if total_num_edges != 0:
        hist_pblur = hist_pblur / total_num_edges
    else:
        hist_pblur = np.zeros(hist_pblur.shape)
    sharpness_metric = np.sum(hist_pblur[0:63])
    return sharpness_metric


def marziliano_method(E, A):
    # print E
    edge_with_map = np.zeros(A.shape)
    gy, gx = np.gradient(A)
    M, N = A.shape
    angle_A = np.zeros(A.shape)
    for m in range(1, M+1):
        for n in range(1, N+1):
            if gx[m-1, n-1] != 0:
                angle_A[m-1, n-1] = math.atan2(gy[m-1,n-1], gx[m-1,n-1]) * (180/np.pi)
            if gx[m-1, n-1] == 0 and gy[m-1, n-1] == 0:
                angle_A[m-1, n-1] = 0
            if gx[m-1, n-1] == 0 and gy[m-1, n-1] == np.pi/2:
                angle_A[m-1, n-1] = 90
    if angle_A.size != 0:
        angle_Arnd = 45 * np.round(angle_A/45.0)
        # print angle_Arnd
        count = 0
        for m in range(2, M):
            for n in range(2, N):
                if E[m-1, n-1] == 1:
                    if angle_Arnd[m-1, n-1] == 180 or angle_Arnd[m-1, n-1] == -180:
                        count += 1
                        for k in range(0, 101):
                            posy1 = n-1-k
                            posy2 = n - 2 - k
                            if posy2 <= 0:
                                break
                            if A[m-1, posy2-1] - A[m-1, posy1-1] <= 0:
                                break
                        width_count_side1 = k + 1
                        for k in range(0, 101):
                            negy1 = n + 1 + k
                            negy2 = n + 2 + k
                            if negy2 > N:
                                break
                            if A[m-1, negy2-1] > A[m-1, negy1-1]:
                                break
                        width_count_side2 = k + 1
                        edge_with_map[m-1, n-1] = width_count_side1 + width_count_side2
                    elif angle_Arnd[m-1, n-1] == 0:
                        count += 1
                        for k in range(0, 101):
                            posy1 = n+1+k
                            posy2 = n + 2 + k
                            if posy2 > N:
                                break
                            # print m, posy2
                            if A[m-1, posy2-1] <= A[m-1, posy1-1]:
                                break
                        width_count_side1 = k + 1
                        for k in range(0, 101):
                            negy1 = n -1-k
                            negy2 = n -2 -k
                            if negy2 <=0:
                                break
                            if A[m-1, negy2-1] >= A[m-1, negy1-1]:
                                break
                        width_count_side2 = k + 1
                        edge_with_map[m-1, n-1] = width_count_side1 + width_count_side2
    return edge_with_map


def get_edge_blk_decision(im_in, T):
    m, n = im_in.shape
    L = m * n
    im_edge_pixels = np.sum(im_in)
    im_out = im_edge_pixels > (L * T)
    return im_out


def matlab_sobel_edge(img):
    mask = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 8.0
    bx = correlate(img, mask)
    b = bx*bx
    # print b
    b = b > 4.0
    return np.array(b, dtype=np.int)


if __name__ == '__main__':
    print cpbd_compute('F:\googleDowload\LIVE_Images_GBlur\img11.bmp')
