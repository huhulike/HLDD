import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import torch
from PIL import Image


def checkboard(im1, im2, d=150):
    im1 = im1 * 1.0
    im2 = im2 * 1.0
    mask = np.zeros_like(im1)
    for i in range(mask.shape[0] // d + 1):
        for j in range(mask.shape[1] // d + 1):
            if (i + j) % 2 == 0:
                mask[i * d:(i + 1) * d, j * d:(j + 1) * d, :] += 1
    return im1 * mask + im2 * (1 - mask)


def image_fusion(img1_np, img2_np, solution, f_path, b_path):
    img1_np = cv2.cvtColor(img1_np, cv2.COLOR_RGB2BGR)
    img2_np = cv2.cvtColor(img2_np, cv2.COLOR_RGB2BGR)

    M1, N1, num1 = img1_np.shape
    M2, N2, num2 = img2_np.shape

    # Create a blank fusion image
    if num1 == 3 and num2 == 3:
        fusion_image = np.zeros((3 * M1, 3 * N1, num1), dtype=np.uint8)
    elif num1 == 1 and num2 == 3:
        fusion_image = np.zeros((3 * M1, 3 * N1), dtype=np.uint8)
        img2_np = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)
    elif num1 == 3 and num2 == 1:
        fusion_image = np.zeros((3 * M1, 3 * N1), dtype=np.uint8)
        img1_np = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
    elif num1 == 1 and num2 == 1:
        fusion_image = np.zeros((3 * M1, 3 * N1), dtype=np.uint8)

    # Create an identity transformation matrix
    solution_1 = np.array([[1, 0, N1], [0, 1, M1], [0, 0, 1]], dtype=np.float32)

    # Apply the transformation to the first image
    f_1 = cv2.warpPerspective(img1_np, solution_1, (3 * N1, 3 * M1))

    # Apply the transformation to the second image using the provided solution
    f_2 = cv2.warpPerspective(img2_np, solution_1 @ solution, (3 * N1, 3 * M1))

    # Find overlapping regions and blend images
    same_index = np.where((f_1 != 0) & (f_2 != 0))  # 相同区域
    index_1 = np.where((f_1 != 0) & (f_2 == 0))  # 在 f_1 中而不在 f_2 中的区域
    index_2 = np.where((f_1 == 0) & (f_2 != 0))  # 在 f_2 中而不在 f_1 中的区域

    fusion_image[same_index] = f_1[same_index] // 2 + f_2[same_index] // 2
    fusion_image[index_1] = f_1[index_1]
    fusion_image[index_2] = f_2[index_2]

    fusion_image = fusion_image.astype(np.uint8)

    # Delete redundant areas
    left_up = np.dot(solution_1 @ solution, [1, 1, 1])
    left_down = np.dot(solution_1 @ solution, [1, M2, 1])
    right_up = np.dot(solution_1 @ solution, [N2, 1, 1])
    right_down = np.dot(solution_1 @ solution, [N2, M2, 1])

    X = [left_up[0] / left_up[2], left_down[0] / left_down[2], right_up[0] / right_up[2], right_down[0] / right_down[2]]
    Y = [left_up[1] / left_up[2], left_down[1] / left_down[2], right_up[1] / right_up[2], right_down[1] / right_down[2]]

    X_min = max(int(np.floor(min(X))), 1)
    X_max = min(int(np.ceil(max(X))), 3 * N1)
    Y_min = max(int(np.floor(min(Y))), 1)
    Y_max = min(int(np.ceil(max(Y))), 3 * M1)

    if X_min > N1 + 1:
        X_min = N1 + 1
    if X_max < 2 * N1:
        X_max = 2 * N1
    if Y_min > M1 + 1:
        Y_min = M1 + 1
    if Y_max < 2 * M1:
        Y_max = 2 * M1

    if num1 == 1:
        fusion_image = fusion_image[Y_min:Y_max, X_min:X_max]
        f_1 = f_1[Y_min:Y_max, X_min:X_max]
        f_2 = f_2[Y_min:Y_max, X_min:X_max]
    elif num1 == 3:
        fusion_image = fusion_image[Y_min:Y_max, X_min:X_max, :]
        f_1 = f_1[Y_min:Y_max, X_min:X_max, :]
        f_2 = f_2[Y_min:Y_max, X_min:X_max, :]

    # save the fusion image
    cv2.imwrite(f_path, fusion_image)

    grid_num = 5  # board nun
    grid_size = min(f_1.shape[0], f_1.shape[1]) // grid_num  # board size

    f_3 = checkboard(f_1, f_2, grid_size)
    # save the board image
    cv2.imwrite(b_path, f_3)


def plot_images(imgs, titles=None, cmaps='gray', dpi=100, pad=.5,
                adaptive=True):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
    else:
        ratios = [4/3] * n
    figsize = [sum(ratios)*4.5, 4.5]
    fig, ax = plt.subplots(
        1, n, figsize=figsize, dpi=dpi, gridspec_kw={'width_ratios': ratios})
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])
    fig.tight_layout(pad=pad)


def plot_keypoints(kpts, colors='lime', ps=4, axes=None, a=1.0):
    """Plot keypoints for existing images.
    Args:
        kpts: list of ndarrays of size (N, 2).
        colors: string, or list of list of tuples (one for each keypoints).
        ps: size of the keypoints as float.
    """
    if not isinstance(colors, list):
        colors = [colors] * len(kpts)
    if not isinstance(a, list):
        a = [a] * len(kpts)
    if axes is None:
        axes = plt.gcf().axes
    for ax, k, c, alpha in zip(axes, kpts, colors, a):
        ax.scatter(k[:, 0], k[:, 1], c=c, s=ps, linewidths=0, alpha=alpha)


def plot_matches(im1, im2, matches, inliers=None, Npts=None, lines=False,
                 unnormalize=True, radius=5, dpi=150, sav_fig=False,
                 colors=None):

    # Read images and resize
    if isinstance(im1, torch.Tensor):
        im1 = im1.squeeze().permute(1, 2, 0).cpu().data.numpy()
        im2 = im2.squeeze().permute(1, 2, 0).cpu().data.numpy()

        if unnormalize:
            im1 = undo_normalize_scale(im1)
            im2 = undo_normalize_scale(im2)
        else:
            im1 = im1.astype(np.uint8)
            im2 = im2.astype(np.uint8)
        I1 = Image.fromarray(im1)
        I2 = Image.fromarray(im2)
    elif isinstance(im1, np.ndarray):
        I1 = Image.fromarray(im1)
        I2 = Image.fromarray(im2)
    elif isinstance(im1, str):
        I1 = Image.open(im1)
        I2 = Image.open(im2)
    else:
        I1 = im1
        I2 = im2

    w1, h1 = I1.size
    w2, h2 = I2.size

    if h1 <= h2:
        scale1 = 1
        scale2 = h1 / h2
        w2 = int(scale2 * w2)
        I2 = I2.resize((w2, h1))
    else:
        scale1 = h2 / h1
        scale2 = 1
        w1 = int(scale1 * w1)
        I1 = I1.resize((w1, h2))
    catI = np.concatenate([np.array(I1), np.array(I2)], axis=1)

    # Load all matches
    match_num = matches.shape[0]
    if inliers is None:
        if Npts is not None:
            Npts = Npts if Npts < match_num else match_num
        else:
            Npts = matches.shape[0]
        inliers = range(Npts)  # Everthing as an inlier
    else:
        if Npts is not None and Npts < len(inliers):
            inliers = inliers[:Npts]
    print('Plotting inliers: ', len(inliers))

    x1 = scale1 * matches[inliers, 0]
    y1 = scale1 * matches[inliers, 1]
    x2 = scale2 * matches[inliers, 2] + w1
    y2 = scale2 * matches[inliers, 3]
    c = np.random.rand(len(inliers), 3)

    if colors is not None:
        c = colors

    # Plot images and matches
    fig = plt.figure(figsize=(30, 20))
    axis = plt.gca()  # fig.add_subplot(1, 1, 1)
    axis.imshow(catI)
    axis.axis('off')

    # plt.imshow(catI)
    # ax = plt.gca()
    for i, inid in enumerate(inliers):
        # Plot
        # color = [0, 1, 0]
        axis.add_artist(plt.Circle((x1[i], y1[i]), radius=radius, color=c[i, :]))
        axis.add_artist(plt.Circle((x2[i], y2[i]), radius=radius, color=c[i, :]))
        if lines:
            axis.plot([x1[i], x2[i]], [y1[i], y2[i]], c=c[i, :], linestyle='-', linewidth=radius)
    if sav_fig:
        fig.savefig(sav_fig, dpi=dpi, bbox_inches='tight')
    # plt.show()


def undo_normalize_scale(im):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    im = im * std + mean
    im *= 255.0
    return im.astype(np.uint8)