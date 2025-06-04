import argparse
import os

import cv2
import numpy
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pydegensac

import viz
#from evaluations.HPatch_match import adapt_resize, recover_scale
from lib.utils import preprocess_image
from loss import get_local_extreme
from model import PCnet

def adapt_resize(image, down_sample=1):
    h = image.height
    w = image.width

#    if max(h, w) > 1000:  # True
#        scale = 1
#    else:
#        scale = 1.5  # 1.5
    scale = 1
    h_ = h * scale
    w_ = w * scale
    H = int((int(h_ / 8) - 1) / down_sample) * 8
    W = int((int(w_ / 8) - 1) / down_sample) * 8
    image = image.resize((W, H), Image.BILINEAR)
    sx = w / W
    sy = h / H

    return image, sx, sy


def recover_scale(kp, sx, sy):
    kpx = kp[:, 0] * sx
    kpy = kp[:, 1] * sy

    kps = [kpx, kpy]
    kps = (np.array(kps)).T
    return kps


def mutual_nn_matching_torch(desc1, desc2, threshold=None, eps=1e-9):
    if len(desc1) == 0 or len(desc2) == 0:
        return torch.empty((0, 2), dtype=torch.int64), torch.empty((0, 2), dtype=torch.int64)

    device = desc1.device
    # desc1 = desc1 / (desc1.norm(dim=1, keepdim=True) + eps)
    # desc2 = desc2 / (desc2.norm(dim=1, keepdim=True) + eps)
    similarity = torch.einsum('id, jd->ij', desc1, desc2)

    nn12 = similarity.max(dim=1)[1]
    nn21 = similarity.max(dim=0)[1]
    ids1 = torch.arange(0, similarity.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]]).t()
    scores = similarity.max(dim=1)[0][mask]
    if threshold:
        mask = scores > threshold
        matches = matches[mask]
        scores = scores[mask]
    return matches, scores


def mutual_nn_matching(desc1, desc2, threshold=None):
    if isinstance(desc1, np.ndarray):
        desc1 = torch.from_numpy(desc1)
        desc2 = torch.from_numpy(desc2)
    matches, scores = mutual_nn_matching_torch(desc1, desc2, threshold=threshold)
    return matches.cpu().numpy(), scores.cpu().numpy()


def upscale_positions(pos, scaling_steps=0):
    for _ in range(scaling_steps):
        pos = pos * 2 + 0.5
    return pos


def get_kps_dens(score_map1, dense_descriptor1, reli_score1, det_threshold):
    array1 = cv2.normalize(score_map1.unsqueeze(2).to('cpu').detach().numpy(), None, 0, 255, cv2.NORM_MINMAX)
    image_rgb1 = cv2.cvtColor(array1, cv2.COLOR_GRAY2BGR)

    fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=False,
                                          type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    kps1 = fast.detect(image_rgb1, None)

    if len(kps1) == 0:
        return None, None

    kp1 = torch.tensor([]).to(device)
    for i in range(len(kps1)):
        p = kps1[i].pt
        p = numpy.array(p)
        p = torch.from_numpy(p).to(device).unsqueeze(1)
        kp1 = torch.cat([kp1, p], dim=1)

    am_kp = kp1
    am_kp = upscale_positions(am_kp, scaling_steps=3)
    am_kp = am_kp.T.to('cpu').detach().numpy()

    kp_reli = reli_score1[kp1[1].long(), kp1[0].long()]
    idx = kp_reli > det_threshold
    kp1 = kp1[:, idx]

    dens1 = (dense_descriptor1[:, kp1[1].long(), kp1[0].long()]).T
    kp1 = upscale_positions(kp1, scaling_steps=3)
    kp1 = kp1.T.to('cpu').detach().numpy()

    return kp1, dens1, am_kp


def norm(map):
    min = torch.min(map)
    max = torch.max(map)
    map = (map-min) / (max-min)
    return map


def refine(matches, dense_descriptor1, dense_descriptor2, radius=8):
    pairs = matches.shape[0]
    kp1 = matches[:, :2]
    kp2 = matches[:, 2:]

    upsample = torch.nn.Upsample(scale_factor=8, mode='bicubic', align_corners=True)
    dense_descriptor1 = upsample(dense_descriptor1.unsqueeze(0))
    dense_descriptor2 = upsample(dense_descriptor2.unsqueeze(0))

    neighbor_kp1, neighbor_kp2 = [], []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            temp_kps1 = (np.concatenate([np.expand_dims((kp1[:, 0] + i), axis=0), np.expand_dims((kp1[:, 1] + j), axis=0)], axis=0)).T
            neighbor_kp1.append(temp_kps1)

            temp_kps2 = (np.concatenate([np.expand_dims((kp2[:, 0] + i), axis=0), np.expand_dims((kp2[:, 1] + j), axis=0)], axis=0)).T
            neighbor_kp2.append(temp_kps2)

    neighbor_kp1 = np.array(neighbor_kp1)
    neighbor_kp2 = np.array(neighbor_kp2)

    rf_kps1, rf_kps2, match_scores = [], [], []
    for id in range(pairs):
        # if scores[id] > 0.8:
        #     rf_kps1.append(kp1[id, :])
        #     rf_kps2.append(kp2[id, :])
        #     match_scores.append(np.array([scores[id]]))
        #     continue
        kps1 = neighbor_kp1[:, id, :]
        kps2 = neighbor_kp2[:, id, :]
        dens1 = (dense_descriptor1[0, :, kps1[:, 1].round(), kps1[:, 0].round()]).T
        dens2 = (dense_descriptor2[0, :, kps2[:, 1].round(), kps2[:, 0].round()]).T

        p1 = kp1[id, :]
        p2 = kp2[id, :]
        d1 = (dense_descriptor1[0, :, int(p1[1]), int(p1[0])]).unsqueeze(0)
        d2 = (dense_descriptor2[0, :, int(p2[1]), int(p2[0])]).unsqueeze(0)

        scores1 = torch.einsum('id, jd->ij', d1, dens2)
        scores2 = torch.einsum('id, jd->ij', dens1, d2)

        max1, id1 = torch.max(scores1, dim=1)
        max2, id2 = torch.max(scores2, dim=0)

        if max1 >= max2:
            rf_kps1.append(p1)
            rf_kps2.append(kps2[id1, :])
            match_scores.append(max1.to('cpu').numpy())
        else:
            rf_kps1.append(kps1[id2, :])
            rf_kps2.append(p2)
            match_scores.append(max2.to('cpu').numpy())

    rf_kps1 = np.array(rf_kps1)
    rf_kps2 = np.array(rf_kps2)
    match_scores = np.array(match_scores)

    return rf_kps1, rf_kps2, match_scores
    pass


def refine2(matches, scores, radius=3):
    idx = scores > 0.3
    fine_matches = matches[idx]
    H12, inliers12 = pydegensac.findHomography(fine_matches[:, :2], fine_matches[:, 2:4], 2)
    H21, inliers21 = pydegensac.findHomography(fine_matches[:, 2:4], fine_matches[:, :2], 2)

    pairs = matches.shape[0]
    kp1 = matches[:, :2]
    kp2 = matches[:, 2:]

    neighbor_kp1, neighbor_kp2 = [], []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            temp_kps1 = (
                np.concatenate([np.expand_dims((kp1[:, 0] + i), axis=0), np.expand_dims((kp1[:, 1] + j), axis=0)],
                               axis=0)).T
            neighbor_kp1.append(temp_kps1)

            temp_kps2 = (
                np.concatenate([np.expand_dims((kp2[:, 0] + i), axis=0), np.expand_dims((kp2[:, 1] + j), axis=0)],
                               axis=0)).T
            neighbor_kp2.append(temp_kps2)

    neighbor_kp1 = np.array(neighbor_kp1)
    neighbor_kp2 = np.array(neighbor_kp2)

    rf_kps1, rf_kps2 = [], []
    for id in range(pairs):
        if scores[id] > 0.8:
            rf_kps1.append(kp1[id, :])
            rf_kps2.append(kp2[id, :])
            continue

        p1 = kp1[id, :]
        p2 = kp2[id, :]
        kps1 = neighbor_kp1[:, id, :]
        kps2 = neighbor_kp2[:, id, :]
        kps1_pred = np.concatenate([kps1, np.ones(((2*radius+1)*(2*radius+1), 1))], axis=1)
        kps2_pred = np.concatenate([kps2, np.ones(((2*radius+1)*(2*radius+1), 1))], axis=1)

        warped_kps1 = (np.matmul(H12, kps1_pred.T)).T
        warped_kps1 = warped_kps1[:, :2] / warped_kps1[:, 2:]
        dist1 = np.linalg.norm(p2 - warped_kps1, axis=1)

        warped_kps2 = (np.matmul(H21, kps2_pred.T)).T
        warped_kps2 = warped_kps2[:, :2] / warped_kps2[:, 2:]
        dist2 = np.linalg.norm(p1 - warped_kps2, axis=1)

        min1, id1 = np.min(dist1), np.argmin(dist1)
        min2, id2 = np.min(dist2), np.argmin(dist2)

        if min1 <= min2:
            rf_kps1.append(kps1[id1, :])
            rf_kps2.append(p2)
        else:
            rf_kps1.append(p1)
            rf_kps2.append(kps2[id2, :])

    rf_kps1 = np.array(rf_kps1)
    rf_kps2 = np.array(rf_kps2)

    return rf_kps1, rf_kps2, scores
    pass


if __name__ == '__main__':
    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Argument parsing
    parser = argparse.ArgumentParser(description='Feature extraction script')

    parser.add_argument(
        '--image_list_file', type=str, default='/home/hmq/Desktop/rgb_test/pairs.txt',
        help='path to a file containing a list of images to process'
    )

    parser.add_argument(
        '--preprocessing', type=str, default='caffe',
        help='image preprocessing (caffe or torch)'
    )

    parser.add_argument(
        '--model_file', type=str, default='model_path/model.pth',
        help='path to the full model'
    )

    parser.add_argument(
        '--det_threshold', type=int, default=0.8,
        help='point nums'
    )

    parser.add_argument(
        '--det_radius', type=int, default=3,
        help='det radius'
    )

    parser.add_argument(
        '--score', type=float, default=0.6,
        help='score'
    )

    parser.add_argument(
        '--resize', type=int, default=[256, 256],
        help='image resize'
    )

    parser.add_argument(
        '--output_dir', type=str, default='/home/hmq/Desktop/rgb_test/out_put',
        help='extension for the output'
    )

    parser.add_argument(
        '--output_type', type=str, default='npz',
        help='output file type (npz or mat)'
    )

    parser.set_defaults(use_relu=True)

    args = parser.parse_args()

    model = PCnet().to(device)
    model.load_state_dict(torch.load(args.model_file, map_location='cuda:0')['state_dict'])
    model = model.eval()

    with open(args.image_list_file, 'r') as f:
        pairs = [l.split() for l in f.readlines()]

    for i, pair in tqdm(enumerate(pairs)):
        img_path1, img_path2 = pair[:2]

        name0 = img_path1.split('/')[-1]
        name1 = img_path2.split('/')[-1]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        # matches_path = args.output_dir + str(i) + '{}_{}_matches.npz'.format(stem0, stem1)
        # viz_path = args.output_dir + str(i) + '{}_{}_matches.{}'.format(stem0, stem1, 'jpg')
        matches_path = args.output_dir + '/{}_{}_matches.npz'.format(stem0, stem1)
        viz_path = args.output_dir + '/{}_{}_matches.{}'.format(stem0, stem1, 'jpg')

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        o_rgb_image1 = Image.open(img_path1)
        # rgb_image1.save(args.output_dir + str(i) + '_rgb1.jpg')
        if o_rgb_image1.mode != 'RGB':
            o_rgb_image1 = o_rgb_image1.convert('RGB')
        rgb_image1, sx1, sy1 = adapt_resize(o_rgb_image1, 1)
        rgb_image1 = np.array(rgb_image1)
        image1 = preprocess_image(rgb_image1, preprocessing=args.preprocessing)
        image1 = image1.unsqueeze(0).cuda()
        # image1 = torch.tensor(image1[np.newaxis, :, :, :].astype(np.float32), device=device)

        o_rgb_image2 = Image.open(img_path2)
        # rgb_image2.save(args.output_dir + str(i) + '_rgb2.jpg')
        if o_rgb_image2.mode != 'RGB':
            o_rgb_image2 = o_rgb_image2.convert('RGB')
        rgb_image2, sx2, sy2 = adapt_resize(o_rgb_image2, 1)
        rgb_image2 = np.array(rgb_image2)
        image2 = preprocess_image(rgb_image2, preprocessing=args.preprocessing)
        image2 = image2.unsqueeze(0).cuda()
        # image2 = torch.tensor(image2[np.newaxis, :, :, :].astype(np.float32), device=device)

        with torch.no_grad():
            out1 = model(image1)
            out2 = model(image2)

            score_map1 = norm(out1['det'].squeeze(0))
            score_map2 = norm(out2['det'].squeeze(0))

            score_img1 = (score_map1.to('cpu').detach().numpy() * 255)
            score_img2 = (score_map2.to('cpu').detach().numpy() * 255)
            score_path1 = args.output_dir + str(i) + '{}_score.{}'.format(stem0, 'jpg')
            score_path2 = args.output_dir + str(i) + '{}_score.{}'.format(stem1, 'jpg')
            # cv2.imwrite(score_path1, score_img1)
            # cv2.imwrite(score_path2, score_img2)

            reli_score1 = norm(out1['reli'].squeeze(0))
            reli_score2 = norm(out2['reli'].squeeze(0))

            reli_img1 = (reli_score1.to('cpu').detach().numpy() * 255)
            reli_img2 = (reli_score2.to('cpu').detach().numpy() * 255)
            reli_path1 = args.output_dir + str(i) + '{}_reli.{}'.format(stem0, 'jpg')
            reli_path2 = args.output_dir + str(i) + '{}_reli.{}'.format(stem1, 'jpg')
            # cv2.imwrite(reli_path1, reli_img1)
            # cv2.imwrite(reli_path2, reli_img2)

            dense_descriptor1 = out1['des'].squeeze(0)
            dense_descriptor2 = out2['des'].squeeze(0)

            kp1, dens1, am_kp1 = get_kps_dens(score_map1, dense_descriptor1, reli_score1, args.det_threshold)
            kp2, dens2, am_kp2 = get_kps_dens(score_map2, dense_descriptor2, reli_score2, args.det_threshold)

            # NN Match
            match_ids, scores = mutual_nn_matching(dens1, dens2)
            p1s = kp1[match_ids[:, 0], :2]
            p2s = kp2[match_ids[:, 1], :2]
            matches = np.concatenate([p1s, p2s], axis=1)

            # # bf_match
            # bf = cv2.BFMatcher()
            # matches = bf.knnMatch(np.array(dens1.cpu()), np.array(dens2.cpu()), k=2)
            #
            # good = []
            # for m, n in matches:
            #     if m.distance < 0.9 * n.distance:
            #         good.append(m)
            # p1s = np.float32([kp1[m.queryIdx] for m in good])
            # p2s = np.float32([kp2[m.trainIdx] for m in good])
            # matches = np.concatenate([p1s, p2s], axis=1)

            # coarse-to-fine
            rf_p1s, rf_p2s, rf_scores = refine(matches, dense_descriptor1, dense_descriptor2)

            rf_p1s = recover_scale(rf_p1s, sx1, sy1)
            rf_p2s = recover_scale(rf_p2s, sx2, sy2)
            rf_matches = np.concatenate([rf_p1s, rf_p2s], axis=1)

            # sorted_scores = sorted(scores, reverse=True)
            # if sorted_scores[3] > 0.45:
            #     idx = rf_scores > 0.45
            # else:
            #     idx = rf_scores >= sorted_scores[3]
            #
            # rf_matches = rf_matches[idx.squeeze(1)]
            H, mask = cv2.findHomography(rf_matches[:, 0:2], rf_matches[:, 2:4], cv2.RANSAC, 4)
            mask = mask.squeeze(1) == 1
            rf_matches = rf_matches[mask]

            #
            viz.plot_matches(o_rgb_image1, o_rgb_image2, rf_matches, radius=0.5, lines=True, sav_fig=viz_path)
            # viz.plot_images([rgb_image1, rgb_image2])
            # viz.plot_keypoints([kp1, kp2], ps=5)
            # key_path = args.output_dir + str(i) + '{}_{}_keys.{}'.format(stem0, stem1, 'jpg')
            # plt.savefig(key_path)
            plt.close()

            # viz.plot_images([rgb_image1])
            # viz.plot_keypoints([kp1], ps=5)
            # key_path = args.output_dir + str(i) + '{}_{}_1111keys.{}'.format(stem0, stem1, 'jpg')
            # plt.savefig(key_path)
            # plt.close()
            #
            # viz.plot_images([rgb_image2])
            # viz.plot_keypoints([kp2], ps=5)
            # key_path = args.output_dir + str(i) + '{}_{}_2222keys.{}'.format(stem0, stem1, 'jpg')
            # plt.savefig(key_path)
            # plt.close()
            #
            # # am_kp1 = recover_scale(am_kp1, sx1, sy1)
            # # am_kp2 = recover_scale(am_kp2, sx2, sy2)
            # viz.plot_images([rgb_image1])
            # viz.plot_keypoints([am_kp1], ps=5, colors='r')
            # key_path = args.output_dir + str(i) + '{}_{}_am111keys.{}'.format(stem0, stem1, 'jpg')
            # plt.savefig(key_path)
            # plt.close()
            #
            # viz.plot_images([rgb_image2])
            # viz.plot_keypoints([am_kp2], ps=5, colors='r')
            # key_path = args.output_dir + str(i) + '{}_{}_am222keys.{}'.format(stem0, stem1, 'jpg')
            # plt.savefig(key_path)
            # plt.close()
    pass