import gc
import random
import numpy
import torch
import cv2
import math
from collections import Counter
from lib.utils import (
    grid_positions,
    upscale_positions,
    downscale_positions,
    savefig,
    imshow_image
)
from lib.exceptions import NoGradientError, EmptyTensorError


def loss_function(model, batch_data, det_radius, dist_threshold, device, des_nums=1600, scaling_steps=3):

    image1 = batch_data['image1'].to(device)
    image2 = batch_data['image2'].to(device)

    out1 = model(image1)
    out2 = model(image2)

    det_score1 = out1['det']
    det_score2 = out2['det']

    reli_score1 = out1['reli']
    reli_score2 = out2['reli']

    dense_descriptor1 = out1['des']
    dense_descriptor2 = out2['des']

    det_losses, des_losses, fine_losses, coarse_losses, no_match_losses, score_losses, reli_losses= 0, 0, 0, 0, 0, 0, 0
    batch = 0
    for idx_in_batch in range(batch_data['image1'].size(0)):

        depth1 = batch_data['depth1'][idx_in_batch].to(device)  # [h1, w1]
        intrinsics1 = batch_data['intrinsics1'][idx_in_batch].to(device)  # [3, 3]
        pose1 = batch_data['pose1'][idx_in_batch].view(4, 4).to(device)  # [4, 4]
        bbox1 = batch_data['bbox1'][idx_in_batch].to(device)  # [2]

        depth2 = batch_data['depth2'][idx_in_batch].to(device)
        intrinsics2 = batch_data['intrinsics2'][idx_in_batch].to(device)
        pose2 = batch_data['pose2'][idx_in_batch].view(4, 4).to(device)
        bbox2 = batch_data['bbox2'][idx_in_batch].to(device)

        image_path1 = batch_data['image_path1'][idx_in_batch]
        image_path2 = batch_data['image_path2'][idx_in_batch]

        score_map1 = det_score1[idx_in_batch]
        score_map2 = det_score2[idx_in_batch]

        reli_map1 = reli_score1[idx_in_batch]
        reli_map2 = reli_score2[idx_in_batch]

        dense1 = dense_descriptor1[idx_in_batch]
        dense2 = dense_descriptor2[idx_in_batch]

        h1, w1 = score_map1.size()
        h2, w2 = score_map2.size()

        # Warp the positions from image 1 to image 2
        f_pos1 = grid_positions(h1, w1, device)
        pos1 = upscale_positions(f_pos1, scaling_steps=scaling_steps)
        try:
            pos1, pos2, ids = warp(pos1, depth1, intrinsics1, pose1, bbox1, depth2, intrinsics2, pose2, bbox2)
        except EmptyTensorError:
            print('Warp EmptyTensorError')
            continue

        f_pos1 = f_pos1[:, ids]
        # Skip the pair if not enough GT correspondences are available
        all_nums = f_pos1.shape[1]
        if all_nums < 128:
            print('Low overlaps')
            continue

        f_pos2 = torch.round(downscale_positions(pos2, scaling_steps=scaling_steps)).long()
        det_loss, score_loss = det_loss_func_ext(score_map1, score_map2, det_radius, f_pos1, f_pos2, device)

        # if all_nums > des_nums:
        #     rand_idx = torch.randint(low=0, high=all_nums, size=[des_nums], device=device)
        #     pos1 = pos1[:, rand_idx]
        #     pos2 = pos2[:, rand_idx]

        des_loss, fine_loss, coarse_loss, no_match_loss = des_loss_func(image_path1, image_path2, dense1, dense2, f_pos1, f_pos2, dist_threshold, device)
        reli_loss = reli_loss_func(dense1, dense2, reli_map1, reli_map2, device)

        det_losses += det_loss
        des_losses += des_loss
        fine_losses += fine_loss
        coarse_losses += coarse_loss
        no_match_losses += no_match_loss
        score_losses += score_loss
        reli_losses += reli_loss
        batch += 1

    if batch == 0:
        return [], [], [], [], [], [], []

    det_loss = det_losses / batch
    des_loss = des_losses / batch
    fine_loss = fine_losses / batch
    coarse_loss = coarse_losses / batch
    no_match_loss = no_match_losses / batch
    score_loss = score_losses / batch
    reli_loss = reli_losses / batch

    return det_loss, des_loss, fine_loss, coarse_loss, no_match_loss, score_loss, reli_loss
    pass


def reli_loss_func(dense1, dense2, reli_map1, reli_map2, device, patch=4):
    h1, w1 = reli_map1.size()
    h2, w2 = reli_map2.size()
    shape1 = (torch.tensor([h1, w1]).to(device)).unsqueeze(1).float()
    shape2 = (torch.tensor([h2, w2]).to(device)).unsqueeze(1).float()

    pos1 = grid_positions(h1, w1, device)
    pos2 = grid_positions(h2, w2, device)

    score1 = get_reli_loss(dense1, reli_map1, pos1, patch, shape1)
    score2 = get_reli_loss(dense2, reli_map2, pos2, patch, shape2)

    reli_loss = (score1 + score2) / 2
    return reli_loss
    pass


def get_reli_loss(dense, reli_map, pos, patch, shape, es_nums=128):
    all_nums = pos.shape[1]
    neibor_pos = torch.tensor([]).to(device=dense.device)
    for i in range(-patch, patch+1, 1):
        for j in range(-patch, patch+1, 1):
            if i != 0 or j != 0:
                temp = torch.cat([(pos[0] + i).unsqueeze(0), (pos[1] + j).unsqueeze(0)], dim=0)
                neibor_pos = torch.cat([neibor_pos, temp.unsqueeze(0)], dim=0)

    rand_idx = torch.randint(low=0, high=all_nums, size=[es_nums], device=dense.device)
    neibor_pos = neibor_pos[:, :, rand_idx]
    estimate_map = torch.tensor([]).to(device=dense.device)
    for k in range(es_nums):
        temp_pos = (neibor_pos[:, :, k]).T
        idx = (temp_pos >= 0) * (temp_pos <= (shape - 1))
        idx = torch.prod(idx, dim=0) == 1
        temp_pos = temp_pos[:, idx]
        temp_dense = dense[:, temp_pos[0].long(), temp_pos[1].long()].detach()
        kps_dense = dense[:, pos[0, k].long(), pos[1, k].long()].detach()
        sim = torch.matmul(kps_dense.unsqueeze(0), temp_dense)
        sim = (1 - sim.mean()).unsqueeze(0)
        estimate_map = torch.cat([estimate_map, sim], dim=0)

    reli_map = reli_map.view(all_nums)
    reli_map = reli_map[rand_idx]
    func_mean = torch.nn.L1Loss(reduction='mean')
    reli_loss = func_mean(reli_map, estimate_map)
    return reli_loss
    pass


def neg_loss(dense1, dense2, p1, p2, grid):
    des1 = dense1[:, p1[0].long(), p1[1].long()]
    all_p2 = get_no_match_coor(p2, grid, 0)
    all_des2 = dense2[:, all_p2[0].long(), all_p2[1].long()]

    all_neg_loss = torch.matmul(des1.unsqueeze(0), all_des2)
    neg_loss = torch.max(all_neg_loss)
    return neg_loss


def des_loss_func(image_path1, image_path2, dense1, dense2, fine_match_pos1, fine_match_pos2, dist_threshold, device, fine_coarse=0.3, fine_nomatch=0.7):
    # dim1, h1, w1 = dense1.shape
    # shape1 = (torch.tensor([h1, w1]).to(device)).unsqueeze(1).float()
    # dim2, h2, w2 = dense1.shape
    # shape2 = (torch.tensor([h2, w2]).to(device)).unsqueeze(1).float()
    num_point = fine_match_pos1.shape[1]

    Sim = torch.tensor([]).to(device)
    coarse_losses, no_match_losses = torch.tensor([]).to(device), torch.tensor([]).to(device)
    for np in range(num_point):
        fine_match_p1 = fine_match_pos1[:, np]
        fine_match_p2 = fine_match_pos2[:, np]
        coarse_loss1, no_match_loss1 = get_coarse_nomatch_loss(fine_match_p1, fine_match_p2, fine_match_pos2, dense1, dense2, dist_threshold, device)
        coarse_loss2, no_match_loss2 = get_coarse_nomatch_loss(fine_match_p2, fine_match_p1, fine_match_pos1, dense2, dense1, dist_threshold, device)
        if coarse_loss1 != 0 and coarse_loss2 != 0:
            no_match_losses = torch.cat([no_match_losses, max(no_match_loss1, no_match_loss2).unsqueeze(0)], dim=0)
            coarse_losses = torch.cat([coarse_losses, max(coarse_loss1, coarse_loss2).unsqueeze(0)], dim=0)
        else:
            no_match_losses = torch.cat([no_match_losses, max(no_match_loss1, no_match_loss2).unsqueeze(0)], dim=0)

        dim, h, w = dense2.shape
        grid = grid_positions(h, w, device)
        all_dense2 = dense2[:, grid[0].long(), grid[1].long()]
        p1_dense1 = dense1[:, fine_match_p1[0].long(), fine_match_p1[1].long()].unsqueeze(0)
        sim_ = torch.matmul(p1_dense1, all_dense2)
        Sim = torch.cat([Sim, sim_], dim=0)

    no_match_loss = no_match_losses.mean()
    coarse_loss = coarse_losses.mean()

    fine_dense1 = dense1[:, fine_match_pos1[0].long(), fine_match_pos1[1].long()]
    fine_dense2 = dense2[:, fine_match_pos2[0].long(), fine_match_pos2[1].long()]

    fine_loss = torch.matmul(fine_dense1.T, fine_dense2)
    temp = fine_loss
    fine_loss = fine_loss.trace()
    fine_loss = fine_loss / num_point

    save_name = 'des/' + image_path1.split('/')[-1] + '+' + image_path2.split('/')[-1] + '.npz'
    # save_name = 'des.npz'
    import scipy.io as scio
    scio.savemat(save_name, {'fine_match_pos1':fine_match_pos1.detach().cpu().numpy(),
            'fine_match_pos2':fine_match_pos2.detach().cpu().numpy(),
            'Sim':Sim.detach().cpu().numpy(),
            'temp':temp.detach().cpu().numpy()})

    des_loss = max(0, 0.6 - fine_loss + coarse_loss) + max(0, 0.4 - coarse_loss + no_match_loss)

    return des_loss, fine_loss, coarse_loss, no_match_loss
    pass


def get_coarse_nomatch_loss(fine_match_p1, fine_match_p2, pos2, dense1, dense2, dist_threshold, device, shape=None, func_mean=None):

    temp_dense1 = dense1[:, fine_match_p1[0].long(), fine_match_p1[1].long()]

    dim, h, w = dense2.shape
    grid = grid_positions(h, w, device)
    # coarse_losses
    # coarse_match_p2 = get_coarse_match_coor(fine_match_p2, pos2, dist_threshold)
    coarse_match_p2 = get_coarse_match_coor(fine_match_p2, grid, dist_threshold)
    coarse_dense2 = dense2[:, coarse_match_p2[0].long(), coarse_match_p2[1].long()]

    # coarse_temp_dense1 = temp_dense1.unsqueeze(0).repeat(coarse_dense2.shape[0], 1)
    if coarse_match_p2.shape[1] == 0:
        coarse_loss = 0
    else:
        # coarse_loss = func_mean(coarse_dense2, coarse_temp_dense1)
        coarse_losses = torch.matmul(temp_dense1.unsqueeze(0), coarse_dense2)
        # coarse_loss = coarse_losses.mean()
        coarse_loss = torch.max(coarse_losses)

    # no_match_losses
    # no_match_p2 = get_no_match_coor(fine_match_p2, pos2, dist_threshold)
    no_match_p2 = get_no_match_coor(fine_match_p2, grid, dist_threshold)
    no_match_dense2 = dense2[:, no_match_p2[0].long(), no_match_p2[1].long()]

    # nomatch_temp_dense1 = temp_dense1.unsqueeze(0).repeat(no_match_dense2.shape[0], 1)
    # no_match_loss = func_mean(no_match_dense2, nomatch_temp_dense1)
    no_match_losses = torch.matmul(temp_dense1.unsqueeze(0), no_match_dense2)
    # no_match_loss = no_match_losses.mean()
    no_match_loss = torch.max(no_match_losses)

    return coarse_loss, no_match_loss
    pass


def get_no_match_coor(fine_match_p, pos, dist_threshold, no_match_nums=600):
    right = dist_threshold[1]

    dist = pos - fine_match_p.unsqueeze(1)
    dist = torch.norm(dist.float(), p=2, dim=0)

    idx = (dist > right)
    no_match_list = pos[:, idx]

    # device = pos.device
    # num = no_match_list.shape[1]
    # if num > no_match_nums:
    #     random_idx = torch.randint(low=0, high=num, size=[no_match_nums], device=device)
    #     no_match_list = no_match_list[:, random_idx]

    return no_match_list
    pass


def get_coarse_match_coor(fine_match_p, pos, dist_threshold):
    left = dist_threshold[0]
    right = dist_threshold[1]

    dist = pos-fine_match_p.unsqueeze(1)
    dist = torch.norm(dist.float(), p=2, dim=0)

    idx = (dist >= left) * (dist <= right)

    coarse_match_list = pos[:, idx]
    return coarse_match_list
    pass


# local extreme values
def det_loss_func_ext(score_map1, score_map2, radius, pos1, pos2, device):
    h1, w1 = score_map1.shape
    h2, w2 = score_map2.shape
    shape1 = (torch.tensor([h1, w1]).to(device)).unsqueeze(1).float()
    shape2 = (torch.tensor([h2, w2]).to(device)).unsqueeze(1).float()

    idx = (pos1 >= radius) * (pos1 <= (shape1 - (radius + 1)))
    idx = torch.prod(idx, dim=0) == 1
    pos1 = pos1[:, idx]
    pos2 = pos2[:, idx]

    idx = (pos2 >= radius) * (pos2 <= (shape2 - (radius + 1)))
    idx = torch.prod(idx, dim=0) == 1
    pos1 = pos1[:, idx]
    pos2 = pos2[:, idx]

    neighbor_score1 = get_neighbor_score(pos1, score_map1, radius, device)
    neighbor_score2 = get_neighbor_score(pos2, score_map2, radius, device)
    # neighbor_score1 = torch.nn.functional.normalize(neighbor_score1.T, dim=1)
    # neighbor_score2 = torch.nn.functional.normalize(neighbor_score2.T, dim=1)

    func = torch.nn.L1Loss(reduction='mean')
    det_loss = func(neighbor_score1, neighbor_score2)
    # det_loss = torch.nn.functional.cosine_similarity(neighbor_score1, neighbor_score2, dim=0)
    # det_loss = 1 - det_loss.mean()

    max1 = torch.max(neighbor_score1, dim=0)[0]
    mean1 = torch.mean(neighbor_score1, dim=0)
    max2 = torch.max(neighbor_score2, dim=0)[0]
    mean2 = torch.mean(neighbor_score2, dim=0)

    score_loss = abs(0.5 - ((max1 - mean1).mean() + (max2 - mean2).mean())/2)
    # score_loss = get_score_loss(score_map1, score_map2, radius)

    return det_loss, score_loss


def get_local_extreme(score_map, radius):
    device = score_map.device
    h, w = score_map.shape
    shape = (torch.tensor([h, w]).to(device)).unsqueeze(1).float()
    pos = grid_positions(h, w, device)

    idx = (pos >= radius) * (pos <= (shape - (radius + 1)))
    idx = torch.prod(idx, dim=0) == 1
    pos = pos[:, idx]

    neighbor_score = get_neighbor_score(pos, score_map, radius, device)
    kps_score = score_map[pos[0].long(), pos[1].long()]

    n_max, _ = torch.max(neighbor_score, dim=0)
    n_min, _ = torch.min(neighbor_score, dim=0)
    id = (kps_score - n_max) * (kps_score - n_min)
    id = id > 0
    pos = pos[:, id]

    neighbor_score = neighbor_score[:, id]
    kps_score = kps_score[id]
    score = torch.abs(torch.mean(neighbor_score, dim=0) - kps_score)

    return pos, score


def get_score_loss(score_map1, score_map2, radius):
    fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=False,
                                          type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)

    array1 = cv2.normalize(score_map1.unsqueeze(2).to('cpu').detach().numpy(), None, 0, 255, cv2.NORM_MINMAX)
    image_rgb1 = cv2.cvtColor(array1, cv2.COLOR_GRAY2BGR)
    array2 = cv2.normalize(score_map2.unsqueeze(2).to('cpu').detach().numpy(), None, 0, 255, cv2.NORM_MINMAX)
    image_rgb2 = cv2.cvtColor(array2, cv2.COLOR_GRAY2BGR)

    kps1 = fast.detect(image_rgb1, None)
    kps2 = fast.detect(image_rgb2, None)

    # print('coor_point_num is {}/{}'.format(coor_point_num, total_num))

    return


def get_neighbor_score(pos, score_map, radius, device):

    neighbor_score = torch.tensor([]).to(device)
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if math.sqrt(i*i+j*j) >= math.sqrt(8) and math.sqrt(i*i+j*j) <= math.sqrt(10):
            # if math.sqrt(i * i + j * j) <= math.sqrt(10):
                temp_kps = torch.cat([(pos[0] + i).unsqueeze(0), (pos[1] + j).unsqueeze(0)], dim=0)
                temp_score = (score_map[temp_kps[0].long(), temp_kps[1].long()]).unsqueeze(0)
                neighbor_score = torch.cat([neighbor_score, temp_score], dim=0)

    temp_score = (score_map[pos[0].long(), pos[1].long()]).unsqueeze(0)
    neighbor_score = torch.cat([neighbor_score, temp_score], dim=0)

    return neighbor_score


def get_keypoints(score_map1, score_map2, det_nums, H12, shape2, device):
    point1, _, topk_val1 = filter_score_map(score_map1, det_nums)
    _, _, topk_val2 = filter_score_map(score_map2, det_nums)
    estimate_point2 = points_wrap(point1, H12)
    estimate_point2 = estimate_point2.long()

    idx = (estimate_point2 >= 0) * (estimate_point2 <= (shape2 - 1))
    idx = torch.prod(idx, dim=0) == 1

    point1 = point1[:, idx]
    estimate_point2 = estimate_point2[:, idx]

    point1_response = score_map1[point1[0].long(), point1[1].long()]
    point2_response = score_map2[estimate_point2[0], estimate_point2[1]]

    zeros = torch.zeros(point1_response.shape[0]).to(device)
    hing = (point1_response - topk_val1) * (point2_response - topk_val2)

    total_num = point1.shape[1]
    coor_point_num = torch.sum(hing > 0)
    print('coor_point_num is {}/{}'.format(coor_point_num, total_num))
    return point1, estimate_point2


def points_wrap(point1, H12):
    device = point1.device
    len = point1.shape[1]

    uv1 = torch.cat([(point1[1, :]).view(1, -1), (point1[0, :]).view(1, -1), torch.ones(1, len).to(device)], dim=0)

    H12 = torch.from_numpy(H12).to(device).float()
    estimate_uv2 = H12 @ uv1
    estimate_uv2 = estimate_uv2[:2, :] / estimate_uv2[2:, :]

    estimate_point2 = uv_to_pos(estimate_uv2)
    return estimate_point2
    pass


def filter_score_map(score_map, det_nums):

    device = score_map.device
    h, w = score_map.size()
    score_map = score_map.view(h * w)
    values, indices = score_map.topk(det_nums, dim=0, largest=True)
    topk_val = values[-1]

    pos = grid_positions(h, w, device)
    kps = pos[:, indices]

    return kps, topk_val


def interpolate_depth(pos, depth):
    device = pos.device

    ids = torch.arange(0, pos.size(1), device=device)

    h, w = depth.size()

    i = pos[0, :]
    j = pos[1, :]

    # Valid corners
    i_top_left = torch.floor(i).long()
    j_top_left = torch.floor(j).long()
    valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

    i_top_right = torch.floor(i).long()
    j_top_right = torch.ceil(j).long()
    valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

    i_bottom_left = torch.ceil(i).long()
    j_bottom_left = torch.floor(j).long()
    valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

    i_bottom_right = torch.ceil(i).long()
    j_bottom_right = torch.ceil(j).long()
    valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

    valid_corners = torch.min(
        torch.min(valid_top_left, valid_top_right),
        torch.min(valid_bottom_left, valid_bottom_right)
    )

    i_top_left = i_top_left[valid_corners]
    j_top_left = j_top_left[valid_corners]

    i_top_right = i_top_right[valid_corners]
    j_top_right = j_top_right[valid_corners]

    i_bottom_left = i_bottom_left[valid_corners]
    j_bottom_left = j_bottom_left[valid_corners]

    i_bottom_right = i_bottom_right[valid_corners]
    j_bottom_right = j_bottom_right[valid_corners]

    ids = ids[valid_corners]
    if ids.size(0) == 0:
        raise EmptyTensorError

    # Valid depth
    valid_depth = torch.min(
        torch.min(
            depth[i_top_left, j_top_left] > 0,
            depth[i_top_right, j_top_right] > 0
        ),
        torch.min(
            depth[i_bottom_left, j_bottom_left] > 0,
            depth[i_bottom_right, j_bottom_right] > 0
        )
    )

    i_top_left = i_top_left[valid_depth]
    j_top_left = j_top_left[valid_depth]

    i_top_right = i_top_right[valid_depth]
    j_top_right = j_top_right[valid_depth]

    i_bottom_left = i_bottom_left[valid_depth]
    j_bottom_left = j_bottom_left[valid_depth]

    i_bottom_right = i_bottom_right[valid_depth]
    j_bottom_right = j_bottom_right[valid_depth]

    ids = ids[valid_depth]
    if ids.size(0) == 0:
        raise EmptyTensorError

    # Interpolation
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    interpolated_depth = (
        w_top_left * depth[i_top_left, j_top_left] +
        w_top_right * depth[i_top_right, j_top_right] +
        w_bottom_left * depth[i_bottom_left, j_bottom_left] +
        w_bottom_right * depth[i_bottom_right, j_bottom_right]
    )

    pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

    return [interpolated_depth, pos, ids]


def filter_depth(pos, depth):
    device = pos.device

    ids = torch.arange(0, pos.size(1), device=device)

    h, w = depth.size()

    i = pos[0, :]
    j = pos[1, :]

    i_long = torch.floor(i).long()
    j_long = torch.floor(j).long()

    # Valid depth
    valid_depth = depth[i_long, j_long] > 0

    ids = ids[valid_depth]
    if ids.size(0) == 0:
        raise EmptyTensorError

    # filter
    i = i[ids]
    j = j[ids]

    i_long = i_long[ids]
    j_long = j_long[ids]

    filter_depth = (depth[i_long, j_long])

    pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

    return [filter_depth, pos, ids]


def uv_to_pos(uv):
    return torch.cat([uv[1, :].view(1, -1), uv[0, :].view(1, -1)], dim=0)


def upscale_positions(pos, scaling_steps=0):
    for _ in range(scaling_steps):
        pos = pos * 2 + 0.5
    return pos


def downscale_positions(pos, scaling_steps=0):
    for _ in range(scaling_steps):
        pos = (pos - 0.5) / 2
    return pos


def warp(
        pos1,
        depth1, intrinsics1, pose1, bbox1,
        depth2, intrinsics2, pose2, bbox2
):
    device = pos1.device

    Z1, pos1, ids = interpolate_depth(pos1, depth1)

    # COLMAP convention  !!!!!!!!!!!  u1 = pos1[1, :]???
    # central_match = np.array([
    #     point2D1[1], point2D1[0],
    #     point2D2[1], point2D2[0]
    # ])
    # bbox1_i = max(int(central_match[0]) - self.image_size // 2, 0)
    # if bbox1_i + self.image_size >= image1.shape[0]:
    #     bbox1_i = image1.shape[0] - self.image_size
    u1 = pos1[1, :] + bbox1[1] + .5
    v1 = pos1[0, :] + bbox1[0] + .5

    X1 = (u1 - intrinsics1[0, 2]) * (Z1 / intrinsics1[0, 0])
    Y1 = (v1 - intrinsics1[1, 2]) * (Z1 / intrinsics1[1, 1])

    # camera1 coordinates
    XYZ1_hom = torch.cat([
        X1.view(1, -1),
        Y1.view(1, -1),
        Z1.view(1, -1),
        torch.ones(1, Z1.size(0), device=device)
    ], dim=0)

    # camera2 coordinates
    XYZ2_hom = torch.chain_matmul(pose2, torch.inverse(pose1), XYZ1_hom)
    XYZ2 = XYZ2_hom[: -1, :] / XYZ2_hom[-1, :].view(1, -1)

    uv2_hom = torch.matmul(intrinsics2, XYZ2)
    uv2 = uv2_hom[: -1, :] / uv2_hom[-1, :].view(1, -1)

    # coordinates of uv1 reflect to camera2
    u2 = uv2[0, :] - bbox2[1] - .5
    v2 = uv2[1, :] - bbox2[0] - .5
    uv2 = torch.cat([u2.view(1, -1),  v2.view(1, -1)], dim=0)

    # uv_to_pos(uv2) !!!!! x,y trap, why?  depth(y,x)???
    annotated_depth, pos2, new_ids = interpolate_depth(uv_to_pos(uv2), depth2)

    ids = ids[new_ids]
    pos1 = pos1[:, new_ids]
    estimated_depth = XYZ2[2, new_ids]

    inlier_mask = torch.abs(estimated_depth - annotated_depth) < 0.05

    ids = ids[inlier_mask]
    if ids.size(0) == 0:
        raise EmptyTensorError

    pos2 = pos2[:, inlier_mask]
    pos1 = pos1[:, inlier_mask]

    return pos1, pos2, ids


