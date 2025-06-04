from argparse import ArgumentParser
import torch
import torch.nn.parallel
import torch.optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from model import PCnet
from lib.dataset import MegaDepthDataset
from loss import loss_function

# Setup
def get_parser():
    parser = ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--dataset_path', type=str, default='/media/hmq/DATA_Maoqing_Hu/deep_matching/MegaDepth/', help='data root')
    parser.add_argument('--scene_info_path', type=str, default='/media/hmq/DATA_Maoqing_Hu/deep_matching/MegaDepth/scene_info/', help='scene info path')
    parser.add_argument('--use_validation', type=bool, default=False, help='')
    parser.add_argument('--preprocessing', type=str, default='caffe', help='image preprocessing (caffe or torch)')
    parser.add_argument('--det_radius', type=float, default=3, help='detected point nums')
    parser.add_argument('--dist_threshold', type=int, default=[1, 6], help='coarse to fine: dist threshold')
    parser.add_argument('--num_workers', type=int, default=8, help='data loader workers')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--base_lr', type=float, default=2.5e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=30, help='training epochs')
    parser.add_argument('--start_epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--print_freq', type=int, default=1, help='print frequency (default: 10)')
    parser.add_argument('--save_step', type=int, default=1, help='model save step (default: 10)')
    parser.add_argument('--valid_freq', type=int, default=1, help='valid frequency')
    parser.add_argument('--save_path', type=str, default='model_path/test', help='model and summary save path')
    parser.add_argument('--resume', type=str, default=None, help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume_epoch', type=int, default='12', help='resume epoch (default: none)')
    parser.add_argument('--weight', type=str, default='', help='path to weight (default: none)')
    return parser


def train(train_loader, model, optimizer, epoch):

    model.train()
    with torch.enable_grad():

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for index, batch_data in progress_bar:

            current_iter = (epoch - 1) * len(train_loader) + index + 1

            det_loss, des_loss, fine_loss, coarse_loss, no_match_loss, score_loss, reli_loss = loss_function(model, batch_data, args.det_radius, args.dist_threshold, device)

            if det_loss == []:
                continue

            loss = det_loss + des_loss + score_loss + reli_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (index + 1) % args.print_freq == 0:
                print('Epoch: [{}/{}][{}/{}] '
                      'det_loss {det_loss:.8f} '
                      'score_loss {score_loss:.6f} '
                      'des_loss {des_loss:.6f} '
                      'reli_loss {reli_loss:.6f} '
                      'fine_loss {fine_loss:.6f} '
                      'coarse_loss {coarse_loss:.6f} '
                      'no_match_loss {no_match_loss:.6f} '.format(epoch, args.epochs, index + 1, len(train_loader),
                                                        det_loss=det_loss,
                                                        score_loss=score_loss,
                                                        des_loss=des_loss,
                                                        reli_loss=reli_loss,
                                                        fine_loss=fine_loss,
                                                        coarse_loss=coarse_loss,
                                                        no_match_loss=no_match_loss))

            writer.add_scalar('det_loss', det_loss, current_iter)
            writer.add_scalar('des_loss', des_loss, current_iter)
            writer.add_scalar('score_loss', score_loss, current_iter)
            writer.add_scalar('reli_loss', reli_loss, current_iter)
            writer.add_scalar('loss', loss, current_iter)

    print('Train result at epoch [{}/{}].'.format(epoch, args.epochs))


def validation(valid_loader, model, epoch):

    model.eval()
    with torch.no_grad():

        valid_losses, valid_det_losses, valid_des_losses, valid_fine_losses, valid_coarse_losses, valid_no_match_losses, valid_score_losses, valid_reli_losses = 0, 0, 0, 0, 0, 0, 0, 0
        progress_bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
        batch = len(valid_loader)
        for index, batch_data in progress_bar:

            valid_det_loss, valid_des_loss, valid_fine_loss, valid_coarse_loss, valid_no_match_loss, valid_score_loss, valid_reli_loss = loss_function(model, batch_data, args.det_radius, args.dist_threshold, device)

            if valid_det_loss == []:
                batch -= 1
                continue

            valid_loss = valid_det_loss + valid_des_loss + valid_score_loss + valid_reli_loss

            valid_losses += valid_loss
            valid_det_losses += valid_det_loss
            valid_des_losses += valid_des_loss
            valid_fine_losses += valid_fine_loss
            valid_coarse_losses += valid_coarse_loss
            valid_no_match_losses += valid_no_match_loss
            valid_score_losses += valid_score_loss
            valid_reli_losses += valid_reli_loss

            if (index + 1) % args.print_freq == 0:
                print('Epoch: [{}/{}][{}/{}] '
                      'valid_det_loss {valid_det_loss:.8f} '
                      'valid_score_loss {valid_score_loss:.6f} '
                      'valid_des_loss {valid_des_loss:.6f} '
                      'valid_reli_loss {valid_reli_loss:.6f} '
                      'valid_fine_loss {valid_fine_loss:.6f} '
                      'valid_coarse_loss {valid_coarse_loss:.6f} '
                      'valid_no_match_loss {valid_no_match_loss:.6f} '.format(epoch, args.epochs, index + 1, len(valid_loader),
                                                        valid_det_loss=valid_det_loss,
                                                        valid_score_loss=valid_score_loss,
                                                        valid_des_loss=valid_des_loss,
                                                        valid_reli_loss=valid_reli_loss,
                                                        valid_fine_loss=valid_fine_loss,
                                                        valid_coarse_loss=valid_coarse_loss,
                                                        valid_no_match_loss=valid_no_match_loss))

        valid_loss = valid_losses / batch
        valid_det_loss = valid_det_losses / batch
        valid_des_loss = valid_des_losses / batch
        valid_score_loss = valid_score_losses / batch
        valid_reli_loss = valid_reli_losses / batch



        writer.add_scalar('valid_loss', valid_loss, epoch)
        writer.add_scalar('valid_det_loss', valid_det_loss, epoch)
        writer.add_scalar('valid_des_loss', valid_des_loss, epoch)
        writer.add_scalar('valid_reli_loss', valid_reli_loss, epoch)
        writer.add_scalar('valid_score_loss', valid_score_loss, epoch)

    print('Eval result at epoch [{}/{}].'.format(epoch, args.epochs))


def main():
    global args, writer, device
    args = get_parser().parse_args()
    device = torch.device('cuda:0')
    writer = SummaryWriter(args.save_path)
    print(args)
    print("=> creating model ...")

    torch.manual_seed(49)
    torch.cuda.manual_seed(69)
    model = PCnet().to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location='cuda:0')['state_dict'])

    optimizer = torch.optim.Adam(
        [{'params': model.parameters()},
         ],
        lr=args.base_lr, betas=(0.9, 0.999))

    # Dataset
    if args.use_validation:
        valid_dataset = MegaDepthDataset(
            scene_list_path='megadepth_utils/valid_scenes.txt',
            scene_info_path=args.scene_info_path,
            base_path=args.dataset_path,
            train=False,
            preprocessing='valid',
            pairs_per_scene=10
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        valid_dataset.build_dataset()

    train_dataset = MegaDepthDataset(
        scene_list_path='megadepth_utils/test_scenes.txt',
        scene_info_path=args.scene_info_path,
        base_path=args.dataset_path,
        preprocessing='train',
        pairs_per_scene=100
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    if args.resume:
        start_epoch = args.resume_epoch
    else:
        start_epoch = args.start_epoch
    for epoch in range(start_epoch, args.epochs + 1):
        train_dataset.build_dataset()
        train(train_loader, model, optimizer, epoch)

        if epoch % args.save_step == 0:
            print('Start to save checkpoint.')
            filename_model = args.save_path + '/epoch_' + str(epoch) + '_model.pth'
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename_model)

        if args.use_validation:
            if epoch % args.valid_freq == 0:
                print('Start to validate')
                validation(valid_loader, model, epoch)

    writer.close()


if __name__ == '__main__':
    main()

