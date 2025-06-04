import torch
import torch.nn as nn


def conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
        # nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


class Encoder(nn.Module):
    def __init__(self, backbone='res2net', layers=50, pretrained=True):
        super(Encoder, self).__init__()

        if backbone == 'res2net':
            import Res2Net.res2net as models
        else:
            raise NameError('Backbone type not defined!')

        if layers == 50:
            res2net = models.res2net50(pretrained=pretrained)

        self.layer0 = nn.Sequential(res2net.conv1, res2net.bn1, res2net.relu, res2net.maxpool)
        self.layer1, self.layer2 = res2net.layer1, res2net.layer2

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class PCnet(torch.nn.Module):
  def __init__(self, dropout=0.1):
    super(PCnet, self).__init__()
    c1, c2, c3, des_dim, det_dim, reli_dim = 64, 128, 512, 128, 1, 1

    # encoder sequential
    self.encoder = Encoder()

    # Detector Head.
    self.det_sq = nn.Sequential(
        nn.Conv2d(c3, c1, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(c1),
        nn.ReLU(inplace=True),
        nn.Conv2d(c1, det_dim, kernel_size=3, stride=1, padding=1),
    )
    self.sig = torch.nn.Sigmoid()

    # Reliability Head.
    self.reli_sq = nn.Sequential(
        nn.Conv2d(c3, c1, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(c1),
        nn.ReLU(inplace=True),
        nn.Conv2d(c1, reli_dim, kernel_size=3, stride=1, padding=1),
    )

    # Descriptor Head.
    self.des_sq = nn.Sequential(
        # GAM(c3, c3),
        nn.Conv2d(c3, c2, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(c2),
        nn.ReLU(inplace=True),
        # GAM(c2, c2),
        nn.Conv2d(c2, des_dim, kernel_size=3, stride=1, padding=1),
    )

  def forward(self, x):
    # Shared Encoder.
    x = self.encoder(x)

    det = self.det_sq(x)
    det = self.sig(det)
    det = det.squeeze(1)

    # Reliability Head.
    reli = self.reli_sq(x)
    reli = reli.squeeze(1)

    # Descriptor Head.
    des = self.des_sq(x)
    des = torch.nn.functional.normalize(des, dim=1)

    return {'det': det, 'des': des, 'reli': reli}