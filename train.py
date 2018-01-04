import argparse

import os
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

import net
from sampler import InfiniteSamplerWrapper
from tqdm import tqdm


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomSizedCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, required=True,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, required=True,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=100000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=1e-2)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=16)
args = parser.parse_args()


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

decoder = net.decoder
vgg = net.vgg

vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])
for param in vgg.parameters():
    param.requires_grad = False
network = net.Net(vgg, decoder)
network.train()
network.cuda()

content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

for i in tqdm(range(args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = Variable(next(content_iter).cuda(), requires_grad=False)
    style_images = Variable(next(style_iter).cuda(), requires_grad=False)
    optimizer.zero_grad()
    loss_c, loss_s = network(content_images, style_images)
    loss = args.content_weight * loss_c + args.style_weight * loss_s
    loss.backward()
    optimizer.step()

    if (i + 1) % 10000 == 0 or (i + 1) == args.max_iter:  # save
        torch.save(
            {'iter': i + 1, 'state_dict': net.decoder.state_dict()},
            '{:s}/decoder_iter_{:d}.pth.tar'.format(args.save_dir, i + 1)
        )
