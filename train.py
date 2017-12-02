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


def train_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Scale(size))
    if crop:
        transform_list.append(transforms.RandomSizedCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


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
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')

# preprocessing options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--resume', help='If specified')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=2)
# parser.add_argument('--target_content_layer', default='relu4_1')
# parser.add_argument('--target_style_layers',
#                     default='relu1_1,relu2_1,relu3_1,relu4_1')
parser.add_argument('--tv_weight', type=float, default=0.0,
                    help='Weight of TV loss')
parser.add_argument('--style_weight', type=float, default=1e-2)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--recon_style', action='store_true',
                    help='If specified, the decoder is also trained to \
                                                    reconstruct style images')
parser.add_argument('--normalize', action='store_true', help='If specified, \
                                gradients at the loss function are normalized')
parser.add_argument('--n_threads', type=int, default=2)
args = parser.parse_args()

# Either --content or --contentDir should be given.
assert (os.path.exists(args.content_dir) and os.path.exists(args.style_dir))
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

decoder = net.decoder
vgg = net.vgg

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])
network = net.Net(vgg, decoder)
network.cuda()

content_tf = train_transform(args.content_size, args.crop)
style_tf = train_transform(args.style_size, args.crop)

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

optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

for i in range(args.max_iter):
    print(i)
    content_images = Variable(next(content_iter).cuda())
    style_images = Variable(next(style_iter).cuda())
    optimizer.zero_grad()
    loss_c, loss_s = network(content_images, style_images)
    loss = args.content_weight * loss_c + args.style_weight * loss_s
    loss.backward()
    optimizer.step()

    if (i + 1) % 10000 == 0:  # save
        torch.save(
            {'iter': i + 1, 'state_dict': net.state_dict()},
            '{:s}/snapshot_iter_{:d}.pth.tar'.format(args.save_dir, i + 1)
        )
