import argparse

import numpy as np
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image

import net
from function import adaptive_instance_normalization
import time


def test_transform():
    return transforms.Compose(
        [transforms.Scale((512, 512)), transforms.ToTensor()])


def style_transfer(vgg, decoder, content, style, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


parser = argparse.ArgumentParser()
parser.add_argument('--content_dir', type=str, required=True,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, required=True,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--image_per_content', type=int, default=10)
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
args = parser.parse_args()

content_paths = [os.path.join(args.content_dir, f) for f in
                 os.listdir(args.content_dir)]
style_paths = [os.path.join(args.style_dir, f) for f in
               os.listdir(args.style_dir)]

if not os.path.exists(args.output):
    os.mkdir(args.output)

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.cuda()
decoder.cuda()

content_tf = test_transform()
style_tf = test_transform()

N_style = len(style_paths)

total_time = 0
for cnt, content_path in enumerate(content_paths):
    # one content image, N style image
    style_inds_cands = np.random.randint(low=0, high=N_style - 1,
                                         size=args.image_per_content)
    i = 0
    while True:
        inds = style_inds_cands[i * args.batch_size:(i + 1) * args.batch_size]
        style = torch.stack(
            [style_tf(Image.open(style_paths[ind]).convert('RGB')) for ind in
             inds])
        content = content_tf(Image.open(content_path)) \
            .unsqueeze(0).expand_as(style)
        output = style_transfer(vgg, decoder,
                                Variable(content.cuda(), volatile=True),
                                Variable(style.cuda(), volatile=True),
                                args.alpha).data
        output = output.cpu()
        for j, ind in enumerate(inds):
            output_name = '{:s}/{:s}_{:s}{:s}'.format(
                args.output,
                splitext(basename(content_path))[0],
                splitext(basename(style_paths[ind]))[0], args.save_ext)
            save_image(output[j], output_name)
        i += 1
        if i * args.batch_size > len(style_inds_cands):
            break
    if cnt > 0 and cnt % 10 == 0:
        print(cnt, time.time() - total_time)
        total_time = time.time()
