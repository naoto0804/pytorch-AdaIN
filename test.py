import argparse

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image

import net
from function import style_transfer
from PIL import Image


def custom_transform(size):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Scale(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image')
# parser.add_argument('--style', type=str,
#                     help='File path to the style image, or multiple style \
#                     images separated by commas if you want to do style \
#                     interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')

# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--gpu', action='store_true',
                    help='Zero-indexed ID of the GPU to use')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# Advanced options
# parser.add_argument('--preserve_color', action='store_true',
#                     help='If specified, \
#                         preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
# parser.add_argument(
#     '--style_interpolation_weights', type=str, default='',
#     help='The weight for blending the style of multiple style images')

args = parser.parse_args()

# Either --content or --contentDir should be given.
assert (args.content or args.content_dir)
# Either --style or --styleDir should be given.
assert (args.style or args.style_dir)

if not os.path.exists(args.output):
    os.mkdir(args.output)

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))

vgg = nn.Sequential(*list(vgg.children())[:31])
if args.gpu:
    decoder.cuda()
    vgg.cuda()

content_transform = custom_transform(args.content_size)
style_transform = custom_transform(args.style_size)

if args.content:
    content_paths = [args.content]
else:
    content_paths = [os.path.join(args.content_dir, f) for f in
                     os.listdir(args.content_dir)]

if args.style:
    # style_paths = args.style.split(',')
    # if len(style_paths) == 1:
    style_paths = [args.style]
else:
    style_paths = [os.path.join(args.style_dir, f) for f in
                   os.listdir(args.style_dir)]

for content_path in content_paths:
    for style_path in style_paths:
        content = content_transform(Image.open(content_path)).unsqueeze(0)
        style = style_transform(Image.open(style_path)).unsqueeze(0)
        if args.gpu:
            style = style.cuda()
            content = content.cuda()
        content = Variable(content, volatile=True)
        style = Variable(style, volatile=True)

        output = style_transfer(vgg, decoder, content, style, args.alpha).data
        if args.gpu:
            output = output.cpu()

        output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
            args.output, os.path.splitext(os.path.basename(content_path))[0],
            os.path.splitext(os.path.basename(style_path))[0], args.save_ext
        )
        save_image(output, output_name)
