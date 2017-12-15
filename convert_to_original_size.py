import argparse
import os
import time

from chainercv.transforms import resize
from chainercv.utils import read_image
from chainercv.utils import write_image

parser = argparse.ArgumentParser()
parser.add_argument('domain', choices=['clipart', 'watercolor', 'comic'])
parser.add_argument('--option', default='')
args = parser.parse_args()

translation = 'photo2{:s}'.format(args.domain)
result_root = '/tmp/adain_{:s}{:s}'.format(translation, args.option)
dev_root = '/home/inoue/data/VOCdevkit'
years = [2007, 2012]

result_files = os.listdir(result_root)
if len(result_files) == 0:
    print('No files to be converted')
    exit()

total_time = 0
for i, f in enumerate(result_files):
    content_id, style = f.rsplit('_', 1)
    if len(content_id.split('_')) == 2:
        year = 2012  # like 2008_000001
    else:
        year = 2007  # like 009965
    style_id, _ = style.split('.')

    original_root = os.path.join(dev_root, 'VOC{:d}'.format(year))
    translated_root = os.path.join(
        dev_root,
        'VOC{:d}_adain_{:s}{:s}'.format(year, translation, args.option))
    _, H, W = read_image(os.path.join(
        original_root, 'JPEGImages', '{:s}.jpg'.format(content_id))).shape

    infile = os.path.join(result_root, f)
    outfile = os.path.join(translated_root, 'JPEGImages',
                           '{:s}_{:s}.jpg'.format(content_id, style_id))

    img = read_image(infile)  # numpy array, CHW, [0~255)
    img = resize(img, (H, W))
    write_image(img, outfile)
    if i > 0 and i % 10 == 0:
        print(i, time.time() - total_time)
        total_time = time.time()
