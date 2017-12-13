import argparse
import os
import subprocess

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('domain')
parser.add_argument('--option', default='')
args = parser.parse_args()

translation = 'photo2{:s}'.format(args.domain)
result_root = '/home/inoue/tmp/adain_{:s}{:s}'.format(translation, args.option)
dev_root = '/home/inoue/data/VOCdevkit'
years = [2007, 2012]

result_files = os.listdir(result_root)
if len(result_files) == 0:
    print('No files to be converted')
    exit()

for i, f in enumerate(result_files):
    content_id, style = f.rsplit('_', 1)
    if len(content_id.split('_')) == 2:
        year = 2012  # like 2008_000001
    else:
        year = 2007  # like 009965
    style_id, _ = style.split('.')

    original_root = os.path.join(dev_root, 'VOC{:d}'.format(year))
    translated_root = os.path.join(
        dev_root, 'VOC{:d}_adain_{:s}'.format(year, translation, args.option))
    W, H = Image.open(os.path.join(
        original_root, 'JPEGImages', '{:s}.jpg'.format(content_id))).size

    infile = os.path.join(result_root, f)
    outfile = os.path.join(translated_root, 'JPEGImages',
                           '{:s}_{:s}.jpg'.format(content_id, style_id))

    # cmd = "convert -verbose "
    cmd = "convert "
    cmd += "{:s} -resize {:d}x{:d}! {:s}".format(infile, W, H, outfile)
    subprocess.call(cmd.strip().split(" "))
    if (i + 1) % 100 == 0:
        print(i + 1)
