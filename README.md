# pytorch-AdaIN

This is an unofficial pytorch implementation of a paper, Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization [Huang+, ICCV2017].
I'm really grateful to the [original implementation](https://github.com/xunhuang1995/AdaIN-style) in Torch by the authors, which is very useful.

## Requirements
- Python 3.5+
- PyTorch
- TorchVision

## Usage

### Download models
This command will download a pre-trained decoder as well as a modified VGG-19 network.
```
bash models/download_models.sh
```

### Convert models
This command will convert the models for Torch to the models for PyTorch.
```
python convert_torch.py --model models/vgg_normalised.t7
python convert_torch.py --model models/decoder.t7
```

### Test
Use `--content` and `--style` to provide the respective path to the content and style image.
```
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content input/content/cornell.jpg --style input/style/woman_with_hat_matisse.jpg
```

You can also run the code on directories of content and style images using `--content_dir` and `--style_dir`. It will save every possible combination of content and styles to the output directory.
```
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content_dir input/content --style_dir input/style
```

Some other options:
* `--content_size`: New (minimum) size for the content image. Keeping the original size if set to 0.
* `--style_size`: New (minimum) size for the content image. Keeping the original size if set to 0.
* `--alpha`: Adjust the degree of stylization. It should be a value between 0.0 and 1.0 (default).
* `--preserve_color`: Preserve the color of the content image.

## TODO
- [x] Implement the preserve color option
- [x] Implement the style interpolation option
- [ ] Implement the spatial control option

## References
- [1]: X. Huang and S. Belongie. "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.", in ICCV, 2017.
- [2]: [Original implementation in Torch](https://github.com/xunhuang1995/AdaIN-style)
