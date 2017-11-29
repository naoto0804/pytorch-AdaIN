# pytorch-AdaIN

This is an unofficial pytorch implementation of a paper, Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization [Huang+, ICCV2017].

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
```
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content input/content/cornell.jpg --style input/style/woman_with_hat_matisse.jpg --gpu
```

Some other options:
* `-contentSize`: New (minimum) size for the content image. Keeping the original size if set to 0.
* `-styleSize`: New (minimum) size for the content image. Keeping the original size if set to 0.

### TODO
- [ ] preserve color
- [ ] spatial control
- [ ] style interpolation

## References
- [1]: X. Huang and S. Belongie. "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.", in ICCV, 2017.
- [2]: [Original implementation in Torch](https://github.com/xunhuang1995/AdaIN-style)
