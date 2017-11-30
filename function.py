import torch


def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.data.size()
    N, C = size[:2]
    assert (size[:2] == style_feat.data.size()[:2])

    style_feat = style_feat.view(N, C, -1)
    style_std = style_feat.std(dim=2).view(N, C, 1, 1).expand(size)
    style_mean = style_feat.mean(dim=2).view(N, C, 1, 1).expand(size)
    content_feat_3d = content_feat.view(N, C, -1)
    content_std = content_feat_3d.std(dim=2).view(N, C, 1, 1).expand(size)
    content_mean = content_feat_3d.mean(dim=2).view(N, C, 1, 1).expand(size)
    normalized_feat = (content_feat - content_mean) / content_std
    return normalized_feat * style_std + style_mean


def style_transfer(vgg, decoder, content, style, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_feat = vgg(content)
    style_feat = vgg(style)
    feat = adaptive_instance_normalization(content_feat, style_feat)
    feat = feat * alpha + content_feat * (1 - alpha)
    return decoder(feat)


def calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())
