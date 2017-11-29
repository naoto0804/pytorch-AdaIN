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
