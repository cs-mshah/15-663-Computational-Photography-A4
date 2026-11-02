import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def linearize(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def encode(rgb):
    rgb = np.clip(rgb, 0, 1)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)


def unpack(image, size=16):
    h, w, c = image.shape
    if h % size or w % size or c != 3:
        raise ValueError("Expected RGB image with dimensions divisible by lenslet size")
    return image.reshape(h // size, size, w // size, size, c).transpose(1, 3, 0, 2, 4)[
        :, ::-1
    ]


def mosaic(lightfield):
    u, v, h, w, c = lightfield.shape
    return lightfield.transpose(0, 2, 1, 3, 4).reshape(u * h, v * w, c)


def render(lightfield, depth, apertures):
    n, _, h, w, _ = lightfield.shape
    coordinates = np.arange(n) - (n - 1) / 2
    y, x = np.mgrid[:h, :w].astype(np.float32)
    sums = np.zeros((len(apertures), h, w, 3), dtype=np.float32)
    counts = np.zeros(len(apertures), dtype=int)
    for i, u in enumerate(coordinates):
        for j, v in enumerate(coordinates):
            selected = np.maximum(abs(u), abs(v)) <= np.asarray(apertures) / 2
            if not selected.any():
                continue
            view = linearize(lightfield[i, j].astype(np.float32) / 255)
            shifted = cv2.remap(
                view,
                x + np.float32(depth * v),
                y + np.float32(depth * u),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            for a in np.flatnonzero(selected):
                sums[a] += shifted
                counts[a] += 1
    if (counts == 0).any():
        raise ValueError("Apertures must contain at least one angular sample")
    return sums / counts[:, None, None, None]


def focus_weights(stack, sigma1=1, sigma2=2):
    luminance = stack @ np.array([0.2126729, 0.7151522, 0.0721750], np.float32)
    high = luminance - gaussian_filter(luminance, (0, sigma1, sigma1))
    return gaussian_filter(high * high, (0, sigma2, sigma2))


def depth_from_focus(stack, depths, sigma1=1, sigma2=2):
    weights = focus_weights(stack, sigma1, sigma2)
    total = weights.sum(axis=0)
    normalized = np.divide(
        weights,
        total[None],
        out=np.full_like(weights, 1 / len(depths)),
        where=total[None] > 1e-15,
    )
    image = np.einsum("dhw,dhwc->hwc", normalized, stack)
    depth = np.einsum("dhw,d->hw", normalized, depths)
    return image, depth, weights


def confocal(stack, depths):
    variance = np.var(stack, axis=0).mean(axis=-1)
    return np.asarray(depths)[variance.argmin(axis=0)], variance
