import numpy as np
from lightfield import encode
from PIL import Image, ImageDraw


def save_rgb(path, image, linear=True):
    rgb = encode(image) if linear else np.clip(image, 0, 1)
    Image.fromarray(np.uint8(np.round(rgb * 255))).save(path, quality=94)


def panel(images, labels, columns=3, width=420):
    height = round(images[0].shape[0] * width / images[0].shape[1])
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width, rows * (height + 28)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (image, label) in enumerate(zip(images, labels)):
        tile = Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255))
        tile = tile.resize((width, height), Image.Resampling.LANCZOS)
        x, y = i % columns * width, i // columns * (height + 28)
        canvas.paste(tile, (x, y + 28))
        draw.text((x + 8, y + 6), label, fill="black")
    return canvas


def depth_rgb(depth, depths):
    value = np.clip((depth - depths[0]) / (depths[-1] - depths[0]), 0, 1)
    return np.repeat(value[..., None], 3, axis=-1)


def save_afis(output, stack):
    _, _, h, w, _ = stack.shape
    points = [
        (round(w * 0.2), round(h * 0.65)),
        (round(w * 0.54), round(h * 0.4)),
        (round(w * 0.59), round(h * 0.12)),
    ]
    images = []
    for i, (x, y) in enumerate(points):
        afi = stack[:, :, y, x]
        np.save(output / f"afi_{i + 1}.npy", afi)
        luminance = afi @ np.array([0.2126729, 0.7151522, 0.0721750])
        normalized = (luminance - luminance.min()) / max(
            float(np.ptp(luminance)), 1e-12
        )
        images.append(
            np.repeat(np.repeat(normalized[..., None], 3, axis=-1), 5, axis=0)
        )
    panel(
        images, [f"({x},{y}); aperture down, d right" for x, y in points], 3, 420
    ).save(output / "afis.jpg")
    marked = Image.fromarray(np.uint8(encode(stack[-1, stack.shape[1] // 2]) * 255))
    draw = ImageDraw.Draw(marked)
    for i, (x, y) in enumerate(points):
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline="red", width=2)
        draw.text((x + 7, y), str(i + 1), fill="red")
    marked.save(output / "afi_locations.jpg")
    return points
