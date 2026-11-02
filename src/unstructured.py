import cv2
import numpy as np
from scipy.signal import correlate


def ncc(image, template):
    image = np.asarray(image, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    centered = template - template.mean()
    energy = np.sum(centered**2)
    if energy < 1e-12:
        raise ValueError("Choose a template with visible texture")
    box = np.ones_like(template)
    local_sum = correlate(image, box, mode="valid", method="fft")
    local_square = correlate(image**2, box, mode="valid", method="fft")
    variance = np.maximum(local_square - local_sum**2 / template.size, 0)
    numerator = correlate(image, centered, mode="valid", method="fft")
    denominator = np.sqrt(energy * variance)
    return np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12
    )


def read_video(path, step=3, width=960):
    if step < 1 or width < 1:
        raise ValueError("Step and width must be positive")
    capture = cv2.VideoCapture(str(path))
    frames, indices = [], []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (width, round(h * width / w)))
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            indices.append(index)
        index += 1
    capture.release()
    if len(frames) < 3:
        raise ValueError("Video must contain at least three sampled frames")
    return np.stack(frames), indices


def refocus_video(frames, point, size=41, radius=80):
    from lightfield import linearize

    gray = frames.astype(np.float32) / 255 @ np.array([0.2126, 0.7152, 0.0722])
    x, y = point
    half = size // 2
    h, w = gray.shape[1:]
    if size < 3 or size % 2 == 0 or radius < 0:
        raise ValueError("Use an odd template size >= 3 and a nonnegative radius")
    if not (half <= x < w - half and half <= y < h - half):
        raise ValueError("Template must lie entirely inside the middle frame")
    template = gray[len(frames) // 2, y - half : y + half + 1, x - half : x + half + 1]
    left, top = max(0, x - half - radius), max(0, y - half - radius)
    right, bottom = min(w, x + half + radius + 1), min(h, y + half + radius + 1)
    total = np.zeros((h, w, 3), np.float64)
    coverage = np.zeros((h, w), np.float64)
    shifts, scores = [], []
    for frame, luminance in zip(frames, gray):
        correlation = ncc(luminance[top:bottom, left:right], template)
        row, col = np.unravel_index(correlation.argmax(), correlation.shape)
        dx, dy = x - half - left - col, y - half - top - row
        transform = np.float32([[1, 0, dx], [0, 1, dy]])
        total += cv2.warpAffine(
            linearize(frame.astype(np.float32) / 255), transform, (w, h)
        )
        coverage += cv2.warpAffine(np.ones((h, w), np.float32), transform, (w, h))
        shifts.append([int(dx), int(dy)])
        scores.append(float(correlation[row, col]))
    result = np.divide(
        total,
        coverage[..., None],
        out=np.zeros_like(total),
        where=coverage[..., None] > 0,
    )
    return result, coverage == len(frames), shifts, scores
