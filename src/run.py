import argparse
import json
from pathlib import Path

import numpy as np
from figures import depth_rgb, panel, save_afis, save_rgb
from lightfield import confocal, depth_from_focus, encode, mosaic, render, unpack
from PIL import Image
from unstructured import read_video, refocus_video


def chessboard(args):
    args.output.mkdir(parents=True, exist_ok=True)
    lightfield = unpack(np.asarray(Image.open(args.input).convert("RGB")))
    Image.fromarray(mosaic(lightfield)).save(
        args.output / "viewpoint_mosaic.jpg", quality=94
    )
    depths = np.linspace(args.depth_min, args.depth_max, args.samples)
    apertures = np.array([2, 4, 8, 12, 16])
    margin = int(np.ceil(np.max(np.abs(depths)) * 7.5)) + 1
    h, w = lightfield.shape[2:4]
    shape = (len(apertures), len(depths), h - 2 * margin, w - 2 * margin, 3)
    if min(shape[2:4]) <= 0:
        raise ValueError("Depth range leaves no common image area")
    stack = np.lib.format.open_memmap(
        args.output / "focal_aperture.npy", mode="w+", dtype=np.float32, shape=shape
    )
    for i, depth in enumerate(depths):
        print(f"Rendering {i + 1}/{len(depths)}: d={depth:.2f}", flush=True)
        stack[:, i] = render(lightfield, depth, apertures)[
            :, margin:-margin, margin:-margin
        ]
    stack.flush()
    chosen = np.linspace(0, len(depths) - 1, 5).astype(int)
    panel(
        [encode(stack[-1, i]) for i in chosen],
        [f"d = {depths[i]:.2f}" for i in chosen],
        5,
        350,
    ).save(args.output / "refocusings.jpg")
    for i in chosen:
        save_rgb(args.output / f"refocus_{i:02d}.jpg", stack[-1, i])
    panel(
        [encode(stack[a, i]) for a in range(len(apertures)) for i in chosen],
        [f"a={a}, d={depths[i]:.2f}" for a in apertures for i in chosen],
        5,
        280,
    ).save(args.output / "focal_aperture_collage.jpg")
    experiments = []
    for sigma1, sigma2 in [(0.5, 1), (1, 2), (1, 4), (2, 4)]:
        image, depth, weights = depth_from_focus(stack[-1], depths, sigma1, sigma2)
        name = f"s{sigma1:g}_{sigma2:g}"
        save_rgb(args.output / f"all_focus_{name}.jpg", image)
        save_rgb(
            args.output / f"depth_{name}.png", depth_rgb(depth, depths), linear=False
        )
        np.save(args.output / f"depth_{name}.npy", depth)
        experiments.extend([encode(image), depth_rgb(depth, depths)])
        del weights
    panel(
        experiments,
        [
            f"sigma1={s1}, sigma2={s2}: {kind}"
            for s1, s2 in [(0.5, 1), (1, 2), (1, 4), (2, 4)]
            for kind in ["all focus", "depth"]
        ],
        2,
        600,
    ).save(args.output / "sigma_comparison.jpg")
    depth, variance = confocal(stack, depths)
    np.save(args.output / "depth_confocal.npy", depth)
    save_rgb(args.output / "depth_confocal.png", depth_rgb(depth, depths), linear=False)
    np.save(args.output / "confocal_variance.npy", variance)
    points = save_afis(args.output, stack)
    metadata = {
        "shape": list(lightfield.shape),
        "depths": depths.tolist(),
        "apertures": apertures.tolist(),
        "crop_margin": margin,
        "afi_points": points,
        "stack_color": "linear RGB",
        "raw_angular_v_reversed": True,
        "depth_units": "shift per angular sample; not metric distance",
    }
    (args.output / "settings.json").write_text(json.dumps(metadata, indent=2))


def own_video(args):
    args.output.mkdir(parents=True, exist_ok=True)
    frames, indices = read_video(args.video, args.step, args.width)
    middle = frames[len(frames) // 2]
    Image.fromarray(middle).save(args.output / "middle_frame.jpg")
    if not args.point:
        print(
            "Select template centers (x,y) on middle_frame.jpg, then pass --point x y"
        )
        return
    records, images, bounds = [], [], []
    for x, y in args.point:
        image, valid, shifts, scores = refocus_video(
            frames, (x, y), args.template, args.radius
        )
        rows, cols = np.where(valid)
        if not len(rows):
            raise ValueError("No common area after alignment; reduce camera motion")
        bounds.append(
            [int(cols.min()), int(rows.min()), int(cols.max() + 1), int(rows.max() + 1)]
        )
        images.append(image)
        records.append({"point": [x, y], "shifts_xy": shifts, "ncc_scores": scores})
    left, top = np.max(np.array(bounds)[:, :2], axis=0)
    right, bottom = np.min(np.array(bounds)[:, 2:], axis=0)
    if left >= right or top >= bottom:
        raise ValueError("No shared crop across focus targets")
    crops, labels = [], []
    for image, record in zip(images, records):
        x, y = record["point"]
        crop = image[top:bottom, left:right]
        save_rgb(args.output / f"focus_{x}_{y}.jpg", crop)
        record["crop_xyxy"] = [int(left), int(top), int(right), int(bottom)]
        crops.append(encode(crop))
        labels.append(f"Template center ({x}, {y})")
    panel(crops, labels, len(crops)).save(args.output / "own_refocusings.jpg")
    (args.output / "alignment.json").write_text(
        json.dumps(
            {
                "frame_indices": indices,
                "template_size": args.template,
                "search_radius": args.radius,
                "processing_width": args.width,
                "results": records,
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="part", required=True)
    chess = sub.add_parser("chess")
    chess.add_argument(
        "--input", type=Path, default=Path("data/chessboard_lightfield.png")
    )
    chess.add_argument("--output", type=Path, default=Path("data/chess"))
    chess.add_argument("--depth-min", type=float, default=-0.5)
    chess.add_argument("--depth-max", type=float, default=2)
    chess.add_argument("--samples", type=int, default=51)
    video = sub.add_parser("video")
    video.add_argument("video", type=Path)
    video.add_argument("--output", type=Path, default=Path("data/own"))
    video.add_argument("--point", nargs=2, type=int, action="append")
    video.add_argument("--step", type=int, default=3)
    video.add_argument("--width", type=int, default=960)
    video.add_argument("--template", type=int, default=41)
    video.add_argument("--radius", type=int, default=80)
    args = parser.parse_args()
    if args.part == "chess":
        if args.samples < 5 or args.depth_min >= args.depth_max:
            parser.error("Use at least five samples and increasing depth limits")
        chessboard(args)
    else:
        own_video(args)


if __name__ == "__main__":
    main()
