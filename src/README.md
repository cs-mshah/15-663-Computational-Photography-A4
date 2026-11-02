# Assignment 4

Run commands from the repository root. Python dependencies are in
`src/requirements.txt`; locally, use `uvenv photo`.

```sh
python -m pip install -r src/requirements.txt
```

## Part 1

Pass the supplied chessboard lightfield path explicitly. The provided input is
excluded from this repository as required by the submission guidelines.

```sh
python src/run.py chess --input /path/to/chessboard_lightfield.png
```

Results go to `data/chess/`. Rendering uses 51 depths from -0.5 to 2 and square
apertures with side lengths 2, 4, 8, 12 and 16. The raw horizontal angular index
is reversed to match the coordinate orientation used for refocusing. Rendering
and merging use linear RGB; saved color images are sRGB-encoded.

`viewpoint_mosaic.jpg` contains all 256 sub-aperture views. `refocus_*.jpg` are the five refocused images, summarized
in `refocusings.jpg`. `all_focus_s*.jpg` and `depth_s*.png` are the four
sharpness experiments shown in `sigma_comparison.jpg`. `focal_aperture_collage.jpg`,
`afis.jpg`, `afi_locations.jpg` and `depth_confocal.png` show the aperture/focus
stack, sample AFIs and their locations, and direct confocal depth. The depth
PNGs share a fixed scale: black is d=-0.5 and white is d=2; d is a refocusing
parameter, not distance in meters.

The run also writes numerical `.npy` caches and `settings.json`. These are
reproducible intermediates and are not included in the submission.

## Part 3

```sh
python src/run.py video data/own/lightfield.mp4 \
  --width 540 --step 3 --template 61 --radius 200 \
  --point 95 622 --point 220 485 --point 365 620
```

The submitted video is the 6.93-second phone capture used for all results.
Every third frame is processed at 540×960. Coordinates are (x,y) in the middle
sampled frame, measured from its top-left corner. The three points select the
wallet label, notebook lettering and bottle lettering. The images `focus_*.jpg`
and `own_refocusings.jpg` use the same common overlap crop.

For another video, first run without `--point` to save `middle_frame.jpg`, then
select textured points and rerun with those coordinates. Adjust `--template`
and `--radius` if necessary. NCC is implemented from correlations and pixel-wise
operations. Running also writes an `alignment.json` log of frame shifts and
scores; this diagnostic log is not included in the submission.

## Report

`report.pdf` is at the repository root. It contains the methods, requested
answers and figures for Parts 1 and 3.
