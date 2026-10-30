"""Cache MediaPipe landmarks for a video, frames read as pipeline.py reads them: whole frame (the
--no-sam path), or with the CPU athlete tracker when a selection is given ('largest' or an --athlete
spec, athlete.py).

    python experiments/camera_coverage/landmarks.py <repo> <video> <out.npz> [largest|x,y|x0,y0,x1,y1|i]
"""

import sys
import time

import imageio_ffmpeg
import mediapy
import numpy as np

sys.path.insert(0, sys.argv[1])
import athlete  # noqa: E402
import pose  # noqa: E402
import track  # noqa: E402

mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
v = mediapy.read_video(sys.argv[2])
fps = v.metadata.fps
frames = np.asarray(v)
t0 = time.time()
sel = sys.argv[4] if len(sys.argv) > 4 else None
if sel is not None:
    masks, tinfo = athlete.track_people(frames, None if sel == "largest" else sel)
    print("tracker", tinfo)
else:
    masks = np.ones(frames.shape[:3], bool)
crops, boxes = track.crop_to_mask(frames, masks)
world, image_xy, vis = pose.landmarks_from_crops(crops, fps)
np.savez_compressed(sys.argv[3], world=world, image_xy=image_xy, vis=vis, boxes=boxes, fps=fps,
                    masks=np.packbits(masks, axis=-1), mask_shape=masks.shape)
print(sys.argv[2], frames.shape, f"pose {(~np.isnan(world[:, 0, 0])).mean():.0%}", f"{time.time() - t0:.0f}s")
