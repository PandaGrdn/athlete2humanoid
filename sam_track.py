"""Run SAM 3.1 on a video and save the athlete's masks. Runs in SAM's own environment
(Python 3.12, numpy<2), which pipeline.py calls as a subprocess:

    $A2H_SAM_PYTHON sam_track.py clip.mp4 --prompt athlete --out masks.npz
"""

import argparse

import numpy as np

import track

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--prompt", default="athlete")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    # "athlete" can miss an ordinary person in a home video, or latch onto something seen only
    # briefly; fall back to "person" and keep whichever prompt tracks someone most of the time.
    best = None
    for prompt in dict.fromkeys([a.prompt, "person"]):
        outputs = track.track(a.video, prompt=prompt)
        try:
            masks, obj_id = track.main_track_masks(outputs, max(outputs) + 1)
        except RuntimeError as e:
            print(f"prompt {prompt!r}: {e}")
            continue
        seen = masks.any((1, 2)).mean()
        print(f"prompt {prompt!r}: object {obj_id} visible in {seen:.0%} of frames")
        if best is None or seen > best[0]:
            best = (seen, prompt, masks, obj_id)
        if seen >= 0.8:
            break
    if best is None:
        raise SystemExit("SAM 3.1 found no person with any prompt.")
    _, prompt, masks, obj_id = best
    np.savez_compressed(a.out, masks=masks, obj_id=obj_id)
    print(f"SAM 3.1 ({prompt!r}): object {obj_id}, visible in {masks.any((1, 2)).mean():.0%} of frames")
