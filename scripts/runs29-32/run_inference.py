"""
Tier 1 inference runner for CrossFuse.

Numerically identical to the repo's `test.py` (same preprocessing, same min-max
normalisation, same YCrCb recomposition), but adds what a 7-10 hour CPU run needs:
  * --datasets selection and resume (skips images already fused)
  * per-image timing and ETA
  * a hard CPU thread setting (4 = physical cores; 8 measured slower)

It also writes a SECOND output that `test.py` does not:

    test_results/<ds>/<name>.png       RGB, exactly as test.py produces
    test_results_gray/<ds>/<name>.png  the raw fused Y channel, uint8

That second output matters for gap G6. Every metric in `utils/Evaluator.py` asserts a
2-D input, but test.py only ever saves RGB. Recovering luma from that RGB is lossy --
the Y->RGB->Y round-trip through the visible image's Cr/Cb clips and quantises. Saving
the network's actual Y output gives an unambiguous reference, so the colour-space
convention can be settled by measurement rather than assumption.

Usage:
    python run_inference.py --repo ../CrossFuse --datasets TNO
    python run_inference.py --repo ../CrossFuse --datasets RoadScene MSRS
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--repo", default=os.path.join(here, "..", "CrossFuse"))
    ap.add_argument("--datasets", nargs="*", default=["RoadScene", "TNO", "MSRS"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    # Tier 3 ablations III/IV change the encoder, so the architecture built here must
    # match the one that was trained or load_state_dict fails. Defaults = published.
    ap.add_argument("--long_scale", choices=["transformer", "cnn"], default="transformer")
    ap.add_argument("--short_scale", choices=["cnn", "transformer"], default="cnn")
    # Ablation V (capacity-matched III) widens the substituted CNN, so inference must be told
    # the same width or load_state_dict fails on a size mismatch. 3 = published.
    ap.add_argument("--long_cnn_layers", type=int, default=3)
    # Run 30 (Tier3_CrossFuse/RUN30_PREREGISTRATION.md): remove the decoder's skip and nothing else. The decoder adds
    # inp_img only when it is not None (network.py, Restormer_Decoder.forward), so passing None is the whole change.
    ap.add_argument("--no-skip", action="store_true", help="decoder without its visible-input skip (Run 30)")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    sys.path.insert(0, repo)
    torch.set_num_threads(args.threads)

    from network import (Restormer_Encoder, Restormer_Decoder,
                         LongScaleFeatureExtraction, ShortScaleFeatureExtraction)
    from utils.img_read_save import image_read_cv2, img_save

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = args.ckpt or os.path.join(repo, "models", "CrossFuse_RSM3.pth")
    ckpt = torch.load(ckpt_path, map_location=device)

    Encoder = nn.DataParallel(Restormer_Encoder(long_scale=args.long_scale,
                                                short_scale=args.short_scale,
                                                long_cnn_layers=args.long_cnn_layers)).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
    BaseFuseLayer = nn.DataParallel(LongScaleFeatureExtraction(dim=64, num_heads=8)).to(device)
    DetailFuseLayer = nn.DataParallel(ShortScaleFeatureExtraction(num_layers=1)).to(device)
    Encoder.load_state_dict(ckpt["DIDF_Encoder"])
    Decoder.load_state_dict(ckpt["DIDF_Decoder"])
    BaseFuseLayer.load_state_dict(ckpt["BaseFuseLayer"])
    DetailFuseLayer.load_state_dict(ckpt["DetailFuseLayer"])
    for m in (Encoder, Decoder, BaseFuseLayer, DetailFuseLayer):
        m.eval()

    print(f"device={device} threads={torch.get_num_threads()} ckpt={os.path.basename(ckpt_path)}")

    for ds in args.datasets:
        test_folder = os.path.join(repo, "test_image", ds)
        out_rgb = os.path.join(repo, "test_results", ds)
        out_gray = os.path.join(repo, "test_results_gray", ds)
        ir_dir = os.path.join(test_folder, "ir")
        vi_dir = os.path.join(test_folder, "vi")
        if not os.path.isdir(ir_dir):
            print(f"[skip] {ds}: no {ir_dir}")
            continue
        os.makedirs(out_rgb, exist_ok=True)
        os.makedirs(out_gray, exist_ok=True)

        names = sorted(os.listdir(ir_dir))
        todo = [n for n in names
                if not os.path.exists(os.path.join(out_rgb, os.path.splitext(n)[0] + ".png"))]
        print(f"\n=== {ds}: {len(names)} images, {len(names)-len(todo)} already done, {len(todo)} to run ===",
              flush=True)

        times = []
        for i, name in enumerate(todo, 1):
            if args.limit and i > args.limit:
                print(f"  reached --limit {args.limit}")
                break
            stem = os.path.splitext(name)[0]
            vi_path = os.path.join(vi_dir, name)
            if not os.path.exists(vi_path):
                print(f"  [skip] {stem}: no matching visible image")
                continue

            t0 = time.time()
            # --- preprocessing: identical to test.py ---
            data_IR = image_read_cv2(os.path.join(ir_dir, name), mode="GRAY")[np.newaxis, np.newaxis, ...] / 255.0
            data_VIS = cv2.split(image_read_cv2(vi_path, mode="YCrCb"))[0][np.newaxis, np.newaxis, ...] / 255.0
            data_VIS_BGR = cv2.imread(vi_path)
            _, data_VIS_Cr, data_VIS_Cb = cv2.split(cv2.cvtColor(data_VIS_BGR, cv2.COLOR_BGR2YCrCb))

            data_IR = torch.FloatTensor(data_IR).to(device)
            data_VIS = torch.FloatTensor(data_VIS).to(device)

            with torch.inference_mode():
                feature_V_L, feature_V_H, _ = Encoder(data_VIS)
                feature_I_L, feature_I_H, _ = Encoder(data_IR)
                feature_F_L = BaseFuseLayer(feature_V_L + feature_I_L)
                feature_F_H = DetailFuseLayer(feature_V_H + feature_I_H)
                Fuse, _ = Decoder(None if args.no_skip else data_VIS, feature_F_L, feature_F_H)
                Fuse = (Fuse - torch.min(Fuse)) / (torch.max(Fuse) - torch.min(Fuse))

            fuse = np.squeeze((Fuse * 255).cpu().numpy()).astype(np.uint8)

            # raw luma (unambiguous, no RGB round-trip) -- for gap G6
            img_save(fuse, stem, out_gray)

            # RGB, exactly as test.py
            ycrcb_fuse = np.dstack((fuse, data_VIS_Cr, data_VIS_Cb))
            rgb_fuse = cv2.cvtColor(ycrcb_fuse, cv2.COLOR_YCrCb2RGB)
            img_save(rgb_fuse, stem, out_rgb)

            dt = time.time() - t0
            times.append(dt)
            if i % 5 == 0 or i == 1 or i == len(todo):
                avg = float(np.mean(times))
                left = (len(todo) - i) * avg
                print(f"  [{i}/{len(todo)}] {stem}  {dt:.1f}s  avg={avg:.1f}s  ETA={left/60:.0f}min",
                      flush=True)

        if times:
            print(f"  {ds} done: n={len(times)} mean={np.mean(times):.2f}s "
                  f"total={np.sum(times)/60:.1f}min", flush=True)

    print("\nInference complete.")


if __name__ == "__main__":
    main()
