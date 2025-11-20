import glob, os, sys, subprocess
import numpy as np

in_dir = 'data/teacher_logits_resnet18_50'
out_dir = 'samples/teacher_logits_vis'
tmp_dir = 'tmp_vis'

os.makedirs(out_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)

files = sorted(glob.glob(os.path.join(in_dir, '*_logits.npy')))
files = files[:10]

print('processing', len(files), 'files')
for f in files:
    stem = os.path.basename(f).replace('_logits.npy', '')
    # try to find matching image path
    candidates = glob.glob(f"data/images/**/*{stem}.*", recursive=True)
    if not candidates:
        print('skip, no image for', stem)
        continue
    img = candidates[0]
    print('vis', stem, img)
    logits = np.load(f)
    # choose first channel if multi-channel
    if logits.ndim == 3:
        if logits.shape[0] <= 4:
            log_ch = logits[0]
        elif logits.shape[2] <= 4:
            log_ch = logits[:, :, 0]
        else:
            log_ch = logits.squeeze()
    else:
        log_ch = logits
    # sigmoid
    probs = 1.0 / (1.0 + np.exp(-log_ch))
    tmp_npy = os.path.join(tmp_dir, stem + '_prob.npy')
    np.save(tmp_npy, probs.astype(np.float32))
    out_png = os.path.join(out_dir, stem + '_overlay.png')
    # call visualize_pred.py
    cmd = [sys.executable, 'tools/visualize_pred.py', img, tmp_npy, out_png]
    subprocess.run(cmd)

print('done')
