import numpy as np
import cv2
import sys
import os

def usage():
	print('Usage: python tools/visualize_pred.py <image> <pred.npy> <out.png>')


if len(sys.argv) < 4:
	usage()
	sys.exit(1)

img_path = sys.argv[1]
pred_npy = sys.argv[2]
out_path = sys.argv[3]

# load image
img = cv2.imread(img_path)
if img is None:
	print(f"ERROR: failed to read image '{img_path}'")
	sys.exit(2)

# load prediction npy
try:
	pred = np.load(pred_npy)
except Exception as e:
	print(f"ERROR: failed to load npy '{pred_npy}': {e}")
	sys.exit(3)

# normalize prediction to 2D float [0,1]
if pred.ndim == 3:
	# handle shapes (C,H,W) or (H,W,C)
	if pred.shape[0] <= 4 and pred.shape[0] != img.shape[0]:
		# assume (C,H,W) where C is channel-first
		pred2 = pred[0]
	elif pred.shape[2] <= 4:
		# assume (H,W,C) channel-last
		pred2 = pred[..., 0]
	else:
		# fallback: pick first channel
		pred2 = pred[0]
else:
	pred2 = pred

pred2 = pred2.astype('float32')
# if values look like 0..255, normalize
mx = pred2.max() if pred2.size else 0.0
if mx > 1.5:
	pred2 = pred2 / 255.0

pred2 = np.clip(pred2, 0.0, 1.0)

# resize prediction to image size
h, w = img.shape[:2]
pred_resized = cv2.resize(pred2, (w, h), interpolation=cv2.INTER_LINEAR)

# compose overlay: alpha in [0,1]
alpha_max = 0.6
alpha = np.clip(pred_resized * alpha_max, 0.0, 1.0).astype('float32')
alpha = alpha[:, :, None]

# cv2.imread returns BGR; green color in BGR is (0,255,0)
green = np.array([0, 255, 0], dtype='float32')
overlay = (img.astype('float32') * (1.0 - alpha) + green * alpha).astype('uint8')

out_dir = os.path.dirname(out_path)
if out_dir and not os.path.exists(out_dir):
	os.makedirs(out_dir, exist_ok=True)

cv2.imwrite(out_path, overlay)
print('wrote', out_path)