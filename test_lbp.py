import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import time
from app.liveness_engine import compute_lbp as original_lbp

def test_lbp_equivalence():
    # Create a random image and mask
    img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    mask = np.ones((100, 100), dtype=np.uint8) * 255
    
    # Original LBP
    start = time.perf_counter()
    val_orig = original_lbp(img, mask)
    time_orig = time.perf_counter() - start
    
    # Skimage LBP
    # The original LBP uses 8 neighbors, radius 1, and a specific ordering.
    # original code starts at top-left, goes clockwise or counter-clockwise?
    # code |= (gray[i-1, j-1] >= center) << 7 (top-left)
    # code |= (gray[i-1, j] >= center) << 6 (top-mid)
    # code |= (gray[i-1, j+1] >= center) << 5 (top-right)
    # code |= (gray[i, j+1] >= center) << 4 (mid-right)
    # code |= (gray[i+1, j+1] >= center) << 3 (bot-right)
    # code |= (gray[i+1, j] >= center) << 2 (bot-mid)
    # code |= (gray[i+1, j-1] >= center) << 1 (bot-left)
    # code |= (gray[i, j-1] >= center) << 0 (mid-left)
    # This is a standard 8-neighbor uniform LBP pattern without rotation invariance.
    
    start = time.perf_counter()
    lbp_sk = local_binary_pattern(img, 8, 1, method='default')
    if mask is not None:
        # skimage returns float array, original uses uint8 logic but sum is mean
        val_sk = np.mean(lbp_sk[mask > 0])
    else:
        val_sk = np.mean(lbp_sk)
    time_sk = time.perf_counter() - start
    
    print(f"Original LBP: {val_orig:.4f} (Time: {time_orig:.4f}s)")
    print(f"Skimage LBP:  {val_sk:.4f} (Time: {time_sk:.4f}s)")
    print(f"Difference:   {abs(val_orig - val_sk):.4f}")

if __name__ == '__main__':
    test_lbp_equivalence()
