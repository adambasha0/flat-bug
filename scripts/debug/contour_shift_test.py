import torch
import numpy as np
from flat_bug.geometric import resize_mask, find_contours, scale_contour


def make_rect_mask(h, w, x1, y1, x2, y2):
    m = torch.zeros((h, w), dtype=torch.bool)
    m[y1:y2, x1:x2] = 1
    return m


def bbox_from_contour(contour):
    # contour: Nx2 numpy array (y,x) or (x,y) depending on cv2
    # We will treat columns as (x, y) if shape matches.
    arr = np.array(contour).astype(float)
    # Ensure shape (N,2)
    if arr.shape[1] != 2:
        raise RuntimeError(f"Unexpected contour shape: {arr.shape}")
    xs = arr[:, 0]
    ys = arr[:, 1]
    xmin = xs.min()
    ymin = ys.min()
    xmax = xs.max()
    ymax = ys.max()
    return xmin, ymin, xmax, ymax


def main():
    # Original mask at 256x256 with a rectangle
    H0 = W0 = 256
    rect = (60, 60, 190, 190)  # x1,y1,x2,y2 in mask coords
    mask256 = make_rect_mask(H0, W0, *rect)

    print('Original mask 256 bbox (x1,y1,x2,y2):', rect)

    # Resize mask to 3x -> 768x768
    H1 = W1 = 256 * 3
    mask768 = resize_mask(mask256, (H1, W1))

    # Find contours on resized mask
    cont_res = find_contours(mask768, largest_only=True, simplify=False)
    cont_res_np = cont_res.cpu().numpy()
    print('Contour from 768 mask shape:', cont_res_np.shape)

    # Map contour to image/tile coords using FlatBug factor (1024/256)/3 = 4/3
    factor = (1024.0 / 256.0) / 3.0
    cont_mapped = cont_res_np * factor
    bbox_mapped = bbox_from_contour(cont_mapped)
    print('Mapped contour bbox (float) after *4/3:', tuple(np.round(bbox_mapped, 3)))

    # Now apply scale_contour with expand_by_one=True and a scale slightly >1 to force path
    # In predictor the scaling to image coords is done by mask_to_image_scale (≈4.0 for 256->1024).
    # We'll simulate a scale of exactly 1.0 here would skip processing; use a representative image-scale of 1.0
    # but to exercise expand_by_one we pass scale=1.1
    cont_after = scale_contour(cont_mapped.copy(), scale=[1.1, 1.1], expand_by_one=True)
    bbox_after = bbox_from_contour(cont_after)
    print('BBox after scale_contour(scale=1.1, expand_by_one=True):', tuple(np.round(bbox_after, 3)))

    # Round/convert to integer as the code does when writing boxes
    xmin, ymin, xmax, ymax = bbox_after
    xmin_floor = np.floor(xmin).astype(int)
    ymin_floor = np.floor(ymin).astype(int)
    xmax_ceil = np.ceil(xmax).astype(int)
    ymax_ceil = np.ceil(ymax).astype(int)
    print('Rounded bbox (floor/ceil):', (xmin_floor, ymin_floor, xmax_ceil, ymax_ceil))

    # Apply pad=5 like FlatBug
    pad = 5
    padded = (xmin_floor - pad, ymin_floor - pad, xmax_ceil + pad, ymax_ceil + pad)
    print('Padded bbox with pad=5:', padded)

    # Show net pixel differences compared to mapped bbox
    mapped_int = (int(np.floor(bbox_mapped[0])), int(np.floor(bbox_mapped[1])), int(np.ceil(bbox_mapped[2])), int(np.ceil(bbox_mapped[3])))
    print('Mapped bbox rounded (floor/ceil):', mapped_int)

    diffs = (xmin_floor - mapped_int[0], ymin_floor - mapped_int[1], xmax_ceil - mapped_int[2], ymax_ceil - mapped_int[3])
    print('Delta (after scale_contour vs mapped) per side:', diffs)

if __name__ == '__main__':
    main()
