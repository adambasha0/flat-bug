import cv2
import numpy as np


def make_rect_mask(h, w, x1, y1, x2, y2):
    m = np.zeros((h, w), dtype=np.uint8)
    m[y1:y2, x1:x2] = 1
    return m


def bbox_from_contour(contour):
    arr = np.array(contour).astype(float)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr.squeeze(1)
    if arr.shape[1] != 2:
        raise RuntimeError(f"Unexpected contour shape: {arr.shape}")
    xs = arr[:, 0]
    ys = arr[:, 1]
    xmin = xs.min()
    ymin = ys.min()
    xmax = xs.max()
    ymax = ys.max()
    return xmin, ymin, xmax, ymax


def linear_interpolate(poly: np.ndarray, scale: int) -> np.ndarray:
    if scale < 1:
        raise ValueError("Scale must be at least 1")
    if len(poly) == 0:
        return poly
    if scale == 1:
        return poly
    new_poly = np.zeros((poly.shape[0] * scale, 2), dtype=np.float32)
    for i in range(poly.shape[0] - 1):
        new_poly[i*scale:(i+1)*scale] = np.linspace(poly[i], poly[i+1], scale, endpoint=False)
    new_poly[-scale:] = np.linspace(poly[-1], poly[0], scale, endpoint=False)
    # remove repeated consecutive points
    mask = ~(new_poly == np.roll(new_poly, -1, axis=0)).all(axis=1)
    return new_poly[mask]


def poly_normals(polygon: np.ndarray) -> np.ndarray:
    v = np.roll(polygon, -1, axis=0) - polygon
    n = np.column_stack([v[:, 1], -v[:, 0]])
    n = (n + np.roll(n, 1, axis=0)) / 2
    return n


def scale_contour(contour: np.ndarray, scale, expand_by_one: bool = False) -> np.ndarray:
    contour = np.array(contour, dtype=np.float32)
    if contour.ndim != 2 or contour.shape[1] != 2:
        if contour.shape[0] == 2:
            contour = contour.reshape(1, 2)
        else:
            raise ValueError(f"Contour must be Nx2 array, not {contour.shape}")
    if isinstance(scale, (int, float)):
        scale = [scale, scale]
    scale = np.array(scale, dtype=np.float32)
    if len(scale) != 2:
        raise ValueError(f"Scale must be scalar or list of 2 scalars, not {scale}")
    if len(contour) == 0:
        return contour
    if len(contour) == 1:
        return np.round(contour * scale).astype(np.int32)
    if np.all(scale == 1):
        return contour
    contour = contour * scale
    centroid = contour.mean(axis=0)
    n_interp = max(1, int(np.ceil(scale.max())) * 2)
    contour = linear_interpolate(contour, n_interp)
    contour_normals = poly_normals(contour)
    if expand_by_one:
        expand_one = np.sign(contour_normals) * (np.abs(contour_normals) > 0)
        contour -= expand_one
    if scale[0] < 1:
        contour[:, 0] += contour_normals[:, 0] / scale[0] / 2
    if scale[1] < 1:
        contour[:, 1] += contour_normals[:, 1] / scale[1] / 2
    # rounding toward outside
    positive = contour_normals > 0
    contour[positive] = np.floor(contour[positive])
    contour[~positive] = np.ceil(contour[~positive])
    contour = np.round(contour)
    drift = centroid - contour.mean(axis=0)
    out = (contour + drift).round().astype(np.int32)
    return out[(n_interp // 2)::n_interp].copy()


def main():
    H0 = W0 = 256
    rect = (60, 60, 190, 190)  # x1,y1,x2,y2
    mask256 = make_rect_mask(H0, W0, *rect)
    print('Original mask 256 bbox (x1,y1,x2,y2):', rect)

    # Resize to 768x768 like FlatBug
    H1 = W1 = 256 * 3
    # use nearest to mimic boolean mask resizing
    mask768 = cv2.resize(mask256.astype(np.uint8), (W1, H1), interpolation=cv2.INTER_NEAREST)

    # find contours
    contours = cv2.findContours((mask768).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if len(contours) == 0:
        print('No contour found')
        return
    cont_res = contours[np.argmax([cv2.contourArea(c) for c in contours])]
    cont_res = cont_res.squeeze(1)  # Nx2 (x,y)
    print('Contour from 768 mask shape:', cont_res.shape)

    factor = (1024.0 / 256.0) / 3.0
    cont_mapped = cont_res * factor
    bbox_mapped = bbox_from_contour(cont_mapped)
    print('Mapped contour bbox (float) after *4/3:', tuple(np.round(bbox_mapped, 3)))

    cont_after = scale_contour(cont_mapped.copy(), scale=[1.1, 1.1], expand_by_one=True)
    bbox_after = bbox_from_contour(cont_after)
    print('BBox after scale_contour(scale=1.1, expand_by_one=True):', tuple(np.round(bbox_after, 3)))

    xmin, ymin, xmax, ymax = bbox_after
    xmin_floor = int(np.floor(xmin))
    ymin_floor = int(np.floor(ymin))
    xmax_ceil = int(np.ceil(xmax))
    ymax_ceil = int(np.ceil(ymax))
    print('Rounded bbox (floor/ceil):', (xmin_floor, ymin_floor, xmax_ceil, ymax_ceil))

    pad = 5
    padded = (xmin_floor - pad, ymin_floor - pad, xmax_ceil + pad, ymax_ceil + pad)
    print('Padded bbox with pad=5:', padded)

    mapped_int = (int(np.floor(bbox_mapped[0])), int(np.floor(bbox_mapped[1])), int(np.ceil(bbox_mapped[2])), int(np.ceil(bbox_mapped[3])))
    print('Mapped bbox rounded (floor/ceil):', mapped_int)

    diffs = (xmin_floor - mapped_int[0], ymin_floor - mapped_int[1], xmax_ceil - mapped_int[2], ymax_ceil - mapped_int[3])
    print('Delta (after scale_contour vs mapped) per side:', diffs)

if __name__ == '__main__':
    main()
