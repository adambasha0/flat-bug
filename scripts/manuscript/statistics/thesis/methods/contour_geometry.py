#!/usr/bin/env python3
"""Polygon-contour geometry used by the greedy matcher (`fb_eval_greedy.py`).

Mask IoU in this thesis is computed by rasterising both polygons onto a shared
canvas and counting overlapping pixels, rather than from RLE — that is what makes
the mask IoU comparable across the models being evaluated.

These five functions are vendored verbatim (numpy + OpenCV only, no other
dependencies) so the evaluation runs standalone. They originate in the FlatBug
reference implementation, whose numbers this thesis reproduces:
`flat_bug.coco_utils` (`format_contour`, `contour_bbox`, `contour_area`) and
`flat_bug.eval_utils` (`bbox_intersect_area`, `contour_intersection`,
`pairwise_contour_intersection`) — https://github.com/darsa-group/flat-bug (AGPL-3.0).
Keeping the maths bit-identical is deliberate: it is what makes the AP numbers
here directly comparable to the published FlatBug baseline.
"""

from typing import List, Optional

import cv2
import numpy as np

__all__ = [
    "format_contour",
    "contour_bbox",
    "contour_area",
    "bbox_intersect_area",
    "contour_intersection",
    "pairwise_contour_intersection",
]


def format_contour(c: List) -> np.ndarray:
    """Convert a COCO polygon `[[x1, y1, x2, y2, ...]]` to an OpenCV Nx2 contour."""
    c = c[0]
    return np.array([[c[i], c[i + 1]] for i in range(0, len(c), 2)], dtype=np.int32)


def contour_bbox(c: np.ndarray) -> np.ndarray:
    """Axis-aligned bounding box `[x1, y1, x2, y2]` of an Nx2 contour."""
    return np.array([c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()])


def contour_area(c: np.ndarray) -> np.ndarray:
    """Filled pixel area of an Nx2 contour (rasterised, not the shoelace area)."""
    min_xy = c.min(axis=0)
    c = c - min_xy
    max_xy = c.max(axis=0) + 1
    mask = np.zeros(max_xy[::-1], dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 1, thickness=cv2.FILLED)
    return np.sum(mask, dtype=np.int64)


def bbox_intersect_area(b1: np.ndarray, b2s: np.ndarray) -> np.ndarray:
    """Intersection area between one axis-aligned box and an array of boxes."""
    if len(b2s.shape) == 1:
        b2s = b2s.copy().reshape(-1, 4)
    ix_max = np.maximum(b1[:2], b2s[:, :2])
    ix_min = np.minimum(b1[2:], b2s[:, 2:])
    return np.prod((ix_min - ix_max).clip(0), axis=1)


def contour_intersection(
        contour1: np.ndarray,
        contour2: np.ndarray,
        box1: np.ndarray,
        box2: np.ndarray,
    ) -> np.ndarray:
    """Overlapping pixel count of two contours.

    Both are rasterised onto a shared canvas spanning the union of their boxes,
    then multiplied element-wise.
    """
    if len(contour1) < 2 or len(contour2) < 2:
        return 0

    min_xy = np.minimum(box1[:2], box2[:2])
    max_xy = np.maximum(box1[2:], box2[2:]) + 1 - min_xy

    contour1 = contour1 - min_xy
    contour2 = contour2 - min_xy
    if contour1.min() < 0 or contour2.min() < 0:
        raise Exception("Negative contour coordinates")

    mask1, mask2 = np.zeros(max_xy[::-1], dtype=np.uint8), np.zeros(max_xy[::-1], dtype=np.uint8)
    cv2.drawContours(mask1, [contour1], -1, 1, thickness=cv2.FILLED)
    cv2.drawContours(mask2, [contour2], -1, 1, thickness=cv2.FILLED)
    return (mask1 * mask2).sum(dtype=np.int64)


def pairwise_contour_intersection(
        contours1: List[np.ndarray],
        contours2: Optional[List[np.ndarray]] = None,
        bboxes1: Optional[np.ndarray] = None,
        bboxes2: Optional[np.ndarray] = None,
        areas1: Optional[np.ndarray] = None,
        areas2: Optional[np.ndarray] = None,
    ) -> np.array:
    """Pairwise intersection matrix (n x m) between two groups of contours.

    Box overlap is used as a cheap pre-filter, so only candidate pairs are
    rasterised.
    """
    if contours2 is None:
        contours2 = contours1

    n = len(contours1)
    m = len(contours2)

    if bboxes1 is None:
        bboxes1 = np.array([contour_bbox(c) for c in contours1])
    if bboxes2 is None:
        bboxes2 = np.array([contour_bbox(c) for c in contours2])
    if areas1 is None:
        areas1 = np.array([contour_area(c) for c in contours1])
    if areas2 is None:
        areas2 = np.array([contour_area(c) for c in contours2])

    intersections = np.zeros((n, m), dtype=np.float32)

    for i in range(n):
        bintersect = bbox_intersect_area(bboxes1[i], bboxes2)
        for j in np.where(bintersect > 0)[0]:
            intersections[i, j] = contour_intersection(
                contours1[i], contours2[j], bboxes1[i], bboxes2[j])

    return intersections
