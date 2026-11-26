"""Co-registration of UAV bands taken by separate cameras (e.g. an RGB + NIR rig).

Two lenses a few centimetres apart never see the same footprint, so NIR must be warped
onto the RGB grid before NDVI-type indices make sense. Feature matching (ORB/SIFT +
RANSAC homography) handles large offsets; ECC refines small ones to sub-pixel accuracy.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np


def _u8(a: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(np.asarray(a, np.float32))
    lo, hi = np.percentile(a, (1, 99))
    return (np.clip((a - lo) / (hi - lo + 1e-12), 0, 1) * 255).astype(np.uint8)


def _cv2():
    try:
        import cv2
    except ImportError as e:
        raise ImportError("co-registration needs `pip install opencv-python`") from e
    return cv2


def estimate_homography(
    moving: np.ndarray,
    reference: np.ndarray,
    *,
    method: Literal["orb", "sift"] = "orb",
    n_features: int = 5000,
    ratio: float = 0.75,
    ransac_px: float = 3.0,
) -> tuple[np.ndarray, dict]:
    """3x3 homography mapping ``moving`` pixel coordinates onto ``reference``.

    Images are contrast-stretched to 8 bit, matched with ORB or SIFT and Lowe's ratio test,
    and the homography is fit with RANSAC. Returns ``(H, info)`` with match counts and the
    RMS reprojection error of the inliers.
    """
    cv2 = _cv2()
    m8, r8 = _u8(moving), _u8(reference)
    if method == "sift":
        det, norm = cv2.SIFT_create(nfeatures=n_features), cv2.NORM_L2
    else:
        det, norm = cv2.ORB_create(nfeatures=n_features), cv2.NORM_HAMMING
    km, dm = det.detectAndCompute(m8, None)
    kr, dr = det.detectAndCompute(r8, None)
    if dm is None or dr is None or len(km) < 4 or len(kr) < 4:
        raise RuntimeError("too few keypoints to register; try method='ecc'")
    pairs = cv2.BFMatcher(norm).knnMatch(dm, dr, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    if len(good) < 8:
        raise RuntimeError(f"only {len(good)} good matches; try method='ecc' or 'sift'")
    src = np.float32([km[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
    dst = np.float32([kr[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, ransac_px)
    if H is None:
        raise RuntimeError("homography estimation failed")
    inl = inl.ravel().astype(bool)
    proj = cv2.perspectiveTransform(src[inl], H)
    rmse = float(np.sqrt(np.mean(np.sum((proj - dst[inl]) ** 2, axis=-1))))
    return H, {"method": method, "matches": len(good), "inliers": int(inl.sum()), "rmse_px": rmse}


def refine_ecc(
    moving: np.ndarray,
    reference: np.ndarray,
    *,
    init: np.ndarray | None = None,
    motion: Literal["translation", "euclidean", "affine", "homography"] = "homography",
    iterations: int = 200,
    eps: float = 1e-6,
    gauss: int = 5,
) -> tuple[np.ndarray, dict]:
    """Enhanced Correlation Coefficient alignment.

    ``init`` is an optional starting homography mapping ``moving`` onto ``reference`` (as
    returned by :func:`estimate_homography`). Returns the refined homography in the same sense.
    """
    cv2 = _cv2()
    modes = {
        "translation": cv2.MOTION_TRANSLATION,
        "euclidean": cv2.MOTION_EUCLIDEAN,
        "affine": cv2.MOTION_AFFINE,
        "homography": cv2.MOTION_HOMOGRAPHY,
    }
    m, r = _u8(moving).astype(np.float32), _u8(reference).astype(np.float32)
    # ECC estimates the warp taking reference coords to moving coords (the inverse of H)
    start = np.eye(3, dtype=np.float32) if init is None else np.linalg.inv(init).astype(np.float32)
    warp = start if motion == "homography" else start[:2].copy()
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, eps)
    cc, warp = cv2.findTransformECC(r, m, warp, modes[motion], crit, None, gauss)
    W = warp if warp.shape == (3, 3) else np.vstack([warp, [0, 0, 1]])
    return np.linalg.inv(W), {"method": f"ecc-{motion}", "ecc": float(cc)}


def warp(
    moving: np.ndarray,
    H: np.ndarray,
    shape: tuple[int, int],
    *,
    order: int = 1,
    fill: float = np.nan,
) -> np.ndarray:
    """Warp ``moving`` with homography ``H`` onto a grid of ``shape`` (rows, cols)."""
    cv2 = _cv2()
    flags = (
        cv2.INTER_LINEAR if order == 1 else (cv2.INTER_NEAREST if order == 0 else cv2.INTER_CUBIC)
    )
    return cv2.warpPerspective(
        np.asarray(moving, np.float32),
        H,
        (shape[1], shape[0]),
        flags=flags,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(fill),
    )


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return -1.0
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def edge_agreement(a: np.ndarray, b: np.ndarray, *, margin: int = 6) -> float:
    """Correlation of the Sobel edge maps of two bands, away from NaN borders.

    A fair alignment score across bands (NIR and green differ in brightness but share edges);
    pixels within ``margin`` of a NaN (e.g. the empty border of a warped band) are ignored.
    """
    from scipy.ndimage import binary_erosion, sobel

    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    ok = np.isfinite(a) & np.isfinite(b)
    if margin:
        ok = binary_erosion(ok, iterations=margin)
    if ok.sum() < 10:
        return -1.0

    def mag(x: np.ndarray) -> np.ndarray:
        x = np.where(np.isfinite(x), x, np.nanmean(x))
        return np.hypot(sobel(x, 0), sobel(x, 1))

    return _ncc(np.where(ok, mag(a), np.nan), np.where(ok, mag(b), np.nan))


def coregister(
    moving: Any,
    reference: Any,
    *,
    method: Literal["orb", "sift", "ecc"] = "orb",
    refine: bool = True,
    fill: float = np.nan,
    validate: bool = True,
    min_inliers: int = 15,
    max_shift: float | None = None,
    **kw: Any,
) -> tuple[np.ndarray, dict]:
    """Align a 2-D ``moving`` band (e.g. NIR) to a 2-D ``reference`` band (e.g. green).

    ``orb``/``sift``: feature homography, then (``refine=True``) an ECC refinement that is
    kept only if it converges and raises the correlation with the reference. ``ecc``: ECC
    alone, for images already within a few pixels.

    With ``validate`` (default) a solution is rejected, and the band returned unwarped
    (identity), when it has fewer than ``min_inliers`` RANSAC inliers, shifts the image
    centre by more than ``max_shift`` pixels (default: a quarter of the shorter side), or
    lowers :func:`edge_agreement` with the reference. Repetitive crop rows can lure every
    method onto a neighbouring row; this check catches it. ``info["accepted"]`` and
    ``info["reason"]`` report the outcome.

    Returns the warped band and ``info`` (``H``, match counts, RMS error or ECC score).
    """
    mv = np.asarray(getattr(moving, "values", moving), np.float32)
    rf = np.asarray(getattr(reference, "values", reference), np.float32)
    if method == "ecc":
        H, info = refine_ecc(mv, rf, **kw)
    else:
        H, info = estimate_homography(mv, rf, method=method, **kw)
        if refine:
            try:
                H2, info2 = refine_ecc(mv, rf, init=H)
                if _ncc(warp(mv, H2, rf.shape), rf) > _ncc(warp(mv, H, rf.shape), rf):
                    H, info = H2, {**info, **info2}
            except Exception:  # noqa: BLE001  ECC may not converge; keep the feature solution
                pass
    out = warp(mv, H, rf.shape, fill=fill)
    info["accepted"], info["reason"] = True, ""
    unwarped = mv.copy() if mv.shape == rf.shape else warp(mv, np.eye(3), rf.shape, fill=fill)
    if validate:
        rows, cols = rf.shape
        centre = np.array([cols / 2, rows / 2, 1.0])
        moved = H @ centre
        shift = float(np.hypot(*(moved[:2] / moved[2] - centre[:2])))
        limit = max_shift if max_shift is not None else min(rows, cols) / 4
        before = edge_agreement(unwarped, rf)
        after = edge_agreement(out, rf)
        info.update(shift_px=shift, edge_before=before, edge_after=after)
        reason = ""
        if "inliers" in info and info["inliers"] < min_inliers:
            reason = f"only {info['inliers']} inliers (< {min_inliers})"
        elif shift > limit:
            reason = f"shift {shift:.0f} px > {limit:.0f} px"
        elif after < before:
            reason = f"edge agreement fell ({before:.3f} -> {after:.3f})"
        if reason:
            H, out = np.eye(3), unwarped
            info["accepted"], info["reason"] = False, reason
    info["H"] = H
    info["correlation"] = _ncc(out, rf)
    return out, info
