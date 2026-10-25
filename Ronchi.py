import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.signal import find_peaks
from scipy.optimize import brentq
from scipy.interpolate import griddata
from scipy.spatial import cKDTree, ConvexHull
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import cumulative_trapezoid
from scipy.special import eval_jacobi
from scipy.io import savemat
from numpy.polynomial import legendre as npleg
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import tempfile
import os
import time

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
# Vertical alignment verification
ANGLE_THRESHOLD   = 0.01   # max residual inclination accepted (degrees). The Hough angle
                           # is only a first estimate: with curved fringes (e.g. spherical
                           # aberration) it can be off by ~0.5 deg, so the line fit refines it
ALIGN_MAX_ITER    = 5      # max number of correction iterations
ALIGN_N_CENTRAL   = 3      # number of central stripes used to check verticality

# Stripe segmentation
BIN_METHOD         = "adaptive"  # "adaptive": local threshold (robust to uneven brightness)
                                 # "fixed": global threshold BIN_THRESHOLD
BIN_THRESHOLD      = 195   # threshold used when BIN_METHOD = "fixed"
ADAPTIVE_C         = -20   # adaptive: a pixel is white if it exceeds its local mean by 20 levels
                           # (after the contrast of the fringe channel is stretched to 0-255)
ADAPTIVE_BLOCK     = 0.25  # adaptive: local window size as a fraction of the image size
MIN_COMPONENT_AREA = 300   # connected components smaller than this are discarded (px).
                           # Low on purpose: dim fringes are kept; false detections are
                           # removed by the user in the fringe review
CLOSE_KERNEL       = 10    # size of the square kernel of the morphological closing (px)
CURVED_CLOSE_KERNEL = (3, 10)   # (width, height) of the closing used with curved fringes
                           # (fringe following): elongated along the fringes, so breaks along
                           # a fringe are still joined but neighbouring fringes that are very
                           # close together (compressed by the aberrations) are not merged
SPURIOUS_GAP       = 0.6   # peaks closer than this fraction of the median gap are spurious

# Centroid lines
N_ANCHORS   = 15           # anchor points per stripe used to fit its center curve
POLY_DEGREE = 2            # minimum degree of the polynomial fitted to each stripe
POLY_MAX_DEGREE = 2        # maximum degree (chosen with the residual, BIC, if > 2).
                           # Kept at 2: with real Ronchigrams higher degrees follow the
                           # irregularities of the fringes (worse agreement between
                           # diffraction orders of the same surface)
CURVED_MAX_DEGREE = 6      # maximum degree when the fringes were found by fringe following
                           # (curved, S-shaped fringes that a parabola cannot follow)
SUBPIXEL_ANCHORS = False   # True: anchor points from intensity-weighted centroids (more
                           # exact with clean simulated fringes, less robust with real ones)
MIN_GOOD_ANCHORS = 5       # sub-pixel anchor points needed to accept a stripe automatically
WINDOW_LEAVE_MAX = 0.38    # fringe following is used when the fringes touch the border of
                           # their vertical search window in more than this fraction of
                           # rows (curved fringes). Real, nearly vertical Ronchigrams stay
                           # below 0.25; strongly curved ones are above 0.5
MIN_FRINGE_CHORD = 0.5     # fringe following: a fringe must span at least this fraction of
                           # the pupil chord at its position (shorter pieces are discarded)
MIN_STRIPE_PIXELS = 200    # white pixels needed in a stripe window to look for anchors
MIN_MANUAL_ANCHORS = 3     # anchor points needed per fringe in the fringe review
EDGE_TOL = 0.015           # a fringe row ending closer than this fraction of the pupil radius
                           # (at least 3 px) to the pupil border is cut by the pupil
TA_POSITION = "experimental"   # where each TA value is located in the pupil:
                           # "experimental" = on the measured fringe (Optical Shop Testing,
                           # Eq. 9.30: m and m0 are compared at the same point (x, y));
                           # "theoretical" = on the theoretical line (previous versions)
EDGE_PARTIAL = 0.35        # a fringe next to the pupil border narrower than this fraction of
                           # its neighbour is only the visible part of a cut fringe
EDGE_MIN_ROWS = 5          # rows not cut by the pupil needed to measure the fringe width
POINT_STEP  = 5            # distance (px) between consecutive points of a centroid line.
                           # Nyquist criterion: points per line = length / 5  (<= image size / 5)

# Point cloud refinement (multilinear interpolation, PCM)
REFINE_K1     = 5          # points inserted between neighbouring lines in the 1st refinement
REFINE_K2     = 10         # points inserted between neighbouring lines in the 2nd refinement
MIN_SPACING   = 1          # refined points are never closer than this (px).
                           # >= 1 so there are never more points than pixels (Optik article)
INTERP_METHOD = "linear"   # vertical/horizontal: "linear" = multilinear (Delaunay),
                           #                      "cubic" = spline (Clough-Tocher)
GRID_INTERP_METHOD = "cubic"  # grid: dx and dy interpolated separately with this method

# Extension of the point clouds to the whole exit pupil
FULL_PUPIL_DEGREE = 4      # total degree of the polynomial used outside the data region
                           # (4 = degree of the Kingslake primary aberrations polynomial).
                           # High degrees (e.g. 11) diverge where there are no data.
FULL_PUPIL_DECAY  = 20     # px; the fit residual at the data border decays with this
                           # length, so the extension joins the data without a step

# Optical surface under test (user inputs in the GUI; defaults = lens L1 of the
# manuscript: D = 39 mm, f = 237 mm, z = 173.5 mm; R assumes a plano-convex BK7 lens)
SURFACE_DEFAULTS = {"R": 122.06,   # radius of curvature of the tested surface (mm)
                    "f": 237.0,    # focal length (mm)
                    "D": 39.0,     # diameter of the aperture (mm)
                    "K": 0.0,      # conic constant (0 sphere, -1 paraboloid, ...)
                    "z": 173.5}    # distance from the surface vertex to the plane where
                                   # the Ronchigram was recorded (camera plane, mm)

PUPIL_MARGIN = 1.08        # size of the working square relative to the pupil diameter

# Wavefront (same procedure as Arriaga et al., Appl. Phys. B 132:14, 2026, Eqs. 9-15)
# Units: TA converted to mm with the scale of the ideal ray mapping (surface
# parameters), W in micrometers
WAVEFRONT_SOURCE = "vertical_horizontal"  # TA_x from the vertical matrix and TA_y from the
                                          # horizontal one; "grid": dx, dy of the grid matrix
TA_FIT_DEGREE  = 11        # Eq. 12: TA ~ sum_{l,h<=11} B_lh cos^l(t) sin^h(t) rho^(l+h)
RADIAL_STEPS   = 2000      # Eq. 14: divisions of rho in the Riemann sum (article: 1e5)
ANGULAR_STEPS  = 360       # Eq. 14: directions theta
ZERNIKE_Q      = 50        # Eq. 15: default Zernike degree n (n = 50 -> 1326 coefficients).
                           # The GUI lets the user change it before running.
ZERNIKE_GRID   = 201       # samples per side of W_r used for the Zernike fit
MAP_GRID       = 151       # samples per side of the maps shown in the GUI
WAVEFRONT_SIGN = -1        # W = -integral(TA): sign convention checked with simulated
                           # Ronchigrams of known wavefront
REMOVE_TERMS   = (0, 1, 2, 4)  # OSA/ANSI indices removed for display: piston, tilts, defocus
FIG_DPI        = 150       # resolution of the plots shown in the GUI

# Fringe colors in the fringe review (same colors in the editor and in its result step)
FE_COLORS = ["#00D000", "#00B8FF", "#FF9F00", "#FF30FF", "#00E8C0", "#FFE000",
             "#FF5050", "#A8A8FF"]
FE_RGB = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in FE_COLORS]

COLORS_ON_WHITE = [
    (0, 150, 0), (0, 110, 220), (230, 120, 0),
    (180, 0, 180), (0, 150, 150), (140, 100, 0)
]

COLORS = [
    (0, 255, 0), (0, 200, 255), (255, 165, 0),
    (255, 0, 255), (0, 255, 200), (255, 255, 0)
]


# ══════════════════════════════════════════════════════════════════════════════
# GENERIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def figToArray(fig):
    # Render a matplotlib figure into an RGB numpy array and close it
    fig.set_dpi(FIG_DPI)
    fig.patch.set_facecolor("white")
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(
        canvas.get_width_height()[::-1] + (4,)
    )
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    plt.close(fig)
    return img


def circleMask(shape, pupil):
    # Binary mask (255 inside) of the circular exit pupil
    cx, cy, r = pupil
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2).astype(np.uint8) * 255


def insidePupil(x, y, pupil):
    cx, cy, r = pupil
    return (np.asarray(x) - cx) ** 2 + (np.asarray(y) - cy) ** 2 <= r ** 2


def drawCurve(img, poly, a0, a1, color, thickness, vertical=True):
    # vertical=True : x = poly(y), y in [a0, a1]
    # vertical=False: y = poly(x), x in [a0, a1]
    a = np.linspace(a0, a1, 400)
    b = poly(a)
    pts = np.stack([b, a], axis=1) if vertical else np.stack([a, b], axis=1)
    h, w = img.shape[:2]
    ok = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    pts = np.round(pts[ok]).astype(np.int32)
    if len(pts) > 1:
        cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness)


def stripeChannel(rgb):
    # Color channel that carries the fringes (manuscript, Eq. 6): the one with the
    # largest contrast (standard deviation), so the color of the fringes does not
    # matter. Its intensity is stretched to 0-255 (percentiles 1 and 99.7) so dim
    # Ronchigrams are processed like bright ones.
    k = int(np.argmax([float(rgb[:, :, c].std()) for c in range(3)]))
    ch = rgb[:, :, k].astype(np.float32)
    lo, hi = np.percentile(ch, [1.0, 99.7])
    if hi - lo < 1.0:
        return rgb[:, :, k].copy()
    return np.clip((ch - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def detectPupil(rgb):
    # Locate the circular exit pupil (bright region of the fringe channel)
    # Returns (cx, cy, r) using the convex hull of the largest bright region
    R = stripeChannel(rgb)
    h, w = R.shape
    k = max(3, (min(h, w) // 40) | 1)
    blur = cv2.GaussianBlur(R, (k, k), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Close the dark gaps between stripes so the pupil becomes a single region
    ks = max(5, min(h, w) // 8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("Could not detect the exit pupil.")
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    M = cv2.moments(hull)
    if M["m00"] == 0:
        raise RuntimeError("Could not detect the exit pupil.")
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    r = np.sqrt(cv2.contourArea(hull) / np.pi)
    return float(cx), float(cy), float(r)


def rotateImage(rgb, angle):
    # Rotation about the image center. Lanczos interpolation keeps the fringe profiles
    # (bilinear interpolation smooths them and shifts their centers by fractions of px)
    h, w = rgb.shape[:2]
    if abs(angle) < 1e-9:
        return rgb.copy()
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT)


# ══════════════════════════════════════════════════════════════════════════════
# STRIPE DETECTION (same algorithm for experimental and theoretical Ronchigrams)
# ══════════════════════════════════════════════════════════════════════════════
def growWideFringes(binary, gray):
    # A fringe much wider than the window of the adaptive threshold is not detected in
    # its middle: every pixel there is about as bright as its neighbourhood, so only
    # its narrow ends are found. The detected fringes are grown into the adjacent
    # pixels that are as bright as the fringes themselves (>= 90 % of the local fringe
    # brightness). Only large regions are added (a missing wide fringe); small changes
    # at the borders of normal fringes are not made, so ordinary Ronchigrams are
    # segmented exactly as before.
    white = binary > 0
    if not white.any():
        return binary
    size = min(binary.shape)
    g = gray.astype(np.float32)
    sigma = max(5.0, 0.06 * size)
    num = cv2.GaussianBlur(g * white, (0, 0), sigma)
    den = cv2.GaussianBlur(white.astype(np.float32), (0, 0), sigma)
    level = np.where(den > 0.05, num / np.maximum(den, 1e-6), float(g[white].mean()))
    cand = ((g >= 0.9 * level) & ~white).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    near = cv2.dilate(white.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    out = binary.copy()
    minArea = (0.12 * size) ** 2
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < minArea:
            continue
        comp = labels == k
        if (comp & near).any():
            out[comp] = 255
    return out


def binarizeStripes(rgb, closeKernel=None):
    # CLAHE on the fringe channel -> threshold -> remove small components -> closing
    # closeKernel: (width, height) of the closing; default CLOSE_KERNEL x CLOSE_KERNEL
    R = stripeChannel(rgb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    Rb = clahe.apply(R)
    if BIN_METHOD == "adaptive":
        # Local (Gaussian-weighted) mean threshold: works when some stripes are
        # dimmer than others (non-uniform illumination of the pupil)
        block = max(3, int(min(Rb.shape) * ADAPTIVE_BLOCK) | 1)
        binary = cv2.adaptiveThreshold(cv2.GaussianBlur(Rb, (5, 5), 0), 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block, ADAPTIVE_C)
    else:
        _, binary = cv2.threshold(Rb, BIN_THRESHOLD, 255, cv2.THRESH_BINARY)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] > MIN_COMPONENT_AREA
    clean = np.where(keep[labels], 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       closeKernel or (CLOSE_KERNEL, CLOSE_KERNEL))
    closed = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    closed = growWideFringes(closed, R)
    return Rb, binary, clean, closed


def columnProfile(binImg, sigma=5):
    # Average intensity per column, smoothed with a 1D Gaussian
    profile = np.mean(binImg, axis=0).astype(np.float32)
    k = int(sigma * 6) | 1
    smooth = cv2.GaussianBlur(profile.reshape(1, -1), (k, 1), sigma).flatten()
    return profile, smooth


def detectStripePeaks(smooth):
    # Two-pass peak detection + removal of spurious peaks (too close to a neighbour)
    rng = float(smooth.max() - smooth.min())
    if rng <= 0:
        raise RuntimeError("No stripes detected (empty intensity profile).")
    minProm = rng * 0.10
    peaksR, _ = find_peaks(smooth, prominence=minProm)
    if len(peaksR) < 2:
        raise RuntimeError(
            f"Only {len(peaksR)} stripe(s) detected. At least 2 are needed to estimate "
            "the spacing. Check the image, the crop or the binarization threshold."
        )
    minDist = max(int(np.median(np.diff(peaksR)) * 0.4), 10)
    peaks, _ = find_peaks(smooth, prominence=minProm, distance=minDist)

    # A peak much closer to its neighbour than the median gap is a fragment
    # (typically at the pupil edge): keep only the stronger one
    peaks = list(peaks)
    removed = []
    while len(peaks) > 2:
        gaps = np.diff(peaks)
        i = int(np.argmin(gaps))
        if gaps[i] >= SPURIOUS_GAP * np.median(gaps):
            break
        a, b = peaks[i], peaks[i + 1]
        weak = a if smooth[a] < smooth[b] else b
        peaks.remove(weak)
        removed.append(int(weak))
    peaks = np.array(peaks)
    if len(peaks) < 2:
        raise RuntimeError(f"Only {len(peaks)} stripe(s) remain after peak filtering.")
    medianGap = float(np.median(np.diff(peaks)))
    return peaksR, peaks, medianGap, minProm, removed


def rowCentroid(grayRow, binRow, guess, halfWin, xMin=None, xMax=None):
    # Sub-pixel center of a bright fringe in one row: intensity-weighted centroid
    # (Eq. 14 of the manuscript, intensity version) in a window centered on the
    # current estimate, re-centered twice so the window is symmetric about the fringe.
    # The background (window minimum) is subtracted from the weights.
    n = len(grayRow)
    c = float(guess)
    for _ in range(3):
        # Near the pupil border the fringe is cut: the window is shrunk symmetrically
        # so it stays inside the pupil (otherwise the centroid is pulled inwards).
        # If it becomes too small, None is returned and the binary center is used.
        hw = halfWin
        if xMin is not None:
            hw = min(halfWin, c - xMin, xMax - c)
            if hw < 0.35 * halfWin:
                return None
        a, b = int(max(0, np.floor(c - hw))), int(min(n, np.ceil(c + hw) + 1))
        if b - a < 3:
            return None
        seg = grayRow[a:b].astype(np.float64)
        wts = seg - seg.min()
        if wts.sum() <= 0:
            return None
        c = a + float((wts * np.arange(b - a)).sum() / wts.sum())
    return c


def fringeRowCenters(rr, cnt, sc, lo, hi, pupil, refWidth=None):
    # Center of a fringe in each of its rows (rr), from its white pixels: cnt = number,
    # sc = sum of their columns, [lo, hi) = extent.
    # Inside the pupil the center is the center of mass of the white pixels. Where the
    # pupil border cuts the fringe, only part of its width is visible and the center of
    # mass is pulled towards the pupil center; there the center is measured from the
    # edge of the fringe that IS visible: (inner edge) +- (fringe width) / 2, with the
    # width measured on the nearest rows that are not cut. A row is discarded (NaN) if
    # the center obtained lies outside the pupil (the crest of the fringe is not in the
    # image) or if the fringe is cut on both sides.
    # Returns (centers, median width of the rows that are not cut or None).
    center = sc / np.maximum(cnt, 1)
    if pupil is None:
        return center, None
    pcx, pcy, pr = pupil
    half = np.sqrt(np.maximum(pr ** 2 - (rr - pcy) ** 2, 0.0))
    xL, xR = pcx - half, pcx + half
    tol = max(3.0, EDGE_TOL * pr)
    cutL, cutR = lo <= xL + tol, hi >= xR - tol
    cut = cutL | cutR
    span = (hi - lo).astype(np.float64)
    free = np.where(~cut)[0]
    width = float(np.median(span[free])) if len(free) >= EDGE_MIN_ROWS else None
    if not cut.any():
        return center, width
    center = center.astype(np.float64).copy()
    for i in np.where(cut)[0]:
        if cutL[i] and cutR[i]:
            center[i] = np.nan
            continue
        if width is not None:
            near = free[np.argsort(np.abs(rr[free] - rr[i]))[:15]]
            wRef = float(np.median(span[near]))
        elif refWidth is not None:
            wRef = float(refWidth)
        else:
            center[i] = np.nan
            continue
        wRef = max(wRef, span[i])
        c = hi[i] - wRef / 2.0 if cutL[i] else lo[i] + wRef / 2.0
        # at least 3/4 of the fringe must be visible
        center[i] = c if (xL[i] + 0.25 * wRef <= c <= xR[i] - 0.25 * wRef) else np.nan
    return center, width


def bandAnchors(rr, cnt, center, nAnchors):
    # Anchor points of a fringe: its rows are divided in nAnchors bands and the anchor
    # point of each band is the mean of the row centers, weighted by the number of
    # white pixels of each row (center of mass, Eqs. 14-15 of the manuscript)
    ok = np.isfinite(center) & (cnt > 0)
    rr, cnt, center = rr[ok], cnt[ok].astype(np.float64), center[ok]
    if len(rr) < 2:
        return [], None, None
    firstRow, lastRow = int(rr[0]), int(rr[-1])
    band = (lastRow - firstRow) // (nAnchors - 1)
    if band < 2:
        return [], firstRow, lastRow
    anchors = []
    for b in range(nAnchors):
        if b == 0:
            r0, r1 = firstRow, firstRow + max(band // 2, 1)
        elif b == nAnchors - 1:
            r0, r1 = lastRow - max(band // 2, 1), lastRow + 1
        else:
            r0 = firstRow + b * band - band // 2
            r1 = r0 + band
        sel = (rr >= r0) & (rr < r1)
        if cnt[sel].sum() == 0:
            continue
        anchors.append((float((rr[sel] * cnt[sel]).sum() / cnt[sel].sum()),
                        float((center[sel] * cnt[sel]).sum() / cnt[sel].sum())))
    return anchors, firstRow, lastRow


def getAnchorPoints(binImg, peaks, halfWidth, nAnchors=N_ANCHORS, gray=None, pupil=None):
    # For each stripe, divide its vertical extent in nAnchors bands. The anchor point
    # of each band is the mean of the sub-pixel intensity centroids of its rows
    # (or the center of mass of the white pixels when no gray image is given).
    h, w = binImg.shape
    if gray is None:
        # Row data of every fringe inside its vertical window
        data = []
        for peakCol in peaks:
            minCol = max(0, int(peakCol) - halfWidth)
            maxCol = min(w, int(peakCol) + halfWidth)
            strip = binImg[:, minCol:maxCol] > 0
            rr = np.where(strip.any(axis=1))[0]
            if len(rr) == 0 or strip.sum() < MIN_STRIPE_PIXELS:
                continue
            sub = strip[rr]
            cnt = sub.sum(axis=1).astype(np.float64)
            sc = (sub * np.arange(minCol, maxCol)).sum(axis=1).astype(np.float64)
            lo = minCol + sub.argmax(axis=1)
            hi = maxCol - sub[:, ::-1].argmax(axis=1)
            data.append((int(peakCol), rr, cnt, sc, lo, hi))
        return fringesFromRows(data, pupil, nAnchors)
    stripes = []
    for peakCol in peaks:
        minCol = max(0, int(peakCol) - halfWidth)
        maxCol = min(w, int(peakCol) + halfWidth)
        strip = binImg[:, minCol:maxCol]
        rows = np.where(np.any(strip > 0, axis=1))[0]
        if len(rows) == 0 or strip.sum() < MIN_STRIPE_PIXELS * 255:
            continue
        firstRow, lastRow = rows[0], rows[-1]
        band = (lastRow - firstRow) // (nAnchors - 1)
        if band < 2:
            continue
        anchors = []
        for b in range(nAnchors):
            if b == 0:
                r0, r1 = firstRow, firstRow + max(band // 2, 1)
            elif b == nAnchors - 1:
                r0, r1 = lastRow - max(band // 2, 1), lastRow + 1
            else:
                r0 = firstRow + b * band - band // 2
                r1 = r0 + band
            r0, r1 = max(0, r0), min(h, r1)
            sub = strip[r0:r1]
            if sub.sum() == 0:
                continue
            if gray is None:
                rr, cc = np.where(sub > 0)
                anchors.append((r0 + rr.mean(), minCol + cc.mean()))
                continue
            pts, nRows = [], 0
            for rr_ in range(r0, r1):
                white = np.where(strip[rr_] > 0)[0]
                if len(white) == 0:
                    continue
                nRows += 1
                xMin = xMax = None
                if pupil is not None:
                    pcx, pcy, pr = pupil
                    half = pr ** 2 - (rr_ - pcy) ** 2
                    if half <= 0:
                        continue
                    half = np.sqrt(half) - 2.0
                    xMin, xMax = pcx - half, pcx + half
                cx_ = rowCentroid(gray[rr_], binImg[rr_], minCol + white.mean(),
                                  0.75 * halfWidth, xMin, xMax)
                if cx_ is not None:
                    pts.append((rr_, cx_))
            if pts and len(pts) >= 0.3 * max(nRows, 1):
                pts = np.array(pts)
                anchors.append((float(pts[:, 0].mean()), float(pts[:, 1].mean())))
            # (bands where the fringe is cut by the pupil border give no anchor point:
            #  their center would be biased; that zone is covered by the extension)
        # Accept the stripe only if it is continuous enough
        minAnchors = MIN_GOOD_ANCHORS if gray is not None else nAnchors * 0.6
        if len(anchors) >= minAnchors:
            stripes.append({"peak": int(peakCol), "anchors": anchors})
    return stripes


def fringesFromRows(data, pupil, nAnchors=N_ANCHORS, minChord=None):
    # data: one (peak, rows, count, sum of columns, lo, hi) per fringe.
    # Row centers (corrected where the pupil cuts the fringe) -> anchor points.
    # A fringe that is cut by the pupil in all its rows takes the width of the nearest
    # fringe that is not; it is kept only if its crest is inside the pupil.
    first = [fringeRowCenters(rr, cnt, sc, lo, hi, pupil) for (_, rr, cnt, sc, lo, hi) in data]
    xm = [float(sc.sum() / max(cnt.sum(), 1)) for (_, rr, cnt, sc, lo, hi) in data]
    known = [(xm[i], first[i][1]) for i in range(len(data)) if first[i][1] is not None]
    stripes = []
    for i, (peak, rr, cnt, sc, lo, hi) in enumerate(data):
        center = first[i][0]
        others = [t for t in known if t[0] != xm[i]]
        ref = min(others, key=lambda t: abs(t[0] - xm[i]))[1] if others else None
        if pupil is not None and ref is not None:
            if first[i][1] is None:
                center, _ = fringeRowCenters(rr, cnt, sc, lo, hi, pupil, refWidth=ref)
            elif (first[i][1] < EDGE_PARTIAL * ref and
                  abs(xm[i] - pupil[0]) > pupil[2] - ref):
                # Sliver next to the pupil border, much narrower than its neighbour:
                # the visible part of a fringe whose crest is outside the pupil
                continue
        anchors, firstRow, lastRow = bandAnchors(rr, cnt, center, nAnchors)
        # Accept the fringe only if it is continuous enough (and long enough)
        if len(anchors) < nAnchors * 0.6:
            continue
        okc = np.isfinite(center)
        meanCol = float((center[okc] * cnt[okc]).sum() / cnt[okc].sum())
        if minChord is not None and pupil is not None:
            half = pupil[2] ** 2 - (meanCol - pupil[0]) ** 2
            chord = 2.0 * np.sqrt(half) if half > 0 else 0.0
            if (lastRow - firstRow) < minChord * chord:
                continue
        stripes.append({"peak": int(peak) if peak is not None else int(round(meanCol)),
                        "anchors": anchors})
    return stripes


def trackFringes(binImg, pupil=None, nAnchors=N_ANCHORS):
    # Fringe following. Each fringe is located on the center row of the pupil and then
    # followed upwards and downwards row by row: in every row the fringe is the white
    # run (or runs) connected to its extent in the previous row. The search therefore
    # moves with the fringe, so curved or S-shaped fringes are followed along their
    # whole length (a fixed vertical window only works for nearly vertical fringes).
    # White regions that do not cross the center row are followed from their own middle
    # row. The anchor points are, as before, the center of mass of the white pixels of
    # the fringe in nAnchors bands along its length (Eqs. 14-15 of the manuscript).
    h, w = binImg.shape
    white = binImg > 0
    claimed = np.zeros((h, w), dtype=bool)
    cy = int(round(pupil[1])) if pupil is not None else h // 2
    cy = min(max(cy, 0), h - 1)
    size = 2.0 * pupil[2] if pupil is not None else float(h)
    maxGap = max(12, int(0.12 * size))        # rows without white pixels tolerated

    def rowRuns(r):
        row = white[r] & ~claimed[r]
        d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        return np.where(d == 1)[0], np.where(d == -1)[0]

    def follow(r0, L0, R0):
        rows = {}
        for direction in (1, -1):
            L, R, wS, gap = L0, R0, float(R0 - L0), 0
            r = r0 if direction == 1 else r0 - 1
            while 0 <= r < h:
                st, en = rowRuns(r)
                m = 2 + 0.5 * gap
                sel = (en > L - m) & (st < R + m)
                found = False
                if sel.any():
                    a, b = int(st[sel].min()), int(en[sel].max())
                    if (b - a) > 1.6 * wS + 4:
                        # Two fringes touch in the binary image: keep only the part
                        # that continues this fringe
                        a, b = max(a, int(L - 0.25 * wS)), min(b, int(R + 0.25 * wS))
                    cols = np.where(white[r, a:b] & ~claimed[r, a:b])[0] + a
                    if len(cols):
                        rows[r] = (float(cols.sum()), len(cols), int(cols.min()),
                                   int(cols.max()) + 1)
                        claimed[r, a:b] = True
                        L, R = int(cols.min()), int(cols.max()) + 1
                        wS = 0.8 * wS + 0.2 * (R - L)
                        gap, found = 0, True
                if not found:
                    gap += 1
                    if gap > maxGap:
                        break
                r += direction
        return rows

    # Seeds: white runs of the center row (averaged over a few rows)
    band = max(2, int(0.01 * size))
    seedRow = white[max(0, cy - band):min(h, cy + band + 1)].mean(axis=0) > 0.5
    d = np.diff(np.concatenate(([0], seedRow.astype(np.int8), [0])))
    fringes = [follow(cy, int(a), int(b))
               for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]) if b - a >= 2]
    # White regions that do not cross the center row
    for _ in range(60):
        rest = (white & ~claimed).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(rest)
        if n <= 1:
            break
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[k, cv2.CC_STAT_AREA] < MIN_STRIPE_PIXELS:
            break
        ys = np.where((labels == k).any(axis=1))[0]
        rm = int(ys[len(ys) // 2])
        xs = np.where(labels[rm] == k)[0]
        rows = follow(rm, int(xs.min()), int(xs.max()) + 1)
        if not rows:
            claimed[labels == k] = True
            continue
        fringes.append(rows)

    data = []
    for rows in fringes:
        if not rows:
            continue
        rr = np.array(sorted(rows))
        sc = np.array([rows[r][0] for r in rr], dtype=np.float64)
        cnt = np.array([rows[r][1] for r in rr], dtype=np.float64)
        if cnt.sum() < MIN_STRIPE_PIXELS:
            continue
        lo = np.array([rows[r][2] for r in rr])
        hi = np.array([rows[r][3] for r in rr])
        data.append((None, rr, cnt, sc, lo, hi))
    stripes = fringesFromRows(data, pupil, nAnchors, minChord=MIN_FRINGE_CHORD)
    stripes.sort(key=lambda s_: float(np.mean([p[1] for p in s_["anchors"]])))
    return stripes


def windowLeaveFraction(binImg, peaks, halfWidth):
    # Fraction of the rows of the fringes in which the white pixels touch the left or
    # right border of the vertical search window: low for nearly vertical fringes,
    # high when the fringes are curved and leave the window
    h, w = binImg.shape
    touched = total = 0
    for p in peaks:
        a, b = max(0, int(p) - halfWidth), min(w, int(p) + halfWidth)
        if b - a < 2:
            continue
        strip = binImg[:, a:b] > 0
        rows = strip.any(axis=1)
        total += int(rows.sum())
        touched += int((strip[:, 0] | strip[:, -1])[rows].sum())
    return touched / max(total, 1)


def detectFringes(binImg, pupil, peaks, halfWidth, gray=None):
    # Anchor points of the fringes.
    #  - Nearly vertical fringes (the case of the Ronchi-SLM setup): vertical search
    #    window around each peak of the column profile, as described in the manuscript.
    #  - Curved fringes (they leave their windows) or no valid peaks: fringe following.
    # Returns (stripes, method, fraction of rows leaving the windows).
    leave = None
    if len(peaks) >= 2:
        leave = windowLeaveFraction(binImg, peaks, halfWidth)
        if leave <= WINDOW_LEAVE_MAX:
            stripes = getAnchorPoints(binImg, peaks, halfWidth, pupil=pupil, gray=gray)
            if len(stripes) >= 2:
                return stripes, "vertical windows on the column-profile peaks", leave
    return trackFringes(binImg, pupil), "fringe following (curved fringes)", leave


def fitCenterline(rows, cols, maxDegree=None):
    # Polynomial x = p(y) through the anchor points. The degree goes from POLY_DEGREE
    # to POLY_MAX_DEGREE and is chosen with the Bayesian information criterion, so a
    # higher degree is used only when the residual really decreases.
    rows, cols = np.asarray(rows, float), np.asarray(cols, float)
    n = len(rows)
    best, bestBic = None, np.inf
    if maxDegree is None:
        maxDegree = POLY_MAX_DEGREE
    for d in range(POLY_DEGREE, maxDegree + 1):
        if n < d + 3 and d > POLY_DEGREE:
            break
        dd = min(d, n - 1)
        poly = np.poly1d(np.polyfit(rows, cols, dd))
        rss = float(np.sum((poly(rows) - cols) ** 2)) + 1e-12
        bic = n * np.log(rss / n) + (dd + 1) * np.log(n)
        if bic < bestBic - 2.0 or best is None:      # require a clear improvement
            best, bestBic = poly, bic
    return best


def fringePositions(smooth, stripes, halfWidth):
    # Position of each confirmed fringe in the column profile: the profile maximum
    # near the fringe (same definition as the automatic peaks), or the mean column of
    # its anchor points if the profile has no signal there (e.g. a dim fringe added
    # by hand). The median difference of these positions is the fringe spacing.
    pos = []
    top = float(np.max(smooth)) if len(smooth) else 0.0
    for s_ in stripes:
        xm = float(np.mean([p[1] for p in s_["anchors"]]))
        a, b = int(max(0, xm - halfWidth)), int(min(len(smooth), xm + halfWidth + 1))
        seg = smooth[a:b]
        if seg.size and top > 0 and seg.max() > 0.05 * top:
            pos.append(float(a + int(np.argmax(seg))))
        else:
            pos.append(xm)
    return np.sort(pos)


def buildCentroidLines(binImg, peaks, medianGap, pupil, stripes=None, halfWidth=None,
                       maxDegree=None):
    # Centroid line of each stripe: polynomial through its anchor points,
    # sampled every POINT_STEP px (rows aligned to multiples of POINT_STEP)
    # and restricted to the exit pupil. If 'stripes' is given (anchor points
    # confirmed by the user), they are used instead of the automatic ones.
    if halfWidth is None:
        halfWidth = max(int(medianGap * 0.4), 10)
    if stripes is None:
        stripes = getAnchorPoints(binImg, peaks, halfWidth)
    lines = []
    for s in stripes:
        rows = np.array([p[0] for p in s["anchors"]])
        cols = np.array([p[1] for p in s["anchors"]])
        if len(rows) < POLY_DEGREE + 1:
            continue
        poly = fitCenterline(rows, cols, maxDegree)
        ys = np.arange(np.ceil(rows.min() / POINT_STEP) * POINT_STEP,
                       rows.max() + 1e-9, POINT_STEP, dtype=np.float64)
        xs = poly(ys)
        ok = insidePupil(xs, ys, pupil)
        ys, xs = ys[ok], xs[ok]
        if len(ys) < 3:
            continue

        # Stripe width: white pixels per row around the curve (central 50% of the stripe)
        widths = []
        yA, yB = ys.min(), ys.max()
        for y in np.arange(yA + 0.25 * (yB - yA), yB - 0.25 * (yB - yA), 1.0):
            xc = int(round(poly(y)))
            row = binImg[int(y), max(0, xc - halfWidth):xc + halfWidth]
            nz = np.count_nonzero(row)
            if nz > 0:                      # manual fringes may have no white pixels
                widths.append(nz)
        width = float(np.median(widths)) if widths else np.nan

        lines.append({"poly": poly, "anchors": s["anchors"], "ys": ys, "xs": xs,
                      "a0": float(ys.min()), "a1": float(ys.max()),
                      "width": width, "xMean": float(xs.mean())})
    lines.sort(key=lambda L: L["xMean"])
    return lines, halfWidth


def surfaceMapping(R, f, D, K, z, n_samples=4000):
    # Ideal Ronchi-SLM geometry (manuscript, Fig. 1): the grating is displayed in the
    # collimated beam, the surface under test (conic of radius R and conic constant K,
    # plano-convex lens of focal length f, so n = 1 + R/f) refracts every ray, and the
    # Ronchigram is recorded at a distance z from the surface vertex.
    # Each ray entering at height h (0 <= h <= D/2) is traced exactly and reaches the
    # camera plane at x(h). This radial mapping entrance -> camera is what bends the
    # straight grating lines in the ideal (theoretical) Ronchigram.
    if min(R, f, D, z) <= 0:
        raise RuntimeError("R, f, D and z must be positive.")
    n = 1.0 + R / f
    h = np.linspace(0.0, D / 2.0, n_samples + 1)
    arg = 1.0 - (1.0 + K) * h ** 2 / R ** 2
    if np.any(arg < 0):
        raise RuntimeError("The conic surface is not defined over the whole diameter "
                           "(check R, K and D).")
    sag = h ** 2 / (R * (1.0 + np.sqrt(arg)))
    # Unit normal of the conic, pointing to the incident side
    nx, nz = h, (1.0 + K) * sag - R
    L = np.hypot(nx, nz)
    nx, nz = nx / L, nz / L
    # Refraction air -> glass (incident direction (0, 1))
    cosI = -nz
    eta = 1.0 / n
    k1 = 1.0 - eta ** 2 * (1.0 - cosI ** 2)
    if np.any(k1 < 0):
        raise RuntimeError("Total internal reflection at the surface (check R, f, K, D).")
    tx = (eta * cosI - np.sqrt(k1)) * nx
    tz = eta + (eta * cosI - np.sqrt(k1)) * nz
    # Refraction glass -> air at the flat back face (thin lens)
    k2 = 1.0 - n ** 2 * (1.0 - tz ** 2)
    if np.any(k2 < 0):
        raise RuntimeError("Total internal reflection at the back face (check R, f, K, D).")
    ux, uz = n * tx, np.sqrt(k2)
    x = h + (z - sag) * ux / uz
    xMarg = float(x[-1])
    if abs(xMarg) < 1e-9:
        raise RuntimeError("z places the camera at the focus: the pupil has zero size.")
    rhoIn = h / h[-1]
    rhoCam = x / xMarg
    if np.any(np.diff(rhoCam) <= 0):
        raise RuntimeError("With these parameters the camera plane z is inside the "
                           "caustic (rays cross before reaching it). Check f, K and z.")
    m0 = float(rhoCam[1] / rhoIn[1])          # magnification at the center
    return {"R": R, "f": f, "D": D, "K": K, "z": z, "n": n,
            "rhoIn": rhoIn, "rhoCam": rhoCam, "m0": m0, "xMarg": xMarg,
            "maxDev": float(np.max(np.abs(rhoCam - m0 * rhoIn)))}


def camToEntrance(mapping, rhoCam):
    # Normalized entrance radius (at the surface) of the ray that reaches rhoCam
    return np.interp(rhoCam, mapping["rhoCam"], mapping["rhoIn"])


def buildTheoreticalRonchigram(shape, pupil, xc, spacing, width, mapping):
    # Ideal Ronchigram for the surface parameters: straight, equally spaced grating
    # lines at the surface, mapped to the camera plane with the exact ray mapping.
    # Spacing and width are taken from the experimental Ronchigram (at the center),
    # and one line passes through the central experimental stripe at the pupil center.
    h, w = shape
    cx, cy, r = pupil
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dx, dy = (xx - cx) / r, (yy - cy) / r
    rc = np.hypot(dx, dy)
    inside = rc <= 1.0
    m0 = mapping["m0"]
    scale = np.full_like(rc, 1.0 / m0)
    ok = inside & (rc > 1e-9)
    scale[ok] = camToEntrance(mapping, rc[ok]) / rc[ok]
    u = dx * scale                                   # entrance coordinate of each pixel
    p = spacing / r / m0                             # grating period at the entrance
    wd = width / r / m0                              # stripe width at the entrance
    d0 = (xc - cx) / r
    u0 = d0 * (camToEntrance(mapping, abs(d0)) / abs(d0) if abs(d0) > 1e-9 else 1.0 / m0)
    t = (u - u0) / p
    white = (np.abs(t - np.round(t)) * p <= wd / 2) & inside
    theo = np.where(white, 255, 0).astype(np.uint8)
    nLines = len(np.unique(np.round(t[white]))) if white.any() else 0
    return theo, nLines


def theoreticalLines(shape, pupil, xc, spacing, mapping, kNeed=0):
    # Analytic centers of the theoretical stripes (manuscript: x_k = x_0 + k * lambda_exp),
    # mapped to the camera plane with the ideal ray mapping of the surface. For every
    # row, the camera abscissa whose entrance coordinate equals u_k is found exactly
    # (Newton), and a smooth polynomial in y is fitted to those points.
    # Every line keeps its order k and a function "exact" that gives its abscissa at
    # any row. Orders up to +-kNeed are returned even if they fall outside the pupil
    # ("outside": True): the ideal ruling continues there (the ray mapping is continued
    # linearly beyond the edge) and those lines are the reference of the outer fringes
    # when the aberrations displace them towards the center.
    h, w = shape
    cx, cy, r = pupil
    m0 = mapping["m0"]
    p = spacing / r / m0
    d0 = (xc - cx) / r
    u0 = d0 * (camToEntrance(mapping, abs(d0)) / abs(d0) if abs(d0) > 1e-9 else 1.0 / m0)

    edgeSlope = float((mapping["rhoIn"][-1] - mapping["rhoIn"][-2]) /
                      (mapping["rhoCam"][-1] - mapping["rhoCam"][-2]))

    def entrance(dx, dy):
        rc = np.hypot(dx, dy)
        rin = np.where(rc <= 1.0, camToEntrance(mapping, np.minimum(rc, 1.0)),
                       1.0 + (rc - 1.0) * edgeSlope)
        sc = np.where(rc > 1e-12, rin / np.maximum(rc, 1e-12), 1.0 / m0)
        return dx * sc

    def solve(uk, dy):
        dx = np.full_like(dy, uk * m0)
        for _ in range(30):
            f = entrance(dx, dy) - uk
            der = (entrance(dx + 1e-6, dy) - entrance(dx - 1e-6, dy)) / 2e-6
            dx = dx - f / np.where(np.abs(der) > 1e-9, der, 1.0)
        return dx

    ys = np.arange(np.ceil(cy - r), np.floor(cy + r) + 1, 1.0)
    dy = (ys - cy) / r
    lines = []
    if not np.isfinite(p) or p <= 0:
        raise RuntimeError("The fringe spacing is zero: the theoretical Ronchigram cannot "
                           "be built. Review the fringes.")
    kMax = max(int(np.ceil(1.0 / p)) + 1, int(kNeed))
    for k in range(-kMax, kMax + 1):
        uk = u0 + k * p
        dx = solve(uk, dy)
        ok = dx ** 2 + dy ** 2 <= 1.0
        exact = (lambda yq, uk=uk: cx + r * solve(uk, (np.asarray(yq, dtype=np.float64) - cy) / r))
        if abs(uk) >= 1.0 or ok.sum() < 3:
            if abs(k) > kNeed:
                continue
            xk = cx + dx * r
            poly = np.polynomial.Polynomial.fit(ys, xk, min(8, len(ys) - 1))
            lines.append({"poly": poly, "anchors": [], "ys": np.array([]), "xs": np.array([]),
                          "a0": float(ys.min()), "a1": float(ys.max()), "width": np.nan,
                          "xMean": float(xk.mean()), "k": k, "exact": exact, "outside": True})
            continue
        yk, xk = ys[ok], cx + dx[ok] * r
        poly = np.polynomial.Polynomial.fit(yk, xk, min(8, len(yk) - 1))
        a0, a1 = float(yk.min()), float(yk.max())
        ysS = np.arange(np.ceil(a0 / POINT_STEP) * POINT_STEP, a1 + 1e-9, POINT_STEP)
        xsS = poly(ysS)
        okS = insidePupil(xsS, ysS, pupil)
        lines.append({"poly": poly, "anchors": [], "ys": ysS[okS], "xs": xsS[okS],
                      "a0": a0, "a1": a1, "width": np.nan, "xMean": float(xk.mean()),
                      "k": k, "exact": exact, "outside": False})
    lines.sort(key=lambda L: L["xMean"])
    return lines


def centralIndex(lines, cx, cy):
    # Index of the line passing closest to the pupil center
    return int(np.argmin([abs(L["poly"](cy) - cx) for L in lines]))


def fringesBetween(binImg, LA, LB):
    # Number of fringes of the binary image lying between two detected fringes (0 for
    # consecutive fringes): white runs found, row by row, in the middle of the gap
    # between both centroid lines. The median over the rows is taken.
    a0, a1 = max(LA["a0"], LB["a0"]), min(LA["a1"], LB["a1"])
    if a1 - a0 < 4:
        return 0
    h, w = binImg.shape
    counts = []
    for y in np.arange(np.ceil(a0), a1, 2.0):
        r = int(round(y))
        if not 0 <= r < h:
            continue
        xa, xb = sorted((float(LA["poly"](y)), float(LB["poly"](y))))
        if xb - xa < 6:
            continue
        i0, i1 = max(0, int(np.floor(xa))), min(w, int(np.ceil(xb)) + 1)
        row = binImg[r, i0:i1] > 0
        d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        mid = i0 + 0.5 * (np.where(d == 1)[0] + np.where(d == -1)[0] - 1)
        length = np.where(d == -1)[0] - np.where(d == 1)[0]
        t = (mid - xa) / (xb - xa)
        counts.append(int(np.sum((t > 0.2) & (t < 0.8) & (length >= 2))))
    return int(round(float(np.median(counts)))) if counts else 0


def fringeOrders(expLines, binImg, ie0, cy, spacing=None):
    # Order of every fringe counted from the central one (order 0), as in the classical
    # theory of the Ronchi test (Optical Shop Testing, Eq. 9.3: each fringe is one line
    # of the ruling, order m). Neighbouring fringes differ by one order whatever their
    # separation, so fringes compressed or spread by the aberrations keep their order.
    # Orders are skipped only where a fringe is missing from the list (not detected, or
    # deleted in the review): the binary image shows a fringe between two of them AND
    # their separation is about twice (or more) that of the neighbouring fringes.
    pos = [float(L["poly"](min(max(cy, L["a0"]), L["a1"]))) for L in expLines]
    seq = list(np.argsort(pos))
    gaps = [pos[seq[j + 1]] - pos[seq[j]] for j in range(len(seq) - 1)]
    step = []
    for j, g in enumerate(gaps):
        n = fringesBetween(binImg, expLines[seq[j]], expLines[seq[j + 1]])
        if n > 0:
            ref = [gaps[i] for i in (j - 1, j + 1) if 0 <= i < len(gaps)]
            if spacing:
                ref.append(float(spacing))
            ref = min(ref) if ref else g
            n = min(n, max(int(round(g / ref)) - 1, 0)) if ref > 0 else 0
        step.append(1 + n)
    c = seq.index(ie0)
    order = {ie0: 0}
    for j in range(c + 1, len(seq)):
        order[seq[j]] = order[seq[j - 1]] + step[j - 1]
    for j in range(c - 1, -1, -1):
        order[seq[j]] = order[seq[j + 1]] - step[j]
    return order


def lineDifferences(expLines, theoLines, pairs, pupil):
    # Point-to-point difference between corresponding experimental and theoretical
    # centroid lines, at the same rows (both positions are kept; TA_POSITION selects
    # which one locates the value in the pupil).
    rows = []
    for k, (ie, it) in enumerate(pairs):
        E, T = expLines[ie], theoLines[it]
        if TA_POSITION == "experimental":
            # Every point of the measured fringe is compared with the ideal position
            # of the same order at the same row (also where the ideal line leaves the pupil)
            ys = np.arange(np.ceil(E["a0"] / POINT_STEP) * POINT_STEP, E["a1"] + 1e-9,
                           POINT_STEP, dtype=np.float64)
            xE = E["poly"](ys)
            xT = T["exact"](ys)
            ok = insidePupil(xE, ys, pupil)
        else:
            a0 = max(E["a0"], T["a0"])
            a1 = min(E["a1"], T["a1"])
            ys = np.arange(np.ceil(a0 / POINT_STEP) * POINT_STEP, a1 + 1e-9,
                           POINT_STEP, dtype=np.float64)
            xE = E["poly"](ys)
            xT = T["poly"](ys)
            ok = insidePupil(xT, ys, pupil) & ~np.asarray(T.get("outside", False))
        for y, xe, xt in zip(ys[ok], xE[ok], xT[ok]):
            rows.append({"line": k, "expIdx": ie, "theoIdx": it,
                         "along": float(y), "acrossTheo": float(xt),
                         "acrossExp": float(xe), "diffRaw": float(xe - xt)})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No corresponding experimental/theoretical points were found.")
    return df


def adjustMin(df, col="diffRaw"):
    # Shift the differences so the minimum is zero (Zernike polynomials are normalized)
    df = df.copy()
    df["diffAdj"] = (df[col] - df[col].min()).astype(np.float64)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def plotDiffMap(df, title, pupil, shape, value="diffAdj", outline=None,
                cbarLabel="TA, adjusted to min = 0 (px)"):
    # Point cloud: positions = measured points, color (intensity) = adjusted difference
    # outline: optional (N, 2) polygon drawn in cyan (border of the measured region)
    fig, ax = plt.subplots(figsize=(8, 8))
    if len(df) > 3000:
        # Dense clouds: drawn as a continuous surface over the points (display only;
        # the data are not modified). Individual markers would leave gaps/bands.
        sc = ax.tripcolor(df["x"].values, df["y"].values, df[value].values,
                          cmap="hot", shading="gouraud")
    else:
        s = float(np.clip(40000 / max(len(df), 1), 1, 60))
        sc = ax.scatter(df["x"], df["y"], c=df[value], cmap="hot", s=s, marker="s")
    plt.colorbar(sc, ax=ax, label=cbarLabel, fraction=0.046)
    cx, cy, r = pupil
    ax.add_patch(plt.Circle((cx, cy), r, fill=False, color="gray", ls="--"))
    if outline is not None:
        poly = np.vstack([outline, outline[:1]])
        ax.plot(poly[:, 0], poly[:, 1], color="#0B5394", lw=1.5, ls="--",
                label="Measured region")
        ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, shape[1])
    ax.set_ylim(shape[0], 0)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    plt.tight_layout()
    return figToArray(fig)


def style3D(ax):
    # White background and a faint grid in the 3D plots, so the grid does not
    # compete with the data
    ax.set_facecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor("#D0D0D0")
        axis._axinfo["grid"].update(color="#EBEBEB", linewidth=0.4, linestyle="-")
        axis.line.set_color("#808080")
    ax.tick_params(colors="#404040", labelsize=8)


# "hot" colormap without its white end, so the highest points stay visible on white
HOT_ON_WHITE = matplotlib.colors.LinearSegmentedColormap.from_list(
    "hotOnWhite", plt.get_cmap("hot")(np.linspace(0.0, 0.85, 256)))


def plot3D(df, title, value="diffAdj"):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    s = float(np.clip(20000 / max(len(df), 1), 1, 30))
    style3D(ax)
    sc = ax.scatter(df["x"], df["y"], df[value], c=df[value], cmap=HOT_ON_WHITE, s=s,
                    depthshade=False)
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label="TA, adjusted to min = 0 (px)")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_zlabel("TA (px)")
    ax.invert_yaxis()
    ax.set_title(title)
    plt.tight_layout()
    return figToArray(fig)


def plotProfile(profile, smooth, axisLabel):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(profile, alpha=0.4, label="Original Profile")
    ax.plot(smooth, label="Smoothed Profile", linewidth=2)
    ax.set_title("Intensity Profile")
    ax.set_xlabel(axisLabel)
    ax.set_ylabel("Average Intensity")
    ax.legend()
    plt.tight_layout()
    return figToArray(fig)


def plotPeaks(smooth, peaksR, peaks, removed, axisLabel):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(smooth, label="Smoothed profile", linewidth=2)
    ax.plot(peaksR, smooth[peaksR], "b+", markersize=14, markeredgewidth=2,
            label=f"Initial detection ({len(peaksR)})")
    ax.plot(peaks, smooth[peaks], "rx", markersize=14, markeredgewidth=2,
            label=f"Final stripes ({len(peaks)})")
    if removed:
        ax.plot(removed, smooth[removed], "ko", markersize=10, mfc="none",
                label=f"Discarded fragments ({len(removed)})")
    ax.set_title("Peaks Detection")
    ax.set_xlabel(axisLabel)
    ax.set_ylabel("Average Intensity")
    ax.legend()
    plt.tight_layout()
    return figToArray(fig)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — VERTICAL ALIGNMENT VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════
def stripeOrientation(rgb, pupil=None):
    # Global orientation of the fringes, for any inclination (Sec. 3.1: Canny + Hough,
    # median of the orientations).
    # The edges are taken from the BINARIZED fringes, not from the gray image: smooth
    # (sinusoidal) fringes have no sharp edges and Canny finds almost nothing on them,
    # while the binary fringes always have well-defined borders.
    # The orientation of a line is an axial quantity (theta and theta + 180 are the
    # same line), so the segments are averaged with the doubled angle and the median
    # is taken around that mean; this also works for nearly horizontal fringes, where
    # a plain median of the angles breaks at the wrap-around.
    # Returns (rotation in degrees that makes the fringes vertical, number of segments,
    # method used).
    # 1) Hough on the edges of the image itself (sharp fringes, as in real Ronchigrams)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    edges0 = cv2.Canny(cv2.GaussianBlur(bgr, (5, 5), 0), 30, 100, apertureSize=3)
    lines0 = cv2.HoughLinesP(edges0, 1, np.pi / 180, threshold=100,
                             minLineLength=100, maxLineGap=20)
    imageRot, nImage = None, 0
    if lines0 is not None:
        a0 = []
        for x1, y1, x2, y2 in lines0.reshape(-1, 4):
            a = np.degrees(np.arctan2(y1 - y2, x1 - x2))
            # Endpoint order is arbitrary: normalize to (-180, 0] (vertical = -90)
            a0.append((a % 180) - 180 if a % 180 != 0 else 0.0)
        imageRot, nImage = float(np.median(a0)) + 90.0, len(a0)

    # 2) Hough on the binarized fringes (smooth fringes have no sharp edges)
    _, _, _, closed = binarizeStripes(rgb)
    h, w = closed.shape
    if pupil is None:
        try:
            pupil = detectPupil(rgb)
        except RuntimeError:
            pupil = None
    inner = None
    if pupil is not None:
        closed = cv2.bitwise_and(closed, circleMask((h, w), pupil))
        # The arcs where the fringes are cut by the pupil border are not fringe edges
        inner = circleMask((h, w), (pupil[0], pupil[1], 0.96 * pupil[2]))
    edges = cv2.Canny(closed, 50, 150)
    if inner is not None:
        edges[inner == 0] = 0
    size = min(h, w) if pupil is None else 2.0 * pupil[2]
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(25, int(0.10 * size)),
                            minLineLength=max(25, int(0.15 * size)),
                            maxLineGap=max(5, int(0.03 * size)))
    ang, wts = [], []
    if lines is not None:
        # Depending on the OpenCV version the shape is (N, 1, 4) or (N, 4)
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            ang.append(np.degrees(np.arctan2(y1 - y2, x1 - x2)) % 180.0)
            wts.append(np.hypot(x1 - x2, y1 - y2))
    # Dominant direction of the intensity gradient inside the pupil (structure tensor):
    # the fringes are perpendicular to it. It uses the whole image, so it is also right
    # for strongly curved fringes, where the Hough segments are only the slanted parts.
    gray = cv2.GaussianBlur(stripeChannel(rgb).astype(np.float32), (0, 0), 2.0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    if inner is not None:
        gx, gy = gx[inner > 0], gy[inner > 0]
    jxx, jyy, jxy = float((gx * gx).sum()), float((gy * gy).sum()), float((gx * gy).sum())
    gradDir = (0.5 * np.degrees(np.arctan2(2.0 * jxy, jxx - jyy)) + 90.0
               if jxx + jyy > 0 else None)

    houghDir = None
    if len(ang) >= 3:
        ang, wts = np.array(ang), np.array(wts)
        d = np.radians(2.0 * ang)
        mean = 0.5 * np.degrees(np.arctan2((wts * np.sin(d)).sum(), (wts * np.cos(d)).sum()))
        dev = ((ang - mean + 90.0) % 180.0) - 90.0
        houghDir = mean + float(np.median(dev))

    def agrees(direction_):
        # A Hough estimate is trusted only if it agrees with the gradient orientation
        return gradDir is None or abs(((direction_ - gradDir + 90.0) % 180.0) - 90.0) <= 3.0

    if imageRot is not None and agrees(imageRot + 90.0):
        return imageRot, nImage, "Hough on the image edges"
    if houghDir is not None and agrees(houghDir):
        direction, method = houghDir, "Hough on the binarized fringes (smooth fringes)"
    elif gradDir is not None:
        direction = gradDir
        method = ("gradient orientation (curved fringes: the Hough segments disagree)"
                  if (houghDir is not None or imageRot is not None)
                  else "gradient orientation (no Hough segments)")
    else:
        return 0.0, 0, "no fringes found (image not rotated)"
    # direction: axial angle of the fringes (vertical = 90). Smallest rotation that
    # makes them vertical, in (-90, 90]
    rotation = ((direction - 90.0 + 90.0) % 180.0) - 90.0
    return float(rotation), len(ang), method


def measureCentralInclination(rgb, pupil=None):
    # Residual inclination of the central stripes (Eq. 10 of the manuscript).
    # Each central stripe is represented by its anchor points along its whole length
    # (same segmentation and centroid recognition used in the analysis) and a straight
    # line x = a + b*y is fitted to them; its slope b = (x_bottom - x_top)/(y_bottom - y_top)
    # is the inclination. Using the whole stripe instead of only its two extremes avoids
    # the noise of the pupil border, where the stripes are cut and fragmented.
    _, _, _, binImg = binarizeStripes(rgb)
    if pupil is None:
        pupil = detectPupil(rgb)
    cx, cy, _ = pupil
    binImg = cv2.bitwise_and(binImg, circleMask(binImg.shape, pupil))
    _, smooth = columnProfile(binImg)
    try:
        _, peaks, gap, _, _ = detectStripePeaks(smooth)
        halfWidth = max(int(gap * 0.4), 10)
    except RuntimeError:
        peaks, gap, halfWidth = np.array([], int), None, max(int(0.15 * pupil[2]), 10)
    stripes, _, _ = detectFringes(binImg, pupil, peaks, halfWidth)
    if len(stripes) < 2:
        raise RuntimeError("Fewer than 2 fringes found to verify the alignment.")
    lines, _ = buildCentroidLines(binImg, peaks, gap, pupil, stripes=stripes,
                                  halfWidth=halfWidth)
    # The ALIGN_N_CENTRAL stripes closest to the pupil center (least aberration effect)
    central = sorted(lines, key=lambda L: abs(L["poly"](cy) - cx))[:ALIGN_N_CENTRAL]
    measures = []
    for L in sorted(central, key=lambda L: L["xMean"]):
        ys = np.array([p[0] for p in L["anchors"]])
        xs = np.array([p[1] for p in L["anchors"]])
        if len(ys) < 3:
            continue
        b, a = np.polyfit(ys, xs, 1)
        yTop, yBot = float(ys.min()), float(ys.max())
        xTop, xBot = a + b * yTop, a + b * yBot
        measures.append({"xTop": xTop, "xBot": xBot, "yTop": yTop, "yBot": yBot,
                         "dist": xBot - xTop, "angle": float(np.degrees(np.arctan(b)))})
    if not measures:
        return 0.0, measures
    return float(np.median([m["angle"] for m in measures])), measures


def verifyVerticalAlignment(rgb0, baseAngle=0.0, pupil=None):
    # Refines the orientation until the residual inclination of the central stripes
    # is below ANGLE_THRESHOLD. Every measurement rotates the ORIGINAL image by the
    # total angle (Hough + corrections), so the rotations are never accumulated
    # (each resampling would smooth the fringes a little more).
    history = []
    total = float(baseAngle)
    h0, w0 = rgb0.shape[:2]
    rgb = rotateImage(rgb0, total)
    for it in range(ALIGN_MAX_ITER + 1):
        pup = None
        if pupil is not None:
            M = cv2.getRotationMatrix2D((w0 / 2, h0 / 2), total, 1.0)
            pcx, pcy = (M @ np.array([pupil[0], pupil[1], 1.0])).tolist()
            pup = (pcx, pcy, pupil[2])
        try:
            angle, measures = measureCentralInclination(rgb, pup)
        except RuntimeError:
            # Too few stripes detected automatically (e.g. a dim Ronchigram):
            # the check is skipped and the Hough orientation is kept
            angle, measures = 0.0, []
        history.append((angle, measures))
        if abs(angle) <= ANGLE_THRESHOLD or it == ALIGN_MAX_ITER:
            break
        # Rotating by -angle cancels the inclination of the stripes
        total -= angle
        rgb = rotateImage(rgb0, total)
    totalCorrection = total - float(baseAngle)

    # Visualization: extremes of the central stripes and a vertical reference line
    vis = rgb.copy()
    finalAngle, finalMeasures = history[-1]
    for m in finalMeasures:
        pTop = (int(round(m["xTop"])), int(m["yTop"]))
        pBot = (int(round(m["xBot"])), int(m["yBot"]))
        cv2.line(vis, pTop, (pTop[0], pBot[1]), (255, 255, 255), 1)   # vertical reference
        cv2.line(vis, pTop, pBot, (0, 255, 0), 2)                       # measured extremes
        cv2.circle(vis, pTop, 6, (0, 255, 255), -1)
        cv2.circle(vis, pBot, 6, (0, 255, 255), -1)

    text = []
    for i, (a, ms) in enumerate(history):
        text.append(f"Iteration {i}: residual angle = {a:.4f}°")
    text.append(f"Threshold: {ANGLE_THRESHOLD}°")
    text.append(f"Total correction applied: {totalCorrection:.4f}°")
    text.append("Central stripes (straight line fitted to all their anchor points):")
    text += [f"  x_top={m['xTop']:.2f}  x_bottom={m['xBot']:.2f}  Δ={m['dist']:.3f}"
             for m in finalMeasures]
    return rgb, vis, text, totalCorrection


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — ANALYSIS OF ONE RONCHIGRAM (vertical or horizontal)
# ══════════════════════════════════════════════════════════════════════════════
def analyzeRonchigram(rgb, pupil, label, horizontal, results, prog, p0, p1, mapping,
                      editor=None, stripesOverride=None, maxDegree=None):
    # Same algorithm for both orientations. The horizontal Ronchigram is analyzed
    # on its transpose (stripes become vertical) and the results are transposed back,
    # so the output is expressed in the coordinates of the rotated image.
    # Output convention: vertical   -> lines x = poly(y), differences in x
    #                    horizontal -> lines y = poly(x), differences in y
    tag = f" - {label}"
    if horizontal:
        work = np.ascontiguousarray(rgb.transpose(1, 0, 2))
        pup = (pupil[1], pupil[0], pupil[2])
        axisLabel = "Row (px)"
    else:
        work = rgb
        pup = pupil
        axisLabel = "Column (px)"

    def out(im):
        if not horizontal:
            return im
        return np.ascontiguousarray(im.T if im.ndim == 2 else im.transpose(1, 0, 2))

    def step(f):
        prog(p0 + (p1 - p0) * f)

    h, w = work.shape[:2]
    cx, cy, rP = pup
    pMask = circleMask((h, w), pup)

    # ── Segmentation ──
    Rb, binary, clean, closed = binarizeStripes(work)
    closed = cv2.bitwise_and(closed, pMask)
    results.append({"title": "CLAHE" + tag, "image": out(Rb)})
    results.append({"title": "Binary Image" + tag, "image": out(binary)})
    results.append({"title": "Connected Components Filtering" + tag, "image": out(clean)})
    results.append({"title": "Closed Stripes inside Pupil" + tag, "image": out(closed)})
    step(0.15)

    # ── Stripe detection ──
    profile, smooth = columnProfile(closed)
    try:
        peaksR, peaks, gap, minProm, removed = detectStripePeaks(smooth)
        peakText = [f"Initial detection: {len(peaksR)} peaks",
                    f"Discarded fragments: {len(removed)}",
                    f"Final stripes: {len(peaks)}",
                    f"Median gap (spacing): {gap:.3f} px"]
    except RuntimeError as ex:
        # No valid peaks in the column profile (e.g. strongly curved fringes): the
        # fringes are found by fringe following
        peaksR, peaks, gap, removed = np.array([], int), np.array([], int), None, []
        peakText = [f"Column profile: {ex}"]
    results.append({"title": "Intensity Profile" + tag,
                    "image": plotProfile(profile, smooth, axisLabel)})
    results.append({"title": "Peaks Detection" + tag,
                    "image": plotPeaks(smooth, peaksR, peaks, removed, axisLabel),
                    "text": peakText})
    step(0.3)

    # ── Anchor points: automatic, then reviewed by the user ──
    halfWidth = max(int(gap * 0.4), 10) if gap else max(int(0.15 * rP), 10)

    def curvedBinary():
        # Curved fringes: the closing must not merge neighbouring fringes that the
        # aberrations bring very close together (near the pupil border), so it is
        # repeated with a kernel elongated along the fringes
        c = cv2.bitwise_and(binarizeStripes(work, CURVED_CLOSE_KERNEL)[3], pMask)
        for r_ in results:
            if r_["title"] == "Closed Stripes inside Pupil" + tag:
                r_["image"] = out(c)
                r_["text"] = [f"Curved fringes: closing of {CURVED_CLOSE_KERNEL[0]} x "
                              f"{CURVED_CLOSE_KERNEL[1]} px (along the fringes), so "
                              f"neighbouring fringes are not merged"]
        return c

    if stripesOverride is not None:
        stripes = stripesOverride
        anchorText = ["Anchor points: the fringes confirmed in the vertical Ronchigram, "
                      "rotated 90° (same fringes)"]
        if maxDegree == CURVED_MAX_DEGREE:
            closed = curvedBinary()
    else:
        stripes, detMethod, leave = detectFringes(
            closed, pup, peaks, halfWidth,
            gray=stripeChannel(work) if SUBPIXEL_ANCHORS else None)
        if detMethod.startswith("fringe following"):
            maxDegree = CURVED_MAX_DEGREE
            closedC = curvedBinary()
            stripesC = trackFringes(closedC, pup)
            closed = closedC
            if len(stripesC) >= 2:
                stripes = stripesC
        if gap is None and len(stripes) >= 2:
            # Spacing from the detected fringes when the column profile has no peaks
            cols_ = np.sort([np.mean([p[1] for p in s_["anchors"]]) for s_ in stripes])
            halfWidth = max(int(0.4 * float(np.median(np.diff(cols_)))), 10)
        anchorText = [f"Detection: {detMethod}",
                      f"Half-width of search window: {halfWidth} px",
                      f"Automatically detected stripes: {len(stripes)}"]
        if leave is not None:
            anchorText.insert(1, f"Rows where a fringe leaves its vertical window: "
                                 f"{100 * leave:.0f} % (limit {100 * WINDOW_LEAVE_MAX:.0f} %)")
    if editor is not None:
        nAuto = len(stripes)
        stripes = editor({"image": work, "stripes": stripes, "pupil": pup,
                          "halfWidth": halfWidth, "channel": stripeChannel(work)})
        if stripes is None:
            raise RuntimeError("Pipeline cancelled by the user in the fringe review.")
        anchorText = [t_ for t_ in anchorText if t_.startswith(("Detection", "Rows where"))] + \
                     [f"Automatically detected fringes: {nAuto}",
                      f"Fringes confirmed by the user: {len(stripes)}",
                      "Anchor points shown: the ones confirmed by the user"]

    # Anchor points used in the analysis (the confirmed ones when there was a review)
    vis = cv2.cvtColor(closed, cv2.COLOR_GRAY2RGB)
    for i, s_ in enumerate(stripes):
        col = FE_RGB[i % len(FE_RGB)] if editor is not None else COLORS[i % len(COLORS)]
        for (ar, ac) in s_["anchors"]:
            pt = (int(round(ac)), int(round(ar)))
            cv2.circle(vis, pt, 7, (255, 255, 255), -1)
            cv2.circle(vis, pt, 7, col, 3)
    results.append({"title": "Anchor Points" + tag, "image": out(vis), "text": anchorText})

    if editor is not None:
        vis = work.copy()
        for i, s_ in enumerate(stripes):
            col = FE_RGB[i % len(FE_RGB)]
            rr = np.array([p[0] for p in s_["anchors"]])
            cc = np.array([p[1] for p in s_["anchors"]])
            if len(rr) > POLY_DEGREE:
                drawCurve(vis, fitCenterline(rr, cc, maxDegree), rr.min(), rr.max(),
                          col, 2)
            for (ar, ac) in s_["anchors"]:
                cv2.circle(vis, (int(round(ac)), int(round(ar))), 6, (255, 255, 255), -1)
                cv2.circle(vis, (int(round(ac)), int(round(ar))), 6, col, 2)
        results.append({"title": "Fringe Review (confirmed by the user)" + tag,
                        "image": out(vis),
                        "text": [f"Automatically detected fringes: {nAuto}",
                                 f"Confirmed fringes: {len(stripes)}",
                                 "Anchor points per fringe: " +
                                 ", ".join(str(len(s_["anchors"])) for s_ in stripes)]})

    # ── Experimental centroid lines (from the confirmed anchor points) ──
    expLines, halfWidth = buildCentroidLines(closed, peaks, gap, pup, stripes=stripes,
                                             halfWidth=halfWidth, maxDegree=maxDegree)
    if len(expLines) < 2:
        raise RuntimeError(f"{label}: at least 2 fringes (with at least "
                           f"{POLY_DEGREE + 1} anchor points each) are needed.")
    # Spacing of the confirmed fringes (same definition as the automatic detection)
    gap = float(np.median(np.diff(fringePositions(smooth, stripes, halfWidth))))
    if maxDegree == CURVED_MAX_DEGREE or not np.isfinite(gap) or gap < 2.0:
        # Curved fringes: the column profile does not separate them (several fringes
        # share the same maximum, which would give a spacing of zero). The spacing is
        # the median separation of the centroid lines on the center row.
        pos_ = np.sort([float(L["poly"](min(max(cy, L["a0"]), L["a1"]))) for L in expLines])
        sep_ = np.diff(pos_)
        sep_ = sep_[sep_ > 0.5]
        if len(sep_) == 0:
            raise RuntimeError(f"{label}: the fringe spacing could not be measured "
                               "(the confirmed fringes coincide). Review the fringes.")
        gap = float(np.median(sep_))
    expWidth = float(np.nanmedian([L["width"] for L in expLines]))
    if not np.isfinite(expWidth):
        expWidth = 0.4 * gap

    vis = work.copy()
    for i, L in enumerate(expLines):
        col = COLORS[i % len(COLORS)]
        drawCurve(vis, L["poly"], L["a0"], L["a1"], col, 2)
        for x, y in zip(L["xs"], L["ys"]):
            cv2.circle(vis, (int(round(x)), int(round(y))), 2, (255, 255, 255), -1)
    nPts = [len(L["ys"]) for L in expLines]
    results.append({"title": "Experimental Centroid Lines" + tag, "image": out(vis),
                    "text": [f"Point spacing along each line: {POINT_STEP} px",
                             f"Max. points allowed (size/{POINT_STEP}): {h // POINT_STEP}",
                             f"Points per line: min {min(nPts)}, max {max(nPts)}",
                             f"Measured stripe width: {expWidth:.3f} px"]})
    step(0.5)

    # ── Theoretical Ronchigram: ideal surface (R, f, D, K, z), same spacing and width
    #    as the experimental one, anchored on the central stripe ──
    ie0 = centralIndex(expLines, cx, cy)
    xc = float(expLines[ie0]["poly"](cy))
    theoBin, nTheo = buildTheoreticalRonchigram((h, w), pup, xc, gap, expWidth, mapping)
    results.append({"title": "Theoretical Ronchigram" + tag, "image": out(theoBin),
                    "text": [f"Ideal surface: R = {mapping['R']:g} mm, f = {mapping['f']:g} mm, "
                             f"D = {mapping['D']:g} mm, K = {mapping['K']:g}, "
                             f"z = {mapping['z']:g} mm",
                             f"Spacing at the center: {gap:.3f} px",
                             f"Stripe width (= experimental): {expWidth:.3f} px",
                             f"Reference (central) stripe at: {xc:.3f} px",
                             f"Theoretical stripes: {nTheo}"]})

    # Theoretical centroid lines: analytic centers of the ideal stripes
    # (the order of every experimental fringe is counted first, so the theoretical
    #  lines of all those orders are available)
    orders = fringeOrders(expLines, closed, ie0, cy, gap)
    theoLines = theoreticalLines((h, w), pup, xc, gap, mapping,
                                 kNeed=max(abs(o) for o in orders.values()))
    byOrder = {L["k"]: i for i, L in enumerate(theoLines)}

    vis = cv2.cvtColor(theoBin, cv2.COLOR_GRAY2RGB)
    for L in theoLines:
        if L["outside"]:
            continue
        drawCurve(vis, L["poly"], L["a0"], L["a1"], (255, 0, 0), 2)
        for x, y in zip(L["xs"], L["ys"]):
            cv2.circle(vis, (int(round(x)), int(round(y))), 2, (0, 120, 255), -1)
    nInside = sum(1 for L in theoLines if not L["outside"])
    theoText = [f"Recognized theoretical lines: {nInside}"]
    if nInside < len(theoLines):
        theoText.append(f"Lines of the ideal ruling beyond the pupil border, used as "
                        f"reference of the outer fringes: {len(theoLines) - nInside}")
    results.append({"title": "Theoretical Centroid Lines" + tag, "image": out(vis),
                    "text": theoText})
    step(0.7)

    # ── Correspondence by order, counted from the central stripe ──
    # Fringe m of the Ronchigram corresponds to line m of the theoretical one
    # (consecutive orders; orders are skipped only where the binary image shows a
    # fringe that is not in the list, so a deleted fringe does not shift the rest)
    pairs = []
    for ie, L in enumerate(expLines):
        it = byOrder.get(orders[ie])
        if it is None or it in [q[1] for q in pairs]:
            continue
        if TA_POSITION != "experimental" and theoLines[it]["outside"]:
            continue
        pairs.append((ie, it))

    dfT = lineDifferences(expLines, theoLines, pairs, pup)
    dfT = adjustMin(dfT)

    vis = np.full_like(work, 255)
    for k, (ie, it) in enumerate(pairs):
        E, T = expLines[ie], theoLines[it]
        drawCurve(vis, E["poly"], E["a0"], E["a1"], COLORS_ON_WHITE[k % len(COLORS_ON_WHITE)], 2)
        drawCurve(vis, T["poly"], T["a0"], T["a1"], (255, 0, 0), 2)
    for _, r in dfT.iloc[::2].iterrows():
        y = int(round(r["along"]))
        cv2.line(vis, (int(round(r["acrossExp"])), y), (int(round(r["acrossTheo"])), y),
                 (150, 150, 150), 1)
    cv2.circle(vis, (int(round(cx)), int(round(cy))), int(round(rP)), (90, 90, 90), 1)
    results.append({"title": "Experimental vs Theoretical Correspondence" + tag,
                    "image": out(vis),
                    "text": [f"Corresponding pairs: {len(pairs)}",
                             "Fringe orders counted from the central fringe (order 0)"] +
                            [f"  Exp. line {ie + 1} -> order {theoLines[it]['k']:+d}"
                             + (" (ideal line beyond the pupil)" if theoLines[it]["outside"]
                                else "") for ie, it in pairs]})

    # ── Output in the frame of the analyzed image ──
    across = dfT["acrossExp"] if TA_POSITION == "experimental" else dfT["acrossTheo"]
    if horizontal:
        df = pd.DataFrame({"line": dfT["line"], "x": dfT["along"], "y": across,
                           "yExp": dfT["acrossExp"], "yTheo": dfT["acrossTheo"],
                           "diffRaw": dfT["diffRaw"], "diffAdj": dfT["diffAdj"]})
    else:
        df = pd.DataFrame({"line": dfT["line"], "x": across, "y": dfT["along"],
                           "xExp": dfT["acrossExp"], "xTheo": dfT["acrossTheo"],
                           "diffRaw": dfT["diffRaw"], "diffAdj": dfT["diffAdj"]})

    summary = df.groupby("line")["diffRaw"].agg(["mean", "max", "min", "std", "count"])
    summary.columns = ["Mean (px)", "Max (px)", "Min (px)", "Std (px)", "Points"]
    results.append({"title": "Difference Summary" + tag, "table": summary.round(4),
                    "excel": df,
                    "text": [f"Compared points: {len(df)}",
                             f"Difference = experimental - theoretical",
                             f"Raw min: {df['diffRaw'].min():.6f} px",
                             f"Raw max: {df['diffRaw'].max():.6f} px",
                             "Adjusted so that min = 0"]})
    results.append({"title": "Difference Matrix" + tag,
                    "image": plotDiffMap(df, f"Difference Matrix{tag}", pupil, rgb.shape[:2]),
                    "text": [f"Points: {len(df)}",
                             f"Adjusted min: {df['diffAdj'].min():.6f} px",
                             f"Adjusted max: {df['diffAdj'].max():.6f} px"]})
    step(1.0)

    return {"exp": expLines, "theo": theoLines, "pairs": pairs, "df": df,
            "spacing": gap, "width": expWidth, "maxDegree": maxDegree,
            "stripes": [{"peak": int(round(L["xMean"])), "anchors": L["anchors"]}
                        for L in expLines]}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — GRID
# ══════════════════════════════════════════════════════════════════════════════
def intersectLines(LV, LH):
    # Vertical line: x = pV(y), y in [LV.a0, LV.a1]
    # Horizontal line: y = pH(x), x in [LH.a0, LH.a1]
    # Intersection: g(y) = pH(pV(y)) - y = 0
    pV, pH = LV["poly"], LH["poly"]
    ys = np.linspace(LV["a0"], LV["a1"], 600)
    xs = pV(ys)
    g = pH(xs) - ys
    valid = (xs >= LH["a0"]) & (xs <= LH["a1"])
    for k in range(len(ys) - 1):
        if not (valid[k] and valid[k + 1]):
            continue
        if g[k] == 0:
            return float(pV(ys[k])), float(ys[k])
        if g[k] * g[k + 1] < 0:
            y = brentq(lambda t: pH(pV(t)) - t, ys[k], ys[k + 1], xtol=1e-12)
            return float(pV(y)), float(y)
    return None


def buildGrid(V, H, pupil, shape):
    # Experimental grid: experimental vertical + horizontal centroid lines
    # Theoretical grid:  theoretical vertical + horizontal centroid lines
    # Intersections are paired keeping the line correspondence of each Ronchigram
    rows = []
    for kv, (ieV, itV) in enumerate(V["pairs"]):
        for kh, (ieH, itH) in enumerate(H["pairs"]):
            pe = intersectLines(V["exp"][ieV], H["exp"][ieH])
            pt = intersectLines(V["theo"][itV], H["theo"][itH])
            if pe is None or pt is None:
                continue
            pos = pe if TA_POSITION == "experimental" else pt
            if not insidePupil(pos[0], pos[1], pupil):
                continue
            dx, dy = pe[0] - pt[0], pe[1] - pt[1]
            rows.append({"lineV": kv, "lineH": kh, "x": pos[0], "y": pos[1],
                         "xTheo": pt[0], "yTheo": pt[1],
                         "xExp": pe[0], "yExp": pe[1], "dx": dx, "dy": dy,
                         "diffRaw": float(np.hypot(dx, dy))})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No intersections found between vertical and horizontal lines.")
    df = adjustMin(df)

    # Drawn on a white background (experimental: green / blue; theoretical: red)
    h, w = shape
    gridExp = np.full((h, w, 3), 255, dtype=np.uint8)
    gridTheo = np.full((h, w, 3), 255, dtype=np.uint8)
    overlay = np.full((h, w, 3), 255, dtype=np.uint8)
    for im, which, colV, colH in ((gridExp, "exp", (0, 150, 0), (0, 110, 220)),
                                  (gridTheo, "theo", (220, 0, 0), (220, 0, 0)),
                                  (overlay, "theo", (240, 150, 150), (240, 150, 150)),
                                  (overlay, "exp", (0, 150, 0), (0, 110, 220))):
        k = 0 if which == "exp" else 1
        for pair in V["pairs"]:
            L = V[which][pair[k]]
            drawCurve(im, L["poly"], L["a0"], L["a1"], colV, 2, vertical=True)
        for pair in H["pairs"]:
            L = H[which][pair[k]]
            drawCurve(im, L["poly"], L["a0"], L["a1"], colH, 2, vertical=False)
    for _, r in df.iterrows():
        pe = (int(round(r["xExp"])), int(round(r["yExp"])))
        pt = (int(round(r["xTheo"])), int(round(r["yTheo"])))
        cv2.line(overlay, pe, pt, (230, 140, 0), 2)
        cv2.circle(overlay, pt, 5, (0, 0, 0), -1)
        cv2.circle(overlay, pe, 5, (150, 0, 200), -1)
    cx, cy, r = pupil
    for im in (gridExp, gridTheo, overlay):
        cv2.circle(im, (int(round(cx)), int(round(cy))), int(round(r)), (120, 120, 120), 1)
    return df, gridExp, gridTheo, overlay


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — REFINEMENT OF THE POINT CLOUDS (multilinear interpolation)
# ══════════════════════════════════════════════════════════════════════════════
def refineCoords(coords, k, minSpacing=MIN_SPACING):
    # Inserts k equally spaced coordinates between each pair of neighbours.
    # k is reduced if the new spacing would be below minSpacing (Nyquist).
    coords = np.unique(np.round(np.asarray(coords, dtype=np.float64), 9))
    if len(coords) < 2 or k <= 0:
        return coords, 0
    minGap = float(np.min(np.diff(coords)))
    kEff = int(max(0, min(k, np.floor(minGap / minSpacing) - 1)))
    new = [coords[0]]
    for a, b in zip(coords[:-1], coords[1:]):
        new.extend(np.linspace(a, b, kEff + 2)[1:])
    return np.array(new, dtype=np.float64), kEff


def refineMatrix(df, xCoords, yCoords, refineX, refineY, k, pupil, vector=False):
    # Builds a finer lattice (k new points between neighbouring lines in the
    # refined directions) and interpolates the adjusted differences on it.
    # Points outside the data region or the pupil are discarded.
    #
    # vector=True (grid): the signed components dx and dy are interpolated
    # separately with GRID_INTERP_METHOD and the difference is recomputed as
    # |(dx, dy)|. Interpolating the magnitude directly produces a cone-shaped
    # artifact where the displacement is close to zero, and facets due to the
    # few intersection points.
    xs, kx = refineCoords(xCoords, k) if refineX else (np.unique(xCoords), 0)
    ys, ky = refineCoords(yCoords, k) if refineY else (np.unique(yCoords), 0)
    X, Y = np.meshgrid(xs, ys)
    pts = (df["x"].values, df["y"].values)
    if vector:
        DX = griddata(pts, df["dx"].values, (X, Y), method=GRID_INTERP_METHOD)
        DY = griddata(pts, df["dy"].values, (X, Y), method=GRID_INTERP_METHOD)
        Z = np.hypot(DX, DY)
    else:
        Z = griddata(pts, df["diffAdj"].values, (X, Y), method=INTERP_METHOD)
        # The signed TA (experimental - theoretical) is interpolated too: it is the
        # value used to integrate the wavefront (the min = 0 map is only for display)
        signedCol = "diffRaw" if "diffRaw" in df else "signed"
        S = griddata(pts, df[signedCol].values, (X, Y), method=INTERP_METHOD)
    ok = ~np.isnan(Z) & insidePupil(X, Y, pupil)
    out = pd.DataFrame({"x": X[ok], "y": Y[ok], "value": Z[ok].astype(np.float64)})
    if vector:
        out["dx"] = DX[ok].astype(np.float64)
        out["dy"] = DY[ok].astype(np.float64)
    else:
        out["signed"] = S[ok].astype(np.float64)
    out = adjustMin(out, "value").drop(columns="value")
    return out, kx, ky


def extendAxis(coords, lo, hi):
    # Extends a set of equally spaced coordinates with the same step until
    # it covers [lo, hi]. The original coordinates are kept unchanged.
    coords = np.unique(np.round(np.asarray(coords, dtype=np.float64), 9))
    step = float(np.median(np.diff(coords)))
    left = np.arange(coords[0] - step, lo - step, -step)[::-1]
    right = np.arange(coords[-1] + step, hi + step, step)
    return np.concatenate([left, coords, right])


def polyBasis(u, v, degree):
    # Monomials u^l v^h with l + h <= degree (u, v normalized to the unit pupil)
    return np.stack([u ** l * v ** h for l in range(degree + 1)
                     for h in range(degree + 1 - l)], axis=1)


def extendToPupil(df, pupil, vector=False):
    # Covers the whole exit pupil:
    #  - lattice points that already have data keep their value
    #  - the rest are estimated with a polynomial of degree FULL_PUPIL_DEGREE
    #    fitted (least squares) to all the data, plus the fit residual of the
    #    nearest data point decaying as exp(-d / FULL_PUPIL_DECAY), so the
    #    extension joins the measured region continuously
    # vector=True (grid): dx and dy are extended separately, then |(dx, dy)|
    cx, cy, r = pupil
    xs = extendAxis(df["x"].values, cx - r, cx + r)
    ys = extendAxis(df["y"].values, cy - r, cy + r)
    X, Y = np.meshgrid(xs, ys)
    inside = insidePupil(X, Y, pupil)
    Xq, Yq = X[inside], Y[inside]

    tree = cKDTree(np.column_stack([df["x"].values, df["y"].values]))
    dist, idx = tree.query(np.column_stack([Xq, Yq]))
    isData = dist < 1e-6

    u, v = (df["x"].values - cx) / r, (df["y"].values - cy) / r
    B = polyBasis(u, v, FULL_PUPIL_DEGREE)
    Bq = polyBasis((Xq - cx) / r, (Yq - cy) / r, FULL_PUPIL_DEGREE)
    weight = np.exp(-dist / FULL_PUPIL_DECAY)

    def extend(values):
        coef = np.linalg.lstsq(B, values, rcond=None)[0]
        residual = values - B @ coef
        est = Bq @ coef + residual[idx] * weight
        est[isData] = values[idx[isData]]
        return est, float(np.sqrt(np.mean(residual ** 2)))

    out = pd.DataFrame({"x": Xq, "y": Yq})
    if vector:
        dx, rmsX = extend(df["dx"].values)
        dy, rmsY = extend(df["dy"].values)
        out["dx"], out["dy"] = dx, dy
        out["value"] = np.hypot(dx, dy)
        rms = float(np.hypot(rmsX, rmsY))
    else:
        out["value"], rms = extend(df["diffAdj"].values)
        out["signed"], _ = extend(df["signed"].values)
    out["source"] = np.where(isData, "data", "extended")
    out = adjustMin(out, "value").drop(columns="value")

    # Border of the measured region, for the plots
    pts = np.column_stack([df["x"].values, df["y"].values])
    outline = pts[ConvexHull(pts).vertices]
    return out, rms, outline


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — WAVEFRONT (Eqs. 9-15 of the article)
# ══════════════════════════════════════════════════════════════════════════════
ZERNIKE_NAMES = {
    (0, 0): "Piston", (1, -1): "Tilt Y", (1, 1): "Tilt X",
    (2, -2): "Astigmatism 45°", (2, 0): "Defocus", (2, 2): "Astigmatism 0°",
    (3, -3): "Trefoil Y", (3, -1): "Coma Y", (3, 1): "Coma X", (3, 3): "Trefoil X",
    (4, -4): "Tetrafoil Y", (4, -2): "Sec. astigmatism 45°", (4, 0): "Spherical",
    (4, 2): "Sec. astigmatism 0°", (4, 4): "Tetrafoil X",
    (5, -1): "Sec. coma Y", (5, 1): "Sec. coma X", (6, 0): "Sec. spherical",
}


def fitTASurface(u, v, values, degree=TA_FIT_DEGREE):
    # Eq. 12: least-squares fit of the TA point cloud with the polynomial
    # sum_{l,h<=degree} A_lh x^l y^h (= B_lh cos^l sin^h rho^(l+h)).
    # A Legendre basis spans exactly the same polynomials and is better conditioned.
    V = npleg.legvander2d(u, v, [degree, degree])
    coef = np.linalg.lstsq(V, values, rcond=None)[0]
    rms = float(np.sqrt(np.mean((V @ coef - values) ** 2)))
    return coef.reshape(degree + 1, degree + 1), rms


def zernikeIndices(Q):
    # ANSI order: n = 0..Q, m = -n, -n+2, ..., n  ->  (Q+1)(Q+2)/2 terms
    return [(n, m) for n in range(Q + 1) for m in range(-n, n + 1, 2)]


def zernike(n, m, rho, theta):
    # Normalized Zernike polynomial (orthonormal on the unit disk).
    # Radial part through Jacobi polynomials, numerically stable for high n.
    k = (n - abs(m)) // 2
    R = (-1) ** k * rho ** abs(m) * eval_jacobi(k, abs(m), 0, 1 - 2 * rho ** 2)
    if m == 0:
        return np.sqrt(n + 1) * R
    ang = np.cos(m * theta) if m > 0 else np.sin(-m * theta)
    return np.sqrt(2 * (n + 1)) * R * ang


def zernikeTerms(idx, rho, theta):
    # Yields (j, Z_j) for every term, in OSA/ANSI order. The radial part of each
    # (n, |m|) is computed once and shared by +m and -m.
    radial = {}
    for j, (n, m) in enumerate(idx):
        am = abs(m)
        if (n, am) not in radial:
            if radial and next(iter(radial))[0] != n:
                radial = {}
            k = (n - am) // 2
            radial[(n, am)] = (-1) ** k * rho ** am * eval_jacobi(k, am, 0, 1 - 2 * rho ** 2)
        R = radial[(n, am)]
        if m == 0:
            yield j, np.sqrt(n + 1) * R
        else:
            ang = np.cos(m * theta) if m > 0 else np.sin(-m * theta)
            yield j, np.sqrt(2 * (n + 1)) * R * ang


def zernikeEval(coefs, idx, rho, theta, skip=(), progress=None):
    W = np.zeros_like(rho)
    for j, Z in zernikeTerms(idx, rho, theta):
        if j not in skip and coefs[j] != 0:
            W += coefs[j] * Z
        if progress and j % 50 == 0:
            progress(j / len(idx))
    return W


def zernikeFit(idx, rho, theta, values, chunk=4000, progress=None):
    # Least squares fit (Eq. 49 of the manuscript) through the normal equations,
    # accumulated in chunks so many points of W_r can be used without much memory
    J = len(idx)
    AtA = np.zeros((J, J))
    Atb = np.zeros(J)
    for s0 in range(0, len(rho), chunk):
        if progress:
            progress(s0 / len(rho))
        r_, t_, v_ = rho[s0:s0 + chunk], theta[s0:s0 + chunk], values[s0:s0 + chunk]
        A = np.empty((len(r_), J))
        for j, Z in zernikeTerms(idx, r_, t_):
            A[:, j] = Z
        AtA += A.T @ A
        Atb += A.T @ v_
    coefs = np.linalg.lstsq(AtA, Atb, rcond=None)[0]
    return coefs


def computeWavefront(taX, taY, pupil, nZernike=ZERNIKE_Q, mapping=None, mmPerPx=None):
    # taX, taY: point clouds (x, y, value) of the transverse aberrations in x and y
    cx, cy, r = pupil

    # ── Eq. 12: TA surfaces ──
    cX, rmsX = fitTASurface((taX[0] - cx) / r, (taX[1] - cy) / r, taX[2])
    cY, rmsY = fitTASurface((taY[0] - cx) / r, (taY[1] - cy) / r, taY[2])

    # ── Eqs. 9, 13, 14: directional derivative along r and radial Riemann sum ──
    rho = np.linspace(0.0, 1.0, RADIAL_STEPS + 1)
    theta = np.linspace(0.0, 2 * np.pi, ANGULAR_STEPS, endpoint=False)
    Rg, Tg = np.meshgrid(rho, theta, indexing="ij")
    U, V = Rg * np.cos(Tg), Rg * np.sin(Tg)
    tx = npleg.legval2d(U, V, cX)
    ty = npleg.legval2d(U, V, cY)
    taR = tx * np.cos(Tg) + ty * np.sin(Tg)            # TA along r (Eq. 9)
    # Rayces: dW/dh = -TA/z, with TA in mm (px * mm/px) and h the height at the surface
    # (h = rho_in * D/2, rho_in from the ideal ray mapping). W in micrometers.
    rhoIn = camToEntrance(mapping, rho)
    dRhoIn = np.gradient(rhoIn, rho)[:, None]
    kW = 1000.0 * mmPerPx * (mapping["D"] / 2.0) / mapping["z"]
    Wpolar = WAVEFRONT_SIGN * kW * cumulative_trapezoid(taR * dRhoIn, rho, axis=0, initial=0)

    # Polar -> Cartesian samples on the unit disk
    thetaExt = np.append(theta, 2 * np.pi)
    interp = RegularGridInterpolator((rho, thetaExt),
                                     np.concatenate([Wpolar, Wpolar[:, :1]], axis=1))
    g = np.linspace(-1, 1, ZERNIKE_GRID)
    GX, GY = np.meshgrid(g, g)
    inside = GX ** 2 + GY ** 2 <= 1.0
    rq = np.sqrt(GX[inside] ** 2 + GY[inside] ** 2)
    tq = np.mod(np.arctan2(GY[inside], GX[inside]), 2 * np.pi)
    Wq = interp(np.column_stack([rq, tq]))

    # ── Eq. 15 / Eq. 49: Zernike fit of degree n (OSA/ANSI order) on W_r ──
    idx = zernikeIndices(nZernike)
    coefs = zernikeFit(idx, rq, tq, Wq)
    fitRms = float(np.sqrt(np.mean((zernikeEval(coefs, idx, rq, tq) - Wq) ** 2)))

    # Maps shown in the GUI, on a smaller regular grid (rows = y downwards)
    g = np.linspace(-1, 1, MAP_GRID)
    GX, GY = np.meshgrid(g, g)
    inside = GX ** 2 + GY ** 2 <= 1.0
    rq = np.sqrt(GX[inside] ** 2 + GY[inside] ** 2)
    tq = np.mod(np.arctan2(GY[inside], GX[inside]), 2 * np.pi)
    Wmap = np.full(GX.shape, np.nan)
    Wmap[inside] = zernikeEval(coefs, idx, rq, tq, skip=REMOVE_TERMS)
    Wraw = np.full(GX.shape, np.nan)
    Wraw[inside] = interp(np.column_stack([rq, tq]))
    def onGrid(c):
        M = np.full(GX.shape, np.nan)
        M[inside] = npleg.legval2d(GX[inside], GY[inside], c)
        return M
    taRmap = np.full(GX.shape, np.nan)
    taRmap[inside] = (onGrid(cX)[inside] * np.cos(tq) + onGrid(cY)[inside] * np.sin(tq))

    return {"idx": idx, "coefs": coefs, "fitRms": fitRms, "rmsX": rmsX, "rmsY": rmsY,
            "nZernike": nZernike, "nPointsFit": int(len(Wq)),
            "surface": {k_: mapping[k_] for k_ in ("R", "f", "D", "K", "z")},
            "taX": onGrid(cX), "taY": onGrid(cY), "taR": taRmap,
            "Wraw": Wraw, "W": Wmap, "grid": g,
            "Wpolar": (rho, thetaExt, np.concatenate([Wpolar, Wpolar[:, :1]], axis=1))}


def computeWF(wf, geom, progress=None):
    # Final fit on the full-size image:
    #  1) W_r (result of the TA integration, Eq. 14) is interpolated to a matrix of
    #     the same size as the crop, [c d], with the same pupil as the crop
    #     (each crop pixel is mapped to the working frame with the same rotation and
    #     translation applied by the pipeline, then normalized to the pupil, Eq. 37).
    #  2) The Zernike polynomials of degree n are fitted by least squares (Eq. 49)
    #     to ALL the pupil pixels of that large W_r.
    #  3) W_F = that fit evaluated on the same [c d] matrix (NaN outside the pupil).
    h, w = geom["cropShape"]
    A = geom["cropToWork"]
    cx, cy, r = geom["pupil"]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xw = A[0, 0] * xx + A[0, 1] * yy + A[0, 2]
    yw = A[1, 0] * xx + A[1, 1] * yy + A[1, 2]
    u, v = (xw - cx) / r, (yw - cy) / r
    rho = np.sqrt(u ** 2 + v ** 2)
    inside = rho <= 1.0
    rin = rho[inside]
    theta = np.mod(np.arctan2(v[inside], u[inside]), 2 * np.pi)
    idx = wf["idx"]

    # 1) W_r interpolated to the crop size
    rG, tG, Wp = wf["Wpolar"]
    interp = RegularGridInterpolator((rG, tG), Wp)
    Wr_in = interp(np.column_stack([rin, theta]))

    # 2) Zernike fit of degree n on the large W_r
    p = (lambda f: progress(0.75 * f)) if progress else None
    coefs = zernikeFit(idx, rin, theta, Wr_in, progress=p)

    # 3) W_F on the same matrix
    p = (lambda f: progress(0.75 + 0.25 * f)) if progress else None
    full = zernikeEval(coefs, idx, rin, theta, progress=p)
    low = np.zeros_like(full)
    for j, Z in zernikeTerms(idx[:max(REMOVE_TERMS) + 1], rin, theta):
        if j in REMOVE_TERMS:
            low += coefs[j] * Z
    WF = np.full((h, w), np.nan)
    WFc = np.full((h, w), np.nan)
    Wr = np.full((h, w), np.nan)
    WF[inside] = full
    WFc[inside] = full - low
    Wr[inside] = Wr_in
    rms = float(np.sqrt(np.mean((full - Wr_in) ** 2)))
    return {"WF": WF, "WFc": WFc, "Wr": Wr, "coefs": coefs, "nPoints": int(inside.sum()),
            "fitRms": rms}


def plotPupilMap(M, title, label, cmap="jet"):
    fig, ax = plt.subplots(figsize=(7.5, 7))
    im = ax.imshow(M, cmap=cmap, extent=[-1, 1, 1, -1])
    plt.colorbar(im, ax=ax, label=label, fraction=0.046)
    ax.set_title(title)
    ax.set_xlabel("x / r (normalized)")
    ax.set_ylabel("y / r (normalized)")
    plt.tight_layout()
    return figToArray(fig)


def plotWavefront3D(M, g, title):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    style3D(ax)
    GX, GY = np.meshgrid(g, g)
    Z = np.ma.masked_invalid(M)
    surf = ax.plot_surface(GX, GY, Z, cmap="jet", linewidth=0, antialiased=True,
                           vmin=np.nanmin(M), vmax=np.nanmax(M))
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="W (µm)")
    ax.set_xlabel("x / r (normalized)")
    ax.set_ylabel("y / r (normalized)")
    ax.set_zlabel("W (µm)")
    ax.invert_yaxis()
    ax.set_title(title)
    plt.tight_layout()
    return figToArray(fig)


def plotZernikeBars(coefs, idx, nMax=6):
    sel = [j for j, (n, m) in enumerate(idx) if n <= nMax and j not in REMOVE_TERMS]
    labels = [ZERNIKE_NAMES.get(idx[j], f"Z({idx[j][0]},{idx[j][1]})") for j in sel]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(sel)), [coefs[j] for j in sel], color="#0B5394")
    ax.set_xticks(range(len(sel)))
    ax.set_xticklabels([f"{j}: {l}" for j, l in zip(sel, labels)], rotation=70,
                       ha="right", fontsize=8)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("Coefficient (µm)")
    ax.set_title(f"Zernike coefficients up to n = {nMax} "
                 "(piston, tilts and defocus removed)")
    plt.tight_layout()
    return figToArray(fig)


def toMatrix(df, shape):
    # Matrix of the image size: each point stores its adjusted difference at its
    # theoretical position; pixels without data are NaN
    M = np.full(shape, np.nan, dtype=np.float64)
    r = np.clip(np.round(df["y"].values).astype(int), 0, shape[0] - 1)
    c = np.clip(np.round(df["x"].values).astype(int), 0, shape[1] - 1)
    M[r, c] = df["diffAdj"].values
    return M


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def processImg(route, progressCb=None, nZernike=None, surface=None, fringeEditor=None,
               pupilFromCrop=False):
    nZernike = ZERNIKE_Q if nZernike is None else int(nZernike)
    surf = dict(SURFACE_DEFAULTS)
    if surface:
        surf.update(surface)
    mapping = surfaceMapping(surf["R"], surf["f"], surf["D"], surf["K"], surf["z"])

    def prog(val):
        if progressCb:
            progressCb(val)

    results = []
    stageStart = {"A": 0}   # index of the first step of each stage (A-G)

    bgrImg = cv2.imread(route)
    if bgrImg is None:
        raise RuntimeError(f"Cannot open image: {route}")
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    results.append({"title": "Input Image", "image": rgbImg.copy()})
    prog(2)

    # ── Initial orientation with the Hough transform ──
    # Exit pupil given by the user: the circle inscribed in the square crop (its center
    # is the crop center, which the rotations about the crop center do not move)
    hC, wC = bgrImg.shape[:2]
    cropCircle = ((wC - 1) / 2.0, (hC - 1) / 2.0, min(wC, hC) / 2.0) if pupilFromCrop else None
    houghRotation, nHough, houghMethod = stripeOrientation(rgbImg, cropCircle)
    stripeAngle = houghRotation - 90.0
    rgbOrig = rgbImg.copy()          # every later transformation starts from the original
    rgbImg = rotateImage(rgbOrig, stripeAngle + 90)
    results.append({"title": "Rotated Image (Hough)", "image": rgbImg.copy(),
                    "text": [f"Detected lines: {nHough}",
                             f"Orientation from: {houghMethod}",
                             f"Rotation applied: {stripeAngle + 90:.4f}°"]})
    prog(5)

    # ── Vertical alignment verification (central stripes) ──
    rgbImg, alignVis, alignText, alignCorrection = verifyVerticalAlignment(
        rgbOrig, stripeAngle + 90, cropCircle)
    results.append({"title": "Vertical Alignment Verification", "image": alignVis,
                    "text": alignText})
    prog(9)

    # ── Exit pupil: center the image on it in a square frame ──
    # The pupil center becomes the rotation center, so the 90° rotated
    # Ronchigram overlaps exactly with the vertical one
    # Crop pixel -> working pixel: both rotations are about the same center, so they
    # add up; then the translation that centers the pupil. Used to build W_F.
    Rtot = cv2.getRotationMatrix2D((wC / 2, hC / 2), stripeAngle + 90 + alignCorrection, 1.0)
    if cropCircle is not None:
        cx, cy = (Rtot @ np.array([cropCircle[0], cropCircle[1], 1.0])).tolist()
        rP = cropCircle[2]
        pupilSource = "circle of the crop (fitted by the user)"
    else:
        cx, cy, rP = detectPupil(rgbImg)
        pupilSource = "automatic detection (no crop)"
    side = int(2 * np.ceil(rP * PUPIL_MARGIN))
    c0 = (side - 1) / 2.0
    cropToWork = np.vstack([Rtot, [0, 0, 1]])
    cropToWork[0, 2] += c0 - cx
    cropToWork[1, 2] += c0 - cy
    # A single resampling of the original crop: rotation (Hough + verticality) and
    # centering of the pupil in one affine transformation
    rgbImg = cv2.warpAffine(rgbOrig, cropToWork[:2], (side, side), flags=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_CONSTANT)
    pupil = (c0, c0, rP)
    vis = rgbImg.copy()
    cv2.circle(vis, (int(round(c0)), int(round(c0))), int(round(rP)), (255, 255, 255), 2)
    cv2.drawMarker(vis, (int(round(c0)), int(round(c0))), (255, 255, 255),
                   cv2.MARKER_CROSS, 20, 2)
    results.append({"title": "Exit Pupil and Centering", "image": vis,
                    "text": [f"Pupil: {pupilSource}",
                             f"Pupil center (original): ({cx:.3f}, {cy:.3f}) px",
                             f"Pupil radius: {rP:.3f} px",
                             f"Working image: {side} x {side} px"]})

    # ── Optical surface and scale ──
    mmPerPx = abs(mapping["xMarg"]) / rP
    surfTable = pd.DataFrame({"Value": [
        f"{mapping['R']:g} mm", f"{mapping['f']:g} mm", f"{mapping['D']:g} mm",
        f"{mapping['K']:g}", f"{mapping['z']:g} mm", f"{mapping['n']:.4f}",
        f"{abs(mapping['xMarg']):.4f} mm", f"{1000 * mmPerPx:.4f} µm/px",
        f"{100 * mapping['maxDev']:.3f} % of the pupil radius"]},
        index=["Radius of curvature R", "Focal length f", "Diameter D",
               "Conic constant K", "Distance z (vertex to Ronchigram plane)",
               "Refractive index n = 1 + R/f", "Pupil radius at the camera plane",
               "Scale", "Ideal stripe bending (max.)"])
    surfTable.index.name = "Optical surface"
    results.append({"title": "Optical Surface Parameters", "table": surfTable,
                    "text": ["Ideal Ronchigram: grating lines traced exactly through the "
                             "surface (conic R, K; n from R/f) to the plane z",
                             "Scale = pupil radius at z / pupil radius in px",
                             "W = -(1/z) integral of TA dh, in micrometers"]})
    prog(12)

    # ── Vertical Ronchigram ──
    stageStart["B"] = len(results)
    V = analyzeRonchigram(rgbImg, pupil, "Vertical", False, results, prog, 12, 40, mapping,
                          editor=fringeEditor)

    # ── Horizontal Ronchigram: 90° rotation, NO inclination correction ──
    stageStart["C"] = len(results)
    rgbH = cv2.rotate(rgbImg, cv2.ROTATE_90_CLOCKWISE)
    results.append({"title": "Rotated 90° (Horizontal Ronchigram)", "image": rgbH.copy(),
                    "text": ["Rotation without inclination correction."]})
    # The horizontal Ronchigram is the vertical one rotated 90° about the pupil center;
    # analyzed on its transpose, that is the vertical image flipped upside down. So the
    # fringes confirmed in the vertical review are reused: row -> side - 1 - row.
    stripesH = [{"peak": s_["peak"], "anchors": [(side - 1 - r_, c_) for (r_, c_) in s_["anchors"]]}
                for s_ in V["stripes"]]
    H = analyzeRonchigram(rgbH, pupil, "Horizontal", True, results, prog, 40, 68, mapping,
                          stripesOverride=stripesH, maxDegree=V["maxDegree"])

    # ── Grid ──
    stageStart["D"] = len(results)
    G, gridExp, gridTheo, gridOverlay = buildGrid(V, H, pupil, rgbImg.shape[:2])
    results.append({"title": "Experimental Grid", "image": gridExp})
    results.append({"title": "Theoretical Grid", "image": gridTheo})
    results.append({"title": "Grid Intersections: Experimental vs Theoretical",
                    "image": gridOverlay,
                    "text": [f"Corresponding intersections: {len(G)}",
                             "Black: theoretical, purple: experimental, orange: displacement"]})
    summaryG = G[["dx", "dy", "diffRaw"]].agg(["mean", "max", "min", "std"])
    summaryG.index = ["Mean", "Max", "Min", "Std"]
    results.append({"title": "Grid Difference Summary", "table": summaryG.round(4),
                    "excel": G,
                    "text": [f"Intersections: {len(G)}",
                             "Difference = |P_exp - P_theo| (Euclidean)",
                             "dx, dy kept with sign in the Excel file",
                             "Adjusted so that min = 0"]})
    results.append({"title": "Difference Matrix - Grid",
                    "image": plotDiffMap(G, "Difference Matrix - Grid", pupil,
                                         rgbImg.shape[:2]),
                    "text": [f"Points: {len(G)}",
                             f"Adjusted max: {G['diffAdj'].max():.6f} px"]})
    prog(74)

    # ── Refinement 1 (REFINE_K1 points between lines) ──
    stageStart["E"] = len(results)
    shape = rgbImg.shape[:2]
    base = {
        "Vertical":   (V["df"], V["df"].groupby("line")["x"].mean().values,
                       V["df"]["y"].values, True, False),
        "Horizontal": (H["df"], H["df"]["x"].values,
                       H["df"].groupby("line")["y"].mean().values, False, True),
        "Grid":       (G, G.groupby("lineV")["x"].mean().values,
                       G.groupby("lineH")["y"].mean().values, True, True),
    }
    ref1, ref2, info = {}, {}, {}
    for name, (df, xc_, yc_, rx, ry) in base.items():
        isGrid = name == "Grid"
        d1, kx1, ky1 = refineMatrix(df, xc_, yc_, rx, ry, REFINE_K1, pupil, vector=isGrid)
        ref1[name] = d1
        info[name] = [len(df), len(d1), max(kx1, ky1)]
        method = (f"{GRID_INTERP_METHOD} on dx, dy -> |d|" if isGrid else INTERP_METHOD)
        results.append({
            "title": f"Refinement 1 - Difference {name}",
            "image": plotDiffMap(d1, f"Refinement 1 ({REFINE_K1} pts) - {name}",
                                 pupil, shape),
            "text": [f"Original points: {len(df)}",
                     f"Refined points: {len(d1)}",
                     f"Points inserted between lines: requested {REFINE_K1}, "
                     f"applied {max(kx1, ky1)}",
                     f"Interpolation: {method}"]})
    prog(84)

    # ── Refinement 2 (REFINE_K2 points between lines) — final point clouds ──
    stageStart["F"] = len(results)
    for name in base:
        d1 = ref1[name]
        rx, ry = base[name][3], base[name][4]
        d2, kx2, ky2 = refineMatrix(d1, d1["x"].values, d1["y"].values,
                                    rx, ry, REFINE_K2, pupil, vector=(name == "Grid"))
        ref2[name] = d2
        info[name] += [len(d2), max(kx2, ky2)]
        txt = [f"Input points: {len(d1)}",
               f"Refined points: {len(d2)}",
               f"Points inserted between lines: requested {REFINE_K2}, "
               f"applied {max(kx2, ky2)}"]
        if max(kx2, ky2) < REFINE_K2:
            txt.append(f"Limited by Nyquist: points never closer than {MIN_SPACING} px")
        results.append({"title": f"Refinement 2 - Difference {name}",
                        "image": plotDiffMap(d2, f"Point Cloud (Refinement 2) - {name}",
                                             pupil, shape),
                        "text": txt})
        results.append({"title": f"Point Cloud 3D - {name}",
                        "image": plot3D(d2, f"Point Cloud - {name}")})
    prog(91)

    # ── Final output ──
    matrices = {f"Difference_Ronchi{n}": toMatrix(ref2[k], shape)
                for n, k in [("Vertical", "Vertical"), ("Horizontal", "Horizontal"),
                             ("Grid", "Grid")]}
    summary = pd.DataFrame(
        [[info[n][0], info[n][1], info[n][2], info[n][3], info[n][4],
          ref2[n]["diffAdj"].min(), ref2[n]["diffAdj"].max()] for n in base],
        index=list(base),
        columns=["Orig. pts", "Ref.1 pts", "k1", "Ref.2 pts", "k2", "Min", "Max"])
    results.append({
        "title": "Output: Point Clouds (Final Matrices)",
        "table": summary,
        "excel": {"Vertical": ref2["Vertical"], "Horizontal": ref2["Horizontal"],
                  "Grid": ref2["Grid"]},
        "arrays": matrices,
        "data": {"pointClouds": ref2, "matrices": matrices, "pupil": pupil,
                 "original": {"Vertical": V["df"], "Horizontal": H["df"], "Grid": G}},
        "text": [f"Matrix size: {shape[0]} x {shape[1]} (float64, NaN = no data)",
                 "Positions = points on the " + ("experimental fringes" if TA_POSITION ==
                                                 "experimental" else "theoretical lines"),
                 "Values = adjusted differences (min = 0)",
                 "Download: Excel (3 sheets) + .npz with the 3 matrices"]})
    prog(93)

    # ── Extension of the point clouds to the whole exit pupil ──
    stageStart["G"] = len(results)
    full = {}
    fullInfo = {}
    for name in base:
        dF, rms, outline = extendToPupil(ref2[name], pupil, vector=(name == "Grid"))
        full[name] = dF
        nExt = int((dF["source"] == "extended").sum())
        fullInfo[name] = [len(ref2[name]), nExt, len(dF), 100 * nExt / len(dF), rms,
                          dF["diffAdj"].max()]
        results.append({
            "title": f"Full Pupil - Difference {name}",
            "image": plotDiffMap(dF, f"Full Pupil - {name}", pupil, shape,
                                 outline=outline),
            "text": [f"Measured points (kept): {len(ref2[name])}",
                     f"Extended points: {nExt} ({100 * nExt / len(dF):.1f}% of the pupil)",
                     f"Extension: polynomial of degree {FULL_PUPIL_DEGREE} "
                     f"+ residual decaying in {FULL_PUPIL_DECAY} px",
                     f"Polynomial fit RMS on the data: {rms:.4f} px",
                     "Cyan line: border of the measured region",
                     "Values outside it are extrapolated, not measured",
                     "Adjusted so that min = 0"]})
        results.append({"title": f"Full Pupil 3D - {name}",
                        "image": plot3D(dF, f"Full Pupil - {name}")})
    prog(95)

    fullMatrices = {f"Difference_Ronchi{n}_FullPupil": toMatrix(full[k], shape)
                    for n, k in [("Vertical", "Vertical"), ("Horizontal", "Horizontal"),
                                 ("Grid", "Grid")]}
    fullSummary = pd.DataFrame(
        [fullInfo[n] for n in base], index=list(base),
        columns=["Measured", "Extended", "Total", "% ext.", "Fit RMS", "Max"])
    results.append({
        "title": "Output: Full Pupil Matrices",
        "table": fullSummary,
        "excel": {"Vertical": full["Vertical"], "Horizontal": full["Horizontal"],
                  "Grid": full["Grid"]},
        "arrays": fullMatrices,
        "data": {"fullPupil": full, "matrices": fullMatrices, "pupil": pupil},
        "text": [f"Matrix size: {shape[0]} x {shape[1]} (float64, NaN = outside pupil)",
                 "Column 'source': data = measured/refined, extended = extrapolated",
                 "Values = adjusted differences (min = 0)",
                 "Download: Excel (3 sheets) + .npz with the 3 matrices"]})
    prog(96)

    # ── Wavefront (Eqs. 9-15 of the article), micrometers ──
    stageStart["H"] = len(results)
    # The integration uses the SIGNED differences (experimental - theoretical);
    # the maps adjusted to min = 0 are only for display (manuscript, Sec. 3.5)
    if WAVEFRONT_SOURCE == "grid":
        g_ = full["Grid"]
        taX = (g_["x"].values, g_["y"].values, g_["dx"].values)
        taY = (g_["x"].values, g_["y"].values, g_["dy"].values)
        srcText = "TA_x, TA_y = signed dx, dy of the grid"
    else:
        taX = (full["Vertical"]["x"].values, full["Vertical"]["y"].values,
               full["Vertical"]["signed"].values)
        taY = (full["Horizontal"]["x"].values, full["Horizontal"]["y"].values,
               full["Horizontal"]["signed"].values)
        srcText = "TA_x = vertical matrix, TA_y = horizontal matrix (signed)"
    wf = computeWavefront(taX, taY, pupil, nZernike, mapping, mmPerPx)
    prog(98)

    results.append({"title": "TA Surface Fit (Eq. 12) - x",
                    "image": plotPupilMap(wf["taX"], "TA_x fitted surface (Eq. 12)",
                                          "TA_x (px)"),
                    "text": [srcText,
                             f"Polynomial degree: {TA_FIT_DEGREE} (l, h <= {TA_FIT_DEGREE})",
                             f"Fit RMS: {wf['rmsX']:.4f} px"]})
    results.append({"title": "TA Surface Fit (Eq. 12) - y",
                    "image": plotPupilMap(wf["taY"], "TA_y fitted surface (Eq. 12)",
                                          "TA_y (px)"),
                    "text": [f"Polynomial degree: {TA_FIT_DEGREE}",
                             f"Fit RMS: {wf['rmsY']:.4f} px"]})
    results.append({"title": "Directional Derivative (Eq. 9)",
                    "image": plotPupilMap(wf["taR"], "TA along r (directional derivative)",
                                          "TA_r (px)"),
                    "text": ["TA_r = TA_x cos(theta) + TA_y sin(theta)",
                             "dW/dh = -TA_r / z  (TA in mm, h = height at the surface)"]})
    results.append({"title": "Wavefront W_r - Radial Integration (Eq. 14)",
                    "image": plotPupilMap(wf["Wraw"], "W_r: integration of TA (Eq. 14)",
                                          "W_r (µm)"),
                    "text": [f"Divisions of rho: {RADIAL_STEPS}",
                             f"Directions theta: {ANGULAR_STEPS}",
                             "Includes piston, tilt and defocus"]})
    names = ", ".join(ZERNIKE_NAMES.get(wf["idx"][j], str(j)) for j in REMOVE_TERMS)
    pv = float(np.nanmax(wf["W"]) - np.nanmin(wf["W"]))
    rmsW = float(np.nanstd(wf["W"]))
    results.append({"title": "Wavefront W_f (final)",
                    "image": plotPupilMap(wf["W"], f"W_f: Zernike fit of W_r (n = {nZernike})",
                                          "W_f (µm)"),
                    "text": [f"Zernike degree n = {nZernike} "
                             f"({len(wf['idx'])} coefficients, OSA/ANSI order)",
                             f"Fitted on {wf['nPointsFit']} points of W_r",
                             f"Removed: {names}",
                             f"P-V: {pv:.4f} µm",
                             f"RMS: {rmsW:.4f} µm",
                             f"R = {mapping['R']:g} mm, f = {mapping['f']:g} mm, "
                             f"D = {mapping['D']:g} mm, K = {mapping['K']:g}, "
                             f"z = {mapping['z']:g} mm"]})
    results.append({"title": "Wavefront W_f 3D",
                    "image": plotWavefront3D(wf["W"], wf["grid"], "W_f(rho, theta)")})

    cropH, cropW = bgrImg.shape[:2]
    summaryW = pd.DataFrame(
        {"Value": [nZernike, len(wf["idx"]), wf["nPointsFit"], pv, rmsW,
                   f"{cropH} x {cropW}"]},
        index=["Zernike degree n", "Coefficients", "Points of W_r used",
               "P-V of W_f (µm)", "RMS of W_f (µm)", "Size of W_F"])
    summaryW.index.name = "Wavefront"
    results.append({
        "title": "Output: Zernike Coefficients and W_F",
        "table": summaryW,
        "wavefrontExport": {"wf": wf, "geom": {"cropShape": (cropH, cropW),
                                                "cropToWork": cropToWork,
                                                "pupil": pupil}},
        "data": {"wavefront": wf},
        "text": [f"1) W_r (integration of TA) is interpolated to the crop size "
                 f"({cropH} x {cropW}), with the same pupil as the crop",
                 f"2) Zernike fit up to n = {nZernike} (OSA/ANSI order) on all the "
                 "pupil pixels of that W_r",
                 "3) W_F = that fit on the same matrix, NaN outside the pupil",
                 "Download: coefficients (.xlsx, .csv) and W_F (.mat, .csv)",
                 "W_F is computed when downloading; it may take a while"]})
    prog(100)

    # Tag every step with its stage (used by the GUI to group the tabs)
    order = sorted(stageStart.items(), key=lambda kv: kv[1])
    for k, (stage, start) in enumerate(order):
        end = order[k + 1][1] if k + 1 < len(order) else len(results)
        for i in range(start, end):
            results[i]["stage"] = stage
    return results

 #------------------------------------------------------------------------------------------------------------------------------------------------------

# ── Colors ──
# ══════════════════════════════════════════════════════════════════════════════
# GRAPHICAL USER INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

# ── Header: institution logos and titles ──
# Put the logo images (PNG/JPG) in the same folder as this script with these
# names, or change the names here. If a file is missing, a placeholder is shown.
# The three logo boxes are square (LOGO_SIZE x LOGO_SIZE px); square images fit best.
LOGO_LEFT_FILE  = "UNAM.png"          # left side, first box
LOGO_LEFT2_FILE = "UASLPlogo.png"     # left side, second box
LOGO_RIGHT_FILE = "INAOE.jpg"         # right side
LOGO_SIZE       = 96
APP_TITLE    = "Variational Ronchi Sensor"
APP_SUBTITLE = ""   # second header bar: empty for now, the space is kept for future text

# ── Color palette ──
NAVY      = "#0B3B78"   # header frame
NAVY_DARK = "#082C5A"   # header title panel
GOLD      = "#E2B33C"   # accent line / selected tab / running stage
BG        = "#F3F5F9"   # window background
WHITE     = "#FFFFFF"
TEXT      = "#1A1A1A"
MUTED     = "#5F6B7A"
TAB_TEXT  = "#1F4E8C"   # unselected tab text
TAB_HOVER = "#EAF0F8"
BORDER    = "#C9D3E0"
GREEN     = "#1E9E4A"   # finished stage
PURPLE    = "#7E3FA3"   # pending stage
RED       = "#C0392B"   # failed stage
BLUE_BTN  = "#0B5394"   # main action button
GREEN_BTN = "#2E8B57"   # download button
FONT      = "Segoe UI"

# ── Pipeline stages (letter, tab name) ──
STAGES = [
    ("A", "A. Preparation"),
    ("B", "B. Vertical Ronchigram"),
    ("C", "C. Horizontal Ronchigram"),
    ("D", "D. Grid"),
    ("E", "E. Refinement 1"),
    ("F", "F. Refinement 2"),
    ("G", "G. Full pupil"),
    ("H", "H. Wavefront"),
]
STAGE_NAME = dict(STAGES)
# Number shown in each status circle and English name shown in the status text
STAGE_NUMBER = {s: i + 1 for i, (s, _) in enumerate(STAGES)}
STAGE_NAME_EN = {
    "A": "Preparation",
    "B": "Vertical Ronchigram",
    "C": "Horizontal Ronchigram",
    "D": "Grid",
    "E": "Refinement 1",
    "F": "Refinement 2",
    "G": "Full pupil",
    "H": "Wavefront",
}
def stageStatusName(stage):
    return f"Stage {STAGE_NUMBER[stage]}: {STAGE_NAME_EN[stage]}"
# Progress value (%) reported by processImg when each stage ends
STAGE_END = {"A": 12, "B": 40, "C": 68, "D": 74, "E": 84, "F": 93, "G": 96, "H": 101}

# ── Global state ──
imgPath    = None
origPil    = None
cropRect   = None
cropActive = False
cropStart  = None
cropRectId = None
results    = []      # list of result dicts from processImg
slideIdx   = 0       # index of the current step
stageSteps = {}      # stage letter -> list of step indices
curStage   = None
running    = False
renderJob  = None
nZernikeRun = ZERNIKE_Q
surfaceRun = dict(SURFACE_DEFAULTS)
exportRunning = False

BARH = 22
UI_SCALE = 1.0       # set after the window is created (screen DPI / 96)

# Crop editing state (preview geometry and current drag)
prevGeom   = None    # (scale, offsetX, offsetY) of the preview image on the canvas
dragMode   = None    # "new", "move" or ("resize", corner)
dragStart  = None
cropAtPress = None
previewJob = None
cropPrevJob = None
HANDLE     = 7       # half size (px) of the corner handles


# ══════════════════════════════════════════════════════════════════════════════
# WIDGETS
# ══════════════════════════════════════════════════════════════════════════════
class TabStrip(tk.Frame):
    # Horizontal row of tabs with scroll arrows (selected tab in gold)
    def __init__(self, parent, onSelect, fontSize=10, pady=7):
        super().__init__(parent, bg=WHITE, highlightthickness=1, highlightbackground=BORDER)
        self.onSelect = onSelect
        self.fontSize = fontSize
        self.pady = pady
        self.items = []
        self.selected = None

        self.btnL = tk.Label(self, text="‹", bg=WHITE, fg=MUTED, font=(FONT, 15), padx=8,
                             cursor="hand2")
        self.btnR = tk.Label(self, text="›", bg=WHITE, fg=MUTED, font=(FONT, 15), padx=8,
                             cursor="hand2")
        self.btnL.pack(side="left", fill="y")
        self.btnR.pack(side="right", fill="y")
        self.canvas = tk.Canvas(self, bg=WHITE, highlightthickness=0, height=10,
                                xscrollincrement=S(40))
        self.canvas.pack(side="left", fill="x", expand=True)
        self.inner = tk.Frame(self.canvas, bg=WHITE)
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._onInner)
        self.btnL.bind("<Button-1>", lambda e: self.scroll(-4))
        self.btnR.bind("<Button-1>", lambda e: self.scroll(4))
        for w in (self.canvas, self.inner):
            w.bind("<MouseWheel>", lambda e: self.scroll(-1 if e.delta > 0 else 1))
            w.bind("<Button-4>", lambda e: self.scroll(-1))
            w.bind("<Button-5>", lambda e: self.scroll(1))

    def _onInner(self, _):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"),
                              height=self.inner.winfo_reqheight())

    def scroll(self, units):
        self.canvas.xview_scroll(units, "units")

    def setItems(self, labels):
        for w in self.inner.winfo_children():
            w.destroy()
        self.items = []
        self.selected = None
        for i, text in enumerate(labels):
            lbl = tk.Label(self.inner, text=text, bg=WHITE, fg=TAB_TEXT,
                           font=(FONT, self.fontSize), padx=14, pady=self.pady, cursor="hand2")
            lbl.pack(side="left")
            lbl.bind("<Button-1>", lambda e, i=i: self.onSelect(i))
            lbl.bind("<Enter>", lambda e, i=i: self._hover(i, True))
            lbl.bind("<Leave>", lambda e, i=i: self._hover(i, False))
            lbl.bind("<MouseWheel>", lambda e: self.scroll(-1 if e.delta > 0 else 1))
            lbl.bind("<Button-4>", lambda e: self.scroll(-1))
            lbl.bind("<Button-5>", lambda e: self.scroll(1))
            self.items.append(lbl)
        self.canvas.xview_moveto(0)

    def _hover(self, i, on):
        if i != self.selected and i < len(self.items):
            self.items[i].config(bg=TAB_HOVER if on else WHITE)

    def select(self, i):
        self.selected = i
        for k, lbl in enumerate(self.items):
            sel = k == i
            lbl.config(bg=GOLD if sel else WHITE, fg=TEXT if sel else TAB_TEXT,
                       font=(FONT, self.fontSize, "bold" if sel else "normal"))
        self.after_idle(lambda: self.ensureVisible(i))

    def ensureVisible(self, i, tries=0):
        if i is None or i >= len(self.items):
            return
        self.update_idletasks()
        total = self.inner.winfo_width()
        if (total <= 1 or self.items[i].winfo_width() <= 1) and tries < 10:
            # Widgets not laid out yet: try again shortly
            self.after(30, lambda: self.ensureVisible(i, tries + 1))
            return
        view = self.canvas.winfo_width()
        if total <= view or total <= 1:
            return
        lbl = self.items[i]
        x0, x1 = lbl.winfo_x(), lbl.winfo_x() + lbl.winfo_width()
        left = self.canvas.canvasx(0)
        if x0 < left:
            self.canvas.xview_moveto(x0 / total)
        elif x1 > left + view:
            self.canvas.xview_moveto((x1 - view) / total)


def makeButton(parent, text, command, bg, fg=WHITE, size=11, bold=True, **kw):
    return tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                     activebackground=bg, activeforeground=fg, disabledforeground="#F0F0F0",
                     font=(FONT, size, "bold" if bold else "normal"),
                     relief="flat", bd=0, cursor="hand2", **kw)


def card(parent, title=None):
    # White panel with thin border and optional navy title (like a group box)
    outer = tk.Frame(parent, bg=WHITE, highlightthickness=1, highlightbackground=BORDER)
    if title:
        tk.Label(outer, text=title, bg=WHITE, fg=NAVY, font=(FONT, 11, "bold"),
                 anchor="w").pack(fill="x", padx=14, pady=(10, 4))
    return outer


def loadLogo(label, fileName, maxW, maxH, placeholder):
    # Shows the logo image if the file exists next to the script, else a placeholder
    # Folders searched, in order:
    #  1) inside the .exe (files added with PyInstaller --add-data)
    #  2) next to the .exe          3) next to this script      4) working folder
    import sys
    bases = []
    if hasattr(sys, "_MEIPASS"):
        bases.append(sys._MEIPASS)
    if getattr(sys, "frozen", False):
        bases.append(os.path.dirname(sys.executable))
    try:
        bases.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    bases.append(os.getcwd())
    path = next((os.path.join(b, fileName) for b in bases
                 if os.path.isfile(os.path.join(b, fileName))), "")
    if path:
        try:
            im = Image.open(path).convert("RGBA")
            im.thumbnail((maxW, maxH), Image.LANCZOS)
            photo = ImageTk.PhotoImage(im)
            label.config(image=photo, text="")
            label.image = photo
            return
        except Exception:
            pass
    label.config(text=f"{placeholder}\n{fileName}", fg="#9AA5B1",
                 font=(FONT, 8, "italic"), wraplength=maxW, justify="center")


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE STATUS
# ══════════════════════════════════════════════════════════════════════════════
def setStageState(stage, state):
    color = {"pending": PURPLE, "running": GOLD, "done": GREEN, "error": RED}[state]
    c = statusCircles[stage]
    c.itemconfig("dot", fill=color, outline=color)
    c.itemconfig("txt", fill=TEXT if state == "running" else WHITE)


def resetStatus():
    for s, _ in STAGES:
        setStageState(s, "pending")


def currentStageFor(pct):
    for s, _ in STAGES:
        if pct < STAGE_END[s]:
            return s
    return STAGES[-1][0]


def onStatusClick(stage):
    if results and stage in stageSteps:
        showStep(stageSteps[stage][0])


# ══════════════════════════════════════════════════════════════════════════════
# INPUT VIEW (select image and crop)
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# FRINGE REVIEW (the user confirms, edits, adds or deletes the anchor points)
# ══════════════════════════════════════════════════════════════════════════════
fe = {}          # state of the fringe review


def fringeEditorBridge(data):
    # Called from the processing thread: shows the review in the GUI thread and
    # waits until the user confirms (or cancels)
    ev = threading.Event()
    box = {}
    root.after(0, openFringeEditor, data, ev, box)
    ev.wait()
    return box.get("stripes")


def openFringeEditor(data, ev, box):
    fe.clear()
    stripes = [[(float(r), float(c)) for (r, c) in s_["anchors"]] for s_ in data["stripes"]]
    fe.update({"img": Image.fromarray(data["image"]), "chan": data["channel"].astype(np.float32),
               "hw": float(data["halfWidth"]), "pupil": data["pupil"], "ev": ev, "box": box,
               "stripes": stripes, "orig": [list(a) for a in stripes], "nAuto": len(stripes),
               "sel": None, "drag": None, "geom": None, "photo": None, "cacheKey": None})
    feSort()
    fe["sel"] = 0 if fe["stripes"] else None
    showView(viewFringes)
    lblStatus.config(text="Waiting for the fringe review…", fg=NAVY)
    lblFooter.config(text="Review the fringes: select one on the right, click on the image to add "
                          "anchor points, drag them to move them, right-click to delete them.")
    root.update_idletasks()
    feRefresh()


def feMeanCol(a):
    return float(np.mean([p[1] for p in a])) if a else float("inf")


def feSort():
    ref = fe["stripes"][fe["sel"]] if fe.get("sel") is not None else None
    fe["stripes"].sort(key=feMeanCol)
    fe["sel"] = next((i for i, a in enumerate(fe["stripes"]) if a is ref), None)


def feImg2Can(r, c):
    sc, ox, oy = fe["geom"]
    return ox + c * sc, oy + r * sc


def feCan2Img(x, y):
    sc, ox, oy = fe["geom"]
    return (y - oy) / sc, (x - ox) / sc


def feRender(_=None):
    if not fe or fe.get("img") is None:
        return
    cw, chh = max(10, feCanvas.winfo_width()), max(10, feCanvas.winfo_height())
    iw, ih = fe["img"].size
    sc = min(cw / iw, chh / ih)
    ox, oy = (cw - iw * sc) / 2, (chh - ih * sc) / 2
    fe["geom"] = (sc, ox, oy)
    if fe["cacheKey"] != (cw, chh):
        fe["photo"] = ImageTk.PhotoImage(fe["img"].resize((max(1, int(iw * sc)),
                                                            max(1, int(ih * sc))), Image.LANCZOS))
        fe["cacheKey"] = (cw, chh)
    feCanvas.delete("all")
    feCanvas.create_image(ox, oy, anchor="nw", image=fe["photo"])
    cx, cy, r = fe["pupil"]
    x0, y0 = feImg2Can(cy - r, cx - r)
    x1, y1 = feImg2Can(cy + r, cx + r)
    feCanvas.create_oval(x0, y0, x1, y1, outline="white", dash=(4, 3))
    rad = S(5)
    for i, a in enumerate(fe["stripes"]):
        col = FE_COLORS[i % len(FE_COLORS)]
        selected = i == fe["sel"]
        for (pr, pc) in a:
            x, y = feImg2Can(pr, pc)
            rr_ = rad + (2 if selected else 0)
            feCanvas.create_oval(x - rr_, y - rr_, x + rr_, y + rr_, fill=col,
                                 outline="white" if selected else "black",
                                 width=2 if selected else 1)
        if a:
            top = min(a, key=lambda p: p[0])
            x, y = feImg2Can(top[0], top[1])
            feCanvas.create_text(x, y - S(14), text=str(i + 1), fill=col,
                                 font=(FONT, 12, "bold"))


def feRefreshList():
    for wdg in feList.winfo_children():
        wdg.destroy()
    if not fe["stripes"]:
        tk.Label(feList, text="No fringes. Press “New fringe” to add one.", bg=WHITE,
                 fg=MUTED, font=(FONT, 9, "italic")).pack(anchor="w")
    for i, a in enumerate(fe["stripes"]):
        row = tk.Frame(feList, bg=WHITE)
        row.pack(fill="x", pady=1)
        tk.Label(row, bg=FE_COLORS[i % len(FE_COLORS)], width=2).pack(side="left", fill="y")
        sel = i == fe["sel"]
        n = len(a)
        warn = "  ⚠" if 0 < n < MIN_MANUAL_ANCHORS else ""
        tk.Button(row, text=f"Fringe {i + 1}  ·  {n} points{warn}", anchor="w",
                  command=lambda i=i: feSelect(i), relief="flat", bd=0, cursor="hand2",
                  bg=GOLD if sel else "#EEF3F9", fg=TEXT,
                  font=(FONT, 10, "bold" if sel else "normal"), padx=8, pady=2
                  ).pack(side="left", fill="x", expand=True, padx=(4, 0))
    nOk = sum(1 for a in fe["stripes"] if len(a) >= MIN_MANUAL_ANCHORS)
    feInfo.config(text=f"Detected automatically: {fe['nAuto']}   ·   Now: {len(fe['stripes'])} "
                       f"({nOk} with at least {MIN_MANUAL_ANCHORS} points)")


def feRefresh():
    feRender()
    feRefreshList()


def feSelect(i):
    fe["sel"] = i
    feRefresh()


def feNearest(r, c, only=None):
    # (fringe, point) closest to (r, c) within the click tolerance
    sc = fe["geom"][0]
    tol = S(10) / sc
    best, bd = None, tol
    for i, a in enumerate(fe["stripes"]):
        if only is not None and i != only:
            continue
        for j, (pr, pc) in enumerate(a):
            d = np.hypot(pr - r, pc - c)
            if d <= bd:
                best, bd = (i, j), d
    return best


def feInside(r, c):
    iw, ih = fe["img"].size
    return 0 <= r < ih and 0 <= c < iw


def feOnPress(event):
    if not fe or fe["geom"] is None:
        return
    r, c = feCan2Img(event.x, event.y)
    if not feInside(r, c):
        return
    hit = feNearest(r, c, only=fe["sel"]) if fe["sel"] is not None else None
    if hit is not None:
        fe["drag"] = hit
        return
    other = feNearest(r, c)
    if other is not None:
        feSelect(other[0])
        return
    if fe["sel"] is None:
        lblFooter.config(text="Select a fringe on the right (or press “New fringe”) before "
                              "adding points.")
        return
    fe["stripes"][fe["sel"]].append((r, c))
    fe["stripes"][fe["sel"]].sort(key=lambda p: p[0])
    feSort()
    feRefresh()


def feOnDrag(event):
    if not fe or fe.get("drag") is None:
        return
    r, c = feCan2Img(event.x, event.y)
    iw, ih = fe["img"].size
    i, j = fe["drag"]
    fe["stripes"][i][j] = (min(max(r, 0), ih - 1), min(max(c, 0), iw - 1))
    feRender()


def feOnRelease(event):
    if not fe or fe.get("drag") is None:
        return
    i, j = fe["drag"]
    fe["stripes"][i].sort(key=lambda p: p[0])
    fe["drag"] = None
    feSort()
    feRefresh()


def feOnDelete(event):
    # Right click (or Ctrl + click): delete the nearest point of the selected fringe
    if not fe or fe["geom"] is None:
        return
    r, c = feCan2Img(event.x, event.y)
    hit = feNearest(r, c, only=fe["sel"]) if fe["sel"] is not None else feNearest(r, c)
    if hit is not None:
        del fe["stripes"][hit[0]][hit[1]]
        feRefresh()


def feNewFringe():
    fe["stripes"].append([])
    fe["sel"] = len(fe["stripes"]) - 1
    lblFooter.config(text=f"New fringe: click along its center on the image to add at least "
                          f"{MIN_MANUAL_ANCHORS} anchor points (from top to bottom).")
    feRefresh()


def feDeleteFringe():
    if fe["sel"] is None:
        return
    del fe["stripes"][fe["sel"]]
    fe["sel"] = None if not fe["stripes"] else min(fe["sel"], len(fe["stripes"]) - 1)
    feRefresh()


def feClearPoints():
    if fe["sel"] is not None:
        fe["stripes"][fe["sel"]].clear()
        feRefresh()


def feDeleteAll():
    if fe["stripes"] and messagebox.askyesno("Delete all fringes",
                                             "Delete all the fringes and their anchor points?"):
        fe["stripes"].clear()
        fe["sel"] = None
        feRefresh()


def feReset():
    fe["stripes"] = [list(a) for a in fe["orig"]]
    fe["sel"] = 0 if fe["stripes"] else None
    feRefresh()


def feFinish(stripes):
    box, ev = fe["box"], fe["ev"]
    box["stripes"] = stripes
    fe["img"] = None
    showView(viewInput)
    lblStatus.config(text="Running…", fg=NAVY)
    ev.set()


def feConfirm():
    stripes = [a for a in fe["stripes"] if len(a) > 0]
    short = [i + 1 for i, a in enumerate(fe["stripes"]) if 0 < len(a) < MIN_MANUAL_ANCHORS]
    if short:
        messagebox.showerror("Fringe review", "Fringe(s) " + ", ".join(map(str, short)) +
                             f" have fewer than {MIN_MANUAL_ANCHORS} anchor points. Add points "
                             "or delete the fringe.")
        return
    if len(stripes) < 2:
        messagebox.showerror("Fringe review", "At least 2 fringes are needed to continue.")
        return
    out = [{"peak": int(round(feMeanCol(a))), "anchors": sorted(a, key=lambda p: p[0])}
           for a in sorted(stripes, key=feMeanCol)]
    lblFooter.config(text=f"{len(out)} fringes confirmed. Running…")
    feFinish(out)


def feCancel():
    lblFooter.config(text="Pipeline cancelled in the fringe review.")
    feFinish(None)


def showView(view):
    viewInput.pack_forget()
    viewResults.pack_forget()
    viewFringes.pack_forget()
    view.pack(fill="both", expand=True)


def S(v):
    # Scales a size in pixels with the screen DPI (sharp and proportional on
    # high-resolution screens)
    return int(round(v * UI_SCALE))


def img2can(x, y):
    sc, ox, oy = prevGeom
    return ox + x * sc, oy + y * sc


def can2img(cx, cy):
    sc, ox, oy = prevGeom
    iw, ih = origPil.size
    return (min(iw, max(0.0, (cx - ox) / sc)), min(ih, max(0.0, (cy - oy) / sc)))


def buildPreview(_=None):
    # Resizes the image once to the canvas size (the expensive part). The crop is
    # drawn on top as canvas items, so moving or resizing it is immediate.
    global previewPhoto, prevGeom, previewJob
    previewJob = None
    canvasPrev.delete("all")
    cw, ch = max(10, canvasPrev.winfo_width()), max(10, canvasPrev.winfo_height())
    if origPil is None:
        canvasPrev.create_text(cw // 2, ch // 2, fill="#9AA5B1", font=(FONT, 12),
                               text="Select a Ronchigram image to start")
        return
    iw, ih = origPil.size
    sc = min(cw / iw, ch / ih)
    nw, nh = max(1, int(iw * sc)), max(1, int(ih * sc))
    ox, oy = (cw - nw) / 2, (ch - nh) / 2
    prevGeom = (sc, ox, oy)
    previewPhoto = ImageTk.PhotoImage(origPil.resize((nw, nh), Image.LANCZOS))
    canvasPrev.create_image(ox, oy, anchor="nw", image=previewPhoto, tags="img")
    drawCropOverlay()


def schedulePreview(_=None):
    global previewJob
    if previewJob:
        root.after_cancel(previewJob)
    previewJob = root.after(120, buildPreview)


def refreshPreview():
    buildPreview()
    updateCropInfo()


def drawCropOverlay():
    # Crop square, inscribed circle (pupil guide), corner handles and a dimmed outside
    canvasPrev.delete("crop")
    if origPil is None or prevGeom is None or not cropRect:
        return
    x0, y0 = img2can(cropRect[0], cropRect[1])
    x1, y1 = img2can(cropRect[2], cropRect[3])
    iw, ih = origPil.size
    ax0, ay0 = img2can(0, 0)
    ax1, ay1 = img2can(iw, ih)
    for (a, b, c, d) in ((ax0, ay0, ax1, y0), (ax0, y1, ax1, ay1),
                         (ax0, y0, x0, y1), (x1, y0, ax1, y1)):
        if c > a and d > b:
            canvasPrev.create_rectangle(a, b, c, d, fill="black", stipple="gray50",
                                        outline="", tags="crop")
    canvasPrev.create_rectangle(x0, y0, x1, y1, outline=GOLD, width=2, tags="crop")
    canvasPrev.create_oval(x0, y0, x1, y1, outline="white", dash=(4, 3), tags="crop")
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    canvasPrev.create_line(mx - 8, my, mx + 8, my, fill=GOLD, tags="crop")
    canvasPrev.create_line(mx, my - 8, mx, my + 8, fill=GOLD, tags="crop")
    hs = S(HANDLE)
    for (hx, hy) in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        canvasPrev.create_rectangle(hx - hs, hy - hs, hx + hs, hy + hs, fill=GOLD,
                                    outline="white", tags="crop")


def updateCropInfo():
    if origPil is None:
        return
    iw, ih = origPil.size
    lblFile.config(text=os.path.basename(imgPath) if imgPath else "—")
    lblSize.config(text=f"{iw} × {ih} px")
    if cropRect:
        x0, y0, x1, y1 = cropRect
        lblCrop.config(text=f"{x1 - x0} × {y1 - y0} px at ({x0}, {y0})")
    else:
        lblCrop.config(text="none (full image)")
    scheduleCropPreview()


def scheduleCropPreview():
    global cropPrevJob
    if cropPrevJob:
        root.after_cancel(cropPrevJob)
    cropPrevJob = root.after(150, drawCropPreview)


def drawCropPreview():
    # Enlarged view of the selected region, to check the crop before running
    global cropPrevPhoto, cropPrevJob
    cropPrevJob = None
    canvasCropPrev.delete("all")
    size = int(canvasCropPrev.cget("width"))
    if origPil is None or not cropRect:
        canvasCropPrev.create_text(size // 2, size // 2, text="No crop", fill="#9AA5B1",
                                   font=(FONT, 10))
        return
    im = origPil.crop(cropRect).resize((size, size), Image.LANCZOS)
    cropPrevPhoto = ImageTk.PhotoImage(im)
    canvasCropPrev.create_image(0, 0, anchor="nw", image=cropPrevPhoto)
    canvasCropPrev.create_oval(1, 1, size - 2, size - 2, outline="white", dash=(4, 3))


def setCrop(x0, y0, side):
    # Square crop inside the image (side and position clamped)
    global cropRect
    iw, ih = origPil.size
    side = int(round(max(10, min(side, iw, ih))))
    x0 = int(round(min(max(0, x0), iw - side)))
    y0 = int(round(min(max(0, y0), ih - side)))
    cropRect = (x0, y0, x0 + side, y0 + side)


def handleAt(cx, cy):
    if not cropRect:
        return None
    x0, y0 = img2can(cropRect[0], cropRect[1])
    x1, y1 = img2can(cropRect[2], cropRect[3])
    tol = S(HANDLE) + 3
    for k, (hx, hy) in enumerate(((x0, y0), (x1, y0), (x0, y1), (x1, y1))):
        if abs(cx - hx) <= tol and abs(cy - hy) <= tol:
            return k
    return None


def insideCrop(cx, cy):
    if not cropRect:
        return False
    x0, y0 = img2can(cropRect[0], cropRect[1])
    x1, y1 = img2can(cropRect[2], cropRect[3])
    return x0 <= cx <= x1 and y0 <= cy <= y1


def selectImage():
    global imgPath, origPil, cropRect
    if running:
        return
    path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif")]
    )
    if not path:
        return
    imgPath = path
    cropRect = None
    origPil = Image.open(path).convert("RGB")
    showView(viewInput)
    root.update_idletasks()
    refreshPreview()
    btnProcess.config(bg=BLUE_BTN, state="normal", cursor="hand2")
    lblFooter.config(text="Image loaded. Press “New crop” and drag a square around the "
                          "diffraction order to analyze.")


def toggleCrop():
    # "New crop" mode: the next drag draws a new square even inside the current one
    global cropActive
    if origPil is None or running:
        return
    cropActive = not cropActive
    canvasPrev.config(cursor="crosshair" if cropActive else "")
    btnCrop.config(bg=GOLD if cropActive else "#44546A",
                   fg=TEXT if cropActive else WHITE,
                   activebackground=GOLD if cropActive else "#44546A")
    lblFooter.config(text="Drag on the image to draw a new square crop." if cropActive else "")


def clearCrop():
    global cropRect
    if running:
        return
    cropRect = None
    drawCropOverlay()
    updateCropInfo()


def onCropPress(event):
    global dragMode, dragStart, cropAtPress
    if origPil is None or running or prevGeom is None:
        return
    dragStart = can2img(event.x, event.y)
    cropAtPress = cropRect
    h = handleAt(event.x, event.y)
    if cropRect and not cropActive and h is not None:
        dragMode = ("resize", h)
    elif cropRect and not cropActive and insideCrop(event.x, event.y):
        dragMode = "move"
    else:
        dragMode = "new"


def onCropDrag(event):
    if dragMode is None:
        return
    ix, iy = can2img(event.x, event.y)
    sx, sy = dragStart
    if dragMode == "new":
        side = max(abs(ix - sx), abs(iy - sy))
        setCrop(sx if ix >= sx else sx - side, sy if iy >= sy else sy - side, side)
    elif dragMode == "move":
        x0, y0, x1, y1 = cropAtPress
        setCrop(x0 + ix - sx, y0 + iy - sy, x1 - x0)
    else:
        # Resize from a corner; the opposite corner stays fixed
        x0, y0, x1, y1 = cropAtPress
        corner = dragMode[1]
        axn = x1 if corner in (0, 2) else x0
        ayn = y1 if corner in (0, 1) else y0
        side = max(abs(ix - axn), abs(iy - ayn))
        nx0 = axn - side if corner in (0, 2) else axn
        ny0 = ayn - side if corner in (0, 1) else ayn
        setCrop(nx0, ny0, side)
    drawCropOverlay()


def onCropRelease(event):
    global dragMode, cropRect
    if dragMode is None:
        return
    if dragMode == "new" and cropRect and cropRect[2] - cropRect[0] <= 10:
        cropRect = cropAtPress
    dragMode = None
    if cropActive:
        toggleCrop()
    drawCropOverlay()
    updateCropInfo()
    lblFooter.config(text="Adjust the crop: drag inside to move it, drag a corner to resize "
                          "it, mouse wheel to zoom, arrow keys to nudge (Shift = 10 px).")


def onCropMotion(event):
    if origPil is None or prevGeom is None or dragMode is not None:
        return
    if cropActive:
        cur = "crosshair"
    elif handleAt(event.x, event.y) is not None:
        cur = "sizing"
    elif insideCrop(event.x, event.y):
        cur = "fleur"
    else:
        cur = "crosshair"
    canvasPrev.config(cursor=cur)


def onCropWheel(event):
    # Mouse wheel: enlarge / reduce the crop around its center
    if not cropRect or running:
        return
    up = (getattr(event, "delta", 0) > 0) or getattr(event, "num", 0) == 4
    x0, y0, x1, y1 = cropRect
    side = (x1 - x0) * (1.03 if up else 1 / 1.03)
    c = ((x0 + x1) / 2, (y0 + y1) / 2)
    setCrop(c[0] - side / 2, c[1] - side / 2, side)
    drawCropOverlay()
    updateCropInfo()


def nudgeCrop(dx, dy):
    if not cropRect or running:
        return
    x0, y0, x1, y1 = cropRect
    setCrop(x0 + dx, y0 + dy, x1 - x0)
    drawCropOverlay()
    updateCropInfo()


def onArrow(dx, dy, event):
    # Arrow keys: next/previous step in the results, or move the crop in the input view
    if isinstance(root.focus_get(), (tk.Entry, tk.Spinbox)):
        return
    if viewResults.winfo_ismapped():
        if dx:
            navigate(dx)
    elif viewInput.winfo_ismapped():
        step = 10 if (event.state & 0x0001) else 1
        nudgeCrop(dx * step, dy * step)


# ══════════════════════════════════════════════════════════════════════════════
# PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def updateBar(pct):
    w = max(1, canvasBar.winfo_width())
    canvasBar.coords(barFill, 0, 0, int(w * pct / 100), BARH)
    lblPercent.config(text=f"{int(pct)}%")
    stage = currentStageFor(pct)
    for s, _ in STAGES:
        if STAGE_END[s] <= pct:
            setStageState(s, "done")
        elif s == stage:
            setStageState(s, "running")
    lblStatus.config(text=f"Running {stageStatusName(stage)} …", fg=NAVY)


def runProcessing():
    try:
        if cropRect:
            imgToProcess = origPil.crop(cropRect)
        else:
            imgToProcess = origPil

        tmpPath = os.path.join(tempfile.gettempdir(), "ronchi_input.png")
        imgToProcess.save(tmpPath)

        def progressCb(val):
            root.after(0, updateBar, val)

        startTime = time.time()
        res = processImg(tmpPath, progressCb=progressCb, nZernike=nZernikeRun,
                         surface=surfaceRun, fringeEditor=fringeEditorBridge,
                         pupilFromCrop=bool(cropRect))
        elapsed = time.time() - startTime
        print(f"Processing time: {elapsed:.2f} seconds")
        root.after(0, onFinished, res, elapsed)

    except Exception as ex:
        # 'ex' is deleted when the except block ends, so capture the message now
        errMsg = str(ex)
        root.after(0, onFailed, errMsg)


def readSurface():
    # Surface parameters typed by the user; validated with the ideal ray mapping
    vals = {}
    for key, entry in surfaceEntries.items():
        try:
            vals[key] = float(entry.get().replace(",", "."))
        except ValueError:
            raise RuntimeError(f"{SURFACE_LABELS[key]} must be a number.")
    surfaceMapping(vals["R"], vals["f"], vals["D"], vals["K"], vals["z"])
    return vals


def startProcess():
    global running, nZernikeRun, surfaceRun
    if imgPath is None or running:
        return
    # Zernike degree chosen by the user
    try:
        nZernikeRun = int(spinZernike.get())
        if not 2 <= nZernikeRun <= 100:
            raise ValueError
    except ValueError:
        messagebox.showerror("Zernike degree", "The Zernike degree n must be an integer "
                                               "between 2 and 100.")
        return
    # Optical surface parameters
    try:
        surfaceRun = readSurface()
    except RuntimeError as ex:
        messagebox.showerror("Optical surface", str(ex))
        return
    # Confirm the crop before running
    if cropRect:
        x0, y0, x1, y1 = cropRect
        ok = messagebox.askyesno(
            "Confirm crop",
            f"Is the crop correct?\n\nThe dashed circle is used as the exit pupil: its "
            f"center and radius must match the pupil of the Ronchigram.\n\n"
            f"Position: ({x0}, {y0}) – ({x1}, {y1})\n"
            f"Size: {x1 - x0} × {y1 - y0} px (pupil radius {(x1 - x0) / 2:g} px)\n\n"
            "Choose “No” to keep adjusting it.")
    else:
        ok = messagebox.askyesno("No crop", "No crop has been selected.\n"
                                            "Process the full image?")
    if not ok:
        lblFooter.config(text="Adjust the crop and press “Run pipeline” again.")
        return
    running = True
    resetStatus()
    btnProcess.config(state="disabled", bg="#9E9E9E", text="Running…")
    updateBar(0)
    threading.Thread(target=runProcessing, daemon=True).start()


def onFinished(res, elapsed):
    global running
    running = False
    btnProcess.config(state="normal", bg=BLUE_BTN, text="Run pipeline")
    for s, _ in STAGES:
        setStageState(s, "done")
    lblStatus.config(text=f"Pipeline finished · {len(res)} steps · {elapsed:.1f} s", fg=GREEN)
    showResults(res)


def onFailed(msg):
    global running
    running = False
    btnProcess.config(state="normal", bg=BLUE_BTN, text="Run pipeline")
    stage = None
    for s, _ in STAGES:
        c = statusCircles[s]
        if c.itemcget("dot", "fill") == GOLD:
            stage = s
    if stage:
        setStageState(stage, "error")
        lblStatus.config(text=f"{stageStatusName(stage)} failed", fg=RED)
    else:
        lblStatus.config(text="Pipeline failed", fg=RED)
    if "cancelled by the user" in msg:
        lblStatus.config(text="Pipeline cancelled", fg=MUTED)
        lblFooter.config(text="Pipeline cancelled in the fringe review. Adjust and run it again.")
        return
    lblFooter.config(text=f"Error: {msg}")
    messagebox.showerror("Error", msg)


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS VIEW
# ══════════════════════════════════════════════════════════════════════════════
def stepLabel(i):
    title = results[i].get("title", f"Step {i + 1}")
    for suffix in (" - Vertical", " - Horizontal"):
        title = title.replace(suffix, "")
    return f"{i + 1}. {title}"


def showResults(res):
    global results, stageSteps, curStage
    results = res
    stageSteps = {}
    for i, r in enumerate(results):
        stageSteps.setdefault(r.get("stage", "A"), []).append(i)
    curStage = None
    stageTabs.setItems([name for s, name in STAGES if s in stageSteps])
    showView(viewResults)
    root.after(50, lambda: showStep(0))


def onStageTab(k):
    present = [s for s, _ in STAGES if s in stageSteps]
    showStep(stageSteps[present[k]][0])


def onStepTab(k):
    showStep(stageSteps[curStage][k])


def showStep(i):
    global slideIdx, curStage
    if not results:
        return
    slideIdx = max(0, min(len(results) - 1, i))
    stage = results[slideIdx].get("stage", "A")
    present = [s for s, _ in STAGES if s in stageSteps]
    if stage != curStage:
        curStage = stage
        stageTabs.select(present.index(stage))
        stepTabs.setItems([stepLabel(j) for j in stageSteps[stage]])
    stepTabs.select(stageSteps[stage].index(slideIdx))
    renderSlide()


def navigate(delta):
    if results and viewResults.winfo_ismapped():
        showStep(slideIdx + delta)


def scheduleRender(_=None):
    global renderJob
    if not results:
        return
    if renderJob:
        root.after_cancel(renderJob)
    renderJob = root.after(120, renderSlide)


def tableToImage(df):
    # Renders a summary table as an image using matplotlib
    def fmt(v):
        if isinstance(v, (int, np.integer)):
            return str(v)
        if isinstance(v, (float, np.floating)):
            return str(int(v)) if float(v).is_integer() else f"{v:.4f}"
        return str(v)

    nRows = len(df)
    fig, ax = plt.subplots(figsize=(9, 0.5 + 0.42 * (nRows + 1)), dpi=100)
    ax.axis("off")
    # The index goes in the first column so it is never cut off
    tbl = ax.table(
        cellText=[[str(idx)] + [fmt(v) for v in row] for idx, row in zip(df.index, df.values)],
        colLabels=[str(df.index.name or "")] + [str(c) for c in df.columns],
        cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER)
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", weight="bold")
        elif c == 0:
            cell.set_facecolor("#EEF3F9")
            cell.set_text_props(weight="bold")
    ax.set_position([0, 0, 1, 1])
    return figToArray(fig)


def renderSlide():
    global slidePhoto, renderJob
    renderJob = None
    if not results:
        return
    result = results[slideIdx]

    lblStepTitle.config(text=result.get("title", ""))
    lblCounter.config(text=f"Step {slideIdx + 1} / {len(results)}")
    lblFooter.config(text=f"{STAGE_NAME.get(result.get('stage', 'A'), '')}  ·  "
                          f"{result.get('title', '')}")
    hasFile = ("excel" in result or "wavefrontExport" in result
               or isinstance(result.get("image"), np.ndarray))
    if "wavefrontExport" in result:
        label = "⬇  Download Zernike coefficients + W_F"
    elif "excel" in result:
        label = "⬇  Download Excel" + (" + matrices" if "arrays" in result else "")
    else:
        label = "⬇  Download image"
    btnDownload.config(text=label,
                       state="normal" if hasFile else "disabled",
                       bg=GREEN_BTN if hasFile else "#9E9E9E")

    canvasSlide.update_idletasks()
    cw = max(50, canvasSlide.winfo_width() - 20)
    ch = max(50, canvasSlide.winfo_height() - 20)
    canvasSlide.delete("all")

    imgNp = None
    if "table" in result:
        imgNp = tableToImage(result["table"])
    elif isinstance(result.get("image"), np.ndarray):
        imgNp = result["image"]

    if imgNp is not None:
        imgPil = (Image.fromarray(imgNp, mode="L").convert("RGB") if imgNp.ndim == 2
                  else Image.fromarray(imgNp))
        iw, ih = imgPil.size
        scale = min(cw / iw, ch / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        slidePhoto = ImageTk.PhotoImage(imgPil.resize((nw, nh), Image.LANCZOS))
        canvasSlide.create_image(cw // 2 + 10, ch // 2 + 10, anchor="center", image=slidePhoto)
    else:
        canvasSlide.create_text(cw // 2 + 10, ch // 2 + 10, text="No image for this step",
                                fill="#9AA5B1", font=(FONT, 12))

    # Text panel
    for widget in frameTextContent.winfo_children():
        widget.destroy()
    lines = result.get("text", [])
    if lines:
        for line in lines:
            tk.Label(frameTextContent, text=str(line), bg=WHITE, fg="#333333",
                     font=(FONT, 10), anchor="w", justify="left",
                     wraplength=S(270)).pack(fill="x", pady=2)
    else:
        tk.Label(frameTextContent, text="No additional info.", bg=WHITE, fg="#9AA5B1",
                 font=(FONT, 10, "italic")).pack(anchor="w")
    canvasText.yview_moveto(0)


def exportWavefront(result):
    # Saves the Zernike coefficients (OSA/ANSI order) and W_F. W_F is computed here,
    # in the background, because it has the full size of the crop.
    global exportRunning
    if exportRunning:
        return
    exp = result["wavefrontExport"]
    wf, geom = exp["wf"], exp["geom"]
    nZ = wf["nZernike"]
    path = filedialog.asksaveasfilename(
        title="Save Zernike coefficients and W_F",
        defaultextension=".mat", initialfile=f"Wavefront_n{nZ}.mat",
        filetypes=[("MATLAB file", "*.mat")])
    if not path:
        return
    base = os.path.splitext(path)[0]

    exportRunning = True
    btnDownload.config(state="disabled", bg="#9E9E9E")
    h, w = geom["cropShape"]
    lblFooter.config(text=f"Interpolating W_r to {h} × {w} and fitting Zernike n = {nZ}… 0%")

    def work():
        global exportRunning
        try:
            def progress(f):
                root.after(0, lambda: lblFooter.config(
                    text=f"Interpolating W_r to {h} × {w} and fitting Zernike n = {nZ}… {int(100 * f)}%"))
            out = computeWF(wf, geom, progress=progress)
            WF = out["WF"]
            # Zernike coefficients of the fit on the large W_r (OSA/ANSI order:
            # j = (n(n+2) + m)/2), the ones that generate W_F
            zdf = pd.DataFrame({"j (OSA/ANSI)": range(len(wf["idx"])),
                                "n": [n for n, m in wf["idx"]],
                                "m": [m for n, m in wf["idx"]],
                                "coefficient": out["coefs"]})
            zdf.to_excel(base + "_Zernike_OSA.xlsx", index=False)
            zdf.to_csv(base + "_Zernike_OSA.csv", index=False)
            savemat(base + "_W_F.mat", {
                "W_F": WF, "W_F_without_piston_tilt_defocus": out["WFc"],
                "W_r_interpolated": out["Wr"],
                "fit_points": out["nPoints"], "fit_rms": out["fitRms"],
                "zernike_coefficients": np.asarray(out["coefs"]),
                "zernike_n": np.array([n for n, m in wf["idx"]]),
                "zernike_m": np.array([m for n, m in wf["idx"]]),
                "zernike_degree": nZ,
                "units": "micrometers (pupil radius normalized to 1)",
                "R_mm": wf["surface"]["R"], "f_mm": wf["surface"]["f"],
                "D_mm": wf["surface"]["D"], "K": wf["surface"]["K"],
                "z_mm": wf["surface"]["z"]})
            np.savetxt(base + "_W_F.csv", WF, delimiter=",", fmt="%.10g")
            files = [base + "_Zernike_OSA.xlsx", base + "_Zernike_OSA.csv",
                     base + "_W_F.mat", base + "_W_F.csv"]
            root.after(0, lambda: messagebox.showinfo(
                "Saved", f"W_F: {h} × {w} (NaN outside the pupil)\nZernike fit on {out['nPoints']} pixels of W_r\n\nSaved to:\n"
                         + "\n".join(files)))
            root.after(0, lambda: lblFooter.config(text="W_F and Zernike coefficients saved."))
        except Exception as ex:
            msg = str(ex)
            root.after(0, lambda: messagebox.showerror("Error", msg))
        finally:
            exportRunning = False
            root.after(0, lambda: btnDownload.config(state="normal", bg=GREEN_BTN))

    threading.Thread(target=work, daemon=True).start()


def downloadCurrent():
    if not results:
        return
    result = results[slideIdx]
    if "wavefrontExport" in result:
        exportWavefront(result)
        return

    # If the step has an Excel dataframe (or several), save it as .xlsx
    if "excel" in result:
        title = result.get("title", f"step_{slideIdx + 1}")
        safeName = "".join(c if c.isalnum() or c in " -" else "_" for c in title)
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            initialfile=f"{safeName}.xlsx",
            filetypes=[("Excel", "*.xlsx")]
        )
        if not path:
            return
        data = result["excel"]
        if isinstance(data, dict):
            # One sheet per dataframe
            with pd.ExcelWriter(path) as writer:
                for sheet, df in data.items():
                    df.to_excel(writer, sheet_name=sheet[:31], index=False)
        else:
            data.to_excel(path, index=True)
        saved = [path]

        # Final matrices (float64, NaN = no data) saved next to the Excel file
        if "arrays" in result:
            npzPath = os.path.splitext(path)[0] + ".npz"
            np.savez(npzPath, **result["arrays"])
            saved.append(npzPath)

        messagebox.showinfo("Saved", "Saved to:\n" + "\n".join(saved))
        return

    # Otherwise save as image
    if "image" not in result or not isinstance(result["image"], np.ndarray):
        messagebox.showinfo("Download", "No file to download for this step.")
        return

    title = result.get("title", f"step_{slideIdx + 1}")
    safeName = "".join(c if c.isalnum() or c in " -" else "_" for c in title)
    path = filedialog.asksaveasfilename(
        defaultextension=".png",
        initialfile=f"{safeName}.png",
        filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
    )
    if not path:
        return

    imgNp = result["image"]
    if imgNp.ndim == 2:
        cv2.imwrite(path, imgNp)
    else:
        cv2.imwrite(path, cv2.cvtColor(imgNp, cv2.COLOR_RGB2BGR))
    messagebox.showinfo("Saved", f"Image saved to:\n{path}")


def goBack():
    if running:
        return
    showView(viewInput)
    lblFooter.config(text="Select another image or adjust the crop and run the pipeline again.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════
# Sharp rendering on Windows high-resolution screens: without this, Windows
# enlarges the window as a bitmap and everything looks blurry (also in the .exe)
try:
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

root = tk.Tk()
root.title(f"{APP_TITLE} - Scientific Pipeline")
root.configure(bg=BG)
UI_SCALE = max(1.0, root.winfo_fpixels("1i") / 96.0)
LOGO_SIZE = S(LOGO_SIZE)
BARH = S(BARH)
scrW, scrH = root.winfo_screenwidth(), root.winfo_screenheight()
winW, winH = min(S(1320), int(scrW * 0.95)), min(S(900), int(scrH * 0.9))
root.geometry(f"{winW}x{winH}")
root.minsize(min(S(1100), winW), min(S(760), winH))
if os.name == "nt":
    try:
        root.state("zoomed")
    except Exception:
        pass

# ── Menu ──
menubar = tk.Menu(root)
menuFile = tk.Menu(menubar, tearoff=0)
menuFile.add_command(label="Open image…", command=selectImage)
menuFile.add_command(label="Back to input", command=goBack)
menuFile.add_command(label="Save current output…", command=downloadCurrent)
menuFile.add_separator()
menuFile.add_command(label="Exit", command=root.destroy)
menubar.add_cascade(label="File", menu=menuFile)
root.config(menu=menubar)

# ── Header: logo | title | logo ──
frameHeader = tk.Frame(root, bg=NAVY, padx=8, pady=8)
frameHeader.pack(fill="x", padx=12, pady=(10, 0))

def logoBox(side, padx=0):
    # Square white box for one logo
    box = tk.Frame(frameHeader, bg=WHITE, width=LOGO_SIZE, height=LOGO_SIZE)
    box.pack(side=side, padx=padx)
    box.pack_propagate(False)
    lbl = tk.Label(box, bg=WHITE)
    lbl.pack(expand=True)
    return lbl

lblLogoLeft  = logoBox("left")
lblLogoLeft2 = logoBox("left", padx=(8, 0))
lblLogoRight = logoBox("right")

frameTitle = tk.Frame(frameHeader, bg=NAVY)
frameTitle.pack(side="left", fill="both", expand=True, padx=8)
tk.Label(frameTitle, text=APP_TITLE, bg=NAVY_DARK, fg=WHITE,
         font=(FONT, 22, "bold"), highlightthickness=2,
         highlightbackground=WHITE).pack(fill="both", expand=True, pady=(0, 4))
tk.Label(frameTitle, text=APP_SUBTITLE, bg=NAVY_DARK, fg="#C9D9F0",
         font=(FONT, 10), highlightthickness=2,
         highlightbackground=WHITE).pack(fill="x")

loadLogo(lblLogoLeft,  LOGO_LEFT_FILE,  LOGO_SIZE - 8, LOGO_SIZE - 8, "Logo 1")
loadLogo(lblLogoLeft2, LOGO_LEFT2_FILE, LOGO_SIZE - 8, LOGO_SIZE - 8, "Logo 2")
loadLogo(lblLogoRight, LOGO_RIGHT_FILE, LOGO_SIZE - 8, LOGO_SIZE - 8, "Logo 3")

tk.Frame(root, bg=GOLD, height=4).pack(fill="x", padx=12)

# ── Pipeline status ──
frameStatus = tk.Frame(root, bg=BG)
frameStatus.pack(fill="x", padx=14, pady=(10, 6))
tk.Label(frameStatus, text="Pipeline status", bg=BG, fg=NAVY,
         font=(FONT, 11, "bold")).pack(side="left", padx=(0, 10))
statusCircles = {}
for s, name in STAGES:
    d = S(36)
    c = tk.Canvas(frameStatus, width=d, height=d, bg=BG, highlightthickness=0,
                  cursor="hand2")
    c.create_oval(2, 2, d - 2, d - 2, fill=PURPLE, outline=PURPLE, tags="dot")
    c.create_text(d // 2, d // 2, text=str(STAGE_NUMBER[s]), fill=WHITE,
                  font=(FONT, 11, "bold"), tags="txt")
    c.pack(side="left", padx=4)
    c.bind("<Button-1>", lambda e, s=s: onStatusClick(s))
    statusCircles[s] = c
lblStatus = tk.Label(frameStatus, text="Waiting for an image", bg=BG, fg=MUTED,
                     font=(FONT, 10))
lblStatus.pack(side="right")

# ── Footer (status line) ──
lblFooter = tk.Label(root, text="Select a Ronchigram image to start (File ▸ Open image…).",
                     bg=BG, fg=MUTED, font=(FONT, 9), anchor="w")
lblFooter.pack(side="bottom", fill="x", padx=16, pady=(0, 2))
tk.Frame(root, bg=NAVY, height=6).pack(side="bottom", fill="x")

# ── Body ──
frameBody = tk.Frame(root, bg=BG)
frameBody.pack(fill="both", expand=True, padx=12, pady=(0, 6))

# ══ INPUT VIEW ══
viewInput = tk.Frame(frameBody, bg=BG)

cardPrev = card(viewInput, "Input Ronchigram")
cardPrev.pack(side="left", fill="both", expand=True, padx=(0, 10))
canvasPrev = tk.Canvas(cardPrev, width=S(560), height=S(460), bg="#F7F9FC",
                       highlightthickness=0)
canvasPrev.pack(fill="both", expand=True, padx=14, pady=(0, 14))
canvasPrev.bind("<Configure>", schedulePreview)
canvasPrev.bind("<ButtonPress-1>", onCropPress)
canvasPrev.bind("<B1-Motion>", onCropDrag)
canvasPrev.bind("<ButtonRelease-1>", onCropRelease)
canvasPrev.bind("<Motion>", onCropMotion)
canvasPrev.bind("<MouseWheel>", onCropWheel)
canvasPrev.bind("<Button-4>", onCropWheel)
canvasPrev.bind("<Button-5>", onCropWheel)

cardCtrl = card(viewInput, "Input and crop")
cardCtrl.pack(side="right", fill="y")
ctrl = tk.Frame(cardCtrl, bg=WHITE, width=S(380))
ctrl.pack(fill="both", expand=True, padx=14, pady=(4, 14))

makeButton(ctrl, "Select image…", selectImage, NAVY, size=11,
           padx=10, pady=7).pack(fill="x", pady=(4, 8))
rowCrop = tk.Frame(ctrl, bg=WHITE)
rowCrop.pack(fill="x", pady=(0, 8))
btnCrop = makeButton(rowCrop, "✂  New crop", toggleCrop, "#44546A", size=10,
                     padx=10, pady=6)
btnCrop.pack(side="left", fill="x", expand=True, padx=(0, 6))
makeButton(rowCrop, "Clear crop", clearCrop, "#DDE3EB", fg=TEXT, size=10, bold=False,
           padx=10, pady=6).pack(side="left", fill="x", expand=True)

tk.Label(ctrl, text="Drag to draw a square and fit its dashed circle to the exit pupil "
                    "(it is used as the pupil). Drag inside to move it, drag a corner to "
                    "resize it, mouse wheel to zoom, arrow keys to nudge (Shift = 10 px).",
         bg=WHITE, fg=MUTED, font=(FONT, 9, "italic"), wraplength=S(360),
         justify="left").pack(fill="x", pady=(0, 8))

rowInfo = tk.Frame(ctrl, bg=WHITE)
rowInfo.pack(fill="x", pady=(0, 8))
info = tk.Frame(rowInfo, bg=WHITE)
info.pack(side="left", fill="both", expand=True, anchor="n")
def infoRow(r, label):
    tk.Label(info, text=label, bg=WHITE, fg=MUTED, font=(FONT, 10), anchor="w"
             ).grid(row=r, column=0, sticky="w", pady=1)
    v = tk.Label(info, text="—", bg=WHITE, fg=TEXT, font=(FONT, 10), anchor="w",
                 justify="left", wraplength=S(190))
    v.grid(row=r, column=1, sticky="w", padx=(10, 0), pady=1)
    return v
lblFile = infoRow(0, "File:")
lblSize = infoRow(1, "Size:")
lblCrop = infoRow(2, "Crop:")

canvasCropPrev = tk.Canvas(rowInfo, width=S(116), height=S(116), bg="#F7F9FC",
                           highlightthickness=1, highlightbackground=BORDER)
canvasCropPrev.pack(side="right", anchor="n", padx=(8, 0))

# Optical surface under test (build the ideal Ronchigram and the scale of W)
tk.Label(ctrl, text="Optical surface", bg=WHITE, fg=NAVY, font=(FONT, 10, "bold"),
         anchor="w").pack(fill="x")
SURFACE_LABELS = {"R": "Radius of curvature R (mm)", "f": "Focal length f (mm)",
                  "D": "Diameter D (mm)", "K": "Conic constant K",
                  "z": "Distance z (mm)"}
SURFACE_SHORT = {"R": "R (mm)", "f": "f (mm)", "D": "D (mm)", "K": "Conic K",
                 "z": "z (mm)"}
gridSurf = tk.Frame(ctrl, bg=WHITE)
gridSurf.pack(fill="x", pady=(4, 2))
surfaceEntries = {}
for idx_, key in enumerate(("R", "f", "D", "K", "z")):
    r_, c_ = divmod(idx_, 2)
    tk.Label(gridSurf, text=SURFACE_SHORT[key], bg=WHITE, fg=TEXT, font=(FONT, 10),
             anchor="w").grid(row=r_, column=2 * c_, sticky="w", pady=2,
                              padx=(0 if c_ == 0 else 14, 6))
    e_ = tk.Entry(gridSurf, width=9, font=(FONT, 10), justify="center")
    e_.insert(0, f"{SURFACE_DEFAULTS[key]:g}")
    e_.grid(row=r_, column=2 * c_ + 1, sticky="w", pady=2)
    surfaceEntries[key] = e_
tk.Label(ctrl, text="z: distance from the surface vertex to the plane where the "
                    "Ronchigram was recorded. K: 0 sphere, -1 paraboloid.",
         bg=WHITE, fg=MUTED, font=(FONT, 9, "italic"), wraplength=S(360),
         justify="left").pack(fill="x", pady=(0, 8))

rowZ = tk.Frame(ctrl, bg=WHITE)
rowZ.pack(fill="x", pady=(0, 10))
tk.Label(rowZ, text="Zernike degree n:", bg=WHITE, fg=TEXT,
         font=(FONT, 10)).pack(side="left")
spinZernike = tk.Spinbox(rowZ, from_=2, to=100, width=5, font=(FONT, 10),
                         justify="center")
spinZernike.delete(0, "end")
spinZernike.insert(0, str(ZERNIKE_Q))
spinZernike.pack(side="left", padx=(8, 0))
tk.Label(rowZ, text="(article: 50)", bg=WHITE, fg=MUTED,
         font=(FONT, 9, "italic")).pack(side="left", padx=(6, 0))

btnProcess = makeButton(ctrl, "Run pipeline", startProcess, "#9E9E9E", size=13,
                        pady=10, state="disabled")
btnProcess.pack(fill="x", pady=(0, 10))

canvasBar = tk.Canvas(ctrl, height=BARH, bg="#DDE3EB", highlightthickness=0)
canvasBar.pack(fill="x")
barFill = canvasBar.create_rectangle(0, 0, 0, BARH, fill=BLUE_BTN, outline="")
lblPercent = tk.Label(ctrl, text="0%", bg=WHITE, fg=MUTED, font=(FONT, 10))
lblPercent.pack(anchor="e", pady=(4, 0))

# ══ FRINGE REVIEW VIEW ══
viewFringes = tk.Frame(frameBody, bg=BG)
cardFe = card(viewFringes, "Fringe review - vertical Ronchigram (aligned image)")
cardFe.pack(side="left", fill="both", expand=True, padx=(0, 10))
feCanvas = tk.Canvas(cardFe, bg="#F7F9FC", highlightthickness=0, cursor="crosshair")
feCanvas.pack(fill="both", expand=True, padx=12, pady=(0, 12))
feCanvas.bind("<Configure>", feRender)
feCanvas.bind("<ButtonPress-1>", feOnPress)
feCanvas.bind("<B1-Motion>", feOnDrag)
feCanvas.bind("<ButtonRelease-1>", feOnRelease)
feCanvas.bind("<Button-3>", feOnDelete)
feCanvas.bind("<Control-Button-1>", feOnDelete)

cardFeCtrl = card(viewFringes, "Fringes")
cardFeCtrl.pack(side="right", fill="y")
feCtrl = tk.Frame(cardFeCtrl, bg=WHITE, width=S(330))
feCtrl.pack(fill="both", expand=True, padx=14, pady=(4, 14))
tk.Label(feCtrl, text="Select a fringe and click on the image to add anchor points. Drag a "
                      "point to move it; right-click (or Ctrl + click) to delete it. The points "
                      "stay where you put them; the line of each fringe is fitted to its "
                      "anchor points after you confirm.",
         bg=WHITE, fg=MUTED, font=(FONT, 9, "italic"), wraplength=S(310),
         justify="left").pack(fill="x", pady=(0, 8))
# The action buttons are packed first at the bottom so they are always visible; the
# list of fringes takes the remaining space above them
feBottom = tk.Frame(feCtrl, bg=WHITE)
feBottom.pack(side="bottom", fill="x")
feList = tk.Frame(feCtrl, bg=WHITE)
feList.pack(side="top", fill="x", pady=(0, 6))
rowFe1 = tk.Frame(feBottom, bg=WHITE)
rowFe1.pack(fill="x", pady=(0, 5))
makeButton(rowFe1, "+  New fringe", feNewFringe, NAVY, size=10, padx=8, pady=4
           ).pack(side="left", fill="x", expand=True, padx=(0, 6))
makeButton(rowFe1, "Delete fringe", feDeleteFringe, "#A94442", size=10, padx=8, pady=4
           ).pack(side="left", fill="x", expand=True)
rowFe2 = tk.Frame(feBottom, bg=WHITE)
rowFe2.pack(fill="x", pady=(0, 5))
makeButton(rowFe2, "Clear points", feClearPoints, "#DDE3EB", fg=TEXT, size=10, bold=False,
           padx=8, pady=4).pack(side="left", fill="x", expand=True, padx=(0, 6))
makeButton(rowFe2, "Delete all", feDeleteAll, "#DDE3EB", fg=TEXT, size=10, bold=False,
           padx=8, pady=4).pack(side="left", fill="x", expand=True)
makeButton(feBottom, "Reset to automatic detection", feReset, "#DDE3EB", fg=TEXT, size=10,
           bold=False, padx=8, pady=4).pack(fill="x", pady=(0, 6))
feInfo = tk.Label(feBottom, text="", bg=WHITE, fg=TEXT, font=(FONT, 9), wraplength=S(310),
                  justify="left", anchor="w")
feInfo.pack(fill="x", pady=(0, 6))
makeButton(feBottom, "✓  Confirm fringes and continue", feConfirm, GREEN_BTN, size=12,
           pady=8).pack(fill="x", pady=(0, 5))
makeButton(feBottom, "Cancel", feCancel, "#9E9E9E", size=10, pady=4).pack(fill="x")

# ══ RESULTS VIEW ══
viewResults = tk.Frame(frameBody, bg=BG)

rowStages = tk.Frame(viewResults, bg=BG)
rowStages.pack(fill="x", pady=(0, 8))
makeButton(rowStages, "↩  Input", goBack, NAVY, size=10, padx=12,
           pady=8).pack(side="left", padx=(0, 8), fill="y")
stageTabs = TabStrip(rowStages, onStageTab, fontSize=10, pady=8)
stageTabs.pack(side="left", fill="x", expand=True)

cardResults = tk.Frame(viewResults, bg=WHITE, highlightthickness=1,
                       highlightbackground=BORDER)
cardResults.pack(fill="both", expand=True)
stepTabs = TabStrip(cardResults, onStepTab, fontSize=9, pady=6)
stepTabs.pack(fill="x", padx=10, pady=(10, 6))

frameDisplay = tk.Frame(cardResults, bg=WHITE)
frameDisplay.pack(fill="both", expand=True, padx=10)

frameInfo = tk.Frame(frameDisplay, bg=WHITE, width=S(300), highlightthickness=1,
                     highlightbackground=BORDER)
frameInfo.pack(side="right", fill="y", pady=(0, 6))
frameInfo.pack_propagate(False)
tk.Label(frameInfo, text="Step details", bg=WHITE, fg=NAVY, font=(FONT, 11, "bold"),
         anchor="w").pack(fill="x", padx=12, pady=(10, 4))
canvasText = tk.Canvas(frameInfo, bg=WHITE, highlightthickness=0)
scrollbar = tk.Scrollbar(frameInfo, orient="vertical", command=canvasText.yview)
frameTextContent = tk.Frame(canvasText, bg=WHITE)
frameTextContent.bind("<Configure>",
                      lambda e: canvasText.configure(scrollregion=canvasText.bbox("all")))
canvasText.create_window((0, 0), window=frameTextContent, anchor="nw")
canvasText.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side="right", fill="y")
canvasText.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=(0, 10))

canvasSlide = tk.Canvas(frameDisplay, bg=WHITE, highlightthickness=1,
                        highlightbackground=BORDER)
canvasSlide.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=(0, 6))
canvasSlide.bind("<Configure>", scheduleRender)

rowBottom = tk.Frame(cardResults, bg=WHITE)
rowBottom.pack(fill="x", padx=10, pady=(4, 10))
btnDownload = makeButton(rowBottom, "⬇  Download", downloadCurrent, GREEN_BTN, size=10,
                         padx=12, pady=6)
btnDownload.pack(side="left")
lblStepTitle = tk.Label(rowBottom, text="", bg=WHITE, fg=TEXT, font=(FONT, 11, "bold"),
                        anchor="w")
lblStepTitle.pack(side="left", padx=12)
makeButton(rowBottom, "›", lambda: navigate(1), NAVY, size=13, padx=12,
           pady=2).pack(side="right")
makeButton(rowBottom, "‹", lambda: navigate(-1), NAVY, size=13, padx=12,
           pady=2).pack(side="right", padx=6)
lblCounter = tk.Label(rowBottom, text="", bg=WHITE, fg=MUTED, font=(FONT, 10))
lblCounter.pack(side="right", padx=10)

# Keyboard navigation between steps
root.bind("<Left>",  lambda e: onArrow(-1, 0, e))
root.bind("<Right>", lambda e: onArrow(1, 0, e))
root.bind("<Up>",    lambda e: onArrow(0, -1, e))
root.bind("<Down>",  lambda e: onArrow(0, 1, e))

# ══════════════════════════════════════════════════════════════
# Start on the input view
# ══════════════════════════════════════════════════════════════
showView(viewInput)
root.update_idletasks()
buildPreview()
root.mainloop()