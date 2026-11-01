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
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
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
# REFERENCES used in the comments of this code (in the texts shown by the GUI they are
# written as "manuscript" and "Supplemental Document")
# REFERENCIAS usadas en los comentarios de este código (en los textos que muestra la GUI
# se escriben como "manuscript" y "Supplemental Document")
#   [M]     J. Arriaga Hernández, B. Cuevas Otahola, A. Jaramillo Núñez, F. Sánchez-Mora,
#           D. Gale, E. Benítez, A. Pozos Rojas, J. C. Cuevas-Tello, "Autonomous
#           Wavefront Reconstruction from SLM-Based Ronchigrams Using a Regularized
#           Variational Model", manuscript (2026).
#   [S]     Same authors, "Autonomous Wavefront Reconstruction from SLM-Based
#           Ronchigrams Using a Regularized Variational Model: Supplemental Document"
#           (2026). Its equations are numbered S1, S2, ...
#   [APB]   J. Arriaga Hernández, B. Cuevas Otahola, A. Jaramillo Núñez,
#           J. Oliveros Oliveros, M. Morín Castillo, "Gaussian quadrature to solve the
#           wavefront in simultaneous Hartmann/Bi-Ronchi tests with SLM using directional
#           derivative", Appl. Phys. B 132, 14 (2026). doi:10.1007/s00340-025-08611-y
#   [Optik] J. Arriaga Hernández, B. Cuevas Otahola, A. Diaz Nayotl, A. Jaramillo Núñez,
#           A. Pérez Villegas, O. Valenzuela, "Machine learning K-means algorithm applied
#           to wavefront sensing in Bi-Ronchi/Hartmann tests with SLM", Optik 346,
#           172607 (2026). doi:10.1016/j.ijleo.2025.172607
#   [OST]   A. Cornejo-Rodríguez, "Ronchi Test", Ch. 9 in D. Malacara (ed.), Optical Shop
#           Testing, 3rd ed., John Wiley & Sons (2007).
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# PARÁMETROS
# ══════════════════════════════════════════════════════════════════════════════
# Vertical alignment verification
# Verificación de la alineación vertical
ANGLE_THRESHOLD   = 0.3    # max residual inclination accepted (degrees)
                           # inclinación residual máxima aceptada (grados)
ALIGN_MAX_ITER    = 3      # max number of correction iterations
                           # número máximo de iteraciones de corrección
ALIGN_N_CENTRAL   = 3      # number of central stripes used to check verticality
                           # número de franjas centrales usadas para revisar la verticalidad

# Stripe segmentation
# Segmentación de las franjas
BIN_METHOD         = "adaptive"  # "adaptive": local threshold (robust to uneven brightness)
                                 # "fixed": global threshold BIN_THRESHOLD
                                 # "adaptive": umbral local (robusto a brillo no uniforme)
                                 # "fixed": umbral global BIN_THRESHOLD
BIN_THRESHOLD      = 195   # threshold used when BIN_METHOD = "fixed"
                           # umbral usado cuando BIN_METHOD = "fixed"
ADAPTIVE_C         = -20   # adaptive: a pixel is white if it exceeds its local mean by 20 levels
                           # (after the contrast of the fringe channel is stretched to 0-255)
                           # adaptive: un píxel es blanco si supera su media local en 20 niveles
                           # (después de estirar el contraste del canal de franjas a 0-255)
ADAPTIVE_BLOCK     = 0.25  # adaptive: local window size as a fraction of the image size
                           # adaptive: tamaño de la ventana local como fracción del tamaño de la imagen
MIN_COMPONENT_AREA = 300   # connected components smaller than this are discarded (px).
                           # Low on purpose: dim fringes are kept; false detections are
                           # removed by the user in the fringe review
                           # se descartan las componentes conectadas más chicas que esto (px).
                           # Bajo a propósito: se conservan las franjas tenues; las detecciones
                           # falsas las quita el usuario en la revisión de franjas
CLOSE_KERNEL       = 10    # size of the square kernel of the morphological closing (px)
                           # tamaño del kernel cuadrado del cierre morfológico (px)
SPURIOUS_GAP       = 0.6   # peaks closer than this fraction of the median gap are spurious
                           # los picos más cercanos que esta fracción de la separación mediana son espurios

# Centroid lines
# Líneas de centroides
N_ANCHORS   = 10           # anchor points per stripe used to fit its center curve
                           # puntos de anclaje por franja usados para ajustar su curva central
POLY_DEGREE = 2            # degree of the polynomial fitted to each stripe
                           # grado del polinomio ajustado a cada franja
MIN_STRIPE_PIXELS = 200    # white pixels needed in a stripe window to look for anchors
                           # píxeles blancos necesarios en la ventana de una franja para buscar anclajes
MIN_MANUAL_ANCHORS = 3     # anchor points needed per fringe in the fringe review
                           # puntos de anclaje necesarios por franja en la revisión de franjas
POINT_STEP  = 5            # distance (px) between consecutive points of a centroid line.
                           # Nyquist criterion: points per line = length / 5  (<= image size / 5)
                           # distancia (px) entre puntos consecutivos de una línea de centroides.
                           # Criterio de Nyquist: puntos por línea = longitud / 5  (<= tamaño de la imagen / 5)

# Point cloud refinement (multilinear interpolation, PCM)
# Refinamiento de la nube de puntos (interpolación multilineal, PCM)
REFINE_K1     = 5          # points inserted between neighbouring lines in the 1st refinement
                           # puntos insertados entre líneas vecinas en el 1er refinamiento
REFINE_K2     = 10         # points inserted between neighbouring lines in the 2nd refinement
                           # puntos insertados entre líneas vecinas en el 2o refinamiento
MIN_SPACING   = 1          # refined points are never closer than this (px).
                           # >= 1 so there are never more points than pixels (Optik article)
                           # los puntos refinados nunca quedan más cerca que esto (px).
                           # >= 1 para que nunca haya más puntos que píxeles (artículo de Optik)
INTERP_METHOD = "linear"   # vertical/horizontal: "linear" = multilinear (Delaunay),
                           #                      "cubic" = spline (Clough-Tocher)
                           # vertical/horizontal: "linear" = multilineal (Delaunay),
                           #                      "cubic" = spline (Clough-Tocher)
GRID_INTERP_METHOD = "cubic"  # grid: dx and dy interpolated separately with this method
                              # malla: dx y dy se interpolan por separado con este método

# Extension of the point clouds to the whole exit pupil
# Extensión de las nubes de puntos a toda la pupila de salida
FULL_PUPIL_DEGREE = 4      # total degree of the polynomial used outside the data region
                           # (4 = degree of the Kingslake primary aberrations polynomial).
                           # High degrees (e.g. 11) diverge where there are no data.
                           # grado total del polinomio usado fuera de la región con datos
                           # (4 = grado del polinomio de aberraciones primarias de Kingslake).
                           # Los grados altos (p. ej. 11) divergen donde no hay datos.
FULL_PUPIL_DECAY  = 20     # px; the fit residual at the data border decays with this
                           # length, so the extension joins the data without a step
                           # px; el residuo del ajuste en el borde de los datos decae con
                           # esta longitud, así la extensión se une a los datos sin escalón

# Optical surface under test (user inputs in the GUI; defaults = lens L1 of the
# manuscript [M] Sec. 2: D = 39 mm, f = 237 mm, z = 173.5 mm; R assumes a plano-convex
# BK7 lens)
# Superficie óptica bajo prueba (datos del usuario en la GUI; valores por defecto = lente
# L1 del manuscrito [M] Sec. 2: D = 39 mm, f = 237 mm, z = 173.5 mm; R supone una lente
# plano-convexa de BK7)
SURFACE_DEFAULTS = {"R": 122.06,   # radius of curvature of the tested surface (mm)
                                   # radio de curvatura de la superficie bajo prueba (mm)
                    "f": 237.0,    # focal length (mm)
                                   # distancia focal (mm)
                    "D": 39.0,     # diameter of the aperture (mm)
                                   # diámetro de la apertura (mm)
                    "K": 0.0,      # conic constant (0 sphere, -1 paraboloid, ...)
                                   # constante cónica (0 esfera, -1 paraboloide, ...)
                    "z": 173.5}    # distance from the surface vertex to the plane where
                                   # the Ronchigram was recorded (camera plane, mm)
                                   # distancia del vértice de la superficie al plano donde
                                   # se registró el Ronchigrama (plano de la cámara, mm)

PUPIL_MARGIN = 1.08        # size of the working square relative to the pupil diameter
                           # tamaño del cuadrado de trabajo respecto al diámetro de la pupila

# ── Wavefront (stage H) ──
# The wavefront is reconstructed from the signed transverse aberrations following the
# manuscript "Autonomous Wavefront Reconstruction from SLM-Based Ronchigrams Using a
# Regularized Variational Model" [M] and its Supplemental Document [S]:
#   TA = Gamma * s * dx                    [M] Eq. 13, [S] Eq. S8; each value is
#                                          located at the nominal point ([M] Eq. 14)
#   p = kappa * TA_x,  q = kappa * TA_y    [M] Eq. 15, [S] Eq. S14
#   W_P: variational reconstruction        [M] Eqs. 19-26, [S] Eqs. S28-S45 (principal)
#   W_G: Gauss-Legendre path integration   [M] Eqs. 27-30, [S] Eqs. S60-S75 (complementary)
#   piston and tilt removal                [M] Eq. 31, [S] Eqs. S81-S82
# Values used by the GUI (the manuscript leaves them as calibration factors):
#   s     = physical size of a camera pixel at the plane z (pupil radius at z from the
#           ideal ray tracing of the surface / pupil radius in pixels), mm/px
#   Gamma = 1 (the TA is taken directly at the observation plane)
#   kappa = WAVEFRONT_SIGN / z, i.e. -1/z (Rayces relation, slope = -TA / z)
# W, W_P, W_G and the Zernike coefficients are given in micrometers; the pupil
# coordinates are normalized to the unit disk (physical pupil radius = D/2).
# ── Frente de onda (etapa H) ──
# El frente de onda se reconstruye a partir de las aberraciones transversales con signo,
# siguiendo el manuscrito "Autonomous Wavefront Reconstruction from SLM-Based
# Ronchigrams Using a Regularized Variational Model" [M] y su documento suplementario [S]:
#   TA = Gamma * s * dx                    [M] Ec. 13, [S] Ec. S8; cada valor se
#                                          ubica en el punto nominal ([M] Ec. 14)
#   p = kappa * TA_x,  q = kappa * TA_y    [M] Ec. 15, [S] Ec. S14
#   W_P: reconstrucción variacional        [M] Ecs. 19-26, [S] Ecs. S28-S45 (principal)
#   W_G: integración de Gauss-Legendre     [M] Ecs. 27-30, [S] Ecs. S60-S75 (complementaria)
#   quitar pistón e inclinación            [M] Ec. 31, [S] Ecs. S81-S82
# Valores que usa la GUI (el manuscrito los deja como factores de calibración):
#   s     = tamaño físico de un píxel de la cámara en el plano z (radio de la pupila en z
#           según el trazo ideal de rayos por la superficie / radio de la pupila en
#           píxeles), mm/px
#   Gamma = 1 (la TA se toma directamente en el plano de observación)
#   kappa = WAVEFRONT_SIGN / z, es decir -1/z (relación de Rayces, pendiente = -TA / z)
# W, W_P, W_G y los coeficientes de Zernike están en micrómetros; las coordenadas de la
# pupila están normalizadas al disco unitario (radio físico de la pupila = D/2).
WAVEFRONT_SOURCE = "vertical_horizontal"  # TA_x from the vertical matrix and TA_y from the
                                          # horizontal one; "grid": dx, dy of the grid matrix
                                          # TA_x de la matriz vertical y TA_y de la
                                          # horizontal; "grid": dx, dy de la matriz de la malla
TA_FIT_DEGREE  = 11        # continuous TA surfaces: products of Legendre polynomials
                           # P_m(x) P_n(y), m, n <= 11 ([M] Sec. 5B, [S] Eq. S18)
                           # superficies continuas de TA: productos de polinomios de Legendre
                           # P_m(x) P_n(y), m, n <= 11 ([M] Sec. 5B, [S] Ec. S18)
WAVEFRONT_GRID = 201       # nodes per side of the Cartesian pupil grid where p, q, W_P and
                           # W_G are computed (nodes with x^2 + y^2 <= 1 are used)
                           # nodos por lado de la malla cartesiana de la pupila donde se calculan
                           # p, q, W_P y W_G (se usan los nodos con x^2 + y^2 <= 1)
WP_OMEGA       = 1.0       # confidence weight omega of every node ([M] Eqs. 19-22). The
                           # manuscript gives no formula for omega; uniform confidence
                           # (omega = 1) is the case of [M] Eq. 21
                           # peso de confianza omega de cada nodo ([M] Ecs. 19-22). El
                           # manuscrito no da una fórmula para omega; confianza uniforme
                           # (omega = 1) es el caso de [M] Ec. 21
WP_MU_CANDIDATES = np.logspace(-9, -1, 17)  # regularization parameters mu tested in the
                                            # parámetros de regularización mu probados en la
                           # L-curve ([S] Eq. S50: mu_l = 10^(a + l*delta), half a decade)
                           # curva L ([S] Ec. S50: mu_l = 10^(a + l*delta), media década)
GL_NODES       = 12        # Gauss-Legendre nodes N_G along each straight path ([M] Eq. 30).
                           # The TA surfaces are polynomials of degree <= 22 along a line,
                           # so 12 nodes integrate them exactly (degree 2*12 - 1 = 23)
                           # nodos de Gauss-Legendre N_G en cada trayectoria recta ([M] Ec. 30).
                           # Las superficies de TA son polinomios de grado <= 22 sobre una
                           # recta, así que 12 nodos las integran exactamente (grado 2*12 - 1 = 23)
ZERNIKE_Q      = 50        # default Zernike degree n (n = 50 -> 1326 coefficients).
                           # The GUI lets the user change it before running.
                           # grado de Zernike n por defecto (n = 50 -> 1326 coeficientes).
                           # La GUI deja que el usuario lo cambie antes de correr.
WAVEFRONT_SIGN = -1        # sign of kappa (kappa = -1/z): convention checked with simulated
                           # Ronchigrams of known wavefront
                           # signo de kappa (kappa = -1/z): convención revisada con
                           # Ronchigramas simulados de frente de onda conocido
REMOVE_TERMS   = (0, 1, 2)  # OSA/ANSI indices removed for display: piston and tilts, i.e.
                            # the least-squares plane of [M] Eq. 31 (the defocus is kept)
                            # índices OSA/ANSI que se quitan para mostrar: pistón e inclinaciones,
                            # es decir, el plano de mínimos cuadrados de [M] Ec. 31 (el desenfoque se queda)
EXPORT_FIRST   = (59, 49)  # besides all the coefficients, the first 59 and the first 49
                           # (OSA/ANSI order) are also saved in separate files ([S] Sec. 5)
                           # además de todos los coeficientes, los primeros 59 y los primeros 49
                           # (orden OSA/ANSI) también se guardan en archivos aparte ([S] Sec. 5)
FIG_DPI        = 150       # resolution of the plots shown in the GUI
                           # resolución de las gráficas que muestra la GUI

# Fringe colors in the fringe review (same colors in the editor and in its result step)
# Colores de las franjas en la revisión de franjas (los mismos en el editor y en su paso de resultado)
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
# FUNCIONES GENERALES
# ══════════════════════════════════════════════════════════════════════════════
def figToArray(fig):
    # Render a matplotlib figure into an RGB numpy array and close it
    # Convierte una figura de matplotlib en un arreglo RGB de numpy y la cierra
    fig.set_dpi(FIG_DPI)
    fig.patch.set_facecolor("white")
    # Titles, axis labels, legends and table cells in the selected language
    # Títulos, ejes, leyendas y celdas de tablas en el idioma seleccionado
    for t in fig.findobj(matplotlib.text.Text):
        t.set_text(T(t.get_text()))
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
    # Máscara binaria (255 adentro) de la pupila de salida circular
    cx, cy, r = pupil
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2).astype(np.uint8) * 255


def insidePupil(x, y, pupil):
    cx, cy, r = pupil
    return (np.asarray(x) - cx) ** 2 + (np.asarray(y) - cy) ** 2 <= r ** 2


def drawCurve(img, poly, a0, a1, color, thickness, vertical=True):
    # vertical=True : x = poly(y), y in [a0, a1]
    # vertical=False: y = poly(x), x in [a0, a1]
    # vertical=True : x = poly(y), y en [a0, a1]
    # vertical=False: y = poly(x), x en [a0, a1]
    a = np.linspace(a0, a1, 400)
    b = poly(a)
    pts = np.stack([b, a], axis=1) if vertical else np.stack([a, b], axis=1)
    h, w = img.shape[:2]
    ok = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    pts = np.round(pts[ok]).astype(np.int32)
    if len(pts) > 1:
        cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness)


def stripeChannel(rgb):
    # Color channel that carries the fringes: the one with the
    # largest contrast (standard deviation), so the color of the fringes does not
    # matter. Its intensity is stretched to 0-255 (percentiles 1 and 99.7) so dim
    # Ronchigrams are processed like bright ones.
    # Canal de color que lleva las franjas: el de mayor contraste (desviación estándar),
    # así no importa el color de las franjas. Su intensidad se estira a 0-255
    # (percentiles 1 y 99.7) para que los Ronchigramas tenues se procesen igual que los
    # brillantes.
    k = int(np.argmax([float(rgb[:, :, c].std()) for c in range(3)]))
    ch = rgb[:, :, k].astype(np.float32)
    lo, hi = np.percentile(ch, [1.0, 99.7])
    if hi - lo < 1.0:
        return rgb[:, :, k].copy()
    return np.clip((ch - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def detectPupil(rgb):
    # Locate the circular exit pupil (bright region of the fringe channel)
    # Returns (cx, cy, r) using the convex hull of the largest bright region
    # Localiza la pupila de salida circular (región brillante del canal de franjas).
    # Regresa (cx, cy, r) usando la envolvente convexa de la región brillante más grande
    R = stripeChannel(rgb)
    h, w = R.shape
    k = max(3, (min(h, w) // 40) | 1)
    blur = cv2.GaussianBlur(R, (k, k), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Close the dark gaps between stripes so the pupil becomes a single region
    # Cierra los huecos oscuros entre franjas para que la pupila quede como una sola región
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
    h, w = rgb.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)


# ══════════════════════════════════════════════════════════════════════════════
# STRIPE DETECTION (same algorithm for experimental and theoretical Ronchigrams)
# DETECCIÓN DE FRANJAS (mismo algoritmo para los Ronchigramas experimental y teórico)
# ══════════════════════════════════════════════════════════════════════════════
def binarizeStripes(rgb):
    # CLAHE on the fringe channel -> threshold -> remove small components -> closing
    # CLAHE en el canal de franjas -> umbral -> quitar componentes chicas -> cierre
    R = stripeChannel(rgb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    Rb = clahe.apply(R)
    if BIN_METHOD == "adaptive":
        # Local (Gaussian-weighted) mean threshold: works when some stripes are
        # dimmer than others (non-uniform illumination of the pupil)
        # Umbral con media local (ponderada con gaussiana): funciona cuando algunas franjas
        # son más tenues que otras (iluminación no uniforme de la pupila)
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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSE_KERNEL, CLOSE_KERNEL))
    closed = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    return Rb, binary, clean, closed


def columnProfile(binImg, sigma=5):
    # Average intensity per column, smoothed with a 1D Gaussian
    # Intensidad promedio por columna, suavizada con una gaussiana 1D
    profile = np.mean(binImg, axis=0).astype(np.float32)
    k = int(sigma * 6) | 1
    smooth = cv2.GaussianBlur(profile.reshape(1, -1), (k, 1), sigma).flatten()
    return profile, smooth


def detectStripePeaks(smooth):
    # Two-pass peak detection + removal of spurious peaks (too close to a neighbour)
    # Detección de picos en dos pasadas + eliminación de picos espurios (muy cerca de un vecino)
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
    # Un pico mucho más cerca de su vecino que la separación mediana es un fragmento
    # (normalmente en el borde de la pupila): solo se conserva el más fuerte
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


def getAnchorPoints(binImg, peaks, halfWidth, nAnchors=N_ANCHORS):
    # For each stripe, divide its vertical extent in nAnchors bands and take
    # the center of mass of the white pixels of each band as anchor point
    # Para cada franja, divide su extensión vertical en nAnchors bandas y toma el centro
    # de masa de los píxeles blancos de cada banda como punto de anclaje
    h, w = binImg.shape
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
            rr, cc = np.where(sub > 0)
            anchors.append((r0 + rr.mean(), minCol + cc.mean()))
        # Accept the stripe only if it is continuous enough
        # La franja solo se acepta si es lo bastante continua
        if len(anchors) >= nAnchors * 0.6:
            stripes.append({"peak": int(peakCol), "anchors": anchors})
    return stripes


def fringePositions(smooth, stripes, halfWidth):
    # Position of each confirmed fringe in the column profile: the profile maximum
    # near the fringe (same definition as the automatic peaks), or the mean column of
    # its anchor points if the profile has no signal there (e.g. a dim fringe added
    # by hand). The median difference of these positions is the fringe spacing.
    # Posición de cada franja confirmada en el perfil por columnas: el máximo del perfil
    # cerca de la franja (misma definición que los picos automáticos), o la columna media
    # de sus puntos de anclaje si el perfil no tiene señal ahí (p. ej. una franja tenue
    # agregada a mano). La mediana de las diferencias de estas posiciones es la
    # separación entre franjas.
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


def buildCentroidLines(binImg, peaks, medianGap, pupil, stripes=None, halfWidth=None):
    # Centroid line of each stripe: polynomial through its anchor points,
    # sampled every POINT_STEP px (rows aligned to multiples of POINT_STEP)
    # and restricted to the exit pupil. If 'stripes' is given (anchor points
    # confirmed by the user), they are used instead of the automatic ones.
    # Línea de centroides de cada franja: polinomio por sus puntos de anclaje, muestreado
    # cada POINT_STEP px (filas alineadas a múltiplos de POINT_STEP) y restringido a la
    # pupila de salida. Si se da 'stripes' (puntos de anclaje confirmados por el
    # usuario), se usan en lugar de los automáticos.
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
        poly = np.poly1d(np.polyfit(rows, cols, POLY_DEGREE))
        ys = np.arange(np.ceil(rows.min() / POINT_STEP) * POINT_STEP,
                       rows.max() + 1e-9, POINT_STEP, dtype=np.float64)
        xs = poly(ys)
        ok = insidePupil(xs, ys, pupil)
        ys, xs = ys[ok], xs[ok]
        if len(ys) < 3:
            continue

        # Stripe width: white pixels per row around the curve (central 50% of the stripe)
        # Ancho de la franja: píxeles blancos por fila alrededor de la curva (50% central de la franja)
        widths = []
        yA, yB = ys.min(), ys.max()
        for y in np.arange(yA + 0.25 * (yB - yA), yB - 0.25 * (yB - yA), 1.0):
            xc = int(round(poly(y)))
            row = binImg[int(y), max(0, xc - halfWidth):xc + halfWidth]
            nz = np.count_nonzero(row)
            if nz > 0:                      # manual fringes may have no white pixels
                                            # las franjas manuales pueden no tener píxeles blancos
                widths.append(nz)
        width = float(np.median(widths)) if widths else np.nan

        lines.append({"poly": poly, "anchors": s["anchors"], "ys": ys, "xs": xs,
                      "a0": float(ys.min()), "a1": float(ys.max()),
                      "width": width, "xMean": float(xs.mean())})
    lines.sort(key=lambda L: L["xMean"])
    return lines, halfWidth


def surfaceMapping(R, f, D, K, z, n_samples=4000):
    # Ideal Ronchi-SLM geometry: the grating is displayed in the
    # collimated beam, the surface under test (conic of radius R and conic constant K,
    # plano-convex lens of focal length f, so n = 1 + R/f) refracts every ray, and the
    # Ronchigram is recorded at a distance z from the surface vertex.
    # Each ray entering at height h (0 <= h <= D/2) is traced exactly and reaches the
    # camera plane at x(h). This radial mapping entrance -> camera is what bends the
    # straight grating lines in the ideal (theoretical) Ronchigram.
    # Geometría ideal Ronchi-SLM: la rejilla se muestra en el haz colimado, la superficie
    # bajo prueba (cónica de radio R y constante cónica K, lente plano-convexa de distancia
    # focal f, así que n = 1 + R/f) refracta cada rayo, y el Ronchigrama se registra a una
    # distancia z del vértice de la superficie.
    # Cada rayo que entra a la altura h (0 <= h <= D/2) se traza exactamente y llega al
    # plano de la cámara en x(h). Este mapeo radial entrada -> cámara es lo que curva las
    # líneas rectas de la rejilla en el Ronchigrama ideal (teórico).
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
    # Normal unitaria de la cónica, apuntando hacia el lado incidente
    nx, nz = h, (1.0 + K) * sag - R
    L = np.hypot(nx, nz)
    nx, nz = nx / L, nz / L
    # Refraction air -> glass (incident direction (0, 1))
    # Refracción aire -> vidrio (dirección incidente (0, 1))
    cosI = -nz
    eta = 1.0 / n
    k1 = 1.0 - eta ** 2 * (1.0 - cosI ** 2)
    if np.any(k1 < 0):
        raise RuntimeError("Total internal reflection at the surface (check R, f, K, D).")
    tx = (eta * cosI - np.sqrt(k1)) * nx
    tz = eta + (eta * cosI - np.sqrt(k1)) * nz
    # Refraction glass -> air at the flat back face (thin lens)
    # Refracción vidrio -> aire en la cara plana posterior (lente delgada)
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
                                              # aumento en el centro
    return {"R": R, "f": f, "D": D, "K": K, "z": z, "n": n,
            "rhoIn": rhoIn, "rhoCam": rhoCam, "m0": m0, "xMarg": xMarg,
            "maxDev": float(np.max(np.abs(rhoCam - m0 * rhoIn)))}


def camToEntrance(mapping, rhoCam):
    # Normalized entrance radius (at the surface) of the ray that reaches rhoCam
    # Radio de entrada normalizado (en la superficie) del rayo que llega a rhoCam
    return np.interp(rhoCam, mapping["rhoCam"], mapping["rhoIn"])


def buildTheoreticalRonchigram(shape, pupil, xc, spacing, width, mapping):
    # Ideal Ronchigram for the surface parameters: straight, equally spaced grating
    # lines at the surface, mapped to the camera plane with the exact ray mapping.
    # Spacing and width are taken from the experimental Ronchigram (at the center),
    # and one line passes through the central experimental stripe at the pupil center.
    # Ronchigrama ideal para los parámetros de la superficie: líneas de rejilla rectas y
    # equiespaciadas en la superficie, mapeadas al plano de la cámara con el mapeo exacto
    # de rayos. La separación y el ancho se toman del Ronchigrama experimental (en el
    # centro), y una línea pasa por la franja experimental central en el centro de la pupila.
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
                                                     # coordenada de entrada de cada píxel
    p = spacing / r / m0                             # grating period at the entrance
                                                     # periodo de la rejilla en la entrada
    wd = width / r / m0                              # stripe width at the entrance
                                                     # ancho de franja en la entrada
    d0 = (xc - cx) / r
    u0 = d0 * (camToEntrance(mapping, abs(d0)) / abs(d0) if abs(d0) > 1e-9 else 1.0 / m0)
    t = (u - u0) / p
    white = (np.abs(t - np.round(t)) * p <= wd / 2) & inside
    theo = np.where(white, 255, 0).astype(np.uint8)
    nLines = len(np.unique(np.round(t[white]))) if white.any() else 0
    return theo, nLines


def centralIndex(lines, cx, cy):
    # Index of the line passing closest to the pupil center
    # Índice de la línea que pasa más cerca del centro de la pupila
    return int(np.argmin([abs(L["poly"](cy) - cx) for L in lines]))


def lineDifferences(expLines, theoLines, pairs, pupil):
    # Point-to-point difference between corresponding experimental and theoretical
    # centroid lines, at the same rows. Position = theoretical point.
    # Diferencia punto a punto entre las líneas de centroides experimental y teórica
    # correspondientes, en las mismas filas. Posición = punto teórico (nominal).
    rows = []
    for k, (ie, it) in enumerate(pairs):
        E, T = expLines[ie], theoLines[it]
        a0 = max(E["a0"], T["a0"])
        a1 = min(E["a1"], T["a1"])
        ys = np.arange(np.ceil(a0 / POINT_STEP) * POINT_STEP, a1 + 1e-9,
                       POINT_STEP, dtype=np.float64)
        xE = E["poly"](ys)
        xT = T["poly"](ys)
        ok = insidePupil(xT, ys, pupil)
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
    # Desplaza las diferencias para que el mínimo sea cero (los polinomios de Zernike están normalizados)
    df = df.copy()
    df["diffAdj"] = (df[col] - df[col].min()).astype(np.float64)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# GRÁFICAS
# ══════════════════════════════════════════════════════════════════════════════
def plotDiffMap(df, title, pupil, shape, value="diffAdj", outline=None,
                cbarLabel="TA, adjusted to min = 0 (px)"):
    # Point cloud: positions = theoretical points, color (intensity) = adjusted difference
    # outline: optional (N, 2) polygon drawn in cyan (border of the measured region)
    # Nube de puntos: posiciones = puntos teóricos, color (intensidad) = diferencia ajustada
    # outline: polígono (N, 2) opcional dibujado en cian (borde de la región medida)
    fig, ax = plt.subplots(figsize=(8, 8))
    if len(df) > 3000:
        # Dense clouds: drawn as a continuous surface over the points (display only;
        # the data are not modified). Individual markers would leave gaps/bands.
        # Nubes densas: se dibujan como una superficie continua sobre los puntos (solo para
        # mostrar; los datos no se modifican). Los marcadores individuales dejarían huecos/bandas.
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
    # Fondo blanco y una cuadrícula tenue en las gráficas 3D, para que la cuadrícula no
    # compita con los datos
    ax.set_facecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor("#D0D0D0")
        axis._axinfo["grid"].update(color="#EBEBEB", linewidth=0.4, linestyle="-")
        axis.line.set_color("#808080")
    ax.tick_params(colors="#404040", labelsize=8)


# "hot" colormap without its white end, so the highest points stay visible on white
# Mapa de color "hot" sin su extremo blanco, para que los puntos más altos se vean sobre blanco
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
# PASO 1 — VERIFICACIÓN DE LA ALINEACIÓN VERTICAL
# ══════════════════════════════════════════════════════════════════════════════
def measureCentralInclination(rgb, pupil=None):
    # Residual inclination of the central stripes.
    # Each central stripe is represented by its anchor points along its whole length
    # (same segmentation and centroid recognition used in the analysis) and a straight
    # line x = a + b*y is fitted to them; its slope b = (x_bottom - x_top)/(y_bottom - y_top)
    # is the inclination. Using the whole stripe instead of only its two extremes avoids
    # the noise of the pupil border, where the stripes are cut and fragmented.
    # Inclinación residual de las franjas centrales.
    # Cada franja central se representa con sus puntos de anclaje a lo largo de toda su
    # longitud (misma segmentación y reconocimiento de centroides que en el análisis) y se
    # les ajusta una recta x = a + b*y; su pendiente b = (x_abajo - x_arriba)/(y_abajo - y_arriba)
    # es la inclinación. Usar toda la franja en lugar de solo sus dos extremos evita el
    # ruido del borde de la pupila, donde las franjas están cortadas y fragmentadas.
    _, _, _, binImg = binarizeStripes(rgb)
    if pupil is None:
        pupil = detectPupil(rgb)
    cx, cy, _ = pupil
    binImg = cv2.bitwise_and(binImg, circleMask(binImg.shape, pupil))
    _, smooth = columnProfile(binImg)
    _, peaks, gap, _, _ = detectStripePeaks(smooth)
    lines, _ = buildCentroidLines(binImg, peaks, gap, pupil)
    # The ALIGN_N_CENTRAL stripes closest to the pupil center (least aberration effect)
    # Las ALIGN_N_CENTRAL franjas más cercanas al centro de la pupila (menor efecto de aberración)
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


def verifyVerticalAlignment(rgb, pupil=None):
    # Iteratively corrects the residual inclination until it is below ANGLE_THRESHOLD
    # Corrige iterativamente la inclinación residual hasta que quede por debajo de ANGLE_THRESHOLD
    history = []
    totalCorrection = 0.0
    for it in range(ALIGN_MAX_ITER + 1):
        try:
            angle, measures = measureCentralInclination(rgb, pupil)
        except RuntimeError:
            # Too few stripes detected automatically (e.g. a dim Ronchigram):
            # the check is skipped and the Hough orientation is kept
            # Muy pocas franjas detectadas automáticamente (p. ej. un Ronchigrama tenue):
            # se omite la revisión y se conserva la orientación de Hough
            angle, measures = 0.0, []
        history.append((angle, measures))
        if abs(angle) <= ANGLE_THRESHOLD or it == ALIGN_MAX_ITER:
            break
        # Rotating by -angle cancels the inclination of the stripes
        # Rotar por -angle cancela la inclinación de las franjas
        rgb = rotateImage(rgb, -angle)
        totalCorrection += -angle

    # Visualization: extremes of the central stripes and a vertical reference line
    # Visualización: extremos de las franjas centrales y una línea vertical de referencia
    vis = rgb.copy()
    finalAngle, finalMeasures = history[-1]
    for m in finalMeasures:
        pTop = (int(round(m["xTop"])), int(m["yTop"]))
        pBot = (int(round(m["xBot"])), int(m["yBot"]))
        cv2.line(vis, pTop, (pTop[0], pBot[1]), (255, 255, 255), 1)   # vertical reference
                                                                      # referencia vertical
        cv2.line(vis, pTop, pBot, (0, 255, 0), 2)                       # measured extremes
                                                                        # extremos medidos
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
# PASO 2 — ANÁLISIS DE UN RONCHIGRAMA (vertical u horizontal)
# ══════════════════════════════════════════════════════════════════════════════
def analyzeRonchigram(rgb, pupil, label, horizontal, results, prog, p0, p1, mapping,
                      editor=None, stripesOverride=None):
    # Same algorithm for both orientations. The horizontal Ronchigram is analyzed
    # on its transpose (stripes become vertical) and the results are transposed back,
    # so the output is expressed in the coordinates of the rotated image.
    # Output convention: vertical   -> lines x = poly(y), differences in x
    #                    horizontal -> lines y = poly(x), differences in y
    # Mismo algoritmo para las dos orientaciones. El Ronchigrama horizontal se analiza
    # sobre su transpuesta (las franjas quedan verticales) y los resultados se vuelven a
    # transponer, así la salida queda en las coordenadas de la imagen rotada.
    # Convención de salida: vertical   -> líneas x = poly(y), diferencias en x
    #                       horizontal -> líneas y = poly(x), diferencias en y
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
    # ── Segmentación ──
    Rb, binary, clean, closed = binarizeStripes(work)
    closed = cv2.bitwise_and(closed, pMask)
    results.append({"title": "CLAHE" + tag, "image": out(Rb)})
    results.append({"title": "Binary Image" + tag, "image": out(binary)})
    results.append({"title": "Connected Components Filtering" + tag, "image": out(clean)})
    results.append({"title": "Closed Stripes inside Pupil" + tag, "image": out(closed)})
    step(0.15)

    # ── Stripe detection ──
    # ── Detección de franjas ──
    profile, smooth = columnProfile(closed)
    try:
        peaksR, peaks, gap, minProm, removed = detectStripePeaks(smooth)
        peakText = [f"Initial detection: {len(peaksR)} peaks",
                    f"Discarded fragments: {len(removed)}",
                    f"Final stripes: {len(peaks)}",
                    f"Median gap (spacing): {gap:.3f} px"]
    except RuntimeError as ex:
        # Fewer than 2 stripes found automatically: allowed only when the user
        # reviews the fringes (or they come from the vertical review)
        # Menos de 2 franjas encontradas automáticamente: solo se permite cuando el usuario
        # revisa las franjas (o cuando vienen de la revisión vertical)
        if editor is None and stripesOverride is None:
            raise
        peaksR, peaks, gap, removed = np.array([], int), np.array([], int), None, []
        peakText = [f"Automatic detection: {ex}"]
    results.append({"title": "Intensity Profile" + tag,
                    "image": plotProfile(profile, smooth, axisLabel)})
    results.append({"title": "Peaks Detection" + tag,
                    "image": plotPeaks(smooth, peaksR, peaks, removed, axisLabel),
                    "text": peakText})
    step(0.3)

    # ── Anchor points: automatic, then reviewed by the user ──
    # ── Puntos de anclaje: automáticos, luego revisados por el usuario ──
    halfWidth = max(int(gap * 0.4), 10) if gap else max(int(0.15 * rP), 10)
    if stripesOverride is not None:
        stripes = stripesOverride
        anchorText = ["Anchor points: the fringes confirmed in the vertical Ronchigram, "
                      "rotated 90° (same fringes)"]
    else:
        stripes = getAnchorPoints(closed, peaks, halfWidth) if len(peaks) else []
        anchorText = [f"Half-width of search window: {halfWidth} px",
                      f"Automatically detected stripes: {len(stripes)}"]
    if editor is not None:
        nAuto = len(stripes)
        stripes = editor({"image": work, "stripes": stripes, "pupil": pup,
                          "halfWidth": halfWidth, "channel": stripeChannel(work)})
        if stripes is None:
            raise RuntimeError("Pipeline cancelled by the user in the fringe review.")
        anchorText = [f"Automatically detected fringes: {nAuto}",
                      f"Fringes confirmed by the user: {len(stripes)}",
                      "Anchor points shown: the ones confirmed by the user"]

    # Anchor points used in the analysis (the confirmed ones when there was a review)
    # Puntos de anclaje usados en el análisis (los confirmados cuando hubo revisión)
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
                drawCurve(vis, np.poly1d(np.polyfit(rr, cc, POLY_DEGREE)), rr.min(), rr.max(),
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
    # ── Líneas de centroides experimentales (de los puntos de anclaje confirmados) ──
    expLines, halfWidth = buildCentroidLines(closed, peaks, gap, pup, stripes=stripes,
                                             halfWidth=halfWidth)
    if len(expLines) < 2:
        raise RuntimeError(f"{label}: at least 2 fringes (with at least "
                           f"{POLY_DEGREE + 1} anchor points each) are needed.")
    # Spacing of the confirmed fringes (same definition as the automatic detection)
    # Separación de las franjas confirmadas (misma definición que la detección automática)
    gap = float(np.median(np.diff(fringePositions(smooth, stripes, halfWidth))))
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
    # ── Ronchigrama teórico: superficie ideal (R, f, D, K, z), misma separación y ancho
    #    que el experimental, anclado en la franja central ──
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

    # Same centroid recognition applied to the theoretical Ronchigram
    # El mismo reconocimiento de centroides se aplica al Ronchigrama teórico
    _, tSmooth = columnProfile(theoBin)
    _, tPeaks, tGap, _, _ = detectStripePeaks(tSmooth)
    theoLines, _ = buildCentroidLines(theoBin, tPeaks, tGap, pup)
    it0 = centralIndex(theoLines, xc, cy)

    vis = cv2.cvtColor(theoBin, cv2.COLOR_GRAY2RGB)
    for L in theoLines:
        drawCurve(vis, L["poly"], L["a0"], L["a1"], (255, 0, 0), 2)
        for x, y in zip(L["xs"], L["ys"]):
            cv2.circle(vis, (int(round(x)), int(round(y))), 2, (0, 120, 255), -1)
    results.append({"title": "Theoretical Centroid Lines" + tag, "image": out(vis),
                    "text": [f"Recognized theoretical lines: {len(theoLines)}"]})
    step(0.7)

    # ── Correspondence by order, counted from the central stripe ──
    # The order of each fringe is its position relative to the central one in
    # units of the spacing, so a missing or deleted fringe does not shift the rest
    # ── Correspondencia por orden, contado desde la franja central ──
    # El orden de cada franja es su posición respecto a la central en unidades de la
    # separación, así una franja faltante o borrada no recorre a las demás
    pairs = []
    for ie, L in enumerate(expLines):
        k = int(round((L["poly"](cy) - xc) / gap))
        it = it0 + k
        if 0 <= it < len(theoLines) and it not in [q[1] for q in pairs]:
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
                    "text": [f"Corresponding pairs: {len(pairs)}"] +
                            [f"  Exp. line {ie + 1} -> Theo. line {it + 1}" for ie, it in pairs]})

    # ── Output in the frame of the analyzed image ──
    # ── Salida en el marco de la imagen analizada ──
    if horizontal:
        df = pd.DataFrame({"line": dfT["line"], "x": dfT["along"], "y": dfT["acrossTheo"],
                           "yExp": dfT["acrossExp"], "yTheo": dfT["acrossTheo"],
                           "diffRaw": dfT["diffRaw"], "diffAdj": dfT["diffAdj"]})
    else:
        df = pd.DataFrame({"line": dfT["line"], "x": dfT["acrossTheo"], "y": dfT["along"],
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
            "spacing": gap, "width": expWidth,
            "stripes": [{"peak": int(round(L["xMean"])), "anchors": L["anchors"]}
                        for L in expLines]}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — GRID
# PASO 3 — MALLA
# ══════════════════════════════════════════════════════════════════════════════
def intersectLines(LV, LH):
    # Vertical line: x = pV(y), y in [LV.a0, LV.a1]
    # Horizontal line: y = pH(x), x in [LH.a0, LH.a1]
    # Intersection: g(y) = pH(pV(y)) - y = 0
    # Línea vertical: x = pV(y), y en [LV.a0, LV.a1]
    # Línea horizontal: y = pH(x), x en [LH.a0, LH.a1]
    # Intersección: g(y) = pH(pV(y)) - y = 0
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
    # Malla experimental: líneas de centroides experimentales verticales + horizontales
    # Malla teórica:      líneas de centroides teóricas verticales + horizontales
    # Las intersecciones se emparejan respetando la correspondencia de líneas de cada Ronchigrama
    rows = []
    for kv, (ieV, itV) in enumerate(V["pairs"]):
        for kh, (ieH, itH) in enumerate(H["pairs"]):
            pe = intersectLines(V["exp"][ieV], H["exp"][ieH])
            pt = intersectLines(V["theo"][itV], H["theo"][itH])
            if pe is None or pt is None or not insidePupil(pt[0], pt[1], pupil):
                continue
            dx, dy = pe[0] - pt[0], pe[1] - pt[1]
            rows.append({"lineV": kv, "lineH": kh, "x": pt[0], "y": pt[1],
                         "xExp": pe[0], "yExp": pe[1], "dx": dx, "dy": dy,
                         "diffRaw": float(np.hypot(dx, dy))})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No intersections found between vertical and horizontal lines.")
    df = adjustMin(df)

    # Drawn on a white background (experimental: green / blue; theoretical: red)
    # Dibujadas sobre fondo blanco (experimental: verde / azul; teórica: rojo)
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
        pt = (int(round(r["x"])), int(round(r["y"])))
        cv2.line(overlay, pe, pt, (230, 140, 0), 2)
        cv2.circle(overlay, pt, 5, (0, 0, 0), -1)
        cv2.circle(overlay, pe, 5, (150, 0, 200), -1)
    cx, cy, r = pupil
    for im in (gridExp, gridTheo, overlay):
        cv2.circle(im, (int(round(cx)), int(round(cy))), int(round(r)), (120, 120, 120), 1)
    return df, gridExp, gridTheo, overlay


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — REFINEMENT OF THE POINT CLOUDS (multilinear interpolation)
# PASO 4 — REFINAMIENTO DE LAS NUBES DE PUNTOS (interpolación multilineal)
# ══════════════════════════════════════════════════════════════════════════════
def refineCoords(coords, k, minSpacing=MIN_SPACING):
    # Inserts k equally spaced coordinates between each pair of neighbours.
    # k is reduced if the new spacing would be below minSpacing (Nyquist).
    # Inserta k coordenadas equiespaciadas entre cada par de vecinos.
    # k se reduce si la nueva separación quedaría por debajo de minSpacing (Nyquist).
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
    # Construye una retícula más fina (k puntos nuevos entre líneas vecinas en las
    # direcciones refinadas) e interpola en ella las diferencias ajustadas.
    # Se descartan los puntos fuera de la región con datos o de la pupila.
    #
    # vector=True (grid): the signed components dx and dy are interpolated
    # separately with GRID_INTERP_METHOD and the difference is recomputed as
    # |(dx, dy)|. Interpolating the magnitude directly produces a cone-shaped
    # artifact where the displacement is close to zero, and facets due to the
    # few intersection points.
    # vector=True (malla): las componentes con signo dx y dy se interpolan por separado
    # con GRID_INTERP_METHOD y la diferencia se recalcula como |(dx, dy)|. Interpolar
    # directamente la magnitud produce un artefacto en forma de cono donde el
    # desplazamiento es casi cero, y facetas por los pocos puntos de intersección.
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
        # También se interpola la TA con signo (experimental - teórica): es el valor que
        # se usa para reconstruir el frente de onda (el mapa con mín = 0 es solo para mostrar)
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
    # Extiende un conjunto de coordenadas equiespaciadas con el mismo paso hasta que
    # cubra [lo, hi]. Las coordenadas originales no cambian.
    coords = np.unique(np.round(np.asarray(coords, dtype=np.float64), 9))
    step = float(np.median(np.diff(coords)))
    left = np.arange(coords[0] - step, lo - step, -step)[::-1]
    right = np.arange(coords[-1] + step, hi + step, step)
    return np.concatenate([left, coords, right])


def polyBasis(u, v, degree):
    # Monomials u^l v^h with l + h <= degree (u, v normalized to the unit pupil)
    # Monomios u^l v^h con l + h <= degree (u, v normalizadas a la pupila unitaria)
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
    # Cubre toda la pupila de salida:
    #  - los puntos de la retícula que ya tienen datos conservan su valor
    #  - los demás se estiman con un polinomio de grado FULL_PUPIL_DEGREE ajustado
    #    (mínimos cuadrados) a todos los datos, más el residuo del ajuste del punto con
    #    datos más cercano, que decae como exp(-d / FULL_PUPIL_DECAY), así la extensión
    #    se une de forma continua a la región medida
    # vector=True (malla): dx y dy se extienden por separado, luego |(dx, dy)|
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
    # Borde de la región medida, para las gráficas
    pts = np.column_stack([df["x"].values, df["y"].values])
    outline = pts[ConvexHull(pts).vertices]
    return out, rms, outline


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — WAVEFRONT: TA surfaces, slopes, W_P (principal), W_G (complementary), Zernike
# PASO 5 — FRENTE DE ONDA: superficies de TA, pendientes, W_P (principal), W_G (complementaria), Zernike
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
    # Continuous TA surface ([M] Sec. 5B, [S] Eqs. S18-S19 with omega = 1): least-squares
    # fit of the TA point cloud (u, v normalized to the unit pupil) with the products
    # of Legendre polynomials P_m(u) P_n(v), m, n <= degree. They span the same
    # polynomials as sum A_lh u^l v^h of the previous articles (Appl. Phys. B, Eq. 12;
    # Optik, Eq. 4) and are numerically better conditioned.
    # Superficie continua de TA ([M] Sec. 5B, [S] Ecs. S18-S19 con omega = 1): ajuste por
    # mínimos cuadrados de la nube de puntos de TA (u, v normalizadas a la pupila unitaria)
    # con los productos de polinomios de Legendre P_m(u) P_n(v), m, n <= degree. Generan
    # los mismos polinomios que sum A_lh u^l v^h de los artículos anteriores (Appl. Phys. B,
    # Ec. 12; Optik, Ec. 4) y están mejor condicionados numéricamente.
    V = npleg.legvander2d(u, v, [degree, degree])
    coef = np.linalg.lstsq(V, values, rcond=None)[0]
    rms = float(np.sqrt(np.mean((V @ coef - values) ** 2)))
    return coef.reshape(degree + 1, degree + 1), rms


def zernikeIndices(Q):
    # ANSI order: n = 0..Q, m = -n, -n+2, ..., n  ->  (Q+1)(Q+2)/2 terms
    # Orden ANSI: n = 0..Q, m = -n, -n+2, ..., n  ->  (Q+1)(Q+2)/2 términos
    return [(n, m) for n in range(Q + 1) for m in range(-n, n + 1, 2)]


def zernike(n, m, rho, theta):
    # Normalized Zernike polynomial (orthonormal on the unit disk).
    # Radial part through Jacobi polynomials, numerically stable for high n.
    # Polinomio de Zernike normalizado (ortonormal en el disco unitario).
    # Parte radial con polinomios de Jacobi, numéricamente estable para n alto.
    k = (n - abs(m)) // 2
    R = (-1) ** k * rho ** abs(m) * eval_jacobi(k, abs(m), 0, 1 - 2 * rho ** 2)
    if m == 0:
        return np.sqrt(n + 1) * R
    ang = np.cos(m * theta) if m > 0 else np.sin(-m * theta)
    return np.sqrt(2 * (n + 1)) * R * ang


def zernikeTerms(idx, rho, theta):
    # Yields (j, Z_j) for every term, in OSA/ANSI order. The radial part of each
    # (n, |m|) is computed once and shared by +m and -m.
    # Entrega (j, Z_j) para cada término, en orden OSA/ANSI. La parte radial de cada
    # (n, |m|) se calcula una vez y la comparten +m y -m.
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
    # Least-squares Zernike fit ([S] Eq. S109, uniform weights) through the normal
    # equations, accumulated in chunks so many points can be used without much memory
    # Ajuste de Zernike por mínimos cuadrados ([S] Ec. S109, pesos uniformes) con las
    # ecuaciones normales, acumuladas por bloques para poder usar muchos puntos sin
    # ocupar mucha memoria
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


def pupilGridOperators(N=WAVEFRONT_GRID):
    # Cartesian grid of N x N nodes on [-1, 1]^2 (rows = y downwards, as the images);
    # the nodes inside the unit pupil (x^2 + y^2 <= 1) are the unknowns of W_P.
    # Discrete operators of [M] Eqs. 23-24 / [S] Eqs. S55-S56, built only with nodes
    # inside the pupil (no value outside the pupil is ever used):
    #   Dx, Dy: centered differences; one-sided at the pupil border when one of the two
    #           neighbours is outside; a node without neighbours along an axis is left
    #           out of that operator. kx, ky = node index of every row of Dx, Dy.
    #   L:      5-point Laplacian, only at nodes whose four neighbours are inside.
    # Malla cartesiana de N x N nodos en [-1, 1]^2 (filas = y hacia abajo, como las
    # imágenes); los nodos dentro de la pupila unitaria (x^2 + y^2 <= 1) son las incógnitas
    # de W_P. Operadores discretos de [M] Ecs. 23-24 / [S] Ecs. S55-S56, construidos solo
    # con nodos dentro de la pupila (nunca se usa un valor fuera de la pupila):
    #   Dx, Dy: diferencias centradas; de un lado en el borde de la pupila cuando uno de
    #           los dos vecinos está afuera; un nodo sin vecinos en un eje queda fuera de
    #           ese operador. kx, ky = índice del nodo de cada fila de Dx, Dy.
    #   L:      laplaciano de 5 puntos, solo en los nodos con sus cuatro vecinos adentro.
    g = np.linspace(-1.0, 1.0, N)
    h = float(g[1] - g[0])
    X, Y = np.meshgrid(g, g)
    inside = X ** 2 + Y ** 2 <= 1.0
    nid = -np.ones((N, N), dtype=int)
    nid[inside] = np.arange(int(inside.sum()))
    nodes = list(zip(*np.where(inside)))

    def isIn(i, j):
        return 0 <= i < N and 0 <= j < N and inside[i, j]

    def derivative(di, dj):
        rows, cols, vals, keep = [], [], [], []
        for (i, j) in nodes:
            fwd, bwd = isIn(i + di, j + dj), isIn(i - di, j - dj)
            k = len(keep)
            if fwd and bwd:          # centered difference
                                     # diferencia centrada
                rows += [k, k]
                cols += [nid[i + di, j + dj], nid[i - di, j - dj]]
                vals += [0.5 / h, -0.5 / h]
            elif fwd:                # one-sided (forward)
                                     # de un lado (hacia adelante)
                rows += [k, k]
                cols += [nid[i + di, j + dj], nid[i, j]]
                vals += [1.0 / h, -1.0 / h]
            elif bwd:                # one-sided (backward)
                                     # de un lado (hacia atrás)
                rows += [k, k]
                cols += [nid[i, j], nid[i - di, j - dj]]
                vals += [1.0 / h, -1.0 / h]
            else:
                continue
            keep.append(nid[i, j])
        D = sparse.csr_matrix((vals, (rows, cols)), shape=(len(keep), len(nodes)))
        return D, np.array(keep, dtype=int)

    Dx, kx = derivative(0, 1)        # x = columns
                                     # x = columnas
    Dy, ky = derivative(1, 0)        # y = rows
                                     # y = filas
    rows, cols, vals = [], [], []
    for (i, j) in nodes:
        nb = [(i, j + 1), (i, j - 1), (i + 1, j), (i - 1, j)]
        if all(isIn(a, b) for a, b in nb):
            k = len(rows) // 5
            rows += [k] * 5
            cols += [nid[i, j]] + [nid[a, b] for a, b in nb]
            vals += [-4.0 / h ** 2] + [1.0 / h ** 2] * 4
    L = sparse.csr_matrix((vals, (rows, cols)), shape=(len(rows) // 5, len(nodes)))
    return {"g": g, "h": h, "X": X, "Y": Y, "inside": inside,
            "Dx": Dx, "kx": kx, "Dy": Dy, "ky": ky, "L": L}


def solveVariational(ops, p, q, mu, omega):
    # Discrete variational reconstruction W_P ([M] Eqs. 22-26, [S] Eqs. S40-S45):
    #   u* = argmin ||W^1/2 (Dx u - p)||^2 + ||W^1/2 (Dy u - q)||^2 + mu ||L u||^2
    # whose optimality condition is the sparse linear system ([M] Eq. 24)
    #   (Dx^T W Dx + Dy^T W Dy + mu L^T L) u = Dx^T W p + Dy^T W q,
    # solved together with the zero-mean (piston) constraint 1^T u = 0 ([M] Eq. 25)
    # through a Lagrange multiplier ([S] Eq. S44). p, q, omega: values at the nodes.
    # Returns u, the weighted slope residual R ([S] Eq. S48) and the smoothness S
    # ([S] Eq. S49).
    # Reconstrucción variacional discreta W_P ([M] Ecs. 22-26, [S] Ecs. S40-S45):
    #   u* = argmin ||W^1/2 (Dx u - p)||^2 + ||W^1/2 (Dy u - q)||^2 + mu ||L u||^2
    # cuya condición de optimalidad es el sistema lineal disperso ([M] Ec. 24)
    #   (Dx^T W Dx + Dy^T W Dy + mu L^T L) u = Dx^T W p + Dy^T W q,
    # resuelto junto con la restricción de media cero (pistón) 1^T u = 0 ([M] Ec. 25)
    # con un multiplicador de Lagrange ([S] Ec. S44). p, q, omega: valores en los nodos.
    # Regresa u, el residuo ponderado de pendientes R ([S] Ec. S48) y la suavidad S
    # ([S] Ec. S49).
    Dx, kx, Dy, ky, L = ops["Dx"], ops["kx"], ops["Dy"], ops["ky"], ops["L"]
    n = Dx.shape[1]
    wx, wy = omega[kx], omega[ky]
    A = (Dx.T @ sparse.diags(wx) @ Dx + Dy.T @ sparse.diags(wy) @ Dy
         + mu * (L.T @ L)).tocsc()
    b = Dx.T @ (wx * p[kx]) + Dy.T @ (wy * q[ky])
    ones = sparse.csc_matrix(np.ones((n, 1)))
    M = sparse.bmat([[A, ones], [ones.T, None]], format="csc")
    u = spsolve(M, np.concatenate([b, [0.0]]))[:n]
    R = float(np.sqrt(np.sum(wx * (Dx @ u - p[kx]) ** 2) + np.sum(wy * (Dy @ u - q[ky]) ** 2)))
    S = float(np.linalg.norm(L @ u))
    return u, R, S


def lCurveCorner(mus, R, S):
    # Discrete L-curve ([S] Eqs. S51-S54): xi = log R(mu), eta = log S(mu), t = log mu;
    # curvature kappa_L(t) = |xi' eta'' - eta' xi''| / (xi'^2 + eta'^2)^(3/2), evaluated
    # with finite differences; the selected mu is the one of maximum curvature (the
    # two extreme candidates are excluded, their derivatives are one-sided).
    # Curva L discreta ([S] Ecs. S51-S54): xi = log R(mu), eta = log S(mu), t = log mu;
    # curvatura kappa_L(t) = |xi' eta'' - eta' xi''| / (xi'^2 + eta'^2)^(3/2), evaluada con
    # diferencias finitas; el mu elegido es el de máxima curvatura (se excluyen los dos
    # candidatos extremos, porque sus derivadas son de un solo lado).
    xi, eta, t = np.log(R), np.log(S), np.log(mus)
    d1x, d1y = np.gradient(xi, t), np.gradient(eta, t)
    d2x, d2y = np.gradient(d1x, t), np.gradient(d1y, t)
    curv = np.abs(d1x * d2y - d1y * d2x) / ((d1x ** 2 + d1y ** 2) ** 1.5 + 1e-30)
    curv[0] = curv[-1] = -np.inf
    return int(np.argmax(curv)), curv


def wavefrontGaussLegendre(slopes, X, Y, nodes=GL_NODES):
    # Complementary reconstruction W_G ([M] Eqs. 27-30, [S] Eqs. S60-S73): line
    # integral of the slope field g = (p, q) along the straight path from the pupil
    # center r0 = (0, 0) to every node r_i,
    #   W_G(r_i) = C + 1/2 sum_m w_m g[r0 + (xi_m + 1)/2 (r_i - r0)] . (r_i - r0),
    # with the Gauss-Legendre nodes xi_m and weights w_m on [-1, 1]. The slopes at the
    # quadrature nodes are evaluated from the continuous TA surfaces ([S] Eq. S73).
    # C = W_G(r0) = 0 here; the piston is fixed afterwards with a zero pupil average.
    # Reconstrucción complementaria W_G ([M] Ecs. 27-30, [S] Ecs. S60-S73): integral de
    # línea del campo de pendientes g = (p, q) por la trayectoria recta desde el centro
    # de la pupila r0 = (0, 0) hasta cada nodo r_i,
    #   W_G(r_i) = C + 1/2 sum_m w_m g[r0 + (xi_m + 1)/2 (r_i - r0)] . (r_i - r0),
    # con los nodos xi_m y pesos w_m de Gauss-Legendre en [-1, 1]. Las pendientes en los
    # nodos de cuadratura se evalúan con las superficies continuas de TA ([S] Ec. S73).
    # Aquí C = W_G(r0) = 0; el pistón se fija después con promedio cero en la pupila.
    xi, w = np.polynomial.legendre.leggauss(nodes)
    W = np.zeros_like(X)
    for xm, wm in zip(xi, w):
        t = 0.5 * (xm + 1.0)
        p, q = slopes(t * X, t * Y)
        W += 0.5 * wm * (p * X + q * Y)
    return W


def removePlane(W, X, Y):
    # [M] Eq. 31: subtracts the least-squares plane a0 + ax x + ay y (piston and tilt)
    # over the pupil nodes. Returns the detrended values and (a0, ax, ay).
    # [M] Ec. 31: resta el plano de mínimos cuadrados a0 + ax x + ay y (pistón e
    # inclinación) sobre los nodos de la pupila. Regresa los valores sin el plano y (a0, ax, ay).
    A = np.column_stack([np.ones_like(X), X, Y])
    coef = np.linalg.lstsq(A, W, rcond=None)[0]
    return W - A @ coef, coef


def computeWavefront(taX, taY, pupil, nZernike=ZERNIKE_Q, mapping=None, mmPerPx=None,
                     progress=None):
    # taX, taY: point clouds (x, y, value) of the signed transverse aberrations in x and
    # y, in pixels of the working image (positions = nominal (theoretical) points of
    # the fringes, [M] Eq. 14).
    # Returns the TA surfaces, the slope field, W_P (principal), W_G (complementary),
    # the Zernike fit of W_P and the maps shown in stage H.
    # taX, taY: nubes de puntos (x, y, valor) de las aberraciones transversales con signo
    # en x y en y, en píxeles de la imagen de trabajo (posiciones = puntos nominales
    # (teóricos) de las franjas, [M] Ec. 14).
    # Regresa las superficies de TA, el campo de pendientes, W_P (principal), W_G
    # (complementaria), el ajuste de Zernike de W_P y los mapas que se muestran en la etapa H.
    cx, cy, r = pupil
    def prog(f):
        if progress:
            progress(f)

    # ── Continuous TA surfaces (pixels), coordinates normalized to the pupil ──
    # ── Superficies continuas de TA (píxeles), coordenadas normalizadas a la pupila ──
    cX, rmsX = fitTASurface((taX[0] - cx) / r, (taX[1] - cy) / r, taX[2])
    cY, rmsY = fitTASurface((taY[0] - cx) / r, (taY[1] - cy) / r, taY[2])

    # ── Calibration ([M] Eqs. 13 and 15) ──
    # TA [mm] = Gamma * s * dx [px], with Gamma = 1 and s = mmPerPx;
    # p = kappa * TA_x, q = kappa * TA_y, with kappa = WAVEFRONT_SIGN / z (-1/z).
    # The slopes are expressed per unit of normalized pupil coordinate (pupil radius
    # a = D/2 mm) and W in micrometers:  dW/du [µm] = 1000 * a * kappa * TA [mm].
    # ── Calibración ([M] Ecs. 13 y 15) ──
    # TA [mm] = Gamma * s * dx [px], con Gamma = 1 y s = mmPerPx;
    # p = kappa * TA_x, q = kappa * TA_y, con kappa = WAVEFRONT_SIGN / z (-1/z).
    # Las pendientes se expresan por unidad de coordenada normalizada de la pupila (radio
    # de la pupila a = D/2 mm) y W en micrómetros:  dW/du [µm] = 1000 * a * kappa * TA [mm].
    gamma = 1.0
    s = float(mmPerPx)
    kappa = WAVEFRONT_SIGN / mapping["z"]
    a = mapping["D"] / 2.0
    kSlope = 1000.0 * a * kappa * gamma * s       # µm per unit of u, per pixel of TA
                                                  # µm por unidad de u, por píxel de TA

    def slopes(U, V):
        # Slope field g = (p, q) at normalized pupil points ([M] Eq. 15, [S] Eq. S20)
        # Campo de pendientes g = (p, q) en puntos normalizados de la pupila ([M] Ec. 15, [S] Ec. S20)
        return (kSlope * npleg.legval2d(U, V, cX), kSlope * npleg.legval2d(U, V, cY))

    ops = pupilGridOperators(WAVEFRONT_GRID)
    X, Y, inside = ops["X"], ops["Y"], ops["inside"]
    xq, yq = X[inside], Y[inside]
    p, q = slopes(xq, yq)
    omega = np.full(len(xq), WP_OMEGA)
    prog(0.05)

    # ── W_P: variational reconstruction, mu selected by the L-curve ──
    # ── W_P: reconstrucción variacional, mu elegido con la curva L ──
    nMu = len(WP_MU_CANDIDATES)
    sols, Rs, Ss = [], [], []
    for k, mu in enumerate(WP_MU_CANDIDATES):
        u, Rk, Sk = solveVariational(ops, p, q, mu, omega)
        sols.append(u)
        Rs.append(Rk)
        Ss.append(Sk)
        prog(0.05 + 0.75 * (k + 1) / nMu)
    kBest, curv = lCurveCorner(WP_MU_CANDIDATES, np.array(Rs), np.array(Ss))
    WPq = sols[kBest]
    mu = float(WP_MU_CANDIDATES[kBest])
    # Differential fidelity of W_P ([S] Eq. S47), informative
    # Fidelidad diferencial de W_P ([S] Ec. S47), informativa
    eGrad = Rs[kBest] / (np.sqrt(np.sum(omega * p ** 2) + np.sum(omega * q ** 2)) + 1e-30)

    # ── W_G: Gauss-Legendre integration along straight paths from the center ──
    # ── W_G: integración de Gauss-Legendre por trayectorias rectas desde el centro ──
    WGq = wavefrontGaussLegendre(slopes, xq, yq)
    WGq = WGq - WGq.mean()                         # zero pupil average (piston)
                                                   # promedio cero en la pupila (pistón)
    prog(0.85)

    # ── Piston and tilt removed for display ([M] Eq. 31) ──
    # ── Pistón e inclinación quitados para mostrar ([M] Ec. 31) ──
    WPd, planeP = removePlane(WPq, xq, yq)
    WGd, planeG = removePlane(WGq, xq, yq)

    # ── Zernike fit of W_P (degree n, OSA/ANSI order, normalized polynomials) ──
    # ── Ajuste de Zernike de W_P (grado n, orden OSA/ANSI, polinomios normalizados) ──
    rq = np.sqrt(xq ** 2 + yq ** 2)
    tq = np.mod(np.arctan2(yq, xq), 2 * np.pi)
    idx = zernikeIndices(nZernike)
    coefs = zernikeFit(idx, rq, tq, WPq)
    Wz = zernikeEval(coefs, idx, rq, tq)
    fitRms = float(np.sqrt(np.mean((Wz - WPq) ** 2)))
    WfQ = zernikeEval(coefs, idx, rq, tq, skip=REMOVE_TERMS)
    prog(1.0)

    def onGrid(values):
        M = np.full(X.shape, np.nan)
        M[inside] = values
        return M

    def filled(values):
        # Values on the whole square grid: the nodes outside the pupil take the value
        # of the nearest pupil node. Used only to interpolate near the pupil border
        # (W_F export); those nodes are never part of the reconstruction.
        # Valores en toda la malla cuadrada: los nodos fuera de la pupila toman el valor del
        # nodo de la pupila más cercano. Solo se usa para interpolar cerca del borde de la
        # pupila (exportación de W_F); esos nodos nunca son parte de la reconstrucción.
        M = onGrid(values)
        _, ind = cKDTree(np.column_stack([xq, yq])).query(
            np.column_stack([X[~inside], Y[~inside]]))
        M[~inside] = values[ind]
        return M

    lcurve = pd.DataFrame({"mu": WP_MU_CANDIDATES, "slope residual R": Rs,
                           "smoothness S": Ss, "curvature": curv})
    return {"idx": idx, "coefs": coefs, "fitRms": fitRms, "rmsX": rmsX, "rmsY": rmsY,
            "nZernike": nZernike, "nPointsFit": int(len(WPq)),
            "surface": {k_: mapping[k_] for k_ in ("R", "f", "D", "K", "z")},
            "calibration": {"s_mm_per_px": s, "Gamma": gamma, "kappa_per_mm": kappa},
            # maps (rows = y downwards), NaN outside the pupil
            # mapas (filas = y hacia abajo), NaN fuera de la pupila
            "grid": ops["g"],
            "taX": onGrid(1000.0 * gamma * s * npleg.legval2d(xq, yq, cX)),   # TA_x, µm
            "taY": onGrid(1000.0 * gamma * s * npleg.legval2d(xq, yq, cY)),   # TA_y, µm
            "WP": onGrid(WPq), "WPd": onGrid(WPd),
            "WG": onGrid(WGq), "WGd": onGrid(WGd),
            "W": onGrid(WfQ),
            "WPfilled": filled(WPq), "WGfilled": filled(WGq),
            "mu": mu, "lcurve": lcurve, "eGrad": float(eGrad),
            "nNodes": int(len(WPq))}


def computeWF(wf, geom, progress=None):
    # Final fit on the full-size image:
    #  1) W_P is interpolated (bilinear, from the pupil grid) to a matrix of the same
    #     size as the crop, [c d], with the same pupil as the crop (each crop pixel is
    #     mapped to the working frame with the same rotation and translation applied by
    #     the pipeline, then normalized to the pupil).
    #  2) The Zernike polynomials of degree n are fitted by least squares to ALL the
    #     pupil pixels of that large W_P.
    #  3) W_F = that fit evaluated on the same [c d] matrix (NaN outside the pupil).
    # W_G is interpolated the same way and saved for reference.
    # Ajuste final en la imagen de tamaño completo:
    #  1) W_P se interpola (bilineal, desde la malla de la pupila) a una matriz del mismo
    #     tamaño que el recorte, [c d], con la misma pupila que el recorte (cada píxel del
    #     recorte se lleva al marco de trabajo con la misma rotación y traslación que
    #     aplica el proceso, y luego se normaliza a la pupila).
    #  2) Los polinomios de Zernike de grado n se ajustan por mínimos cuadrados a TODOS
    #     los píxeles de la pupila de ese W_P grande.
    #  3) W_F = ese ajuste evaluado en la misma matriz [c d] (NaN fuera de la pupila).
    # W_G se interpola igual y se guarda como referencia.
    h, w = geom["cropShape"]
    A = geom["cropToWork"]
    cx, cy, r = geom["pupil"]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xw = A[0, 0] * xx + A[0, 1] * yy + A[0, 2]
    yw = A[1, 0] * xx + A[1, 1] * yy + A[1, 2]
    u, v = (xw - cx) / r, (yw - cy) / r
    rho = np.sqrt(u ** 2 + v ** 2)
    inside = rho <= 1.0
    uin, vin = u[inside], v[inside]
    rin = rho[inside]
    theta = np.mod(np.arctan2(vin, uin), 2 * np.pi)
    idx = wf["idx"]

    # 1) W_P (and W_G) interpolated to the crop size; grid axes = (y, x)
    # 1) W_P (y W_G) interpolados al tamaño del recorte; ejes de la malla = (y, x)
    g = wf["grid"]
    pts = np.column_stack([vin, uin])
    WP_in = RegularGridInterpolator((g, g), wf["WPfilled"])(pts)
    WG_in = RegularGridInterpolator((g, g), wf["WGfilled"])(pts)

    # 2) Zernike fit of degree n on the large W_P
    # 2) Ajuste de Zernike de grado n sobre el W_P grande
    pr = (lambda f: progress(0.75 * f)) if progress else None
    coefs = zernikeFit(idx, rin, theta, WP_in, progress=pr)

    # 3) W_F on the same matrix, and W_F without piston and tilt ([M] Eq. 31)
    # 3) W_F en la misma matriz, y W_F sin pistón ni inclinación ([M] Ec. 31)
    pr = (lambda f: progress(0.75 + 0.25 * f)) if progress else None
    full = zernikeEval(coefs, idx, rin, theta, progress=pr)
    low = np.zeros_like(full)
    for j, Z in zernikeTerms(idx[:max(REMOVE_TERMS) + 1], rin, theta):
        if j in REMOVE_TERMS:
            low += coefs[j] * Z
    WF = np.full((h, w), np.nan)
    WFc = np.full((h, w), np.nan)
    WP = np.full((h, w), np.nan)
    WG = np.full((h, w), np.nan)
    WF[inside] = full
    WFc[inside] = full - low
    WP[inside] = WP_in
    WG[inside] = WG_in
    rms = float(np.sqrt(np.mean((full - WP_in) ** 2)))
    return {"WF": WF, "WFc": WFc, "WP": WP, "WG": WG, "coefs": coefs,
            "nPoints": int(inside.sum()), "fitRms": rms}


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
    # Matriz del tamaño de la imagen: cada punto guarda su diferencia ajustada en su
    # posición teórica; los píxeles sin datos son NaN
    M = np.full(shape, np.nan, dtype=np.float64)
    r = np.clip(np.round(df["y"].values).astype(int), 0, shape[0] - 1)
    c = np.clip(np.round(df["x"].values).astype(int), 0, shape[1] - 1)
    M[r, c] = df["diffAdj"].values
    return M


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# PROCESO PRINCIPAL
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
                            # índice del primer paso de cada etapa (A-G)

    bgrImg = cv2.imread(route)
    if bgrImg is None:
        raise RuntimeError(f"Cannot open image: {route}")
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    results.append({"title": "Input Image", "image": rgbImg.copy()})
    prog(2)

    # ── Initial orientation with the Hough transform ──
    # ── Orientación inicial con la transformada de Hough ──
    blurred = cv2.GaussianBlur(bgrImg, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=100, maxLineGap=20)
    angles = []
    if lines is not None:
        # Depending on the OpenCV version the shape is (N, 1, 4) or (N, 4)
        # Según la versión de OpenCV la forma es (N, 1, 4) o (N, 4)
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            angle = np.degrees(np.arctan2(y1 - y2, x1 - x2))
            # Endpoint order is arbitrary: normalize to (-180, 0] (vertical = -90)
            # El orden de los extremos es arbitrario: se normaliza a (-180, 0] (vertical = -90)
            angle = (angle % 180) - 180 if angle % 180 != 0 else 0.0
            angles.append(angle)
    stripeAngle = float(np.median(angles)) if angles else -90.0
    rgbImg = rotateImage(rgbImg, stripeAngle + 90)
    results.append({"title": "Rotated Image (Hough)", "image": rgbImg.copy(),
                    "text": [f"Detected lines: {len(angles)}",
                             f"Rotation applied: {stripeAngle + 90:.4f}°"]})
    prog(5)

    # Exit pupil given by the user: the circle inscribed in the square crop (its center
    # is the crop center, which the rotations about the crop center do not move)
    # Pupila de salida dada por el usuario: el círculo inscrito en el recorte cuadrado (su
    # centro es el centro del recorte, que las rotaciones alrededor de ese centro no mueven)
    hC, wC = bgrImg.shape[:2]
    cropCircle = ((wC - 1) / 2.0, (hC - 1) / 2.0, min(wC, hC) / 2.0) if pupilFromCrop else None

    # ── Vertical alignment verification (central stripes) ──
    # ── Verificación de la alineación vertical (franjas centrales) ──
    rgbImg, alignVis, alignText, alignCorrection = verifyVerticalAlignment(rgbImg, cropCircle)
    results.append({"title": "Vertical Alignment Verification", "image": alignVis,
                    "text": alignText})
    prog(9)

    # ── Exit pupil: center the image on it in a square frame ──
    # The pupil center becomes the rotation center, so the 90° rotated
    # Ronchigram overlaps exactly with the vertical one
    # Crop pixel -> working pixel: both rotations are about the same center, so they
    # add up; then the translation that centers the pupil. Used to build W_F.
    # ── Pupila de salida: centra la imagen en ella en un marco cuadrado ──
    # El centro de la pupila se vuelve el centro de rotación, así el Ronchigrama rotado
    # 90° se empalma exactamente con el vertical.
    # Píxel del recorte -> píxel de trabajo: las dos rotaciones son alrededor del mismo
    # centro, así que se suman; luego la traslación que centra la pupila. Se usa para W_F.
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
    T = np.float32([[1, 0, c0 - cx], [0, 1, c0 - cy]])
    cropToWork = np.vstack([Rtot, [0, 0, 1]])
    cropToWork[0, 2] += c0 - cx
    cropToWork[1, 2] += c0 - cy
    rgbImg = cv2.warpAffine(rgbImg, T, (side, side), flags=cv2.INTER_LINEAR,
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
    # ── Superficie óptica y escala ──
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
                             "Slopes: p = kappa * TA_x, q = kappa * TA_y with kappa = -1/z "
                             "(manuscript, Eq. 15)"]})
    prog(12)

    # ── Vertical Ronchigram ──
    # ── Ronchigrama vertical ──
    stageStart["B"] = len(results)
    V = analyzeRonchigram(rgbImg, pupil, "Vertical", False, results, prog, 12, 40, mapping,
                          editor=fringeEditor)

    # ── Horizontal Ronchigram: 90° rotation, NO inclination correction ──
    # ── Ronchigrama horizontal: rotación de 90°, SIN corrección de inclinación ──
    stageStart["C"] = len(results)
    rgbH = cv2.rotate(rgbImg, cv2.ROTATE_90_CLOCKWISE)
    results.append({"title": "Rotated 90° (Horizontal Ronchigram)", "image": rgbH.copy(),
                    "text": ["Rotation without inclination correction."]})
    # The horizontal Ronchigram is the vertical one rotated 90° about the pupil center;
    # analyzed on its transpose, that is the vertical image flipped upside down. So the
    # fringes confirmed in the vertical review are reused: row -> side - 1 - row.
    # El Ronchigrama horizontal es el vertical rotado 90° alrededor del centro de la
    # pupila; analizado sobre su transpuesta, es la imagen vertical volteada de arriba a
    # abajo. Así se reutilizan las franjas confirmadas en la revisión vertical:
    # fila -> side - 1 - fila.
    stripesH = [{"peak": s_["peak"], "anchors": [(side - 1 - r_, c_) for (r_, c_) in s_["anchors"]]}
                for s_ in V["stripes"]]
    H = analyzeRonchigram(rgbH, pupil, "Horizontal", True, results, prog, 40, 68, mapping,
                          stripesOverride=stripesH)

    # ── Grid ──
    # ── Malla ──
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
    # ── Refinamiento 1 (REFINE_K1 puntos entre líneas) ──
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
    # ── Refinamiento 2 (REFINE_K2 puntos entre líneas) — nubes de puntos finales ──
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
    # ── Salida final ──
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
                 "Positions = theoretical points",
                 "Values = adjusted differences (min = 0)",
                 "Download: Excel (3 sheets) + .npz with the 3 matrices"]})
    prog(93)

    # ── Extension of the point clouds to the whole exit pupil ──
    # ── Extensión de las nubes de puntos a toda la pupila de salida ──
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
    # ── Frente de onda (manuscrito [M] y suplementario [S]), en micrómetros ──
    stageStart["H"] = len(results)
    # The reconstruction uses the SIGNED differences (experimental - theoretical);
    # the maps adjusted to min = 0 are only for display ([M] Sec. 3B, [S] Sec. 3A)
    # La reconstrucción usa las diferencias CON SIGNO (experimental - teórica); los mapas
    # ajustados a mín = 0 son solo para mostrar ([M] Sec. 3B, [S] Sec. 3A)
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
    wf = computeWavefront(taX, taY, pupil, nZernike, mapping, mmPerPx,
                          progress=lambda f: prog(96 + 3 * f))

    cal = wf["calibration"]
    taText = [f"TA = Gamma * s * dx  (manuscript, Eq. 13), with Gamma = {cal['Gamma']:g} and "
              f"s = {1000 * cal['s_mm_per_px']:.4f} µm/px (pupil radius at the plane z from the "
              "ray tracing / pupil radius in pixels)",
              "Signed TA located at the nominal (theoretical) position of each fringe (manuscript, Eq. 14)",
              f"Continuous surface: least-squares fit with products of Legendre polynomials "
              f"P_m(x) P_n(y), m, n <= {TA_FIT_DEGREE}  (manuscript, Sec. 5B; Supplemental Document, Eq. S18), omega = 1"]
    results.append({"title": "TA Surface Fit - x",
                    "image": plotPupilMap(wf["taX"], "TA_x: continuous surface (Legendre, "
                                          f"m, n <= {TA_FIT_DEGREE})", "TA_x (µm)"),
                    "text": [srcText] + taText +
                            [f"Fit RMS: {wf['rmsX']:.4f} px"]})
    results.append({"title": "TA Surface Fit - y",
                    "image": plotPupilMap(wf["taY"], "TA_y: continuous surface (Legendre, "
                                          f"m, n <= {TA_FIT_DEGREE})", "TA_y (µm)"),
                    "text": ["TA_y from the pseudo-orthogonal representation (the same "
                             "Ronchigram rotated by pi/2, manuscript, Sec. 4A)"] + taText[:1] +
                            [f"Fit RMS: {wf['rmsY']:.4f} px"]})
    names = ", ".join(ZERNIKE_NAMES.get(wf["idx"][j], str(j)) for j in REMOVE_TERMS)
    lc = wf["lcurve"]["mu"].values
    results.append({"title": "Wavefront W_P - Variational (Eq. 19)",
                    "image": plotPupilMap(wf["WPd"], "W_P: variational reconstruction "
                                          "(piston and tilt removed)", "W_P (µm)"),
                    "text": [f"Slopes: p = kappa * TA_x, q = kappa * TA_y  (manuscript, Eq. 15), "
                             f"kappa = -1/z = {cal['kappa_per_mm']:.6g} 1/mm",
                             "W_P = argmin_u  ∫ omega |grad u - g|^2 dA + mu ∫ (lap u)^2 dA  "
                             "(manuscript, Eq. 19)",
                             "Discrete system (manuscript, Eqs. 22-26): centered differences, "
                             "5-point Laplacian, zero-mean constraint 1^T u = 0",
                             f"Grid: {WAVEFRONT_GRID} x {WAVEFRONT_GRID}, "
                             f"{wf['nNodes']} nodes in the pupil, omega = {WP_OMEGA:g}",
                             f"mu = {wf['mu']:.3g}: maximum curvature of the L-curve "
                             f"(Supplemental Document, Eqs. S50-S54), {len(lc)} candidates in "
                             f"[{lc[0]:.0e}, {lc[-1]:.0e}]",
                             f"Differential fidelity E_grad = {wf['eGrad']:.4f}  (Supplemental Document, Eq. S47)",
                             "Piston and tilt removed for display (manuscript, Eq. 31)",
                             f"P-V: {np.nanmax(wf['WPd']) - np.nanmin(wf['WPd']):.4f} µm, "
                             f"RMS: {np.nanstd(wf['WPd']):.4f} µm",
                             "The L-curve values are saved in the .mat file of the "
                             "last step"]})
    results.append({"title": "Wavefront W_G - Gauss-Legendre (Eqs. 27-30)",
                    "image": plotPupilMap(wf["WGd"], "W_G: Gauss-Legendre path integration "
                                          "(piston and tilt removed)", "W_G (µm)"),
                    "text": ["W_G(r_i) = C + 1/2 sum_m w_m g[r0 + (xi_m + 1)/2 (r_i - r0)] . "
                             "(r_i - r0)  (manuscript, Eqs. 27-30)",
                             f"Straight paths from the pupil center r0; N_G = {GL_NODES} "
                             "Gauss-Legendre nodes; slopes from the continuous TA surfaces",
                             "Piston: zero pupil average; piston and tilt removed for "
                             "display (manuscript, Eq. 31)",
                             "Complementary reconstruction from the same slope field; the "
                             "Zernike coefficients and W_F come from W_P",
                             f"P-V: {np.nanmax(wf['WGd']) - np.nanmin(wf['WGd']):.4f} µm, "
                             f"RMS: {np.nanstd(wf['WGd']):.4f} µm"]})
    pv = float(np.nanmax(wf["W"]) - np.nanmin(wf["W"]))
    rmsW = float(np.nanstd(wf["W"]))
    results.append({"title": "Wavefront W_f (final)",
                    "image": plotPupilMap(wf["W"], f"W_f: Zernike fit of W_P (n = {nZernike})",
                                          "W_f (µm)"),
                    "text": [f"Zernike degree n = {nZernike} "
                             f"({len(wf['idx'])} coefficients, OSA/ANSI order, normalized)",
                             f"Fitted on the {wf['nPointsFit']} pupil nodes of W_P "
                             f"(fit RMS {wf['fitRms']:.2e} µm)",
                             f"Removed: {names}  (manuscript, Eq. 31)",
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
        index=["Zernike degree n", "Coefficients", "Pupil nodes of W_P used",
               "P-V of W_f (µm)", "RMS of W_f (µm)", "Size of W_F"])
    summaryW.index.name = "Wavefront"
    results.append({
        "title": "Output: Zernike Coefficients and W_F",
        "table": summaryW,
        "wavefrontExport": {"wf": wf, "geom": {"cropShape": (cropH, cropW),
                                                "cropToWork": cropToWork,
                                                "pupil": pupil}},
        "data": {"wavefront": wf},
        "text": [f"1) W_P (variational reconstruction) is interpolated to the crop size "
                 f"({cropH} x {cropW}), with the same pupil as the crop",
                 f"2) Zernike fit up to n = {nZernike} (OSA/ANSI order) on all the "
                 "pupil pixels of that W_P",
                 "3) W_F = that fit on the same matrix, NaN outside the pupil",
                 f"Download: all the coefficients, the first {EXPORT_FIRST[0]} and the "
                 f"first {EXPORT_FIRST[1]} (.xlsx, .csv), and W_F (.mat, .csv)",
                 "W_F is computed when downloading; it may take a while"]})
    prog(100)

    # Tag every step with its stage (used by the GUI to group the tabs)
    # Marca cada paso con su etapa (la GUI lo usa para agrupar las pestañas)
    order = sorted(stageStart.items(), key=lambda kv: kv[1])
    for k, (stage, start) in enumerate(order):
        end = order[k + 1][1] if k + 1 < len(order) else len(results)
        for i in range(start, end):
            results[i]["stage"] = stage
    return results

 #------------------------------------------------------------------------------------------------------------------------------------------------------

# ── Colors ──
# ── Colores ──
# ══════════════════════════════════════════════════════════════════════════════
# GRAPHICAL USER INTERFACE
# INTERFAZ GRÁFICA
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# LANGUAGE (English / Spanish)
# IDIOMA (inglés / español)
# ══════════════════════════════════════════════════════════════════════════════
# The texts of the program are written in English. When Spanish is selected, every
# text shown to the user goes through T(), which translates it with the dictionary
# TEXTS_ES below:
#   - a text that is exactly a key is replaced by its translation;
#   - otherwise each key found inside the text is replaced (longest keys first), so
#     texts with numbers, e.g. "Pupil radius: 85.2 px", are translated too.
# Texts without a translation are shown in English. To add or correct a translation,
# edit TEXTS_ES. The texts are translated where they are shown:
#   - labels and buttons (classes Label and Button below), also when the language is
#     changed while the program is open;
#   - message boxes and file dialogs (wrappers "messagebox" and "filedialog");
#   - plots and tables (in figToArray). The plots are drawn when the pipeline runs,
#     so they keep the language that was selected at that moment.
# Files written to disk (Excel, .csv, .mat) keep English column names.
# Los textos del programa están escritos en inglés. Cuando se elige español, cada
# texto que ve el usuario pasa por T(), que lo traduce con el diccionario TEXTS_ES de
# abajo:
#   - un texto que es exactamente una clave se cambia por su traducción;
#   - si no, se cambia cada clave que aparezca dentro del texto (primero las más
#     largas), así también se traducen textos con números, p. ej. "Pupil radius: 85.2 px".
# Los textos sin traducción se muestran en inglés. Para agregar o corregir una
# traducción, edita TEXTS_ES. Los textos se traducen donde se muestran:
#   - etiquetas y botones (clases Label y Button de abajo), también cuando se cambia el
#     idioma con el programa abierto;
#   - cuadros de mensaje y diálogos de archivos (envolturas "messagebox" y "filedialog");
#   - gráficas y tablas (en figToArray). Las gráficas se dibujan cuando corre el
#     proceso, así que se quedan en el idioma que estaba elegido en ese momento.
# Los archivos que se guardan (Excel, .csv, .mat) conservan los nombres de columnas en inglés.
LANGUAGES = {"en": "English", "es": "Español"}
LANG = "en"                # language at start: "en" or "es"
                           # idioma al iniciar: "en" o "es"

TEXTS_ES = {
    # ── Window, menu and header ──
    # ── Ventana, menú y encabezado ──
    "Scientific Pipeline": "Procesamiento científico",
    "File": "Archivo",
    "Open image…": "Abrir imagen…",
    "Back to input": "Volver a la entrada",
    "Save current output…": "Guardar el resultado actual…",
    "Exit": "Salir",
    "Language": "Idioma",
    "Pipeline status": "Estado del proceso",
    "Waiting for an image": "Esperando una imagen",
    "Select a Ronchigram image to start (File ▸ Open image…).":
        "Selecciona una imagen de Ronchigrama para empezar (Archivo ▸ Abrir imagen…).",
    "Select a Ronchigram image to start": "Selecciona una imagen de Ronchigrama para empezar",

    # ── Stages ──
    # ── Etapas ──
    "A. Preparation": "A. Preparación",
    "B. Vertical Ronchigram": "B. Ronchigrama vertical",
    "C. Horizontal Ronchigram": "C. Ronchigrama horizontal",
    "D. Grid": "D. Malla",
    "E. Refinement 1": "E. Refinamiento 1",
    "F. Refinement 2": "F. Refinamiento 2",
    "G. Full pupil": "G. Pupila completa",
    "H. Wavefront": "H. Frente de onda",
    "Preparation": "Preparación",
    "Vertical Ronchigram": "Ronchigrama vertical",
    "Horizontal Ronchigram": "Ronchigrama horizontal",
    "Grid": "Malla",
    "Refinement 1": "Refinamiento 1",
    "Refinement 2": "Refinamiento 2",
    "Full pupil": "Pupila completa",
    "Wavefront": "Frente de onda",
    "Stage ": "Etapa ",

    # ── Input view ──
    # ── Vista de entrada ──
    "Input Ronchigram": "Ronchigrama de entrada",
    "Input and crop": "Entrada y recorte",
    "Select image…": "Seleccionar imagen…",
    "✂  New crop": "✂  Nuevo recorte",
    "Clear crop": "Quitar recorte",
    "Drag to draw a square and fit its dashed circle to the exit pupil (it is used as "
    "the pupil). Drag inside to move it, drag a corner to resize it, mouse wheel to "
    "zoom, arrow keys to nudge (Shift = 10 px).":
        "Arrastra para dibujar un cuadrado y ajusta su círculo punteado a la pupila de "
        "salida (se usa como la pupila). Arrastra por dentro para moverlo, arrastra una "
        "esquina para cambiar su tamaño, rueda del ratón para hacer zoom, flechas para "
        "moverlo poco a poco (Shift = 10 px).",
    "File:": "Archivo:",
    "Size:": "Tamaño:",
    "Crop:": "Recorte:",
    "No crop": "Sin recorte",
    "none (full image)": "ninguno (imagen completa)",
    " px at (": " px en (",
    "Optical surface": "Superficie óptica",
    "Radius of curvature R (mm)": "Radio de curvatura R (mm)",
    "Focal length f (mm)": "Distancia focal f (mm)",
    "Diameter D (mm)": "Diámetro D (mm)",
    "Conic constant K": "Constante cónica K",
    "Distance z (mm)": "Distancia z (mm)",
    "Conic K": "Cónica K",
    "z: distance from the surface vertex to the plane where the Ronchigram was "
    "recorded. K: 0 sphere, -1 paraboloid.":
        "z: distancia del vértice de la superficie al plano donde se registró el "
        "Ronchigrama. K: 0 esfera, -1 paraboloide.",
    "Zernike degree n:": "Grado de Zernike n:",
    "(article: 50)": "(artículo: 50)",
    "Run pipeline": "Ejecutar proceso",
    "Image files": "Imágenes",
    "MATLAB file": "Archivo de MATLAB",
    "Image loaded. Press “New crop” and drag a square around the diffraction order to "
    "analyze.":
        "Imagen cargada. Presiona “Nuevo recorte” y arrastra un cuadrado alrededor del "
        "orden de difracción que quieres analizar.",
    "Drag on the image to draw a new square crop.":
        "Arrastra sobre la imagen para dibujar un nuevo recorte cuadrado.",
    "Adjust the crop: drag inside to move it, drag a corner to resize it, mouse wheel "
    "to zoom, arrow keys to nudge (Shift = 10 px).":
        "Ajusta el recorte: arrastra por dentro para moverlo, arrastra una esquina para "
        "cambiar su tamaño, rueda del ratón para hacer zoom, flechas para moverlo poco "
        "a poco (Shift = 10 px).",
    "Zernike degree": "Grado de Zernike",
    "The Zernike degree n must be an integer between 2 and 100.":
        "El grado de Zernike n debe ser un entero entre 2 y 100.",
    " must be a number.": " debe ser un número.",
    "Confirm crop": "Confirmar recorte",
    "Is the crop correct?": "¿El recorte es correcto?",
    "The dashed circle is used as the exit pupil: its center and radius must match "
    "the pupil of the Ronchigram.":
        "El círculo punteado se usa como pupila de salida: su centro y su radio deben "
        "coincidir con la pupila del Ronchigrama.",
    "Position: ": "Posición: ",
    "Size: ": "Tamaño: ",
    " px (pupil radius ": " px (radio de la pupila ",
    "Choose “No” to keep adjusting it.": "Elige “No” para seguir ajustándolo.",
    "No crop has been selected.\nProcess the full image?":
        "No se seleccionó ningún recorte.\n¿Procesar la imagen completa?",
    "Adjust the crop and press “Run pipeline” again.":
        "Ajusta el recorte y presiona “Ejecutar proceso” otra vez.",

    # ── Running ──
    # ── Ejecución ──
    "Running…": "Ejecutando…",
    "Running ": "Ejecutando ",
    "Processing time: ": "Tiempo de procesamiento: ",
    " seconds": " segundos",
    "Pipeline finished · ": "Proceso terminado · ",
    " steps · ": " pasos · ",
    " failed": " falló",
    "Pipeline failed": "El proceso falló",
    "Pipeline cancelled": "Proceso cancelado",
    "Pipeline cancelled in the fringe review. Adjust and run it again.":
        "Proceso cancelado en la revisión de franjas. Ajusta y ejecútalo otra vez.",
    "Pipeline cancelled in the fringe review.":
        "Proceso cancelado en la revisión de franjas.",
    "Pipeline cancelled by the user in the fringe review.":
        "Proceso cancelado por el usuario en la revisión de franjas.",
    "Error: ": "Error: ",

    # ── Fringe review ──
    # ── Revisión de franjas ──
    "Fringe review - vertical Ronchigram (aligned image)":
        "Revisión de franjas - Ronchigrama vertical (imagen alineada)",
    "Fringes": "Franjas",
    "Select a fringe and click on the image to add anchor points. Drag a point to move "
    "it; right-click (or Ctrl + click) to delete it. The points stay where you put "
    "them; the line of each fringe is fitted to its anchor points after you confirm.":
        "Selecciona una franja y haz clic en la imagen para agregar puntos de anclaje. "
        "Arrastra un punto para moverlo; clic derecho (o Ctrl + clic) para borrarlo. "
        "Los puntos se quedan donde los pones; la línea de cada franja se ajusta a sus "
        "puntos de anclaje cuando confirmas.",
    "No fringes. Press “New fringe” to add one.":
        "No hay franjas. Presiona “Nueva franja” para agregar una.",
    "Fringe ": "Franja ",
    " points": " puntos",
    "Detected automatically: ": "Detectadas automáticamente: ",
    "   ·   Now: ": "   ·   Ahora: ",
    " with at least ": " con al menos ",
    "+  New fringe": "+  Nueva franja",
    "Delete fringe": "Borrar franja",
    "Clear points": "Borrar puntos",
    "Delete all": "Borrar todo",
    "Reset to automatic detection": "Volver a la detección automática",
    "✓  Confirm fringes and continue": "✓  Confirmar franjas y continuar",
    "Cancel": "Cancelar",
    "Waiting for the fringe review…": "Esperando la revisión de franjas…",
    "Review the fringes: select one on the right, click on the image to add anchor "
    "points, drag them to move them, right-click to delete them.":
        "Revisa las franjas: selecciona una a la derecha, haz clic en la imagen para "
        "agregar puntos de anclaje, arrástralos para moverlos, clic derecho para "
        "borrarlos.",
    "Select a fringe on the right (or press “New fringe”) before adding points.":
        "Selecciona una franja a la derecha (o presiona “Nueva franja”) antes de "
        "agregar puntos.",
    "New fringe: click along its center on the image to add at least ":
        "Nueva franja: haz clic a lo largo de su centro en la imagen para agregar al "
        "menos ",
    " anchor points (from top to bottom).": " puntos de anclaje (de arriba hacia abajo).",
    "Delete all fringes": "Borrar todas las franjas",
    "Delete all the fringes and their anchor points?":
        "¿Borrar todas las franjas y sus puntos de anclaje?",
    "Fringe review": "Revisión de franjas",
    "Fringe(s) ": "Franja(s) ",
    " have fewer than ": " tienen menos de ",
    " anchor points. Add points or delete the fringe.":
        " puntos de anclaje. Agrega puntos o borra la franja.",
    "At least 2 fringes are needed to continue.":
        "Se necesitan al menos 2 franjas para continuar.",
    " fringes confirmed. Running…": " franjas confirmadas. Ejecutando…",

    # ── Results view ──
    # ── Vista de resultados ──
    "↩  Input": "↩  Entrada",
    "Step details": "Detalles del paso",
    "⬇  Download": "⬇  Descargar",
    "⬇  Download Zernike coefficients + W_F": "⬇  Descargar coeficientes de Zernike + W_F",
    "⬇  Download Excel": "⬇  Descargar Excel",
    " + matrices": " + matrices",
    "⬇  Download image": "⬇  Descargar imagen",
    "Step ": "Paso ",
    "No image for this step": "No hay imagen en este paso",
    "No additional info.": "Sin información adicional.",
    "Select another image or adjust the crop and run the pipeline again.":
        "Selecciona otra imagen o ajusta el recorte y ejecuta el proceso otra vez.",

    # ── Saving ──
    # ── Guardado ──
    "Save Zernike coefficients and W_F": "Guardar los coeficientes de Zernike y W_F",
    " and fitting Zernike n = ": " y ajustando Zernike n = ",
    " (NaN outside the pupil)": " (NaN fuera de la pupila)",
    "Zernike fit on ": "Ajuste de Zernike sobre ",
    "Saved to:": "Guardado en:",
    "Not saved:": "No se guardó:",
    "Saved with errors": "Guardado con errores",
    "Saved": "Guardado",
    "Saved ": "Se guardaron ",
    " files; ": " archivos; ",
    " could not be saved.": " no se pudieron guardar.",
    "W_F and Zernike coefficients saved.": "W_F y coeficientes de Zernike guardados.",
    "No file was saved.": "No se guardó ningún archivo.",
    "Download": "Descargar",
    "No file to download for this step.": "No hay archivo para descargar en este paso.",
    "Image saved to:": "Imagen guardada en:",
    "permission denied (is the file open in Excel or another program?)":
        "permiso denegado (¿el archivo está abierto en Excel u otro programa?)",
    "the package openpyxl is not available, so .xlsx files cannot be written (the .csv "
    "files have the same data)":
        "el paquete openpyxl no está disponible, así que no se pueden escribir archivos "
        ".xlsx (los archivos .csv tienen los mismos datos)",
    "the image could not be encoded": "no se pudo codificar la imagen",

    # ── Errors of the pipeline ──
    # ── Errores del proceso ──
    "Could not detect the exit pupil.": "No se pudo detectar la pupila de salida.",
    "No stripes detected (empty intensity profile).":
        "No se detectaron franjas (perfil de intensidad vacío).",
    "Only ": "Solo ",
    " stripe(s) detected. At least 2 are needed to estimate the spacing. Check the "
    "image, the crop or the binarization threshold.":
        " franja(s) detectada(s). Se necesitan al menos 2 para estimar la separación. "
        "Revisa la imagen, el recorte o el umbral de binarización.",
    " stripe(s) remain after peak filtering.":
        " franja(s) quedan después de filtrar los picos.",
    "R, f, D and z must be positive.": "R, f, D y z deben ser positivos.",
    "The conic surface is not defined over the whole diameter (check R, K and D).":
        "La superficie cónica no está definida en todo el diámetro (revisa R, K y D).",
    "Total internal reflection at the surface (check R, f, K, D).":
        "Reflexión interna total en la superficie (revisa R, f, K, D).",
    "Total internal reflection at the back face (check R, f, K, D).":
        "Reflexión interna total en la cara posterior (revisa R, f, K, D).",
    "z places the camera at the focus: the pupil has zero size.":
        "z pone la cámara en el foco: la pupila tiene tamaño cero.",
    "With these parameters the camera plane z is inside the caustic (rays cross "
    "before reaching it). Check f, K and z.":
        "Con estos parámetros el plano de la cámara z está dentro de la cáustica (los "
        "rayos se cruzan antes de llegar). Revisa f, K y z.",
    "No corresponding experimental/theoretical points were found.":
        "No se encontraron puntos experimentales/teóricos correspondientes.",
    "No intersections found between vertical and horizontal lines.":
        "No se encontraron intersecciones entre las líneas verticales y horizontales.",
    "Cannot open image: ": "No se puede abrir la imagen: ",
    ": at least 2 fringes (with at least ": ": se necesitan al menos 2 franjas (con al menos ",
    " anchor points each) are needed.": " puntos de anclaje cada una).",
    "cancelled by the user": "cancelado por el usuario",

    # ── Step titles ──
    # ── Títulos de los pasos ──
    "Input Image": "Imagen de entrada",
    "Rotated Image (Hough)": "Imagen rotada (Hough)",
    "Vertical Alignment Verification": "Verificación de la alineación vertical",
    "Exit Pupil and Centering": "Pupila de salida y centrado",
    "Optical Surface Parameters": "Parámetros de la superficie óptica",
    "Rotated 90° (Horizontal Ronchigram)": "Rotada 90° (Ronchigrama horizontal)",
    "Binary Image": "Imagen binaria",
    "Connected Components Filtering": "Filtrado de componentes conectados",
    "Closed Stripes inside Pupil": "Franjas cerradas dentro de la pupila",
    "Intensity Profile": "Perfil de intensidad",
    "Peaks Detection": "Detección de picos",
    "Anchor Points": "Puntos de anclaje",
    "Fringe Review (confirmed by the user)": "Revisión de franjas (confirmadas por el usuario)",
    "Experimental Centroid Lines": "Líneas de centroides experimentales",
    "Theoretical Ronchigram": "Ronchigrama teórico",
    "Theoretical Centroid Lines": "Líneas de centroides teóricas",
    "Experimental vs Theoretical Correspondence": "Correspondencia experimental vs teórica",
    "Difference Summary": "Resumen de diferencias",
    "Difference Matrix - Grid": "Matriz de diferencias - Malla",
    "Difference Matrix": "Matriz de diferencias",
    "Experimental Grid": "Malla experimental",
    "Theoretical Grid": "Malla teórica",
    "Grid Intersections: Experimental vs Theoretical":
        "Intersecciones de la malla: experimental vs teórica",
    "Grid Difference Summary": "Resumen de diferencias de la malla",
    "Refinement 1 - Difference ": "Refinamiento 1 - Diferencia ",
    "Refinement 2 - Difference ": "Refinamiento 2 - Diferencia ",
    "Refinement 1 (": "Refinamiento 1 (",
    " pts) - ": " pts) - ",
    "Point Cloud (Refinement 2) - ": "Nube de puntos (refinamiento 2) - ",
    "Point Cloud 3D - ": "Nube de puntos 3D - ",
    "Point Cloud - ": "Nube de puntos - ",
    "Output: Point Clouds (Final Matrices)": "Salida: nubes de puntos (matrices finales)",
    "Full Pupil - Difference ": "Pupila completa - Diferencia ",
    "Full Pupil 3D - ": "Pupila completa 3D - ",
    "Full Pupil - ": "Pupila completa - ",
    "Output: Full Pupil Matrices": "Salida: matrices de pupila completa",
    "Wavefront W_f (final)": "Frente de onda W_f (final)",
    "Wavefront W_f 3D": "Frente de onda W_f 3D",
    "Output: Zernike Coefficients and W_F": "Salida: coeficientes de Zernike y W_F",

    # ── Step details ──
    # ── Detalles de los pasos ──
    "Iteration ": "Iteración ",
    ": residual angle = ": ": ángulo residual = ",
    "Threshold: ": "Umbral: ",
    "Total correction applied: ": "Corrección total aplicada: ",
    "Central stripes (straight line fitted to all their anchor points):":
        "Franjas centrales (recta ajustada a todos sus puntos de anclaje):",
    "Detected lines: ": "Líneas detectadas: ",
    "Rotation applied: ": "Rotación aplicada: ",
    "Pupil: ": "Pupila: ",
    "circle of the crop (fitted by the user)": "círculo del recorte (ajustado por el usuario)",
    "automatic detection (no crop)": "detección automática (sin recorte)",
    "Pupil center (original): (": "Centro de la pupila (original): (",
    "Pupil radius: ": "Radio de la pupila: ",
    "Working image: ": "Imagen de trabajo: ",
    "Value": "Valor",
    "Diameter D": "Diámetro D",
    "Focal length f": "Distancia focal f",
    "Radius of curvature R": "Radio de curvatura R",
    "Conic constant": "Constante cónica",
    "Distance z (vertex to Ronchigram plane)": "Distancia z (vértice al plano del Ronchigrama)",
    "Pupil radius at the camera plane": "Radio de la pupila en el plano de la cámara",
    "Refractive index n = 1 + R/f": "Índice de refracción n = 1 + R/f",
    "Scale": "Escala",
    "Ideal stripe bending (max.)": "Curvatura ideal de las franjas (máx.)",
    " % of the pupil radius": " % del radio de la pupila",
    "Ideal Ronchigram: grating lines traced exactly through the surface (conic R, K; "
    "n from R/f) to the plane z":
        "Ronchigrama ideal: líneas de la rejilla trazadas exactamente a través de la "
        "superficie (cónica R, K; n a partir de R/f) hasta el plano z",
    "Scale = pupil radius at z / pupil radius in px":
        "Escala = radio de la pupila en z / radio de la pupila en px",
    "Rotation without inclination correction.": "Rotación sin corrección de inclinación.",
    "Initial detection: ": "Detección inicial: ",
    " peaks": " picos",
    "Discarded fragments: ": "Fragmentos descartados: ",
    "Final stripes: ": "Franjas finales: ",
    "Median gap (spacing): ": "Separación mediana: ",
    "Automatic detection: ": "Detección automática: ",
    "Anchor points: the fringes confirmed in the vertical Ronchigram, rotated 90° "
    "(same fringes)":
        "Puntos de anclaje: las franjas confirmadas en el Ronchigrama vertical, rotadas "
        "90° (las mismas franjas)",
    "Half-width of search window: ": "Semiancho de la ventana de búsqueda: ",
    "Automatically detected stripes: ": "Franjas detectadas automáticamente: ",
    "Automatically detected fringes: ": "Franjas detectadas automáticamente: ",
    "Fringes confirmed by the user: ": "Franjas confirmadas por el usuario: ",
    "Anchor points shown: the ones confirmed by the user":
        "Puntos de anclaje mostrados: los confirmados por el usuario",
    "Confirmed fringes: ": "Franjas confirmadas: ",
    "Anchor points per fringe: ": "Puntos de anclaje por franja: ",
    "Point spacing along each line: ": "Separación de los puntos a lo largo de cada línea: ",
    "Max. points allowed (size/": "Máx. de puntos permitidos (tamaño/",
    "Points per line: min ": "Puntos por línea: mín ",
    ", max ": ", máx ",
    "Measured stripe width: ": "Ancho de franja medido: ",
    "Ideal surface: ": "Superficie ideal: ",
    "Spacing at the center: ": "Separación en el centro: ",
    "Stripe width (= experimental): ": "Ancho de franja (= experimental): ",
    "Reference (central) stripe at: ": "Franja de referencia (central) en: ",
    "Theoretical stripes: ": "Franjas teóricas: ",
    "Recognized theoretical lines: ": "Líneas teóricas reconocidas: ",
    "Corresponding pairs: ": "Pares correspondientes: ",
    "  Exp. line ": "  Línea exp. ",
    " -> Theo. line ": " -> línea teór. ",
    "Points": "Puntos",
    "Min (px)": "Mín (px)",
    "Max (px)": "Máx (px)",
    "Mean (px)": "Media (px)",
    "Std (px)": "Desv. est. (px)",
    "Mean": "Media",
    "Min": "Mín",
    "Max": "Máx",
    "Std": "Desv. est.",
    "Compared points: ": "Puntos comparados: ",
    "Difference = experimental - theoretical": "Diferencia = experimental - teórica",
    "Raw min: ": "Mín sin ajustar: ",
    "Raw max: ": "Máx sin ajustar: ",
    "Adjusted so that min = 0": "Ajustada para que mín = 0",
    "Points: ": "Puntos: ",
    "Adjusted min: ": "Mín ajustado: ",
    "Adjusted max: ": "Máx ajustado: ",
    "Corresponding intersections: ": "Intersecciones correspondientes: ",
    "Black: theoretical, purple: experimental, orange: displacement":
        "Negro: teórica, morado: experimental, naranja: desplazamiento",
    "Intersections: ": "Intersecciones: ",
    "Difference = |P_exp - P_theo| (Euclidean)": "Diferencia = |P_exp - P_teo| (euclidiana)",
    "dx, dy kept with sign in the Excel file": "dx, dy se guardan con signo en el Excel",
    "Original points: ": "Puntos originales: ",
    "Refined points: ": "Puntos refinados: ",
    "Points inserted between lines: requested ": "Puntos insertados entre líneas: pedidos ",
    ", applied ": ", aplicados ",
    "Interpolation: ": "Interpolación: ",
    "Input points: ": "Puntos de entrada: ",
    "Limited by Nyquist: points never closer than ":
        "Limitado por Nyquist: los puntos nunca están a menos de ",
    "Orig. pts": "Pts orig.",
    "Ref.1 pts": "Pts ref.1",
    "Ref.2 pts": "Pts ref.2",
    "Matrix size: ": "Tamaño de la matriz: ",
    " (float64, NaN = no data)": " (float64, NaN = sin datos)",
    " (float64, NaN = outside pupil)": " (float64, NaN = fuera de la pupila)",
    "Positions = theoretical points": "Posiciones = puntos teóricos",
    "Values = adjusted differences (min = 0)": "Valores = diferencias ajustadas (mín = 0)",
    "Download: Excel (3 sheets) + .npz with the 3 matrices":
        "Descarga: Excel (3 hojas) + .npz con las 3 matrices",
    "Measured points (kept): ": "Puntos medidos (conservados): ",
    "Extended points: ": "Puntos extendidos: ",
    "% of the pupil)": "% de la pupila)",
    "Extension: polynomial of degree ": "Extensión: polinomio de grado ",
    " + residual decaying in ": " + residuo que decae en ",
    "Polynomial fit RMS on the data: ": "RMS del ajuste polinomial sobre los datos: ",
    "Cyan line: border of the measured region": "Línea cian: borde de la región medida",
    "Values outside it are extrapolated, not measured":
        "Los valores fuera de ella son extrapolados, no medidos",
    "Measured": "Medidos",
    "Extended": "Extendidos",
    "Total": "Total",
    "% ext.": "% ext.",
    "Fit RMS": "RMS del ajuste",
    "Column 'source': data = measured/refined, extended = extrapolated":
        "Columna 'source': data = medido/refinado, extended = extrapolado",
    "TA_x, TA_y = signed dx, dy of the grid": "TA_x, TA_y = dx, dy con signo de la malla",
    "TA_x = vertical matrix, TA_y = horizontal matrix (signed)":
        "TA_x = matriz vertical, TA_y = matriz horizontal (con signo)",
    "Polynomial degree: ": "Grado del polinomio: ",
    "Fit RMS: ": "RMS del ajuste: ",
    "Zernike degree n = ": "Grado de Zernike n = ",
    "Removed: ": "Quitados: ",
    "Coefficients": "Coeficientes",
    "Zernike degree n": "Grado de Zernike n",
    "P-V of W_f (µm)": "P-V de W_f (µm)",
    "RMS of W_f (µm)": "RMS de W_f (µm)",
    "Size of W_F": "Tamaño de W_F",
    "), with the same pupil as the crop": "), con la misma pupila que el recorte",
    "2) Zernike fit up to n = ": "2) Ajuste de Zernike hasta n = ",
    "3) W_F = that fit on the same matrix, NaN outside the pupil":
        "3) W_F = ese ajuste en la misma matriz, NaN fuera de la pupila",
    "W_F is computed when downloading; it may take a while":
        "W_F se calcula al descargar; puede tardar un poco",

    # ── Wavefront: TA, W_P, W_G (manuscript and Supplemental Document) ──
    # ── Frente de onda: TA, W_P, W_G (manuscrito y documento suplementario) ──
    "Interpolating W_P to ": "Interpolando W_P a ",
    " pixels of W_P": " píxeles de W_P",
    "TA Surface Fit - x": "Ajuste de la superficie TA - x",
    "TA Surface Fit - y": "Ajuste de la superficie TA - y",
    "Wavefront W_P - Variational (Eq. 19)": "Frente de onda W_P - Variacional (Ec. 19)",
    "Wavefront W_G - Gauss-Legendre (Eqs. 27-30)": "Frente de onda W_G - Gauss-Legendre (Ecs. 27-30)",
    "Slopes: p = kappa * TA_x, q = kappa * TA_y with kappa = -1/z (manuscript, Eq. 15)":
        "Pendientes: p = kappa * TA_x, q = kappa * TA_y con kappa = -1/z (manuscrito, Ec. 15)",
    "TA = Gamma * s * dx  (manuscript, Eq. 13), with Gamma = ":
        "TA = Gamma * s * dx  (manuscrito, Ec. 13), con Gamma = ",
    " and s = ": " y s = ",
    " µm/px (pupil radius at the plane z from the ray tracing / pupil radius in pixels)":
        " µm/px (radio de la pupila en el plano z según el trazo de rayos / radio de la "
        "pupila en píxeles)",
    "Signed TA located at the nominal (theoretical) position of each fringe (manuscript, Eq. 14)":
        "TA con signo, ubicada en la posición nominal (teórica) de cada franja "
        "(manuscrito, Ec. 14)",
    "Continuous surface: least-squares fit with products of Legendre polynomials P_m(x) "
    "P_n(y), m, n <= ":
        "Superficie continua: ajuste por mínimos cuadrados con productos de polinomios de "
        "Legendre P_m(x) P_n(y), m, n <= ",
    "  (manuscript, Sec. 5B; Supplemental Document, Eq. S18), omega = 1":
        "  (manuscrito, Sec. 5B; documento suplementario, Ec. S18), omega = 1",
    "TA_y from the pseudo-orthogonal representation (the same Ronchigram rotated by pi/2, "
    "manuscript, Sec. 4A)":
        "TA_y de la representación pseudo-ortogonal (el mismo Ronchigrama rotado pi/2, "
        "manuscrito, Sec. 4A)",
    "TA_x: continuous surface (Legendre, ": "TA_x: superficie continua (Legendre, ",
    "TA_y: continuous surface (Legendre, ": "TA_y: superficie continua (Legendre, ",
    "Slopes: p = kappa * TA_x, q = kappa * TA_y  (manuscript, Eq. 15), kappa = -1/z = ":
        "Pendientes: p = kappa * TA_x, q = kappa * TA_y  (manuscrito, Ec. 15), kappa = -1/z = ",
    "W_P = argmin_u  ∫ omega |grad u - g|^2 dA + mu ∫ (lap u)^2 dA  (manuscript, Eq. 19)":
        "W_P = argmin_u  ∫ omega |grad u - g|^2 dA + mu ∫ (lap u)^2 dA  (manuscrito, Ec. 19)",
    "Discrete system (manuscript, Eqs. 22-26): centered differences, 5-point Laplacian, "
    "zero-mean constraint 1^T u = 0":
        "Sistema discreto (manuscrito, Ecs. 22-26): diferencias centradas, laplaciano de "
        "5 puntos, restricción de media cero 1^T u = 0",
    "Grid: ": "Malla: ",
    " nodes in the pupil, omega = ": " nodos en la pupila, omega = ",
    ": maximum curvature of the L-curve (Supplemental Document, Eqs. S50-S54), ":
        ": máxima curvatura de la curva L (documento suplementario, Ecs. S50-S54), ",
    " candidates in [": " candidatos en [",
    "Differential fidelity E_grad = ": "Fidelidad diferencial E_grad = ",
    "  (Supplemental Document, Eq. S47)": "  (documento suplementario, Ec. S47)",
    "Piston and tilt removed for display (manuscript, Eq. 31)":
        "Pistón e inclinación quitados para mostrar (manuscrito, Ec. 31)",
    "The L-curve values are saved in the .mat file of the last step":
        "Los valores de la curva L se guardan en el archivo .mat del último paso",
    "W_P: variational reconstruction (piston and tilt removed)":
        "W_P: reconstrucción variacional (sin pistón ni inclinación)",
    "W_G: Gauss-Legendre path integration (piston and tilt removed)":
        "W_G: integración de Gauss-Legendre por trayectorias (sin pistón ni inclinación)",
    "  (manuscript, Eqs. 27-30)": "  (manuscrito, Ecs. 27-30)",
    "Straight paths from the pupil center r0; N_G = ":
        "Trayectorias rectas desde el centro de la pupila r0; N_G = ",
    " Gauss-Legendre nodes; slopes from the continuous TA surfaces":
        " nodos de Gauss-Legendre; pendientes de las superficies continuas de TA",
    "Piston: zero pupil average; piston and tilt removed for display (manuscript, Eq. 31)":
        "Pistón: promedio cero en la pupila; pistón e inclinación quitados para mostrar "
        "(manuscrito, Ec. 31)",
    "Complementary reconstruction from the same slope field; the Zernike coefficients and "
    "W_F come from W_P":
        "Reconstrucción complementaria a partir del mismo campo de pendientes; los "
        "coeficientes de Zernike y W_F salen de W_P",
    "W_f: Zernike fit of W_P (n = ": "W_f: ajuste de Zernike de W_P (n = ",
    " coefficients, OSA/ANSI order, normalized)": " coeficientes, orden OSA/ANSI, normalizados)",
    "Fitted on the ": "Ajustado sobre los ",
    " pupil nodes of W_P (fit RMS ": " nodos de la pupila de W_P (RMS del ajuste ",
    "  (manuscript, Eq. 31)": "  (manuscrito, Ec. 31)",
    "Pupil nodes of W_P used": "Nodos de la pupila de W_P usados",
    "1) W_P (variational reconstruction) is interpolated to the crop size (":
        "1) W_P (reconstrucción variacional) se interpola al tamaño del recorte (",
    " (OSA/ANSI order) on all the pupil pixels of that W_P":
        " (orden OSA/ANSI) sobre todos los píxeles de la pupila de ese W_P",
    "Download: all the coefficients, the first ": "Descarga: todos los coeficientes, los primeros ",
    " and the first ": " y los primeros ",
    " (.xlsx, .csv), and W_F (.mat, .csv)": " (.xlsx, .csv), y W_F (.mat, .csv)",

    # ── Plots ──
    # ── Gráficas ──
    "TA, adjusted to min = 0 (px)": "TA, ajustada a mín = 0 (px)",
    "Measured region": "Región medida",
    "Original Profile": "Perfil original",
    "Smoothed Profile": "Perfil suavizado",
    "Smoothed profile": "Perfil suavizado",
    "Average Intensity": "Intensidad promedio",
    "Initial detection (": "Detección inicial (",
    "Final stripes (": "Franjas finales (",
    "Discarded fragments (": "Fragmentos descartados (",
    "Row (px)": "Fila (px)",
    "Column (px)": "Columna (px)",
    "x / r (normalized)": "x / r (normalizado)",
    "y / r (normalized)": "y / r (normalizado)",
    "Coefficient (µm)": "Coeficiente (µm)",
    "Zernike coefficients up to n = ": "Coeficientes de Zernike hasta n = ",
    " (piston, tilts and defocus removed)": " (sin pistón, inclinaciones ni desenfoque)",

    # ── Zernike names ──
    # ── Nombres de Zernike ──
    "Piston": "Pistón",
    "Tilt X": "Inclinación X",
    "Tilt Y": "Inclinación Y",
    "Defocus": "Desenfoque",
    "Astigmatism 0°": "Astigmatismo 0°",
    "Astigmatism 45°": "Astigmatismo 45°",
    "Coma X": "Coma X",
    "Coma Y": "Coma Y",
    "Trefoil X": "Trébol X",
    "Trefoil Y": "Trébol Y",
    "Spherical": "Esférica",
    "Sec. astigmatism 0°": "Astigmatismo sec. 0°",
    "Sec. astigmatism 45°": "Astigmatismo sec. 45°",
    "Tetrafoil X": "Tetrafolio X",
    "Tetrafoil Y": "Tetrafolio Y",
    "Sec. coma X": "Coma sec. X",
    "Sec. coma Y": "Coma sec. Y",
    "Sec. spherical": "Esférica sec.",
}

_trRegex = None
import re
_pathRegex = re.compile(r"(?:(?<=\s)|^)(?:[A-Za-z]:[\\/]|/)\S")


def T(text):
    # Text in the selected language (see the comment at the top of this section)
    # Texto en el idioma seleccionado (ver el comentario al inicio de esta sección)
    global _trRegex
    if LANG == "en" or not isinstance(text, str) or not text:
        return text
    if text in TEXTS_ES:
        return TEXTS_ES[text]
    if "\n" in text:
        return "\n".join(T(line) for line in text.split("\n"))
    # A file path (C:\..., C:/... or /Users/...) is never translated: only the text
    # before it is
    # Una ruta de archivo (C:\..., C:/... o /Users/...) nunca se traduce: solo el texto
    # que va antes de ella
    m = _pathRegex.search(text)
    if m:
        return T(text[:m.start()]) + text[m.start():]
    if _trRegex is None:
        parts = []
        for key in sorted(TEXTS_ES, key=len, reverse=True):
            p = re.escape(key)
            # Keys that start/end with a letter or digit only match whole words
            # Las claves que empiezan/terminan con letra o dígito solo coinciden con palabras completas
            if key[0].isalnum():
                p = r"(?<![\w])" + p
            if key[-1].isalnum():
                p = p + r"(?![\w])"
            parts.append(p)
        _trRegex = re.compile("|".join(parts))
    return _trRegex.sub(lambda m: TEXTS_ES[m.group(0)], text)


class Label(tk.Label):
    # tk.Label whose text is written in English and shown in the selected language.
    # The English text is kept, so the label can be translated again when the user
    # changes the language.
    # tk.Label cuyo texto se escribe en inglés y se muestra en el idioma seleccionado.
    # Se guarda el texto en inglés, así la etiqueta se puede volver a traducir cuando el
    # usuario cambia el idioma.
    def __init__(self, master=None, **kw):
        self.textEn = kw.get("text")
        if "text" in kw:
            kw["text"] = T(kw["text"])
        super().__init__(master, **kw)
        _translated.add(self)

    def configure(self, cnf=None, **kw):
        if "text" in kw:
            self.textEn = kw["text"]
            kw["text"] = T(kw["text"])
        return super().configure(cnf, **kw)
    config = configure

    def retranslate(self):
        if self.textEn is not None:
            super().configure(text=T(self.textEn))


class Button(tk.Button):
    # tk.Button with the same behaviour as Label above
    # tk.Button con el mismo comportamiento que Label de arriba
    def __init__(self, master=None, **kw):
        self.textEn = kw.get("text")
        if "text" in kw:
            kw["text"] = T(kw["text"])
        super().__init__(master, **kw)
        _translated.add(self)

    def configure(self, cnf=None, **kw):
        if "text" in kw:
            self.textEn = kw["text"]
            kw["text"] = T(kw["text"])
        return super().configure(cnf, **kw)
    config = configure

    def retranslate(self):
        if self.textEn is not None:
            super().configure(text=T(self.textEn))


import weakref
_translated = weakref.WeakSet()    # every Label and Button created
                                   # cada Label y Button creado


class _TranslatedMessagebox:
    # Same functions as tkinter.messagebox, with the title and the message translated
    # Mismas funciones que tkinter.messagebox, con el título y el mensaje traducidos
    def __getattr__(self, name):
        func = getattr(_tkMessagebox, name)

        def call(title=None, message=None, **kw):
            return func(T(title), T(message), **kw)
        return call


class _TranslatedFiledialog:
    # Same functions as tkinter.filedialog, with the title and file types translated
    # Mismas funciones que tkinter.filedialog, con el título y los tipos de archivo traducidos
    def __getattr__(self, name):
        func = getattr(_tkFiledialog, name)

        def call(**kw):
            if "title" in kw:
                kw["title"] = T(kw["title"])
            if "filetypes" in kw:
                kw["filetypes"] = [(T(a), b) for a, b in kw["filetypes"]]
            return func(**kw)
        return call


_tkMessagebox, _tkFiledialog = messagebox, filedialog
messagebox = _TranslatedMessagebox()
filedialog = _TranslatedFiledialog()


# ── Header: institution logos and titles ──
# Put the logo images (PNG/JPG) in the "img" folder (or next to this script) with these
# names, or change the names here. If a file is missing, a placeholder is shown.
# The three logo boxes are square (LOGO_SIZE x LOGO_SIZE px); square images fit best.
# ── Encabezado: logos de las instituciones y títulos ──
# Pon las imágenes de los logos (PNG/JPG) en la carpeta "img" (o junto a este script) con
# estos nombres, o cambia los nombres aquí. Si falta un archivo, se muestra un recuadro
# de reemplazo. Los tres recuadros de logos son cuadrados (LOGO_SIZE x LOGO_SIZE px);
# las imágenes cuadradas quedan mejor.
LOGO_LEFT_FILE  = "UNAM.png"          # left side, first box
                                      # lado izquierdo, primer recuadro
LOGO_LEFT2_FILE = "UASLPlogo.png"     # left side, second box
                                      # lado izquierdo, segundo recuadro
LOGO_RIGHT_FILE = "INAOE.jpg"         # right side
                                      # lado derecho
LOGO_SIZE       = 96
APP_TITLE    = "Variational Ronchi Sensor"
APP_SUBTITLE = ""   # second header bar: empty for now, the space is kept for future text
                    # segunda barra del encabezado: vacía por ahora, el espacio se guarda para texto futuro

# ── Color palette ──
# ── Paleta de colores ──
NAVY      = "#0B3B78"   # header frame
                        # marco del encabezado
NAVY_DARK = "#082C5A"   # header title panel
                        # panel del título del encabezado
GOLD      = "#E2B33C"   # accent line / selected tab / running stage
                        # línea de acento / pestaña seleccionada / etapa en ejecución
BG        = "#F3F5F9"   # window background
                        # fondo de la ventana
WHITE     = "#FFFFFF"
TEXT      = "#1A1A1A"
MUTED     = "#5F6B7A"
TAB_TEXT  = "#1F4E8C"   # unselected tab text
                        # texto de las pestañas no seleccionadas
TAB_HOVER = "#EAF0F8"
BORDER    = "#C9D3E0"
GREEN     = "#1E9E4A"   # finished stage
                        # etapa terminada
PURPLE    = "#7E3FA3"   # pending stage
                        # etapa pendiente
RED       = "#C0392B"   # failed stage
                        # etapa que falló
BLUE_BTN  = "#0B5394"   # main action button
                        # botón de acción principal
GREEN_BTN = "#2E8B57"   # download button
                        # botón de descarga
FONT      = "Segoe UI"

# ── Pipeline stages (letter, tab name) ──
# ── Etapas del proceso (letra, nombre de la pestaña) ──
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
# Número que se muestra en cada círculo de estado y nombre (en inglés) del texto de estado
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
# Valor de avance (%) que reporta processImg cuando termina cada etapa
STAGE_END = {"A": 12, "B": 40, "C": 68, "D": 74, "E": 84, "F": 93, "G": 96, "H": 101}

# ── Global state ──
# ── Estado global ──
imgPath    = None
origPil    = None
cropRect   = None
cropActive = False
cropStart  = None
cropRectId = None
results    = []      # list of result dicts from processImg
                     # lista de diccionarios de resultados de processImg
slideIdx   = 0       # index of the current step
                     # índice del paso actual
stageSteps = {}      # stage letter -> list of step indices
                     # letra de la etapa -> lista de índices de pasos
curStage   = None
running    = False
renderJob  = None
nZernikeRun = ZERNIKE_Q
surfaceRun = dict(SURFACE_DEFAULTS)
exportRunning = False

BARH = 22
UI_SCALE = 1.0       # set after the window is created (screen DPI / 96)
                     # se define al crear la ventana (DPI de la pantalla / 96)

# Crop editing state (preview geometry and current drag)
# Estado de la edición del recorte (geometría de la vista previa y arrastre actual)
prevGeom   = None    # (scale, offsetX, offsetY) of the preview image on the canvas
                     # (escala, offsetX, offsetY) de la imagen de vista previa en el canvas
dragMode   = None    # "new", "move" or ("resize", corner)
                     # "new", "move" o ("resize", esquina)
dragStart  = None
cropAtPress = None
previewJob = None
cropPrevJob = None
HANDLE     = 7       # half size (px) of the corner handles
                     # medio tamaño (px) de las manijas de las esquinas


# ══════════════════════════════════════════════════════════════════════════════
# WIDGETS
# ══════════════════════════════════════════════════════════════════════════════
class TabStrip(tk.Frame):
    # Horizontal row of tabs with scroll arrows (selected tab in gold)
    # Fila horizontal de pestañas con flechas de desplazamiento (pestaña seleccionada en dorado)
    def __init__(self, parent, onSelect, fontSize=10, pady=7):
        super().__init__(parent, bg=WHITE, highlightthickness=1, highlightbackground=BORDER)
        self.onSelect = onSelect
        self.fontSize = fontSize
        self.pady = pady
        self.items = []
        self.selected = None

        self.btnL = Label(self, text="‹", bg=WHITE, fg=MUTED, font=(FONT, 15), padx=8,
                             cursor="hand2")
        self.btnR = Label(self, text="›", bg=WHITE, fg=MUTED, font=(FONT, 15), padx=8,
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
            lbl = Label(self.inner, text=text, bg=WHITE, fg=TAB_TEXT,
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
            # Los widgets todavía no tienen tamaño: se intenta otra vez en un momento
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
    return Button(parent, text=text, command=command, bg=bg, fg=fg,
                     activebackground=bg, activeforeground=fg, disabledforeground="#F0F0F0",
                     font=(FONT, size, "bold" if bold else "normal"),
                     relief="flat", bd=0, cursor="hand2", **kw)


def card(parent, title=None):
    # White panel with thin border and optional navy title (like a group box)
    # Panel blanco con borde delgado y título azul marino opcional (como un group box)
    outer = tk.Frame(parent, bg=WHITE, highlightthickness=1, highlightbackground=BORDER)
    if title:
        Label(outer, text=title, bg=WHITE, fg=NAVY, font=(FONT, 11, "bold"),
                 anchor="w").pack(fill="x", padx=14, pady=(10, 4))
    return outer


def loadLogo(label, fileName, maxW, maxH, placeholder):
    # Shows the logo image if the file is found, else a placeholder.
    # Folders searched, in order:
    #  1) inside the .exe (files added with PyInstaller --add-data)
    #  2) next to the .exe          3) next to this script      4) working folder
    # Each folder is searched directly and in its "img" subfolder (the layout of the
    # repository: img/UNAM.png, img/UASLPlogo.png, img/INAOE.jpg).
    # Muestra la imagen del logo si se encuentra el archivo; si no, un recuadro de reemplazo.
    # Carpetas donde se busca, en orden:
    #  1) dentro del .exe (archivos agregados con PyInstaller --add-data)
    #  2) junto al .exe          3) junto a este script      4) carpeta de trabajo
    # En cada carpeta se busca directamente y en su subcarpeta "img" (la estructura del
    # repositorio: img/UNAM.png, img/UASLPlogo.png, img/INAOE.jpg).
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
    candidates = [os.path.join(b, sub, fileName) for b in bases for sub in ("", "img")]
    path = next((c for c in candidates if os.path.isfile(c)), "")
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
# ESTADO DEL PROCESO
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
# VISTA DE ENTRADA (seleccionar imagen y recorte)
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# FRINGE REVIEW (the user confirms, edits, adds or deletes the anchor points)
# REVISIÓN DE FRANJAS (el usuario confirma, edita, agrega o borra los puntos de anclaje)
# ══════════════════════════════════════════════════════════════════════════════
fe = {}          # state of the fringe review
                 # estado de la revisión de franjas


def fringeEditorBridge(data):
    # Called from the processing thread: shows the review in the GUI thread and
    # waits until the user confirms (or cancels)
    # Se llama desde el hilo de procesamiento: muestra la revisión en el hilo de la GUI
    # y espera hasta que el usuario confirme (o cancele)
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


def feListWheel(event):
    # Scrolls the list of fringes of the fringe review
    # Desplaza la lista de franjas de la revisión de franjas
    if getattr(event, "num", None) == 4:
        step = -1
    elif getattr(event, "num", None) == 5:
        step = 1
    else:
        step = -1 if event.delta > 0 else 1
    feListCanvas.yview_scroll(step, "units")


def feListBindWheel(_=None):
    for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
        feListCanvas.bind_all(ev, feListWheel)


def feListUnbindWheel(_=None):
    for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
        feListCanvas.unbind_all(ev)


def feShowSelected(row):
    # Scrolls the list so the row of the selected fringe is visible (e.g. a new fringe,
    # which is added at the end of the list)
    # Desplaza la lista para que se vea la fila de la franja seleccionada (p. ej. una
    # franja nueva, que se agrega al final de la lista)
    feListCanvas.update_idletasks()
    total = max(feList.winfo_height(), 1)
    top, bottom = feListCanvas.yview()
    y0, y1 = row.winfo_y() / total, (row.winfo_y() + row.winfo_height()) / total
    if y0 < top:
        feListCanvas.yview_moveto(y0)
    elif y1 > bottom:
        feListCanvas.yview_moveto(max(0.0, y1 - (bottom - top)))


def feRefreshList():
    # Rebuilds the list of fringes (one button per fringe) inside its scrollable area
    # Reconstruye la lista de franjas (un botón por franja) dentro de su área con scroll
    for wdg in feList.winfo_children():
        wdg.destroy()
    if not fe["stripes"]:
        Label(feList, text="No fringes. Press “New fringe” to add one.", bg=WHITE,
                 fg=MUTED, font=(FONT, 9, "italic")).pack(anchor="w")
    selRow = None
    for i, a in enumerate(fe["stripes"]):
        row = tk.Frame(feList, bg=WHITE)
        row.pack(fill="x", pady=1)
        Label(row, bg=FE_COLORS[i % len(FE_COLORS)], width=2).pack(side="left", fill="y")
        sel = i == fe["sel"]
        n = len(a)
        warn = "  ⚠" if 0 < n < MIN_MANUAL_ANCHORS else ""
        Button(row, text=f"Fringe {i + 1}  ·  {n} points{warn}", anchor="w",
                  command=lambda i=i: feSelect(i), relief="flat", bd=0, cursor="hand2",
                  bg=GOLD if sel else "#EEF3F9", fg=TEXT,
                  font=(FONT, 10, "bold" if sel else "normal"), padx=8, pady=2
                  ).pack(side="left", fill="x", expand=True, padx=(4, 0))
        if sel:
            selRow = row
    if selRow is not None:
        feShowSelected(selRow)
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
    # (franja, punto) más cercano a (r, c) dentro de la tolerancia del clic
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
    # Clic derecho (o Ctrl + clic): borra el punto más cercano de la franja seleccionada
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
    # Escala un tamaño en píxeles con el DPI de la pantalla (nítido y proporcional en
    # pantallas de alta resolución)
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
    # Redimensiona la imagen una vez al tamaño del canvas (la parte costosa). El recorte
    # se dibuja encima como elementos del canvas, así moverlo o redimensionarlo es inmediato.
    global previewPhoto, prevGeom, previewJob
    previewJob = None
    canvasPrev.delete("all")
    cw, ch = max(10, canvasPrev.winfo_width()), max(10, canvasPrev.winfo_height())
    if origPil is None:
        canvasPrev.create_text(cw // 2, ch // 2, fill="#9AA5B1", font=(FONT, 12),
                               text=T("Select a Ronchigram image to start"))
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
    # Cuadrado del recorte, círculo inscrito (guía de la pupila), manijas de las esquinas y
    # el exterior oscurecido
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
        lblCrop.config(text=T(f"{x1 - x0} × {y1 - y0} px at ({x0}, {y0})"))
    else:
        lblCrop.config(text=T("none (full image)"))
    scheduleCropPreview()


def scheduleCropPreview():
    global cropPrevJob
    if cropPrevJob:
        root.after_cancel(cropPrevJob)
    cropPrevJob = root.after(150, drawCropPreview)


def drawCropPreview():
    # Enlarged view of the selected region, to check the crop before running
    # Vista ampliada de la región seleccionada, para revisar el recorte antes de correr
    global cropPrevPhoto, cropPrevJob
    cropPrevJob = None
    canvasCropPrev.delete("all")
    size = int(canvasCropPrev.cget("width"))
    if origPil is None or not cropRect:
        canvasCropPrev.create_text(size // 2, size // 2, text=T("No crop"), fill="#9AA5B1",
                                   font=(FONT, 10))
        return
    im = origPil.crop(cropRect).resize((size, size), Image.LANCZOS)
    cropPrevPhoto = ImageTk.PhotoImage(im)
    canvasCropPrev.create_image(0, 0, anchor="nw", image=cropPrevPhoto)
    canvasCropPrev.create_oval(1, 1, size - 2, size - 2, outline="white", dash=(4, 3))


def setCrop(x0, y0, side):
    # Square crop inside the image (side and position clamped)
    # Recorte cuadrado dentro de la imagen (lado y posición limitados)
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
    # Modo "Nuevo recorte": el siguiente arrastre dibuja un cuadrado nuevo aunque sea dentro del actual
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
        # Redimensiona desde una esquina; la esquina opuesta se queda fija
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
    # Rueda del ratón: agranda / achica el recorte alrededor de su centro
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
    # Flechas: paso siguiente/anterior en los resultados, o mover el recorte en la vista de entrada
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
# PROCESAMIENTO
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
        # 'ex' se borra cuando termina el bloque except, así que el mensaje se guarda ahora
        errMsg = str(ex)
        root.after(0, onFailed, errMsg)


def readSurface():
    # Surface parameters typed by the user; validated with the ideal ray mapping
    # Parámetros de la superficie que escribe el usuario; se validan con el mapeo ideal de rayos
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
    # Grado de Zernike que elige el usuario
    try:
        nZernikeRun = int(spinZernike.get())
        if not 2 <= nZernikeRun <= 100:
            raise ValueError
    except ValueError:
        messagebox.showerror("Zernike degree", "The Zernike degree n must be an integer "
                                               "between 2 and 100.")
        return
    # Optical surface parameters
    # Parámetros de la superficie óptica
    try:
        surfaceRun = readSurface()
    except RuntimeError as ex:
        messagebox.showerror("Optical surface", str(ex))
        return
    # Confirm the crop before running
    # Confirmar el recorte antes de correr
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
# VISTA DE RESULTADOS
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
    # Dibuja una tabla de resumen como imagen usando matplotlib
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
    # Texts in the selected language (the table is drawn again when it changes)
    # El índice va en la primera columna para que nunca se corte.
    # Textos en el idioma seleccionado (la tabla se vuelve a dibujar cuando cambia)
    tbl = ax.table(
        cellText=[[T(str(idx))] + [T(fmt(v)) for v in row]
                  for idx, row in zip(df.index, df.values)],
        colLabels=[T(str(df.index.name or ""))] + [T(str(c)) for c in df.columns],
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
        canvasSlide.create_text(cw // 2 + 10, ch // 2 + 10, text=T("No image for this step"),
                                fill="#9AA5B1", font=(FONT, 12))

    # Text panel
    # Panel de texto
    for widget in frameTextContent.winfo_children():
        widget.destroy()
    lines = result.get("text", [])
    if lines:
        for line in lines:
            Label(frameTextContent, text=str(line), bg=WHITE, fg="#333333",
                     font=(FONT, 10), anchor="w", justify="left",
                     wraplength=S(270)).pack(fill="x", pady=2)
    else:
        Label(frameTextContent, text="No additional info.", bg=WHITE, fg="#9AA5B1",
                 font=(FONT, 10, "italic")).pack(anchor="w")
    canvasText.yview_moveto(0)


def explainSaveError(ex):
    # Readable reason why a file could not be written
    # Motivo legible por el que no se pudo escribir un archivo
    if isinstance(ex, PermissionError):
        return "permission denied (is the file open in Excel or another program?)"
    if isinstance(ex, ImportError) or "openpyxl" in str(ex):
        return ("the package openpyxl is not available, so .xlsx files cannot be "
                "written (the .csv files have the same data)")
    return f"{type(ex).__name__}: {ex}"


def saveTable(df, path):
    # Saves a table as .xlsx (needs the openpyxl package) or as .csv
    # Guarda una tabla como .xlsx (necesita el paquete openpyxl) o como .csv
    if path.endswith(".xlsx"):
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)


def exportWavefront(result):
    # Saves the Zernike coefficients (OSA/ANSI order) and W_F. W_F is computed here,
    # in the background, because it has the full size of the crop.
    # Every file is written independently: if one of them cannot be written (a file
    # open in Excel, a missing package, no permission in the folder...) the others are
    # still saved and the message lists which one failed and why. (Before, all the
    # files were written in one block, so the first failure, usually the .xlsx file,
    # stopped the whole export and nothing was saved.)
    # Guarda los coeficientes de Zernike (orden OSA/ANSI) y W_F. W_F se calcula aquí, en
    # segundo plano, porque tiene el tamaño completo del recorte.
    # Cada archivo se escribe por separado: si uno no se puede escribir (un archivo abierto
    # en Excel, un paquete que falta, sin permiso en la carpeta...) los demás sí se guardan
    # y el mensaje dice cuál falló y por qué. (Antes todos los archivos se escribían en un
    # solo bloque, así que la primera falla, casi siempre el .xlsx, detenía toda la
    # exportación y no se guardaba nada.)
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
    lblFooter.config(text=f"Interpolating W_P to {h} × {w} and fitting Zernike n = {nZ}… 0%")

    def work():
        global exportRunning
        saved, failed = [], []

        def write(fname, func):
            try:
                func(fname)
                saved.append(fname)
            except Exception as ex:
                failed.append(f"{os.path.basename(fname)}: {explainSaveError(ex)}")

        try:
            def progress(f):
                root.after(0, lambda: lblFooter.config(
                    text=f"Interpolating W_P to {h} × {w} and fitting Zernike n = {nZ}… {int(100 * f)}%"))
            out = computeWF(wf, geom, progress=progress)
            WF = out["WF"]
            # Zernike coefficients of the fit on the large W_P (OSA/ANSI order:
            # j = (n(n+2) + m)/2), the ones that generate W_F, in micrometers
            # Coeficientes de Zernike del ajuste sobre el W_P grande (orden OSA/ANSI:
            # j = (n(n+2) + m)/2), los que generan W_F, en micrómetros
            zdf = pd.DataFrame({"j (OSA/ANSI)": range(len(wf["idx"])),
                                "n": [n for n, m in wf["idx"]],
                                "m": [m for n, m in wf["idx"]],
                                "coefficient (µm)": out["coefs"]})
            for ext in (".xlsx", ".csv"):
                write(base + "_Zernike_OSA" + ext, lambda f: saveTable(zdf, f))
            # The first 59 and the first 49 coefficients (OSA/ANSI order) in separate files
            # Los primeros 59 y los primeros 49 coeficientes (orden OSA/ANSI) en archivos aparte
            for k in EXPORT_FIRST:
                if k <= len(zdf):
                    for ext in (".xlsx", ".csv"):
                        write(f"{base}_Zernike_OSA_first{k}{ext}",
                              lambda f, k=k: saveTable(zdf.iloc[:k], f))
            lc = wf["lcurve"]
            write(base + "_W_F.mat", lambda f: savemat(f, {
                "W_F": WF, "W_F_without_piston_tilt": out["WFc"],
                "W_P_interpolated": out["WP"], "W_G_interpolated": out["WG"],
                "fit_points": out["nPoints"], "fit_rms": out["fitRms"],
                "zernike_coefficients": np.asarray(out["coefs"]),
                "zernike_n": np.array([n for n, m in wf["idx"]]),
                "zernike_m": np.array([m for n, m in wf["idx"]]),
                "zernike_degree": nZ,
                "mu": wf["mu"], "lcurve_mu": lc["mu"].values,
                "lcurve_R": lc["slope residual R"].values,
                "lcurve_S": lc["smoothness S"].values,
                "s_mm_per_px": wf["calibration"]["s_mm_per_px"],
                "Gamma": wf["calibration"]["Gamma"],
                "kappa_per_mm": wf["calibration"]["kappa_per_mm"],
                "units": "micrometers (pupil radius normalized to 1)",
                "R_mm": wf["surface"]["R"], "f_mm": wf["surface"]["f"],
                "D_mm": wf["surface"]["D"], "K": wf["surface"]["K"],
                "z_mm": wf["surface"]["z"]}))
            write(base + "_W_F.csv",
                  lambda f: np.savetxt(f, WF, delimiter=",", fmt="%.10g"))
            msg = (f"W_F: {h} × {w} (NaN outside the pupil)\n"
                   f"Zernike fit on {out['nPoints']} pixels of W_P\n\nSaved to:\n"
                   + "\n".join(saved))
            if failed:
                msg += "\n\nNot saved:\n" + "\n".join(failed)
                root.after(0, lambda: messagebox.showwarning("Saved with errors", msg))
                root.after(0, lambda: lblFooter.config(
                    text=f"Saved {len(saved)} files; {len(failed)} could not be saved."))
            else:
                root.after(0, lambda: messagebox.showinfo("Saved", msg))
                root.after(0, lambda: lblFooter.config(text="W_F and Zernike coefficients saved."))
        except Exception as ex:
            msg = explainSaveError(ex)
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
    # Si el paso tiene un dataframe de Excel (o varios), se guarda como .xlsx
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
        saved, failed = [], []
        try:
            if isinstance(data, dict):
                # One sheet per dataframe
                # Una hoja por dataframe
                with pd.ExcelWriter(path) as writer:
                    for sheet, df in data.items():
                        df.to_excel(writer, sheet_name=sheet[:31], index=False)
            else:
                data.to_excel(path, index=True)
            saved.append(path)
        except Exception as ex:
            failed.append(f"{os.path.basename(path)}: {explainSaveError(ex)}")

        # Final matrices (float64, NaN = no data) saved next to the Excel file
        # Matrices finales (float64, NaN = sin datos) guardadas junto al archivo de Excel
        if "arrays" in result:
            npzPath = os.path.splitext(path)[0] + ".npz"
            try:
                np.savez(npzPath, **result["arrays"])
                saved.append(npzPath)
            except Exception as ex:
                failed.append(f"{os.path.basename(npzPath)}: {explainSaveError(ex)}")

        msg = "Saved to:\n" + "\n".join(saved) if saved else "No file was saved."
        if failed:
            messagebox.showwarning("Saved with errors", msg + "\n\nNot saved:\n" + "\n".join(failed))
        else:
            messagebox.showinfo("Saved", msg)
        return

    # Otherwise save as image
    # Si no, se guarda como imagen
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
    if imgNp.ndim == 3:
        imgNp = cv2.cvtColor(imgNp, cv2.COLOR_RGB2BGR)
    # cv2.imencode + tofile instead of cv2.imwrite: cv2.imwrite fails silently on
    # Windows when the folder name has accents or other non-ASCII characters
    # cv2.imencode + tofile en lugar de cv2.imwrite: cv2.imwrite falla sin avisar en
    # Windows cuando el nombre de la carpeta tiene acentos u otros caracteres no ASCII
    try:
        ext = os.path.splitext(path)[1].lower() or ".png"
        ok, buf = cv2.imencode(ext if ext in (".png", ".jpg", ".jpeg") else ".png", imgNp)
        if not ok:
            raise ValueError("the image could not be encoded")
        buf.tofile(path)
    except Exception as ex:
        messagebox.showerror("Error", f"{os.path.basename(path)}: {explainSaveError(ex)}")
        return
    messagebox.showinfo("Saved", f"Image saved to:\n{path}")


def goBack():
    if running:
        return
    showView(viewInput)
    lblFooter.config(text="Select another image or adjust the crop and run the pipeline again.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# VENTANA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
# Sharp rendering on Windows high-resolution screens: without this, Windows
# enlarges the window as a bitmap and everything looks blurry (also in the .exe)
# Dibujo nítido en pantallas de alta resolución de Windows: sin esto, Windows agranda
# la ventana como mapa de bits y todo se ve borroso (también en el .exe)
try:
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

root = tk.Tk()
root.title(f"{APP_TITLE} - {T('Scientific Pipeline')}")
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
# ── Menú ──
def buildMenu():
    # (Re)built when the language changes
    # Se (re)construye cuando cambia el idioma
    menubar = tk.Menu(root)
    menuFile = tk.Menu(menubar, tearoff=0)
    menuFile.add_command(label=T("Open image…"), command=selectImage)
    menuFile.add_command(label=T("Back to input"), command=goBack)
    menuFile.add_command(label=T("Save current output…"), command=downloadCurrent)
    menuFile.add_separator()
    menuFile.add_command(label=T("Exit"), command=root.destroy)
    menubar.add_cascade(label=T("File"), menu=menuFile)
    menuLang = tk.Menu(menubar, tearoff=0)
    for code, name in LANGUAGES.items():
        menuLang.add_command(label=("✓ " if code == LANG else "   ") + name,
                             command=lambda c=code: setLanguage(c))
    menubar.add_cascade(label=T("Language"), menu=menuLang)
    root.config(menu=menubar)


def setLanguage(code):
    # Changes the language of the whole interface while the program is open
    # Cambia el idioma de toda la interfaz con el programa abierto
    global LANG
    if code == LANG:
        return
    LANG = code
    for w in list(_translated):
        try:
            w.retranslate()
        except tk.TclError:          # widget already destroyed
                                     # widget ya destruido
            pass
    root.title(f"{APP_TITLE} - {T('Scientific Pipeline')}")
    buildMenu()
    for c, lbl in langButtons.items():
        sel = c == LANG
        lbl.config(bg=GOLD if sel else BG, fg=TEXT if sel else TAB_TEXT,
                   font=(FONT, 9, "bold" if sel else "normal"))
    # Texts drawn on canvases and tables are drawn again in the new language. The
    # plots of a pipeline already run keep their language (run it again to change it).
    # Los textos dibujados en los canvas y las tablas se vuelven a dibujar en el nuevo
    # idioma. Las gráficas de un proceso ya corrido conservan su idioma (hay que correrlo
    # otra vez para cambiarlo).
    buildPreview()
    drawCropPreview()
    updateCropInfo()
    if results:
        renderSlide()


buildMenu()

# ── Header: logo | title | logo ──
# ── Encabezado: logo | título | logo ──
frameHeader = tk.Frame(root, bg=NAVY, padx=8, pady=8)
frameHeader.pack(fill="x", padx=12, pady=(10, 0))

def logoBox(side, padx=0):
    # Square white box for one logo
    # Recuadro blanco cuadrado para un logo
    box = tk.Frame(frameHeader, bg=WHITE, width=LOGO_SIZE, height=LOGO_SIZE)
    box.pack(side=side, padx=padx)
    box.pack_propagate(False)
    lbl = Label(box, bg=WHITE)
    lbl.pack(expand=True)
    return lbl

lblLogoLeft  = logoBox("left")
lblLogoLeft2 = logoBox("left", padx=(8, 0))
lblLogoRight = logoBox("right")

frameTitle = tk.Frame(frameHeader, bg=NAVY)
frameTitle.pack(side="left", fill="both", expand=True, padx=8)
Label(frameTitle, text=APP_TITLE, bg=NAVY_DARK, fg=WHITE,
         font=(FONT, 22, "bold"), highlightthickness=2,
         highlightbackground=WHITE).pack(fill="both", expand=True, pady=(0, 4))
Label(frameTitle, text=APP_SUBTITLE, bg=NAVY_DARK, fg="#C9D9F0",
         font=(FONT, 10), highlightthickness=2,
         highlightbackground=WHITE).pack(fill="x")

loadLogo(lblLogoLeft,  LOGO_LEFT_FILE,  LOGO_SIZE - 8, LOGO_SIZE - 8, "Logo 1")
loadLogo(lblLogoLeft2, LOGO_LEFT2_FILE, LOGO_SIZE - 8, LOGO_SIZE - 8, "Logo 2")
loadLogo(lblLogoRight, LOGO_RIGHT_FILE, LOGO_SIZE - 8, LOGO_SIZE - 8, "Logo 3")

tk.Frame(root, bg=GOLD, height=4).pack(fill="x", padx=12)

# ── Pipeline status ──
# ── Estado del proceso ──
frameStatus = tk.Frame(root, bg=BG)
frameStatus.pack(fill="x", padx=14, pady=(10, 6))
Label(frameStatus, text="Pipeline status", bg=BG, fg=NAVY,
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
# Language selector (the language names are not translated)
# Selector de idioma (los nombres de los idiomas no se traducen)
frameLang = tk.Frame(frameStatus, bg=BG)
frameLang.pack(side="right", padx=(12, 0))
langButtons = {}
for code, name in LANGUAGES.items():
    sel = code == LANG
    lb = tk.Label(frameLang, text=name, bg=GOLD if sel else BG, fg=TEXT if sel else TAB_TEXT,
                  font=(FONT, 9, "bold" if sel else "normal"), padx=8, pady=2,
                  cursor="hand2", highlightthickness=1, highlightbackground=BORDER)
    lb.pack(side="left", padx=(0, 4))
    lb.bind("<Button-1>", lambda e, c=code: setLanguage(c))
    langButtons[code] = lb
lblStatus = Label(frameStatus, text="Waiting for an image", bg=BG, fg=MUTED,
                     font=(FONT, 10))
lblStatus.pack(side="right")

# ── Footer (status line) ──
# ── Pie (línea de estado) ──
lblFooter = Label(root, text="Select a Ronchigram image to start (File ▸ Open image…).",
                     bg=BG, fg=MUTED, font=(FONT, 9), anchor="w")
lblFooter.pack(side="bottom", fill="x", padx=16, pady=(0, 2))
tk.Frame(root, bg=NAVY, height=6).pack(side="bottom", fill="x")

# ── Body ──
# ── Cuerpo ──
frameBody = tk.Frame(root, bg=BG)
frameBody.pack(fill="both", expand=True, padx=12, pady=(0, 6))

# ══ INPUT VIEW ══
# ══ VISTA DE ENTRADA ══
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

Label(ctrl, text="Drag to draw a square and fit its dashed circle to the exit pupil "
                    "(it is used as the pupil). Drag inside to move it, drag a corner to "
                    "resize it, mouse wheel to zoom, arrow keys to nudge (Shift = 10 px).",
         bg=WHITE, fg=MUTED, font=(FONT, 9, "italic"), wraplength=S(360),
         justify="left").pack(fill="x", pady=(0, 8))

rowInfo = tk.Frame(ctrl, bg=WHITE)
rowInfo.pack(fill="x", pady=(0, 8))
info = tk.Frame(rowInfo, bg=WHITE)
info.pack(side="left", fill="both", expand=True, anchor="n")
def infoRow(r, label):
    Label(info, text=label, bg=WHITE, fg=MUTED, font=(FONT, 10), anchor="w"
             ).grid(row=r, column=0, sticky="w", pady=1)
    # The values (file name, size, crop) are not translated: tk.Label
    # Los valores (nombre del archivo, tamaño, recorte) no se traducen: tk.Label
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
# Superficie óptica bajo prueba (construye el Ronchigrama ideal y la escala de W)
Label(ctrl, text="Optical surface", bg=WHITE, fg=NAVY, font=(FONT, 10, "bold"),
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
    Label(gridSurf, text=SURFACE_SHORT[key], bg=WHITE, fg=TEXT, font=(FONT, 10),
             anchor="w").grid(row=r_, column=2 * c_, sticky="w", pady=2,
                              padx=(0 if c_ == 0 else 14, 6))
    e_ = tk.Entry(gridSurf, width=9, font=(FONT, 10), justify="center")
    e_.insert(0, f"{SURFACE_DEFAULTS[key]:g}")
    e_.grid(row=r_, column=2 * c_ + 1, sticky="w", pady=2)
    surfaceEntries[key] = e_
Label(ctrl, text="z: distance from the surface vertex to the plane where the "
                    "Ronchigram was recorded. K: 0 sphere, -1 paraboloid.",
         bg=WHITE, fg=MUTED, font=(FONT, 9, "italic"), wraplength=S(360),
         justify="left").pack(fill="x", pady=(0, 8))

rowZ = tk.Frame(ctrl, bg=WHITE)
rowZ.pack(fill="x", pady=(0, 10))
Label(rowZ, text="Zernike degree n:", bg=WHITE, fg=TEXT,
         font=(FONT, 10)).pack(side="left")
spinZernike = tk.Spinbox(rowZ, from_=2, to=100, width=5, font=(FONT, 10),
                         justify="center")
spinZernike.delete(0, "end")
spinZernike.insert(0, str(ZERNIKE_Q))
spinZernike.pack(side="left", padx=(8, 0))
Label(rowZ, text="(article: 50)", bg=WHITE, fg=MUTED,
         font=(FONT, 9, "italic")).pack(side="left", padx=(6, 0))

btnProcess = makeButton(ctrl, "Run pipeline", startProcess, "#9E9E9E", size=13,
                        pady=10, state="disabled")
btnProcess.pack(fill="x", pady=(0, 10))

canvasBar = tk.Canvas(ctrl, height=BARH, bg="#DDE3EB", highlightthickness=0)
canvasBar.pack(fill="x")
barFill = canvasBar.create_rectangle(0, 0, 0, BARH, fill=BLUE_BTN, outline="")
lblPercent = Label(ctrl, text="0%", bg=WHITE, fg=MUTED, font=(FONT, 10))
lblPercent.pack(anchor="e", pady=(4, 0))

# ══ FRINGE REVIEW VIEW ══
# ══ VISTA DE REVISIÓN DE FRANJAS ══
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
Label(feCtrl, text="Select a fringe and click on the image to add anchor points. Drag a "
                      "point to move it; right-click (or Ctrl + click) to delete it. The points "
                      "stay where you put them; the line of each fringe is fitted to its "
                      "anchor points after you confirm.",
         bg=WHITE, fg=MUTED, font=(FONT, 9, "italic"), wraplength=S(310),
         justify="left").pack(fill="x", pady=(0, 8))
# The action buttons are packed first at the bottom so they are always visible. The
# list of fringes goes in its own scrollable area above them: its requested height is
# fixed, so a long list (many fringes) never makes the window grow and never pushes
# the buttons out of the screen; the list is scrolled with its scrollbar or the wheel.
# Los botones de acción se colocan primero abajo para que siempre se vean. La lista de
# franjas va en su propia área con scroll arriba de ellos: su altura pedida es fija,
# así una lista larga (muchas franjas) nunca hace crecer la ventana ni saca los botones
# de la pantalla; la lista se recorre con su barra o con la rueda del ratón.
feBottom = tk.Frame(feCtrl, bg=WHITE)
feBottom.pack(side="bottom", fill="x")
feListArea = tk.Frame(feCtrl, bg=WHITE)
feListArea.pack(side="top", fill="both", expand=True, pady=(0, 6))
feListCanvas = tk.Canvas(feListArea, bg=WHITE, highlightthickness=0, width=S(290),
                         height=S(120))
feListScroll = tk.Scrollbar(feListArea, orient="vertical", command=feListCanvas.yview)
feListCanvas.configure(yscrollcommand=feListScroll.set)
feListScroll.pack(side="right", fill="y")
feListCanvas.pack(side="left", fill="both", expand=True)
feList = tk.Frame(feListCanvas, bg=WHITE)          # the rows of feRefreshList go here
                                                   # aquí van las filas de feRefreshList
feListWin = feListCanvas.create_window((0, 0), window=feList, anchor="nw")
feList.bind("<Configure>",
            lambda e: feListCanvas.configure(scrollregion=feListCanvas.bbox("all")))
feListCanvas.bind("<Configure>",
                  lambda e: feListCanvas.itemconfigure(feListWin, width=e.width))
# Mouse wheel over the list: <MouseWheel> on Windows and macOS, buttons 4/5 on Linux.
# It is bound only while the pointer is over the list, so the wheel keeps working as
# before in the rest of the GUI.
# Rueda del ratón sobre la lista: <MouseWheel> en Windows y macOS, botones 4/5 en Linux.
# Solo se asigna mientras el puntero está sobre la lista, así la rueda sigue
# funcionando como antes en el resto de la GUI.
feListArea.bind("<Enter>", feListBindWheel)
feListArea.bind("<Leave>", feListUnbindWheel)
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
feInfo = Label(feBottom, text="", bg=WHITE, fg=TEXT, font=(FONT, 9), wraplength=S(310),
                  justify="left", anchor="w")
feInfo.pack(fill="x", pady=(0, 6))
makeButton(feBottom, "✓  Confirm fringes and continue", feConfirm, GREEN_BTN, size=12,
           pady=8).pack(fill="x", pady=(0, 5))
makeButton(feBottom, "Cancel", feCancel, "#9E9E9E", size=10, pady=4).pack(fill="x")

# ══ RESULTS VIEW ══
# ══ VISTA DE RESULTADOS ══
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
Label(frameInfo, text="Step details", bg=WHITE, fg=NAVY, font=(FONT, 11, "bold"),
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
lblStepTitle = Label(rowBottom, text="", bg=WHITE, fg=TEXT, font=(FONT, 11, "bold"),
                        anchor="w")
lblStepTitle.pack(side="left", padx=12)
makeButton(rowBottom, "›", lambda: navigate(1), NAVY, size=13, padx=12,
           pady=2).pack(side="right")
makeButton(rowBottom, "‹", lambda: navigate(-1), NAVY, size=13, padx=12,
           pady=2).pack(side="right", padx=6)
lblCounter = Label(rowBottom, text="", bg=WHITE, fg=MUTED, font=(FONT, 10))
lblCounter.pack(side="right", padx=10)

# Keyboard navigation between steps
# Navegación con el teclado entre pasos
root.bind("<Left>",  lambda e: onArrow(-1, 0, e))
root.bind("<Right>", lambda e: onArrow(1, 0, e))
root.bind("<Up>",    lambda e: onArrow(0, -1, e))
root.bind("<Down>",  lambda e: onArrow(0, 1, e))

# ══════════════════════════════════════════════════════════════
# Start on the input view
# Empieza en la vista de entrada
# ══════════════════════════════════════════════════════════════
showView(viewInput)
root.update_idletasks()
buildPreview()
root.mainloop()