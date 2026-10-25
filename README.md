# Variational Ronchi Sensor

A graphical application for processing Ronchigrams, estimating transverse aberrations, reconstructing the optical wavefront, and exporting Zernike coefficients and full-pupil wavefront matrices.

The software implements the computational pipeline described in the accompanying paper. It provides an interactive workflow for selecting the exit pupil, detecting and reviewing fringes, comparing experimental and theoretical Ronchigrams, refining transverse-aberration point clouds, and reconstructing the wavefront through radial integration and an OSA/ANSI-ordered Zernike fit.

> **Research software notice**  
> This repository contains software associated with a scientific publication. If you use it in academic work, please cite the paper and the software release. Add the final citation and DOI in the [Citation](#citation) section when they become available.

## Key features

- Desktop graphical user interface written in Python with Tkinter.
- Input support for PNG, JPEG, BMP, and TIFF Ronchigrams.
- Interactive square crop whose inscribed circle defines the exit pupil.
- Automatic selection and contrast normalization of the image channel carrying the fringes.
- Automatic coarse and fine fringe alignment.
- Adaptive fringe segmentation for nonuniform pupil illumination.
- Two automatic detection strategies:
  - vertical windows around column-profile peaks;
  - fringe following for strongly curved or S-shaped fringes.
- Interactive fringe review, including adding, moving, and deleting anchor points and fringes.
- Experimental-to-theoretical fringe correspondence by diffraction order.
- Transverse-aberration estimation in two orthogonal directions.
- Grid construction from experimental and theoretical centroid-line intersections.
- Two-stage interpolation and point-cloud refinement.
- Extension of the reconstructed quantities to the full exit pupil, with measured and extrapolated points explicitly identified.
- Wavefront reconstruction by radial integration of the transverse aberration.
- Zernike fitting in OSA/ANSI order, with a user-selectable maximum degree from 2 to 100.
- Export of intermediate figures, tables, point clouds, matrices, Zernike coefficients, and wavefront products.
- Background processing so that the interface remains responsive during a run.

## Pipeline overview

A complete run contains 62 processing and output steps grouped into eight stages:

| Stage | Steps | Main operation | Main result |
|---|---:|---|---|
| A. Preparation | 1-5 | Orientation, exit-pupil definition, and optical parameters | Vertically aligned Ronchigram centered on the pupil |
| B. Vertical Ronchigram | 6-19 | Segmentation, fringe detection and review, centroid lines, theoretical reference | Transverse aberration in x |
| C. Horizontal Ronchigram | 20-33 | Analysis after an exact 90-degree rotation | Transverse aberration in y |
| D. Grid | 34-38 | Intersections of both centroid-line families | Displacements dx and dy |
| E. Refinement 1 | 39-43 | First interpolation pass | Refined point clouds |
| F. Refinement 2 | 44-48 | Second interpolation pass | Dense point clouds |
| G. Full pupil | 49-55 | Extension beyond the measured region | Full-pupil transverse-aberration fields |
| H. Wavefront | 56-62 | Surface fitting, radial integration, and Zernike decomposition | Integrated and fitted wavefronts and Zernike coefficients |

## Requirements

### Supported environment

The application is designed primarily as a Windows desktop program. The Python source can also run on other systems that provide Tkinter and the required scientific Python packages.

Recommended setup:

- Python 3.10 or newer
- A working Tk installation
- At least 8 GB of RAM for typical images
- Additional memory and processing time for large crops or high Zernike degrees

### Python dependencies

The source code imports the following third-party packages:

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `opencv-python`
- `Pillow`
- `openpyxl` for Excel export

Tkinter is part of the standard Python distribution on Windows. On some Linux distributions it must be installed separately through the system package manager.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<OWNER>/<REPOSITORY>.git
cd <REPOSITORY>
```

Replace `<OWNER>` and `<REPOSITORY>` with the final GitHub organization/user and repository name.

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
python -m pip install --upgrade pip
python -m pip install numpy scipy pandas matplotlib opencv-python Pillow openpyxl
```

Alternatively, create a `requirements.txt` file containing:

```text
numpy
scipy
pandas
matplotlib
opencv-python
Pillow
openpyxl
```

and install it with:

```bash
python -m pip install -r requirements.txt
```

## Running the application

From the repository directory, run:

```bash
python Ronchi.py
```

The GUI optionally displays three institutional logos. Place the following image files next to `Ronchi.py` if they are to be included:

```text
UNAM.png
UASLPlogo.png
INAOE.jpg
```

If a logo file is missing, the application displays a placeholder and the scientific pipeline remains usable.

## Quick start

1. Launch the program and select **Open image** or **Select image**.
2. Draw a square crop around the diffraction order to be analyzed.
3. Fit the dashed circle to the exit pupil. The circle inscribed in the crop defines the pupil center and radius.
4. Enter the optical-surface parameters:
   - radius of curvature, `R`;
   - focal length, `f`;
   - aperture diameter, `D`;
   - conic constant, `K`;
   - distance from the surface vertex to the Ronchigram plane, `z`;
   - maximum Zernike degree, `n`.
5. Select **Run pipeline** and confirm the crop.
6. In the fringe-review view, verify that the anchor points follow the centers of the fringes.
7. Add, move, or remove points and fringes when required, then select **Confirm fringes and continue**.
8. Browse the results by stage and processing step.
9. Use the download button to save images, Excel tables, NumPy matrices, Zernike coefficients, and wavefront files.

## Input data

### Accepted image formats

- PNG
- JPG/JPEG
- BMP
- TIFF

The image is loaded as an 8-bit RGB image. Processing is performed on copies, and the input file is not modified.

### Exit-pupil selection

The recommended workflow is to draw a square crop around the diffraction order of interest and fit its inscribed dashed circle to the pupil boundary. This user-defined pupil is preferred because outer diffraction orders can be dim, unevenly illuminated, or contaminated by neighboring orders.

Crop controls include:

- drag to draw a new square;
- drag inside the square to move it;
- drag a corner to resize it;
- use the mouse wheel to scale it;
- use the arrow keys for one-pixel shifts;
- hold Shift while using the arrow keys for ten-pixel shifts.

If no crop is provided, the program attempts automatic pupil detection using thresholding, morphological closing, and the convex hull of the largest bright region.

### Optical parameters

All dimensional surface parameters are entered in millimeters.

| Parameter | Meaning |
|---|---|
| `R` | Radius of curvature of the tested surface |
| `f` | Focal length |
| `D` | Aperture diameter |
| `K` | Conic constant, e.g. 0 for a sphere and -1 for a paraboloid |
| `z` | Distance from the surface vertex to the Ronchigram recording plane |
| `n` | Maximum Zernike radial degree, from 2 to 100 |

Before processing, the software verifies that the values define a valid surface and ray mapping. In particular, the inferred refractive index is `1 + R/f`, the conic must be defined across the full aperture, and the recording plane must not lie inside the caustic.

The default parameters in the current source are:

```text
R = 122.06 mm
f = 237.0 mm
D = 39.0 mm
K = 0.0
z = 173.5 mm
Zernike degree = 50
```

These defaults are experiment-specific examples and should be replaced with the parameters of the surface and acquisition geometry being analyzed.

## Fringe review

Automatic fringe detection is followed by a mandatory interactive review. This step is important because the confirmed anchor points determine the centroid lines and all subsequent calculations.

Available operations include:

- select a fringe from the list;
- click to add an anchor point;
- drag an anchor point to move it;
- right-click, or Ctrl-click, to delete an anchor point;
- add a new fringe;
- clear or delete a fringe;
- delete all fringes;
- restore the automatic detection.

At least two fringes with at least three anchor points each are required to continue.

## Method summary

### Orientation and resampling

The program first estimates fringe orientation using Hough transforms and the dominant image-gradient direction. It then refines the alignment from the central fringes until the residual inclination is at most 0.01 degrees or five correction iterations have been completed.

The final working image is generated from the original crop with a single affine transformation and Lanczos interpolation, avoiding repeated resampling.

### Segmentation and fringe detection

The fringe channel is selected automatically as the RGB channel with the largest standard deviation. Its intensities are stretched between the 1st and 99.7th percentiles. Segmentation then applies CLAHE, Gaussian smoothing, adaptive thresholding, connected-component filtering, morphological closing, and the pupil mask.

Nearly vertical fringes are detected from peaks in the column profile. When fringes leave their vertical search windows in more than 38% of the sampled rows, or valid profile peaks cannot be found, the software switches to row-by-row fringe following.

### Centroid lines and transverse aberration

A polynomial is fitted to the confirmed anchor points of each fringe. The default centroid-line degree is 2. For fringes found by the curved-fringe tracker, the degree may increase to 6 when justified by the Bayesian information criterion.

Experimental centroid lines are matched to theoretical lines by fringe order relative to the central fringe. Signed transverse aberrations are calculated as experimental minus theoretical positions at corresponding rows.

### Refinement and full-pupil extension

The measured point clouds undergo two interpolation passes. The first requests five inserted points between neighboring lines, and the second requests ten. The applied number is reduced when necessary to keep refined samples at least one pixel apart.

Values outside the measured region are extrapolated to the full pupil with a fourth-degree polynomial plus a decaying nearest-data residual. Exported full-pupil tables label each point as `data` or `extended` so that measured/refined values can be distinguished from extrapolated values.

### Wavefront reconstruction

The signed transverse-aberration components are fitted with polynomial surfaces in a Legendre basis. Their radial component is integrated along 360 angular directions using 2,000 radial steps. The resulting wavefront is represented in micrometers.

The integrated wavefront is fitted with normalized Zernike polynomials in OSA/ANSI order. Maps displayed in the GUI omit piston, x/y tilt, and defocus, while the exported data retain the full coefficient set and provide corrected products separately.

## Outputs

### Per-step downloads

Depending on the selected step, the GUI can export:

- PNG or JPEG images;
- Excel workbooks containing summaries or point clouds;
- `.npz` archives containing difference matrices.

### Final wavefront products

The final export writes four files using the selected base name:

```text
<name>_Zernike_OSA.xlsx
<name>_Zernike_OSA.csv
<name>_W_F.mat
<name>_W_F.csv
```

The Zernike tables contain:

- OSA/ANSI index `j`;
- radial degree `n`;
- azimuthal order `m`;
- fitted coefficient.

The MATLAB file contains:

- `W_F`: full Zernike-fitted wavefront;
- `W_F_without_piston_tilt_defocus`;
- `W_r_interpolated`: radially integrated wavefront on the crop grid;
- Zernike coefficients and their `(n, m)` indices;
- fit-point count and fit RMS;
- optical parameters and units.

Wavefront values are in micrometers. Pixels outside the pupil are stored as `NaN`.

> Computing and exporting the full-resolution `W_F` matrix can take substantially longer than the main GUI run, especially for large image crops and high Zernike degrees.

## Main numerical settings

The following values are defined as constants near the beginning of `Ronchi.py` and can be adjusted for controlled experiments:

| Setting | Default |
|---|---:|
| Fine-alignment tolerance | 0.01 degrees |
| Maximum fine-alignment iterations | 5 |
| Adaptive-threshold window | 25% of image size |
| Adaptive-threshold offset | 20 gray levels above the local mean |
| Minimum connected-component area | 300 px |
| Standard morphological closing | 10 x 10 px |
| Anchor-point bands per fringe | 15 |
| Curved-fringe switch threshold | 38% of rows |
| Default centroid-line degree | 2 |
| Maximum curved-fringe degree | 6 |
| Sampling interval along a centroid line | 5 px |
| Refinement passes | 5 and 10 inserted points |
| Minimum refined-point spacing | 1 px |
| Full-pupil extension polynomial degree | 4 |
| TA surface-fit degree per variable | 11 |
| Radial integration steps | 2,000 |
| Angular directions | 360 |
| Default maximum Zernike degree | 50 |

## Limitations

1. The method is designed for a grating that produces a family of vertical fringes after alignment. The orthogonal transverse-aberration component is inferred from an exact 90-degree rotation of the same Ronchigram, which assumes the rotational symmetry required by the method.
2. When a manual crop is used, the pupil center and radius are only as accurate as the fitted inscribed circle.
3. Values in the extended region near the pupil boundary are extrapolated rather than measured. Use the `source` column in exported tables to distinguish them.
4. Very compressed, closed, merged, low-contrast, or partially visible fringes may require manual correction during fringe review.
5. Processing time and memory use increase with image dimensions and maximum Zernike degree.
6. The optical parameters and acquisition geometry must be physically consistent. Invalid surfaces, caustic crossings, or a pupil collapsed at focus are rejected before processing.

## Suggested repository structure

```text
.
├── Ronchi.py
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
├── docs/
│   └── Supplemental_GUI_Document.pdf
├── examples/
│   ├── input/
│   └── output/
└── assets/
    └── screenshots/
```

If the logo files are kept in `assets/`, update the corresponding constants in `Ronchi.py` or copy the images next to the script when building the executable.

## Windows executable

A Windows executable can be distributed separately from the Python source, for example through the GitHub **Releases** page. The executable should be built from a tagged commit so that users can identify the corresponding source version.

If institutional logos are bundled with PyInstaller, include them as data files and retain their expected names. Example commands are intentionally not fixed here because the appropriate separator and packaging options may vary by operating system and release process.

## Reproducibility recommendations

For scientific use, record the following with every analysis:

- software version or Git commit;
- input image name and checksum;
- crop coordinates and pupil radius;
- `R`, `f`, `D`, `K`, and `z`;
- selected Zernike degree;
- any manual changes made during fringe review;
- exported full-pupil point tables, including the `source` column;
- Zernike coefficients and fitted wavefront matrices.

A tagged release and an archived copy in a research repository are recommended for the version associated with the paper.

## Citation

Please replace this placeholder with the citation of the associated paper:

```bibtex
@article{AUTHOR_YEAR_VARIATIONAL_RONCHI,
  author  = {Author list},
  title   = {Paper title},
  journal = {Journal name},
  year    = {Year},
  volume  = {Volume},
  pages   = {Pages},
  doi     = {DOI}
}
```

If the software receives a DOI, add a separate software citation and a `CITATION.cff` file.

## License

No license information was provided with the current source. Before publishing the repository, add a `LICENSE` file and replace this section with the selected license and any institutional requirements.

Without an explicit license, others may view the public repository but do not automatically receive permission to copy, modify, or redistribute the software.

## Authors and contact

Add the final author list, institutional affiliations, and a contact address here.

```text
Corresponding authors: Jesús Alonso Arriaga Hernández^1, Bolivia Cuevas Otahola^2
Email: dr.j.a.arriaga.hernandez@gmail.com, bcuevas@astro.unam.mx
Institution:
1. FCFM, BUAP
2. Instituto de Astronomía, UNAM
```

## Acknowledgments

The authors thank Instituto de Astronomía UNAM and Instituto Nacional de Astrofísica, Óptica y Electrónica for the support given during this research work.

## Disclaimer

This software is intended for research use. Users are responsible for validating the acquisition geometry, optical parameters, fringe selection, and reconstructed quantities for their experimental configuration. Scientific conclusions should not rely on the automatically detected fringes without reviewing the anchor points and the intermediate diagnostic products.
