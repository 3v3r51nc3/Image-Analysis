#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere; we cd to this script directory (your bin/)
BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BIN_DIR}"

FIG_DIR="${BIN_DIR}/figures"
mkdir -p "${FIG_DIR}"

# -------------------------
# INPUTS (existing in bin/)
# -------------------------
IMG_NOISE="camera_bruit_poivre_et_sel.png"
IMG_EDGES="cat.jpg"
IMG_FINE="camera.png"
BIN_IMG="binary.png"

# Kernels / masks (existing in bin/)
K_MEAN_5="maskMean5x5.png"
K_GAUSS_5="maskGauss5x5.png"

# Bilateral uses a kernel image + required intensity scale
K_BILAT="${K_GAUSS_5}"
BILAT_C="0.1"

# Morphology SEs
E_LINE_V="morphoLineV.png"
E_LINE_H="morphoLineH.png"
E_CROSS="morphoCross.png"
E_CIRCLE="morphoCircle.png"

# -------------------------
# helpers
# -------------------------
must_exist() {
    if [[ ! -f "$1" ]]; then
        echo "ERROR: missing file: $1" >&2
        exit 1
    fi
}
must_exec() {
    if [[ ! -x "$1" ]]; then
        echo "ERROR: missing executable: $1" >&2
        exit 1
    fi
}
run() {
    echo "[RUN] $*"
    "$@"
}
copy() {
    must_exist "$1"
    cp -f "$1" "$2"
}

# Validate required executables for your LaTeX figures
must_exec "./meanFilter"
must_exec "./convolution"
must_exec "./edgeSobel"
must_exec "./bilateralFilter"
must_exec "./transpose"
must_exec "./expand"
must_exec "./rotate"

# Validate required inputs
must_exist "${IMG_NOISE}"
must_exist "${IMG_EDGES}"
must_exist "${IMG_FINE}"
must_exist "${BIN_IMG}"
must_exist "${K_MEAN_5}"
must_exist "${K_GAUSS_5}"
must_exist "${K_BILAT}"

# =========================================================
# EXACT NAMES USED IN YOUR LaTeX
# =========================================================

# --- input_A / input_B / input_C
copy "${IMG_NOISE}" "${FIG_DIR}/input_A.png"
copy "${IMG_EDGES}" "${FIG_DIR}/input_B.png"
copy "${IMG_FINE}"  "${FIG_DIR}/input_C.png"

# --- mean filter (requires -M/--filterSize)
copy "${IMG_NOISE}" "${FIG_DIR}/mean_input.png"
run ./meanFilter -I "${IMG_NOISE}" -M 2 -O "${FIG_DIR}/mean_output.png"

# --- convolution (kernel image path)
copy "${IMG_EDGES}" "${FIG_DIR}/conv_input.png"
run ./convolution -I "${IMG_EDGES}" -K "${K_GAUSS_5}" -O "${FIG_DIR}/conv_output.png"

# --- sobel (single output program; duplicate to satisfy LaTeX filenames)
copy "${IMG_EDGES}" "${FIG_DIR}/sobel_input.png"
run ./edgeSobel -I "${IMG_EDGES}" -O "${FIG_DIR}/sobel_norm.png"
cp -f "${FIG_DIR}/sobel_norm.png" "${FIG_DIR}/sobel_gx.png"

# --- bilateral (requires -K kernel image and -C intensityScale)
copy "${IMG_NOISE}" "${FIG_DIR}/bilat_input.png"
run ./bilateralFilter -I "${IMG_NOISE}" -K "${K_BILAT}" -C "${BILAT_C}" -O "${FIG_DIR}/bilat_output.png"

# --- transpose
copy "${IMG_EDGES}" "${FIG_DIR}/transpose_input.png"
run ./transpose -I "${IMG_EDGES}" -O "${FIG_DIR}/transpose_output.png"

# --- interpolation (expand): -F INT, -P ('nearest' or 'bilinear')
copy "${IMG_FINE}" "${FIG_DIR}/interp_input.png"
run ./expand -I "${IMG_FINE}" -F 2 -P nearest  -O "${FIG_DIR}/interp_nearest.png"
run ./expand -I "${IMG_FINE}" -F 2 -P bilinear -O "${FIG_DIR}/interp_bilinear.png"

# --- rotation: -A FLOAT, -P ('nearest' or 'bilinear')
copy "${IMG_FINE}" "${FIG_DIR}/rot_input.png"
run ./rotate -I "${IMG_FINE}" -A 30 -P nearest  -O "${FIG_DIR}/rot_nearest.png"
run ./rotate -I "${IMG_FINE}" -A 30 -P bilinear -O "${FIG_DIR}/rot_bilinear.png"

# =========================================================
# OPTIONAL: extra outputs (not referenced by your LaTeX yet)
# =========================================================

if [[ -x "./dilate" ]]; then
    run ./dilate -I "${BIN_IMG}"    -E "${E_LINE_V}" -O "${FIG_DIR}/dilate_binary_lineV.png"
    run ./dilate -I "${IMG_EDGES}"  -E "${E_LINE_V}" -O "${FIG_DIR}/dilate_cat_lineV.png"
    run ./dilate -I "${IMG_EDGES}"  -E "${E_LINE_H}" -O "${FIG_DIR}/dilate_cat_lineH.png"
fi

if [[ -x "./erode" ]]; then
    run ./erode  -I "${BIN_IMG}"    -E "${E_LINE_V}" -O "${FIG_DIR}/erode_binary_lineV.png"
    run ./erode  -I "${IMG_EDGES}"  -E "${E_CROSS}"  -O "${FIG_DIR}/erode_cat_cross.png"
fi

if [[ -x "./open" ]]; then
    run ./open   -I "${BIN_IMG}"    -E "${E_LINE_V}" -O "${FIG_DIR}/open_binary_lineV.png"
    run ./open   -I "${IMG_EDGES}"  -E "${E_LINE_V}" -O "${FIG_DIR}/open_cat_lineV.png"
fi

if [[ -x "./close" ]]; then
    run ./close  -I "${BIN_IMG}"    -E "${E_CIRCLE}" -O "${FIG_DIR}/close_binary_circle.png"
    run ./close  -I "${IMG_EDGES}"  -E "${E_CIRCLE}" -O "${FIG_DIR}/close_cat_circle.png"
fi

if [[ -x "./morphologicalGradient" ]]; then
    run ./morphologicalGradient -I "${BIN_IMG}"   -E "${E_CROSS}" -O "${FIG_DIR}/mgrad_binary_cross.png"
    run ./morphologicalGradient -I "${IMG_EDGES}" -E "${E_CROSS}" -O "${FIG_DIR}/mgrad_cat_cross.png"
fi

if [[ -x "./median" ]]; then
    run ./median -I "${IMG_NOISE}" -M 2 -O "${FIG_DIR}/median_sp_M2.png"
fi

if [[ -x "./ccLabel" ]]; then
    run ./ccLabel -I "${BIN_IMG}" -O "${FIG_DIR}/ccLabel_binary.png"
fi

if [[ -x "./ccAreaFilter" ]]; then
    run ./ccAreaFilter -I "${BIN_IMG}" -F 200 -O "${FIG_DIR}/ccAreaFilter_binary_F200.png"
fi

if [[ -x "./ccLabel2pass" ]]; then
    run ./ccLabel2pass -I "${BIN_IMG}" -O "${FIG_DIR}/ccLabel2pass_binary.png"
fi

echo "[OK] LaTeX-matching images generated in: ${FIG_DIR}"
echo "Upload the whole 'figures/' folder into Overleaf next to your .tex."
