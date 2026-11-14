// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Gaussian blur filter implementation using multi-scale separable convolution.
//!
//! This implementation uses a multi-scale approach for efficient blurring:
//! - **Small blurs** (σ ≤ 2): Direct separable convolution at full resolution
//! - **Large blurs** (σ > 2): Iterative downsample → blur → upsample pyramid
//!
//! The algorithm automatically determines the optimal number of decimation levels
//! using variance analysis. Each 2× decimation applies a \[1,3,3,1\]/8 binomial filter
//! (adding variance = 3.0), then downsamples, reducing the remaining blur work needed.
//! This exploits the variance additivity property: `σ²_total = σ²_downsample + σ²_blur`.
//!
//! **Variance Addition Reference**: The convolution of two Gaussians with variances σ₁²
//! and σ₂² produces a Gaussian with variance σ₁² + σ₂². This fundamental property comes
//! from probability theory and applies to Gaussian convolution in image processing. See:
//! - Torralba & Freeman, "Foundations of Computer Vision" (MIT Press), Section 2.2:
//!   <https://visionbook.mit.edu/blurring_2.html#properties-of-the-continuous-gaussian>

use super::FilterEffect;
use crate::filter::context::ScratchBuffer;
use vello_common::fearless_simd::{Level, Simd, dispatch};
use vello_common::filter::gaussian_blur::{
    DecimationAxes, DecimationSizer, GaussianBlur, MAX_KERNEL_SIZE, decimation_levels,
};
use vello_common::filter_effects::EdgeMode;
use vello_common::peniko::color::PremulRgba8;
use vello_common::pixmap::Pixmap;

use alloc::vec::Vec;

impl FilterEffect for GaussianBlur {
    fn execute_lowp(&self, pixmap: &mut Pixmap, filter_scratch: &mut ScratchBuffer) {
        if self.std_deviation_x <= 0.0 && self.std_deviation_y <= 0.0 {
            return;
        }

        let (scratch, buffers) = filter_scratch.get_blur_scratch(pixmap.width(), pixmap.height());
        apply_blur(
            pixmap,
            scratch,
            buffers,
            self.n_decimations_x,
            self.n_decimations_y,
            self.kernel_x(),
            self.kernel_y(),
            self.edge_mode,
        );
    }

    fn execute_highp(&self, pixmap: &mut Pixmap, filter_scratch: &mut ScratchBuffer) {
        // TODO: Currently only lowp is implemented and used for highp as well.
        // This needs to be updated to use proper high-precision arithmetic.
        Self::execute_lowp(self, pixmap, filter_scratch);
    }
}

/// Reusable buffers for [`apply_blur`], so that blurring does not allocate per layer.
#[derive(Debug)]
pub(crate) struct BlurBuffers {
    /// The SIMD level that the blur is compiled for.
    level: Level,
    /// Rows of f32 channel values for the convolution passes.
    floats: Vec<f32>,
    /// Rows of pixels holding the sliding window of the vertical resampling passes.
    pixels: Vec<PremulRgba8>,
}

impl Default for BlurBuffers {
    fn default() -> Self {
        Self::new(Level::try_detect().unwrap_or(Level::baseline()))
    }
}

impl BlurBuffers {
    pub(crate) fn new(level: Level) -> Self {
        Self {
            level,
            floats: Vec::new(),
            pixels: Vec::new(),
        }
    }

    /// Return an f32 row with `len` elements.
    fn float_row(&mut self, len: usize) -> &mut [f32] {
        self.floats.resize(len, 0.0);
        &mut self.floats[..len]
    }

    /// Return `N` pixel rows with `width` pixels each.
    fn pixel_rows<const N: usize>(&mut self, width: usize) -> [&mut [PremulRgba8]; N] {
        self.pixels.clear();
        self.pixels.resize(N * width, TRANSPARENT_BLACK);
        let mut rows = self.pixels.chunks_exact_mut(width.max(1));
        core::array::from_fn(|_| rows.next().unwrap())
    }
}

/// Apply Gaussian blur using multi-scale decimation and upsampling.
///
/// Uses a precomputed decimation plan and kernel per axis for optimal performance.
/// Operates in-place using a single pixmap buffer with logical dimension tracking to
/// minimize memory allocations. With no decimation levels, applies direct convolution.
///
/// The `scratch` buffer is used for separable convolution and must be at least as
/// large as the source pixmap.
pub(crate) fn apply_blur(
    pixmap: &mut Pixmap,
    scratch: &mut Pixmap,
    buffers: &mut BlurBuffers,
    n_decimations_x: usize,
    n_decimations_y: usize,
    kernel_x: &[f32],
    kernel_y: &[f32],
    edge_mode: EdgeMode,
) {
    // The passes are plain loops written to be auto-vectorized, so compile them for the
    // best available SIMD level. Everything they call must be inlined for this to apply.
    dispatch!(buffers.level, simd => simd.vectorize(
        #[inline(always)]
        || blur(
            pixmap,
            scratch,
            buffers,
            n_decimations_x,
            n_decimations_y,
            kernel_x,
            kernel_y,
            edge_mode,
        ),
    ));
}

#[inline(always)]
fn blur(
    pixmap: &mut Pixmap,
    scratch: &mut Pixmap,
    buffers: &mut BlurBuffers,
    n_decimations_x: usize,
    n_decimations_y: usize,
    kernel_x: &[f32],
    kernel_y: &[f32],
    edge_mode: EdgeMode,
) {
    let levels = decimation_levels(n_decimations_x, n_decimations_y);
    let width = pixmap.width();
    let height = pixmap.height();

    if width == 0 || height == 0 {
        return;
    }

    // Track logical dimensions through decimation (physical buffer stays the same size)
    let mut sizer = DecimationSizer::new(width, height);

    // Downsample level by level, finest first
    for axes in levels.clone() {
        let (w, h) = sizer.current();
        downscale_axes(pixmap, buffers, w, h, axes, edge_mode);
        sizer.downscale_axes(axes);
    }

    // Apply the reduced blur at the coarsest resolution
    let (w, h) = sizer.current();
    convolve(
        pixmap, scratch, buffers, w, h, kernel_x, kernel_y, edge_mode,
    );

    // Upsample back to original resolution, undoing the levels coarsest first
    for axes in levels.rev() {
        let (w, h) = sizer.current();
        upscale_axes(pixmap, buffers, w, h, axes, edge_mode);
        sizer.upscale_axes(axes);
    }

    debug_assert_eq!(
        sizer.current(),
        (width, height),
        "Final dimensions should match original"
    );
}

/// Apply separable Gaussian convolution with logical dimensions.
///
/// Performs horizontal blur followed by vertical blur, each with its own kernel; an
/// identity kernel (a single weight) skips its pass. Works with a logical view of the
/// pixmap, using only the top-left region defined by width × height. The `scratch`
/// buffer is provided by the caller to avoid allocations.
#[inline(always)]
pub(crate) fn convolve(
    src: &mut Pixmap,
    scratch: &mut Pixmap,
    buffers: &mut BlurBuffers,
    width: u16,
    height: u16,
    kernel_x: &[f32],
    kernel_y: &[f32],
    edge_mode: EdgeMode,
) {
    let radius_x = (kernel_x.len() / 2) as u8;
    let radius_y = (kernel_y.len() / 2) as u8;

    match (kernel_x.len() > 1, kernel_y.len() > 1) {
        (true, true) => {
            convolve_x(
                src, scratch, buffers, width, height, kernel_x, radius_x, edge_mode,
            );
            convolve_y(scratch, src, width, height, kernel_y, radius_y, edge_mode);
        }
        (true, false) => {
            convolve_x(
                src, scratch, buffers, width, height, kernel_x, radius_x, edge_mode,
            );
            copy_region(scratch, src, width, height);
        }
        (false, true) => {
            convolve_y(src, scratch, width, height, kernel_y, radius_y, edge_mode);
            copy_region(scratch, src, width, height);
        }
        (false, false) => {}
    }
}

/// Copy the top-left `width × height` region from `src` to `dst`.
#[inline(always)]
fn copy_region(src: &Pixmap, dst: &mut Pixmap, width: u16, height: u16) {
    let src_stride = usize::from(src.width());
    let dst_stride = usize::from(dst.width());
    let width = usize::from(width);

    for y in 0..usize::from(height) {
        let src_row = &src.data()[y * src_stride..][..width];
        dst.data_mut()[y * dst_stride..][..width].copy_from_slice(src_row);
    }
}

/// Downsample the given axes by 2x, see [`downscale`].
#[inline(always)]
fn downscale_axes(
    src: &mut Pixmap,
    buffers: &mut BlurBuffers,
    src_width: u16,
    src_height: u16,
    axes: DecimationAxes,
    edge_mode: EdgeMode,
) -> (u16, u16) {
    match axes {
        DecimationAxes::Both => downscale(src, buffers, src_width, src_height, edge_mode),
        DecimationAxes::X => {
            let dst_width = src_width.div_ceil(2);
            downscale_x(src, buffers, src_width, src_height, dst_width, edge_mode);
            (dst_width, src_height)
        }
        DecimationAxes::Y => {
            let dst_height = src_height.div_ceil(2);
            downscale_y(src, buffers, src_width, src_height, dst_height, edge_mode);
            (src_width, dst_height)
        }
    }
}

/// Upsample the given axes by 2x, see [`upscale`].
#[inline(always)]
fn upscale_axes(
    src: &mut Pixmap,
    buffers: &mut BlurBuffers,
    src_width: u16,
    src_height: u16,
    axes: DecimationAxes,
    edge_mode: EdgeMode,
) -> (u16, u16) {
    match axes {
        DecimationAxes::Both => upscale(src, buffers, src_width, src_height, edge_mode),
        DecimationAxes::X => {
            upscale_x(src, buffers, src_width, src_height, edge_mode);
            (src_width * 2, src_height)
        }
        DecimationAxes::Y => {
            upscale_y(src, buffers, src_width, src_height, edge_mode);
            (src_width, src_height * 2)
        }
    }
}

/// Apply horizontal blur pass (1D convolution along x-axis).
///
/// For each output pixel, computes a weighted sum of horizontally neighboring pixels
/// using the Gaussian kernel. Handles edge cases according to the specified edge mode.
/// Writes results to a destination buffer to avoid overwriting source data.
///
/// Each row is first widened to f32 into a buffer padded by `radius` pixels on both sides
/// according to the edge mode, so the inner loops are branch-free and vectorize over the
/// row's channels. Weights are accumulated in kernel order, so the result is identical to
/// summing each pixel's taps one by one.
#[inline(always)]
pub(crate) fn convolve_x(
    src: &Pixmap,
    dst: &mut Pixmap,
    buffers: &mut BlurBuffers,
    src_width: u16,
    src_height: u16,
    kernel: &[f32],
    radius: u8,
    edge_mode: EdgeMode,
) {
    if src_width == 0 {
        return;
    }

    let row_len = usize::from(src_width) * 4;
    let pad_len = usize::from(radius) * 4;
    let src_stride = usize::from(src.width()) * 4;
    let dst_stride = usize::from(dst.width()) * 4;
    let padded = buffers.float_row(row_len + 2 * pad_len);
    let src_data = src.data_as_u8_slice();
    let dst_data = dst.data_as_u8_slice_mut();

    for y in 0..usize::from(src_height) {
        let src_row = &src_data[y * src_stride..][..row_len];

        let (left, rest) = padded.split_at_mut(pad_len);
        let (middle, right) = rest.split_at_mut(row_len);
        for (d, &s) in middle.iter_mut().zip(src_row) {
            *d = f32::from(s);
        }
        for (i, pixel) in left.chunks_exact_mut(4).enumerate() {
            let x = i as i32 - i32::from(radius);
            pad_pixel(pixel, src_row, x, src_width, edge_mode);
        }
        for (i, pixel) in right.chunks_exact_mut(4).enumerate() {
            let x = i32::from(src_width) + i as i32;
            pad_pixel(pixel, src_row, x, src_width, edge_mode);
        }

        // Sum contributions from all kernel positions: output = Σ(weight[j] × pixel[x+j-radius])
        let dst_row = &mut dst_data[y * dst_stride..][..row_len];
        let mut blocks = dst_row.chunks_exact_mut(BLOCK);
        let mut start = 0;
        for out in &mut blocks {
            convolve_x_block::<BLOCK>(padded, kernel, start, out);
            start += BLOCK;
        }
        for out in blocks.into_remainder().chunks_exact_mut(4) {
            convolve_x_block::<4>(padded, kernel, start, out);
            start += 4;
        }
    }
}

/// Number of channel values accumulated at once by the convolution passes, small enough for the
/// accumulators to stay in registers.
const BLOCK: usize = 16;

/// Convolve the `N` channel values starting at `start` of a padded row.
#[inline(always)]
fn convolve_x_block<const N: usize>(padded: &[f32], kernel: &[f32], start: usize, out: &mut [u8]) {
    let mut acc = [0.0_f32; N];
    for (j, &k) in kernel.iter().enumerate() {
        let taps: &[f32; N] = padded[start + j * 4..][..N].try_into().unwrap();
        for (a, &p) in acc.iter_mut().zip(taps) {
            *a += p * k;
        }
    }
    for (d, &a) in out.iter_mut().zip(&acc) {
        *d = round_to_u8(a);
    }
}

/// Write the f32 channels of the source pixel at `x` (extended by the edge mode) into `pixel`.
#[inline(always)]
fn pad_pixel(pixel: &mut [f32], src_row: &[u8], x: i32, width: u16, edge_mode: EdgeMode) {
    match source_index(x, width, edge_mode) {
        Some(src_x) => {
            for (d, &s) in pixel.iter_mut().zip(&src_row[src_x * 4..][..4]) {
                *d = f32::from(s);
            }
        }
        None => pixel.fill(0.0),
    }
}

/// Apply vertical blur pass (1D convolution along y-axis).
///
/// For each output pixel, computes a weighted sum of vertically neighboring pixels
/// using the Gaussian kernel. Handles edge cases according to the specified edge mode.
/// Writes results to a destination buffer to avoid overwriting source data.
///
/// Works a row at a time: each output row accumulates whole weighted source rows, which keeps
/// memory accesses sequential and lets the inner loop vectorize. Weights are accumulated in
/// kernel order, so the result is identical to summing each pixel's taps one by one.
#[inline(always)]
pub(crate) fn convolve_y(
    src: &Pixmap,
    dst: &mut Pixmap,
    src_width: u16,
    src_height: u16,
    kernel: &[f32],
    radius: u8,
    edge_mode: EdgeMode,
) {
    if src_height == 0 {
        return;
    }

    let row_len = usize::from(src_width) * 4;
    let src_stride = usize::from(src.width()) * 4;
    let dst_stride = usize::from(dst.width()) * 4;
    let src_data = src.data_as_u8_slice();
    let dst_data = dst.data_as_u8_slice_mut();
    let mut taps = [(0_usize, 0.0_f32); MAX_KERNEL_SIZE];

    for y in 0..usize::from(src_height) {
        // Collect the source rows and weights of all kernel positions:
        // output = Σ(weight[j] × pixel[y+j-radius])
        let mut n_taps = 0;
        for (j, &k) in kernel.iter().enumerate() {
            let src_y = y as i32 + j as i32 - i32::from(radius);
            // Out-of-bounds rows are transparent black, which add nothing to the sum.
            if let Some(src_y) = source_index(src_y, src_height, edge_mode) {
                taps[n_taps] = (src_y * src_stride, k);
                n_taps += 1;
            }
        }
        let taps = &taps[..n_taps];

        let dst_row = &mut dst_data[y * dst_stride..][..row_len];
        let mut blocks = dst_row.chunks_exact_mut(BLOCK);
        let mut start = 0;
        for out in &mut blocks {
            convolve_y_block::<BLOCK>(src_data, taps, start, out);
            start += BLOCK;
        }
        for out in blocks.into_remainder().chunks_exact_mut(4) {
            convolve_y_block::<4>(src_data, taps, start, out);
            start += 4;
        }
    }
}

/// Convolve the `N` channel values starting at `start` of each tap's source row.
#[inline(always)]
fn convolve_y_block<const N: usize>(
    src_data: &[u8],
    taps: &[(usize, f32)],
    start: usize,
    out: &mut [u8],
) {
    let mut acc = [0.0_f32; N];
    for &(offset, k) in taps {
        let row: &[u8; N] = src_data[offset + start..][..N].try_into().unwrap();
        for (a, &p) in acc.iter_mut().zip(row) {
            *a += f32::from(p) * k;
        }
    }
    for (d, &a) in out.iter_mut().zip(&acc) {
        *d = round_to_u8(a);
    }
}

/// Round a non-negative channel sum to the nearest `u8`, with ties away from zero.
///
/// This gives the same result as `v.round() as u8` for the sums produced by the convolution
/// passes, but avoids a call to `roundf` (there is no rounding instruction in baseline x86-64),
/// so the conversion loops vectorize.
#[inline(always)]
fn round_to_u8(v: f32) -> u8 {
    // Adding and subtracting 2^23 rounds any value in [0, 2^23) to the nearest integer, with
    // ties to even. Ties that went down are then bumped up. `v - nearest` is exact.
    // Since the offset sum is in [2^23, 2^23 + 255], its low mantissa byte is the integer.
    const ROUNDING_OFFSET: f32 = 8_388_608.0;
    let v = v.min(255.0);
    let offset = v + ROUNDING_OFFSET;
    let nearest = offset.to_bits() as u8;
    let is_tie_rounded_down = v - (offset - ROUNDING_OFFSET) == 0.5;
    nearest + u8::from(is_tie_rounded_down)
}

/// Downsample image by 2x using separable \[1,3,3,1\]/8 binomial filter.
///
/// Performs horizontal and vertical decimation in sequence. Returns the new
/// logical dimensions (ceil(width/2), ceil(height/2)).
#[inline(always)]
pub(crate) fn downscale(
    src: &mut Pixmap,
    buffers: &mut BlurBuffers,
    src_width: u16,
    src_height: u16,
    edge_mode: EdgeMode,
) -> (u16, u16) {
    let dst_width = src_width.div_ceil(2);
    let dst_height = src_height.div_ceil(2);
    downscale_x(src, buffers, src_width, src_height, dst_width, edge_mode);
    // We can pass `dst_width` instead of `src_width` here, since we already decimated
    // horizontally.
    downscale_y(src, buffers, dst_width, src_height, dst_height, edge_mode);
    (dst_width, dst_height)
}

/// Horizontal decimation pass using \[1,3,3,1\]/8 filter.
///
/// Reduces width by 2x while applying a binomial blur kernel. The \[1,3,3,1\] weights
/// approximate a Gaussian and contribute variance=0.75 before the 2x downsampling.
#[inline(always)]
fn downscale_x(
    src: &mut Pixmap,
    buffers: &mut BlurBuffers,
    src_width: u16,
    src_height: u16,
    dst_width: u16,
    edge_mode: EdgeMode,
) {
    let stride = usize::from(src.width());
    let width = usize::from(src_width).min(stride);
    if width == 0 {
        return;
    }
    let dst_width = usize::from(dst_width);
    // Outputs in `1..interior_end` only read source pixels inside the row.
    let interior_end = ((width - 1) / 2).clamp(1, dst_width);
    let [copy] = buffers.pixel_rows::<1>(width);

    for row in src
        .data_mut()
        .chunks_exact_mut(stride)
        .take(usize::from(src_height))
    {
        // Interior outputs read from a copy of the row so they can be computed independently
        // of the outputs written before them. Since output `x` only reads source pixels from
        // `2x-1` onwards, which have not been overwritten yet, this gives the same result as
        // computing them in place.
        copy.copy_from_slice(&row[..width]);
        row[0] = downscale_pixel(row, 0, src_width, edge_mode);

        let src_bytes: &[u8] = bytemuck::cast_slice(copy);
        let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut row[1..interior_end]);
        for (i, dst) in dst_bytes.chunks_exact_mut(4).enumerate() {
            // Apply [1,3,3,1]/8 weights to pixels [x*2-1, x*2, x*2+1, x*2+2] with x = i + 1
            let p = &src_bytes[(i * 2 + 1) * 4..][..16];
            for (c, d) in dst.iter_mut().enumerate() {
                let sum = u16::from(p[c])
                    + u16::from(p[4 + c]) * 3
                    + u16::from(p[8 + c]) * 3
                    + u16::from(p[12 + c]);
                *d = ((sum + 4) >> 3) as u8;
            }
        }

        // The remaining outputs may read extended pixels, which may already have been
        // overwritten, so compute them in place in order.
        for x in interior_end..dst_width {
            row[x] = downscale_pixel(row, x, src_width, edge_mode);
        }
    }
}

/// Compute decimated pixel `x` of a row from the pixels `[x*2-1, x*2, x*2+1, x*2+2]`.
#[inline(always)]
fn downscale_pixel(row: &[PremulRgba8], x: usize, width: u16, edge_mode: EdgeMode) -> PremulRgba8 {
    let src_x = (x * 2) as i32;
    decimate_weighted(
        sample_row(row, src_x - 1, width, edge_mode),
        sample_row(row, src_x, width, edge_mode),
        sample_row(row, src_x + 1, width, edge_mode),
        sample_row(row, src_x + 2, width, edge_mode),
    )
}

/// Vertical decimation pass using \[1,3,3,1\]/8 filter.
///
/// Reduces logical height by 2x while applying a binomial blur kernel.
/// Operates in-place by writing to the beginning of the same buffer.
///
/// Works a row at a time, keeping copies of the rows in the sliding window, so that each column
/// sees exactly the same reads and writes as when processing it on its own.
#[inline(always)]
fn downscale_y(
    src: &mut Pixmap,
    buffers: &mut BlurBuffers,
    src_width: u16,
    src_height: u16,
    dst_height: u16,
    edge_mode: EdgeMode,
) {
    let stride = usize::from(src.width());
    let width = usize::from(src_width).min(stride);
    let data = src.data_mut();
    let [mut p0, mut p1, mut p2, mut p3] = buffers.pixel_rows::<4>(width);

    // Sliding window: maintains 2 previous rows (y0, y1) to form 4-tap filter
    // with current rows (y2, y3). Start with row -1 for implicit top padding.
    load_row(p0, data, stride, -1, src_height, edge_mode);
    load_row(p1, data, stride, 0, src_height, edge_mode);

    for y in 0..usize::from(dst_height) {
        // Rows [y*2-1, y*2, y*2+1, y*2+2] for the [1,3,3,1] kernel
        let src_y = (y * 2) as i32;
        load_row(p2, data, stride, src_y + 1, src_height, edge_mode);
        load_row(p3, data, stride, src_y + 2, src_height, edge_mode);

        // Apply [1,3,3,1]/8 weights → output = (p0 + 3×p1 + 3×p2 + p3) / 8
        let dst: &mut [u8] = bytemuck::cast_slice_mut(&mut data[y * stride..][..width]);
        let rows: [&[u8]; 4] = [
            bytemuck::cast_slice(p0),
            bytemuck::cast_slice(p1),
            bytemuck::cast_slice(p2),
            bytemuck::cast_slice(p3),
        ];
        for (i, d) in dst.iter_mut().enumerate() {
            let sum = u16::from(rows[0][i])
                + u16::from(rows[1][i]) * 3
                + u16::from(rows[2][i]) * 3
                + u16::from(rows[3][i]);
            *d = ((sum + 4) >> 3) as u8;
        }

        // Advance window: previous p2,p3 become next p0,p1
        core::mem::swap(&mut p0, &mut p2);
        core::mem::swap(&mut p1, &mut p3);
    }
}

/// Upsample a pixmap by 2x using linear interpolation with [0.75, 0.25] weights.
///
/// Uses separable passes: horizontal doubling followed by vertical doubling.
///
/// ## Phase Alignment Theory
///
/// The downsampling \[1,3,3,1\]/8 filter creates a half-pixel offset. Its center of mass
/// is at position `2k + 0.5`, not `2k`. This means each downsampled pixel at discrete
/// position `k` represents a sample at continuous position `2k + 0.5`.
///
/// When upsampling, we reconstruct pixels at positions `2k` and `2k+1` from downsampled
/// pixels whose centers are at `..., 2k-1.5, 2k+0.5, 2k+2.5, ...`
///
/// **Interpolation weights** are derived from linear interpolation based on distances:
/// - Position `2k`:   distance 0.5 from center at `2k+0.5`, distance 1.5 from center at `2k-1.5`
///   → weights: 0.75×pixel\[k\] + 0.25×pixel\[k-1\]
/// - Position `2k+1`: distance 0.5 from center at `2k+0.5`, distance 1.5 from center at `2k+2.5`
///   → weights: 0.75×pixel\[k\] + 0.25×pixel\[k+1\]
#[inline(always)]
pub(crate) fn upscale(
    src: &mut Pixmap,
    buffers: &mut BlurBuffers,
    src_width: u16,
    src_height: u16,
    edge_mode: EdgeMode,
) -> (u16, u16) {
    let dst_width = src_width * 2;
    let dst_height = src_height * 2;
    upscale_x(src, buffers, src_width, src_height, edge_mode);
    upscale_y(src, buffers, dst_width, src_height, edge_mode);
    (dst_width, dst_height)
}

/// Horizontal upsampling pass using [0.75, 0.25] interpolation with logical dimensions.
///
/// Doubles the logical width using phase-aligned interpolation. Each input pixel
/// generates two output pixels with different weights based on their distance from
/// the downsampled pixel's center position.
/// Operates in-place by processing backwards to avoid overwriting source data.
#[inline(always)]
fn upscale_x(
    src: &mut Pixmap,
    buffers: &mut BlurBuffers,
    src_width: u16,
    src_height: u16,
    edge_mode: EdgeMode,
) {
    let stride = usize::from(src.width());
    let width = usize::from(src_width);
    if stride == 0 || width == 0 {
        return;
    }
    let [copy] = buffers.pixel_rows::<1>(width);

    for row in src
        .data_mut()
        .chunks_exact_mut(stride)
        .take(usize::from(src_height))
    {
        // Process backwards (right to left) to avoid overwriting source data. Input pixel `x`
        // only depends on `[x-1, x, x+1]`, none of which are overwritten before it is processed
        // unless they are extended pixels, so interior pixels can be computed from a copy.
        copy.copy_from_slice(&row[..width]);
        if width > 1 {
            upscale_pixel(row, width - 1, src_width, edge_mode);
        }

        if width > 2 {
            let src_bytes: &[u8] = bytemuck::cast_slice(copy);
            let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut row[2..(width - 1) * 2]);
            for (i, dst) in dst_bytes.chunks_exact_mut(8).enumerate() {
                // Pixels [x-1, x, x+1] with x = i + 1
                let p = &src_bytes[i * 4..][..12];
                let (even, odd) = dst.split_at_mut(4);
                for c in 0..4 {
                    // output[2x]   = 0.25×p[x-1] + 0.75×p[x]
                    // output[2x+1] = 0.75×p[x]   + 0.25×p[x+1]
                    even[c] = ((u16::from(p[c]) + u16::from(p[4 + c]) * 3 + 2) >> 2) as u8;
                    odd[c] = ((u16::from(p[4 + c]) * 3 + u16::from(p[8 + c]) + 2) >> 2) as u8;
                }
            }
        }

        upscale_pixel(row, 0, src_width, edge_mode);
    }
}

/// Write the two upsampled pixels `[x*2, x*2+1]` of a row in place, from the input pixels
/// `[x-1, x, x+1]`.
#[inline(always)]
fn upscale_pixel(row: &mut [PremulRgba8], x: usize, width: u16, edge_mode: EdgeMode) {
    let prev = sample_row(row, x as i32 - 1, width, edge_mode);
    let current = sample_row(row, x as i32, width, edge_mode);
    let next = sample_row(row, x as i32 + 1, width, edge_mode);
    // Phase-aligned interpolation:
    // output[2x]   = 0.25×prev + 0.75×current  (position 2x   is 0.5 from center at 2x+0.5)
    // output[2x+1] = 0.75×current + 0.25×next  (position 2x+1 is 0.5 from center at 2x+0.5)
    row[x * 2] = interpolate_25_75(prev, current);
    // On odd physical widths the last output pixel falls outside the pixmap.
    if let Some(pixel) = row.get_mut(x * 2 + 1) {
        *pixel = interpolate_75_25(current, next);
    }
}

/// Vertical upsampling pass using [0.75, 0.25] interpolation with logical dimensions.
///
/// Doubles the logical height using phase-aligned interpolation. Each input pixel
/// generates two output pixels with different weights based on their distance from
/// the downsampled pixel's center position.
/// Operates in-place by processing backwards to avoid overwriting source data.
///
/// Works a row at a time, keeping copies of the rows in the sliding window, so that each column
/// sees exactly the same reads and writes as when processing it on its own.
#[inline(always)]
fn upscale_y(
    src: &mut Pixmap,
    buffers: &mut BlurBuffers,
    src_width: u16,
    src_height: u16,
    edge_mode: EdgeMode,
) {
    let stride = usize::from(src.width());
    let physical_height = usize::from(src.height());
    let width = usize::from(src_width).min(stride);
    let data = src.data_mut();
    let [mut p0, mut p1, mut p2] = buffers.pixel_rows::<3>(width);

    // Maintain sliding window of three rows: prev, current, next
    // This allows us to compute both output rows that depend on current row y
    let height = i32::from(src_height);
    load_row(p0, data, stride, height, src_height, edge_mode);
    load_row(p1, data, stride, height - 1, src_height, edge_mode);

    // Process backwards (bottom to top) to avoid overwriting source data
    for y in (0..usize::from(src_height)).rev() {
        load_row(p2, data, stride, y as i32 - 1, src_height, edge_mode);

        // Generate two output rows per input with phase-aligned interpolation:
        // output[2y]   = 0.25×p2 + 0.75×p1  (position 2y   is 0.5 from center at 2y+0.5)
        // output[2y+1] = 0.75×p1 + 0.25×p0  (position 2y+1 is 0.5 from center at 2y+0.5)
        let dst_y = y * 2;
        let prev: &[u8] = bytemuck::cast_slice(p2);
        let current: &[u8] = bytemuck::cast_slice(p1);
        let next: &[u8] = bytemuck::cast_slice(p0);
        let dst: &mut [u8] = bytemuck::cast_slice_mut(&mut data[dst_y * stride..][..width]);
        for ((d, &a), &b) in dst.iter_mut().zip(prev).zip(current) {
            *d = ((u16::from(a) + u16::from(b) * 3 + 2) >> 2) as u8;
        }
        // On odd physical heights the last output row falls outside the pixmap.
        if dst_y + 1 < physical_height {
            let dst: &mut [u8] =
                bytemuck::cast_slice_mut(&mut data[(dst_y + 1) * stride..][..width]);
            for ((d, &a), &b) in dst.iter_mut().zip(current).zip(next) {
                *d = ((u16::from(a) * 3 + u16::from(b) + 2) >> 2) as u8;
            }
        }

        // Advance sliding window for next iteration
        core::mem::swap(&mut p0, &mut p1);
        core::mem::swap(&mut p1, &mut p2);
    }
}

/// Copy the source row at `y` (extended by the edge mode) into `dst`.
#[inline(always)]
fn load_row(
    dst: &mut [PremulRgba8],
    data: &[PremulRgba8],
    stride: usize,
    y: i32,
    height: u16,
    edge_mode: EdgeMode,
) {
    match source_index(y, height, edge_mode) {
        Some(src_y) => dst.copy_from_slice(&data[src_y * stride..][..dst.len()]),
        None => dst.fill(TRANSPARENT_BLACK),
    }
}

/// Transparent black pixel constant.
const TRANSPARENT_BLACK: PremulRgba8 = PremulRgba8 {
    r: 0,
    g: 0,
    b: 0,
    a: 0,
};

/// Sample a pixel from a row with edge mode handling, with a fast path for in-bounds pixels.
#[inline(always)]
fn sample_row(row: &[PremulRgba8], x: i32, width: u16, edge_mode: EdgeMode) -> PremulRgba8 {
    if x >= 0 && x < i32::from(width) {
        row[x as usize]
    } else {
        sample(x, width, edge_mode, |src_x| row[usize::from(src_x)])
    }
}

/// Map a coordinate to the index of the pixel it samples, or `None` if it samples transparent
/// black (out of bounds with [`EdgeMode::None`]).
#[inline(always)]
fn source_index(coord: i32, size: u16, edge_mode: EdgeMode) -> Option<usize> {
    if edge_mode == EdgeMode::None && (coord < 0 || coord >= i32::from(size)) {
        return None;
    }
    Some(usize::from(extend(coord, size, edge_mode)))
}

/// Sample a pixel with edge mode handling (generic implementation).
///
/// Handles both horizontal and vertical sampling based on the provided closure.
/// The `sample_fn` closure receives the clamped/extended coordinate and returns the pixel.
/// For `EdgeMode::None`, returns transparent black if the coordinate is out of bounds.
#[inline(always)]
fn sample<F>(coord: i32, size: u16, edge_mode: EdgeMode, sample_fn: F) -> PremulRgba8
where
    F: FnOnce(u16) -> PremulRgba8,
{
    // For EdgeMode::None, return transparent black if out of bounds
    if edge_mode == EdgeMode::None && (coord < 0 || coord >= size as i32) {
        return TRANSPARENT_BLACK;
    }
    let extended_coord = extend(coord, size, edge_mode);
    sample_fn(extended_coord)
}

/// Extend a coordinate beyond image boundaries according to the edge mode.
///
/// Transforms out-of-bounds coordinates for sampling: clamped, wrapped, or mirrored
/// depending on the mode. For `EdgeMode::None`, the coordinate is guaranteed to be
/// in-bounds (already checked by caller, which returns transparent black for out-of-bounds).
#[inline(always)]
fn extend(coord: i32, size: u16, edge_mode: EdgeMode) -> u16 {
    match edge_mode {
        EdgeMode::Duplicate => {
            // Clamp to image bounds: pixels outside use nearest edge pixel
            coord.clamp(0, size as i32 - 1) as u16
        }
        EdgeMode::None => {
            // Coordinate is already validated as in-bounds by caller
            coord as u16
        }
        EdgeMode::Wrap => {
            // Wrap around using modulo: image tiles infinitely
            let mut c = coord % size as i32;
            if c < 0 {
                c += size as i32;
            }
            c as u16
        }
        EdgeMode::Mirror => {
            // Mirror at boundaries: image reflects across edges
            let period = size as i32 * 2;
            let mut c = coord % period;
            if c < 0 {
                c += period;
            }
            if c >= size as i32 {
                c = period - c - 1;
            }
            c as u16
        }
    }
}

/// Blend 4 RGBA pixels using \[1,3,3,1\]/8 binomial weights.
///
/// Computes `(p0 + 3×p1 + 3×p2 + p3) / 8` using efficient integer arithmetic (right shift).
/// This binomial pattern approximates a Gaussian and contributes variance=0.75 to the blur.
/// Adds 4 before the shift to implement round-to-nearest instead of floor division.
#[inline(always)]
fn decimate_weighted(
    p0: PremulRgba8,
    p1: PremulRgba8,
    p2: PremulRgba8,
    p3: PremulRgba8,
) -> PremulRgba8 {
    let r = ((p0.r as u32 + p1.r as u32 * 3 + p2.r as u32 * 3 + p3.r as u32 + 4) >> 3) as u8;
    let g = ((p0.g as u32 + p1.g as u32 * 3 + p2.g as u32 * 3 + p3.g as u32 + 4) >> 3) as u8;
    let b = ((p0.b as u32 + p1.b as u32 * 3 + p2.b as u32 * 3 + p3.b as u32 + 4) >> 3) as u8;
    let a = ((p0.a as u32 + p1.a as u32 * 3 + p2.a as u32 * 3 + p3.a as u32 + 4) >> 3) as u8;
    PremulRgba8 { r, g, b, a }
}

/// Blend 2 RGBA pixels using [0.25, 0.75] weights (right-weighted interpolation).
///
/// Used for upsampling to generate the second of two output pixels.
/// Favors the right/next pixel (p1) with 75% weight.
/// Adds 2 before the shift to implement round-to-nearest instead of floor division.
#[inline(always)]
fn interpolate_25_75(p0: PremulRgba8, p1: PremulRgba8) -> PremulRgba8 {
    let r = ((p0.r as u32 + p1.r as u32 * 3 + 2) >> 2) as u8;
    let g = ((p0.g as u32 + p1.g as u32 * 3 + 2) >> 2) as u8;
    let b = ((p0.b as u32 + p1.b as u32 * 3 + 2) >> 2) as u8;
    let a = ((p0.a as u32 + p1.a as u32 * 3 + 2) >> 2) as u8;
    PremulRgba8 { r, g, b, a }
}

/// Blend 2 RGBA pixels using [0.75, 0.25] weights.
///
/// Computes `0.75×p0 + 0.25×p1` using efficient integer arithmetic.
/// Used during upsampling for positions closer to the first pixel.
/// Adds 2 before the shift to implement round-to-nearest instead of floor division.
#[inline(always)]
fn interpolate_75_25(p0: PremulRgba8, p1: PremulRgba8) -> PremulRgba8 {
    let r = ((p0.r as u32 * 3 + p1.r as u32 + 2) >> 2) as u8;
    let g = ((p0.g as u32 * 3 + p1.g as u32 + 2) >> 2) as u8;
    let b = ((p0.b as u32 * 3 + p1.b as u32 + 2) >> 2) as u8;
    let a = ((p0.a as u32 * 3 + p1.a as u32 + 2) >> 2) as u8;
    PremulRgba8 { r, g, b, a }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vello_common::filter::gaussian_blur::{
        GaussianBlur, MAX_KERNEL_SIZE, compute_gaussian_kernel, plan_decimated_blur,
    };

    fn random_pixmap(width: u16, height: u16, seed: &mut u32) -> Pixmap {
        let mut next = || {
            // xorshift32
            *seed ^= *seed << 13;
            *seed ^= *seed >> 17;
            *seed ^= *seed << 5;
            *seed
        };
        let mut pixmap = Pixmap::new(width, height);
        for pixel in pixmap.data_mut() {
            let bits = next();
            // Mix fully transparent, fully opaque and partially transparent pixels.
            let a = match bits % 4 {
                0 => 0,
                1 => 255,
                _ => (bits >> 8) as u8,
            };
            let channel = |shift: u32| ((((bits >> shift) & 0xff) * u32::from(a)) / 255) as u8;
            *pixel = PremulRgba8 {
                r: channel(16),
                g: channel(24),
                b: channel(4),
                a,
            };
        }
        pixmap
    }

    /// The optimized blur must be bit-identical to the original scalar implementation.
    #[test]
    fn test_matches_reference_implementation() {
        use crate::filter::gaussian_blur_reference as reference;

        let mut seed = 0x1234_5678;
        // Reuse one set of buffers across sizes, as the filter context does across layers.
        let mut buffers = BlurBuffers::default();
        let edge_modes = [
            EdgeMode::None,
            EdgeMode::Duplicate,
            EdgeMode::Wrap,
            EdgeMode::Mirror,
        ];
        let sizes: [(u16, u16); 10] = [
            (1, 1),
            (2, 2),
            (1, 9),
            (7, 3),
            (13, 11),
            (16, 16),
            (60, 60),
            (64, 52),
            (34, 70),
            (130, 18),
        ];

        let std_deviations = [0.3, 0.8, 1.0, 1.7, 2.0, 2.5, 3.0, 4.0, 6.5, 8.0, 20.0];
        // Isotropic blurs, plus anisotropic ones including blurs along a single axis.
        let mut std_deviation_pairs: Vec<(f32, f32)> =
            std_deviations.iter().map(|&s| (s, s)).collect();
        std_deviation_pairs.extend([
            (0.0, 1.7),
            (2.5, 0.0),
            (0.0, 8.0),
            (20.0, 0.0),
            (1.0, 6.5),
            (8.0, 2.0),
            (3.0, 20.0),
        ]);

        for (width, height) in sizes {
            for &(std_deviation_x, std_deviation_y) in &std_deviation_pairs {
                let blur = GaussianBlur::new(std_deviation_x, std_deviation_y, EdgeMode::None);
                // The original implementation writes out of bounds when upscaling to an odd
                // physical size, so only compare decimated blurs on even sizes.
                if (blur.n_decimations_x > 0 && width % 2 == 1)
                    || (blur.n_decimations_y > 0 && height % 2 == 1)
                {
                    continue;
                }

                for edge_mode in edge_modes {
                    let source = random_pixmap(width, height, &mut seed);
                    let mut expected = source.clone();
                    let mut actual = source;
                    // Use a larger scratch buffer, as the filter context may have one from a
                    // previous, larger layer.
                    reference::apply_blur(
                        &mut expected,
                        &mut Pixmap::new(width + 3, height + 1),
                        blur.n_decimations_x,
                        blur.n_decimations_y,
                        blur.kernel_x(),
                        blur.kernel_y(),
                        edge_mode,
                    );
                    apply_blur(
                        &mut actual,
                        &mut Pixmap::new(width + 3, height + 1),
                        &mut buffers,
                        blur.n_decimations_x,
                        blur.n_decimations_y,
                        blur.kernel_x(),
                        blur.kernel_y(),
                        edge_mode,
                    );

                    assert!(
                        actual.data() == expected.data(),
                        "blur differs from the reference for {width}x{height}, \
                         σ=({std_deviation_x}, {std_deviation_y}), {edge_mode:?}"
                    );
                }
            }
        }
    }

    /// Rounding must match `f32::round` (ties away from zero) for every reachable value.
    #[test]
    fn test_round_to_u8_matches_round() {
        let check = |v: f32| assert_eq!(round_to_u8(v), v.round() as u8, "rounding {v}");
        // Values around every integer and every tie, where the rounding direction changes.
        for n in 0..=256_u16 {
            for center in [f32::from(n), f32::from(n) + 0.5] {
                let (mut below, mut above) = (center, center);
                for _ in 0..64 {
                    check(below);
                    check(above);
                    below = below.next_down().max(0.0);
                    above = above.next_up();
                }
            }
        }
        // A dense sweep over the whole range.
        for i in 0..=2_570_000_u32 {
            check(i as f32 * 1e-4);
        }
    }

    /// Test edge extension with Duplicate mode.
    #[test]
    fn test_extend_duplicate() {
        let size = 10;

        // In-bounds coordinate
        assert_eq!(extend(5, size, EdgeMode::Duplicate), 5);

        // Below bounds: clamp to 0
        assert_eq!(extend(-1, size, EdgeMode::Duplicate), 0);
        assert_eq!(extend(-10, size, EdgeMode::Duplicate), 0);

        // Above bounds: clamp to size-1
        assert_eq!(extend(10, size, EdgeMode::Duplicate), 9);
        assert_eq!(extend(20, size, EdgeMode::Duplicate), 9);
    }

    /// Test edge extension with Wrap mode.
    #[test]
    fn test_extend_wrap() {
        let size = 10;

        // In-bounds: identity
        assert_eq!(extend(5, size, EdgeMode::Wrap), 5);

        // Above bounds: wrap around
        assert_eq!(extend(10, size, EdgeMode::Wrap), 0);
        assert_eq!(extend(11, size, EdgeMode::Wrap), 1);
        assert_eq!(extend(25, size, EdgeMode::Wrap), 5);

        // Below bounds: wrap from end
        assert_eq!(extend(-1, size, EdgeMode::Wrap), 9);
        assert_eq!(extend(-2, size, EdgeMode::Wrap), 8);
    }

    /// Test edge extension with Mirror mode.
    #[test]
    fn test_extend_mirror() {
        let size = 10;

        // In-bounds: identity
        assert_eq!(extend(5, size, EdgeMode::Mirror), 5);

        // Just past boundary: mirror back
        // For coord=10: period=20, c=10, c>=10 so c = 20-10-1 = 9
        assert_eq!(extend(10, size, EdgeMode::Mirror), 9);
        // For coord=11: period=20, c=11, c>=10 so c = 20-11-1 = 8
        assert_eq!(extend(11, size, EdgeMode::Mirror), 8);

        // Full reflection cycle (period = 2*size = 20)
        // For coord=19: c=19, c>=10 so c = 20-19-1 = 0
        assert_eq!(extend(19, size, EdgeMode::Mirror), 0);
        // For coord=20: c=20%20=0, c<10 so c=0
        assert_eq!(extend(20, size, EdgeMode::Mirror), 0);

        // Negative coordinates
        // For coord=-1: c=-1%20=-1, c<0 so c+=20 → c=19, c>=10 so c=20-19-1=0
        assert_eq!(extend(-1, size, EdgeMode::Mirror), 0);
        // For coord=-2: c=-2%20=-2, c<0 so c+=20 → c=18, c>=10 so c=20-18-1=1
        assert_eq!(extend(-2, size, EdgeMode::Mirror), 1);
    }

    /// Test edge extension with None mode (coordinate should pass through).
    #[test]
    fn test_extend_none() {
        let size = 10;
        // None mode just passes the coordinate as u16
        assert_eq!(extend(5, size, EdgeMode::None), 5);
        assert_eq!(extend(9, size, EdgeMode::None), 9);
        assert_eq!(extend(10, size, EdgeMode::None), 10);
        assert_eq!(extend(11, size, EdgeMode::None), 11);
    }

    /// Test [1,3,3,1]/8 decimation weights.
    #[test]
    fn test_decimate_weighted() {
        let p0 = PremulRgba8 {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        };
        let p1 = PremulRgba8 {
            r: 8,
            g: 8,
            b: 8,
            a: 8,
        };
        let p2 = PremulRgba8 {
            r: 8,
            g: 8,
            b: 8,
            a: 8,
        };
        let p3 = PremulRgba8 {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        };

        // (0 + 3*8 + 3*8 + 0 + 4) / 8 = (48 + 4) / 8 = 52 / 8 = 6.5 → 6 (rounds down)
        let result = decimate_weighted(p0, p1, p2, p3);
        assert_eq!(result.r, 6);
        assert_eq!(result.g, 6);
        assert_eq!(result.b, 6);
        assert_eq!(result.a, 6);
    }

    /// Test [1,3,3,1]/8 with all same values (should be identity).
    #[test]
    fn test_decimate_weighted_uniform() {
        let p = PremulRgba8 {
            r: 100,
            g: 150,
            b: 200,
            a: 255,
        };
        let result = decimate_weighted(p, p, p, p);
        // All same: (100 + 300 + 300 + 100 + 4) / 8 = 804/8 = 100.5 → 100 (rounds down)
        assert_eq!(result.r, 100);
        assert_eq!(result.g, 150);
        assert_eq!(result.b, 200);
        assert_eq!(result.a, 255);
    }

    /// Test [0.25, 0.75] interpolation.
    #[test]
    fn test_interpolate_25_75() {
        let p0 = PremulRgba8 {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        };
        let p1 = PremulRgba8 {
            r: 100,
            g: 100,
            b: 100,
            a: 100,
        };

        // (0 + 300 + 2) / 4 = 302 / 4 = 75.5 → 75 (rounds down)
        let result = interpolate_25_75(p0, p1);
        assert_eq!(result.r, 75);
        assert_eq!(result.g, 75);
        assert_eq!(result.b, 75);
        assert_eq!(result.a, 75);
    }

    /// Test [0.75, 0.25] interpolation.
    #[test]
    fn test_interpolate_75_25() {
        let p0 = PremulRgba8 {
            r: 100,
            g: 100,
            b: 100,
            a: 100,
        };
        let p1 = PremulRgba8 {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        };

        // (300 + 0 + 2) / 4 = 302 / 4 = 75.5 → 75 (rounds down)
        let result = interpolate_75_25(p0, p1);
        assert_eq!(result.r, 75);
        assert_eq!(result.g, 75);
        assert_eq!(result.b, 75);
        assert_eq!(result.a, 75);
    }

    /// Test interpolation symmetry.
    #[test]
    fn test_interpolation_symmetry() {
        let p0 = PremulRgba8 {
            r: 50,
            g: 100,
            b: 150,
            a: 200,
        };
        let p1 = PremulRgba8 {
            r: 200,
            g: 150,
            b: 100,
            a: 50,
        };

        let r1 = interpolate_25_75(p0, p1);
        let r2 = interpolate_75_25(p1, p0);

        // Should be symmetric
        assert!((r1.r as i32 - r2.r as i32).abs() <= 0);
        assert!((r1.g as i32 - r2.g as i32).abs() <= 0);
        assert!((r1.b as i32 - r2.b as i32).abs() <= 0);
        assert!((r1.a as i32 - r2.a as i32).abs() <= 0);
    }

    /// Test that very small image sizes don't panic.
    #[test]
    fn test_small_image_sizes() {
        let mut pixmap = Pixmap::new(1, 1);
        let (n_decimations, kernel, kernel_size) = plan_decimated_blur(2.0);

        // Should not panic
        let result = std::panic::catch_unwind(move || {
            let mut scratch = Pixmap::new(1, 1);
            let kernel = &kernel[..usize::from(kernel_size)];
            apply_blur(
                &mut pixmap,
                &mut scratch,
                &mut BlurBuffers::default(),
                n_decimations,
                n_decimations,
                kernel,
                kernel,
                EdgeMode::None,
            );
        });

        assert!(result.is_ok());
    }

    /// A blur along one axis only must leave the other axis untouched, with and
    /// without decimation.
    #[test]
    fn test_single_axis_blur_keeps_other_axis_sharp() {
        for std_deviation in [1.5, 6.0] {
            let blur = GaussianBlur::new(std_deviation, 0.0, EdgeMode::None);
            let mut pixmap = Pixmap::new(32, 32);
            let mut scratch = Pixmap::new(32, 32);
            // One opaque row in the middle.
            for x in 0..32 {
                pixmap.set_pixel(
                    x,
                    16,
                    PremulRgba8 {
                        r: 255,
                        g: 0,
                        b: 0,
                        a: 255,
                    },
                );
            }

            apply_blur(
                &mut pixmap,
                &mut scratch,
                &mut BlurBuffers::default(),
                blur.n_decimations_x,
                blur.n_decimations_y,
                blur.kernel_x(),
                blur.kernel_y(),
                EdgeMode::None,
            );

            for y in 0..32 {
                for x in 0..32 {
                    let alpha = pixmap.sample(x, y).a;
                    if y == 16 {
                        assert!(alpha > 0, "σ={std_deviation}: row stayed opaque at x={x}");
                    } else {
                        assert_eq!(alpha, 0, "σ={std_deviation}: row {y} was blurred into");
                    }
                }
            }
        }
    }

    /// A blur along the y-axis only is the transposed x-only blur.
    #[test]
    fn test_axis_blurs_are_transposes() {
        let mut horizontal = Pixmap::new(24, 24);
        let mut vertical = Pixmap::new(24, 24);
        horizontal.set_pixel(
            12,
            12,
            PremulRgba8 {
                r: 255,
                g: 0,
                b: 0,
                a: 255,
            },
        );
        vertical.set_pixel(
            12,
            12,
            PremulRgba8 {
                r: 255,
                g: 0,
                b: 0,
                a: 255,
            },
        );
        let mut scratch = Pixmap::new(24, 24);

        let blur_x = GaussianBlur::new(4.0, 0.0, EdgeMode::None);
        apply_blur(
            &mut horizontal,
            &mut scratch,
            &mut BlurBuffers::default(),
            blur_x.n_decimations_x,
            blur_x.n_decimations_y,
            blur_x.kernel_x(),
            blur_x.kernel_y(),
            EdgeMode::None,
        );
        let blur_y = GaussianBlur::new(0.0, 4.0, EdgeMode::None);
        apply_blur(
            &mut vertical,
            &mut scratch,
            &mut BlurBuffers::default(),
            blur_y.n_decimations_x,
            blur_y.n_decimations_y,
            blur_y.kernel_x(),
            blur_y.kernel_y(),
            EdgeMode::None,
        );

        for y in 0..24 {
            for x in 0..24 {
                assert_eq!(
                    horizontal.sample(x, y),
                    vertical.sample(y, x),
                    "at ({x}, {y})"
                );
            }
        }
    }

    /// Test downscale with odd dimensions.
    #[test]
    fn test_downscale_odd_dimensions() {
        let mut pixmap = Pixmap::new(5, 5);
        // Fill with white
        for y in 0..5 {
            for x in 0..5 {
                pixmap.set_pixel(
                    x,
                    y,
                    PremulRgba8 {
                        r: 255,
                        g: 255,
                        b: 255,
                        a: 255,
                    },
                );
            }
        }

        let (new_width, new_height) = downscale(
            &mut pixmap,
            &mut BlurBuffers::default(),
            5,
            5,
            EdgeMode::Duplicate,
        );
        // 5 / 2 = 2.5 → ceil = 3
        assert_eq!(new_width, 3);
        assert_eq!(new_height, 3);
    }

    /// Test upscale dimensions.
    #[test]
    fn test_upscale_dimensions() {
        let mut pixmap = Pixmap::new(6, 6);
        let (new_width, new_height) = upscale(
            &mut pixmap,
            &mut BlurBuffers::default(),
            3,
            3,
            EdgeMode::Duplicate,
        );
        // 3 * 2 = 6
        assert_eq!(new_width, 6);
        assert_eq!(new_height, 6);
    }

    /// Test that horizontal convolution preserves uniform colors.
    #[test]
    fn test_convolve_x_uniform() {
        let mut src = Pixmap::new(5, 3);
        let mut dst = Pixmap::new(5, 3);
        // Fill with uniform gray
        src.data_mut().fill(PremulRgba8 {
            r: 128,
            g: 128,
            b: 128,
            a: 255,
        });

        let (kernel, kernel_size) = compute_gaussian_kernel(1.0);
        convolve_x(
            &src,
            &mut dst,
            &mut BlurBuffers::default(),
            5,
            3,
            &kernel[..usize::from(kernel_size)],
            kernel_size / 2,
            EdgeMode::Duplicate,
        );

        // Uniform input should produce uniform output
        for y in 0..3 {
            for x in 0..5 {
                let pixel = dst.sample(x, y);
                assert_eq!(
                    pixel,
                    PremulRgba8 {
                        r: 128,
                        g: 128,
                        b: 128,
                        a: 255,
                    }
                );
            }
        }
    }

    /// Test that vertical convolution preserves uniform colors.
    #[test]
    fn test_convolve_y_uniform() {
        let mut src = Pixmap::new(3, 5);
        let mut dst = Pixmap::new(3, 5);
        // Fill with uniform gray
        src.data_mut().fill(PremulRgba8 {
            r: 128,
            g: 128,
            b: 128,
            a: 255,
        });

        let (kernel, kernel_size) = compute_gaussian_kernel(1.0);
        convolve_y(
            &src,
            &mut dst,
            3,
            5,
            &kernel[..usize::from(kernel_size)],
            kernel_size / 2,
            EdgeMode::Duplicate,
        );

        // Uniform input should produce uniform output
        for y in 0..5 {
            for x in 0..3 {
                let pixel = dst.sample(x, y);
                assert_eq!(
                    pixel,
                    PremulRgba8 {
                        r: 128,
                        g: 128,
                        b: 128,
                        a: 255,
                    }
                );
            }
        }
    }

    /// Test that convolution with identity kernel is a no-op.
    #[test]
    fn test_convolve_identity_kernel() {
        let mut src = Pixmap::new(3, 3);
        let mut dst = Pixmap::new(3, 3);
        src.set_pixel(
            1,
            1,
            PremulRgba8 {
                r: 255,
                g: 100,
                b: 50,
                a: 200,
            },
        );

        let kernel = [1.0]; // Identity kernel
        convolve_x(
            &src,
            &mut dst,
            &mut BlurBuffers::default(),
            3,
            3,
            &kernel,
            0,
            EdgeMode::None,
        );

        // Should be unchanged
        assert_eq!(dst.sample(1, 1), src.sample(1, 1));
    }

    /// Test that large sigma values are clamped to `MAX_KERNEL_SIZE`.
    #[test]
    fn test_large_sigma_clamped_to_max() {
        let (kernel, kernel_size) = compute_gaussian_kernel(100.0);
        // For σ=100, radius = ceil(300) = 300, size would be 601
        // But it should be clamped to MAX_KERNEL_SIZE
        assert_eq!(usize::from(kernel_size), MAX_KERNEL_SIZE);

        // The clamped kernel should still sum to 1.0 (normalized)
        let sum: f32 = kernel.iter().take(usize::from(kernel_size)).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    /// Test that decimation prevents kernel from exceeding `MAX_KERNEL_SIZE`.
    #[test]
    fn test_decimation_reduces_kernel_size() {
        let (n_decimations, _kernel, kernel_size) = plan_decimated_blur(100.0);
        assert_eq!(kernel_size, 11);
        assert_eq!(n_decimations, 6, "Large sigma should trigger decimation");
    }

    /// Test sampling behavior at exact boundaries for each edge mode.
    #[test]
    fn test_sample_x_at_boundaries() {
        let mut pixmap = Pixmap::new(3, 1);
        pixmap.set_pixel(
            0,
            0,
            PremulRgba8 {
                r: 10,
                g: 0,
                b: 0,
                a: 255,
            },
        );
        pixmap.set_pixel(
            1,
            0,
            PremulRgba8 {
                r: 20,
                g: 0,
                b: 0,
                a: 255,
            },
        );
        pixmap.set_pixel(
            2,
            0,
            PremulRgba8 {
                r: 30,
                g: 0,
                b: 0,
                a: 255,
            },
        );

        // Test left boundary with Duplicate
        let mut p = sample_row(pixmap.data(), -1, 3, EdgeMode::Duplicate);
        assert_eq!(p.r, 10); // Should clamp to first pixel

        // Test right boundary with Duplicate
        p = sample_row(pixmap.data(), 3, 3, EdgeMode::Duplicate);
        assert_eq!(p.r, 30); // Should clamp to last pixel

        // Test with None mode (should return transparent black)
        p = sample_row(pixmap.data(), -1, 3, EdgeMode::None);
        assert_eq!(p.a, 0);
    }

    /// Test sampling behavior at exact boundaries for vertical sampling.
    #[test]
    fn test_sample_y_at_boundaries() {
        let mut pixmap = Pixmap::new(1, 3);
        pixmap.set_pixel(
            0,
            0,
            PremulRgba8 {
                r: 10,
                g: 0,
                b: 0,
                a: 255,
            },
        );
        pixmap.set_pixel(
            0,
            1,
            PremulRgba8 {
                r: 20,
                g: 0,
                b: 0,
                a: 255,
            },
        );
        pixmap.set_pixel(
            0,
            2,
            PremulRgba8 {
                r: 30,
                g: 0,
                b: 0,
                a: 255,
            },
        );

        // Test top boundary with Duplicate
        let mut p = sample_row(pixmap.data(), -1, 3, EdgeMode::Duplicate);
        assert_eq!(p.r, 10); // Should clamp to first pixel

        // Test bottom boundary with Duplicate
        p = sample_row(pixmap.data(), 3, 3, EdgeMode::Duplicate);
        assert_eq!(p.r, 30); // Should clamp to last pixel

        // Test with None mode (should return transparent black)
        p = sample_row(pixmap.data(), -1, 3, EdgeMode::None);
        assert_eq!(p.a, 0);
    }

    /// Test that kernel normalization is precise for various sigma values.
    #[test]
    fn test_kernel_normalization_precision() {
        for sigma in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let (kernel, kernel_size) = compute_gaussian_kernel(sigma);
            let sum: f32 = kernel.iter().take(usize::from(kernel_size)).sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Kernel for σ={} not normalized: sum={}",
                sigma,
                sum
            );
        }
    }

    /// Test that downscale → upscale preserves dimensions.
    #[test]
    fn test_downscale_upscale_roundtrip() {
        let mut pixmap = Pixmap::new(8, 8);
        // Fill with a pattern
        pixmap.data_mut().fill(PremulRgba8 {
            r: 128,
            g: 128,
            b: 128,
            a: 255,
        });

        let (w1, h1) = downscale(
            &mut pixmap,
            &mut BlurBuffers::default(),
            8,
            8,
            EdgeMode::Duplicate,
        );
        assert_eq!(w1, 4);
        assert_eq!(h1, 4);

        let (w2, h2) = upscale(
            &mut pixmap,
            &mut BlurBuffers::default(),
            w1,
            h1,
            EdgeMode::Duplicate,
        );
        assert_eq!(w2, 8);
        assert_eq!(h2, 8);
    }

    fn pixmap_from_red(width: u16, height: u16, values: &[&[u8]]) -> Pixmap {
        let mut pixmap = Pixmap::new(width, height);
        for (y, row) in values.iter().enumerate() {
            for (x, &r) in row.iter().enumerate() {
                pixmap.set_pixel(
                    x as u16,
                    y as u16,
                    PremulRgba8 {
                        r,
                        g: 0,
                        b: 0,
                        a: 255,
                    },
                );
            }
        }
        pixmap
    }

    fn assert_red_values<const W: usize, const H: usize>(pixmap: &Pixmap, expected: [[u8; W]; H]) {
        for (y, row) in expected.iter().enumerate() {
            for (x, &want) in row.iter().enumerate() {
                let got = pixmap.sample(x as u16, y as u16).r;

                assert_eq!(
                    got, want,
                    "red mismatch at ({x}, {y}): got {got}, expected {want}"
                );
            }
        }
    }

    #[test]
    fn test_downscale_x_non_uniform() {
        //   (0,0)=0    (1,0)=40   (2,0)=80   (3,0)=120
        //   (0,1)=20   (1,1)=60   (2,1)=100  (3,1)=140
        let mut pixmap = pixmap_from_red(4, 2, &[&[0, 40, 80, 120], &[20, 60, 100, 140]]);

        let dst_width = 4_u16.div_ceil(2);
        downscale_x(
            &mut pixmap,
            &mut BlurBuffers::default(),
            4,
            2,
            dst_width,
            EdgeMode::Duplicate,
        );

        // Row 0: p[-1]=0,p[0]=0,p[1]=40,p[2]=80    → (0+0*3+40*3+80+4)>>3       = 25
        //        p[1]=40,p[2]=80,p[3]=120,p[4]=120  → (40+80*3+120*3+120+4)>>3   = 95
        // Row 1: p[-1]=20,p[0]=20,p[1]=60,p[2]=100  → (20+20*3+60*3+100+4)>>3    = 45
        //        p[1]=60,p[2]=100,p[3]=140,p[4]=140  → (60+100*3+140*3+140+4)>>3   = 115
        assert_red_values(&pixmap, [[25, 95], [45, 115]]);
    }

    #[test]
    fn test_downscale_y_non_uniform() {
        // Use a 2x4 image (output of a prior downscale_x).
        //   (0,0)=25   (1,0)=95
        //   (0,1)=45   (1,1)=115
        //   (0,2)=35   (1,2)=105
        //   (0,3)=55   (1,3)=125
        let mut pixmap = pixmap_from_red(2, 4, &[&[25, 95], &[45, 115], &[35, 105], &[55, 125]]);

        let dst_height = 4_u16.div_ceil(2);
        downscale_y(
            &mut pixmap,
            &mut BlurBuffers::default(),
            2,
            4,
            dst_height,
            EdgeMode::Duplicate,
        );

        // Col 0: p[-1]=25,p[0]=25,p[1]=45,p[2]=35 → (25+25*3+45*3+35+4)>>3   = 34
        //        p[1]=45,p[2]=35,p[3]=55,p[4]=55   → (45+35*3+55*3+55+4)>>3  = 46
        // Col 1: p[-1]=95,p[0]=95,p[1]=115,p[2]=105 → (95+95*3+115*3+105+4)>>3 = 104
        //        p[1]=115,p[2]=105,p[3]=125,p[4]=125 → (115+105*3+125*3+125+4)>>3 = 116
        assert_red_values(&pixmap, [[34, 104], [46, 116]]);
    }

    #[test]
    fn test_upscale_x_non_uniform() {
        let mut pixmap = pixmap_from_red(4, 2, &[&[34, 104], &[46, 116]]);

        upscale_x(
            &mut pixmap,
            &mut BlurBuffers::default(),
            2,
            2,
            EdgeMode::Duplicate,
        );

        // Row 0 [34, 104]:
        //   x=0: interp25_75(34,34)=34,  interp75_25(34,104)=(34*3+104+2)>>2 = 52
        //   x=1: interp25_75(34,104)=(34+104*3+2)>>2=87, interp75_25(104,104)=104
        // Row 1 [46, 116]:
        //   x=0: 46, interp75_25(46,116)=(46*3+116+2)>>2=64
        //   x=1: interp25_75(46,116)=(46+116*3+2)>>2=99, 116
        assert_red_values(&pixmap, [[34, 52, 87, 104], [46, 64, 99, 116]]);
    }

    #[test]
    fn test_upscale_y_non_uniform() {
        let mut pixmap = pixmap_from_red(4, 4, &[&[34, 52, 87, 104], &[46, 64, 99, 116]]);

        upscale_y(
            &mut pixmap,
            &mut BlurBuffers::default(),
            4,
            2,
            EdgeMode::Duplicate,
        );

        #[rustfmt::skip]
        assert_red_values(&pixmap, [
            [ 34,  52,  87, 104],
            [ 37,  55,  90, 107],
            [ 43,  61,  96, 113],
            [ 46,  64,  99, 116],
        ]);
    }
}
