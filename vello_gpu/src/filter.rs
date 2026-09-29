// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! GPU filter encoding and executable pass planning.

use crate::copy::GpuCopyInstance;
use crate::schedule::round::FilterOp;
use crate::util::pack_u16_pair;
use alloc::vec::Vec;
use bytemuck::{Pod, Zeroable};
use core::ops::Range;
use vello_common::filter::color_matrix::ColorMatrix;
use vello_common::filter::drop_shadow::DropShadow;
use vello_common::filter::flood::Flood;
use vello_common::filter::gaussian_blur::{DecimationSizer, GaussianBlur, MAX_KERNEL_SIZE};
use vello_common::filter::offset::Offset;
use vello_common::filter::{FilterData, PreparedFilter};
use vello_common::filter_effects::EdgeMode;
use vello_common::geometry::{RectU16, SizeU16};
use vello_common::util::{Clear, RetainVec};

/// How much transparent padding to reserve for filter layers within the image. Needed so
/// that the various shader programs can assume transparent pixels on the outside, making
/// the code significantly easier since we don't need to special-case border pixels. Since we
/// do use checked accesses for the offset filter, the bottleneck is formed by the gaussian blur
/// convolution.
///
/// Keep this in sync with `FILTER_ATLAS_PADDING` in `filter.wesl`!
#[expect(clippy::cast_possible_truncation, reason = "safe in this case")]
pub(crate) const FILTER_ATLAS_PADDING: u16 = MAX_KERNEL_SIZE as u16 / 2;

// Note: Keep these variables and struct layouts in sync with `filter.wesl`!

// Since we store in RGBA32 texture.
const BYTES_PER_TEXEL: usize = 16;
const COMPOSITE_ORIGINAL_SHIFT: u32 = 13;
const COMPOSITE_ORIGINAL_MASK: u32 = 1 << COMPOSITE_ORIGINAL_SHIFT;

// The shader reads the parameters of each filter by texel index.
const _: () = assert!(
    size_of::<GpuOffset>() == BYTES_PER_TEXEL,
    "offset must span one texel"
);
const _: () = assert!(
    size_of::<GpuFlood>() == BYTES_PER_TEXEL,
    "flood must span one texel"
);
const _: () = assert!(
    size_of::<GpuGaussianBlur>() == 2 * BYTES_PER_TEXEL,
    "gaussian blur must span two texels"
);
const _: () = assert!(
    size_of::<GpuDropShadow>() == 3 * BYTES_PER_TEXEL,
    "drop shadow must span three texels"
);
const _: () = assert!(
    size_of::<GpuColorMatrix>() == 6 * BYTES_PER_TEXEL,
    "color matrix must span six texels"
);

pub(crate) mod filter_type {
    pub(crate) const OFFSET: u32 = 0;
    pub(crate) const FLOOD: u32 = 1;
    pub(crate) const GAUSSIAN_BLUR: u32 = 2;
    pub(crate) const DROP_SHADOW: u32 = 3;
    pub(crate) const COLOR_MATRIX: u32 = 4;
}

pub(crate) mod edge_mode {
    pub(crate) const DUPLICATE: u32 = 0;
    pub(crate) const WRAP: u32 = 1;
    pub(crate) const MIRROR: u32 = 2;
    pub(crate) const NONE: u32 = 3;
}

pub(crate) mod pass_kind {
    pub(crate) const COPY: u32 = 0;
    pub(crate) const FLOOD: u32 = 1;
    pub(crate) const OFFSET: u32 = 2;
    pub(crate) const DOWNSCALE: u32 = 3;
    pub(crate) const BLUR_H: u32 = 4;
    pub(crate) const BLUR_V: u32 = 5;
    pub(crate) const UPSCALE: u32 = 6;
    pub(crate) const COMPOSITE_DROP_SHADOW: u32 = 7;
    pub(crate) const COLORIZE: u32 = 8;
    pub(crate) const COLOR_MATRIX: u32 = 9;
}

pub(crate) fn edge_mode_to_gpu(mode: EdgeMode) -> u32 {
    match mode {
        EdgeMode::Duplicate => edge_mode::DUPLICATE,
        EdgeMode::Wrap => edge_mode::WRAP,
        EdgeMode::Mirror => edge_mode::MIRROR,
        EdgeMode::None => edge_mode::NONE,
    }
}

fn pack_header(filter_type: u32) -> u32 {
    debug_assert!(filter_type <= 31, "filter_type must fit in 5 bits");

    filter_type
}

const fn pack_header_with_gaussian_params(
    filter_type: u32,
    edge_mode: u32,
    n_decimations: u32,
    n_linear_taps: u32,
) -> u32 {
    debug_assert!(filter_type <= 31, "filter_type must fit in 5 bits");
    debug_assert!(edge_mode <= 3, "edge_mode must fit in 2 bits");
    debug_assert!(n_decimations <= 15, "n_decimations must fit in 4 bits");
    debug_assert!(n_linear_taps <= 3, "n_linear_taps must fit in 2 bits");

    filter_type | (edge_mode << 5) | (n_decimations << 7) | (n_linear_taps << 11)
}

const _: () = assert!(
    pack_header_with_gaussian_params(31, 3, 15, 3) & COMPOSITE_ORIGINAL_MASK == 0,
    "Gaussian filter parameters overlap the composite_original bit"
);

// To a large degree, the vello_gpu implementation of gaussian blur follows the one in vello_cpu.
// However, we apply a specific optimization, where instead of averaging and weighting each sample
// one after the other, we use linear sampling to sample two pixels at once, and adjust the
// weights accordingly so the gaussian blur filter is still valid. See
// https://www.rastergrid.com/blog/2010/09/efficient-gaussian-blur-with-linear-sampling/
// for more information.

/// Maximum number of linear-sampling tap pairs per side.
const MAX_TAPS_PER_SIDE: usize = (MAX_KERNEL_SIZE / 2).div_ceil(2);

/// A linear-sampling kernel derived from a discrete Gaussian kernel.
struct LinearKernel {
    /// Weight of the center tap.
    center_weight: f32,
    // Note that we only need to store one side since they are symmetrical.
    /// Merged weights for each tap pair. Only the first `n_taps` entries are valid.
    weights: [f32; MAX_TAPS_PER_SIDE],
    /// The fractional offsets for each tap pair for linear sampling. Only the first `n_taps` entries are valid.
    offsets: [f32; MAX_TAPS_PER_SIDE],
    /// The actual number of taps per side.
    n_taps: u8,
}

impl LinearKernel {
    fn new(kernel: &[f32; MAX_KERNEL_SIZE], kernel_size: u8) -> Self {
        let kernel_size = kernel_size as usize;
        let radius = kernel_size / 2;
        let center_weight = kernel[radius];

        let mut weights = [0.0_f32; MAX_TAPS_PER_SIDE];
        let mut offsets = [0.0_f32; MAX_TAPS_PER_SIDE];
        let mut n_taps = 0_u8;

        // The kernel is symmetric, so we can only process the positive side.
        let positive_side = &kernel[radius + 1..kernel_size];
        let (pairs, remainder) = positive_side.as_chunks::<2>();

        // Merge each consecutive pair into a single bilinear tap. See the
        // formulas on the website linked above.
        for (k, &[w1, w2]) in pairs.iter().enumerate() {
            let merged_weight = w1 + w2;
            let offset1 = (2 * k + 1) as f32;
            let merged_offset = if merged_weight > 0.0 {
                (w1 * offset1 + w2 * (offset1 + 1.0)) / merged_weight
            } else {
                offset1
            };
            weights[n_taps as usize] = merged_weight;
            offsets[n_taps as usize] = merged_offset;
            n_taps += 1;
        }

        // If there is a leftover tap, we sample with no fractional offset so that just
        // that single pixel is sampled fully.
        if let [leftover] = remainder {
            weights[n_taps as usize] = *leftover;
            offsets[n_taps as usize] = radius as f32;
            n_taps += 1;
        }

        Self {
            center_weight,
            weights,
            offsets,
            n_taps,
        }
    }
}

// The encoded filters are packed back to back into the filter data texture, each spanning only
// as many texels as its parameters need (like encoded paints).

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
pub(crate) struct GpuOffset {
    pub header: u32,
    pub dx: f32,
    pub dy: f32,
    pub _padding: [u32; 1],
}

impl From<&Offset> for GpuOffset {
    fn from(offset: &Offset) -> Self {
        Self {
            header: pack_header(filter_type::OFFSET),
            dx: offset.dx,
            dy: offset.dy,
            _padding: [0; 1],
        }
    }
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
pub(crate) struct GpuFlood {
    pub header: u32,
    pub color: u32,
    pub _padding: [u32; 2],
}

impl From<&Flood> for GpuFlood {
    fn from(flood: &Flood) -> Self {
        Self {
            header: pack_header(filter_type::FLOOD),
            color: flood.color.premultiply().to_rgba8().to_u32(),
            _padding: [0; 2],
        }
    }
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
pub(crate) struct GpuGaussianBlur {
    pub header: u32,
    pub center_weight: f32,
    pub linear_weights: [f32; MAX_TAPS_PER_SIDE],
    pub linear_offsets: [f32; MAX_TAPS_PER_SIDE],
}

impl From<&GaussianBlur> for GpuGaussianBlur {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "n_decimations fits in 4 bits"
    )]
    fn from(blur: &GaussianBlur) -> Self {
        let lk = LinearKernel::new(&blur.kernel, blur.kernel_size);

        Self {
            header: pack_header_with_gaussian_params(
                filter_type::GAUSSIAN_BLUR,
                edge_mode_to_gpu(blur.edge_mode),
                // Note that this could be exceeded in theory, but it would have to be a huge
                // standard deviation! If it turns out to be a problem we can reserve additional
                // bits for it in the future.
                blur.n_decimations as u32,
                lk.n_taps as u32,
            ),
            center_weight: lk.center_weight,
            linear_weights: lk.weights,
            linear_offsets: lk.offsets,
        }
    }
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
pub(crate) struct GpuDropShadow {
    pub header: u32,
    pub center_weight: f32,
    pub linear_weights: [f32; MAX_TAPS_PER_SIDE],
    pub linear_offsets: [f32; MAX_TAPS_PER_SIDE],
    pub dx: f32,
    pub dy: f32,
    pub color: u32,
    pub _padding: [u32; 1],
}

impl From<&DropShadow> for GpuDropShadow {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "n_decimations fits in 4 bits"
    )]
    fn from(shadow: &DropShadow) -> Self {
        let lk = LinearKernel::new(&shadow.kernel, shadow.kernel_size);
        let composite_original = if shadow.composite_original {
            COMPOSITE_ORIGINAL_MASK
        } else {
            0
        };

        Self {
            header: pack_header_with_gaussian_params(
                filter_type::DROP_SHADOW,
                edge_mode_to_gpu(shadow.edge_mode),
                shadow.n_decimations as u32,
                lk.n_taps as u32,
            ) | composite_original,
            center_weight: lk.center_weight,
            linear_weights: lk.weights,
            linear_offsets: lk.offsets,
            dx: shadow.dx,
            dy: shadow.dy,
            color: shadow.color.premultiply().to_rgba8().to_u32(),
            _padding: [0; 1],
        }
    }
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Zeroable, Pod)]
pub(crate) struct GpuColorMatrix {
    pub header: u32,
    pub _padding: [u32; 3],
    /// The weights of the input channels for each output channel, one row per texel.
    pub weights: [[f32; 4]; 4],
    /// The constant offset of each output channel.
    pub offsets: [f32; 4],
}

impl From<&ColorMatrix> for GpuColorMatrix {
    fn from(color_matrix: &ColorMatrix) -> Self {
        let matrix = &color_matrix.matrix;

        Self {
            header: pack_header(filter_type::COLOR_MATRIX),
            _padding: [0; 3],
            weights: core::array::from_fn(|row| core::array::from_fn(|col| matrix[row * 5 + col])),
            offsets: core::array::from_fn(|row| matrix[row * 5 + 4]),
        }
    }
}

/// The packed header of an encoded filter, see `filter.wesl` for its layout.
#[derive(Debug, Clone, Copy, Zeroable)]
pub(crate) struct GpuFilterHeader(u32);

impl GpuFilterHeader {
    pub(crate) fn filter_type(&self) -> u32 {
        self.0 & 0x1F
    }

    /// Returns the number of decimation levels encoded in the header.
    pub(crate) fn n_decimations(&self) -> usize {
        ((self.0 >> 7) & 0xF) as usize
    }

    pub(crate) fn composite_original(&self) -> bool {
        self.0 & COMPOSITE_ORIGINAL_MASK != 0
    }

    pub(crate) fn needs_copy_pass(&self) -> bool {
        // For drop shadows, we need to retain the input of the filter because in the end
        // we need to composite it _on top_ of the actual shadow.
        self.filter_type() == filter_type::DROP_SHADOW && self.composite_original()
    }
}

/// Per-instance data for one filter pass.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(crate) struct FilterInstanceData {
    /// Origin of the current ping-pong source region, packed as `u16x2`.
    pub source_origin: u32,
    /// Size of the source region, packed as `u16x2`.
    pub source_size: u32,
    /// Origin of the opposite ping-pong destination region, packed as `u16x2`.
    pub dest_origin: u32,
    /// Size of the destination region, packed as `u16x2`.
    pub dest_size: u32,
    /// Dimensions of the destination texture page, packed as `u16x2`.
    pub dest_texture_size: u32,
    /// Texel offset into `filter_data` where this filter's data is stored.
    pub filter_data_offset: u32,
    /// Origin of the original region, packed as `u16x2`.
    pub original_origin: u32,
    /// Size of the original region, packed as `u16x2`.
    pub original_size: u32,
    /// The filter pass that should be executed.
    pub filter_pass_kind: u32,
}

/// Context used for keeping track of state necessary for filter rendering.
#[derive(Debug, Default)]
pub(crate) struct FilterContext {
    /// The encoded filters used in the current scene, in the layout of the filter data texture.
    texels: Vec<[u32; 4]>,
    /// The encoded filters of all filter layers in the scene, in the order in which they are
    /// applied. Each filter layer references a contiguous range of this list.
    prepared: Vec<PreparedGpuFilter>,
}

/// Offset and header of one filter recorded in [`FilterContext`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct PreparedGpuFilter {
    /// Texel offset of the parameter block in the filter data texture.
    pub(crate) data_offset: u32,
    /// The header of the encoded filter.
    pub(crate) header: GpuFilterHeader,
}

/// The chain of prepared filters of one filter layer, referencing a range of the filters
/// recorded in [`FilterContext`]. The filters are applied in order, each operating on the
/// result of the previous one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PreparedGpuFilterChain {
    /// Index of the first filter in [`FilterContext::prepared`].
    start: u32,
    /// Index one past the last filter in [`FilterContext::prepared`].
    end: u32,
    /// Whether any filter in the chain needs to preserve its input in the scratch texture.
    needs_copy_pass: bool,
}

impl PreparedGpuFilterChain {
    #[cfg(test)]
    pub(crate) fn new(range: Range<u32>, needs_copy_pass: bool) -> Self {
        Self {
            start: range.start,
            end: range.end,
            needs_copy_pass,
        }
    }

    /// The range of this chain's filters in [`FilterContext::prepared`].
    pub(crate) fn range(self) -> Range<usize> {
        self.start as usize..self.end as usize
    }

    /// Whether the scratch texture is needed while applying this chain.
    pub(crate) fn needs_copy_pass(self) -> bool {
        self.needs_copy_pass
    }
}

/// The passes to execute at one step of a [`FilterPassPlan`].
///
/// The copies must be executed before the filter passes of the same step.
#[derive(Debug, Default)]
pub(crate) struct FilterStep {
    /// Copies preserving the input of a filter in the shared scratch texture, reading from
    /// the original region.
    copies: Vec<GpuCopyInstance>,
    /// Filter passes reading from the source of this step and writing to its destination.
    filters: Vec<FilterInstanceData>,
}

impl FilterStep {
    pub(crate) fn copy_pass(&self) -> Option<&[GpuCopyInstance]> {
        (!self.copies.is_empty()).then_some(&self.copies)
    }

    pub(crate) fn filters(&self) -> &[FilterInstanceData] {
        &self.filters
    }
}

impl Clear for FilterStep {
    fn clear(&mut self) {
        self.copies.clear();
        self.filters.clear();
    }
}

/// The concrete filter-execution plan for a batch of scheduled filters.
///
/// The passes of all filters are grouped into steps. Within one step, all filter passes read
/// from the same texture and write to the other one, alternating between the two with each
/// step. A step can additionally start with copies that move the current contents of a
/// filter's original region into the scratch texture, for filters that need to preserve
/// their input across their own passes (like drop shadows).
#[derive(Debug, Default)]
pub(crate) struct FilterPassPlan {
    /// The passes grouped by their index in each filter's pass sequence.
    steps: RetainVec<FilterStep>,
}

impl FilterPassPlan {
    /// Build the plan for the given filter operations, whose chains reference `prepared`.
    pub(crate) fn init(
        &mut self,
        filters: impl IntoIterator<Item = FilterOp>,
        prepared: &[PreparedGpuFilter],
        texture_size: SizeU16,
    ) {
        self.clear();

        for op in filters {
            let mut builder = FilterPassBuilder::new(op, texture_size, self);

            for filter in &prepared[op.filters.range()] {
                builder.emit_filter(filter);
            }

            builder.ensure_result_in_original();
        }
    }

    pub(crate) fn steps(&self) -> impl Iterator<Item = &FilterStep> {
        self.steps.as_slice().iter()
    }

    fn clear(&mut self) {
        self.steps.clear();
    }

    fn step_mut(&mut self, step: usize) -> &mut FilterStep {
        if self.steps.len() <= step {
            self.steps.resize_with(step + 1, FilterStep::default);
        }

        &mut self.steps[step]
    }
}

/// Expands one scheduled filter chain into entries in a shared [`FilterPassPlan`].
#[derive(Debug)]
struct FilterPassBuilder<'a> {
    /// Scheduled filter chain and its original/temporary texture regions.
    op: FilterOp,
    /// Full dimensions of the intermediate texture pages.
    texture_size: SizeU16,
    /// The filter pass plan we are writing into.
    passes: &'a mut FilterPassPlan,
    /// Tracks dimensions through blur downscaling and upscaling.
    sizer: DecimationSizer,
    /// Whether the next pass reads from the original region; it writes to the other region.
    current_is_original: bool,
    /// Index of the next pass in this chain's sequence.
    step: usize,
    /// Texel offset of the parameters of the filter whose passes are currently emitted.
    filter_data_offset: u32,
}

impl<'a> FilterPassBuilder<'a> {
    fn new(op: FilterOp, texture_size: SizeU16, passes: &'a mut FilterPassPlan) -> Self {
        let sizer = DecimationSizer::new(
            op.textures.original.rect.width(),
            op.textures.original.rect.height(),
        );

        Self {
            op,
            texture_size,
            passes,
            sizer,
            current_is_original: true,
            step: 0,
            filter_data_offset: 0,
        }
    }

    /// Emit the sequence of passes for one filter of the chain.
    fn emit_filter(&mut self, filter: &PreparedGpuFilter) {
        self.filter_data_offset = filter.data_offset;
        let original = self.op.textures.original.rect;
        self.sizer.reset(original.width(), original.height());

        match filter.header.filter_type() {
            filter_type::OFFSET => {
                self.emit(pass_kind::OFFSET);
            }
            filter_type::FLOOD => {
                self.emit(pass_kind::FLOOD);
            }
            filter_type::COLOR_MATRIX => {
                self.emit(pass_kind::COLOR_MATRIX);
            }
            filter_type::GAUSSIAN_BLUR => {
                self.emit_blur_sequence(filter.header.n_decimations());
            }
            filter_type::DROP_SHADOW => {
                if filter.header.composite_original() {
                    // The input of the drop shadow gets composited on top of the shadow in
                    // the end, but the passes below overwrite both ping-pong regions. So we
                    // preserve the input in the scratch texture, which requires it to be in
                    // the original region first.
                    self.ensure_result_in_original();
                    self.push_copy_to_scratch_pass();
                }
                self.emit(pass_kind::OFFSET);
                self.emit_blur_sequence(filter.header.n_decimations());
                if filter.header.composite_original() {
                    self.emit(pass_kind::COMPOSITE_DROP_SHADOW);
                } else {
                    self.emit(pass_kind::COLORIZE);
                }
            }
            _ => unreachable!("unsupported filter type was encoded"),
        }
    }

    /// Compute and update source and destination sizes based on the pass kind.
    fn apply_pass_dimensions(&mut self, kind: u32) -> (SizeU16, SizeU16) {
        match kind {
            pass_kind::DOWNSCALE => {
                let (sw, sh) = self.sizer.current();
                let (dw, dh) = self.sizer.downscale();
                (SizeU16::from_wh(sw, sh), SizeU16::from_wh(dw, dh))
            }
            pass_kind::UPSCALE => {
                let (sw, sh) = self.sizer.current();
                let (dw, dh) = self.sizer.upscale();
                (SizeU16::from_wh(sw, sh), SizeU16::from_wh(dw, dh))
            }
            _ => {
                let (w, h) = self.sizer.current();
                let size = SizeU16::from_wh(w, h);
                (size, size)
            }
        }
    }

    /// Emit one pass, reading from the current region and writing to the other texture parity.
    fn emit(&mut self, kind: u32) {
        let (source_size, dest_size) = self.apply_pass_dimensions(kind);
        let original = self.op.textures.original;
        let temporary = self.op.textures.temporary;
        let (source_rect, dest_rect) = if self.current_is_original {
            (original.rect, temporary.rect)
        } else {
            (temporary.rect, original.rect)
        };
        let dest_texture_size = self.texture_size;
        let rect_origin = |rect: RectU16| pack_u16_pair(rect.x0, rect.y0);
        let size = |size: SizeU16| pack_u16_pair(size.width(), size.height());

        self.passes
            .step_mut(self.step)
            .filters
            .push(FilterInstanceData {
                source_origin: rect_origin(source_rect),
                source_size: size(source_size),
                dest_origin: rect_origin(dest_rect),
                dest_size: size(dest_size),
                dest_texture_size: size(dest_texture_size),
                filter_data_offset: self.filter_data_offset,
                original_origin: rect_origin(original.rect),
                original_size: pack_u16_pair(original.rect.width(), original.rect.height()),
                filter_pass_kind: kind,
            });

        self.step += 1;
        self.current_is_original = !self.current_is_original;
    }

    /// Apply the sequences of passes that is needed to create a full Gaussian blur with
    /// the given number of decimations.
    fn emit_blur_sequence(&mut self, n_decimations: usize) {
        // TODO: From my experiments, it would very much be worth it to add a
        // UPSCALE_4x and DOWNSCALE_4x pass, since unlike the CPU we can use bilinear
        // filtering for sampling and therefore don't need as many samples, and can reduce
        // the number of render passes for large standard deviations. However, this unfortunately
        // causes higher pixel differences for some tests compared to vello_cpu, since edge
        // pixels will inevitably exhibit different behavior. Therefore, for now we stick to
        // this more straight-forward approach.

        for _ in 0..n_decimations {
            self.emit(pass_kind::DOWNSCALE);
        }
        self.emit(pass_kind::BLUR_H);

        let mut final_pass = pass_kind::BLUR_V;

        if n_decimations > 0 {
            self.emit(pass_kind::BLUR_V);
            for _ in 0..n_decimations - 1 {
                self.emit(pass_kind::UPSCALE);
            }

            final_pass = pass_kind::UPSCALE;
        }

        self.emit(final_pass);
    }

    fn ensure_result_in_original(&mut self) {
        if !self.current_is_original {
            self.emit(pass_kind::COPY);
        }
    }

    /// Copy the current contents of the original region into the scratch texture, before the
    /// passes of the current step run.
    fn push_copy_to_scratch_pass(&mut self) {
        debug_assert!(
            self.current_is_original,
            "the input to preserve must be in the original region"
        );

        let original = self.op.textures.original;
        let dest_texture_size = self.texture_size;
        let copy_instance = GpuCopyInstance {
            dest_texture_origin: pack_u16_pair(original.rect.x0, original.rect.y0),
            source_texture_origin: pack_u16_pair(original.rect.x0, original.rect.y0),
            copy_rect_size: pack_u16_pair(original.rect.width(), original.rect.height()),
            dest_texture_size: pack_u16_pair(dest_texture_size.width(), dest_texture_size.height()),
        };

        self.passes.step_mut(self.step).copies.push(copy_instance);
    }
}

impl FilterContext {
    pub(crate) fn clear(&mut self) {
        self.texels.clear();
        self.prepared.clear();
    }

    /// Encode all filters of a filter layer and return the chain referencing them.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the number of filters is bounded by the size of the filter data texture"
    )]
    pub(crate) fn push(&mut self, filter_data: &FilterData) -> PreparedGpuFilterChain {
        let start = self.prepared.len();

        for filter in PreparedFilter::chain(&filter_data.filter, &filter_data.transform) {
            let prepared = match &filter {
                PreparedFilter::Offset(f) => self.encode(&GpuOffset::from(f)),
                PreparedFilter::Flood(f) => self.encode(&GpuFlood::from(f)),
                PreparedFilter::GaussianBlur(f) => self.encode(&GpuGaussianBlur::from(f)),
                PreparedFilter::DropShadow(f) => self.encode(&GpuDropShadow::from(f)),
                PreparedFilter::ColorMatrix(f) => self.encode(&GpuColorMatrix::from(f)),
            };
            self.prepared.push(prepared);
        }

        let chain = &self.prepared[start..];

        PreparedGpuFilterChain {
            start: start as u32,
            end: self.prepared.len() as u32,
            needs_copy_pass: chain.iter().any(|f| f.header.needs_copy_pass()),
        }
    }

    /// All filters encoded so far, referenced by the chains returned from [`Self::push`].
    pub(crate) fn prepared(&self) -> &[PreparedGpuFilter] {
        &self.prepared
    }

    /// Append an encoded filter, which must start with its packed header.
    fn encode<T: Pod>(&mut self, filter: &T) -> PreparedGpuFilter {
        let data_offset = self.total_texels();
        self.texels
            .extend_from_slice(bytemuck::cast_slice(core::slice::from_ref(filter)));
        let header = GpuFilterHeader(self.texels[data_offset as usize][0]);

        PreparedGpuFilter {
            data_offset,
            header,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.texels.is_empty()
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the texel count is bounded by the size of the filter data texture"
    )]
    pub(crate) fn total_texels(&self) -> u32 {
        self.texels.len() as u32
    }

    pub(crate) fn serialize_to_buffer(&self, buffer: &mut [u8]) {
        let src = bytemuck::cast_slice::<[u32; 4], u8>(&self.texels);
        debug_assert!(
            buffer.len() >= src.len(),
            "filter data buffer too small: {} < {}",
            buffer.len(),
            src.len()
        );
        buffer[..src.len()].copy_from_slice(src);
    }

    /// Calculate the required height for the filter data texture.
    /// Returns `None` if no filters are present.
    pub(crate) fn required_filter_data_height(
        &self,
        resource_texture_dimension_2d: u32,
    ) -> Option<u32> {
        let required_texels = self.total_texels();

        if required_texels == 0 {
            return None;
        }
        let height = required_texels.div_ceil(resource_texture_dimension_2d);
        // TODO: Turn into error.
        assert!(
            height <= resource_texture_dimension_2d,
            "Filter texture height exceeds resource texture dimensions"
        );

        Some(height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::round::FilterTextureRegions;
    use crate::target::{LayerTextureId, TextureParity, TextureRegion};
    use vello_common::color::AlphaColor;
    use vello_common::filter::gaussian_blur::{compute_gaussian_kernel, plan_decimated_blur};
    use vello_common::filter_effects::matrices;

    fn region(parity: TextureParity) -> TextureRegion {
        TextureRegion {
            target: LayerTextureId::new(parity, 0),
            rect: RectU16::new(0, 0, 32, 24),
        }
    }

    fn filter_op(filters: Range<u32>, needs_copy_pass: bool) -> FilterOp {
        FilterOp {
            textures: FilterTextureRegions::new(
                region(TextureParity::Odd),
                region(TextureParity::Even),
            ),
            filters: PreparedGpuFilterChain::new(filters, needs_copy_pass),
        }
    }

    fn prepared(header: GpuFilterHeader, data_offset: u32) -> PreparedGpuFilter {
        PreparedGpuFilter {
            data_offset,
            header,
        }
    }

    fn gpu_offset() -> GpuFilterHeader {
        GpuFilterHeader(pack_header(filter_type::OFFSET))
    }

    fn gpu_flood() -> GpuFilterHeader {
        GpuFilterHeader(pack_header(filter_type::FLOOD))
    }

    fn gpu_blur(std_deviation: f32) -> GpuFilterHeader {
        GpuFilterHeader(
            GpuGaussianBlur::from(&GaussianBlur::new(std_deviation, EdgeMode::None)).header,
        )
    }

    fn gpu_shadow() -> GpuFilterHeader {
        GpuFilterHeader(
            GpuDropShadow::from(&DropShadow::new(
                3.0,
                -4.0,
                8.0,
                EdgeMode::None,
                AlphaColor::new([0.0, 0.0, 0.0, 1.0]),
            ))
            .header,
        )
    }

    fn gpu_color_matrix() -> GpuFilterHeader {
        GpuFilterHeader(pack_header(filter_type::COLOR_MATRIX))
    }

    #[test]
    #[should_panic(expected = "Filter texture height exceeds resource texture dimensions")]
    fn filter_data_height_must_fit_resource_texture_limit() {
        let mut context = FilterContext::default();
        context.encode(&GpuOffset::from(&Offset::new(1.0, 2.0)));
        context.encode(&GpuOffset::from(&Offset::new(1.0, 2.0)));

        let _ = context.required_filter_data_height(1);
    }

    #[test]
    fn filters_are_packed_with_variable_stride() {
        let mut context = FilterContext::default();
        let offsets = [
            context.encode(&GpuOffset::from(&Offset::new(1.0, 2.0))),
            context.encode(&GpuGaussianBlur::from(&GaussianBlur::new(
                2.0,
                EdgeMode::None,
            ))),
            context.encode(&GpuColorMatrix::from(&ColorMatrix::new(matrices::SEPIA))),
            context.encode(&GpuFlood::from(&Flood::new(AlphaColor::new([
                0.2, 0.4, 0.6, 0.8,
            ])))),
        ]
        .map(|filter| filter.data_offset);

        assert_eq!(offsets, [0, 1, 3, 9]);
        assert_eq!(context.total_texels(), 10);
        assert_eq!(context.required_filter_data_height(4), Some(3));
    }

    fn step_layout(plan: &FilterPassPlan) -> Vec<Vec<(u32, u32)>> {
        plan.steps()
            .map(|step| {
                step.filters()
                    .iter()
                    .map(|instance| (instance.filter_data_offset, instance.filter_pass_kind))
                    .collect()
            })
            .collect()
    }

    fn copy_layout(plan: &FilterPassPlan) -> Vec<usize> {
        plan.steps()
            .map(|step| step.copy_pass().map_or(0, <[GpuCopyInstance]>::len))
            .collect()
    }

    #[test]
    fn pass_batching() {
        let mut plan = FilterPassPlan::default();
        plan.init(
            [
                filter_op(0..1, false),
                filter_op(1..2, false),
                filter_op(2..3, true),
            ],
            &[
                prepared(gpu_offset(), 0),
                prepared(gpu_blur(8.0), 1),
                prepared(gpu_shadow(), 2),
            ],
            SizeU16::new(64),
        );

        assert_eq!(
            step_layout(&plan),
            alloc::vec![
                alloc::vec![
                    (0, pass_kind::OFFSET),
                    (1, pass_kind::DOWNSCALE),
                    (2, pass_kind::OFFSET),
                ],
                alloc::vec![
                    (0, pass_kind::COPY),
                    (1, pass_kind::DOWNSCALE),
                    (2, pass_kind::DOWNSCALE),
                ],
                alloc::vec![(1, pass_kind::BLUR_H), (2, pass_kind::DOWNSCALE)],
                alloc::vec![(1, pass_kind::BLUR_V), (2, pass_kind::BLUR_H)],
                alloc::vec![(1, pass_kind::UPSCALE), (2, pass_kind::BLUR_V)],
                alloc::vec![(1, pass_kind::UPSCALE), (2, pass_kind::UPSCALE)],
                alloc::vec![(2, pass_kind::UPSCALE)],
                alloc::vec![(2, pass_kind::COMPOSITE_DROP_SHADOW)],
            ]
        );
        // The drop shadow preserves its input before its first pass.
        assert_eq!(copy_layout(&plan), [1, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn plan_reinit() {
        let mut plan = FilterPassPlan::default();
        plan.init(
            [filter_op(0..1, true), filter_op(1..2, false)],
            &[prepared(gpu_shadow(), 0), prepared(gpu_blur(8.0), 1)],
            SizeU16::new(64),
        );
        assert!(plan.steps().next().unwrap().copy_pass().is_some());
        assert!(plan.steps().count() > 2);

        plan.init(
            [filter_op(0..1, false)],
            &[prepared(gpu_offset(), 0)],
            SizeU16::new(64),
        );

        assert_eq!(copy_layout(&plan), [0, 0]);
        assert_eq!(
            step_layout(&plan),
            [
                alloc::vec![(0, pass_kind::OFFSET)],
                alloc::vec![(0, pass_kind::COPY)],
            ]
        );
    }

    #[test]
    fn chain_passes_run_back_to_back() {
        let mut plan = FilterPassPlan::default();
        plan.init(
            [filter_op(0..3, false)],
            &[
                prepared(gpu_color_matrix(), 0),
                prepared(gpu_offset(), 6),
                prepared(gpu_blur(8.0), 7),
            ],
            SizeU16::new(64),
        );

        assert_eq!(copy_layout(&plan), [0; 8]);
        assert_eq!(
            step_layout(&plan),
            [
                alloc::vec![(0, pass_kind::COLOR_MATRIX)],
                alloc::vec![(6, pass_kind::OFFSET)],
                alloc::vec![(7, pass_kind::DOWNSCALE)],
                alloc::vec![(7, pass_kind::DOWNSCALE)],
                alloc::vec![(7, pass_kind::BLUR_H)],
                alloc::vec![(7, pass_kind::BLUR_V)],
                alloc::vec![(7, pass_kind::UPSCALE)],
                // The chain has an even number of passes, so the result is already in
                // the original region and no copy pass is needed.
                alloc::vec![(7, pass_kind::UPSCALE)],
            ]
        );
    }

    #[test]
    fn drop_shadow_in_chain_preserves_its_input() {
        let mut plan = FilterPassPlan::default();
        plan.init(
            [filter_op(0..2, true)],
            &[prepared(gpu_offset(), 0), prepared(gpu_shadow(), 1)],
            SizeU16::new(64),
        );

        // After the offset pass, the result lives in the temporary region. It first has to be
        // moved back into the original region, from where it is copied into the scratch
        // texture right before the drop shadow's passes overwrite both regions.
        assert_eq!(
            step_layout(&plan)[..3],
            [
                alloc::vec![(0, pass_kind::OFFSET)],
                alloc::vec![(1, pass_kind::COPY)],
                alloc::vec![(1, pass_kind::OFFSET)],
            ]
        );
        assert_eq!(copy_layout(&plan)[..3], [0, 0, 1]);
        assert_eq!(
            step_layout(&plan).last().unwrap(),
            &alloc::vec![(1, pass_kind::COMPOSITE_DROP_SHADOW)]
        );
        // The chain has an even number of passes, so the composite result lands in the
        // original region without a trailing copy.
        assert_eq!(plan.steps().count() % 2, 0);
    }

    #[test]
    fn empty_chain_emits_no_passes() {
        let mut plan = FilterPassPlan::default();
        plan.init([filter_op(0..0, false)], &[], SizeU16::new(64));

        assert_eq!(plan.steps().count(), 0);
    }

    #[test]
    fn single_pass_filters_finish_in_original() {
        let mut plan = FilterPassPlan::default();
        plan.init(
            [
                filter_op(0..1, false),
                filter_op(1..2, false),
                filter_op(2..3, false),
            ],
            &[
                prepared(gpu_offset(), 0),
                prepared(gpu_flood(), 1),
                prepared(gpu_color_matrix(), 2),
            ],
            SizeU16::new(64),
        );

        assert_eq!(copy_layout(&plan), [0, 0]);
        assert_eq!(
            step_layout(&plan),
            [
                alloc::vec![
                    (0, pass_kind::OFFSET),
                    (1, pass_kind::FLOOD),
                    (2, pass_kind::COLOR_MATRIX),
                ],
                alloc::vec![
                    (0, pass_kind::COPY),
                    (1, pass_kind::COPY),
                    (2, pass_kind::COPY),
                ],
            ]
        );
    }

    #[test]
    fn test_offset_conversion() {
        let offset = Offset::new(10.5, -20.3);
        let gpu_offset = GpuOffset::from(&offset);

        assert_eq!(gpu_offset.header & 0x1F, filter_type::OFFSET);
        assert_eq!(gpu_offset.dx, 10.5);
        assert_eq!(gpu_offset.dy, -20.3);
    }

    #[test]
    fn test_color_matrix_conversion() {
        let matrix = core::array::from_fn(|i| i as f32);
        let gpu_color_matrix = GpuColorMatrix::from(&ColorMatrix::new(matrix));

        assert_eq!(
            gpu_color_matrix.weights,
            [
                [0.0, 1.0, 2.0, 3.0],
                [5.0, 6.0, 7.0, 8.0],
                [10.0, 11.0, 12.0, 13.0],
                [15.0, 16.0, 17.0, 18.0],
            ]
        );
        assert_eq!(gpu_color_matrix.offsets, [4.0, 9.0, 14.0, 19.0]);
    }

    fn check_round_trip<T>(gpu: T, expected_type: u32)
    where
        T: PartialEq + core::fmt::Debug + Pod,
    {
        let mut context = FilterContext::default();
        let prepared = context.encode(&gpu);

        assert_eq!(prepared.header.filter_type(), expected_type);
        assert_eq!(
            bytemuck::pod_read_unaligned::<T>(bytemuck::cast_slice(&context.texels)),
            gpu
        );
    }

    #[test]
    fn test_offset_round_trip() {
        check_round_trip(GpuOffset::from(&Offset::new(1.0, 2.0)), filter_type::OFFSET);
    }

    #[test]
    fn test_flood_round_trip() {
        check_round_trip(
            GpuFlood::from(&Flood::new(AlphaColor::new([0.2, 0.4, 0.6, 0.8]))),
            filter_type::FLOOD,
        );
    }

    #[test]
    fn test_gaussian_blur_round_trip() {
        check_round_trip(
            GpuGaussianBlur::from(&GaussianBlur::new(2.0, EdgeMode::None)),
            filter_type::GAUSSIAN_BLUR,
        );
    }

    #[test]
    fn test_color_matrix_round_trip() {
        check_round_trip(
            GpuColorMatrix::from(&ColorMatrix::new(matrices::SEPIA)),
            filter_type::COLOR_MATRIX,
        );
    }

    #[test]
    fn test_drop_shadow_round_trip() {
        check_round_trip(
            GpuDropShadow::from(&DropShadow::new(
                3.0,
                -4.0,
                1.5,
                EdgeMode::Duplicate,
                AlphaColor::new([0.0, 0.0, 0.0, 1.0]),
            )),
            filter_type::DROP_SHADOW,
        );
    }

    fn check_linear_kernel(kernel: &[f32; MAX_KERNEL_SIZE], size: u8, expected_taps: u8) {
        let lk = LinearKernel::new(kernel, size);
        assert_eq!(lk.n_taps, expected_taps);

        let sum = lk.center_weight + 2.0 * lk.weights.iter().take(lk.n_taps as usize).sum::<f32>();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "weights must sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn linear_kernel_size_1() {
        let (_n_dec, kernel, size) = plan_decimated_blur(0.0);
        assert_eq!(size, 1);
        check_linear_kernel(&kernel, size, 0);
    }

    #[test]
    fn linear_kernel_size_3() {
        let (kernel, size) = compute_gaussian_kernel(0.1);
        assert_eq!(size, 3);
        check_linear_kernel(&kernel, size, 1);
    }

    #[test]
    fn linear_kernel_size_7() {
        let (kernel, size) = compute_gaussian_kernel(1.0);
        assert_eq!(size, 7);
        check_linear_kernel(&kernel, size, 2);
    }

    #[test]
    fn linear_kernel_size_13() {
        let (kernel, size) = compute_gaussian_kernel(2.0);
        assert_eq!(size, 13);
        check_linear_kernel(&kernel, size, 3);
    }
}
