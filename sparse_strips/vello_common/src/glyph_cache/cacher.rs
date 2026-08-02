// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Atlas-backed implementation of glifo's [`GlyphCacher`] hook.
//!
//! [`AtlasGlyphCacher`] wraps a [`GlyphAtlas`] + [`ImageCache`] pair and
//! implements [`glifo::GlyphCacher`] for any renderer that additionally
//! implements [`AtlasGlyphRenderer`] (i.e. can resolve an [`AtlasSlot`] to an
//! image source and supports tinted image painting).

use super::cache::GlyphAtlas;
use super::commands::AtlasCommandRecorder;
use super::key::{GlyphCacheKey, SUBPIXEL_BITMAP, SUBPIXEL_COLR, join_bits, subpixel_offset};
use super::region::{AtlasSlot, RasterMetrics};
use crate::image_cache::ImageCache;
use crate::paint::{Image, ImageSource, PaintType, Tint, TintMode};
use alloc::sync::Arc;
use glifo::{
    BitmapGlyphData, CacheRunConfig, CacheableGlyph, CacheableGlyphKind, ColrGlyphData, DrawSink,
    GlyphCacher, GlyphRenderer, OutlineGlyphData,
};
use peniko::color::palette::css::BLACK;
use peniko::kurbo::{Affine, Join, Rect, Vec2};
use peniko::{Extend, ImageQuality, ImageSampler};
use smallvec::SmallVec;

/// Renderer capabilities required for atlas-cached glyph rendering, on top of
/// glifo's renderer-neutral [`GlyphRenderer`] trait.
pub trait AtlasGlyphRenderer: GlyphRenderer {
    /// Set (or clear) the tint applied to subsequent image paints.
    fn set_tint(&mut self, tint: Option<Tint>);

    /// Set the current paint to a (typically atlas-page) image.
    fn set_paint_atlas_image(&mut self, image: Image);

    /// The current paint. Used to decide whether outline glyphs can be
    /// tint-rendered from the atlas (solid paints only).
    fn current_paint(&self) -> &PaintType;

    /// Resolve an atlas slot to the image source of its atlas page.
    fn atlas_image_source(&self, atlas_slot: &AtlasSlot) -> ImageSource;

    /// The paint transform mapping the fill rect onto the slot's atlas region.
    fn atlas_paint_transform(&self, atlas_slot: &AtlasSlot) -> Affine {
        Affine::translate((-(atlas_slot.x as f64), -(atlas_slot.y as f64)))
    }
}

/// A [`GlyphCacher`] backed by a shared [`GlyphAtlas`] and [`ImageCache`].
#[derive(Debug)]
pub struct AtlasGlyphCacher<'a> {
    /// The glyph atlas cache (entries, LRU, pending uploads/commands).
    pub glyph_atlas: &'a mut GlyphAtlas,
    /// The image cache providing atlas page allocation.
    pub image_cache: &'a mut ImageCache,
}

impl<'a> AtlasGlyphCacher<'a> {
    /// Create a cacher for the given atlas and image cache.
    pub fn new(glyph_atlas: &'a mut GlyphAtlas, image_cache: &'a mut ImageCache) -> Self {
        Self {
            glyph_atlas,
            image_cache,
        }
    }
}

/// Build a [`GlyphCacheKey`] from glifo's cache-key ingredients.
fn cache_key(glyph: &CacheableGlyph<'_>) -> GlyphCacheKey {
    match glyph.kind {
        CacheableGlyphKind::Outline => GlyphCacheKey::new(
            glyph.font_id,
            glyph.font_index,
            glyph.glyph_id,
            glyph.font_size,
            glyph.hinted,
            glyph.fractional_x,
            glyph.context_color,
            glyph.context_color_packed,
            glyph.embolden,
            glyph.var_coords,
        ),
        CacheableGlyphKind::Colr(_) | CacheableGlyphKind::Bitmap => GlyphCacheKey {
            font_id: glyph.font_id,
            font_index: glyph.font_index,
            glyph_id: glyph.glyph_id,
            size_bits: glyph.font_size.to_bits(),
            hinted: false,
            subpixel_x: match glyph.kind {
                CacheableGlyphKind::Colr(_) => SUBPIXEL_COLR,
                _ => SUBPIXEL_BITMAP,
            },
            context_color: glyph.context_color,
            context_color_packed: glyph.context_color_packed,
            embolden_x_bits: 0,
            embolden_y_bits: 0,
            embolden_join_bits: join_bits(Join::Miter),
            embolden_miter_limit_bits: 4.0_f32.to_bits(),
            embolden_tolerance_bits: 0.1_f32.to_bits(),
            var_coords: SmallVec::from_slice(glyph.var_coords),
        },
    }
}

/// Returns `true` if the transform is safe for atlas-cached glyph rendering.
#[inline]
fn supports_atlas_caching(transform: &Affine, kind: &CacheableGlyphKind) -> bool {
    // TODO: Investigate whether we can support arbitrary mirroring. From some
    // initial experiments, allowing x-mirroring leads to slightly shifted glyphs, so
    // we don't support this now. Y-mirroring also needs more consideration.
    let [a, b, c, d, _, _] = transform.as_coeffs();
    let has_skew = b.abs() > SCALAR_NEARLY_ZERO || c.abs() > SCALAR_NEARLY_ZERO;

    match kind {
        // For those glyphs, we expect any scaling factor to have been completely absorbed. Due to the fact
        // that we had to apply a flip transform for outlines, the y-scaling factor is expected to be negative.
        CacheableGlyphKind::Outline | CacheableGlyphKind::Colr(_) => {
            let unit_scale = (1.0 - a.abs()).abs() <= SCALAR_NEARLY_ZERO
                && (1.0 - d.abs()).abs() <= SCALAR_NEARLY_ZERO;
            !has_skew && unit_scale && a.is_sign_positive() && d.is_sign_negative()
        }
        // For bitmap glyphs, we need to relax the condition a bit, since bitmap glyphs already have a fixed
        // size and thus might not correspond 100% to the font size. Therefore, they likely don't have a unit
        // transform.
        CacheableGlyphKind::Bitmap => !has_skew && a.is_sign_positive() && d.is_sign_positive(),
    }
}

// From <https://github.com/linebender/tiny-skia/blob/68b198a7210a6bbf752b43d6bc4db62445730313/path/src/scalar.rs#L12>
const SCALAR_NEARLY_ZERO: f64 = 1.0 / (1 << 12) as f64;

/// Choose image sampling quality based on downscale factor.
#[inline]
fn quality_for_scale(transform: &Affine) -> ImageQuality {
    let [a, _, _, d, _, _] = transform.as_coeffs();
    if a < 0.5 || d < 0.5 {
        ImageQuality::High
    } else {
        ImageQuality::Medium
    }
}

/// Calculate raster metrics (pixel bounds, bearings) from a glyph's bounding box.
#[expect(
    clippy::cast_possible_truncation,
    reason = "glyph bounds fit in i32/u16/i16 at reasonable ppem values"
)]
#[inline]
fn calculate_raster_metrics(bounds: &Rect) -> RasterMetrics {
    // Floor/ceil round outward from the fractional bounding box. Width gets an
    // extra pixel to accommodate the horizontal subpixel offset (up to 0.75 px)
    // applied when rasterising into the atlas; the Y axis has no subpixel shift
    // so floor/ceil alone is sufficient. GLYPH_PADDING in the atlas allocator
    // provides the guard band needed by the hybrid renderer's Extend::Pad sampling.
    let min_x = bounds.x0.floor() as i32;
    let max_x = bounds.x1.ceil() as i32 + 1;

    // For Y, we flip the coordinate system: font Y up -> screen Y down
    // After flipping Y, min_y becomes -max_y and max_y becomes -min_y
    let flipped_min_y = (-bounds.y1).floor() as i32;
    let flipped_max_y = (-bounds.y0).ceil() as i32;

    let width = (max_x - min_x) as u16;
    let height = (flipped_max_y - flipped_min_y) as u16;

    RasterMetrics {
        width,
        height,
        bearing_x: min_x as i16,
        bearing_y: flipped_min_y as i16,
    }
}

/// Render from the atlas, constructing the appropriate image from the slot.
fn render_from_atlas(
    renderer: &mut impl AtlasGlyphRenderer,
    atlas_slot: AtlasSlot,
    rect_transform: Affine,
    area: Rect,
    quality: ImageQuality,
    tint: Option<Tint>,
) {
    let paint_transform = renderer.atlas_paint_transform(&atlas_slot);
    let image_source = renderer.atlas_image_source(&atlas_slot);
    let image = Image {
        image: image_source,
        sampler: ImageSampler {
            x_extend: Extend::Pad,
            y_extend: Extend::Pad,
            quality,
            alpha: 1.0,
        },
    };

    let state = renderer.save_state();
    renderer.set_tint(tint);
    renderer.set_transform(rect_transform);
    renderer.set_paint_atlas_image(image);
    renderer.set_paint_transform(paint_transform);
    renderer.fill_rect(&area);
    renderer.set_tint(None);
    renderer.restore_state(state);
}

/// Render an outline glyph from the atlas using bearing-based positioning.
#[inline]
fn render_outline_glyph_from_atlas(
    renderer: &mut impl AtlasGlyphRenderer,
    atlas_slot: AtlasSlot,
    outline_transform: Affine,
    tint_color: peniko::color::AlphaColor<peniko::color::Srgb>,
) {
    let [_, _, _, _, tx, ty] = outline_transform.as_coeffs();
    let rect_transform = Affine::translate((
        tx.floor() + atlas_slot.bearing_x as f64,
        ty.floor() + atlas_slot.bearing_y as f64,
    ));
    let area = Rect::new(0.0, 0.0, atlas_slot.width as f64, atlas_slot.height as f64);
    render_from_atlas(
        renderer,
        atlas_slot,
        rect_transform,
        area,
        ImageQuality::Low,
        Some(Tint {
            color: tint_color,
            mode: TintMode::AlphaMask,
        }),
    );
}

/// Choose image sampling quality based on skew presence.
#[inline]
fn quality_for_skew(transform: &Affine) -> ImageQuality {
    let [_, b, c, _, _, _] = transform.as_coeffs();
    if b.abs() > SCALAR_NEARLY_ZERO || c.abs() > SCALAR_NEARLY_ZERO {
        ImageQuality::Medium
    } else {
        ImageQuality::Low
    }
}

impl<R: AtlasGlyphRenderer> GlyphCacher<R> for AtlasGlyphCacher<'_> {
    fn run_config(&self, renderer: &R, font_size: f32) -> CacheRunConfig {
        let size_ok = font_size <= self.glyph_atlas.config().max_cached_font_size;
        CacheRunConfig {
            // We use image tinting to color cached outline glyphs, which is
            // not supported for complex paints.
            cache_outlines: size_ok && matches!(renderer.current_paint(), PaintType::Solid(_)),
            cache_colr_bitmap: size_ok,
        }
    }

    fn draw_cached_glyph(&mut self, renderer: &mut R, glyph: &CacheableGlyph<'_>) -> bool {
        let key = cache_key(glyph);
        let Some(slot) = self.glyph_atlas.get(&key) else {
            return false;
        };
        match glyph.kind {
            CacheableGlyphKind::Outline => {
                let tint = renderer.get_context_color();
                render_outline_glyph_from_atlas(renderer, slot, glyph.transform, tint);
            }
            CacheableGlyphKind::Bitmap => {
                let area = Rect::new(0.0, 0.0, slot.width as f64, slot.height as f64);
                render_from_atlas(
                    renderer,
                    slot,
                    glyph.transform,
                    area,
                    quality_for_scale(&glyph.transform),
                    None,
                );
            }
            CacheableGlyphKind::Colr(area) => {
                render_from_atlas(
                    renderer,
                    slot,
                    glyph.transform,
                    area,
                    quality_for_skew(&glyph.transform),
                    None,
                );
            }
        }
        true
    }

    fn draw_and_cache_outline(
        &mut self,
        renderer: &mut R,
        data: &OutlineGlyphData<'_>,
        glyph: &CacheableGlyph<'_>,
    ) -> bool {
        if !supports_atlas_caching(&glyph.transform, &glyph.kind) {
            return false;
        }

        let key = cache_key(glyph);
        let bounds = data.bbox.scale_from_origin(data.scale);
        let raster_metrics = calculate_raster_metrics(&bounds);
        let subpixel = subpixel_offset(key.subpixel_x);

        let Some((atlas_slot, recorder)) =
            self.glyph_atlas
                .insert(self.image_cache, key, raster_metrics)
        else {
            return false;
        };

        record_outline_to_atlas(data, subpixel, recorder, atlas_slot, raster_metrics);

        let tint = renderer.get_context_color();
        render_outline_glyph_from_atlas(renderer, atlas_slot, glyph.transform, tint);
        true
    }

    fn draw_and_cache_bitmap(
        &mut self,
        renderer: &mut R,
        data: &BitmapGlyphData<'_>,
        glyph: &CacheableGlyph<'_>,
    ) -> bool {
        if !supports_atlas_caching(&glyph.transform, &glyph.kind) {
            return false;
        }

        let key = cache_key(glyph);
        let raster_metrics = RasterMetrics {
            width: data.pixmap.width(),
            height: data.pixmap.height(),
            bearing_x: 0,
            bearing_y: 0,
        };

        // Bitmap glyphs already have pixel data — no draw commands to record,
        // so we discard the returned recorder.
        let Some((atlas_slot, _)) = self
            .glyph_atlas
            .insert(self.image_cache, key, raster_metrics)
        else {
            return false;
        };

        // Both backends defer the actual pixel copy/upload; it completes before
        // the render pass that resolves image references.
        self.glyph_atlas.push_pending_upload(
            atlas_slot.image_id,
            Arc::clone(data.pixmap),
            atlas_slot,
        );

        render_from_atlas(
            renderer,
            atlas_slot,
            glyph.transform,
            data.area,
            quality_for_scale(&glyph.transform),
            None,
        );
        true
    }

    fn draw_and_cache_colr(
        &mut self,
        renderer: &mut R,
        data: &ColrGlyphData,
        glyph: &CacheableGlyph<'_>,
        paint: &mut dyn FnMut(&mut dyn DrawSink),
    ) -> bool {
        if !supports_atlas_caching(&glyph.transform, &glyph.kind) {
            return false;
        }

        let key = cache_key(glyph);
        let raster_metrics = RasterMetrics {
            width: data.pix_width,
            height: data.pix_height,
            bearing_x: 0,
            bearing_y: 0,
        };

        let Some((atlas_slot, recorder)) =
            self.glyph_atlas
                .insert(self.image_cache, key, raster_metrics)
        else {
            return false;
        };

        recorder.set_transform(Affine::translate((
            atlas_slot.x as f64,
            atlas_slot.y as f64,
        )));
        paint(recorder);

        let CacheableGlyphKind::Colr(area) = glyph.kind else {
            unreachable!("COLR glyph offered with non-COLR kind");
        };
        render_from_atlas(
            renderer,
            atlas_slot,
            glyph.transform,
            area,
            quality_for_skew(&glyph.transform),
            None,
        );
        true
    }
}

/// Record outline glyph draw commands into the atlas command recorder.
fn record_outline_to_atlas(
    data: &OutlineGlyphData<'_>,
    subpixel_offset: f32,
    recorder: &mut AtlasCommandRecorder,
    atlas_slot: AtlasSlot,
    raster_metrics: RasterMetrics,
) {
    let outline_transform =
        Affine::scale_non_uniform(data.scale, -data.scale).then_translate(Vec2::new(
            atlas_slot.x as f64 - raster_metrics.bearing_x as f64 + subpixel_offset as f64,
            atlas_slot.y as f64 - raster_metrics.bearing_y as f64,
        ));
    recorder.set_transform(outline_transform);
    recorder.set_paint(BLACK.into());
    recorder.fill_path(data.path);
}

/// Replay recorded atlas commands into a [`DrawSink`].
///
/// The commands `Vec` is drained, freeing memory as each command is consumed.
pub fn replay_atlas_commands(
    commands: &mut alloc::vec::Vec<super::commands::AtlasCommand>,
    target: &mut impl DrawSink,
) {
    use super::commands::AtlasCommand;
    for cmd in commands.drain(..) {
        match cmd {
            AtlasCommand::SetTransform(t) => target.set_transform(t),
            AtlasCommand::SetPaint(p) => target.set_paint(p),
            AtlasCommand::SetPaintTransform(t) => target.set_paint_transform(t),
            AtlasCommand::FillPath(p) => target.fill_path(&p),
            AtlasCommand::FillRect(r) => target.fill_rect(&r),
            AtlasCommand::PushClipLayer(c) => target.push_clip_layer(&c),
            AtlasCommand::PushClipPath(c) => target.push_clip_path(&c),
            AtlasCommand::PushBlendLayer(m) => target.push_blend_layer(m),
            AtlasCommand::PopLayer => target.pop_layer(),
            AtlasCommand::PopClipPath => target.pop_clip_path(),
        }
    }
}
