// Copyright 2026 the Vello Authors and the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared glyph rendering logic for rendering backends.

use crate::bitmap::GlyphImage;
use crate::cache::{BitmapGlyphData, ColrGlyphData, GlyphCacher, OutlineGlyphData};
use crate::colr::ColrPainter;
use crate::glyph::{GlyphBitmap, GlyphColr, GlyphType, OutlineCacheSession, PreparedGlyph};
use crate::interface::GlyphRenderer;
use crate::kurbo;
use crate::peniko;
use kurbo::{Affine, BezPath, Shape};
use peniko::ImageQuality;
use peniko::color::{AlphaColor, Srgb};

/// Fill a prepared glyph, offering it to the backend glyph cacher first and
/// falling back to direct rendering otherwise.
pub(crate) fn fill_glyph<R: GlyphRenderer>(
    renderer: &mut R,
    cacher: &mut impl GlyphCacher<R>,
    prepared_glyph: PreparedGlyph<'_>,
    outline_cache: &mut OutlineCacheSession<'_>,
) {
    let PreparedGlyph {
        glyph_type,
        outline_transform: transform,
        relative_paint_transform: paint_transform,
        cacheable,
    } = prepared_glyph;

    match glyph_type {
        GlyphType::Outline(glyph) => {
            if let Some(info) = &cacheable {
                let data = OutlineGlyphData {
                    path: &glyph.path,
                    bbox: glyph.bbox,
                    scale: glyph.scale,
                };
                if cacher.draw_and_cache_outline(renderer, &data, info) {
                    return;
                }
            }

            fill_uncached_outline_glyph(
                renderer,
                &glyph.path,
                glyph.scale,
                transform,
                paint_transform,
            );
        }
        GlyphType::Bitmap(glyph) => {
            if let Some(info) = &cacheable {
                let data = BitmapGlyphData {
                    pixmap: &glyph.pixmap,
                    area: glyph.area,
                };
                if cacher.draw_and_cache_bitmap(renderer, &data, info) {
                    return;
                }
            }

            render_uncached_bitmap_glyph(renderer, glyph, transform);
        }
        GlyphType::Colr(glyph) => {
            let context_color = renderer.get_context_color();
            if let Some(info) = &cacheable {
                let data = ColrGlyphData {
                    area: glyph.area,
                    pix_width: glyph.pix_width,
                    pix_height: glyph.pix_height,
                    has_non_default_blend: glyph.has_non_default_blend,
                };
                let glyph_ref = &*glyph;
                let outline_cache_ref = &mut *outline_cache;
                if cacher.draw_and_cache_colr(renderer, &data, info, &mut |sink| {
                    paint_colr_glyph(glyph_ref, context_color, sink, outline_cache_ref);
                }) {
                    return;
                }
            }

            render_uncached_colr_glyph(renderer, &glyph, transform, context_color, outline_cache);
        }
    }
}

/// Stroke a prepared glyph.
///
/// Stroked outlines are never offered to the cacher (the stroke parameters
/// would need to be part of the cache key); COLR and bitmap glyphs are always
/// filled and delegate to [`fill_glyph`].
pub(crate) fn stroke_glyph<R: GlyphRenderer>(
    renderer: &mut R,
    cacher: &mut impl GlyphCacher<R>,
    prepared_glyph: PreparedGlyph<'_>,
    outline_cache: &mut OutlineCacheSession<'_>,
) {
    match prepared_glyph.glyph_type {
        GlyphType::Outline(ref glyph) => {
            stroke_uncached_outline_glyph(
                renderer,
                &glyph.path,
                glyph.scale,
                prepared_glyph.outline_transform,
                prepared_glyph.relative_paint_transform,
            );
        }
        GlyphType::Bitmap(_) | GlyphType::Colr(_) => {
            fill_glyph(renderer, cacher, prepared_glyph, outline_cache);
        }
    }
}

fn fill_uncached_outline_glyph(
    renderer: &mut impl GlyphRenderer,
    path: &BezPath,
    scale: f64,
    outline_transform: Affine,
    paint_transform: Affine,
) {
    let state = renderer.save_state();
    renderer.set_transform(outline_transform.pre_scale(scale));
    renderer.set_paint_transform(paint_transform);
    renderer.fill_path(path);
    renderer.restore_state(state);
}

fn stroke_uncached_outline_glyph(
    renderer: &mut impl GlyphRenderer,
    path: &BezPath,
    scale: f64,
    outline_transform: Affine,
    paint_transform: Affine,
) {
    let state = renderer.save_state();
    renderer.set_transform(outline_transform.pre_scale(scale));
    renderer.set_paint_transform(paint_transform);
    renderer.stroke_path(path);
    renderer.restore_state(state);
}

fn render_uncached_bitmap_glyph(
    renderer: &mut impl GlyphRenderer,
    glyph: GlyphBitmap,
    outline_transform: Affine,
) {
    let image = GlyphImage {
        pixmap: glyph.pixmap,
        quality: quality_for_scale(&outline_transform),
    };

    let state = renderer.save_state();
    renderer.set_transform(outline_transform);
    renderer.set_paint_image(image);
    renderer.fill_rect(&glyph.area);
    renderer.restore_state(state);
}

/// Paint a COLR glyph into a [`DrawSink`], wrapping it in the appropriate
/// clip/blend layer.
///
/// The sink is expected to already have the desired base transform set.
///
/// [`DrawSink`]: crate::DrawSink
fn paint_colr_glyph(
    glyph: &GlyphColr<'_>,
    context_color: AlphaColor<Srgb>,
    sink: &mut dyn crate::DrawSink,
    outline_cache: &mut OutlineCacheSession<'_>,
) {
    // Two reasons why we wrap COLR glyphs in a clip layer:
    // 1) We need a layer to make sure they are isolated and don't blend into the main surface (unless
    // the glyph is guaranteed to only use default blending, in which case we don't need this).
    // Otherwise, blend modes that are part of the glyph could affect already drawn contents.
    // 2) We do the clipping as a temporary measure to allow the Vello renderers to get a bounding box
    // of the glyph, necessary to keep the cost of blending operations with
    // destructive blend modes to a minimum.
    if glyph.has_non_default_blend {
        sink.push_clip_layer(&glyph.area.to_path(0.1));
    } else {
        sink.push_clip_path(&glyph.area.to_path(0.1));
    }

    // TODO: Maybe ColrPainter can be reused across glyphs?
    let mut colr_painter = ColrPainter::new(glyph, context_color, sink, outline_cache);
    colr_painter.paint();

    if glyph.has_non_default_blend {
        sink.pop_layer();
    } else {
        sink.pop_clip_path();
    }
}

fn render_uncached_colr_glyph(
    renderer: &mut impl GlyphRenderer,
    glyph: &GlyphColr<'_>,
    outline_transform: Affine,
    context_color: AlphaColor<Srgb>,
    outline_cache: &mut OutlineCacheSession<'_>,
) {
    let state = renderer.save_state();
    renderer.set_transform(outline_transform);
    paint_colr_glyph(glyph, context_color, renderer, outline_cache);
    renderer.restore_state(state);
}

/// Choose image sampling quality based on downscale factor.
///
/// Returns `High` when the transform scales below 50% (where aliasing is
/// visible), `Medium` otherwise.
#[inline]
pub fn quality_for_scale(transform: &Affine) -> ImageQuality {
    let [a, _, _, d, _, _] = transform.as_coeffs();
    if a < 0.5 || d < 0.5 {
        ImageQuality::High
    } else {
        ImageQuality::Medium
    }
}
