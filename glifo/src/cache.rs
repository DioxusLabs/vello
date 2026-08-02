// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Backend glyph caching hooks.
//!
//! Glifo itself does not own any glyph bitmap cache. Instead, the glyph
//! rendering loop offers each glyph to a backend-provided [`GlyphCacher`]
//! before (and instead of) rendering it directly. A cacher that recognises a
//! glyph (by its [`CacheableGlyph`] description) can draw it from its own
//! atlas; otherwise glifo falls back to direct rendering, giving the cacher a
//! chance to record/insert the freshly prepared glyph on the way.
//!
//! `vello_common` provides an atlas-backed implementation of this trait that
//! is shared by the Vello CPU and Vello Hybrid renderers.

use crate::color::{AlphaColor, Srgb};
use crate::glyph::{FontEmbolden, NormalizedCoord};
use crate::interface::{DrawSink, GlyphRenderer};
use crate::kurbo::{Affine, BezPath, Rect};
use alloc::sync::Arc;
use vello_pixmap::Pixmap;

/// Premultiply and pack an RGBA color into a `u32` for bitwise hashing/comparison.
#[inline]
pub fn pack_color(color: AlphaColor<Srgb>) -> u32 {
    color.premultiply().to_rgba8().to_u32()
}

/// Which glyph kinds a [`GlyphCacher`] wants to be offered for the current run.
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheRunConfig {
    /// Whether outline glyphs should be offered to the cacher.
    pub cache_outlines: bool,
    /// Whether COLR and bitmap glyphs should be offered to the cacher.
    pub cache_colr_bitmap: bool,
}

/// The kind of a cacheable glyph.
#[derive(Clone, Copy, Debug)]
pub enum CacheableGlyphKind {
    /// An outline glyph.
    Outline,
    /// A bitmap glyph.
    Bitmap,
    /// A COLR glyph. The `Rect` contains the fractional area dimensions used
    /// to preserve sub-pixel accuracy when drawing from a cache.
    Colr(Rect),
}

/// Everything a backend cache needs to identify and position a glyph.
///
/// This contains the full set of cache key "ingredients" (font identity,
/// size, hinting, subpixel position, context color, embolden parameters and
/// variation coordinates) plus the transform at which the glyph would be
/// drawn.
#[derive(Clone, Debug)]
pub struct CacheableGlyph<'a> {
    /// Unique identifier for the font blob.
    pub font_id: u64,
    /// Index within font collection (for TTC files).
    pub font_index: u32,
    /// Glyph index within the font.
    pub glyph_id: u32,
    /// Font size in pixels per em. For bitmap glyphs this is the strike's own
    /// ppem (the size the embedded image was pre-rendered at), not the run's.
    pub font_size: f32,
    /// Whether hinting was applied. Always `false` for COLR/bitmap glyphs.
    pub hinted: bool,
    /// Horizontal fractional pixel offset of the glyph. Only meaningful for
    /// outline glyphs; `0.0` otherwise.
    pub fractional_x: f32,
    /// Context color for COLR glyphs (the run's solid paint color).
    /// `BLACK` for outline and bitmap glyphs.
    pub context_color: AlphaColor<Srgb>,
    /// Pre-packed premultiplied RGBA8 version of `context_color` (see
    /// [`pack_color`]), suitable for hashing/comparison.
    pub context_color_packed: u32,
    /// Synthetic embolden settings. Only meaningful for outline glyphs;
    /// `FontEmbolden::default()` otherwise.
    pub embolden: FontEmbolden,
    /// Normalized variation coordinates for variable fonts. Empty for bitmap
    /// glyphs (fixed strikes are unaffected by variations).
    pub var_coords: &'a [NormalizedCoord],
    /// The kind of glyph.
    pub kind: CacheableGlyphKind,
    /// The transform at which the glyph is drawn (glifo's per-glyph outline
    /// transform; any font-size scaling has already been absorbed).
    pub transform: Affine,
}

/// A prepared outline glyph offered to a [`GlyphCacher`] for insertion.
#[derive(Debug)]
pub struct OutlineGlyphData<'a> {
    /// The glyph path (in outline-cache units).
    pub path: &'a Arc<BezPath>,
    /// Precise bounding box of the path at the cached outline size.
    pub bbox: Rect,
    /// Scale from the cached outline size to the requested draw size.
    pub scale: f64,
}

/// A prepared bitmap glyph offered to a [`GlyphCacher`] for insertion.
#[derive(Debug)]
pub struct BitmapGlyphData<'a> {
    /// The decoded pixel data.
    pub pixmap: &'a Arc<Pixmap>,
    /// The rectangular area that should be filled with the bitmap when painting.
    pub area: Rect,
}

/// A prepared COLR glyph offered to a [`GlyphCacher`] for insertion.
///
/// The actual paint graph is only accessible through the `paint` closure
/// passed to [`GlyphCacher::draw_and_cache_colr`], which replays the glyph's
/// draw commands into any [`DrawSink`].
#[derive(Debug)]
pub struct ColrGlyphData {
    /// The rectangular area covered by the rendered glyph.
    pub area: Rect,
    /// The width in pixels of the texture the glyph should be rendered to.
    pub pix_width: u16,
    /// The height in pixels of the texture the glyph should be rendered to.
    pub pix_height: u16,
    /// Whether the glyph paint graph uses a non-default blend mode.
    pub has_non_default_blend: bool,
}

/// A backend-provided glyph cache, called from glifo's glyph rendering loop.
///
/// All methods return `true` if the cacher drew the glyph (from cache or
/// after inserting it), in which case glifo skips direct rendering.
pub trait GlyphCacher<R: GlyphRenderer> {
    /// Decide which glyph kinds should be offered to the cacher for a run at
    /// the given (post-absorption) font size.
    ///
    /// Note that outline offers additionally require a `Fill` style; glifo
    /// never offers stroked outlines.
    fn run_config(&self, renderer: &R, font_size: f32) -> CacheRunConfig;

    /// Attempt to draw an already-cached glyph.
    ///
    /// This is called *before* glifo does any font-table lookups or outline
    /// construction, so a hit here also skips glyph preparation entirely
    /// (the speculative cache probe).
    fn draw_cached_glyph(&mut self, renderer: &mut R, glyph: &CacheableGlyph<'_>) -> bool;

    /// Insert a freshly prepared outline glyph and draw it from the cache.
    fn draw_and_cache_outline(
        &mut self,
        renderer: &mut R,
        data: &OutlineGlyphData<'_>,
        glyph: &CacheableGlyph<'_>,
    ) -> bool;

    /// Insert a freshly decoded bitmap glyph and draw it from the cache.
    fn draw_and_cache_bitmap(
        &mut self,
        renderer: &mut R,
        data: &BitmapGlyphData<'_>,
        glyph: &CacheableGlyph<'_>,
    ) -> bool;

    /// Insert a freshly prepared COLR glyph and draw it from the cache.
    ///
    /// `paint` replays the glyph's draw commands into any [`DrawSink`]
    /// (typically a command recorder targeting an atlas page). The sink is
    /// expected to already have the appropriate transform set.
    fn draw_and_cache_colr(
        &mut self,
        renderer: &mut R,
        data: &ColrGlyphData,
        glyph: &CacheableGlyph<'_>,
        paint: &mut dyn FnMut(&mut dyn DrawSink),
    ) -> bool;
}

impl<R: GlyphRenderer, C: GlyphCacher<R> + ?Sized> GlyphCacher<R> for &mut C {
    fn run_config(&self, renderer: &R, font_size: f32) -> CacheRunConfig {
        (**self).run_config(renderer, font_size)
    }

    fn draw_cached_glyph(&mut self, renderer: &mut R, glyph: &CacheableGlyph<'_>) -> bool {
        (**self).draw_cached_glyph(renderer, glyph)
    }

    fn draw_and_cache_outline(
        &mut self,
        renderer: &mut R,
        data: &OutlineGlyphData<'_>,
        glyph: &CacheableGlyph<'_>,
    ) -> bool {
        (**self).draw_and_cache_outline(renderer, data, glyph)
    }

    fn draw_and_cache_bitmap(
        &mut self,
        renderer: &mut R,
        data: &BitmapGlyphData<'_>,
        glyph: &CacheableGlyph<'_>,
    ) -> bool {
        (**self).draw_and_cache_bitmap(renderer, data, glyph)
    }

    fn draw_and_cache_colr(
        &mut self,
        renderer: &mut R,
        data: &ColrGlyphData,
        glyph: &CacheableGlyph<'_>,
        paint: &mut dyn FnMut(&mut dyn DrawSink),
    ) -> bool {
        (**self).draw_and_cache_colr(renderer, data, glyph, paint)
    }
}

/// A [`GlyphCacher`] that never caches; all glyphs render directly.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoCache;

impl<R: GlyphRenderer> GlyphCacher<R> for NoCache {
    fn run_config(&self, _renderer: &R, _font_size: f32) -> CacheRunConfig {
        CacheRunConfig::default()
    }

    fn draw_cached_glyph(&mut self, _renderer: &mut R, _glyph: &CacheableGlyph<'_>) -> bool {
        false
    }

    fn draw_and_cache_outline(
        &mut self,
        _renderer: &mut R,
        _data: &OutlineGlyphData<'_>,
        _glyph: &CacheableGlyph<'_>,
    ) -> bool {
        false
    }

    fn draw_and_cache_bitmap(
        &mut self,
        _renderer: &mut R,
        _data: &BitmapGlyphData<'_>,
        _glyph: &CacheableGlyph<'_>,
    ) -> bool {
        false
    }

    fn draw_and_cache_colr(
        &mut self,
        _renderer: &mut R,
        _data: &ColrGlyphData,
        _glyph: &CacheableGlyph<'_>,
        _paint: &mut dyn FnMut(&mut dyn DrawSink),
    ) -> bool {
        false
    }
}
