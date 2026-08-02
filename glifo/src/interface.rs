// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Traits for rendering glyphs and replaying them.

use crate::bitmap::GlyphImage;
use crate::color::{AlphaColor, Srgb};
use crate::kurbo::{Affine, BezPath, Rect};
use crate::peniko::{BlendMode, Gradient};

/// A renderer-neutral paint for glyph drawing.
///
/// COLR glyphs only require solid colors and gradients, both of which are
/// expressible with `peniko` types.
#[derive(Clone, Debug)]
pub enum GlyphPaint {
    /// A solid color.
    Solid(AlphaColor<Srgb>),
    /// A gradient.
    Gradient(Gradient),
}

impl From<AlphaColor<Srgb>> for GlyphPaint {
    fn from(color: AlphaColor<Srgb>) -> Self {
        Self::Solid(color)
    }
}

impl From<Gradient> for GlyphPaint {
    fn from(gradient: Gradient) -> Self {
        Self::Gradient(gradient)
    }
}

// TODO: This trait is only temporary and will hopefully be replaced once we have a better
// unifying imaging API.
/// A sink for low-level glyph drawing commands.
pub trait DrawSink {
    /// Set the current transform.
    fn set_transform(&mut self, t: Affine);
    /// Set the current paint.
    fn set_paint(&mut self, paint: GlyphPaint);
    /// Set the paint transform.
    fn set_paint_transform(&mut self, t: Affine);
    /// Fill a path with the current paint and transform.
    fn fill_path(&mut self, path: &BezPath);
    /// Fill a rectangle with the current paint and transform.
    fn fill_rect(&mut self, rect: &Rect);
    // TODO: `push/pop_clip_layer` are only needed temporarily for Vello renderers so the impact of
    // destructive blend modes can be limited to a smaller area. Once the limitation is lifted,
    // we can remove this method and just use `push_clip_path` and `push_blend_layer`.
    /// Push a clip layer defined by a path.
    fn push_clip_layer(&mut self, clip: &BezPath);
    /// Push a clip path.
    ///
    /// This method _can_ be implemented in terms of `push_clip_layer` without losing correctness.
    /// However, clients are allowed to choose a more efficient clipping implementation that
    /// doesn't require pushing an isolated layer.
    ///
    /// When providing a custom implementation of this method, the `pop_clip_path` method also
    /// needs to be overridden correspondingly.
    fn push_clip_path(&mut self, clip: &BezPath) {
        self.push_clip_layer(clip);
    }
    /// Push a blend/compositing layer.
    fn push_blend_layer(&mut self, blend_mode: BlendMode);
    /// Pop the most recent clip or blend layer.
    fn pop_layer(&mut self);
    /// Pop the most recent clip path.
    ///
    /// See the documentation for [`DrawSink::push_clip_path`].
    fn pop_clip_path(&mut self) {
        self.pop_layer();
    }
    /// Width of the surface.
    fn width(&self) -> u16;
    /// Height of the surface.
    fn height(&self) -> u16;
}

impl<T: DrawSink + ?Sized> DrawSink for &mut T {
    fn set_transform(&mut self, t: Affine) {
        (**self).set_transform(t);
    }
    fn set_paint(&mut self, paint: GlyphPaint) {
        (**self).set_paint(paint);
    }
    fn set_paint_transform(&mut self, t: Affine) {
        (**self).set_paint_transform(t);
    }
    fn fill_path(&mut self, path: &BezPath) {
        (**self).fill_path(path);
    }
    fn fill_rect(&mut self, rect: &Rect) {
        (**self).fill_rect(rect);
    }
    fn push_clip_layer(&mut self, clip: &BezPath) {
        (**self).push_clip_layer(clip);
    }
    fn push_clip_path(&mut self, clip: &BezPath) {
        (**self).push_clip_path(clip);
    }
    fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        (**self).push_blend_layer(blend_mode);
    }
    fn pop_layer(&mut self) {
        (**self).pop_layer();
    }
    fn pop_clip_path(&mut self) {
        (**self).pop_clip_path();
    }
    fn width(&self) -> u16 {
        (**self).width()
    }
    fn height(&self) -> u16 {
        (**self).height()
    }
}

/// A stateful renderer that can draw sequences of glyphs.
pub trait GlyphRenderer: DrawSink {
    /// The type of state used by the renderer.
    type SavedState;

    /// Save the current state.
    fn save_state(&mut self) -> Self::SavedState;

    /// Restore the current state.
    fn restore_state(&mut self, state: Self::SavedState);

    /// Stroke a path with the current paint and stroke settings.
    fn stroke_path(&mut self, path: &BezPath);

    /// Set the current paint to a bitmap glyph image.
    fn set_paint_image(&mut self, image: GlyphImage);

    /// Get the context color from the renderer's current paint, used for resolving the
    /// context-dependent colors of COLR glyphs.
    fn get_context_color(&self) -> AlphaColor<Srgb>;
}
