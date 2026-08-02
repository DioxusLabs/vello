// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Image types for bitmap (emoji) glyphs.

use alloc::sync::Arc;
use peniko::ImageQuality;
use pixmap::Pixmap;

/// An image paint for an uncached bitmap glyph.
#[derive(Clone, Debug)]
pub struct GlyphImage {
    /// The pixel data of the glyph.
    pub pixmap: Arc<Pixmap>,
    /// The suggested sampling quality, derived from the glyph's transform.
    pub quality: ImageQuality,
}
