// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Glyph bitmap atlas cache for efficient text rendering.
//!
//! This module provides a glyph bitmap atlas cache that:
//! - Rasterizes glyphs once and reuses the bitmaps for subsequent draws
//! - Packs glyph bitmaps into shared atlas images using guillotiere
//! - Uses stable glyph keys for reliable cache hits
//! - Supports multiple atlas pages for scalability
//! - Implements simple age-based eviction
//!
//! The core type is [`GlyphAtlas`], which handles allocation, eviction,
//! and pending-command queues. [`AtlasGlyphCacher`] implements glifo's
//! `GlyphCacher` hook on top of it for renderers implementing
//! [`AtlasGlyphRenderer`].

pub mod cache;
mod cacher;
pub mod commands;
pub mod key;
mod region;

#[cfg(all(debug_assertions, feature = "std"))]
pub use cache::GlyphCacheStats;
pub use cache::{
    GLYPH_PADDING, GlyphAtlas, GlyphCacheConfig, PendingBitmapUpload, PendingClearRect,
};
pub use cacher::{AtlasGlyphCacher, AtlasGlyphRenderer, replay_atlas_commands};
pub use commands::{AtlasCommand, AtlasCommandRecorder, paint_type_from_glyph_paint};
pub use key::GlyphCacheKey;
pub use region::{AtlasSlot, RasterMetrics};
