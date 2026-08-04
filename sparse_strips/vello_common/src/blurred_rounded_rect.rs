// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Blurred, rounded rectangles.
use crate::color::{AlphaColor, Srgb};
use crate::kurbo::{Rect, RoundedRectRadii, Vec2};

/// Per-corner, possibly elliptical, radii of a rounded rectangle.
///
/// Each corner has an x radius and a y radius (the `x`/`y` components of the
/// corresponding [`Vec2`]), matching the semantics of CSS `border-radius`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CornerRadii {
    /// The radii of the top-left corner.
    pub top_left: Vec2,
    /// The radii of the top-right corner.
    pub top_right: Vec2,
    /// The radii of the bottom-left corner.
    pub bottom_left: Vec2,
    /// The radii of the bottom-right corner.
    pub bottom_right: Vec2,
}

impl CornerRadii {
    /// Create a new `CornerRadii` from per-corner elliptical radii.
    pub fn new(top_left: Vec2, top_right: Vec2, bottom_left: Vec2, bottom_right: Vec2) -> Self {
        Self {
            top_left,
            top_right,
            bottom_left,
            bottom_right,
        }
    }
}

impl From<RoundedRectRadii> for CornerRadii {
    fn from(radii: RoundedRectRadii) -> Self {
        Self {
            top_left: Vec2::new(radii.top_left, radii.top_left),
            top_right: Vec2::new(radii.top_right, radii.top_right),
            bottom_left: Vec2::new(radii.bottom_left, radii.bottom_left),
            bottom_right: Vec2::new(radii.bottom_right, radii.bottom_right),
        }
    }
}

impl From<f64> for CornerRadii {
    fn from(radius: f64) -> Self {
        RoundedRectRadii::from(radius).into()
    }
}

/// A blurred, rounded rectangle.
#[derive(Debug)]
pub struct BlurredRoundedRectangle {
    /// The base rectangle to use for the blur effect.
    pub rect: Rect,
    /// The color of the blurred rectangle.
    pub color: AlphaColor<Srgb>,
    /// The radii of the rounded rectangle's corners.
    ///
    /// Each corner may have a different, possibly elliptical, radius.
    pub radii: CornerRadii,
    /// The standard deviation of the blur effect.
    pub std_dev: f32,
    /// Whether to paint the inverse (`1 - alpha`) of the blur coverage.
    ///
    /// When `true`, the paint is fully opaque outside the blurred rectangle and fades to
    /// transparent inside it. This is useful for implementing inset box shadows.
    pub invert: bool,
}
