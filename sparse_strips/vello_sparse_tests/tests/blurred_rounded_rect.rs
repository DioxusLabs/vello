// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::renderer::Renderer;
use vello_common::blurred_rounded_rect::CornerRadii;
use vello_common::color::palette::css::REBECCA_PURPLE;
use vello_common::kurbo::{Affine, Point, Rect, RoundedRectRadii, Vec2};
use vello_dev_macros::vello_test;

fn rect_with(ctx: &mut impl Renderer, radius: f32, std_dev: f32, affine: Affine) {
    rect_with_radii(ctx, f64::from(radius).into(), std_dev, affine);
}

fn rect_with_radii(ctx: &mut impl Renderer, radii: CornerRadii, std_dev: f32, affine: Affine) {
    let rect = Rect::new(20.0, 20.0, 80.0, 80.0);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.set_transform(affine);
    ctx.fill_blurred_rounded_rect(&rect, radii, std_dev, false);
}

#[vello_test]
fn blurred_rounded_rect_zero(ctx: &mut impl Renderer) {
    rect_with(ctx, 0.0, 0.0, Affine::IDENTITY);
}

#[vello_test]
fn blurred_rounded_rect_zero_with_radius(ctx: &mut impl Renderer) {
    rect_with(ctx, 10.0, 0.0, Affine::IDENTITY);
}

#[vello_test]
fn blurred_rounded_rect_none(ctx: &mut impl Renderer) {
    rect_with(ctx, 0.0, 0.1, Affine::IDENTITY);
}

#[vello_test]
fn blurred_rounded_rect_small_std_dev(ctx: &mut impl Renderer) {
    rect_with(ctx, 0.0, 5.0, Affine::IDENTITY);
}

#[vello_test]
fn blurred_rounded_rect_medium_std_dev(ctx: &mut impl Renderer) {
    rect_with(ctx, 0.0, 10.0, Affine::IDENTITY);
}

#[vello_test]
fn blurred_rounded_rect_large_std_dev(ctx: &mut impl Renderer) {
    rect_with(ctx, 0.0, 20.0, Affine::IDENTITY);
}

#[vello_test]
fn blurred_rounded_rect_with_radius(ctx: &mut impl Renderer) {
    rect_with(ctx, 10.0, 10.0, Affine::IDENTITY);
}

#[vello_test]
fn blurred_rounded_rect_with_large_radius(ctx: &mut impl Renderer) {
    rect_with(ctx, 30.0, 10.0, Affine::IDENTITY);
}

#[vello_test]
fn blurred_rounded_rect_with_transform(ctx: &mut impl Renderer) {
    rect_with(
        ctx,
        10.0,
        10.0,
        Affine::rotate_about(45.0_f64.to_radians(), Point::new(50.0, 50.0)),
    );
}

#[vello_test]
fn blurred_rounded_rect_non_uniform_radii(ctx: &mut impl Renderer) {
    rect_with_radii(
        ctx,
        RoundedRectRadii::new(20.0, 0.0, 10.0, 5.0).into(),
        3.0,
        Affine::IDENTITY,
    );
}

#[vello_test]
fn blurred_rounded_rect_non_uniform_radii_large_std_dev(ctx: &mut impl Renderer) {
    rect_with_radii(
        ctx,
        RoundedRectRadii::new(30.0, 0.0, 15.0, 5.0).into(),
        10.0,
        Affine::IDENTITY,
    );
}

#[vello_test]
fn blurred_rounded_rect_non_uniform_radii_with_transform(ctx: &mut impl Renderer) {
    rect_with_radii(
        ctx,
        RoundedRectRadii::new(20.0, 0.0, 10.0, 5.0).into(),
        3.0,
        Affine::rotate_about(45.0_f64.to_radians(), Point::new(50.0, 50.0)),
    );
}

#[vello_test]
fn blurred_rounded_rect_elliptical_radii(ctx: &mut impl Renderer) {
    rect_with_radii(
        ctx,
        CornerRadii::new(
            Vec2::new(25.0, 10.0),
            Vec2::new(5.0, 20.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(15.0, 30.0),
        ),
        2.0,
        Affine::IDENTITY,
    );
}

#[vello_test]
fn blurred_rounded_rect_elliptical_radii_std_dev(ctx: &mut impl Renderer) {
    rect_with_radii(
        ctx,
        CornerRadii::new(
            Vec2::new(25.0, 10.0),
            Vec2::new(5.0, 20.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(15.0, 30.0),
        ),
        6.0,
        Affine::IDENTITY,
    );
}

#[vello_test]
fn blurred_rounded_rect_elliptical_pill(ctx: &mut impl Renderer) {
    // A fully elliptical box: all radii at half the rectangle's size.
    ctx.set_paint(REBECCA_PURPLE);
    ctx.set_transform(Affine::IDENTITY);
    let rect = Rect::new(10.0, 25.0, 90.0, 75.0);
    let radii = CornerRadii::new(
        Vec2::new(40.0, 25.0),
        Vec2::new(40.0, 25.0),
        Vec2::new(40.0, 25.0),
        Vec2::new(40.0, 25.0),
    );
    ctx.fill_blurred_rounded_rect(&rect, radii, 3.0, false);
}

fn inverse_rect_with(ctx: &mut impl Renderer, radius: f32, std_dev: f32, affine: Affine) {
    let rect = Rect::new(20.0, 20.0, 80.0, 80.0);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.set_transform(affine);
    ctx.fill_blurred_rounded_rect(&rect, f64::from(radius).into(), std_dev, true);
}

#[vello_test]
fn inverse_blurred_rounded_rect_small_std_dev(ctx: &mut impl Renderer) {
    inverse_rect_with(ctx, 0.0, 5.0, Affine::IDENTITY);
}

#[vello_test]
fn inverse_blurred_rounded_rect_medium_std_dev(ctx: &mut impl Renderer) {
    inverse_rect_with(ctx, 0.0, 10.0, Affine::IDENTITY);
}

#[vello_test]
fn inverse_blurred_rounded_rect_large_std_dev(ctx: &mut impl Renderer) {
    inverse_rect_with(ctx, 0.0, 20.0, Affine::IDENTITY);
}

#[vello_test]
fn inverse_blurred_rounded_rect_with_radius(ctx: &mut impl Renderer) {
    inverse_rect_with(ctx, 10.0, 10.0, Affine::IDENTITY);
}

#[vello_test]
fn inverse_blurred_rounded_rect_with_large_radius(ctx: &mut impl Renderer) {
    inverse_rect_with(ctx, 30.0, 10.0, Affine::IDENTITY);
}

#[vello_test]
fn inverse_blurred_rounded_rect_with_transform(ctx: &mut impl Renderer) {
    inverse_rect_with(
        ctx,
        10.0,
        10.0,
        Affine::rotate_about(45.0_f64.to_radians(), Point::new(50.0, 50.0)),
    );
}

#[vello_test]
fn inverse_blurred_rounded_rect_non_uniform_radii(ctx: &mut impl Renderer) {
    let rect = Rect::new(20.0, 20.0, 80.0, 80.0);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.set_transform(Affine::IDENTITY);
    ctx.fill_blurred_rounded_rect(
        &rect,
        RoundedRectRadii::new(20.0, 0.0, 10.0, 5.0).into(),
        3.0,
        true,
    );
}

#[vello_test]
fn inverse_blurred_rounded_rect_elliptical_radii(ctx: &mut impl Renderer) {
    let rect = Rect::new(20.0, 20.0, 80.0, 80.0);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.set_transform(Affine::IDENTITY);
    let radii = CornerRadii::new(
        Vec2::new(25.0, 10.0),
        Vec2::new(5.0, 20.0),
        Vec2::new(0.0, 0.0),
        Vec2::new(15.0, 30.0),
    );
    ctx.fill_blurred_rounded_rect(&rect, radii, 3.0, true);
}
