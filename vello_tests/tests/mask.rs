// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::renderer::Renderer;
use smallvec::smallvec;
use vello_common::color::DynamicColor;
use vello_common::color::palette::css::{BLACK, BLUE, GRAY, LIME, RED, WHITE, YELLOW};
use vello_common::kurbo::{Point, Rect, Shape};
use vello_common::mask::Mask;
use vello_common::peniko::{BlendMode, ColorStop, ColorStops, Compose, Gradient, Mix};
use vello_cpu::peniko::LinearGradientPosition;
use vello_cpu::{Level, RasterizerSettings, RenderMode, RenderSettings};
use vello_cpu::{Pixmap, RenderContext};
use vello_dev_macros::vello_test;

pub(crate) fn example_mask(alpha_mask: bool) -> Mask {
    let mut mask_pix = Pixmap::new(100, 100);
    // TODO: Would be nice to take the settings from the current test context.
    let settings = RenderSettings {
        level: Level::baseline(),
        num_threads: 0,
    };
    let mut mask_ctx = RenderContext::new_with(100, 100, settings);
    let mut resources = vello_cpu::Resources::new();

    let grad = Gradient {
        kind: LinearGradientPosition {
            start: Point::new(10.0, 0.0),
            end: Point::new(90.0, 0.0),
        }
        .into(),
        stops: ColorStops(smallvec![
            ColorStop {
                offset: 0.0,
                color: DynamicColor::from_alpha_color(RED),
            },
            ColorStop {
                offset: 0.5,
                color: DynamicColor::from_alpha_color(YELLOW.with_alpha(0.5)),
            },
            ColorStop {
                offset: 1.0,
                color: DynamicColor::from_alpha_color(LIME.with_alpha(0.0)),
            },
        ]),
        ..Default::default()
    };

    mask_ctx.set_paint(grad);
    mask_ctx.fill_rect(&Rect::new(10.0, 10.0, 90.0, 90.0));
    mask_ctx.render(&mut mask_pix, &mut resources);

    if alpha_mask {
        Mask::new_alpha(&mask_pix)
    } else {
        Mask::new_luminance(&mask_pix)
    }
}

fn mask(ctx: &mut impl Renderer, alpha_mask: bool) {
    let mask = example_mask(alpha_mask);

    ctx.set_paint(BLACK);
    ctx.fill_rect(&Rect::new(10.0, 10.0, 90.0, 90.0));
    ctx.push_mask_layer(mask);
    ctx.set_paint(RED);
    ctx.fill_rect(&Rect::new(10.0, 10.0, 90.0, 90.0));
    ctx.pop_layer();
}

#[vello_test]
fn mask_alpha(ctx: &mut impl Renderer) {
    mask(ctx, true);
}

#[vello_test]
fn mask_luminance(ctx: &mut impl Renderer) {
    mask(ctx, false);
}

#[vello_test(skip_gpu)]
fn mask_non_isolated(ctx: &mut impl Renderer) {
    let mask = example_mask(false);

    ctx.set_paint(BLACK);
    ctx.fill_rect(&Rect::new(10.0, 10.0, 90.0, 90.0));
    ctx.set_mask(mask);
    ctx.set_paint(RED);
    ctx.fill_rect(&Rect::new(10.0, 10.0, 90.0, 90.0));
}

fn luminance_mask_gradient(start: f64, end: f64) -> Gradient {
    Gradient {
        kind: LinearGradientPosition {
            start: Point::new(start, 0.0),
            end: Point::new(end, 0.0),
        }
        .into(),
        stops: ColorStops(smallvec![
            ColorStop {
                offset: 0.0,
                color: DynamicColor::from_alpha_color(WHITE),
            },
            ColorStop {
                offset: 1.0,
                color: DynamicColor::from_alpha_color(BLACK),
            },
        ]),
        ..Default::default()
    }
}

#[vello_test(skip_gpu)]
fn luminance_mask_layer_gradient(ctx: &mut impl Renderer) {
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);
    ctx.set_paint(RED);
    ctx.fill_rect(&rect);
    ctx.push_luminance_mask_layer(Some(&rect.to_path(0.1)), 1.0);
    ctx.set_paint(luminance_mask_gradient(10.0, 90.0));
    ctx.fill_rect(&rect);
    ctx.pop_layer();
}

// Columns masked with white, lime, translucent white, gray and black; the bottom strip has no
// mask content and must be removed.
#[vello_test(skip_gpu)]
fn luminance_mask_layer_colors(ctx: &mut impl Renderer) {
    ctx.set_paint(BLUE);
    ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
    ctx.push_luminance_mask_layer(Some(&Rect::new(0.0, 0.0, 100.0, 100.0).to_path(0.1)), 1.0);
    let colors = [WHITE, LIME, WHITE.with_alpha(0.5), GRAY, BLACK];
    for (i, color) in colors.into_iter().enumerate() {
        let x = i as f64 * 20.0;
        ctx.set_paint(color);
        ctx.fill_rect(&Rect::new(x, 0.0, x + 20.0, 80.0));
    }
    ctx.pop_layer();
}

// Content outside of the clip path is kept, content inside of it without mask content is removed.
#[vello_test(skip_gpu)]
fn luminance_mask_layer_clipped(ctx: &mut impl Renderer) {
    ctx.set_paint(RED);
    ctx.fill_rect(&Rect::new(10.0, 10.0, 90.0, 90.0));
    let clip = Rect::new(30.0, 30.0, 70.0, 70.0);
    ctx.push_luminance_mask_layer(Some(&clip.to_path(0.1)), 1.0);
    ctx.set_paint(WHITE);
    ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 50.0));
    ctx.pop_layer();
}

#[vello_test(skip_gpu)]
fn luminance_mask_layer_unclipped(ctx: &mut impl Renderer) {
    ctx.set_paint(RED);
    ctx.fill_rect(&Rect::new(10.0, 10.0, 90.0, 90.0));
    ctx.push_luminance_mask_layer(None, 1.0);
    ctx.set_paint(luminance_mask_gradient(30.0, 70.0));
    ctx.fill_rect(&Rect::new(30.0, 30.0, 70.0, 70.0));
    ctx.pop_layer();
}

#[vello_test(skip_gpu)]
fn luminance_mask_layer_opacity(ctx: &mut impl Renderer) {
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);
    ctx.set_paint(RED);
    ctx.fill_rect(&rect);
    ctx.push_luminance_mask_layer(Some(&rect.to_path(0.1)), 0.5);
    ctx.set_paint(luminance_mask_gradient(10.0, 90.0));
    ctx.fill_rect(&rect);
    ctx.pop_layer();
}

// A luminance mask layer inside the content of another one masks only that content.
#[vello_test(skip_gpu)]
fn luminance_mask_layer_nested(ctx: &mut impl Renderer) {
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);
    ctx.set_paint(RED);
    ctx.fill_rect(&rect);
    ctx.push_luminance_mask_layer(Some(&rect.to_path(0.1)), 1.0);
    ctx.set_paint(WHITE);
    ctx.fill_rect(&rect);
    ctx.push_luminance_mask_layer(Some(&rect.to_path(0.1)), 1.0);
    ctx.set_paint(WHITE);
    ctx.fill_rect(&Rect::new(10.0, 10.0, 90.0, 50.0));
    ctx.set_paint(GRAY);
    ctx.fill_rect(&Rect::new(10.0, 50.0, 90.0, 90.0));
    ctx.pop_layer();
    ctx.push_luminance_mask_layer(Some(&rect.to_path(0.1)), 1.0);
    ctx.set_paint(luminance_mask_gradient(10.0, 90.0));
    ctx.fill_rect(&rect);
    ctx.pop_layer();
    ctx.pop_layer();
}

// Pushing a luminance mask layer matches compositing an opaque layer with a pre-rasterized
// `Mask::new_luminance` mask using `Compose::DestIn`.
#[test]
fn luminance_mask_layer_matches_luminance_mask() {
    for render_mode in [RenderMode::OptimizeSpeed, RenderMode::OptimizeQuality] {
        for num_threads in [0, 4] {
            let settings = RenderSettings {
                level: Level::baseline(),
                num_threads,
            };
            let backdrop = Rect::new(0.0, 0.0, 100.0, 100.0);
            let draw_mask = |ctx: &mut RenderContext| {
                ctx.set_paint(luminance_mask_gradient(10.0, 90.0));
                ctx.fill_rect(&Rect::new(10.0, 10.0, 90.0, 50.0));
                ctx.set_paint(LIME.with_alpha(0.5));
                ctx.fill_rect(&Rect::new(10.0, 50.0, 90.0, 90.0));
            };
            let render = |ctx: &mut RenderContext| {
                let mut pixmap = Pixmap::new(100, 100);
                ctx.flush();
                ctx.render_with(
                    &mut pixmap,
                    &mut vello_cpu::Resources::new(),
                    RasterizerSettings {
                        render_mode,
                        ..Default::default()
                    },
                );
                pixmap
            };

            let mut ctx = RenderContext::new_with(100, 100, settings);
            draw_mask(&mut ctx);
            let mask = Mask::new_luminance(&render(&mut ctx));

            let mut ctx = RenderContext::new_with(100, 100, settings);
            ctx.set_paint(RED);
            ctx.fill_rect(&backdrop);
            ctx.push_layer(
                None,
                Some(BlendMode::new(Mix::Normal, Compose::DestIn)),
                None,
                Some(mask),
                None,
            );
            ctx.set_paint(WHITE);
            ctx.fill_rect(&backdrop);
            ctx.pop_layer();
            let expected = render(&mut ctx);

            let mut ctx = RenderContext::new_with(100, 100, settings);
            ctx.set_paint(RED);
            ctx.fill_rect(&backdrop);
            ctx.push_luminance_mask_layer(None, 1.0);
            draw_mask(&mut ctx);
            ctx.pop_layer();
            let actual = render(&mut ctx);

            for (i, (e, a)) in expected.data().iter().zip(actual.data()).enumerate() {
                assert!(
                    e.r.abs_diff(a.r) <= 1 && e.a.abs_diff(a.a) <= 1,
                    "pixel {i} differs ({render_mode:?}, {num_threads} threads): expected {e:?}, got {a:?}"
                );
            }
        }
    }
}
