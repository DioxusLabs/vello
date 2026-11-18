// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests to validate our snapshot testing ability

use std::sync::Arc;

use scenes::SimpleText;
use vello::{
    Glyph, Scene,
    kurbo::{Affine, Circle, Rect},
    peniko::{Blob, Brush, Fill, FontData, Gradient, color::palette},
};
use vello_research_tests::{TestParams, smoke_snapshot_test_sync};

fn filled_square(use_cpu: bool) {
    let mut scene = Scene::new();
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &Brush::Solid(palette::css::BLUE),
        None,
        &Rect::from_center_size((10., 10.), (6., 6.)),
    );
    let params = TestParams {
        use_cpu,
        ..TestParams::new("filled_square", 20, 20)
    };
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.01);
}

fn filled_circle(use_cpu: bool) {
    let mut scene = Scene::new();
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &Brush::Solid(palette::css::BLUE),
        None,
        &Circle::new((10., 10.), 7.),
    );
    let params = TestParams {
        use_cpu,
        ..TestParams::new("filled_circle", 20, 20)
    };
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.01);
}

fn two_emoji(use_cpu: bool) {
    let mut scene = Scene::new();
    let mut text = SimpleText::new();
    text.add_colr_emoji_run(
        &mut scene,
        24.,
        Affine::translate((0., 24.)),
        None,
        Fill::NonZero,
        "🤠",
    );
    text.add_bitmap_emoji_run(
        &mut scene,
        24.,
        Affine::translate((30., 24.)),
        None,
        Fill::NonZero,
        "🤠",
    );
    let params = TestParams {
        use_cpu,
        ..TestParams::new("two_emoji", 60, 30)
    };
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.01);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn emoji_glyph_transform() {
    let mut scene = Scene::new();
    let colr = include_bytes!("../../../assets/noto_color_emoji/NotoColorEmoji-Subset.ttf");
    let bitmap = include_bytes!("../../../assets/noto_color_emoji/NotoColorEmoji-CBTF-Subset.ttf");
    let glyph_transform = Affine::translate((4.0, 6.0))
        * Affine::skew(0.5, 0.0)
        * Affine::scale_non_uniform(1.25, 0.75);
    for (data, id, x) in [(colr.as_slice(), 4, 8.0), (bitmap.as_slice(), 1, 58.0)] {
        let font = FontData::new(Blob::new(Arc::new(data)), 0);
        scene
            .draw_glyphs(&font)
            .font_size(32.0)
            .transform(Affine::translate((6.0, 8.0)))
            .glyph_transform(Some(glyph_transform))
            .draw(Fill::NonZero, [Glyph { id, x, y: 32.0 }].into_iter());
    }
    smoke_snapshot_test_sync(scene, &TestParams::new("emoji_glyph_transform", 122, 46))
        .unwrap()
        .assert_mean_less_than(0.002);
}

fn glyph_gradient_brush_transform(use_cpu: bool) {
    let mut scene = Scene::new();
    let mut text = SimpleText::new();
    // The gradient starts to the right of the text. Without a brush transform,
    // pad extension clamps the whole run to red; with the transform below, the
    // gradient is translated over the glyphs and becomes visibly red-lime-blue.
    let gradient = Gradient::new_linear((200.0, 0.0), (320.0, 0.0)).with_stops([
        palette::css::RED,
        palette::css::LIME,
        palette::css::BLUE,
    ]);

    text.add_run(
        &mut scene,
        None,
        40.0,
        &gradient,
        Affine::translate((8.0, 38.0)),
        None,
        None,
        Fill::NonZero,
        "GRAD",
    );
    text.add_run(
        &mut scene,
        None,
        40.0,
        &gradient,
        Affine::translate((8.0, 82.0)),
        None,
        Some(Affine::translate((-200.0, 0.0))),
        Fill::NonZero,
        "GRAD",
    );

    let params = TestParams {
        use_cpu,
        ..TestParams::new("glyph_gradient_brush_transform", 150, 92)
    };
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.01);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn filled_square_gpu() {
    filled_square(false);
}

#[test]
// The fine shader still requires a GPU, and so we still get a wgpu device
// skip this for now
#[cfg_attr(skip_gpu_tests, ignore)]
fn filled_square_cpu() {
    filled_square(true);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn filled_circle_gpu() {
    filled_circle(false);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn filled_circle_cpu() {
    filled_circle(true);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn two_emoji_gpu() {
    two_emoji(false);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn two_emoji_cpu() {
    two_emoji(true);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn glyph_gradient_brush_transform_gpu() {
    glyph_gradient_brush_transform(false);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn glyph_gradient_brush_transform_cpu() {
    glyph_gradient_brush_transform(true);
}
