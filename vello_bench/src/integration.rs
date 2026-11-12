// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Integration benchmarks for full rendering pipelines.

use std::sync::Arc;

use criterion::Criterion;
use smallvec::smallvec;
use vello_common::color::DynamicColor;
use vello_common::color::palette::css::{BLACK, WHITE};
use vello_common::color::{AlphaColor, Srgb};
use vello_common::kurbo::{Affine, Point, Rect, Shape};
use vello_common::mask::Mask;
use vello_common::paint::{Image, ImageSource};
use vello_common::peniko::{BlendMode, Compose, ImageSampler, Mix};
use vello_common::peniko::{ColorStop, ColorStops, Gradient, LinearGradientPosition};
use vello_common::peniko::{Extend, ImageAlphaType, ImageQuality};
use vello_common::pixmap::{PixelMetadata, Pixmap};
use vello_cpu::{RenderContext, Resources};

/// Image scene rendering benchmark.
pub fn images(c: &mut Criterion) {
    if !crate::EXTENDED {
        return;
    }

    let mut g = c.benchmark_group("images");

    let flower_image = load_flower_image();

    const VIEWPORT_WIDTH: u16 = 1280;
    const VIEWPORT_HEIGHT: u16 = 960;

    let ImageSource::Pixmap(ref image_pixmap) = flower_image else {
        panic!("Expected Pixmap");
    };
    let original_width = f64::from(image_pixmap.width());
    let original_height = f64::from(image_pixmap.height());
    let image_count = VIEWPORT_WIDTH / 256;

    g.bench_function("overlapping", |b| {
        let mut renderer = RenderContext::new(VIEWPORT_WIDTH, VIEWPORT_HEIGHT);
        let mut resources = Resources::default();
        let mut pixmap = Pixmap::new(VIEWPORT_WIDTH, VIEWPORT_HEIGHT);

        b.iter(|| {
            renderer.reset();

            for i in (1..=image_count).rev() {
                let width = 256.0 * i as f64;
                let scale = width / original_width;
                let height = original_height * scale;

                renderer.set_paint_transform(Affine::scale(scale));
                renderer.set_paint(Image {
                    image: flower_image.clone(),
                    sampler: ImageSampler {
                        x_extend: Extend::Pad,
                        y_extend: Extend::Pad,
                        quality: ImageQuality::Medium,
                        alpha: 1.0,
                    },
                });
                renderer.fill_rect(&Rect::new(0.0, 0.0, width, height));
            }

            renderer.flush();
            renderer.render(&mut pixmap, &mut resources);
            std::hint::black_box(&pixmap);
        });
    });

    g.finish();
}

#[derive(Clone, Copy)]
enum MaskKind {
    None,
    Alpha,
    Luminance,
    /// The approach of rendering each luminance mask into a full-size pixmap and converting it
    /// with [`Mask::new_luminance`].
    FullSizeLuminance,
}

/// Many small luminance-masked shapes on a large canvas, compared with alpha masks and with
/// pre-rasterizing full-size luminance masks.
pub fn masks(c: &mut Criterion) {
    const WIDTH: u16 = 1600;
    const HEIGHT: u16 = 1200;
    const COUNT: usize = 100;
    const SIZE: f64 = 40.0;

    let mut g = c.benchmark_group("masks");

    let draw = |ctx: &mut RenderContext, kind: MaskKind| {
        let dest_in = BlendMode::new(Mix::Normal, Compose::DestIn);
        ctx.set_paint(AlphaColor::<Srgb>::from_rgb8(240, 240, 240));
        ctx.fill_rect(&Rect::new(0.0, 0.0, f64::from(WIDTH), f64::from(HEIGHT)));

        for i in 0..COUNT {
            let x = 20.0 + (i % 20) as f64 * 75.0;
            let y = 20.0 + (i / 20) as f64 * 75.0;
            let rect = Rect::new(x, y, x + SIZE, y + SIZE);
            let clip = rect.to_path(0.1);
            let gradient = Gradient {
                kind: LinearGradientPosition {
                    start: Point::new(x, y),
                    end: Point::new(x + SIZE, y),
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
            };

            ctx.push_clip_layer(&clip);
            ctx.set_paint(AlphaColor::<Srgb>::from_rgb8(200, 30, 30));
            ctx.fill_rect(&rect);
            match kind {
                MaskKind::None => {}
                MaskKind::Alpha => {
                    ctx.push_layer(Some(&clip), Some(dest_in), None, None, None);
                    ctx.set_paint(gradient);
                    ctx.fill_rect(&rect);
                    ctx.pop_layer();
                }
                MaskKind::Luminance => {
                    ctx.push_luminance_mask_layer(Some(&clip), 1.0);
                    ctx.set_paint(gradient);
                    ctx.fill_rect(&rect);
                    ctx.pop_layer();
                }
                MaskKind::FullSizeLuminance => {
                    let mut mask_ctx = RenderContext::new(WIDTH, HEIGHT);
                    let mut mask_pixmap = Pixmap::new(WIDTH, HEIGHT);
                    mask_ctx.set_paint(gradient);
                    mask_ctx.fill_rect(&rect);
                    mask_ctx.flush();
                    mask_ctx.render(&mut mask_pixmap, &mut Resources::default());
                    let mask = Mask::new_luminance(&mask_pixmap);
                    ctx.push_layer(None, Some(dest_in), None, Some(mask), None);
                    ctx.set_paint(WHITE);
                    ctx.fill_rect(&rect);
                    ctx.pop_layer();
                }
            }
            ctx.pop_layer();
        }
    };

    for (name, kind) in [
        ("no_mask", MaskKind::None),
        ("alpha", MaskKind::Alpha),
        ("luminance", MaskKind::Luminance),
        ("full_size_luminance", MaskKind::FullSizeLuminance),
    ] {
        g.bench_function(name, |b| {
            let mut renderer = RenderContext::new(WIDTH, HEIGHT);
            let mut resources = Resources::default();
            let mut pixmap = Pixmap::new(WIDTH, HEIGHT);

            b.iter(|| {
                renderer.reset();
                draw(&mut renderer, kind);
                renderer.flush();
                renderer.render(&mut pixmap, &mut resources);
                std::hint::black_box(&pixmap);
            });
        });
    }

    g.finish();
}

fn load_flower_image() -> ImageSource {
    let image_data = include_bytes!("../../assets/splash-flower.jpg");
    let image = image::load_from_memory(image_data).expect("Failed to decode image");
    let width = image.width();
    let height = image.height();
    let rgba_data = image.into_rgba8().into_vec();

    #[expect(
        clippy::cast_possible_truncation,
        reason = "Image dimensions fit in u16"
    )]
    let pixmap = Pixmap::from_parts(
        rgba_data,
        width as u16,
        height as u16,
        PixelMetadata::new(ImageAlphaType::Alpha, true),
    );

    ImageSource::Pixmap(Arc::new(pixmap))
}
