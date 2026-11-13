// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Integration benchmarks for full rendering pipelines.

use std::sync::Arc;

use criterion::Criterion;
use vello_common::color::palette::css::{REBECCA_PURPLE, WHITE};
use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_common::kurbo::{Affine, Rect};
use vello_common::paint::{Image, ImageSource};
use vello_common::peniko::ImageSampler;
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

/// Many small gaussian-blurred rectangles on a large canvas.
pub fn filters(c: &mut Criterion) {
    const WIDTH: u16 = 1600;
    const HEIGHT: u16 = 1200;
    const COLUMNS: usize = 20;
    const ROWS: usize = 15;
    const SIZE: f64 = 60.0;

    let mut g = c.benchmark_group("filters");

    let draw = |ctx: &mut RenderContext, std_deviation: Option<f32>| {
        ctx.set_paint(WHITE);
        ctx.fill_rect(&Rect::new(0.0, 0.0, f64::from(WIDTH), f64::from(HEIGHT)));
        ctx.set_paint(REBECCA_PURPLE);

        for i in 0..COLUMNS * ROWS {
            let x = 20.0 + (i % COLUMNS) as f64 * 78.0;
            let y = 20.0 + (i / COLUMNS) as f64 * 78.0;
            if let Some(std_deviation) = std_deviation {
                ctx.push_filter_layer(Filter::from_primitive(FilterPrimitive::GaussianBlur {
                    std_deviation,
                    edge_mode: EdgeMode::None,
                }));
            }
            ctx.fill_rect(&Rect::new(x, y, x + SIZE, y + SIZE));
            if std_deviation.is_some() {
                ctx.pop_layer();
            }
        }
    };

    for (name, std_deviation) in [
        ("no_filter", None),
        ("blur_sigma_2", Some(2.0)),
        ("blur_sigma_4", Some(4.0)),
        ("blur_sigma_8", Some(8.0)),
    ] {
        g.bench_function(name, |b| {
            let mut renderer = RenderContext::new(WIDTH, HEIGHT);
            let mut resources = Resources::default();
            let mut pixmap = Pixmap::new(WIDTH, HEIGHT);

            b.iter(|| {
                renderer.reset();
                draw(&mut renderer, std_deviation);
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
