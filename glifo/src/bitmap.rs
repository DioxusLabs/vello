// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A minimal, renderer-neutral bitmap representation for bitmap (emoji) glyphs.

use alloc::sync::Arc;
use alloc::vec::Vec;
use peniko::ImageQuality;

/// A bitmap glyph image in premultiplied RGBA8 format.
///
/// This is Glifo's renderer-neutral pixel container. Renderer backends convert
/// this into their own image representation when drawing uncached bitmap
/// glyphs or uploading them to an atlas.
#[derive(Clone, Debug)]
pub struct GlyphPixmap {
    width: u16,
    height: u16,
    /// Premultiplied RGBA8 pixel data in row-major order (`width * height * 4` bytes).
    data: Vec<u8>,
}

impl GlyphPixmap {
    /// Create a pixmap from premultiplied RGBA8 bytes in row-major order.
    ///
    /// Panics if `data` is not exactly `width * height * 4` bytes long.
    pub fn from_parts(data: Vec<u8>, width: u16, height: u16) -> Self {
        assert_eq!(
            data.len(),
            usize::from(width) * usize::from(height) * 4,
            "GlyphPixmap data must be width * height * 4 bytes"
        );
        Self {
            width,
            height,
            data,
        }
    }

    /// Return the width of the pixmap.
    pub fn width(&self) -> u16 {
        self.width
    }

    /// Return the height of the pixmap.
    pub fn height(&self) -> u16 {
        self.height
    }

    /// Return the premultiplied RGBA8 pixel data in row-major order.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Consume the pixmap, returning the premultiplied RGBA8 pixel data.
    pub fn into_data(self) -> Vec<u8> {
        self.data
    }

    /// Create a pixmap by decoding a PNG (used for CBDT/sbix bitmap glyphs).
    #[cfg(feature = "png")]
    pub fn from_png(
        data: impl std::io::BufRead + std::io::Seek,
    ) -> Result<Self, png::DecodingError> {
        let mut decoder = png::Decoder::new(data);
        decoder.set_transformations(
            png::Transformations::normalize_to_color8() | png::Transformations::ALPHA,
        );

        let mut reader = decoder.read_info()?;
        let (width, height) = {
            let info = reader.info();
            let width: u16 = info
                .width
                .try_into()
                .map_err(|_| png::DecodingError::LimitsExceeded)?;
            let height: u16 = info
                .height
                .try_into()
                .map_err(|_| png::DecodingError::LimitsExceeded)?;
            (width, height)
        };
        let mut buf = alloc::vec![0_u8; usize::from(width) * usize::from(height) * 4];

        let (color_type, bit_depth) = reader.output_color_type();
        debug_assert_eq!(
            bit_depth,
            png::BitDepth::Eight,
            "normalize_to_color8 means the bit depth is always 8."
        );

        match color_type {
            png::ColorType::Rgb | png::ColorType::Grayscale => {
                unreachable!("We set a transformation to always convert to alpha")
            }
            png::ColorType::Indexed => {
                unreachable!("Transformation should have expanded indexed images")
            }
            png::ColorType::Rgba => {
                debug_assert_eq!(
                    Some(buf.len()),
                    reader.output_buffer_size(),
                    "The pixel buffer should have the same number of bytes as the image."
                );
                reader.next_frame(&mut buf)?;
            }
            png::ColorType::GrayscaleAlpha => {
                let mut grayscale_data =
                    alloc::vec![0; reader.output_buffer_size().unwrap_or_default()];
                reader.next_frame(&mut grayscale_data)?;

                for (grayscale_pixel, pixel) in
                    grayscale_data.chunks_exact(2).zip(buf.chunks_exact_mut(4))
                {
                    let [gray, alpha] = grayscale_pixel.try_into().unwrap();
                    pixel[0] = gray;
                    pixel[1] = gray;
                    pixel[2] = gray;
                    pixel[3] = alpha;
                }
            }
        };

        for pixel in buf.chunks_exact_mut(4) {
            let alpha = u16::from(pixel[3]);
            #[expect(
                clippy::cast_possible_truncation,
                reason = "Overflow should be impossible."
            )]
            let premultiply = |e: u8| ((u16::from(e) * alpha) / 255) as u8;
            pixel[0] = premultiply(pixel[0]);
            pixel[1] = premultiply(pixel[1]);
            pixel[2] = premultiply(pixel[2]);
        }

        Ok(Self {
            width,
            height,
            data: buf,
        })
    }
}

/// An image paint for an uncached bitmap glyph.
#[derive(Clone, Debug)]
pub struct GlyphImage {
    /// The pixel data of the glyph.
    pub pixmap: Arc<GlyphPixmap>,
    /// The suggested sampling quality, derived from the glyph's transform.
    pub quality: ImageQuality,
}
