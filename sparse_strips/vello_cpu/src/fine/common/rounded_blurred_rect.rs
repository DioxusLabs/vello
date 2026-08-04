// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Drawing blurred, rounded rectangles.
//!
//! Implementation is adapted from: <https://git.sr.ht/~raph/blurrr/tree/master/src/distfield.rs>.

use crate::fine::{NumericVec, PosExt, ShaderResultF32};
use crate::kurbo::{Point, Vec2};
use vello_common::encode::EncodedBlurredRoundedRectangle;
use vello_common::fearless_simd::{Simd, SimdBase, SimdFloat, f32x8, u8x16};

#[cfg(not(feature = "std"))]
use vello_common::kurbo::common::FloatFuncs as _;

#[derive(Debug)]
pub(crate) struct BlurredRoundedRectFiller<S: Simd> {
    r: f32x8<S>,
    g: f32x8<S>,
    b: f32x8<S>,
    a: f32x8<S>,
    invert: bool,
    alpha_calculator: AlphaCalculator<S>,
}

impl<S: Simd> BlurredRoundedRectFiller<S> {
    pub(crate) fn new(
        simd: S,
        rect: &EncodedBlurredRoundedRectangle,
        start_x: f64,
        start_y: f64,
    ) -> Self {
        simd.vectorize(
            #[inline(always)]
            || {
                let start_pos = rect.transform * Point::new(start_x, start_y);
                let color_components = rect.color.as_premul_f32().components;
                let r = f32x8::splat(simd, color_components[0]);
                let g = f32x8::splat(simd, color_components[1]);
                let b = f32x8::splat(simd, color_components[2]);
                let a = f32x8::splat(simd, color_components[3]);
                let simd_rect = SimdRoundedBlurredRect::new(rect, simd);
                let alpha_calculator = AlphaCalculator::new(
                    start_pos,
                    rect.x_advance,
                    rect.y_advance,
                    simd_rect,
                    simd,
                );

                Self {
                    alpha_calculator,
                    r,
                    g,
                    b,
                    a,
                    invert: rect.invert,
                }
            },
        )
    }
}

impl<S: Simd> Iterator for BlurredRoundedRectFiller<S> {
    type Item = ShaderResultF32<S>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let mut next = self.alpha_calculator.next().unwrap();
        if self.invert {
            next = f32x8::splat(next.simd, 1.0) - next;
        }
        let r = self.r * next;
        let g = self.g * next;
        let b = self.b * next;
        let a = self.a * next;

        Some(ShaderResultF32 { r, g, b, a })
    }
}

impl<S: Simd> crate::fine::Painter for BlurredRoundedRectFiller<S> {
    fn paint_u8(mut self, buf: &mut [u8]) {
        self.a.simd.vectorize(
            #[inline(always)]
            || {
                for chunk in buf.chunks_exact_mut(64) {
                    let first = self.next().unwrap();
                    let simd = first.r.simd;
                    let second = self.next().unwrap();

                    let r = u8x16::from_f32(simd, simd.combine_f32x8(first.r, second.r));
                    let g = u8x16::from_f32(simd, simd.combine_f32x8(first.g, second.g));
                    let b = u8x16::from_f32(simd, simd.combine_f32x8(first.b, second.b));
                    let a = u8x16::from_f32(simd, simd.combine_f32x8(first.a, second.a));

                    let combined =
                        simd.combine_u8x32(simd.combine_u8x16(r, g), simd.combine_u8x16(b, a));

                    simd.store_interleaved_128_u8x64(
                        combined,
                        (&mut chunk[..]).try_into().unwrap(),
                    );
                }
            },
        );
    }

    fn paint_f32(mut self, buf: &mut [f32]) {
        self.a.simd.vectorize(
            #[inline(always)]
            || {
                for chunk in buf.chunks_exact_mut(32) {
                    let (c1, c2) = self.next().unwrap().get();
                    c1.simd
                        .store_interleaved_128_f32x16(c1, (&mut chunk[..16]).try_into().unwrap());
                    c2.simd
                        .store_interleaved_128_f32x16(c2, (&mut chunk[16..]).try_into().unwrap());
                }
            },
        );
    }
}

#[derive(Debug)]
struct AlphaCalculator<S: Simd> {
    cur_pos: Point,
    x_advance: Vec2,
    y_advance: Vec2,
    r: SimdRoundedBlurredRect<S>,
    simd: S,
}

impl<S: Simd> AlphaCalculator<S> {
    fn new(
        start_pos: Point,
        x_advance: Vec2,
        y_advance: Vec2,
        r: SimdRoundedBlurredRect<S>,
        simd: S,
    ) -> Self {
        Self {
            cur_pos: start_pos,
            x_advance,
            y_advance,
            r,
            simd,
        }
    }
}

impl<S: Simd> Iterator for AlphaCalculator<S> {
    type Item = f32x8<S>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let i = f32x8::splat_pos(
            self.simd,
            self.cur_pos.x as f32,
            self.x_advance.x as f32,
            self.y_advance.x as f32,
        );
        let j = f32x8::splat_pos(
            self.simd,
            self.cur_pos.y as f32,
            self.x_advance.y as f32,
            self.y_advance.y as f32,
        );
        let r = &self.r;

        let y = j - r.v1 * r.height;
        let x = i - r.v1 * r.width;

        // Select the parameters of the corner in whose quadrant the pixel lies.
        // Per-corner values are stored as [top-left, top-right, bottom-left, bottom-right].
        let simd = self.simd;
        let right = simd.simd_ge_f32x8(x, r.v0);
        let bottom = simd.simd_ge_f32x8(y, r.v0);
        let select_corner = |v: &[f32x8<S>; 4]| {
            simd.select_f32x8(
                bottom,
                simd.select_f32x8(right, v[3], v[2]),
                simd.select_f32x8(right, v[1], v[0]),
            )
        };
        let r1_x = select_corner(&r.r1_x);
        let r1_y = select_corner(&r.r1_y);
        let w_x = select_corner(&r.w_x);
        let w_y = select_corner(&r.w_y);
        let exponent = select_corner(&r.exponent);
        let recip_exponent = select_corner(&r.recip_exponent);

        // Equivalent to r1_y + y.abs() - (r.h * r.v1)
        let y0 = r1_y - r.h.mul_sub(r.v1, y.abs());
        let y1 = y0.max(r.v0);

        // Equivalent to r1_x + x.abs() - (r.w * r.v1)
        let x0 = r1_x - r.w.mul_sub(r.v1, x.abs());
        let x1 = x0.max(r.v0);
        let a = x1.powf(exponent);
        let b = y1.powf(exponent);
        let d_pos = (a + b).powf(recip_exponent);
        // The direction-dependent effective corner radius: the Euclidean-style radius of
        // the superellipse in the direction of the current point, which reduces to the
        // corner radius itself for circular corners. The epsilon terms keep the ratio
        // well-defined when both `a` and `b` are zero.
        let eps = f32x8::splat(simd, 1e-18);
        let r_eff_num = a + b + eps;
        let r_eff_den = a.mul_add(w_x, b.mul_add(w_y, eps * (r.v1 * (w_x + w_y))));
        let r_eff = simd.select_f32x8(
            simd.simd_eq_f32x8(r1_x, r1_y),
            r1_x,
            simd.div_f32x8(r_eff_num, r_eff_den).powf(recip_exponent),
        );
        let d_neg = x0.max(y0).min(r.v0);
        let d = d_pos + d_neg - r_eff;
        let z = r.scale
            * (f32x8::compute_erf7(self.simd, r.std_dev_inv * (r.min_edge + d))
                - f32x8::compute_erf7(self.simd, r.std_dev_inv * d));

        self.cur_pos += 2.0 * self.x_advance;

        Some(z)
    }
}

#[derive(Debug)]
struct SimdRoundedBlurredRect<S: Simd> {
    pub exponent: [f32x8<S>; 4],
    pub recip_exponent: [f32x8<S>; 4],
    pub scale: f32x8<S>,
    pub std_dev_inv: f32x8<S>,
    pub min_edge: f32x8<S>,
    pub w: f32x8<S>,
    pub h: f32x8<S>,
    pub width: f32x8<S>,
    pub height: f32x8<S>,
    pub r1_x: [f32x8<S>; 4],
    pub r1_y: [f32x8<S>; 4],
    pub w_x: [f32x8<S>; 4],
    pub w_y: [f32x8<S>; 4],
    pub v0: f32x8<S>,
    pub v1: f32x8<S>,
}

impl<S: Simd> SimdRoundedBlurredRect<S> {
    fn new(encoded: &EncodedBlurredRoundedRectangle, s: S) -> Self {
        s.vectorize(
            #[inline(always)]
            || {
                let h = f32x8::splat(s, encoded.h);
                let w = f32x8::splat(s, encoded.w);
                let width = f32x8::splat(s, encoded.width);
                let height = f32x8::splat(s, encoded.height);
                let r1_x = encoded.r1_x.map(|v| f32x8::splat(s, v));
                let r1_y = encoded.r1_y.map(|v| f32x8::splat(s, v));
                let w_x = encoded.w_x.map(|v| f32x8::splat(s, v));
                let w_y = encoded.w_y.map(|v| f32x8::splat(s, v));
                let exponent = encoded.exponent.map(|v| f32x8::splat(s, v));
                let recip_exponent = encoded.recip_exponent.map(|v| f32x8::splat(s, v));
                let scale = f32x8::splat(s, encoded.scale);
                let min_edge = f32x8::splat(s, encoded.min_edge);
                let std_dev_inv = f32x8::splat(s, encoded.std_dev_inv);
                let v0 = f32x8::splat(s, 0.0);
                let v1 = f32x8::splat(s, 0.5);

                Self {
                    exponent,
                    recip_exponent,
                    scale,
                    std_dev_inv,
                    min_edge,
                    w,
                    v0,
                    v1,
                    h,
                    width,
                    height,
                    r1_x,
                    r1_y,
                    w_x,
                    w_y,
                }
            },
        )
    }
}

trait FloatExt<S: Simd> {
    // See https://raphlinus.github.io/audio/2018/09/05/sigmoid.html for a little
    // explanation of this approximation to the erf function.
    /// Approximate the erf function.
    fn compute_erf7(simd: S, x: Self) -> Self;
    fn powf(self, x: Self) -> Self;
}

impl<S: Simd> FloatExt<S> for f32x8<S> {
    #[inline(always)]
    fn compute_erf7(simd: S, x: Self) -> Self {
        // Clamp `x`, because for large `x` the terms here become `inf`, causing the result to be 0 or
        // `NaN`. This clamping doesn't lose any information, because `erf(±10) ≈ 1` well within `f64`
        // machine precision, let alone `f32`.
        let x = x.max(Self::splat(simd, -10.0)).min(Self::splat(simd, 10.0));
        let x = x * Self::splat(simd, core::f32::consts::FRAC_2_SQRT_PI);
        let xx = x * x;
        let p1 = Self::splat(simd, 0.0104).mul_add(xx, Self::splat(simd, 0.03395));
        let p2 = p1.mul_add(xx, Self::splat(simd, 0.24295));
        let p3 = x * xx;
        let x = p2.mul_add(p3, x);
        let denom = x.mul_add(x, Self::splat(simd, 1.0)).sqrt();
        x / denom
    }

    #[inline]
    fn powf(mut self, x: Self) -> Self {
        // TODO: SIMD
        self[0] = self[0].powf(x[0]);
        self[1] = self[1].powf(x[1]);
        self[2] = self[2].powf(x[2]);
        self[3] = self[3].powf(x[3]);
        self[4] = self[4].powf(x[4]);
        self[5] = self[5].powf(x[5]);
        self[6] = self[6].powf(x[6]);
        self[7] = self[7].powf(x[7]);

        self
    }
}
