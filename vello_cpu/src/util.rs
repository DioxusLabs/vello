// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::peniko::ImageQuality;
use vello_common::encode::EncodedImage;
use vello_common::fearless_simd::{f32x4, prelude::*, u8x32};
use vello_common::math::FloatExt;
use vello_common::util::{narrow, normalized_mul_u8};

pub(crate) mod scalar {
    /// Perform an approximate division by 255.
    ///
    /// There are three reasons for having this method.
    /// 1) Divisions are slower than shifting + adding, and the compiler does not seem to replace
    ///    divisions by 255 with an equivalent (this was verified by benchmarking; doing / 255 was
    ///    significantly slower).
    /// 2) Integer divisions are usually not available in SIMD, so this provides a good baseline
    ///    implementation.
    /// 3) There are two options for performing the division: One is to perform the division
    ///    in a way that completely preserves the rounding semantics of a integer division by
    ///    255. This could be achieved using the implementation `(val + 1 + (val >> 8)) >> 8`.
    ///    The second approach (used here) has slightly different rounding behavior to a
    ///    normal division by 255, but is much faster (see <https://github.com/linebender/vello/issues/904>)
    ///    and therefore preferable for the high-performance pipeline.
    ///
    /// Four properties worth mentioning:
    /// - This actually calculates the ceiling of `val / 256`.
    /// - Within the allowed range for `val`, rounding errors do not appear for values divisible by 255, i.e. any call `div_255(val * 255)` will always yield `val`.
    /// - If there is a discrepancy, this division will always yield a value 1 higher than the original.
    /// - This holds for values of `val` up to and including `65279`. You should not call this function with higher values.
    #[inline(always)]
    pub(crate) const fn div_255(val: u16) -> u16 {
        debug_assert!(
            val < 65280,
            "the properties of `div_255` do not hold for values of `65280` or greater"
        );
        (val + 255) >> 8
    }

    #[cfg(test)]
    mod tests {
        use crate::util::scalar::div_255;

        #[test]
        fn div_255_properties() {
            for i in 0_u16..256 * 255 {
                let expected = i / 255;
                let actual = div_255(i);

                assert!(
                    expected <= actual,
                    "In case of a discrepancy, the division should yield a value higher than the original."
                );

                let diff = expected.abs_diff(actual);
                assert!(diff <= 1, "Rounding error shouldn't be higher than 1.");

                if i % 255 == 0 {
                    assert_eq!(diff, 0, "Division should be accurate for multiples of 255.");
                }
            }
        }
    }
}

pub(crate) trait NormalizedMulExt {
    fn normalized_mul(self, other: Self) -> Self;
}

impl<S: Simd> NormalizedMulExt for u8x32<S> {
    #[inline(always)]
    fn normalized_mul(self, other: Self) -> Self {
        narrow(normalized_mul_u8(self, other))
    }
}

pub(crate) trait EncodedImageExt {
    fn has_skew(&self) -> bool;
    fn nearest_neighbor(&self) -> bool;
    fn integer_translation(&self) -> Option<(i32, i32)>;
}

impl EncodedImageExt for EncodedImage {
    fn has_skew(&self) -> bool {
        !(self.x_advance.y as f32).is_nearly_zero() || !(self.y_advance.x as f32).is_nearly_zero()
    }

    fn nearest_neighbor(&self) -> bool {
        self.sampler.quality == ImageQuality::Low
    }

    /// If the image is sampled with nearest-neighbor filtering and its transform is a
    /// pure integer translation, return the offset that maps a pixel position to
    /// the image pixel it samples.
    #[inline]
    fn integer_translation(&self) -> Option<(i32, i32)> {
        // Beyond this magnitude, the `f32` math of the image painters might no longer
        // be exact, so don't bother.
        const MAX_OFFSET: f64 = (1 << 20) as f64;

        // Note that we purposefully avoid `f64::round`, since it is a libm call on
        // some targets and this is called for each command.
        let to_integer = |v: f64| {
            if v.is_nan() || v.abs() >= MAX_OFFSET {
                return None;
            }
            let rounded = if v >= 0.0 { v + 0.5 } else { v - 0.5 } as i32;
            ((v - f64::from(rounded)) as f32)
                .is_nearly_zero()
                .then_some(rounded)
        };

        let [a, b, c, d, e, f] = self.transform.as_coeffs();
        if !self.nearest_neighbor() || a != 1.0 || b != 0.0 || c != 0.0 || d != 1.0 {
            return None;
        }

        Some((to_integer(e)?, to_integer(f)?))
    }
}

pub(crate) trait Premultiply {
    fn premultiply(self, alphas: Self) -> Self;
    fn unpremultiply(self, alphas: Self) -> Self;
}

impl<S: Simd> Premultiply for f32x4<S> {
    #[inline(always)]
    fn premultiply(self, alphas: Self) -> Self {
        self * alphas
    }

    #[inline(always)]
    fn unpremultiply(self, alphas: Self) -> Self {
        let zero = Self::splat(alphas.simd, 0.0);
        let divided = self / alphas;

        self.simd
            .select_f32x4(self.simd.simd_eq_f32x4(alphas, zero), zero, divided)
    }
}
