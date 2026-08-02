// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A simple premultiplied RGBA8 pixmap type shared across Vello crates.
//!
//! # Features
//!
//! - `std` (enabled by default): Get floating point functions from the standard library
//!   (likely using your target's libc).
//! - `png` (enabled by default): Allow loading [`Pixmap`]s from PNG images.
//!   Implies `std`.

// LINEBENDER LINT SET - lib.rs - v3
// See https://linebender.org/wiki/canonical-lints/
// These lints shouldn't apply to examples or tests.
#![cfg_attr(not(test), warn(unused_crate_dependencies))]
// These lints shouldn't apply to examples.
#![warn(clippy::print_stdout, clippy::print_stderr)]
// Targeting e.g. 32-bit means structs containing usize can give false positives for 64-bit.
#![cfg_attr(target_pointer_width = "64", warn(clippy::trivially_copy_pass_by_ref))]
// END LINEBENDER LINT SET
#![cfg_attr(docsrs, feature(doc_cfg))]
#![forbid(unsafe_code)]
#![no_std]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

mod pixmap;

pub use pixmap::{Pixmap, PixmapMut};
