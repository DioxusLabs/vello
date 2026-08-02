<div align="center">

# Pixmap

**A simple premultiplied RGBA8 pixmap type**

[![Latest published version.](https://img.shields.io/crates/v/pixmap.svg)](https://crates.io/crates/pixmap)
[![Documentation build status.](https://img.shields.io/docsrs/pixmap.svg)](https://docs.rs/pixmap)
[![Apache 2.0 or MIT license.](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-blue.svg)](#license)
\
[![Linebender Zulip chat.](https://img.shields.io/badge/Linebender-%23vello-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/channel/197075-vello)
[![GitHub Actions CI status.](https://img.shields.io/github/actions/workflow/status/linebender/vello/ci.yml?logo=github&label=CI)](https://github.com/linebender/vello/actions)
[![Dependency staleness status.](https://deps.rs/crate/pixmap/latest/status.svg)](https://deps.rs/crate/pixmap)

</div>

This crate provides `Pixmap`, a simple premultiplied RGBA8 pixel buffer shared across Vello crates (`glifo`, `vello_common`, and the sparse strips renderers).

## Usage

This crate should not be used on its own, and you should instead use one of the renderers which use it.

## Features

- `std` (enabled by default): Get floating point functions from the standard library
  (likely using your target's libc).
- `png` (enabled by default): Allow loading `Pixmap`s from PNG images.
  Implies `std`.

## Minimum supported Rust Version (MSRV)

This version of Pixmap has been verified to compile with **Rust 1.88** and later.

Future versions of Pixmap might increase the Rust version requirement.
It will not be treated as a breaking change and as such can even happen with small patch releases.

## Community

Discussion of Pixmap development happens in the [Linebender Zulip](https://xi.zulipchat.com/), specifically the [#vello channel](https://xi.zulipchat.com/#narrow/channel/197075-vello).
All public content can be read without logging in.

Contributions are welcome by pull request.
The [Rust code of conduct] applies.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

[Rust code of conduct]: https://www.rust-lang.org/policies/code-of-conduct
