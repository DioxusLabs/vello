// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::sync::Arc;
use alloc::vec::Vec;

use super::gaussian_blur::BlurBuffers;
use vello_common::fearless_simd::Level;
use vello_common::pixmap::Pixmap;

#[derive(Debug, Default)]
pub(crate) struct FilterContext {
    /// The rendered pixmaps for each filter layer.
    layers: Vec<Option<Arc<Pixmap>>>,
    scratch: ScratchBuffer,
}

impl FilterContext {
    pub(crate) fn new(num_layers: usize, level: Level) -> Self {
        Self {
            layers: (0..num_layers).map(|_| None).collect(),
            scratch: ScratchBuffer::new(level),
        }
    }

    pub(crate) fn scratch(&mut self) -> &mut ScratchBuffer {
        &mut self.scratch
    }

    pub(crate) fn set_layer(&mut self, id: usize, pixmap: Pixmap) {
        if id >= self.layers.len() {
            self.layers.resize_with(id + 1, || None);
        }
        self.layers[id] = Some(Arc::new(pixmap));
    }

    pub(crate) fn filter_layer(&self, id: usize) -> Option<Arc<Pixmap>> {
        self.layers.get(id).and_then(Option::as_ref).cloned()
    }
}

#[derive(Debug, Default)]
pub(crate) struct ScratchBuffer {
    scratch_buffer: Option<Pixmap>,
    blur_buffers: BlurBuffers,
}

impl ScratchBuffer {
    pub(crate) fn new(level: Level) -> Self {
        Self {
            scratch_buffer: None,
            blur_buffers: BlurBuffers::new(level),
        }
    }

    pub(crate) fn get_scratch_buffer(&mut self, width: u16, height: u16) -> &mut Pixmap {
        match &mut self.scratch_buffer {
            None => {
                self.scratch_buffer = Some(Pixmap::new(width, height));
            }
            Some(buf) if buf.width() < width || buf.height() < height => {
                buf.resize(width, height);
            }
            Some(_) => {}
        }

        self.scratch_buffer.as_mut().unwrap()
    }

    /// Like [`Self::get_scratch_buffer`], but also returns the reusable buffers for blurring.
    pub(crate) fn get_blur_scratch(
        &mut self,
        width: u16,
        height: u16,
    ) -> (&mut Pixmap, &mut BlurBuffers) {
        self.get_scratch_buffer(width, height);
        (
            self.scratch_buffer.as_mut().unwrap(),
            &mut self.blur_buffers,
        )
    }
}
