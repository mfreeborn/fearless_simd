// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![expect(
    clippy::missing_safety_doc,
    reason = "TODO: https://github.com/linebender/fearless_simd/issues/40"
)]

//! Access to AVX512 intrinsics.

use crate::impl_macros::delegate;
#[cfg(target_arch = "x86")]
use core::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as arch;

use arch::*;

/// A token for AVX512 intrinsics on `x86` and `x86_64`.
#[derive(Clone, Copy, Debug)]
pub struct Avx512 {
    _private: (),
}

impl Avx512 {
    /// Create a SIMD token.
    ///
    /// # Safety
    ///
    /// The required CPU features must be available.
    pub const unsafe fn new_unchecked() -> Self {
        Self { _private: () }
    }
}
