use std::marker::PhantomData;

use libc::{c_double, c_longlong, c_void};

use crate::{Booster, Error, Result};

pub struct SingleRowPredictor<'a> {
	pub(crate) handle: lightgbm_sys::FastConfigHandle,
	pub(crate) input_size: usize,
	pub(crate) output_size: usize,
	/// Holds lifetime information with regards to `handle`
	pub(crate) booster: PhantomData<&'a Booster>,
}

unsafe impl Send for SingleRowPredictor<'_> {}
// Although it *technically* is also `Sync`, there's actually much overhead if it's used
// in parallel across threads due to cached resources being behind a mutex, so it is recommended to
// just create one per user thread. Mutex is inexpensive if there is no contention.

impl Drop for SingleRowPredictor<'_> {
	fn drop(&mut self) {
		lgbm_call!(lightgbm_sys::LGBM_FastConfigFree(self.handle))
			.expect("Calling LGBM_FastConfigFree should always succeed");
	}
}

impl SingleRowPredictor<'_> {
	pub fn predict(&self, data: &[f64]) -> Result<Vec<f64>> {
		if data.len() != self.input_size {
			return Err(Error::new(format!(
				"Input data size {} does not match number of features {}",
				data.len(),
				self.input_size
			)));
		}

		let out_result: Vec<f64> = vec![Default::default(); self.output_size];

		let mut out_length: c_longlong = 0;
		lgbm_call!(lightgbm_sys::LGBM_BoosterPredictForMatSingleRowFast(
			self.handle,
			data.as_ptr() as *const c_void,
			&mut out_length,
			out_result.as_ptr() as *mut c_double,
		))?;

		assert!(
			usize::try_from(out_length).is_ok_and(|l| l == out_result.len()),
			"Unexpected written output length"
		);

		Ok(out_result)
	}
}
