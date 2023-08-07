#[cfg(feature = "dataframe")]
extern crate polars;

macro_rules! lgbm_call {
	($x:expr) => {
		Error::check_return_value(unsafe { $x })
	};
}

mod error;
pub use crate::error::{Error, Result};

mod dataset;
pub use crate::dataset::Dataset;

mod booster;
pub use crate::booster::Booster;
