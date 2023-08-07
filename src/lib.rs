#[cfg(feature = "dataframe")]
extern crate polars;

macro_rules! lgbm_call {
	($x:expr) => {
		Error::check_return_value(unsafe { $x })
	};
}

mod error;
pub use error::{Error, Result};

mod dataset;
pub use dataset::Dataset;

mod booster;
pub use booster::Booster;

mod single_row_predictor;
pub use single_row_predictor::SingleRowPredictor;
