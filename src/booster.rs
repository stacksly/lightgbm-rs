use {
	libc::{c_char, c_double, c_longlong, c_void},
	std::{
		self,
		ffi::{CStr, CString},
	},
};

use serde_json::Value;

use lightgbm_sys;

use crate::{Dataset, Error, Result};

const DEFAULT_MAX_FEATURE_NAME_SIZE: u64 = 64;

/// Core model in LightGBM, containing functions for training, evaluating and predicting.
pub struct Booster {
	handle: lightgbm_sys::BoosterHandle,
	param_overrides: CString,
}

struct FeatureNames {
	features: Vec<Vec<u8>>,
	actual_feature_name_len: u64,
	num_feature_names: i32,
}

impl Booster {
	fn new(handle: lightgbm_sys::BoosterHandle, param_overrides: CString) -> Self {
		Booster {
			handle,
			param_overrides,
		}
	}

	/// Init from model file.
	pub fn from_file_with_param_overrides(filename: &str, param_overrides: &str) -> Result<Self> {
		let filename_str = CString::new(filename)
			.map_err(|e| Error::new(format!("Failed to create cstring: {}", e)))?;
		let mut out_num_iterations = 0;
		let mut handle = std::ptr::null_mut();

		lgbm_call!(lightgbm_sys::LGBM_BoosterCreateFromModelfile(
			filename_str.as_ptr() as *const c_char,
			&mut out_num_iterations,
			&mut handle
		))?;

		Ok(Booster::new(
			handle,
			CString::new(param_overrides).map_err(|e| {
				Error::new(format!("Failed to convert param_overrides to CString: {e}"))
			})?,
		))
	}

	pub fn from_file(filename: &str) -> Result<Self> {
		Self::from_file_with_param_overrides(filename, "")
	}

	pub fn from_bytes_with_param_overrides(bytes: &[u8], param_overrides: &str) -> Result<Self> {
		let str_bytes = CString::new(bytes)
			.map_err(|e| Error::new(format!("Failed to create cstring: {}", e)))?;
		let mut out_num_iterations = 0;
		let mut handle = std::ptr::null_mut();

		lgbm_call!(lightgbm_sys::LGBM_BoosterLoadModelFromString(
			str_bytes.as_ptr() as *const c_char,
			&mut out_num_iterations,
			&mut handle
		))?;

		Ok(Booster::new(
			handle,
			CString::new(param_overrides).map_err(|e| {
				Error::new(format!("Failed to convert param_overrides to CString: {e}"))
			})?,
		))
	}

	pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
		Self::from_bytes_with_param_overrides(bytes, "")
	}

	/// Create a new Booster model with given Dataset and parameters.
	///
	/// Example
	/// ```
	/// extern crate serde_json;
	/// use {
	/// 	lightgbm::{Booster, Dataset},
	/// 	serde_json::json,
	/// };
	///
	/// let data = vec![
	/// 	vec![1.0, 0.1, 0.2, 0.1],
	/// 	vec![0.7, 0.4, 0.5, 0.1],
	/// 	vec![0.9, 0.8, 0.5, 0.1],
	/// 	vec![0.2, 0.2, 0.8, 0.7],
	/// 	vec![0.1, 0.7, 1.0, 0.9],
	/// ];
	/// let label = vec![0.0, 0.0, 0.0, 1.0, 1.0];
	/// let dataset = Dataset::from_mat(data, label).unwrap();
	/// let params = json! {
	///    {
	/// 		"num_iterations": 3,
	/// 		"objective": "binary",
	/// 		"metric": "auc"
	/// 	}
	/// };
	/// let bst = Booster::train(dataset, &params).unwrap();
	/// ```
	pub fn train(dataset: Dataset, parameter: &Value) -> Result<Self> {
		// get num_iterations
		let num_iterations: i64 = if parameter["num_iterations"].is_null() {
			100
		} else {
			parameter["num_iterations"]
				.as_i64()
				.ok_or_else(|| Error::new("failed to unwrap num_iterations"))?
		};

		// exchange params {"x": "y", "z": 1} => "x=y z=1"
		let params_string = parameter
			.as_object()
			.ok_or_else(|| Error::new("failed to convert param to object"))?
			.iter()
			.map(|(k, v)| format!("{}={}", k, v))
			.collect::<Vec<_>>()
			.join(" ");
		let params_cstring = CString::new(params_string)
			.map_err(|e| Error::from_other("failed to make cstring", e))?;

		let mut handle = std::ptr::null_mut();
		lgbm_call!(lightgbm_sys::LGBM_BoosterCreate(
			dataset.handle,
			params_cstring.as_ptr() as *const c_char,
			&mut handle
		))?;

		let mut is_finished: i32 = 0;
		for _ in 1..num_iterations {
			lgbm_call!(lightgbm_sys::LGBM_BoosterUpdateOneIter(
				handle,
				&mut is_finished
			))?;
		}
		Ok(Booster::new(
			handle,
			CString::new("").map_err(|e| Error::new(format!("Failed to allocate CString: {e}")))?,
		))
	}

	/// Predict results for given data.
	///
	/// Input data example
	/// ```
	/// let data = vec![
	/// 	vec![1.0, 0.1, 0.2],
	/// 	vec![0.7, 0.4, 0.5],
	/// 	vec![0.1, 0.7, 1.0],
	/// ];
	/// ```
	///
	/// Output data example
	/// ```
	/// let output = vec![vec![1.0, 0.109, 0.433]];
	/// ```
	pub fn predict(&self, data: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>> {
		let data_length = data.len();
		let feature_length = data[0].len();
		let mut out_length: c_longlong = 0;
		let flat_data = data.into_iter().flatten().collect::<Vec<_>>();

		// get num_class
		let mut num_class = 0;
		lgbm_call!(lightgbm_sys::LGBM_BoosterGetNumClasses(
			self.handle,
			&mut num_class
		))?;

		let out_result: Vec<f64> = vec![Default::default(); data_length * num_class as usize];

		lgbm_call!(lightgbm_sys::LGBM_BoosterPredictForMat(
			self.handle,
			flat_data.as_ptr() as *const c_void,
			lightgbm_sys::C_API_DTYPE_FLOAT64 as i32,
			data_length as i32,                        // nrow
			feature_length as i32,                     // ncol
			1_i32,                                     // is_row_major
			lightgbm_sys::C_API_PREDICT_NORMAL as i32, // predict_type
			0_i32,                                     // start_iteration
			-1_i32,                                    // num_iteration
			self.param_overrides.as_ptr() as *const c_char,
			&mut out_length,
			out_result.as_ptr() as *mut c_double
		))?;

		// reshape for multiclass [1,2,3,4,5,6] -> [[1,2,3], [4,5,6]]  # 3 class
		let reshaped_output = if num_class > 1 {
			out_result
				.chunks(num_class as usize)
				.map(|x| x.to_vec())
				.collect()
		} else {
			vec![out_result]
		};
		Ok(reshaped_output)
	}

	pub fn predict_single_row(&self, data: Vec<f64>) -> Result<Vec<f64>> {
		// get num_class
		let mut num_class = 0;
		lgbm_call!(lightgbm_sys::LGBM_BoosterGetNumClasses(
			self.handle,
			&mut num_class
		))?;

		let mut out_length: c_longlong = 0;
		let out_result: Vec<f64> = vec![Default::default(); num_class as usize];

		lgbm_call!(lightgbm_sys::LGBM_BoosterPredictForMatSingleRow(
			self.handle,
			data.as_ptr() as *const c_void,
			lightgbm_sys::C_API_DTYPE_FLOAT64 as i32,
			data.len() as i32,
			1_i32, // is_row_major
			lightgbm_sys::C_API_PREDICT_NORMAL as i32,
			0_i32,  // start_iteration
			-1_i32, // num_iteration,
			self.param_overrides.as_ptr() as *const c_char,
			&mut out_length,
			out_result.as_ptr() as *mut c_double,
		))?;

		Ok(out_result)
	}

	/// Get Feature Num.
	pub fn num_feature(&self) -> Result<i32> {
		let mut out_len = 0;
		lgbm_call!(lightgbm_sys::LGBM_BoosterGetNumFeature(
			self.handle,
			&mut out_len
		))?;
		Ok(out_len)
	}

	fn _feature_names(&self, num_features: i32, feature_name_size: u64) -> Result<FeatureNames> {
		let mut features = (0..num_features)
			.map(|_| (0..feature_name_size).map(|_| 0).collect::<Vec<u8>>())
			.collect::<Vec<_>>();

		let out_strs = features
			.iter_mut()
			.map(|v| v.as_mut_ptr())
			.collect::<Vec<_>>();

		let mut num_feature_names = 0;
		let mut actual_feature_name_len = 0;

		lgbm_call!(lightgbm_sys::LGBM_BoosterGetFeatureNames(
			self.handle,
			num_features,
			&mut num_feature_names,
			feature_name_size,
			&mut actual_feature_name_len,
			out_strs.as_ptr() as *mut *mut c_char
		))?;

		Ok(FeatureNames {
			features,
			actual_feature_name_len,
			num_feature_names,
		})
	}

	/// Get Feature Names.
	pub fn feature_names(&self) -> Result<Vec<String>> {
		let num_features = self.num_feature()?;

		let mut feature_result =
			self._feature_names(num_features, DEFAULT_MAX_FEATURE_NAME_SIZE)?;

		// If the feature name size was larger than the default max, try again with the actual size
		if feature_result.actual_feature_name_len > DEFAULT_MAX_FEATURE_NAME_SIZE {
			feature_result =
				self._feature_names(num_features, feature_result.actual_feature_name_len)?;
		}

		Ok(feature_result
			.features
			.into_iter()
			.take(feature_result.num_feature_names as usize)
			.map(|s| unsafe {
				CStr::from_ptr(s.as_ptr() as *const i8)
					.to_string_lossy()
					.into()
			})
			.collect())
	}

	// Get Feature Importance
	pub fn feature_importance(&self) -> Result<Vec<f64>> {
		let num_feature = self.num_feature()?;
		let out_result: Vec<f64> = vec![Default::default(); num_feature as usize];
		lgbm_call!(lightgbm_sys::LGBM_BoosterFeatureImportance(
			self.handle,
			0_i32,
			0_i32,
			out_result.as_ptr() as *mut c_double
		))?;
		Ok(out_result)
	}

	/// Save model to file.
	pub fn save_file(&self, filename: &str) -> Result<()> {
		let filename_str =
			CString::new(filename).map_err(|e| Error::from_other("failed to create cstring", e))?;
		lgbm_call!(lightgbm_sys::LGBM_BoosterSaveModel(
			self.handle,
			0_i32,
			-1_i32,
			0_i32,
			filename_str.as_ptr() as *const c_char
		))?;
		Ok(())
	}
}

// LGBM_BoosterPredictForMat is always thread-safe
// https://github.com/Microsoft/LightGBM/issues/666#issuecomment-312254519
unsafe impl Send for Booster {}
unsafe impl Sync for Booster {}

impl Drop for Booster {
	fn drop(&mut self) {
		lgbm_call!(lightgbm_sys::LGBM_BoosterFree(self.handle))
			.expect("Calling LGBM_BoosterFree should always succeed");
	}
}

#[cfg(test)]
mod tests {
	use {
		super::*,
		serde_json::json,
		std::{fs, path::Path},
	};

	fn _read_train_file() -> Result<Dataset> {
		Dataset::from_file("lightgbm-sys/lightgbm/examples/binary_classification/binary.train")
	}

	fn _train_booster(params: &Value) -> Booster {
		let dataset = _read_train_file().unwrap();
		Booster::train(dataset, params).unwrap()
	}

	fn _default_params() -> Value {
		let params = json! {
			{
				"num_iterations": 1,
				"objective": "binary",
				"metric": "auc",
				"data_random_seed": 0
			}
		};
		params
	}

	#[test]
	fn predict() {
		let params = json! {
			{
				"num_iterations": 10,
				"objective": "binary",
				"metric": "auc",
				"data_random_seed": 0
			}
		};
		let bst = _train_booster(&params);
		let feature = vec![vec![0.5; 28], vec![0.0; 28], vec![0.9; 28]];
		let result = bst.predict(feature).unwrap();
		let mut normalized_result: Vec<i32> = Vec::new();
		for r in &result[0] {
			normalized_result.push((*r > 0.5).into());
		}
		assert_eq!(normalized_result, vec![0, 0, 1]);
	}

	#[test]
	fn predict_single_row() {
		let params = json! {
			{
				"num_iterations": 10,
				"objective": "binary",
				"metric": "auc",
				"data_random_seed": 0
			}
		};
		let bst = _train_booster(&params);
		let feature = vec![vec![0.5; 28], vec![0.0; 28], vec![0.9; 28]];

		let result: Vec<f64> = feature
			.iter()
			.flat_map(|f| bst.predict_single_row(f.clone()).unwrap())
			.collect();

		let mut normalized_result: Vec<i32> = Vec::new();
		for r in result {
			normalized_result.push((r > 0.5).into());
		}
		assert_eq!(normalized_result, vec![0, 0, 1]);
	}

	#[test]
	fn num_feature() {
		let params = _default_params();
		let bst = _train_booster(&params);
		let num_feature = bst.num_feature().unwrap();
		assert_eq!(num_feature, 28);
	}

	#[test]
	fn feature_importance() {
		let params = _default_params();
		let bst = _train_booster(&params);
		let feature_importance = bst.feature_importance().unwrap();
		assert_eq!(feature_importance, vec![0.0; 28]);
	}

	#[test]
	fn feature_name() {
		let params = _default_params();
		let bst = _train_booster(&params);
		let feature_name = bst.feature_names().unwrap();
		let target = (0..28).map(|i| format!("Column_{}", i)).collect::<Vec<_>>();
		assert_eq!(feature_name, target);
	}

	#[test]
	fn save_file() {
		let params = _default_params();
		let bst = _train_booster(&params);
		assert_eq!(bst.save_file("./test/test_save_file.output"), Ok(()));
		assert!(Path::new("./test/test_save_file.output").exists());
		let _ = fs::remove_file("./test/test_save_file.output");
	}

	#[test]
	fn from_file() {
		let _ = Booster::from_file("./test/test_from_file.input").unwrap();
	}

	#[test]
	fn from_bytes() {
		let file = fs::read_to_string("./test/test_from_file.input").unwrap();
		let _ = Booster::from_bytes(file.as_bytes()).unwrap();
	}
}
