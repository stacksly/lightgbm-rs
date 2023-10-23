use {
	libc::{c_char, c_double, c_longlong, c_void},
	std::{
		self,
		ffi::{CStr, CString},
	},
};

use serde_json::Value;

use lightgbm_sys;

use crate::{Dataset, Error, Result, SingleRowPredictor};

/// Core model in LightGBM, containing functions for training, evaluating and predicting.
pub struct Booster {
	handle: lightgbm_sys::BoosterHandle,
	pub(crate) param_overrides: CString,
	/// This is necessary because of https://github.com/microsoft/LightGBM/issues/6142
	race_workaround_mutex: std::sync::Mutex<()>,
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

impl Booster {
	fn new(handle: lightgbm_sys::BoosterHandle, param_overrides: CString) -> Self {
		Booster {
			handle,
			param_overrides,
			race_workaround_mutex: std::sync::Mutex::new(()),
		}
	}

	/// Init from model file.
	pub fn from_file_with_param_overrides(filename: &str, param_overrides: &str) -> Result<Self> {
		let filename_str = CString::new(filename)
			.map_err(|e| Error::new(format!("Failed to create cstring: {e}")))?;

		let param_overrides = CString::new(param_overrides).map_err(|e| {
			Error::new(format!("Failed to convert param_overrides to CString: {e}"))
		})?;

		let mut out_num_iterations = 0;
		let mut handle = std::ptr::null_mut();

		lgbm_call!(lightgbm_sys::LGBM_BoosterCreateFromModelfile(
			filename_str.as_ptr() as *const c_char,
			&mut out_num_iterations,
			&mut handle
		))?;
		// It is very important to create the booster immediately after a successful call to avoid
		// memory leak on subsequent error (as we rely on the drop impl of Booster to be called)
		Ok(Booster::new(handle, param_overrides))
	}

	pub fn from_file(filename: &str) -> Result<Self> {
		Self::from_file_with_param_overrides(filename, "")
	}

	pub fn from_string_with_param_overrides(string: &[u8], param_overrides: &str) -> Result<Self> {
		let model_cstring = CString::new(string)
			.map_err(|e| Error::new(format!("Failed to create cstring: {e}")))?;

		let param_overrides = CString::new(param_overrides).map_err(|e| {
			Error::new(format!("Failed to convert param_overrides to CString: {e}"))
		})?;

		let mut out_num_iterations = 0;
		let mut handle = std::ptr::null_mut();

		lgbm_call!(lightgbm_sys::LGBM_BoosterLoadModelFromString(
			model_cstring.as_ptr() as *const c_char,
			&mut out_num_iterations,
			&mut handle
		))?;
		// It is very important to create the booster immediately after a successful call to avoid
		// memory leak on subsequent error (as we rely on the drop impl of Booster to be called)
		Ok(Booster::new(handle, param_overrides))
	}

	pub fn from_string(bytes: &[u8]) -> Result<Self> {
		Self::from_string_with_param_overrides(bytes, "")
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
	/// let data = &[
	/// 	[1.0, 0.1, 0.2, 0.1],
	/// 	[0.7, 0.4, 0.5, 0.1],
	/// 	[0.9, 0.8, 0.5, 0.1],
	/// 	[0.2, 0.2, 0.8, 0.7],
	/// 	[0.1, 0.7, 1.0, 0.9],
	/// ];
	/// let label = &[0.0, 0.0, 0.0, 1.0, 1.0];
	/// let dataset = Dataset::from_mat(
	/// 	&data.iter().flatten().copied().collect::<Vec<_>>(),
	/// 	data.len(),
	/// 	label,
	/// )
	/// .unwrap();
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
		let mut params_string = String::new();
		for (k, v) in parameter
			.as_object()
			.ok_or_else(|| Error::new("failed to convert param to object"))?
			.iter()
		{
			if !params_string.is_empty() {
				params_string.push(' ');
			}
			params_string.push_str(&format!("{k}={v}"));
		}
		let params_cstring = CString::new(params_string)
			.map_err(|e| Error::from_other("failed to make cstring", e))?;

		let mut handle = std::ptr::null_mut();
		lgbm_call!(lightgbm_sys::LGBM_BoosterCreate(
			dataset.handle,
			params_cstring.as_ptr() as *const c_char,
			&mut handle
		))?;
		// It is very important to create the booster immediately after a successful call to avoid
		// memory leak on subsequent error (as we rely on the drop impl of Booster to be called)
		let booster = Booster::new(
			handle,
			CString::new("").map_err(|e| Error::new(format!("Failed to allocate CString: {e}")))?,
		);

		let mut is_finished: i32 = 0;
		for _ in 1..num_iterations {
			lgbm_call!(lightgbm_sys::LGBM_BoosterUpdateOneIter(
				handle,
				&mut is_finished
			))?;
		}

		Ok(booster)
	}

	/// Predict results for given data.
	///
	/// Input data example
	///
	/// ```
	/// let data = [[1.0, 0.1], [0.7, 0.4], [0.1, 0.7]]
	/// 	.into_iter()
	/// 	.flatten()
	/// 	.collect::<Vec<_>>();
	/// let n_rows = 3;
	/// ```
	///
	/// Output data example
	/// ```
	/// let output = vec![1.0, 0.109, 0.433];
	/// ```
	///
	/// There is one entry per class for each line in the output vector.
	/// `output.chunks(output.len() / n_rows)` gives the output for each line.
	///
	/// This takes &mut because there's currently a data race in LightGBM:
	/// [#6142](https://github.com/microsoft/LightGBM/issues/6142)
	pub fn predict(&mut self, data: &[f64]) -> Result<Vec<f64>> {
		if data.is_empty() {
			return Ok(Vec::new());
		}
		let num_feature: i32 = self.num_feature()?;
		let n_features: usize = num_feature
			.try_into()
			.map_err(|_| Error::new("number of features doesn't fit into an usize"))?;
		if data.len() % n_features != 0 {
			return Err(Error::new(format!(
				"data len is not a multiple of n_features ({n_features}), \
					but all rows should have the same length",
			)));
		}
		let n_rows = data.len() / n_features;
		let nrow = n_rows
			.try_into()
			.map_err(|_| Error::new("number of rows doesn't fit into an i32"))?;

		let predict_output_len = self.predict_output_len(nrow)?;
		let out_result: Vec<f64> = vec![Default::default(); predict_output_len];

		let mut out_length: c_longlong = 0;
		lgbm_call!(lightgbm_sys::LGBM_BoosterPredictForMat(
			self.handle,
			data.as_ptr() as *const c_void,
			lightgbm_sys::C_API_DTYPE_FLOAT64,
			nrow,
			num_feature,                        // ncol
			1_i32,                              // is_row_major
			lightgbm_sys::C_API_PREDICT_NORMAL, // predict_type
			0_i32,                              // start_iteration
			-1_i32,                             // num_iteration
			self.param_overrides.as_ptr() as *const c_char,
			&mut out_length,
			out_result.as_ptr() as *mut c_double
		))?;

		assert!(
			usize::try_from(out_length).is_ok_and(|l| l == out_result.len()),
			"Unexpected written output length"
		);

		Ok(out_result)
	}

	pub fn predict_single_row(&self, data: &[f64]) -> Result<Vec<f64>> {
		let num_feature: i32 = self.num_feature()?;
		let n_features: usize = num_feature
			.try_into()
			.map_err(|_| Error::new("number of features doesn't fit into an usize"))?;
		if data.len() != n_features {
			return Err(Error::new(format!(
				"data len ({}) is equal to n_features ({n_features}), \
					but this is a single-row prediction",
				data.len(),
			)));
		}

		let predict_output_len = self.predict_output_len(1)?;
		let out_result: Vec<f64> = vec![Default::default(); predict_output_len];

		let mut out_length: c_longlong = 0;
		lgbm_call!(lightgbm_sys::LGBM_BoosterPredictForMatSingleRow(
			self.handle,
			data.as_ptr() as *const c_void,
			lightgbm_sys::C_API_DTYPE_FLOAT64,
			data.len() as i32,
			1_i32, // is_row_major
			lightgbm_sys::C_API_PREDICT_NORMAL,
			0_i32,  // start_iteration
			-1_i32, // num_iteration,
			self.param_overrides.as_ptr() as *const c_char,
			&mut out_length,
			out_result.as_ptr() as *mut c_double,
		))?;

		assert!(
			usize::try_from(out_length).is_ok_and(|l| l == out_result.len()),
			"Unexpected written output length"
		);

		Ok(out_result)
	}

	pub fn single_row_predictor<'a>(&'a self) -> Result<SingleRowPredictor<'a>> {
		let num_feature: i32 = self.num_feature()?;
		let input_size: usize = num_feature.try_into().map_err(|_| {
			Error::new("Number of features returned by LGBM C API doesn't fit in a usize")
		})?;

		// https://github.com/microsoft/LightGBM/issues/6142
		// Because at lower level atomic operations don't actually point to the data that needs to
		// be synchronized, this fixes the issue.
		let _guard = self.race_workaround_mutex.lock().unwrap();

		let output_size = self.predict_output_len(1)?;

		let mut handle = std::ptr::null_mut();

		lgbm_call!(lightgbm_sys::LGBM_BoosterPredictForMatSingleRowFastInit(
			self.handle,
			lightgbm_sys::C_API_PREDICT_NORMAL, // predict_type
			0_i32,                              // start_iteration
			-1_i32,                             // num_iteration
			lightgbm_sys::C_API_DTYPE_FLOAT64,
			num_feature,
			self.param_overrides.as_ptr() as *const c_char,
			&mut handle,
		))?;

		Ok(SingleRowPredictor {
			handle,
			input_size,
			output_size,
			booster: std::marker::PhantomData::<&'a Self>,
		})
	}

	/// Get the size of the output array that will be required for this prediction
	pub(crate) fn predict_output_len(&self, n_rows: i32) -> Result<usize> {
		let mut output_size: i64 = 0;
		lgbm_call!(lightgbm_sys::LGBM_BoosterCalcNumPredict(
			self.handle,
			n_rows,
			lightgbm_sys::C_API_PREDICT_NORMAL, // predict_type
			0_i32,                              // start_iteration
			-1_i32,                             // num_iteration
			&mut output_size
		))?;
		output_size
			.try_into()
			.map_err(|_| Error::new("Output size returned by LGBM C API doesn't fit in a usize"))
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

	fn _feature_names(&self, num_features: i32, feature_name_size: usize) -> Result<FeatureNames> {
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

		const DEFAULT_MAX_FEATURE_NAME_SIZE: usize = 64;
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

	/// Save model to string. This returns the same content that `save_file` writes into a file.
	pub fn save_string(&self) -> Result<Vec<u8>> {
		// get nessesary buffer size
		let mut out_size = 0_i64;
		lgbm_call!(lightgbm_sys::LGBM_BoosterSaveModelToString(
			self.handle,
			0_i32,
			-1_i32,
			0_i32,
			0,
			&mut out_size as *mut _,
			std::ptr::null_mut() as *mut i8
		))?;

		// write data to buffer and convert
		let mut buffer = vec![
			0u8;
			out_size.try_into().map_err(|_| Error::new(
				"string size returned by C API is negative"
			))?
		];
		lgbm_call!(lightgbm_sys::LGBM_BoosterSaveModelToString(
			self.handle,
			0_i32,
			-1_i32,
			0_i32,
			out_size,
			&mut out_size as *mut _,
			buffer.as_mut_ptr() as *mut c_char
		))?;

		if buffer.pop() != Some(0) {
			// this should never happen, unless lightgbm has a bug
			panic!("write out of bounds happened in lightgbm call");
		}

		let cstring = CString::new(buffer)
			.map_err(|e| Error::from_other("Parsing CString returned by C API failed", e))?;
		Ok(cstring.into_bytes())
	}

	/// Save model to string. This returns the same content that `save_file` writes into a file.
	pub fn dump_model_json(&self) -> Result<Vec<u8>> {
		// get nessesary buffer size
		let mut out_size = 0_i64;
		lgbm_call!(lightgbm_sys::LGBM_BoosterDumpModel(
			self.handle,
			0_i32,
			-1_i32,
			0_i32,
			0,
			&mut out_size as *mut _,
			std::ptr::null_mut() as *mut i8
		))?;

		// write data to buffer and convert
		let mut buffer = vec![
			0u8;
			out_size.try_into().map_err(|_| Error::new(
				"string size returned by C API is negative"
			))?
		];
		lgbm_call!(lightgbm_sys::LGBM_BoosterDumpModel(
			self.handle,
			0_i32,
			-1_i32,
			0_i32,
			out_size,
			&mut out_size as *mut _,
			buffer.as_mut_ptr() as *mut c_char
		))?;

		if buffer.pop() != Some(0) {
			// this should never happen, unless lightgbm has a bug
			panic!("write out of bounds happened in lightgbm call");
		}

		let cstring = CString::new(buffer)
			.map_err(|e| Error::from_other("Parsing CString returned by C API failed", e))?;
		Ok(cstring.into_bytes())
	}
}

struct FeatureNames {
	features: Vec<Vec<u8>>,
	actual_feature_name_len: usize,
	num_feature_names: i32,
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
		let mut bst = _train_booster(&params);
		let features = [[0.5; 28], [0.0; 28], [0.9; 28]]
			.into_iter()
			.flatten()
			.collect::<Vec<_>>();
		let result = bst.predict(&features).unwrap();
		let mut normalized_result: Vec<i32> = Vec::new();
		for &r in &result {
			normalized_result.push((r > 0.5).into());
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
		let feature = [[0.5; 28], [0.0; 28], [0.9; 28]];

		let result: Vec<f64> = feature
			.iter()
			.flat_map(|f| bst.predict_single_row(f).unwrap())
			.collect();

		let mut normalized_result: Vec<i32> = Vec::new();
		for r in result {
			normalized_result.push((r > 0.5).into());
		}
		assert_eq!(normalized_result, vec![0, 0, 1]);
	}

	#[test]
	fn predict_single_row_fast() {
		let params = json! {
			{
				"num_iterations": 10,
				"objective": "binary",
				"metric": "auc",
				"data_random_seed": 0
			}
		};
		let bst = _train_booster(&params);
		let single_row_predictor = bst.single_row_predictor().unwrap();

		let feature = [[0.5; 28], [0.0; 28], [0.9; 28]];

		let result: Vec<f64> = feature
			.iter()
			.flat_map(|f| single_row_predictor.predict(f).unwrap())
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
	fn save_string() {
		let params = _default_params();
		let bst = _train_booster(&params);
		let filename = "./test/test_save_string.output";
		assert_eq!(bst.save_file(&filename), Ok(()));
		assert!(Path::new(&filename).exists());
		let booster_file_content = fs::read(&filename).unwrap();
		let _ = fs::remove_file("./test/test_save_string.output");

		assert!(!booster_file_content.is_empty());
		assert_eq!(Ok(booster_file_content), bst.save_string())
	}

	#[test]
	fn from_file() {
		let _ = Booster::from_file("./test/test_from_file.input").unwrap();
	}

	#[test]
	fn from_string() {
		let file = fs::read_to_string("./test/test_from_file.input").unwrap();
		let _ = Booster::from_string(file.as_bytes()).unwrap();
	}

	#[test]
	fn dump_model_json() {
		let params = _default_params();
		let bst = _train_booster(&params);
		let json_dump = String::from_utf8(bst.dump_model_json().expect("Failed to dump model"))
			.expect("Failed to convert JSON dump to string");

		let sjv = serde_json::from_str::<serde_json::Value>(&json_dump).unwrap();

		assert_eq!(
			sjv,
			json!({
			  "average_output": false,
			  "feature_importances": {},
			  "feature_infos": {
				"Column_0": {
				  "max_value": 6.695,
				  "min_value": 0.275,
				  "values": []
				},
				"Column_1": {
				  "max_value": 2.43,
				  "min_value": -2.417,
				  "values": []
				},
				"Column_10": {
				  "max_value": 2.909,
				  "min_value": -2.904,
				  "values": []
				},
				"Column_11": {
				  "max_value": 1.743,
				  "min_value": -1.742,
				  "values": []
				},
				"Column_12": {
				  "max_value": 2.215,
				  "min_value": 0,
				  "values": []
				},
				"Column_13": {
				  "max_value": 6.523,
				  "min_value": 0.264,
				  "values": []
				},
				"Column_14": {
				  "max_value": 2.727,
				  "min_value": -2.728,
				  "values": []
				},
				"Column_15": {
				  "max_value": 1.742,
				  "min_value": -1.742,
				  "values": []
				},
				"Column_16": {
				  "max_value": 2.548,
				  "min_value": 0,
				  "values": []
				},
				"Column_17": {
				  "max_value": 6.068,
				  "min_value": 0.365,
				  "values": []
				},
				"Column_18": {
				  "max_value": 2.496,
				  "min_value": -2.495,
				  "values": []
				},
				"Column_19": {
				  "max_value": 1.743,
				  "min_value": -1.74,
				  "values": []
				},
				"Column_2": {
				  "max_value": 1.743,
				  "min_value": -1.743,
				  "values": []
				},
				"Column_20": {
				  "max_value": 3.102,
				  "min_value": 0,
				  "values": []
				},
				"Column_21": {
				  "max_value": 13.098,
				  "min_value": 0.172,
				  "values": []
				},
				"Column_22": {
				  "max_value": 7.392,
				  "min_value": 0.419,
				  "values": []
				},
				"Column_23": {
				  "max_value": 3.682,
				  "min_value": 0.461,
				  "values": []
				},
				"Column_24": {
				  "max_value": 6.583,
				  "min_value": 0.384,
				  "values": []
				},
				"Column_25": {
				  "max_value": 7.86,
				  "min_value": 0.09300000000000001,
				  "values": []
				},
				"Column_26": {
				  "max_value": 4.543,
				  "min_value": 0.389,
				  "values": []
				},
				"Column_27": {
				  "max_value": 4.316,
				  "min_value": 0.489,
				  "values": []
				},
				"Column_3": {
				  "max_value": 5.7,
				  "min_value": 0.019,
				  "values": []
				},
				"Column_4": {
				  "max_value": 1.743,
				  "min_value": -1.743,
				  "values": []
				},
				"Column_5": {
				  "max_value": 4.19,
				  "min_value": 0.159,
				  "values": []
				},
				"Column_6": {
				  "max_value": 2.97,
				  "min_value": -2.941,
				  "values": []
				},
				"Column_7": {
				  "max_value": 1.741,
				  "min_value": -1.741,
				  "values": []
				},
				"Column_8": {
				  "max_value": 2.173,
				  "min_value": 0,
				  "values": []
				},
				"Column_9": {
				  "max_value": 5.193,
				  "min_value": 0.19,
				  "values": []
				}
			  },
			  "feature_names": [
				"Column_0",
				"Column_1",
				"Column_2",
				"Column_3",
				"Column_4",
				"Column_5",
				"Column_6",
				"Column_7",
				"Column_8",
				"Column_9",
				"Column_10",
				"Column_11",
				"Column_12",
				"Column_13",
				"Column_14",
				"Column_15",
				"Column_16",
				"Column_17",
				"Column_18",
				"Column_19",
				"Column_20",
				"Column_21",
				"Column_22",
				"Column_23",
				"Column_24",
				"Column_25",
				"Column_26",
				"Column_27"
			  ],
			  "label_index": 0,
			  "max_feature_idx": 27,
			  "monotone_constraints": [],
			  "name": "tree",
			  "num_class": 1,
			  "num_tree_per_iteration": 1,
			  "objective": "binary sigmoid:1",
			  "tree_info": [],
			  "version": "v4"
			})
		);
	}
}
