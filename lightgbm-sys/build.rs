use std::{
	env,
	path::{Path, PathBuf},
	process::Command,
};

fn main() {
	let target = env::var("TARGET").unwrap();
	let out_dir = env::var("OUT_DIR").unwrap();
	let lgbm_root = Path::new(&out_dir).join("lightgbm");

	// copy source code
	if !lgbm_root.exists() {
		let status = if target.contains("windows") {
			Command::new("cmd")
				.args(&[
					"/C",
					"echo D | xcopy /S /Y lightgbm",
					lgbm_root.to_str().unwrap(),
				])
				.status()
		} else {
			Command::new("cp")
				.args(&["-r", "lightgbm", lgbm_root.to_str().unwrap()])
				.status()
		};
		if let Some(err) = status.err() {
			panic!(
				"Failed to copy ./lightgbm to {}: {}",
				lgbm_root.display(),
				err
			);
		}
	}

	// CMake
	let dst = cmake::Config::new(&lgbm_root)
		.profile("Release")
		.uses_cxx11()
		.define("BUILD_STATIC_LIB", "ON")
		.build();

	// bindgen build
	let bindings = bindgen::Builder::default()
		.header("wrapper.h")
		.clang_args(&["-x", "c++", "-std=c++14"])
		.clang_arg(format!("-I{}", lgbm_root.join("include").display()))
		.default_macro_constant_type(bindgen::MacroTypeVariation::Signed)
		.generate()
		.expect("Unable to generate bindings");
	let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
	bindings
		.write_to_file(out_path.join("bindings.rs"))
		.expect("Couldn't write bindings.");

	// link to appropriate C++ lib
	if target.contains("apple") {
		println!("cargo:rustc-link-lib=c++");
		println!("cargo:rustc-link-lib=dylib=omp");
		if let Ok(homebrew_libomp_path) = get_homebrew_libpath("libomp") {
			println!("cargo:rustc-link-search={}", homebrew_libomp_path);
		}
	} else if target.contains("linux") {
		println!("cargo:rustc-link-lib=stdc++");
		println!("cargo:rustc-link-lib=dylib=gomp");
	}

	println!("cargo:rustc-link-search={}", out_path.join("lib").display());
	println!("cargo:rustc-link-search=native={}", dst.display());
	if target.contains("windows") {
		println!("cargo:rustc-link-lib=static=lib_lightgbm");
	} else {
		println!("cargo:rustc-link-lib=static=_lightgbm");
	}
}

#[derive(Debug)]
enum HomebrewError {
	Brew,
	Path(String),
	LibNotFound,
}

fn get_homebrew_libpath(lib: &str) -> Result<String, HomebrewError> {
	let cellar_path = Command::new("brew")
		.args(&["--cellar", lib])
		.output()
		.map_err(|_| HomebrewError::Brew)?
		.stdout;

	let cellar_path = Path::new(
		std::str::from_utf8(&cellar_path)
			.map_err(|e| HomebrewError::Path(format!("from_utf8: {}", e)))?
			.trim(),
	);

	for dir in cellar_path.read_dir().map_err(|e| {
		HomebrewError::Path(format!(
			"read_dir({}): {}",
			cellar_path.to_string_lossy(),
			e
		))
	})? {
		if let Ok(d) = dir {
			if d.metadata()
				.map_err(|e| HomebrewError::Path(format!("metadata: {}", e)))?
				.file_type()
				.is_dir()
			{
				return d.path().join("lib").to_str().map(|s| s.to_string()).ok_or(
					HomebrewError::Path(format!("Could not convert path to string")),
				);
			}
		}
	}
	Err(HomebrewError::LibNotFound)
}
