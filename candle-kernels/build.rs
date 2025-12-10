use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");
    println!("cargo::rerun-if-env-changed=CANDLE_CUDA_MODULE_FORMAT");
    
    // Declare the cfg for conditional compilation
    println!("cargo::rustc-check-cfg=cfg(candle_cuda_cubin)");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    let module_format = env::var("CANDLE_CUDA_MODULE_FORMAT")
        .unwrap_or_else(|_| "ptx".to_string())
        .to_lowercase();
    
    match module_format.as_str() {
        "ptx" => build_ptx_modules(&out_dir),
        "cubin" => build_cubin_modules(&out_dir),
        other => panic!(
            "Invalid CANDLE_CUDA_MODULE_FORMAT: '{}'. Valid values: 'ptx' or 'cubin'",
            other
        ),
    }
}

/// Build PTX modules using bindgen_cuda (default)
fn build_ptx_modules(out_dir: &PathBuf) {
    println!("cargo::warning=Building CUDA kernels in PTX mode (default)");
    
    let ptx_path = out_dir.join("ptx.rs");
    let builder = bindgen_cuda::Builder::default();
    println!("cargo::info={builder:?}");
    
    let bindings = builder.build_ptx()
        .expect("Failed to build PTX modules with bindgen_cuda");
    
    bindings.write(ptx_path)
        .expect("Failed to write PTX bindings");
}

/// Build CUBIN modules using bindgen_cuda
fn build_cubin_modules(out_dir: &PathBuf) {
    // Set cfg flag for conditional compilation in lib.rs
    println!("cargo::rustc-cfg=candle_cuda_cubin");
    
    println!("cargo::warning=Building CUDA kernels in CUBIN mode");
    
    let cubin_path = out_dir.join("cubin.rs");
    let builder = bindgen_cuda::Builder::default();
    println!("cargo::info={builder:?}");
    
    let bindings = builder.build_cubin()
        .expect("Failed to build CUBIN modules with bindgen_cuda");
    
    bindings.write(cubin_path)
        .expect("Failed to write CUBIN bindings");
}
