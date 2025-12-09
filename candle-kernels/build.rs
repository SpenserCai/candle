use std::env;
use std::path::PathBuf;
use std::process::Command;
use std::io::Write;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");
    println!("cargo::rerun-if-env-changed=CANDLE_CUDA_MODULE_FORMAT");
    println!("cargo::rerun-if-env-changed=CUDA_COMPUTE_CAP");
    println!("cargo::rerun-if-env-changed=CUDA_NVCC_FLAGS");

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
    println!("cargo::warning=Using bindgen_cuda: {builder:?}");
    
    let bindings = builder.build_ptx()
        .expect("Failed to build PTX modules with bindgen_cuda");
    
    bindings.write(ptx_path)
        .expect("Failed to write PTX bindings");
    
    println!("cargo::warning=Successfully built PTX modules");
}

/// Build CUBIN modules using direct nvcc invocation
fn build_cubin_modules(out_dir: &PathBuf) {
    // Set cfg flag for conditional compilation in lib.rs
    println!("cargo::rustc-cfg=candle_cuda_cubin");
    
    let compute_cap = get_compute_capability();
    println!("cargo::warning=Building CUDA kernels in CUBIN mode");
    println!("cargo::warning=Target compute capability: sm_{}", compute_cap);
    
    // Kernel list matching Id enum in lib.rs
    let kernels = [
        "affine",
        "binary",
        "cast",
        "conv",
        "fill",
        "indexing",
        "quantized",
        "reduce",
        "sort",
        "ternary",
        "unary",
    ];
    
    // Add dependency tracking for all kernel source files
    for kernel_name in &kernels {
        let src_path = format!("src/{}.cu", kernel_name);
        println!("cargo::rerun-if-changed={}", src_path);
    }
    
    // Compile each kernel to CUBIN
    for kernel_name in &kernels {
        let src = PathBuf::from(format!("src/{}.cu", kernel_name));
        let dst = out_dir.join(format!("{}.cubin", kernel_name));
        
        if !src.exists() {
            panic!(
                "Kernel source file not found: {}\n\
                 Expected kernels: {:?}",
                src.display(),
                kernels
            );
        }
        
        compile_cubin(&src, &dst, &compute_cap);
    }
    
    // Generate cubin.rs file (parallel to ptx.rs)
    generate_cubin_rs(out_dir, &kernels);
    
    println!("cargo::warning=Successfully built {} CUBIN modules", kernels.len());
}

/// Compile a single CUDA source file to CUBIN format
fn compile_cubin(src: &PathBuf, dst: &PathBuf, compute_cap: &str) {
    let mut cmd = Command::new("nvcc");
    
    cmd.arg("--cubin")
        .arg(format!("--gpu-architecture=sm_{}", compute_cap))
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--std=c++17")
        .arg("--default-stream").arg("per-thread")
        .arg("-I").arg("src")
        .arg("-o").arg(dst)
        .arg(src);
    
    // Allow custom compiler flags
    if let Ok(extra_flags) = env::var("CUDA_NVCC_FLAGS") {
        for flag in extra_flags.split_whitespace() {
            cmd.arg(flag);
        }
    }
    
    println!(
        "cargo::warning=Compiling {} -> {}",
        src.file_name().unwrap().to_string_lossy(),
        dst.file_name().unwrap().to_string_lossy()
    );
    
    let output = cmd.output().expect(
        "Failed to execute nvcc.\n\
         Ensure CUDA toolkit (or PPU toolkit) is installed and nvcc is in PATH."
    );
    
    if !output.status.success() {
        let args_str = format!("{:?}", cmd);
        
        panic!(
            "nvcc compilation failed for {:?}\n\
             \n\
             Command: {}\n\
             \n\
             stdout:\n{}\n\
             \n\
             stderr:\n{}",
            src,
            args_str,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

/// Generate cubin.rs file (mimics bindgen_cuda's ptx.rs format)
fn generate_cubin_rs(out_dir: &PathBuf, kernels: &[&str]) {
    let cubin_rs_path = out_dir.join("cubin.rs");
    let mut file = std::fs::File::create(&cubin_rs_path)
        .unwrap_or_else(|_| panic!("Failed to create {:?}", cubin_rs_path));
    
    // Generate format matching ptx.rs
    for kernel_name in kernels {
        let const_name = kernel_name.to_uppercase();
        let line = format!(
            "pub const {}: &[u8] = include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{}.cubin\"));\n",
            const_name, kernel_name
        );
        file.write_all(line.as_bytes())
            .expect("Failed to write to cubin.rs");
    }
    
    println!("cargo::warning=Generated cubin.rs with {} modules", kernels.len());
}

/// Get CUDA compute capability
fn get_compute_capability() -> String {
    // 1. Environment variable takes priority
    if let Ok(cap) = env::var("CUDA_COMPUTE_CAP") {
        println!("cargo::warning=Using CUDA compute capability from environment: sm_{}", cap);
        return cap;
    }
    
    // 2. Try to auto-detect from nvidia-smi
    if let Ok(cap) = detect_from_nvidia_smi() {
        println!("cargo::warning=Auto-detected CUDA compute capability: sm_{}", cap);
        println!("cargo::warning=To override, set CUDA_COMPUTE_CAP environment variable");
        return cap;
    }
    
    // 3. For CUBIN builds, compute capability must be specified
    panic!(
        "Cannot detect CUDA compute capability for CUBIN build.\n\
         \n\
         Please set CUDA_COMPUTE_CAP environment variable.\n\
         \n\
         Examples:\n\
         - CUDA_COMPUTE_CAP=75 for Turing (RTX 20xx, T4)\n\
         - CUDA_COMPUTE_CAP=80 for Ampere (A100, RTX 30xx)\n\
         - CUDA_COMPUTE_CAP=86 for Ampere (RTX 30xx mobile)\n\
         - CUDA_COMPUTE_CAP=89 for Ada Lovelace (RTX 40xx)\n\
         - CUDA_COMPUTE_CAP=90 for Hopper (H100)\n\
         \n\
         Find your GPU's compute capability at:\n\
         https://developer.nvidia.com/cuda-gpus\n\
         \n\
         For PPU or other devices, consult your device documentation."
    );
}

fn detect_from_nvidia_smi() -> Result<String, ()> {
    let output = Command::new("nvidia-smi")
        .args(&["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .map_err(|_| ())?;
    
    if output.status.success() {
        let cap = String::from_utf8_lossy(&output.stdout);
        let cap = cap.trim().replace('.', "");
        if !cap.is_empty() && cap.chars().all(|c| c.is_ascii_digit()) {
            return Ok(cap);
        }
    }
    
    Err(())
}
