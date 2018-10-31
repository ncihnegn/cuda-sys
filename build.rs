use std::env;

fn main() {
    let cuda_dir = match env::var("CUDA_HOME") {
        Ok(path) => path,
        Err(_) => "/opt/cuda".to_owned(),
    };

    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-search={}/lib64", cuda_dir);
    println!("cargo:rerun-if-changed=build.rs");
}
