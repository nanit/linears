

fn main() {
    cc::Build::new()
        .cpp(true)
        .flag("-O2")
        .file("liblinear/linear.cpp")
        .file("liblinear/newton.cpp")
        .file("liblinear/blas/daxpy.c")
        .file("liblinear/blas/ddot.c")
        .file("liblinear/blas/dnrm2.c")
        .file("liblinear/blas/dscal.c")
        .include("liblinear")
        .include("liblinear/blas")
        .compile("liblinear");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=wrapper.h");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("wrapper.h")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    bindings
        .write_to_file("src/bindings.rs")
        .expect("Couldn't write bindings!");
}