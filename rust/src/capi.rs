// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//! Thin wrapper of the SVS C API for dynamic Vamana indices.
//!
//! Mirrors `svsbench/capi.py`. The shared library is loaded at run time
//! from the SVS_C_API_LIB environment variable (path to libsvs_c_api.so)
//! or, failing that, the system library search path.
//!
//! Error handles are per thread, as required by the C API for indices
//! with internal synchronization.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::Path;
use std::ptr;
use std::sync::OnceLock;

use anyhow::{Context, Result, anyhow, bail};
use libloading::Library;

pub const LIB_ENV: &str = "SVS_C_API_LIB";

/// Version of svs_c.h whose struct layouts are mirrored below.
const HEADER_VERSION: u32 = 5 << 8;

pub const THREADPOOLS: [&str; 2] = ["native", "omp"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum SyncKind {
    None = 0,
    Global = 1,
    FineGrain = 2,
}

impl SyncKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "NONE",
            Self::Global => "GLOBAL",
            Self::FineGrain => "FINE_GRAIN",
        }
    }
}

const METRIC_EUCLIDEAN: c_int = 0;
const METRIC_COSINE: c_int = 1;
const METRIC_DOT_PRODUCT: c_int = 2;

const DATA_TYPE_NONE: c_int = 0;
const DATA_TYPE_UINT4: c_int = (1 << 8) | 4;
const DATA_TYPE_UINT8: c_int = (1 << 8) | 8;
const DATA_TYPE_FLOAT16: c_int = (3 << 8) | 16;

const THREADPOOL_NATIVE: c_int = 0;
const THREADPOOL_OMP: c_int = 1;
const THREADPOOL_SINGLE_THREAD: c_int = 2;

fn metric(distance: &str) -> Result<c_int> {
    Ok(match distance {
        "cosine" => METRIC_COSINE,
        "l2" => METRIC_EUCLIDEAN,
        "mip" => METRIC_DOT_PRODUCT,
        _ => bail!("Unknown distance={distance}"),
    })
}

#[repr(C)]
struct DynamicIndexParams {
    version: u32,
    struct_size: usize,
    blocksize_bytes: usize,
    blocksize_elements: usize,
    sync_kind: u32,
}

impl DynamicIndexParams {
    fn new(sync_kind: SyncKind) -> Self {
        Self {
            version: HEADER_VERSION,
            struct_size: size_of::<Self>(),
            blocksize_bytes: 0,
            blocksize_elements: 0,
            sync_kind: sync_kind as u32,
        }
    }
}

#[repr(C)]
struct SearchResults {
    version: u32,
    struct_size: usize,
    num_queries: usize,
    total_results: usize,
    offsets: *mut usize,
    indices: *mut usize,
    distances: *mut f32,
    offsets_capacity: usize,
    results_capacity: usize,
    owns_buffers: bool,
}

type H = *mut c_void;

macro_rules! api {
    ($($name:ident: fn($($arg:ty),*) $(-> $ret:ty)?;)*) => {
        #[allow(clippy::type_complexity)]
        pub struct Api {
            $($name: unsafe extern "C" fn($($arg),*) $(-> $ret)?,)*
            _lib: Library,
        }

        impl Api {
            /// # Safety
            /// The library must export the functions with these signatures.
            unsafe fn load(lib: Library) -> Result<Self> {
                Ok(Self {
                    $($name: unsafe {
                        *lib.get(concat!(stringify!($name), "\0").as_bytes())
                            .with_context(|| stringify!($name))?
                    },)*
                    _lib: lib,
                })
            }
        }
    };
}

api! {
    svs_get_version_string: fn() -> *const c_char;
    svs_error_create: fn() -> H;
    svs_error_get_code: fn(H) -> c_int;
    svs_error_get_message: fn(H) -> *const c_char;
    svs_error_free: fn(H);
    svs_algorithm_create_vamana: fn(usize, usize, usize, H) -> H;
    svs_algorithm_vamana_set_alpha: fn(H, f32, H) -> bool;
    svs_algorithm_free: fn(H);
    svs_storage_create_simple: fn(c_int, H) -> H;
    svs_storage_create_lvq: fn(c_int, c_int, H) -> H;
    svs_storage_create_leanvec: fn(usize, c_int, c_int, H) -> H;
    svs_storage_create_leanvec_trained: fn(H, c_int, c_int, H) -> H;
    svs_storage_free: fn(H);
    svs_leanvec_training_data_build:
        fn(H, usize, usize, *const f32, usize, *const f32, H) -> H;
    svs_leanvec_training_data_free: fn(H);
    svs_index_builder_create: fn(c_int, usize, H, H) -> H;
    svs_index_builder_set_storage: fn(H, H, H) -> bool;
    svs_index_builder_set_threadpool: fn(H, c_int, usize, H) -> bool;
    svs_index_builder_free: fn(H);
    svs_index_build_dynamic_ex: fn(
        H, *const f32, *const usize, usize, *const DynamicIndexParams, H
    ) -> H;
    svs_index_convert_dynamic_ex:
        fn(H, H, *const DynamicIndexParams, H) -> H;
    svs_index_load_dynamic_ex:
        fn(H, *const c_char, *const DynamicIndexParams, H) -> H;
    svs_index_free: fn(H);
    svs_index_get_size: fn(H, *mut usize, H) -> bool;
    svs_index_save: fn(H, *const c_char, H) -> bool;
    svs_index_set_num_threads: fn(H, usize, H) -> bool;
    svs_index_dynamic_get_sync_kind: fn(H, *mut u32, H) -> bool;
    svs_index_dynamic_add_points:
        fn(H, *const f32, *const usize, usize, *mut usize, H) -> bool;
    svs_search_params_create_vamana: fn(usize, H) -> H;
    svs_search_params_free: fn(H);
    svs_index_search_topk: fn(
        H, *const f32, usize, usize, *mut SearchResults, H, H, H
    ) -> bool;
}

static API: OnceLock<Api> = OnceLock::new();

fn library_name() -> String {
    std::env::var(LIB_ENV).unwrap_or_else(|_| "libsvs_c_api.so".to_owned())
}

/// Load the SVS C API shared library on first use.
pub fn api() -> &'static Api {
    API.get_or_init(|| {
        let name = library_name();
        // SAFETY: loading the library runs its initializers, and the
        // function signatures above mirror svs_c.h.
        unsafe { Library::new(&name).and_then(|lib| Ok(Api::load(lib))) }
            .unwrap_or_else(|error| {
                panic!(
                    "SVS C API library {name} not found ({error}). \
                     Set {LIB_ENV} to the path of libsvs_c_api.so."
                )
            })
            .unwrap_or_else(|error| panic!("{error:#}"))
    })
}

/// Return the path of the SVS C API library, resolved if it exists.
pub fn library_path() -> String {
    let name = library_name();
    std::fs::canonicalize(&name)
        .map(|path| path.display().to_string())
        .unwrap_or(name)
}

pub fn version() -> String {
    // SAFETY: the library returns a static NUL-terminated string.
    unsafe { CStr::from_ptr((api().svs_get_version_string)()) }
        .to_string_lossy()
        .into_owned()
}

struct ErrorHandle(H);

impl Drop for ErrorHandle {
    fn drop(&mut self) {
        unsafe { (api().svs_error_free)(self.0) }
    }
}

thread_local! {
    static ERROR: ErrorHandle = ErrorHandle(unsafe { (api().svs_error_create)() });
}

/// Return the error handle of the current thread.
fn error() -> H {
    ERROR.with(|error| error.0)
}

fn to_error(function: &str, err: H) -> anyhow::Error {
    let api = api();
    let code = unsafe { (api.svs_error_get_code)(err) };
    let message = unsafe { (api.svs_error_get_message)(err) };
    let message = if message.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(message) }.to_string_lossy().into_owned()
    };
    anyhow!("{function} failed with code {code}: {message}")
}

/// Call a C API function returning a handle and check errors.
macro_rules! call_h {
    ($name:ident($($arg:expr),*)) => {{
        let err = error();
        let handle = unsafe { (api().$name)($($arg,)* err) };
        if handle.is_null() {
            Err(to_error(stringify!($name), err))
        } else {
            Ok(handle)
        }
    }};
}

/// Call a C API function returning bool and check errors.
macro_rules! call_b {
    ($name:ident($($arg:expr),*)) => {{
        let err = error();
        if unsafe { (api().$name)($($arg,)* err) } {
            Ok(())
        } else {
            Err(to_error(stringify!($name), err))
        }
    }};
}

/// Resolve negative LeanVec dims as dim / -leanvec_dims.
pub fn effective_leanvec_dims(leanvec_dims: i64, dim: usize) -> usize {
    if leanvec_dims < 0 {
        dim / (-leanvec_dims) as usize
    } else {
        leanvec_dims as usize
    }
}

/// Algorithm and storage configuration of an index builder.
#[derive(Clone, Debug)]
pub struct BuilderConfig {
    pub svs_type: String,
    pub distance: String,
    pub graph_max_degree: usize,
    pub window_size: usize,
    pub alpha: Option<f32>,
    pub num_threads: usize,
    pub threadpool: String,
    pub leanvec_dims: i64,
}

/// Row-major float32 matrix borrowed from the caller.
#[derive(Clone, Copy)]
pub struct Matrix<'a> {
    pub data: &'a [f32],
    pub dim: usize,
}

impl Matrix<'_> {
    pub fn rows(&self) -> usize {
        self.data.len() / self.dim
    }
}

/// Index builder with algorithm and storage configuration.
pub struct Builder {
    handle: H,
    // Freed in reverse order after the builder.
    handles: Vec<(unsafe extern "C" fn(H), H)>,
}

impl Builder {
    /// If train_queries is given, the LeanVec matrices are trained
    /// out-of-distribution on train_vectors and train_queries.
    pub fn new(
        config: &BuilderConfig,
        dim: usize,
        training: Option<(Matrix, Matrix)>,
    ) -> Result<Self> {
        let api = api();
        let mut builder = Self {
            handle: ptr::null_mut(),
            handles: Vec::new(),
        };
        let algorithm = call_h!(svs_algorithm_create_vamana(
            config.graph_max_degree,
            config.window_size,
            config.window_size
        ))?;
        builder.handles.push((api.svs_algorithm_free, algorithm));
        if let Some(alpha) = config.alpha {
            call_b!(svs_algorithm_vamana_set_alpha(algorithm, alpha))?;
        }
        builder.handle = call_h!(svs_index_builder_create(
            metric(&config.distance)?,
            dim,
            algorithm
        ))?;
        builder.set_threadpool(config.num_threads, &config.threadpool)?;
        if let Some(storage) = builder.create_storage(config, dim, training)? {
            builder.handles.push((api.svs_storage_free, storage));
            call_b!(svs_index_builder_set_storage(builder.handle, storage))?;
        }
        Ok(builder)
    }

    fn create_storage(
        &self,
        config: &BuilderConfig,
        dim: usize,
        training: Option<(Matrix, Matrix)>,
    ) -> Result<Option<H>> {
        let leanvec = |primary, secondary| -> Result<H> {
            let dims = effective_leanvec_dims(config.leanvec_dims, dim);
            let Some((vectors, queries)) = training else {
                return call_h!(svs_storage_create_leanvec(dims, primary, secondary));
            };
            let training_data = call_h!(svs_leanvec_training_data_build(
                self.handle,
                dims,
                vectors.rows(),
                vectors.data.as_ptr(),
                queries.rows(),
                queries.data.as_ptr()
            ))?;
            let storage = call_h!(svs_storage_create_leanvec_trained(
                training_data,
                primary,
                secondary
            ));
            unsafe { (api().svs_leanvec_training_data_free)(training_data) };
            storage
        };
        Ok(Some(match config.svs_type.as_str() {
            "float32" => return Ok(None),
            "float16" => call_h!(svs_storage_create_simple(DATA_TYPE_FLOAT16))?,
            "lvq8" => call_h!(svs_storage_create_lvq(DATA_TYPE_UINT8, DATA_TYPE_NONE))?,
            "lvq4x4" => call_h!(svs_storage_create_lvq(DATA_TYPE_UINT4, DATA_TYPE_UINT4))?,
            "lvq4x8" => call_h!(svs_storage_create_lvq(DATA_TYPE_UINT4, DATA_TYPE_UINT8))?,
            "leanvec4x4" => leanvec(DATA_TYPE_UINT4, DATA_TYPE_UINT4)?,
            "leanvec4x8" => leanvec(DATA_TYPE_UINT4, DATA_TYPE_UINT8)?,
            "leanvec8x8" => leanvec(DATA_TYPE_UINT8, DATA_TYPE_UINT8)?,
            svs_type => bail!("Unknown svs_type={svs_type}"),
        }))
    }

    /// Set the thread pool of indices created by this builder.
    ///
    /// A native pool with one thread is replaced by running every call
    /// on its caller's thread, because a native pool serializes
    /// concurrent calls unless the library wraps it for
    /// SVS_SYNC_KIND_FINE_GRAIN.
    fn set_threadpool(&self, num_threads: usize, threadpool: &str) -> Result<()> {
        let kind = match threadpool {
            "omp" => THREADPOOL_OMP,
            "native" if num_threads == 1 => THREADPOOL_SINGLE_THREAD,
            "native" => THREADPOOL_NATIVE,
            _ => bail!("Unknown threadpool={threadpool}"),
        };
        call_b!(svs_index_builder_set_threadpool(self.handle, kind, num_threads))
    }
}

impl Drop for Builder {
    fn drop(&mut self) {
        let api = api();
        if !self.handle.is_null() {
            unsafe { (api.svs_index_builder_free)(self.handle) };
        }
        while let Some((free, handle)) = self.handles.pop() {
            unsafe { free(handle) };
        }
    }
}

fn path_cstring(path: &Path) -> Result<CString> {
    CString::new(path.as_os_str().as_encoded_bytes()).context("path contains NUL")
}

/// Dynamic Vamana index of the SVS C API.
pub struct DynamicIndex {
    handle: H,
    pub dim: usize,
    // Dropped after the index.
    _builder: Builder,
}

// SAFETY: the handle is only used through C API calls, which are
// thread-safe for indices with internal synchronization.
unsafe impl Send for DynamicIndex {}
unsafe impl Sync for DynamicIndex {}

impl DynamicIndex {
    /// Build a dynamic index from vectors with the given ids.
    pub fn build(
        vectors: Matrix,
        ids: &[usize],
        config: &BuilderConfig,
        training: Option<(Matrix, Matrix)>,
        sync_kind: SyncKind,
    ) -> Result<Self> {
        assert_eq!(vectors.rows(), ids.len());
        let builder = Builder::new(config, vectors.dim, training)?;
        let params = DynamicIndexParams::new(sync_kind);
        let handle = call_h!(svs_index_build_dynamic_ex(
            builder.handle,
            vectors.data.as_ptr(),
            ids.as_ptr(),
            ids.len(),
            &params
        ))?;
        Ok(Self {
            handle,
            dim: vectors.dim,
            _builder: builder,
        })
    }

    /// Load a dynamic index saved with save().
    pub fn load(
        directory: &Path,
        dim: usize,
        config: &BuilderConfig,
        sync_kind: SyncKind,
    ) -> Result<Self> {
        let builder = Builder::new(config, dim, None)?;
        let params = DynamicIndexParams::new(sync_kind);
        let directory = path_cstring(directory)?;
        let handle = call_h!(svs_index_load_dynamic_ex(
            builder.handle,
            directory.as_ptr(),
            &params
        ))?;
        Ok(Self {
            handle,
            dim,
            _builder: builder,
        })
    }

    /// Convert to a new dynamic index with another storage type.
    ///
    /// Without training data, LeanVec uses PCA of the data of this index.
    pub fn convert(
        &self,
        config: &BuilderConfig,
        training: Option<(Matrix, Matrix)>,
        sync_kind: SyncKind,
    ) -> Result<Self> {
        let builder = Builder::new(config, self.dim, training)?;
        let params = DynamicIndexParams::new(sync_kind);
        let handle = call_h!(svs_index_convert_dynamic_ex(
            builder.handle,
            self.handle,
            &params
        ))?;
        Ok(Self {
            handle,
            dim: self.dim,
            _builder: builder,
        })
    }

    pub fn size(&self) -> Result<usize> {
        let mut size = 0;
        call_b!(svs_index_get_size(self.handle, &mut size))?;
        Ok(size)
    }

    pub fn set_num_threads(&self, num_threads: usize) -> Result<()> {
        call_b!(svs_index_set_num_threads(self.handle, num_threads))
    }

    pub fn sync_kind(&self) -> Result<SyncKind> {
        let mut sync_kind = 0;
        call_b!(svs_index_dynamic_get_sync_kind(self.handle, &mut sync_kind))?;
        Ok(match sync_kind {
            0 => SyncKind::None,
            1 => SyncKind::Global,
            2 => SyncKind::FineGrain,
            _ => bail!("Unknown sync_kind={sync_kind}"),
        })
    }

    pub fn save(&self, directory: &Path) -> Result<()> {
        std::fs::create_dir_all(directory)?;
        let directory = path_cstring(directory)?;
        call_b!(svs_index_save(self.handle, directory.as_ptr()))
    }

    /// Add vectors with the given ids in a single call.
    pub fn add(&self, vectors: &[f32], ids: &[usize]) -> Result<()> {
        assert_eq!(vectors.len(), ids.len() * self.dim);
        call_b!(svs_index_dynamic_add_points(
            self.handle,
            vectors.as_ptr(),
            ids.as_ptr(),
            ids.len(),
            ptr::null_mut()
        ))
    }
}

impl Drop for DynamicIndex {
    fn drop(&mut self) {
        unsafe { (api().svs_index_free)(self.handle) };
    }
}

/// Vamana search parameters.
pub struct SearchParams {
    handle: H,
}

// SAFETY: the parameters are only read by the library.
unsafe impl Send for SearchParams {}
unsafe impl Sync for SearchParams {}

impl SearchParams {
    pub fn new(search_window_size: usize) -> Result<Self> {
        Ok(Self {
            handle: call_h!(svs_search_params_create_vamana(search_window_size))?,
        })
    }
}

impl Drop for SearchParams {
    fn drop(&mut self) {
        unsafe { (api().svs_search_params_free)(self.handle) };
    }
}

/// Searches with caller-owned result buffers, by one thread at a time.
pub struct Searcher<'a> {
    index: &'a DynamicIndex,
    k: usize,
    offsets: Vec<usize>,
    results: SearchResults,
}

impl<'a> Searcher<'a> {
    pub fn new(index: &'a DynamicIndex, k: usize, max_queries_per_call: usize) -> Self {
        let mut offsets = vec![0; max_queries_per_call + 1];
        let results = SearchResults {
            version: HEADER_VERSION,
            struct_size: size_of::<SearchResults>(),
            num_queries: 0,
            total_results: 0,
            offsets: offsets.as_mut_ptr(),
            indices: ptr::null_mut(),
            distances: ptr::null_mut(),
            offsets_capacity: offsets.len(),
            results_capacity: max_queries_per_call * k,
            owns_buffers: false,
        };
        Self {
            index,
            k,
            offsets,
            results,
        }
    }

    /// Search queries, writing k results per query to indices and distances.
    pub fn search(
        &mut self,
        queries: &[f32],
        indices: &mut [usize],
        distances: &mut [f32],
        params: &SearchParams,
    ) -> Result<()> {
        let num_queries = queries.len() / self.index.dim;
        let expected = num_queries * self.k;
        assert!(num_queries < self.offsets.len());
        assert!(indices.len() >= expected && distances.len() >= expected);
        self.results.indices = indices.as_mut_ptr();
        self.results.distances = distances.as_mut_ptr();
        call_b!(svs_index_search_topk(
            self.index.handle,
            queries.as_ptr(),
            num_queries,
            self.k,
            &mut self.results,
            params.handle,
            ptr::null_mut()
        ))?;
        if self.results.total_results != expected {
            bail!(
                "svs_index_search_topk: expected {expected} results but got {}",
                self.results.total_results
            );
        }
        Ok(())
    }
}

pub fn runtime_info() -> serde_json::Value {
    serde_json::json!({
        "capi_runtime": {
            "library_path": library_path(),
            "library_version": version(),
            "implementation": "rust",
        }
    })
}
