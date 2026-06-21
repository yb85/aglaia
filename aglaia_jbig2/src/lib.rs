use pyo3::prelude::*;
use pyo3::types::PyBytes;
use ndarray::Array2;
use jbig2enc_rust::{
    encode_single_image, encode_single_image_lossless,
    jbig2enc::encode_document, Jbig2Config,
};

fn pick_config(
    mode: &str,
    match_tolerance: Option<u32>,
    sym_unify_max_err: Option<u32>,
) -> PyResult<Jbig2Config> {
    let mut cfg = match mode {
        "lossless" => Jbig2Config::lossless(),
        "lossy_text" => Jbig2Config::text(),
        "lossy_unify" => Jbig2Config::text_symbol_unify(),
        "default" => Jbig2Config::default(),
        other => return Err(pyo3::exceptions::PyValueError::new_err(
            format!("unknown mode: {other}; expected one of lossless / lossy_text / lossy_unify / default"))),
    };
    cfg.auto_thresh = false;
    cfg.want_full_headers = false;
    if let Some(tol) = match_tolerance {
        cfg.match_tolerance = tol;
    }
    if let Some(e) = sym_unify_max_err {
        cfg.sym_unify_max_err = e;
    }
    Ok(cfg)
}

/// Single-page encode. Returns (globals_or_None, page_bytes).
/// pdf_mode=true emits an embedded JBIG2 stream (no file header).
#[pyfunction]
#[pyo3(signature = (data, width, height, *, pdf_mode=true, lossless=false))]
fn encode_page<'py>(
    py: Python<'py>,
    data: &[u8],
    width: u32,
    height: u32,
    pdf_mode: bool,
    lossless: bool,
) -> PyResult<(Option<Bound<'py, PyBytes>>, Bound<'py, PyBytes>)> {
    let expected = (width as usize) * (height as usize);
    if data.len() != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("data length {} != width*height {}", data.len(), expected)));
    }
    let result = py.allow_threads(|| {
        if lossless {
            encode_single_image_lossless(data, width, height, pdf_mode)
        } else {
            encode_single_image(data, width, height, pdf_mode)
        }
    }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;

    let globals = result.global_data.as_ref().map(|g| PyBytes::new_bound(py, g));
    let page = PyBytes::new_bound(py, &result.page_data);
    Ok((globals, page))
}

/// Multi-page chunk encode using shared symbol dictionary.
/// Pages are passed as a list of (data_bytes, w, h) tuples.
/// Returns one complete JBIG2 document blob covering all pages.
///
/// mode: "lossless" | "lossy_text" | "lossy_unify" | "default".
#[pyfunction]
#[pyo3(signature = (pages, *, mode="lossy_unify", match_tolerance=None, sym_unify_max_err=None))]
fn encode_chunk<'py>(
    py: Python<'py>,
    pages: Vec<(Vec<u8>, u32, u32)>,
    mode: &str,
    match_tolerance: Option<u32>,
    sym_unify_max_err: Option<u32>,
) -> PyResult<Bound<'py, PyBytes>> {
    let cfg = pick_config(mode, match_tolerance, sym_unify_max_err)?;
    // Materialise the ndarray copies first; encode_document wants &[Array2<u8>].
    let mut imgs: Vec<Array2<u8>> = Vec::with_capacity(pages.len());
    for (i, (data, w, h)) in pages.into_iter().enumerate() {
        let expected = (w as usize) * (h as usize);
        if data.len() != expected {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("page {i}: data length {} != w*h {}", data.len(), expected)));
        }
        let arr = Array2::from_shape_vec((h as usize, w as usize), data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("ndarray shape: {e}")))?;
        imgs.push(arr);
    }

    let blob = py.allow_threads(|| encode_document(&imgs, &cfg))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("encode_document: {e:?}")))?;
    Ok(PyBytes::new_bound(py, &blob))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_page, m)?)?;
    m.add_function(wrap_pyfunction!(encode_chunk, m)?)?;
    Ok(())
}
