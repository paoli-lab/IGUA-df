extern crate kodama;
extern crate pyo3;
extern crate numpy;
extern crate rayon;

use pyo3::prelude::*;

mod clustering;
mod distance;

/// A Python module for metagenomic sequence comparison with ``skani``.
///
#[pymodule]
#[pyo3(name = "hca")]
pub fn init(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__package__", "htgcf")?;
    m.add_function(wrap_pyfunction!(distance::manhattan, m)?)?;
    m.add_function(wrap_pyfunction!(clustering::linkage, m)?)?;
    Ok(())
}
