mod types;
mod utils;
mod cluster;
mod query;
mod index;
pub mod helpers;

use pyo3::prelude::*;
use crate::index::VectorCube;
use crate::helpers::same_search::same_search;

#[pymodule]
fn schemantic(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<VectorCube>()?;
    m.add_function(wrap_pyfunction!(same_search, m)?)?;
    Ok(())
}