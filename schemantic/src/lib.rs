mod types;
mod utils;
mod mean_cluster;
mod query;
mod index;

use pyo3::prelude::*;
use crate::index::VectorCube;

#[pymodule]
fn schemantic(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<VectorCube>()?;
    Ok(())
}