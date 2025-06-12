use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct Record {
    #[pyo3(get)]
    pub id: usize,
    pub embed: Vec<f32>,
    pub string: String,
}