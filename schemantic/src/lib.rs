use pyo3::prelude::*;

#[pyfunction]
fn hello(name: &str) -> String {
    format!("Merhaba, {}!", name)
}

#[pymodule]
fn schemantic(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    Ok(())
}
