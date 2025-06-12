use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::collections::HashMap;
use crate::types::Record;
use crate::utils::chunked_means;
use crate::mean_cluster::cluster_by_chunked_means;
use crate::query::find_closest_column_vec;

#[pyclass]
pub struct VectorCube {
    records: Vec<Record>,
    columns: Vec<Vec<usize>>,
    id_to_string: HashMap<usize, String>,
}

#[pymethods]
impl VectorCube {
    #[new]
    fn new(py_list: &PyList) -> Self {
        let records: Vec<Record> = py_list.iter().enumerate().map(|(i, item)| {
            let tup: &PyTuple = item.downcast().unwrap();
            let embed: Vec<f32> = tup.get_item(1).unwrap().extract().unwrap();
            let string: String = tup.get_item(2).unwrap().extract().unwrap();
            Record { id: i, embed, string }
        }).collect();

        let mut id_to_string = HashMap::new();
        for r in &records {
            id_to_string.insert(r.id, r.string.clone());
        }

        let columns = cluster_by_chunked_means(&records, 5);
        Self { records, columns, id_to_string }
    }

    fn query(&self, query_embed: Vec<f32>) -> Vec<String> {
        let id_to_meanvec: HashMap<usize, Vec<f32>> = self
            .records
            .iter()
            .map(|r| (r.id, chunked_means(&r.embed, 4)))
            .collect();

        let col_idx = find_closest_column_vec(&self.columns, &id_to_meanvec, &query_embed);
        self.columns[col_idx]
            .iter()
            .map(|id| self.id_to_string[id].clone())
            .collect()
    }
}
