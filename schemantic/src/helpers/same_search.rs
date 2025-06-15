use std::collections::HashMap;
use crate::cluster::mean_cluster::mean_cluster;
use pyo3::prelude::*;
use pyo3::types::PyAny;

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[pyfunction]
pub fn same_search(
    py_list: &PyAny,
    threshold: f32,
    brute_force: bool,
) -> PyResult<Vec<((usize, Vec<f32>, String), (usize, Vec<f32>, String))>> {
    let records: Vec<(usize, Vec<f32>, String)> = py_list.extract()?;

    let id_to_embed: HashMap<usize, Vec<f32>> =
        records.iter().map(|(id, embed, _)| (*id, embed.clone())).collect();

    let mut results = Vec::new();

    if brute_force {
        for i in 0..records.len() {
            for j in (i + 1)..records.len() {
                let (id1, _, _) = &records[i];
                let (id2, _, _) = &records[j];
                let sim = cosine_similarity(&id_to_embed[id1], &id_to_embed[id2]);
                if sim > threshold {
                    results.push((records[i].clone(), records[j].clone()));
                }
            }
        }
    } else {
        let wrapped: Vec<crate::types::Record> = records
            .iter()
            .map(|(id, embed, string)| crate::types::Record {
                id: *id,
                embed: embed.clone(),
                string: string.clone(),
            })
            .collect();

        let clusters = mean_cluster(&wrapped, 5);

        for cluster in clusters {
            for i in 0..cluster.len() {
                for j in (i + 1)..cluster.len() {
                    let id1 = cluster[i];
                    let id2 = cluster[j];
                    let sim = cosine_similarity(&id_to_embed[&id1], &id_to_embed[&id2]);
                    if sim > threshold {
                        let r1 = records.iter().find(|(id, _, _)| *id == id1).unwrap();
                        let r2 = records.iter().find(|(id, _, _)| *id == id2).unwrap();
                        results.push((r1.clone(), r2.clone()));
                    }
                }
            }
        }
    }

    Ok(results)
}
