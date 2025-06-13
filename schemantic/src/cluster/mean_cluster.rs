use crate::types::Record;
use crate::utils::chunked_means;

pub fn mean_cluster(records: &[Record], k: usize) -> Vec<Vec<usize>> {
    let mut data: Vec<(usize, Vec<f32>)> = records.iter().map(|r| (r.id, chunked_means(&r.embed, 4))).collect();

    data.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let chunk_size = (data.len() + k - 1) / k;

    let mut clusters = vec![vec![]; k];
    for (i, (id, _)) in data.into_iter().enumerate() {
        clusters[i / chunk_size].push(id);
    }

    clusters
}