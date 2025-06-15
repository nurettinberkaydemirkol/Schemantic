use crate::types::Record;
use crate::utils::chunked_means;
use crate::utils::mean_vector;

fn l1_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

pub fn l1_cluster(records: &[Record], k: usize) -> Vec<Vec<usize>> {
    let mut data: Vec<(usize, Vec<f32>)> = records
        .iter()
        .map(|r| (r.id, chunked_means(&r.embed, 4)))
        .collect();

    let centroid = mean_vector(&data.iter().map(|(_, v)| v.clone()).collect::<Vec<_>>());

    data.sort_by(|a, b| {
        l1_distance(&a.1, &centroid)
            .partial_cmp(&l1_distance(&b.1, &centroid))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let chunk_size = (data.len() + k - 1) / k;
    let mut clusters = vec![Vec::new(); k];
    for (i, (id, _)) in data.into_iter().enumerate() {
        clusters[i / chunk_size].push(id);
    }
    clusters
}
