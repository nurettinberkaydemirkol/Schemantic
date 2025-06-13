use crate::cluster::knn_cluster::l2_distance;
use crate::types::Record;
use crate::utils::chunked_means;

fn mean_vector(vectors: &[Vec<f32>]) -> Vec<f32> {
    let len = vectors[0].len();
    let mut mean = vec![0.0; len];
    for vec in vectors {
        for i in 0..len {
            mean[i] += vec[i];
        }
    }
    for x in &mut mean {
        *x /= vectors.len() as f32;
    }
    mean
}

pub fn l2_cluster(records: &[Record], k: usize) -> Vec<Vec<usize>> {
    let mut data: Vec<(usize, Vec<f32>)> = records
        .iter()
        .map(|r| (r.id, chunked_means(&r.embed, 4)))
        .collect();

    let centroid = mean_vector(&data.iter().map(|(_, v)| v.clone()).collect::<Vec<_>>());

    data.sort_by(|a, b| {
        l2_distance(&a.1, &centroid)
            .partial_cmp(&l2_distance(&b.1, &centroid))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let chunk_size = (data.len() + k - 1) / k;
    let mut clusters = vec![vec![]; k];
    for (i, (id, _)) in data.into_iter().enumerate() {
        clusters[i / chunk_size].push(id);
    }

    clusters
}