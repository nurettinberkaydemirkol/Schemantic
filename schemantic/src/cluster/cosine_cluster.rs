use crate::types::Record;
use crate::utils::mean_vector;

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        1.0
    } else {
        1.0 - (dot / (norm_a * norm_b))
    }
}

pub fn cosine_cluster(records: &[Record], k: usize) -> Vec<Vec<usize>> {
    let mut data: Vec<(usize, Vec<f32>)> = records
        .iter()
        .map(|r| (r.id, r.embed.clone()))
        .collect();

    let centroid = mean_vector(&data.iter().map(|(_, v)| v.clone()).collect::<Vec<_>>());

    data.sort_by(|a, b| {
        cosine_distance(&a.1, &centroid)
            .partial_cmp(&cosine_distance(&b.1, &centroid))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let chunk_size = (data.len() + k - 1) / k;
    let mut clusters = vec![Vec::new(); k];
    for (i, (id, _)) in data.into_iter().enumerate() {
        clusters[i / chunk_size].push(id);
    }
    clusters
}