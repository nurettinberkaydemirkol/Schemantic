pub fn chunked_means(v: &[f32], chunks: usize) -> Vec<f32> {
    let chunk_size = v.len() / chunks;
    (0..chunks).map(|i| {
        let start = i * chunk_size;
        let end = if i == chunks - 1 {v.len()} else {start + chunk_size};
        let slice = &v[start..end];
        slice.iter().sum::<f32>() / slice.len() as f32
    }).collect()
}

pub fn l2_distance_vec(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

pub fn mean_vector(vectors: &[Vec<f32>]) -> Vec<f32> {
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