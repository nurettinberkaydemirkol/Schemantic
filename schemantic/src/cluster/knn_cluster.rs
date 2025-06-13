use crate::types::Record;
use crate::utils::chunked_means;


fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a,b)| (a-b).powi(2)).sum()
}

pub fn knn_cluster(records: &[Records], k: usize) -> Vec<Vec<usize>> {
    let data: Vec<(usize, Vec<f32>)> = records
        .iter()
        .map(|r| (r.id, chunked_means(&r.embed, 4)))
        .collect();

    let centers: Vec<Vec<f32>> = data.iter().take(k).map(|(_, v)| v.clone()).collect();

    let mut clusters = vec![vec![]; k];

    for (id, vec) in data {
        let mut best_center = 0;
        let mut best_dist = l2_distance(&vec, &centers[0]);

        for (i, center) in centers.iter().enumerate().skip(1) {
            let dist = l2_distance(&vec, center);
            if dist < best_dist {
                best_center = i;
                best_dist = dist;
            }
        }

        clusters[best_center].push(id);
    }

    clusters
}