use std::collections::HashMap;
use crate::utils::{chunked_means, l2_distance_vec};

pub fn find_closest_column_vec(
    columns: &[Vec<usize>],
    id_to_vecmean: &HashMap<usize, Vec<f32>>,
    query_embed: &[f32]
) -> usize {
    let query_vec = chunked_means(query_embed, 4);

    let mut best = (f32::MAX, 0);
    for (i, col) in columns.iter().enumerate() {
        let col_avg: Vec<f32> = {
            let mut acc = vec![0.0; 4];
            for id in col {
                let vec = &id_to_vecmean[id];
                for (j, val) in vec.iter().enumerate() {
                    acc[j] += val;
                }
            }
            acc.iter_mut().for_each(|x| *x /= col.len() as f32);
            acc
        };

        let dist = l2_distance_vec(&col_avg, &query_vec);
        if dist < best.0 {
            best = (dist, i);
        }
    }

    best.1
}

pub fn find_closest_column_l2(
    columns: &[Vec<usize>],
    id_to_vecmean: &HashMap<usize, Vec<f32>>,
    query_embed: &[f32]
) -> usize {
    let query_vec = chunked_means(query_embed, 4);

    let mut best = (f32::MAX, 0, 0); 

    for (i, col) in columns.iter().enumerate() {
        for id in col {
            let dist = l2_distance_vec(&id_to_vecmean[id], &query_vec);
            if dist < best.0 {
                best = (dist, *id, i);
            }
        }
    }

    best.2
}

pub fn find_closest_column_knn(
    columns: &[Vec<usize>],
    id_to_vecmean: &HashMap<usize, Vec<f32>>,
    query_embed: &[f32],
    k: usize,
) -> usize {
    let query_vec = chunked_means(query_embed, 4);
    let mut all_dists: Vec<(usize, usize, f32)> = vec![]; 

    for (i, col) in columns.iter().enumerate() {
        for id in col {
            let dist = l2_distance_vec(&id_to_vecmean[id], &query_vec);
            all_dists.push((*id, i, dist));
        }
    }

    all_dists.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    let k_nearest = &all_dists[..k.min(all_dists.len())];

    let mut count: HashMap<usize, usize> = HashMap::new();
    for (_, col_idx, _) in k_nearest {
        *count.entry(*col_idx).or_default() += 1;
    }

    count.into_iter().max_by_key(|(_, v)| *v).map(|(k, _)| k).unwrap_or(0)
}