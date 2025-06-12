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