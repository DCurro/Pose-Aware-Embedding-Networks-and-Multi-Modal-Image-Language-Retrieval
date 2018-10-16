from scipy.spatial.distance import cdist
import numpy as np
from path_manager import PathManager


embedding_path = '../embeddings/embeddings_0.npy'
embeddings = np.load(embedding_path)

gt_distances = np.load(PathManager.path_distance_frame0_path_procrustes_valtest)

embedding_val = embeddings[:1919]
embedding_tst = embeddings[1919:]
distances = cdist(embedding_val, embedding_tst)
nearest_indices = np.argsort(distances, axis=1)

gt_distances_val = gt_distances[:1919,1919:]
gt_near_indices = np.argsort(gt_distances_val, axis=1)

chances = 10

indexes_at_k = np.zeros(shape=(embedding_val.shape[0],chances))
distances_at_k = np.zeros(shape=(embedding_val.shape[0],chances))
chance_indexes_at_k = np.zeros(shape=(embedding_val.shape[0],chances))
chance_distances_at_k = np.zeros(shape=(embedding_val.shape[0],chances))


for val_idx in range(embedding_val.shape[0]):
    # if val_idx%100:
    #     print val_idx

    for k in range(chances):
        gt_best = gt_near_indices[val_idx, k]
        index = np.argwhere(nearest_indices[val_idx,:] == gt_best)
        # index = k # ORALE

        gt_chance = np.random.randint(embedding_tst.shape[0])
        chance_index = np.argwhere(nearest_indices[val_idx, :] == gt_chance)

        indexes_at_k[val_idx, k] = index
        distances_at_k[val_idx, k] = gt_distances_val[val_idx, gt_near_indices[val_idx, index]]
        chance_indexes_at_k[val_idx, k] = chance_index
        chance_distances_at_k[val_idx, k] = gt_distances_val[val_idx, gt_near_indices[val_idx, chance_index]]


best_at_k = np.minimum.accumulate(indexes_at_k, axis=1)
worst_at_k = np.maximum.accumulate(indexes_at_k, axis=1)
mean_at_k = np.zeros(shape=(embedding_val.shape[0],chances))
for k in range(chances):
    mean_at_k[:,k] = np.mean(indexes_at_k[:,:k+1], axis=1)

mean_best_at_k = np.mean(best_at_k, axis=0)
mean_mean_at_k = np.mean(mean_at_k, axis=0)
mean_worst_at_k = np.mean(worst_at_k, axis=0)

chance_best_at_k = np.minimum.accumulate(chance_indexes_at_k, axis=1)
chance_worst_at_k = np.maximum.accumulate(chance_indexes_at_k, axis=1)
chance_mean_at_k = np.zeros(shape=(embedding_val.shape[0],chances))
for k in range(chances):
    chance_mean_at_k[:,k] = np.mean(chance_indexes_at_k[:,:k+1], axis=1)

chance_mean_best_at_k = np.mean(chance_best_at_k, axis=0)
chance_mean_mean_at_k = np.mean(chance_mean_at_k, axis=0)
chance_mean_worst_at_k = np.mean(chance_worst_at_k, axis=0)

print('mean distance@k:' + str(np.round(np.mean(distances_at_k,axis=0),decimals=2)) )
print('random chance:' + str(np.round(np.mean(chance_distances_at_k,axis=0),decimals=2)) )
print('')
print('hit@k: ' + str(np.round(100.0*np.mean(best_at_k <= 50, axis=0),decimals=2)) + '%')
print('random chance: ' + str(np.round(100.0*np.mean(chance_best_at_k <= 50, axis=0),decimals=2)) + '%')
