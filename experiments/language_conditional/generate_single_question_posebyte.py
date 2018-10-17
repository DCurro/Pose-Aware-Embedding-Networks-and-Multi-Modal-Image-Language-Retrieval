import numpy as np
from numpy import linalg as np_lin
from scipy import delete
from path_manager import PathManager


def condtionMultivariateNormal(mean_vec, covariance_mat, known_quanitity_index, known_quantity_value):
    C_aa = delete(delete(covariance_mat, known_quanitity_index, axis=1), known_quanitity_index, axis=0)
    C_ab = delete(covariance_mat[known_quanitity_index,:], known_quanitity_index)
    C_ba = delete(covariance_mat[:, known_quanitity_index], known_quanitity_index)
    C_bb = covariance_mat[known_quanitity_index,known_quanitity_index]

    u_a = delete(mean_vec, known_quanitity_index)
    u_b = mean_vec[known_quanitity_index]

    # new_mean_vec = u_a + C_ab*np_lin.inv(C_bb)*(known_quantity_value - u_b)
    new_mean_vec = u_a + C_ab*(1.0/C_bb)*(known_quantity_value - u_b)

    return new_mean_vec


if __name__ == '__main__':
    angles_train = np.load(PathManager.path_annotations_hamming_train_angle)
    distances_train = np.load(PathManager.path_annotations_hamming_train_distance)
    reldistances_train = np.load(PathManager.path_annotations_hamming_train_reldistance)
    posebyte_train = np.concatenate((angles_train,
                                     distances_train,
                                     reldistances_train,), axis=1)

    gt_trn_relevent = posebyte_train

    mean_vec = np.mean(gt_trn_relevent, axis=0)
    std_vec = np.std(gt_trn_relevent, axis=0)

    gt_trn_prime = gt_trn_relevent - mean_vec

    covariance_mat = np.dot(gt_trn_prime.T, gt_trn_prime)/gt_trn_prime.shape[0]
    np_lin.inv(covariance_mat)

    is_high = mean_vec + 4.0*std_vec
    is_low = mean_vec - 4.0*std_vec

    mean_posebyte_one_hot = np.zeros(shape=(gt_trn_relevent.shape[1] * 2, gt_trn_relevent.shape[1]))

    valid_data_count = 0

    for bit_idx in range(gt_trn_relevent.shape[1]):
        valid_data_count += 2

        first_idx = valid_data_count - 2
        second_idx = valid_data_count - 2 + 1

        conditioned_high = condtionMultivariateNormal(mean_vec, covariance_mat, known_quanitity_index=first_idx//2, known_quantity_value=1.0)
        conditioned_high = np.insert(conditioned_high, first_idx//2, [1.0])

        conditioned_low = condtionMultivariateNormal(mean_vec, covariance_mat, known_quanitity_index=first_idx//2, known_quantity_value=0.0)
        conditioned_low = np.insert(conditioned_low, first_idx//2, [0.0])

        mean_posebyte_one_hot[first_idx,:] = conditioned_high
        mean_posebyte_one_hot[second_idx,:] = conditioned_low

    np.save('posebytes/posebyte_conditioned.npy', np.clip(np.round(mean_posebyte_one_hot),0.0,1.0))
