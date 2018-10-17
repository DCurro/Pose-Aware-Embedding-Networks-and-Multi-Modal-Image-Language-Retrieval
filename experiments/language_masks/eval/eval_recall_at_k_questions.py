import numpy as np
from scipy.spatial.distance import cdist
import math
from path_manager import PathManager


#
# Questions
#

reldist_filter = np.load(PathManager.path_questions_hamming_reldistance_keep_bit_idxs)
questions = np.concatenate([np.load(PathManager.path_questions_hamming_angles),
                            np.load(PathManager.path_questions_hamming_distances),
                            np.load(PathManager.path_questions_hamming_reldistances)[reldist_filter]])

#
# Posebyte
#

angles_val = np.load(PathManager.path_annotations_hamming_valtest_angle)
distances_val = np.load(PathManager.path_annotations_hamming_valtest_distance)
reldistances_val = np.load(PathManager.path_annotations_hamming_valtest_reldistance)
posebyte_valtest = np.concatenate((angles_val,
                                   distances_val,
                                   reldistances_val,), axis=1)[1919:]

angles_val = np.load(PathManager.path_annotations_hamming_valtest_angle)
distances_val = np.load(PathManager.path_annotations_hamming_valtest_distance)
reldistances_val = np.load(PathManager.path_annotations_hamming_valtest_reldistance)
posebyte_val = np.concatenate((angles_val,
                               distances_val,
                               reldistances_val,), axis=1)[:1919]

#
# Embeddings
#


embedding_val_raw = np.load('../../language/embeddings/embeddings_valtest_0.npy')[:1919]
embedding_tst_raw = np.load('../../image/hamming/embeddings/embeddings_valtest_0.npy')[1919:]

embedding_query = np.zeros(shape=(1,embedding_tst_raw.shape[1]))

distances_raw = cdist(embedding_val_raw, embedding_tst_raw)
nearest_raw = np.argsort(distances_raw, axis=1)

embedding_val_vggs = np.load('../../image/vggs/embeddings/embeddings_valtest_0.npy')[:1919]
embedding_tst_vggs = np.load('../../image/vggs/embeddings/embeddings_valtest_0.npy')[1919:]
distances_vggs = cdist(embedding_val_vggs, embedding_tst_vggs)
nearest_vggs = np.argsort(distances_vggs, axis=1)

#
# Evaluation Criteria
#

question_keys = ['angle', 'distance', 'beyond']
answer_keys = [True, False]
ks = [1,2,3,4,5,6,7,8,9,10]

#
# Masks: RECALL @ K
#

aps_at_k = []
for k in ks:
    aps = []

    for bit in range(embedding_tst_raw.shape[1]):
        for bit_value in [1, 0]:

            try:
                mask = np.load('../masks/mask_' + str(bit) + '_' + str(bit_value) + '.npy')
            except:
                continue

            embedding_tst = embedding_tst_raw * mask

            distances = cdist(embedding_query, embedding_tst)
            nearest_indices = np.argsort(distances, axis=1)

            question = questions[bit]
            answer = bit_value

            near_indices = nearest_indices[0, :][0:k]

            predictions = posebyte_valtest[near_indices, bit] == answer

            y_score = np.ones(shape=(k,))
            y_true = predictions
            ap = np.sum(y_true) / np.sum(y_score)
            aps += [ap]

    aps_at_k += [np.mean(aps)]
print('masks: ' + str((100.0*np.array(aps_at_k)).round(1)))

#
# Conditional Posebytes: RECALL @ K
#

posebyte_conditional = np.load('../../language_conditional/posebytes/posebyte_conditioned.npy')
embedding_conditional = np.load('../../language_conditional/embeddings/embeddings_conditional.npy')
distances = cdist(embedding_conditional, embedding_tst_raw)
nearest_indices = np.argsort(distances, axis=1)

aps_at_k = []
for k in ks:
    aps = []

    for question_key in question_keys:
        for answer_key in answer_keys:

            for idx in range(posebyte_conditional.shape[0]):
                bit_idx = int(idx/2)

                question = questions[bit_idx]
                answer = posebyte_conditional[idx, bit_idx]

                try:
                    mask = np.load('../masks/mask_' + str(bit_idx) + '_' + str(int(answer)) + '.npy')
                except:
                    continue

                if question_key in str(question):
                    pass
                else:
                    continue

                if answer_key == answer:
                    pass
                else:
                    continue

                near_indices = nearest_indices[idx, :][0:k]

                predictions = posebyte_valtest[near_indices, bit_idx] == answer

                y_score = np.ones(shape=(k,))
                y_true = predictions
                ap = np.sum(y_true) / np.sum(y_score)
                if math.isnan(ap):
                    continue
                aps += [ap]

    aps_at_k += [np.mean(aps)]
print('posebyte: ' + str((100.0 * np.array(aps_at_k)).round(1)))

#
# Mean Query
#

aps_at_k = []
for k in ks:
    aps = []

    for bit in range(embedding_tst_raw.shape[1]):
        # print bit

        for bit_value in [1, 0]:

            try:
                mask = np.load('../masks/mask_' + str(bit) + '_' + str(bit_value) + '.npy')
            except:
                continue

            relevant_indices = np.where(posebyte_val[:,bit] == bit_value)[0]

            nearest_indices_list = nearest_raw[relevant_indices]

            question = questions[bit]
            answer = bit_value

            near_indices = nearest_indices_list[:,0:k]

            predictions = posebyte_valtest[near_indices, bit] == answer

            y_score = np.ones(shape=(k,))
            y_true = predictions
            ap = np.sum(y_true,axis=1) / np.sum(y_score)
            aps += [np.mean(ap)]

    aps_at_k += [np.mean(aps)]
print('mean query: ' + str((100.0 * np.array(aps_at_k)).round(1)))

#
# VGG-S
#

aps_at_k = []
for k in ks:
    aps = []

    for bit in range(embedding_tst_raw.shape[1]):
        for bit_value in [1, 0]:

            try:
                mask = np.load('../masks/mask_' + str(bit) + '_' + str(bit_value) + '.npy')
            except:
                continue

            relevant_indices = np.where(posebyte_val[:,bit] == bit_value)[0]

            nearest_indices_list = nearest_vggs[relevant_indices]

            question = questions[bit]
            answer = bit_value

            near_indices = nearest_indices_list[:,0:k]

            predictions = posebyte_valtest[near_indices, bit] == answer

            y_score = np.ones(shape=(k,))
            y_true = predictions
            ap = np.sum(y_true,axis=1) / np.sum(y_score)
            aps += [np.mean(ap)]

    aps_at_k += [np.mean(aps)]
print('vgg-s: ' + str((100.0*np.array(aps_at_k)).round(1)))

#
# Chance
#

np.random.shuffle(embedding_tst_raw) # for random

aps_at_k = []
for k in ks:
    aps = []

    for bit in range(embedding_tst_raw.shape[1]):
        for bit_value in [1, 0]:

            try:
                mask = np.load('../masks/mask_' + str(bit) + '_' + str(bit_value) + '.npy')
            except:
                continue

            embedding_tst = embedding_tst_raw * mask

            distances = cdist(embedding_query, embedding_tst)
            nearest_indices = np.argsort(distances, axis=1)

            question = questions[bit]
            answer = bit_value

            near_indices = nearest_indices[0, :][0:k]

            predictions = posebyte_valtest[near_indices, bit] == answer

            y_score = np.ones(shape=(k,))
            y_true = predictions
            ap = np.sum(y_true) / np.sum(y_score)
            aps += [ap]

    aps_at_k += [np.mean(aps)]
print('chance: ' + str((100.0 * np.array(aps_at_k)).round(1)))
