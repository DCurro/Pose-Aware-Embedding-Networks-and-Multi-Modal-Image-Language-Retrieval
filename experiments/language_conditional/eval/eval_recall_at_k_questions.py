import numpy as np
from scipy.spatial.distance import cdist
import math
from glob import glob
from path_manager import PathManager


#
# Pb Filter
#

mask_filenames = glob(PathManager.path_valid_masks+'mask*.npy')
valid_bits = np.sort(np.unique([int(name.replace('\\','/').split('/')[-1].split('_')[1]) for name in mask_filenames]))

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

posebyte_conditional = np.load('../posebytes/posebyte_conditioned.npy')

angles_val = np.load(PathManager.path_annotations_hamming_valtest_angle)
distances_val = np.load(PathManager.path_annotations_hamming_valtest_distance)
reldistances_val = np.load(PathManager.path_annotations_hamming_valtest_reldistance)
posebyte_valtest = np.concatenate((angles_val,
                                   distances_val,
                                   reldistances_val,), axis=1)[1919:]

#
# Embeddings
#

embedding_conditional = np.load('../embeddings/embeddings_conditional.npy')
embedding_test = np.load('../../image/hamming/embeddings/embeddings_valtest_0.npy')[1919:]

#
# Distances
#

distances = cdist(embedding_conditional, embedding_test)
nearest_indices = np.argsort(distances, axis=1)


#############################
## RECALL @ K

distances = cdist(embedding_conditional, embedding_test)
nearest_indices = np.argsort(distances, axis=1)

question_keys = ['angle', 'distance', 'beyond']
# question_keys = ['angle']
# question_keys = ['distance']
# question_keys = ['beyond']
answer_keys = [True, False]
# answer_keys = [True]
# answer_keys = [False]

ks = [1,2,3,4,5,6,7,8,9,10]

for k in ks:
    aps = []

    for question_key in question_keys:
        for answer_key in answer_keys:

            for idx in range(posebyte_conditional.shape[0]):
                bit_idx = int(idx/2)

                if bit_idx in valid_bits:
                    pass
                else:
                    continue

                question = questions[bit_idx]
                answer = posebyte_conditional[idx, bit_idx]

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
                # ap = average_precision_score(y_true=y_true, y_score=y_score)
                ap = np.sum(y_true) / np.sum(y_score)
                if math.isnan(ap):
                    continue
                aps += [ap]

    print(np.mean(aps))
print('')


#############################
## RANDOM RECALL @ K

np.random.shuffle(embedding_test) # for random

distances = cdist(embedding_conditional, embedding_test)
nearest_indices = np.argsort(distances, axis=1)

question_keys = ['angle', 'distance', 'beyond']
# question_keys = ['angle']
# question_keys = ['distance']
# question_keys = ['beyond']
answer_keys = [True, False]
# answer_keys = [True]
# answer_keys = [False]

ks = [1,2,3,4,5,6,7,8,9,10]

for k in ks:
    aps = []

    for question_key in question_keys:
        for answer_key in answer_keys:

            for idx in range(posebyte_conditional.shape[0]):
                bit_idx = int(idx/2)

                question = questions[bit_idx]
                answer = posebyte_conditional[idx, bit_idx]

                # print question, bool(answer)

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
                # ap = average_precision_score(y_true=y_true, y_score=y_score)
                ap = np.sum(y_true) / np.sum(y_score)
                if math.isnan(ap):
                    continue
                aps += [ap]

    print(np.mean(aps))
