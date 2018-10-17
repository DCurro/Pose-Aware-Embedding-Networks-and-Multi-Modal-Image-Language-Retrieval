import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from path_manager import PathManager


reldist_filter = np.load(PathManager.path_questions_hamming_reldistance_keep_bit_idxs)
questions = np.concatenate([np.load(PathManager.path_questions_hamming_angles),
                            np.load(PathManager.path_questions_hamming_distances),
                            np.load(PathManager.path_questions_hamming_reldistances)[reldist_filter]])


angles_val = np.load(PathManager.path_annotations_hamming_valtest_angle)
distances_val = np.load(PathManager.path_annotations_hamming_valtest_distance)
reldistances_val = np.load(PathManager.path_annotations_hamming_valtest_reldistance)
posebyte_test = np.concatenate((angles_val,
                                distances_val,
                                reldistances_val,), axis=1)[1919:]

root_img_dir = PathManager.path_image_root
sequence_file = PathManager.path_dataset_valtest_txt
with open(sequence_file, 'r') as in_file:
    label_lines = in_file.readlines()
    image_list = [x.strip() for x in label_lines]
    image_list = [[' '.join(x.strip().split(' ')[:-16]) + '/'] + x.strip().split(' ')[-16:] for x in image_list]

embedding_tst_raw = np.load('../../language/embeddings/embeddings_valtest_0.npy')[1919:]

for bit in range(posebyte_test.shape[1]):
    for bit_value in [1,0]:

        try:
            mask_a = np.load('../masks/mask_'+str(bit)+'_'+str(bit_value)+'.npy')
        except:
            continue

        embedding_val = np.zeros(shape=(1,embedding_tst_raw.shape[1]))
        embedding_tst = embedding_tst_raw * mask_a
        distances = cdist(embedding_val, embedding_tst)

        nearest_indices = np.argsort(distances, axis=1)

        positive_indices = np.where(posebyte_test[:,bit] == bit_value)[0]

        print(questions[bit])

        positive_skip_size = int(float(posebyte_test.shape[0] - (posebyte_test.shape[0] - positive_indices.shape[0])) / (3+1))
        negative_skip_size = int(float((posebyte_test.shape[0] - positive_indices.shape[0])) / (2))

        for anno_idx in [0]:
            nearest = nearest_indices[anno_idx]

            fig = plt.figure(tight_layout=True)

            for frame_idx in range(5):
                if frame_idx < 3:
                    start = 0
                    skip_size = positive_skip_size
                    nearest_temp = list(nearest[start::skip_size])
                else:
                    start = positive_indices.shape[0]
                    skip_size = negative_skip_size
                    nearest_temp = [-1, -1, -1] + list(nearest[start::skip_size])

                near_idx = nearest_temp[frame_idx] + 1919
                image_name = root_img_dir + image_list[near_idx][0] + image_list[near_idx][1].split('_')[1] + '.png'

                axes = fig.add_subplot(1, 5, frame_idx + 1)

                image_to_show = scipy.misc.imread(image_name)
                plt.imshow(scipy.misc.imresize(image_to_show, (288, 288)))
                plt.setp(axes.get_xticklabels(), visible=False)
                plt.setp(axes.get_yticklabels(), visible=False)
                plt.axis('off')

            plt.suptitle(str(questions[bit]))
            plt.show()