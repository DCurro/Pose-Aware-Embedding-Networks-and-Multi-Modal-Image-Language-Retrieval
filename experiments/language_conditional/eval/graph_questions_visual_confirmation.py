from matplotlib import pyplot as plt
from scipy.misc import imread
import numpy as np
from scipy.spatial.distance import cdist
from scipy.misc import imread, imresize
from glob import glob
from path_manager import PathManager


def graph():
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

    #
    # Display
    #

    output_path = 'predictions/'

    root_img_dir = PathManager.path_image_root
    sequence_file = PathManager.path_dataset_valtest_txt
    with open(sequence_file, 'r') as in_file:
        label_lines = in_file.readlines()
        image_list = [x.strip() for x in label_lines]
        image_list = [[' '.join(x.strip().split(' ')[:-16]) + '/'] + x.strip().split(' ')[-16:] for x in image_list]
    image_list = image_list[1919:]

    for anno_idx, anno in enumerate(embedding_conditional):
        question_idx = int(anno_idx/2)
        answer = posebyte_conditional[anno_idx, question_idx]

        if question_idx in valid_bits:
            pass
        else:
            continue

        answer = bool(answer)
        question = str(question_idx)+': '+str(questions[question_idx])
        question = question.replace('angle:', 'is bent:')
        question = question.replace('distance:', 'is near:')
        question = question.replace('beyond:', 'is beyond:')
        question = question+'? '+str(answer)

        output_file_name = output_path + question + '.png'

        nearest = nearest_indices[anno_idx]

        fig = plt.figure()
        fig.set_size_inches(8.0, 8.0)

        for frame_idx in range(25):
            near_idx = nearest[frame_idx]
            image_name = root_img_dir + image_list[near_idx][0] + image_list[near_idx][1].split('_')[1] + '.png'

            axes = fig.add_subplot(5, 5, frame_idx + 1)

            if posebyte_valtest[near_idx, question_idx] == answer:
                for spine in axes.spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(8)
            else:
                for spine in axes.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(8)

            image_to_show = imread(image_name)
            plt.suptitle(question, fontsize=16)
            plt.imshow(imresize(image_to_show, (288, 288)))
            plt.setp(axes.get_xticklabels(), visible=False)
            plt.setp(axes.get_yticklabels(), visible=False)

        plt.show()


if __name__ == '__main__':
    graph()
