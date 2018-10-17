import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import scipy.misc
from path_manager import PathManager


embedding_val = np.load('../embeddings/embeddings_valtest_0.npy')[:1919]
embedding_tst = np.load('../../image/hamming/embeddings/embeddings_valtest_0.npy')[1919:]

distances = cdist(embedding_val, embedding_tst)
nearest_indices = np.argsort(distances, axis=1)

root_img_dir = PathManager.path_image_root

sequence_file = PathManager.path_dataset_valtest_txt
with open(sequence_file, 'r') as in_file:
    label_lines = in_file.readlines()
    image_list = [x.strip() for x in label_lines]
    image_list = [[' '.join(x.strip().split(' ')[:-16]) + '/'] + x.strip().split(' ')[-16:] for x in image_list]

for anno_idx in range(embedding_val.shape[0]):
    nearest = nearest_indices[anno_idx]

    print(anno_idx)

    fig = plt.figure(figsize=(15, 5), tight_layout=True)

    for frame_idx in range(6):

        if frame_idx == 0:
            image_name = root_img_dir + image_list[anno_idx][0] + image_list[anno_idx][1].split('_')[1] + '.png'
            near_idx = anno_idx
            title = 'Query (language equivalent)'
        else:
            near_idx = nearest[frame_idx-1] + 1919
            image_name = root_img_dir + image_list[near_idx][0] + image_list[near_idx][1].split('_')[1] + '.png'
            title = 'retrieval ' + str(frame_idx)

        fig.add_subplot(1, 6, frame_idx + 1)
        image_to_show = scipy.misc.imread(image_name)
        plt.imshow(imresize(image_to_show, (288, 288)))
        plt.axis('off')
        plt.title(title)

    plt.show()