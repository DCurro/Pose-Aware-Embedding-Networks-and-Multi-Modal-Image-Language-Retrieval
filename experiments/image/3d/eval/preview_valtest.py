import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
from other.visualize_3d import visualize_3d
from scipy.spatial.distance import cdist
from path_manager import PathManager
import scipy.misc


embedding_path = '../embeddings/embeddings_0.npy'
embeddings = np.load(embedding_path)

embedding_val = embeddings[:1919]
embedding_tst = embeddings[1919:]
distances = cdist(embedding_val, embedding_tst)
nearest_indices = np.argsort(distances, axis=1)

root_img_dir = PathManager.path_image_root
sequence_file = PathManager.path_dataset_valtest_txt
with open(sequence_file, 'r') as in_file:
    label_lines = in_file.readlines()
    image_list = [x.strip() for x in label_lines]
    image_list = [[' '.join(x.strip().split(' ')[:-16]) + '/'] + x.strip().split(' ')[-16:] for x in image_list]

annos_valtest = np.load(PathManager.path_annotations_3D_valtest)[:,0,:,:]

for anno_idx, _ in enumerate(annos_valtest):
    nearest = nearest_indices[anno_idx]

    print(anno_idx)

    fig = plt.figure(figsize=(15, 5))
    # fig = plt.figure(figsize=(15, 5), tight_layout=True)

    for frame_idx in range(6):

        if frame_idx == 0:
            image_name = root_img_dir + image_list[anno_idx][0] + image_list[anno_idx][1].split('_')[1] + '.png'
            near_idx = anno_idx
            title = 'Query'
        else:
            near_idx = nearest[frame_idx-1] + 1919
            image_name = root_img_dir + image_list[near_idx][0] + image_list[near_idx][1].split('_')[1] + '.png'
            title = 'retrieval ' + str(frame_idx)

        fig.add_subplot(2, 6, frame_idx + 1)
        image_to_show = scipy.misc.imread(image_name)
        plt.imshow(imresize(image_to_show, (288, 288)))
        plt.axis('off')
        plt.title(title)
        visualize_3d(data=annos_valtest[near_idx], fig=fig, subplot_idx=frame_idx+6+1, subplot_dims=(2,6))

    plt.show()


