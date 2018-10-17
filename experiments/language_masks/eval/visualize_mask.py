import matplotlib.pyplot as plt
import numpy as np
from path_manager import PathManager


output_path = 'visualize_masks/'

reldist_filter = np.load(PathManager.path_questions_hamming_reldistance_keep_bit_idxs)
questions = np.concatenate([np.load(PathManager.path_questions_hamming_angles),
                            np.load(PathManager.path_questions_hamming_distances),
                            np.load(PathManager.path_questions_hamming_reldistances)[reldist_filter]])

for mask_bit in range(111):
    try:
        mask_0 = np.load('../masks/mask_' + str(mask_bit) + '_0.npy').clip(0, 1).flatten()
        mask_1 = np.load('../masks/mask_' + str(mask_bit) + '_1.npy').clip(0, 1).flatten()
    except:
        continue

    question = str(mask_bit) + ': ' + str(questions[mask_bit])
    question = question.replace('angle:', 'is bent:')
    question = question.replace('distance:', 'is near:')
    question = question.replace('beyond:', 'is beyond:')
    question = question + '?'

    question = question.replace('->','_')
    question = question.replace('<->','_')
    question = question.replace(' ','_')

    c_0 = np.full((128,), 'b')
    c_0[np.where(mask_0 <= 0.3)[0]] = 'r'

    c_1 = np.full((128,), 'b')
    c_1[np.where(mask_1 <= 0.3)[0]] = 'r'

    # fig = plt.figure(figsize=(10, 5), tight_layout=True)
    fig = plt.figure(figsize=(10, 5))
    # fig = plt.figure(tight_layout=True)
    # fig = plt.figure()

    fig.add_subplot(2, 1, 1)
    plt.axis('off')
    plt.bar(range(mask_0.shape[0]), mask_0, color=c_0, alpha=0.7)

    fig.add_subplot(2, 1, 2)
    plt.axis('off')
    plt.bar(range(mask_1.shape[0]), mask_1, color=c_1, alpha=0.7)

    plt.suptitle(str(question))
    plt.show()
    # plt.savefig(output_path+'mask_'+str(mask_bit)+'_'+question+'.png')
    # plt.close()
