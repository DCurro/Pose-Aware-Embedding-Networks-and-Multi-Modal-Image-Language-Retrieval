import torch.utils.data.dataset
import numpy as np
from nets.net_poseemb_mask import Net
from path_manager import PathManager


angles_train = np.load(PathManager.path_annotations_hamming_train_angle)
distances_train = np.load(PathManager.path_annotations_hamming_train_distance)
reldistances_train = np.load(PathManager.path_annotations_hamming_train_reldistance)
posebyte_train = np.concatenate((angles_train,
                                 distances_train,
                                 reldistances_train,), axis=1)

results_path = 'results/'

for bit in range(posebyte_train.shape[1]):
    for bit_value in [1,0]:
        posebit_count = posebyte_train.shape[1]
        model = Net(posebit_count=posebit_count)
        model.cuda()
        state = model.state_dict()
        try:
            state_dict = torch.load(results_path + 'bit_'+str(bit)+'_value_'+str(bit_value)+'_model_9.pth')
        except:
            continue
        state.update(state_dict)
        model.load_state_dict(state)

        np.save('masks/mask_'+str(bit)+'_'+str(bit_value)+'.npy', state['masklayer.mask'].cpu().numpy()[0])


