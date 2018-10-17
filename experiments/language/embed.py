import torch
import torch.utils.data.dataset
from torch.autograd import Variable
import numpy as np
from path_manager import PathManager
from nets.net_poseemb_language import Net


angles_train = np.load(PathManager.path_annotations_hamming_train_angle)
distances_train = np.load(PathManager.path_annotations_hamming_train_distance)
reldistances_train = np.load(PathManager.path_annotations_hamming_train_reldistance)
posebyte_train = np.concatenate((angles_train,
                                distances_train,
                                reldistances_train,), axis=1)

angles_val = np.load(PathManager.path_annotations_hamming_valtest_angle)
distances_val = np.load(PathManager.path_annotations_hamming_valtest_distance)
reldistances_val = np.load(PathManager.path_annotations_hamming_valtest_reldistance)
posebyte_valtest = np.concatenate((angles_val,
                                   distances_val,
                                   reldistances_val,), axis=1)


class ThinSlicingTrainset(torch.utils.data.dataset.Dataset):
    def __init__(self):
        super(ThinSlicingTrainset, self).__init__()

        self.posebyte = posebyte_train
        self.embedding = np.load('../image/hamming/embeddings/embeddings_train_0.npy')

    def __getitem__(self, index):
        return self.load_batch(index)

    def __len__(self):
        return self.posebyte.shape[0]

    def load_batch(self, iter):
        return self.posebyte[iter], self.embedding[iter]


class ThinSlicingValset(torch.utils.data.dataset.Dataset):
    def __init__(self):
        super(ThinSlicingValset, self).__init__()

        self.posebyte = posebyte_valtest
        self.embedding = np.load('../image/hamming/embeddings/embeddings_valtest_0.npy')

    def __getitem__(self, index):
        return self.load_batch(index)

    def __len__(self):
        return self.posebyte.shape[0]

    def load_batch(self, iter):
        return self.posebyte[iter], self.embedding[iter]


def train(frame, suffix):
    model.eval()

    embeddings = []

    for batch_idx, (data, target) in enumerate(val_loader):
        batch = data.numpy()

        input_data = Variable(torch.squeeze(torch.from_numpy(batch)).float().cuda())
        input_target = Variable(torch.squeeze(target).float().cuda())
        loss, l2_norm = model(input_data, input_target)

        embeddings += [l2_norm.data.cpu().numpy()]

    embeddings = np.concatenate(embeddings)

    print(embeddings.shape)

    np.save('embeddings/embeddings_'+suffix+'_'+str(frame)+'.npy', embeddings)


if __name__ == '__main__':
    # Model

    results_path = 'results/'

    posebit_count = posebyte_train.shape[1]

    model = Net(posebit_count=posebit_count)
    model.cuda()

    state = model.state_dict()
    state_dict = torch.load(results_path + 'thinslicing2_20.pth')
    state.update(state_dict)
    model.load_state_dict(state)

    # RUN

    train_dataset = ThinSlicingTrainset()
    val_loader = torch.utils.data.DataLoader(train_dataset, num_workers=1, shuffle=False, batch_size=128, drop_last=False)
    train(frame=0, suffix='train')

    val_dataset = ThinSlicingValset()
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=1, shuffle=False, batch_size=128, drop_last=False)
    train(frame=0, suffix='valtest')

