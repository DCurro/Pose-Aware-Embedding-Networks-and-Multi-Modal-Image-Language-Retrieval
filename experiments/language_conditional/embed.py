import torch
import torch.utils.data.dataset
from torch.autograd import Variable
import numpy as np
from path_manager import PathManager
from nets.net_poseemb_language import Net


class ThinSlicingConditional(torch.utils.data.dataset.Dataset):
    def __init__(self):
        super(ThinSlicingConditional, self).__init__()

        self.posebyte = posebyte_conditional
        self.embedding = np.load('../language/embeddings/embeddings_train_0.npy')

    def __getitem__(self, index):
        return self.load_batch(index)

    def __len__(self):
        return self.posebyte.shape[0]

    def load_batch(self, iter):
        return self.posebyte[iter], self.embedding[iter]


def embed():
    model.eval()

    embeddings = []

    for batch_idx, (data, target) in enumerate(data_loader):
        batch = data.numpy()

        input_data = Variable(torch.squeeze(torch.from_numpy(batch)).float().cuda())
        input_target = Variable(torch.squeeze(target).float().cuda())
        loss, l2_norm = model(input_data, input_target)

        embeddings += [l2_norm.data.cpu().numpy()]

    embeddings = np.concatenate(embeddings)

    print(embeddings.shape)

    np.save('embeddings/embeddings_conditional.npy', embeddings)


if __name__ == '__main__':
    posebyte_conditional = np.load('posebytes/posebyte_conditioned.npy')

    angles_val = np.load(PathManager.path_annotations_hamming_valtest_angle)
    distances_val = np.load(PathManager.path_annotations_hamming_valtest_distance)
    reldistances_val = np.load(PathManager.path_annotations_hamming_valtest_reldistance)
    posebyte_valtest = np.concatenate((angles_val,
                                       distances_val,
                                       reldistances_val,), axis=1)

    # Model

    posebit_count = posebyte_conditional.shape[1]

    model = Net(posebit_count=posebit_count)
    model.cuda()

    state = model.state_dict()
    state_dict = torch.load('../language/results/'+'thinslicing2_20.pth')
    state.update(state_dict)
    model.load_state_dict(state)

    # RUN

    conditional_dataset = ThinSlicingConditional()
    data_loader = torch.utils.data.DataLoader(conditional_dataset, num_workers=1, shuffle=False, batch_size=128, drop_last=False)
    embed()

