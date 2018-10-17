import torch
import torch.optim as optim
import torch.utils.data.dataset
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
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

        self.posebyte = posebyte_valtest[:1919]
        self.embedding = np.load('../image/hamming/embeddings/embeddings_valtest_0.npy')[:1919]

    def __getitem__(self, index):
        return self.load_batch(index)

    def __len__(self):
        return self.posebyte.shape[0]

    def load_batch(self, iter):
        return self.posebyte[iter], self.embedding[iter]


def draw_plot(train_losses, val_losses, iter_display):
    x = np.array(range(0, len(train_losses))) * iter_display

    fig, ax = plt.subplots()
    ax.grid(True)
    # ax.set_ylim([0.0, 0.1])

    plt.plot(x, train_losses, label="trn")
    plt.plot(x, val_losses, label="val")
    plt.legend()
    plt.savefig(results_path+'_lr_'+str(learning_rate)+'_loss.png')
    plt.close()


### Training ###

def train(epoch):
    #
    # TRAIN
    #

    train_loss_acc = 0.0

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        batch = data.numpy()

        optimizer.zero_grad()
        input_data = Variable(torch.squeeze(torch.from_numpy(batch)).float().cuda())
        input_target = Variable(torch.squeeze(target).float().cuda())
        loss, l2_norm = model(input_data, input_target)
        loss.backward()
        optimizer.step()

        train_loss_acc += float(loss.data)

    train_loss_acc /= (batch_idx+1)

    #
    # VAL
    #

    val_loss_acc = 0.0

    model.eval()

    for batch_idx, (data, target) in enumerate(val_loader):
        batch = data.numpy()

        input_data = Variable(torch.squeeze(torch.from_numpy(batch)).float().cuda())
        input_target = Variable(torch.squeeze(target).float().cuda())
        loss, l2_norm = model(input_data, input_target)

        val_loss_acc += float(loss.data)

    val_loss_acc /= (batch_idx + 1)

    #
    # Wrap up epoch
    #

    global train_losses
    global val_losses
    train_losses += [train_loss_acc]
    val_losses += [val_loss_acc]

    draw_plot(train_losses, val_losses, 1)

    torch.save(model.state_dict(), results_path + 'thinslicing2_' + str(epoch) + '.pth')


if __name__ == '__main__':
    # Model

    posebit_count = posebyte_train.shape[1]

    model = Net(posebit_count=posebit_count)
    model.cuda()

    state = model.state_dict()

    state_dict = torch.load('../image/hamming/results/model_6.pth')
    state_dict_embedding_only = dict()
    state_dict_embedding_only['fc3.weight'] = state_dict['fc7.weight']
    state_dict_embedding_only['fc3.bias'] = state_dict['fc7.bias']
    state_dict_embedding_only['fc2.weight'] = state_dict['fc6.weight']
    state_dict_embedding_only['fc2.bias'] = state_dict['fc6.bias']
    state.update(state_dict_embedding_only)

    model.load_state_dict(state)

    # Params

    results_path = 'results/'

    train_losses = []
    val_losses = []

    # Run

    train_dataset = ThinSlicingTrainset()
    val_dataset = ThinSlicingValset()

    learning_rate = 0.083895513

    for epoch in range(0, 21):
        optimizer = optim.SGD([
            {'params': model.fc1.parameters(), 'lr': learning_rate, 'momentum': 0.9, 'weight_decay': 0.0005, 'nesterov': True},
            {'params': model.fc2.parameters(), 'lr': 0.0, 'momentum': 0.9, 'weight_decay': 0.0005, 'nesterov': True},
            {'params': model.fc3.parameters(), 'lr': 0.0, 'momentum': 0.9, 'weight_decay': 0.0005, 'nesterov': True},
        ])

        train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=1, shuffle=True, batch_size=128, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=1, shuffle=True, batch_size=128, drop_last=True)

        train(epoch)
