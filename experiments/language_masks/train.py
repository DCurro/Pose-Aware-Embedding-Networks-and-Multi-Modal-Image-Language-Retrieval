import torch
import torch.optim as optim
import torch.utils.data.dataset
from torch.autograd import Variable
import numpy as np
import random
import matplotlib.pyplot as plt
from path_manager import PathManager
from nets.net_poseemb_mask import Net


if __name__ == '__main__':
    def train_network(bit, bit_value):

        class ThinSlicingTrainset(torch.utils.data.dataset.Dataset):
            def __init__(self, bit, bit_value):
                super(ThinSlicingTrainset, self).__init__()

                self.bit = bit
                self.bit_value = bit_value

                self.posebyte = posebyte_train

            def __getitem__(self, index):
                return self.load_batch(index)

            def __len__(self):
                return self.posebyte.shape[0]

            def load_batch(self, iter):
                positives = np.where(self.posebyte[:, self.bit] == self.bit_value)[0]
                negatives = np.where(self.posebyte[:, self.bit] != self.bit_value)[0]

                shuffled = lambda seq, rnd=random.random: sorted(seq, key=lambda _: rnd())

                selections = shuffled(positives)[0:55] + shuffled(negatives)[0:56]
                # selections = positives[0:55].tolist() + negatives[0:56].tolist()

                posebytes = []

                for selection in selections:
                    posebytes += [self.posebyte[selection]]

                posebytes = torch.from_numpy(np.vstack(posebytes))

                return posebytes

        class ThinSlicingValset(torch.utils.data.dataset.Dataset):
            def __init__(self, bit, bit_value):
                super(ThinSlicingValset, self).__init__()

                self.bit = bit
                self.bit_value = bit_value

                self.posebyte = posebyte_valtest

            def __getitem__(self, index):
                return self.load_batch(index)

            def __len__(self):
                return self.posebyte.shape[0]

            def load_batch(self, iter):
                positives = np.where(self.posebyte[:, self.bit] == self.bit_value)[0]
                negatives = np.where(self.posebyte[:, self.bit] != self.bit_value)[0]

                shuffled = lambda seq, rnd=random.random: sorted(seq, key=lambda _: rnd())

                selections = shuffled(positives)[0:55] + shuffled(negatives)[0:56]
                # selections = positives[0:55].tolist() + negatives[0:56].tolist()

                posebytes = []

                for selection in selections:
                    posebytes += [self.posebyte[selection]]

                posebytes = torch.from_numpy(np.vstack(posebytes))

                return posebytes

        def draw_plot(train_losses, val_losses, iter_display):
            x = np.array(range(0, len(train_losses))) * iter_display

            fig, ax = plt.subplots()
            ax.grid(True)
            # ax.set_ylim([0.0, 0.1])

            plt.plot(x, train_losses, label="trn")
            plt.plot(x, val_losses, label="val")
            plt.legend()
            plt.savefig(results_path+str(bit)+'_'+str(bit_value)+'_lr_'+str(learning_rate)+'_loss.png')
            plt.close()

        def train(epoch, bit, bit_value, train_losses, val_losses):
            #
            # TRAIN
            #

            train_loss_acc = 0.0

            model.train()
            for batch_idx, data in enumerate(train_loader):
                if batch_idx == 10:
                    break

                batch = data.numpy()

                optimizer.zero_grad()
                input_data = Variable(torch.squeeze(torch.from_numpy(batch)).float().cuda())
                loss, l2_norm = model(input_data)
                loss.backward()
                optimizer.step()

                train_loss_acc += float(loss.data)

            train_loss_acc /= (batch_idx+1)

            #
            # VAL
            #

            val_loss_acc = 0.0

            model.eval()

            for batch_idx, data in enumerate(val_loader):
                if batch_idx == 1:
                    break

                batch = data.numpy()

                input_data = Variable(torch.squeeze(torch.from_numpy(batch)).float().cuda())
                loss, l2_norm = model(input_data)

                val_loss_acc += float(loss.data)

            val_loss_acc /= (batch_idx + 1)

            #
            # Wrap up epoch
            #

            train_losses += [train_loss_acc]
            val_losses += [val_loss_acc]

            draw_plot(train_losses, val_losses, 10)

            if epoch == 9:
                torch.save(model.state_dict(), results_path + 'bit_' + str(bit) + '_value_' + str(bit_value) + '_model_' + str(epoch) + '.pth')

        # Params

        results_path = 'results/'

        # RUN

        train_losses = []
        val_losses = []

        learning_rate = 1.08

        posebit_count = posebyte_train.shape[1]

        model = Net(posebit_count=posebit_count)
        model.cuda()
        state = model.state_dict()
        state_dict = torch.load('../language/results/model_20.pth')
        state.update(state_dict)
        model.load_state_dict(state)

        train_dataset = ThinSlicingTrainset(bit, bit_value=bit_value)
        val_dataset = ThinSlicingValset(bit, bit_value=bit_value)
        train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, shuffle=True, batch_size=1, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=0, shuffle=True, batch_size=1, drop_last=True)

        for epoch in range(0, 10):
            optimizer = optim.SGD([
                {'params': model.fc1.parameters(), 'lr': 0.0, 'momentum': 0.9, 'weight_decay': 0.0005, 'nesterov': True},
                {'params': model.fc2.parameters(), 'lr': 0.0, 'momentum': 0.9, 'weight_decay': 0.0005, 'nesterov': True},
                {'params': model.fc3.parameters(), 'lr': 0.0, 'momentum': 0.9, 'weight_decay': 0.0005, 'nesterov': True},

                {'params': model.masklayer.parameters(), 'lr': learning_rate, 'momentum': 0.9, 'weight_decay': 0.0005, 'nesterov': True},
            ])

            train(epoch, bit, bit_value, train_losses, val_losses)

        return model.cpu().masklayer.mask.data.numpy()


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

    for bit in range(posebyte_train.shape[1]):
        if np.sum(posebyte_train[:, bit] == 0) < 55:
            print('skipping', bit)
            continue
        if np.sum(posebyte_train[:, bit] == 1) < 55:
            print('skipping', bit)
            continue

        print(bit)

        for bit_value in [0, 1]:
            file_output_name = 'masks/mask_'+str(bit)+'_'+str(bit_value)+'.npy'

            try:
                np.load(file_output_name)
                continue
            except:
                pass

            mask = train_network(bit, bit_value)
            np.save(file_output_name, mask)
