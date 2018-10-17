import torch
import torch.optim as optim
import torch.utils.data.dataset
from torch.autograd import Variable
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
import scipy.misc
from glob import glob
from other.spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor)
from other.mean_vggs import get_mean
from path_manager import PathManager
from nets.net_poseemb import Net


# set the distance files
distances_train_file = PathManager.path_distance_frame0_path_2d_train
distances_valtest_file = PathManager.path_distance_frame0_path_2d_valtest

# set the train and val dataset files
dataset_train_txt = PathManager.path_dataset_train_txt
dataset_valtest_txt = PathManager.path_dataset_valtest_txt


class ThinSlicingTrainset(torch.utils.data.dataset.Dataset):
    def __init__(self, far_max=None):
        super(ThinSlicingTrainset, self).__init__()
        print('Loading Thin-Motion Trainset')

        self.far_min = 31
        self.far_max = far_max

        self.image_root = PathManager.path_image_root

        print('- loading distances')
        distances_filename = distances_train_file
        distances = np.load(distances_filename)
        self.nearest_indices = np.argsort(distances, axis=1)
        self.nearest_indices[:, 0] = np.array(range(self.nearest_indices.shape[0]))

        with open(dataset_train_txt) as infile:
            image_list = [x.strip() for x in infile.readlines()]
            self.dataset = [[' '.join(x.strip().split(' ')[:-16]) + '/'] + x.strip().split(' ')[-16:] for x in image_list]

    def __getitem__(self, index):
        return self.load_batch(index)

    def __len__(self):
        return len(self.dataset)

    def load_batch(self, index):
        # print index

        shuffled = lambda seq, rnd=random.random: sorted(seq, key=lambda _: rnd())
        should_horizontally_flip = random.getrandbits(1)

        selection = [0] + shuffled(range(1, 31))[0:5] + shuffled(range(31, self.far_max))[0:105]

        spatial_transform = Compose([Scale(112),
                                     CenterCrop(112),
                                     ToTensor(),
                                     Normalize(get_mean(), [1, 1, 1])])

        near_indices = self.nearest_indices[index]

        image_crops_list = []
        prepped_tensors = []
        for idx, near_idx in enumerate(near_indices[selection]):
            image_name = self.image_root + self.dataset[near_idx][0] + self.dataset[near_idx][1].split('_')[1] + '.png'
            raw_imgs = scipy.misc.imread(image_name)[np.newaxis,:]

            r1, g1, b1 = [115, 108, 99]
            r2, g2, b2, = get_mean()
            red, green, blue = raw_imgs[:, :, :, 0], raw_imgs[:, :, :, 1], raw_imgs[:, :, :, 2]
            mask = (red == r1) & (green == g1) & (blue == b1)
            raw_imgs[:, :, :, :3][mask] = [r2, g2, b2]

            image_crops = self.augment(raw_imgs, do_flip=should_horizontally_flip)
            image_crops_list += [image_crops]

            prepped_images = [spatial_transform(Image.fromarray(image_crop)) for image_crop in image_crops][0]
            prepped_tensor = prepped_images

            prepped_tensors += [prepped_tensor]

        # labels = ['anchor', 'similar', 'similar', 'similar', 'similar', 'similar', 'different', 'different', 'different']
        # image_crops_list = image_crops_list[0:9]
        # for frame in [0]:
        #     fig = plt.figure()
        #     for axes_idx in range(1,10):
        #         fig.add_subplot(3, 3, axes_idx)
        #         plt.imshow(image_crops_list[axes_idx-1][frame])
        #         plt.title(labels[axes_idx-1])
        #         plt.axis('off')
        # plt.show()

        batch = torch.stack(prepped_tensors, 0)

        batch2 = torch.zeros(batch.size())
        batch2[:, 0] = batch[:, 2]
        batch2[:, 1] = batch[:, 1]
        batch2[:, 2] = batch[:, 0]

        return batch2

    def augment(self, raw_imgs, do_flip):
        image0 = raw_imgs[0]

        if do_flip:
            for raw_img_idx, raw_img in enumerate(raw_imgs):
                raw_imgs[raw_img_idx] = np.fliplr(raw_img)

        stacked_images = raw_imgs

        shuffled = lambda seq, rnd=random.random: sorted(seq, key=lambda _: rnd())

        image_crops = []

        xmin = 16
        ymin = 16
        xmax = xmin + 112
        ymax = ymin + 112

        size_img = image0.shape[0]

        s_ratio = 0.050
        t_ratio = 0.050

        smag_max = int(np.ceil(s_ratio * 112))
        tmag_max = int(np.ceil(t_ratio * 112))

        scale = shuffled(range(smag_max * 2))[0]

        smag_rand = scale - smag_max
        xmin = xmin - smag_rand
        xmax = xmax + smag_rand
        ymin = ymin - smag_rand
        ymax = ymax + smag_rand

        tmag_xmin = np.max([1 - xmin, -tmag_max])
        tmag_ymin = np.max([1 - ymin, -tmag_max])
        tmag_xmax = np.min([size_img - xmax, tmag_max])
        tmag_ymax = np.min([size_img - ymax, tmag_max])

        translate_x = shuffled(range(tmag_xmax - tmag_xmin))[0]
        translate_y = shuffled(range(tmag_ymax - tmag_ymin))[0]

        tmag_rand_x = range(tmag_xmin, tmag_xmax)[translate_x]
        tmag_rand_y = range(tmag_ymin, tmag_ymax)[translate_y]
        xmin = xmin + tmag_rand_x
        xmax = xmax + tmag_rand_x
        ymin = ymin + tmag_rand_y
        ymax = ymax + tmag_rand_y

        image_sliced = stacked_images[:, ymin:ymax, xmin:xmax, :]

        for image_slice in image_sliced:
            img_crop = cv2.resize(image_slice, (112, 112))
            image_crops += [img_crop]

        return image_crops


class ThinSlicingValset(torch.utils.data.dataset.Dataset):
    def __init__(self):
        super(ThinSlicingValset, self).__init__()
        print('Loading Thin-Motion Valset')

        self.image_root = PathManager.path_image_root

        print('- loading distances')
        distances_filename = distances_valtest_file
        distances = np.load(distances_filename)
        distances = distances[:1919, :1919]

        self.nearest_indices = np.argsort(distances, axis=1)
        self.nearest_indices[:, 0] = np.array(range(self.nearest_indices.shape[0]))

        with open(dataset_valtest_txt) as infile:
            image_list = [x.strip() for x in infile.readlines()]
            self.dataset = [[' '.join(x.strip().split(' ')[:-16]) + '/'] + x.strip().split(' ')[-16:] for x in image_list]
            self.dataset = self.dataset[:1919]

    def __getitem__(self, index):
        return self.load_batch(index)

    def __len__(self):
        return self.nearest_indices.shape[0]

    def load_batch(self, index):
        # print index

        shuffled = lambda seq, rnd=random.random: sorted(seq, key=lambda _: rnd())

        selection = [0] + shuffled(range(1, 31))[0:5] + shuffled(range(31, self.nearest_indices.shape[1]))[0:105]

        spatial_transform = Compose([Scale(112),
                                     CenterCrop(112),
                                     ToTensor(),
                                     Normalize(get_mean(), [1, 1, 1])])

        near_indices = self.nearest_indices[index]

        image_crops_list = []
        prepped_tensors = []
        for idx, near_idx in enumerate(near_indices[selection]):
            image_name = self.image_root + self.dataset[near_idx][0] + self.dataset[near_idx][1].split('_')[1] + '.png'
            raw_imgs = scipy.misc.imread(image_name)[np.newaxis,:]

            r1, g1, b1 = [115, 108, 99]
            r2, g2, b2, = get_mean()
            red, green, blue = raw_imgs[:, :, :, 0], raw_imgs[:, :, :, 1], raw_imgs[:, :, :, 2]
            mask = (red == r1) & (green == g1) & (blue == b1)
            raw_imgs[:, :, :, :3][mask] = [r2, g2, b2]

            image_crops = self.augment(raw_imgs)
            image_crops_list += [image_crops]

            prepped_images = [spatial_transform(Image.fromarray(image_crop)) for image_crop in image_crops][0]
            prepped_tensor = prepped_images

            prepped_tensors += [prepped_tensor]

        # labels = ['anchor', 'similar', 'similar', 'similar', 'similar', 'similar', 'different', 'different', 'different']
        # image_crops_list = image_crops_list[0:9]
        # for frame in [0]:
        #     fig = plt.figure()
        #     for axes_idx in range(1,10):
        #         fig.add_subplot(3, 3, axes_idx)
        #         plt.imshow(image_crops_list[axes_idx-1][frame])
        #         plt.title(labels[axes_idx-1])
        #         plt.axis('off')
        # plt.show()

        batch = torch.stack(prepped_tensors, 0)

        batch2 = torch.zeros(batch.size())
        batch2[:, 0] = batch[:, 2]
        batch2[:, 1] = batch[:, 1]
        batch2[:, 2] = batch[:, 0]

        return batch2

    def augment(self, raw_imgs):
        image0 = raw_imgs[0]

        stacked_images = raw_imgs

        image_crops = []

        xmin = 16
        ymin = 16
        xmax = xmin + 112
        ymax = ymin + 112

        size_img = image0.shape[0]

        s_ratio = 0.050
        t_ratio = 0.050

        smag_max = int(np.ceil(s_ratio * 112))
        tmag_max = int(np.ceil(t_ratio * 112))

        smag_rand = range(smag_max * 2)[6] - smag_max
        xmin = xmin - smag_rand
        xmax = xmax + smag_rand
        ymin = ymin - smag_rand
        ymax = ymax + smag_rand

        tmag_xmin = np.max([1 - xmin, -tmag_max])
        tmag_ymin = np.max([1 - ymin, -tmag_max])
        tmag_xmax = np.min([size_img - xmax, tmag_max])
        tmag_ymax = np.min([size_img - ymax, tmag_max])
        tmag_rand_x = range(tmag_xmin, tmag_xmax)[range(tmag_xmax - tmag_xmin)[6]]
        tmag_rand_y = range(tmag_ymin, tmag_ymax)[range(tmag_ymax - tmag_ymin)[6]]
        xmin = xmin + tmag_rand_x
        xmax = xmax + tmag_rand_x
        ymin = ymin + tmag_rand_y
        ymax = ymax + tmag_rand_y

        image_sliced = stacked_images[:, ymin:ymax, xmin:xmax, :]

        for image_slice in image_sliced:
            img_crop = cv2.resize(image_slice, (112, 112))
            image_crops += [img_crop]

        return image_crops


def draw_plot(train_losses, val_losses, iter_display):
    x = np.array(range(0, len(train_losses))) * iter_display

    plt.ylim([0, 0.25])

    plt.plot(x, train_losses, label="trn")
    plt.plot(x, val_losses, label="val")
    plt.legend()
    global base_lr
    plt.savefig(results_path + 'loss.png')
    plt.close()


def train(epoch):
    global train_losses
    global val_losses

    val_iters = 500

    train_loss_mean = []
    for batch_idx, data in enumerate(train_loader):
        model.train()

        data = data[0].cuda()

        optimizer.zero_grad()
        (loss, l2_norm) = model(Variable(data))
        loss.backward()
        optimizer.step()

        train_loss_mean += [float(loss.data)]

        if batch_idx % val_iters == 0:
            print('epoch '+str(epoch)+', batch '+str(batch_idx)+': '+ str(float(loss.data)))

            train_losses += [np.mean(train_loss_mean)]
            val_losses += [val(5)]
            draw_plot(train_losses, val_losses, iter_display=val_iters)
            train_loss_mean = []

    torch.save(model.state_dict(), results_path + 'model_' + str(epoch) + '.pth')


def val(val_iterations=1):
    model.eval()

    loss_mean = 0

    for batch_idx, data in enumerate(val_loader):
        if batch_idx == val_iterations:
            break

        data = data[0].cuda()
        (loss, l2_norm) = model(Variable(data))
        loss_mean += float(loss.data) / float(val_iterations)

    return loss_mean


if __name__ == '__main__':
    base_lr = 0.01

    train_losses = []
    val_losses = []
    results_path = 'results/'

    start_epoch = np.max([int(weights.split('_')[-1].split('.pth')[0])+1 for weights in glob(results_path+'model_*.pth')] + [0])

    model = Net()
    if start_epoch > 0:
        print('Resuming from model_'+str(start_epoch-1)+'.pth')
        model.load_state_dict(torch.load(results_path+'model_'+str(start_epoch-1)+'.pth'))
    else:
        raw_state_dict = model.state_dict()
        state_dict = torch.load(PathManager.path_vggs_conv_weights) # vgg convs, random fcs
        state_dict['conv1.weight'] = state_dict.pop('0.weight')
        state_dict['conv1.bias'] = state_dict.pop('0.bias')
        state_dict['conv2.weight'] = state_dict.pop('4.weight')
        state_dict['conv2.bias'] = state_dict.pop('4.bias')
        state_dict['conv3.weight'] = state_dict.pop('7.weight')
        state_dict['conv3.bias'] = state_dict.pop('7.bias')
        state_dict['conv4.weight'] = state_dict.pop('9.weight')
        state_dict['conv4.bias'] = state_dict.pop('9.bias')
        state_dict['conv5.weight'] = state_dict.pop('11.weight')
        state_dict['conv5.bias'] = state_dict.pop('11.bias')
        state_dict.pop('15.1.weight')
        state_dict.pop('15.1.bias')
        state_dict.pop('18.1.weight')
        state_dict.pop('18.1.bias')
        raw_state_dict.update(state_dict)
        model.load_state_dict(raw_state_dict)
    model = model.cuda()

    # val_dataset = ThinSlicingValset()
    # val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=0, shuffle=True, batch_size=1)
    # for batch_idx, data in enumerate(val_loader):
    #     x = 5
    # dataset_trn = ThinSlicingTrainset(far_max=len(ThinSlicingTrainset()))
    # train_loader = torch.utils.data.DataLoader(dataset_trn, num_workers=0, shuffle=True, batch_size=1)
    # for batch_idx, data in enumerate(train_loader):
    #     x = 5

    for epoch in range(start_epoch, 7):
        anneal_factor = 0.2**epoch

        optimizer = optim.SGD([
            {'params': model.conv1.parameters(), 'lr': 0.1*base_lr*anneal_factor, 'momentum': 0.9, 'weight_decay': 0.0005, 'nesterov': True},
            {'params': model.conv2.parameters(), 'lr': 0.1*base_lr*anneal_factor, 'momentum': 0.9, 'weight_decay': 0.0005, 'nesterov': True},
            {'params': model.conv3.parameters(), 'lr': 0.1*base_lr*anneal_factor, 'momentum': 0.9, 'weight_decay': 0.0005, 'nesterov': True},
            {'params': model.conv4.parameters(), 'lr': 0.1*base_lr*anneal_factor, 'momentum': 0.9, 'weight_decay': 0.0005, 'nesterov': True},
            {'params': model.conv5.parameters(), 'lr': 0.1*base_lr*anneal_factor, 'momentum': 0.9, 'weight_decay': 0.0005, 'nesterov': True},
            {'params': model.fc6.parameters(), 'lr': base_lr*anneal_factor, 'momentum': 0.9, 'weight_decay': 0.0005, 'nesterov': True},
            {'params': model.fc7.parameters(), 'lr': base_lr*anneal_factor, 'momentum': 0.9, 'weight_decay': 0.0005, 'nesterov': True},
        ])

        far_min = 31
        far_max = len(ThinSlicingTrainset()) - epoch * 3000
        if far_max < far_min + 1000:
            far_max = far_min + 1000

        dataset_trn = ThinSlicingTrainset(far_max=far_max)
        train_loader = torch.utils.data.DataLoader(dataset_trn, num_workers=3, shuffle=True, batch_size=1)
        dataset_val = ThinSlicingValset()
        val_loader = torch.utils.data.DataLoader(dataset_val, num_workers=0, shuffle=True, batch_size=1)

        train(epoch)

    print('Training complete')
