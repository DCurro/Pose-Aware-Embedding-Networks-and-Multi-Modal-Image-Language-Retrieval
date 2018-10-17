import torch
import torch.utils.data.dataset
from torch.autograd import Variable
import cv2
import numpy as np
from PIL import Image
from other.spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor)
from other.mean_vggs import get_mean
from path_manager import PathManager
from nets.net_poseemb import Net
import scipy.misc


class ThinSlicingValset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset_txt):
        super(ThinSlicingValset, self).__init__()

        self.image_root = PathManager.path_image_root
        with open(dataset_txt) as infile:
            image_list = [x.strip() for x in infile.readlines()]
            self.dataset = [[' '.join(x.strip().split(' ')[:-16]) + '/'] + x.strip().split(' ')[-16:] for x in image_list]

    def __getitem__(self, index):
        return self.load_batch(index)

    def __len__(self):
        return len(self.dataset)

    def load_batch(self, index):
        # print index

        selection = [index]

        spatial_transform = Compose([Scale(112),
                                     CenterCrop(112),
                                     ToTensor(),
                                     Normalize(get_mean(), [1, 1, 1])])

        image_crops_list = []
        prepped_tensors = []
        for idx, near_idx in enumerate(selection):
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
            img_crop = cv2.resize(image_slice, (112,112))
            image_crops += [img_crop]

        return image_crops


def embed(suffix):
    model.eval()

    embeddings = []

    for batch_idx, data in enumerate(val_loader):
        data = data[:,0].cuda()

        (_, l2_norm) = model(Variable(data))
        embeddings += [l2_norm.data.cpu().numpy()]

    embeddings = np.concatenate(embeddings)
    np.save('embeddings/embeddings_' + suffix + '_0.npy', embeddings)


if __name__ == '__main__':
    results_path = 'results/'

    model = Net(l2_norm_dump=True)
    raw_state_dict = model.state_dict()
    state_dict = torch.load(results_path+'model_6.pth')
    raw_state_dict.update(state_dict)
    model.load_state_dict(raw_state_dict)
    model = model.cuda()

    dataset_txt = PathManager.path_dataset_valtest_txt

    dataset_val = ThinSlicingValset(dataset_txt)
    val_loader = torch.utils.data.DataLoader(dataset_val, num_workers=1, shuffle=False, batch_size=111)
    embed('valtest')

    dataset_txt = PathManager.path_dataset_train_txt

    dataset_val = ThinSlicingValset(dataset_txt)
    val_loader = torch.utils.data.DataLoader(dataset_val, num_workers=1, shuffle=False, batch_size=111)
    embed('train')
