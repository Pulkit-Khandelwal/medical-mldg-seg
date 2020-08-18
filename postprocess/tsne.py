import os
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.nn as nn
from utils import sgd, crossentropyloss, fix_seed, write_log, compute_accuracy, binarycrossentropy, rmsprop
from dice_loss import GeneralizedDiceLoss, WeightedCrossEntropyLoss
import numpy as np
import torch.utils.data as data
import torch.optim as optim
import os.path
import torch
from utils3D import cross_entropy_dice
from data_reader_unet_spine import *
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import Unet3D_meta_learning
import scipy.misc
import sklearn as sk
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter
from sklearn.utils import shuffle
import torchvision

class Model:
    def __init__(self):
        self.batch_size = 1
        self.num_classes = 4
        self.unseen_index = 3
        self.lr = 0.001
        self.inner_loops = 1200
        self.step_size = 20
        self.weight_decay = 0.00005
        self.momentum = 0.9
        self.state_dict = ''
        self.logs = 'logs'
        self.patch_size = 64
        self.test_every = 20
        self.test_unseen = 50
        self.epochs = 10

        self.writer = SummaryWriter(comment='fewshot-mldg')

        torch.set_default_tensor_type('torch.cuda.DoubleTensor')

        self.TrainMetaData = []
        self.ValidMetaData = []
        self.TestMetaData = []
        self.FewShotData = []

        self.count1 = 0
        self.count2 = 0
        self.count3 = 0

        # fix the random seed or not
        fix_seed()

        self.setup_path()
        self.network = Unet3D_meta_learning.Unet3D()  # load the vanilla 3D-Unet
        self.network = self.network.cuda()

        self.configure()

    def setup_path(self):
        modality = 'dummy_string'

        self.TrainMetaData = []
        self.ValidMetaData = []
        self.TestMetaData = []
        self.FewShotData = []

        self.test_paths = ['/path/to/images/test_lumbar.txt',
                         '/path/to/images/test_thoracic_lower.txt',
                         '/path/to/images/test_thoracic_middle.txt',
                         '/path/to/images/test_thoracic_upper.txt']

        self.train_paths = ['/path/to/images/few_lumbar.txt',
                            '/path/to/images/few_thoracic_lower.txt',
                            '/path/to/images/few_thoracic_middle.txt',
                            '/path/to/images/few_thoracic_upper.txt']

        self.val_paths = ['/path/to/images/val_lumbar.txt',
                            '/path/to/images/val_thoracic_lower.txt',
                            '/path/to/images/val_thoracic_middle.txt',
                            '/path/to/images/val_thoracic_upper.txt']

        # for tsne
        test_path = '/path/to/images/tsne.txt'

        self.TestMetaData = BatchImageGenerator(test_path, modality, transform=False,
                                        patch_size=self.patch_size, n_patches_transform=5, is_test=True)

        self.batImageGenTests = torch.utils.data.DataLoader(self.TestMetaData, batch_size=self.batch_size,
                                                       sampler=SequentialSampler(self.TestMetaData),
                                                       num_workers=0,
                                                       pin_memory=False)


    def configure(self):
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                   momentum=self.momentum)
        self.scheduler = lr_scheduler.StepLR(optimizer=self.optimizer, step_size=self.step_size, gamma=0.1)
        self.loss_fn = GeneralizedDiceLoss()


    def tsnePlot(self):
        self.network.eval()
        writer = SummaryWriter(comment='tsne-mldg')

        ds = iter(self.batImageGenTests)

        PATH = '/path/to/trained/model/trained_model.tar' # load the trained model
        tmp=torch.load(PATH)
        pretrained_dict=tmp['state']
        model_dict=self.network.state_dict()
        pretrained_dict={k:v for k, v in pretrained_dict.items() if k in model_dict and v.size()==model_dict[k].size()}
        model_dict.update(pretrained_dict)
        self.network.load_state_dict(model_dict)

        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        self.network.convd5.register_forward_hook(get_activation('convd5'))

        num_images = len(ds)
        input_images = []

        index = 1
        DN = []
        for im in range(num_images):
            if index == 5:
                index = 1
            trains, labels = next(ds)
            trains = trains.squeeze(0)

            print(len(trains))
            domain_number = torch.tensor(np.full((len(trains), 1), index))
            index = index + 1

            input_images.append([trains, domain_number])
            DN.append(domain_number)

        DN2 = torch.cat(DN)
        full_output = []
        bs = 1

        for x in range(num_images):
            print("image number", x)
            curr_img = input_images[x]
            curr_img = curr_img[0]

            images_train_three_domains_shape = np.shape(curr_img)
            total_batches_per_image = int(images_train_three_domains_shape[0] / bs)
            print("total number of batches >>>>>>>>", total_batches_per_image)
            batch_index = 0

            for batch in range(1, total_batches_per_image + 1):

                images_test, labels_test = trains[batch_index: batch_index + bs, :, :, :, :], labels[batch_index: batch_index + bs, :, :, :]

                images_test, labels_test = Variable(images_test, requires_grad=False).cuda(), Variable(labels_test, requires_grad=False).float().cuda()

                images_test = Variable(images_test).cuda()
                labels_test = Variable(labels_test).float().cuda()

                outputs, _ = self.network.forward(images_test, meta_step_size=0.001, meta_loss=None, stop_gradient=False)
                feature = activation['convd5']
                print(feature.shape)
                full_output.append(feature)

        full_output = torch.cat(full_output)
        print(np.shape(full_output))
        print(len(full_output))
        embed = full_output.view(len(full_output), -1)
        print(np.shape(embed))

        writer.add_embedding(mat = embed.data, metadata=DN2.data, global_step=index)
        writer.close()


###### call the model here and the tsnePlot method ######
call_model = Model()
call_model.tsnePlot()
