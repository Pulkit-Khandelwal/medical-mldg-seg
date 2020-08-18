import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
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

class SubsetRandomSampler(SubsetRandomSampler):
    r"""Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source=data_source

    def __iter__(self):
        cpu=torch.device('cpu')
        #return iter(torch.randperm(len(self.data_source), device=cpu).tolist())
        return iter(self.data_source)

    def __len__(self):
        return len(self.data_source)


class ModelBaseline:
    def __init__(self):
        self.batch_size = 1
        self.num_classes = 2
        self.unseen_index = 3
        self.lr = 0.001
        self.inner_loops = 1200
        self.step_size = 20
        self.weight_decay = 0.00005
        self.momentum = 0.9
        self.state_dict = ''
        self.logs = 'logs'
        self.patch_size = 64
        self.test_every = 15
        self.test_unseen = 15
        self.epochs = 20

        self.writer = SummaryWriter(comment='fewshot-meta-learning')

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
        modality = 'dummy_for_now'

        self.TrainMetaData = []
        self.ValidMetaData = []
        self.TestMetaData = []
        self.FewShotData = []

        self.test_paths = ['/path/to/images/spine_images/test_lumbar.txt',
                         '/path/to/images/spine_images/test_thoracic_lower.txt',
                         '/path/to/images/spine_images/test_thoracic_middle.txt',
                         '/path/to/images/spine_images/test_thoracic_upper.txt']

        self.train_paths = ['/path/to/images/spine_images/few_lumbar.txt',
                            '/path/to/images/spine_images/few_thoracic_lower.txt',
                            '/path/to/images/spine_images/few_thoracic_middle.txt',
                            '/path/to/images/spine_images/few_thoracic_upper.txt']

        self.val_paths = ['/path/to/images/spine_images/val_lumbar.txt',
                            '/path/to/images/spine_images/val_thoracic_lower.txt',
                            '/path/to/images/spine_images/val_thoracic_middle.txt',
                            '/path/to/images/spine_images/val_thoracic_upper.txt']

        test_path = self.test_paths[3]

        self.train_paths = ['/path/to/images/june25/verse_few.txt'] # this is fewshot (four image)
        self.val_paths = ['/path/to/images/june25/verse_val.txt']   # this is validation (one image)

        img_path = self.train_paths[0]
        val_path = self.val_paths[0]

        dataset_img = BatchImageGenerator(img_path, modality, transform=True,
                                        patch_size=self.patch_size, n_patches_transform=5)

        self.TrainMetaData.append(dataset_img)



        dataset_val = BatchImageGenerator(val_path, modality, transform=False,
                                        patch_size=self.patch_size, n_patches_transform=5, val=True)

        self.ValidMetaData.append(dataset_val)


        self.TestMetaData = BatchImageGenerator(test_path, modality, transform=False,
                                        patch_size=self.patch_size, n_patches_transform=5, is_test=True)

        self.batImageGenTests = torch.utils.data.DataLoader(self.TestMetaData, batch_size=self.batch_size,
                                                       sampler=SequentialSampler(self.TestMetaData),
                                                       num_workers=0,
                                                       pin_memory=False)


    def configure(self):

        PATH = '/path/to/images/code_unet/saved_models/meta_learning_verse_exp3_june23.tar' #meta_learning_verse_exp3_june23
        tmp=torch.load(PATH)
        pretrained_dict=tmp['state']
        print(">>>> let us see the state pretrained_dict", tmp['ite'])
        model_dict=self.network.state_dict()
        pretrained_dict={k:v for k, v in pretrained_dict.items() if k in model_dict and v.size()==model_dict[k].size()}
        model_dict.update(pretrained_dict)
        self.network.load_state_dict(model_dict)

        for k, v in self.network.state_dict().items():
            name = k
            param = torch.nn.Parameter(v)
            if 'convd' in name:
                print(name, param.requires_grad)
                param.requires_grad=False
                print(name, 'set to', param.requires_grad)



        self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                   momentum=self.momentum)
        self.scheduler = lr_scheduler.StepLR(optimizer=self.optimizer, step_size=self.step_size, gamma=0.1)
        self.loss_fn = GeneralizedDiceLoss()

    def train(self):

        print('>>>>>>>>>TRAINING<<<<<<<<')
        self.network.train()
        self.best_accuracy_val=-1

        self.batImageGenTrains=[]
        self.batImageGenVals=[]
        self.batImageGenFews=[]
        #self.batImageGenTests=[]

        for dataset in self.TrainMetaData:

            train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                       sampler=SequentialSampler(dataset),
                                                       num_workers=0,
                                                       pin_memory=False)

            self.batImageGenTrains.append(train_loader)

        for dataset in self.ValidMetaData:

            val_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                       sampler=SequentialSampler(dataset),
                                                       num_workers=0,
                                                       pin_memory=False)

            self.batImageGenVals.append(val_loader)

        d1 = iter(self.batImageGenTrains[0])
        ite = 0

        # just have the three training modalities
        # validation set comes from the same three modalities but different images

        for epoch in range(self.epochs):
            print("<<<<<<<<< epoch >>>>>>>>", epoch)
            d1 = iter(self.batImageGenTrains[0])

            for im in range(len(d1)):
                trains_d1, labels_d1 = next(d1)
                trains = trains_d1.squeeze(0)
                labels = labels_d1.squeeze(0)

                bs = 2
                images_train_three_domains_shape = np.shape(trains)
                total_batches_per_image = int(images_train_three_domains_shape[0] / bs)
                batch_index = 0
                print("total number of batches", total_batches_per_image)

                for batch in range(1, total_batches_per_image + 1):
                    ite += 1
                    print("-------in epoch", epoch + 1, "image number-----", im+1, "batch number-------", batch)

                    images_train, labels_train = trains[batch_index: batch_index + bs, :, :, :, :], labels[batch_index: batch_index + bs, :, :, :]

                    images_train, labels_train = Variable(images_train, requires_grad=False).cuda(), Variable(labels_train, requires_grad=False).float().cuda()

                    outputs, predictions_train = self.network(x=images_train, meta_step_size=0.001, meta_loss=None, stop_gradient=False)

                    loss = self.loss_fn(outputs, labels_train.long())

                    predictions_train = predictions_train.cpu().data.numpy()
                    predicted_classes = np.argmax(predictions_train, axis=1)

                    train_acc = compute_accuracy(predictions=predicted_classes, labels=labels_train.cpu().data.numpy())
                    print("---------train accuracy-------- ", train_acc, "----current loss------ ", loss.cpu().data.numpy())

                    batch_index += bs

                    self.writer.add_scalar('Train/Loss', loss.data, ite)
                    self.writer.add_scalar('Train/Accuracy', train_acc, ite)

                    # init the grad to zeros first
                    self.optimizer.zero_grad()

                    # backpropagate your network
                    loss.backward()

                    # optimize the parameters
                    self.optimizer.step()

                    print('ite:', ite, 'loss:', loss.cpu().data.numpy(), 'lr:', self.scheduler.get_lr()[0])

                    del loss, outputs
                    torch.cuda.empty_cache()

                    if ite % self.test_every == 0 and ite is not 0:
                        print('>>>>>>>>> VALIDATION <<<<<<<<')
                        self.test(ite)

            self.scheduler.step()


    def test(self, ite):
        self.network.eval()

        d1 = iter(self.batImageGenVals[0])

        print("length of the three loaders >>>>> ", len(d1))
        accuracies = []
        val_losses = []

        for im in range(len(d1)):
            try:
                trains, labels = next(d1)
            except StopIteration:
                d1 = iter(self.batImageGenVals[0])
                trains, labels = next(d1)

            bs = 2
            trains = trains.squeeze(0)
            labels = labels.squeeze(0)
            print("check", np.shape(trains), np.shape(labels))
            images_train_three_domains_shape = np.shape(trains)
            total_batches_per_image = int(images_train_three_domains_shape[0] / bs)
            print("total number of batches >>>>>>>>", total_batches_per_image)
            batch_index = 0

            for batch in range(1, total_batches_per_image + 1):

                images_test, labels_test = trains[batch_index: batch_index + bs, :, :, :, :], labels[batch_index: batch_index + bs, :, :, :]

                images_test, labels_test = Variable(images_test, requires_grad=False).cuda(), Variable(labels_test, requires_grad=False).float().cuda()

                images_test = Variable(images_test).cuda()
                labels_test = Variable(labels_test).float().cuda()

                outputs, predictions = self.network(images_test, meta_step_size=0.001, meta_loss=None,
                                                        stop_gradient=False)

                val_loss = self.loss_fn(outputs, labels_test.long())
                val_loss_data = val_loss.cpu().data.numpy()

                predictions = predictions.cpu().data.numpy()
                predicted_classes = np.argmax(predictions, axis=1)

                accuracy_val = compute_accuracy(predictions=predicted_classes, labels=labels_test.cpu().data.numpy())

                accuracies.append(accuracy_val)
                val_losses.append(val_loss_data)

                #print("----------accuracy val----------", accuracy_val)

                batch_index += bs
                self.writer.add_scalar('Validation/Accuracy', accuracy_val, ite) #mean_acc
                self.writer.add_scalar('Validation/Loss', val_loss.data, ite) #mean_val_loss

                del outputs, val_loss

        self.network.train()
        mean_acc = np.mean(accuracies)
        mean_val_loss = np.mean(val_losses)

        if mean_acc > self.best_accuracy_val:
            self.best_accuracy_val = mean_acc
            print("--------best validation accuracy--------", self.best_accuracy_val)

            outfile = os.path.join('/path/to/images/code_unet/saved_models/', 'fewshot_meta_learning_verse_june26_images_1.tar')
            torch.save({'ite': ite, 'state': self.network.state_dict()}, outfile)

        self.writer.close()


    def unseen_fourth_mod(self):
        self.network.eval()

        ds = iter(self.batImageGenTests)
        print(" <<<<<< length of the test loader >>>>> ", len(ds))

        PATH = '/path/to/images/code_unet/saved_models/model.tar'
        tmp=torch.load(PATH)
        pretrained_dict=tmp['state']
        print(">>>> let us see the state pretrained_dict", tmp['ite'])
        model_dict=self.network.state_dict()
        pretrained_dict={k:v for k, v in pretrained_dict.items() if k in model_dict and v.size()==model_dict[k].size()}
        model_dict.update(pretrained_dict)
        self.network.load_state_dict(model_dict)

        num_images = len(ds)
        accuracies = []
        val_losses = []
        it=0
        acc_four_image = []

        for im in range(num_images):
            trains, labels = next(ds)
            trains = trains.squeeze(0)
            labels = labels.squeeze(0)

            bs = 1
            images_train_three_domains_shape = np.shape(trains)
            total_batches_per_image = int(images_train_three_domains_shape[0] / bs)
            print("total number of batches >>>>>>>>", total_batches_per_image)
            batch_index = 0

            acc_this_image = []

            for batch in range(1, total_batches_per_image + 1):

                images_test, labels_test = trains[batch_index: batch_index + bs, :, :, :, :], labels[batch_index: batch_index + bs, :, :, :]

                images_test, labels_test = Variable(images_test, requires_grad=False).cuda(), Variable(labels_test, requires_grad=False).float().cuda()

                images_test = Variable(images_test).cuda()
                labels_test = Variable(labels_test).float().cuda()

                outputs, predictions = self.network(images_test, meta_step_size=0.001, meta_loss=None,
                                                        stop_gradient=False)

                val_loss = self.loss_fn(outputs, labels_test.long())
                val_loss_data = val_loss.cpu().data.numpy()

                predictions = predictions.cpu().data.numpy()
                predicted_classes = np.argmax(predictions, axis=1)

                l_test = labels_test.cpu().data.numpy()

                if np.sum(l_test) != 0:

                    accuracy_val = compute_accuracy(predictions=predicted_classes, labels=labels_test.cpu().data.numpy())
                    accuracies.append(accuracy_val)

                    imk = np.reshape(images_test.cpu(), (64,64,64))
                    imk = imk[32,:,:]
                    imk = np.reshape(imk, (1,64,64))

                    lb = np.reshape(labels_test.cpu(), (64,64,64))
                    lb = lb[32,:,:]
                    lb = np.reshape(lb, (1,64,64))

                    pre = np.reshape(predicted_classes, (64,64,64))
                    pre = pre[32,:,:]
                    pre = np.reshape(pre, (1,64,64))

                    img_batch = np.stack((imk,lb,pre))
                    self.writer.add_image('Test/three' + str(it), img_batch, dataformats='NCHW') # bs,1,64,64,64
                    self.writer.add_scalar('Test/Dice', accuracy_val, it) #mean_acc

                    it = it + 1
                    acc_this_image.append(accuracy_val)

                val_losses.append(val_loss_data)

                del outputs, val_loss

                batch_index += bs

            mean_acc_this_image = np.mean(acc_this_image)
            acc_four_image.append(mean_acc_this_image)
            print("----------accuracy this image----------", mean_acc_this_image)

        mean_acc = np.mean(accuracies)
        mean_val_loss = np.mean(val_losses)
        print("---------- final test accuracy for unseen fourth modality data----------", mean_acc)

        std_acc = np.std(acc_four_image)
        print("--- the four values ---", acc_four_image)
        print("---------- final test std for unseen fourth modality data----------", std_acc)


###### call the model here
baseline = ModelBaseline()
baseline.train()
