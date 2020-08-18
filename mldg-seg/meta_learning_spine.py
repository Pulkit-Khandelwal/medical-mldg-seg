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

class SubsetRandomSampler(SubsetRandomSampler):
    r"""Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source=data_source

    def __iter__(self):
        cpu=torch.device('cpu')
        return iter(self.data_source)

    def __len__(self):
        return len(self.data_source)


class MLDG:
    def __init__(self):
        self.batch_size = 1
        self.num_classes = 2
        self.unseen_index = 0
        self.lr = 0.001
        self.inner_loops = 1200
        self.step_size = 20
        self.weight_decay = 0.00005
        self.momentum = 0.9
        self.state_dict = ''
        self.logs = 'logs'
        self.patch_size = 64
        self.test_every = 50
        self.test_unseen = 100
        self.epochs = 20
        self.meta_step_scale = 1000

        self.writer = SummaryWriter(comment='spine-meta-learning')

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

        self.test_path = ['/path/to/images//xvert.txt']

        self.train_paths = ['/path/to/images//spine_images/lumbar.txt',
                            '/path/to/images//spine_images/thoracic_lower.txt',
                            '/path/to/images//spine_images/thoracic_middle.txt',
                            '/path/to/images//spine_images/thoracic_upper.txt']

        self.train_paths = ['/path/to/images//spine_images/xvert_lumbar.txt',
                            '/path/to/images//spine_images/xvert_thoracic_lower.txt',
                            '/path/to/images//spine_images/xvert_thoracic_middle.txt',
                            '/path/to/images//spine_images/xvert_thoracic_upper.txt']

        self.few_paths = ['/path/to/images//spine_images/few_lumbar.txt',
                            '/path/to/images//spine_images/few_thoracic_lower.txt',
                            '/path/to/images//spine_images/few_thoracic_middle.txt',
                            '/path/to/images//spine_images/few_thoracic_upper.txt']

        self.val_paths = ['/path/to/images//spine_images/val_lumbar.txt',
                            '/path/to/images//spine_images/val_thoracic_lower.txt',
                            '/path/to/images//spine_images/val_thoracic_middle.txt',
                            '/path/to/images//spine_images/val_thoracic_upper.txt']

        for x in range(4):
            img_path = self.train_paths[x]

            dataset_img = BatchImageGenerator(img_path, modality, transform=True,
                                        patch_size=self.patch_size, n_patches_transform=30)

            self.TrainMetaData.append(dataset_img)

            val_path = self.val_paths[x]

            dataset_val = BatchImageGenerator(val_path, modality, transform=False,
                                        patch_size=self.patch_size, n_patches_transform=30)

            self.ValidMetaData.append(dataset_val)

        self.TestMetaData = BatchImageGenerator(self.test_path[0], modality, transform=False,
                                        patch_size=self.patch_size, n_patches_transform=30, is_test=True)

        self.batImageGenTests = torch.utils.data.DataLoader(self.TestMetaData, batch_size=self.batch_size,
                                                       sampler=SequentialSampler(self.TestMetaData),
                                                       num_workers=0,
                                                       pin_memory=False)


    def load_state_dict(self, state_dict=''):

        if state_dict:
            tmp=torch.load(state_dict)
            pretrained_dict=tmp['state']

            model_dict=self.network.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict={k:v for k, v in pretrained_dict.items() if
                             k in model_dict and v.size()==model_dict[k].size()}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.network.load_state_dict(model_dict)

    def configure(self):
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
        d2 = iter(self.batImageGenTrains[1])
        d3 = iter(self.batImageGenTrains[2])
        d4 = iter(self.batImageGenTrains[3])

        print("length of the three loaders >>>>> ", len(d1), len(d2), len(d3))
        ite = 0
        bs = 2

        for epoch in range(self.epochs):
            print("<<<<<<<<< epoch >>>>>>>>", epoch)
            d1 = iter(self.batImageGenTrains[0])
            d2 = iter(self.batImageGenTrains[1])
            d3 = iter(self.batImageGenTrains[2])
            d4 = iter(self.batImageGenTrains[3])

            for im in range(8):
                trains_d1, labels_d1 = next(d1)
                trains_d1 = trains_d1.squeeze(0)
                labels_d1 = labels_d1.squeeze(0)

                trains_d2, labels_d2 = next(d2)
                trains_d2 = trains_d2.squeeze(0)
                labels_d2 = labels_d2.squeeze(0)

                trains_d3, labels_d3 = next(d3)
                trains_d3 = trains_d3.squeeze(0)
                labels_d3 = labels_d3.squeeze(0)

                trains_d4, labels_d4 = next(d4)
                trains_d4 = trains_d4.squeeze(0)
                labels_d4 = labels_d4.squeeze(0)


                images_train_three_domains_shape = np.shape(trains_d1)
                total_batches_per_image = int(images_train_three_domains_shape[0] / bs)
                batch_index = 0
                print("total number of batches", total_batches_per_image)

                for batch in range(1, total_batches_per_image + 1):
                    print("-------in epoch", epoch + 1, "image number-----", im+1, "batch number-------", batch)
                    meta_train_loss=0.0
                    meta_val_loss=0.0
                    total_train_acc=[]
                    ite += 1

                    # meta-val data
                    index_val=np.random.choice(a=np.arange(0, len(self.batImageGenTrains)), size=1)[0]

                    for index in range(4):

                        if index==index_val:
                            continue

                        else:

                            if index == 0:
                                trains, labels = trains_d1, labels_d1

                            if index == 1:
                                trains, labels = trains_d2, labels_d2

                            if index == 2:
                                trains, labels = trains_d3, labels_d3

                            if index == 3:
                                trains, labels = trains_d4, labels_d4

                            images_train, labels_train = trains[batch_index: batch_index + bs, :, :, :, :], labels[batch_index: batch_index + bs, :, :, :]


                            images_train, labels_train = Variable(images_train, requires_grad=False).cuda(), Variable(labels_train, requires_grad=False).float().cuda()

                            outputs, predictions_train = self.network(x=images_train, meta_step_size=0.001, meta_loss=None,
                                                                      stop_gradient=False)

                            loss = self.loss_fn(outputs, labels_train.long())
                            meta_train_loss+=loss

                            predictions_train = predictions_train.cpu().data.numpy()
                            predicted_classes = np.argmax(predictions_train, axis=1)

                            train_acc = compute_accuracy(predictions=predicted_classes, labels=labels_train.cpu().data.numpy())
                            print("---------meta-train accuracy--------", train_acc, "for index ", index)

                            total_train_acc.append(train_acc)

                            del outputs

                    if index_val==0:
                        images_val, labels_val = trains_d1, labels_d1

                    if index_val==1:
                        images_val, labels_val = trains_d2, labels_d2

                    if index_val==2:
                        images_val, labels_val = trains_d3, labels_d3

                    if index_val==3:
                        images_val, labels_val = trains_d4, labels_d4

                #print("batch number >>>>", batch, "in image number>>>>>>", index_val, "in epoch", epoch + 1)

                    images_val = images_val.squeeze(0)
                    labels_val = labels_val.squeeze(0)

                    images_v, labels_v = images_val[batch_index: batch_index + bs, :, :, :, :], labels_val[batch_index: batch_index + bs, :, :, :]

                    images_v, labels_v = Variable(images_v, requires_grad=False).cuda(), Variable(labels_v, requires_grad=False).float().cuda()


                    outputs_val, predictions_val = self.network(x=images_v, meta_step_size=1, meta_loss=meta_train_loss, stop_gradient=False) #alpha

                    predictions_val = predictions_val.cpu().data.numpy()
                    predicted_classes = np.argmax(predictions_val, axis=1)

                    meta_val_loss=self.loss_fn(outputs_val, labels_v)

                    train_acc=compute_accuracy(predictions=predicted_classes, labels=labels_v.cpu().data.numpy())
                    print("---------meta-val accuracy--------", train_acc, "for index ", index_val)


                    total_train_acc.append(train_acc)
                    total_loss=meta_train_loss + meta_val_loss #beta


                    self.writer.add_scalar('Train/Loss', total_loss.data, ite)
                    self.writer.add_scalar('Train/Accuracy', np.mean(total_train_acc), ite)

                    # init the grad to zeros first
                    self.optimizer.zero_grad()

                    # backpropagate your network
                    total_loss.backward()

                    # optimize the parameters
                    self.optimizer.step()

                    print('epoch:',epoch+1, 'ite:', ite, 'batch:', batch, 'meta_train_loss:', meta_train_loss.cpu().data.numpy(), 'meta_val_loss:', meta_val_loss.cpu().data.numpy(),'lr:', self.scheduler.get_lr()[0], 'val_index', index_val)

                    del total_loss, outputs_val
                    torch.cuda.empty_cache()

                    if ite%self.test_every==0 and ite is not 0:
                        print('>>>>>>>>>VALIDATION<<<<<<<<')
                        self.test(ite)


                    batch_index += bs

            self.scheduler.step()


    def test(self, ite):
        self.network.eval()

        d1 = iter(self.batImageGenVals[0])
        d2 = iter(self.batImageGenVals[1])
        d3 = iter(self.batImageGenVals[2])
        d4 = iter(self.batImageGenVals[3])

        print("length of the three loaders >>>>> ", len(d1), len(d2), len(d3))
        accuracies = []
        val_losses = []

        for im in range(2):

            for index in range(4):
                if index == 0:
                    try:
                        trains, labels = next(d1)
                    except StopIteration:
                        d1 = iter(self.batImageGenVals[0])
                        trains, labels = next(d1)

                if index == 1:
                    try:
                        trains, labels = next(d2)
                    except StopIteration:
                        d2 = iter(self.batImageGenVals[1])
                        trains, labels = next(d2)

                if index == 2:
                    try:
                        trains, labels = next(d3)
                    except StopIteration:
                        d3 = iter(self.batImageGenVals[2])
                        trains, labels = next(d3)

                if index == 3:
                    try:
                        trains, labels = next(d4)
                    except StopIteration:
                        d3 = iter(self.batImageGenVals[3])
                        trains, labels = next(d4)

                bs = 10
                trains = trains.squeeze(0)
                labels = labels.squeeze(0)
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

                    print("----------accuracy val----------", accuracy_val)

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

            outfile = os.path.join('/path/to/images//code_unet/saved_models/', 'meta_learning_spine.tar')
            torch.save({'ite': ite, 'state': self.network.state_dict()}, outfile)

        self.writer.close()


    def unseen_fourth_mod(self):
        self.network.eval()

        ds = iter(self.batImageGenTests)
        print(" <<<<<< length of the test loader >>>>> ", len(ds))

        PATH = '/path/to/images//code_unet/saved_models/meta_learning.tar' # unseen modality
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

        it = 0
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
                    #print(np.sum(l_test))

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
                    del img_batch

                    #if it%100==0:
                    #    print("----------accuracy unseen fourth modality data----------", accuracy_val)
                    it = it + 1
                    acc_this_image.append(accuracy_val)

                #val_losses.append(val_loss_data)
                del outputs, val_loss

                batch_index += bs

            mean_acc_this_image = np.mean(acc_this_image)
            acc_four_image.append(mean_acc_this_image)
            print("----------accuracy this image----------", mean_acc_this_image)

        mean_acc = np.mean(accuracies)
        #mean_val_loss = np.mean(val_losses)
        print("---------- final test accuracy for unseen fourth modality data----------", mean_acc)

        std_acc = np.std(acc_four_image)
        print("--- the four values ---", acc_four_image)
        print("---------- final test std for unseen fourth modality data----------", std_acc)


###### call the model here
metalearn = MLDG()
metalearn.train()
