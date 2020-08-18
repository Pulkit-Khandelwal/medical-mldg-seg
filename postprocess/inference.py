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

class ModelBaseline:
    def __init__(self):
        self.batch_size = 1
        self.num_classes = 2
        self.unseen_index = 0 ####### Attention!! to need to change #######
        self.lr = 0.001
        self.inner_loops = 1200
        self.step_size = 20
        self.step_size_unpad = 20
        self.weight_decay = 0.00005
        self.momentum = 0.9
        self.state_dict = ''
        self.logs = 'logs'
        self.patch_size = 64
        self.test_every = 40
        self.test_unseen = 100
        self.epochs = 20

        self.writer = SummaryWriter(comment='inference')

        torch.set_default_tensor_type('torch.cuda.DoubleTensor')

        self.TrainMetaData = []
        self.ValidMetaData = []
        self.TestMetaData = []
        self.FewShotData = []

        self.count1 = 0
        self.count2 = 0
        self.count3 = 0


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

        self.train_paths = ['/home/pulkit/code_tmi/train_fat.txt',
                            '/home/pulkit/code_tmi/train_inn.txt',
                            '/home/pulkit/code_tmi/train_opp.txt',
                            '/home/pulkit/code_tmi/train_wat.txt']

        self.few_paths = ['/home/pulkit/code_tmi/train_fat.txt',
                            '/home/pulkit/code_tmi/train_fat.txt',
                            '/home/pulkit/code_tmi/train_fat.txt',
                            '/home/pulkit/code_tmi/train_fat.txt']

        self.val_paths = ['/home/pulkit/code_tmi/val_fat.txt',
                            '/home/pulkit/code_tmi/val_inn.txt',
                            '/home/pulkit/code_tmi/val_opp.txt',
                            '/home/pulkit/code_tmi/val_wat.txt']

        # 10 images
        self.test_paths = ['/path/to/imagesfull_test_lumbar.txt',
                            '/path/to/imagesfull_test_thoracic_lower.txt',
                            '/path/to/imagesfull_test_thoracic_middle.txt',
                            '/path/to/imagesfull_test_thoracic_upper.txt']

        # 4 images
        self.test_paths = ['/path/to/imagestest_lumbar.txt',
                            '/path/to/imagestest_thoracic_lower.txt',
                            '/path/to/imagestest_thoracic_middle.txt',
                            '/path/to/imagestest_thoracic_upper.txt']

        self.test_paths = ['/home/pulkit/june25/verse_test_fov.txt']

        for x in range(4):
            img_path = self.train_paths[x]

            dataset_img = BatchImageGenerator(img_path, modality, transform=True,
                                        patch_size=self.patch_size, n_patches_transform=30)

            self.TrainMetaData.append(dataset_img)

            val_path = self.val_paths[x]

            dataset_val = BatchImageGenerator(val_path, modality, transform=False,
                                        patch_size=self.patch_size, n_patches_transform=30)

            self.ValidMetaData.append(dataset_val)


        curr_test_modality = self.test_paths[self.unseen_index]
        self.TestMetaData = BatchImageGenerator(curr_test_modality, modality, transform=False,
                                        patch_size=self.patch_size, n_patches_transform=30, is_test=True)

        self.batImageGenTests = torch.utils.data.DataLoader(self.TestMetaData, batch_size=self.batch_size,
                                                       sampler=SequentialSampler(self.TestMetaData),
                                                       num_workers=0,
                                                       pin_memory=False)

    def configure(self):

        self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                   momentum=self.momentum)
        self.scheduler = lr_scheduler.StepLR(optimizer=self.optimizer, step_size=self.step_size, gamma=0.1)
        self.loss_fn = GeneralizedDiceLoss()


    def inference(self, model_path, test_file):
        self.network.eval()
        image_path = read_text_files(test_file)
        image_files = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
                       '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']
        print(image_path)

        ds = iter(self.batImageGenTests)
        print(" <<<<<< length of the test loader >>>>> ", len(ds))

        tmp=torch.load(model_path)
        pretrained_dict=tmp['state']
        print(">>>> which iteration was saved for the inference?", tmp['ite'])
        model_dict=self.network.state_dict()
        pretrained_dict={k:v for k, v in pretrained_dict.items() if k in model_dict and v.size()==model_dict[k].size()}
        model_dict.update(pretrained_dict)
        self.network.load_state_dict(model_dict)

        num_images = len(ds)
        accuracies = []
        val_losses = []
        accuracies_all_image = []
        it=0

        for im in range(num_images):
            trains, labels, patch_indices, img_size = next(ds)
            trains = trains.squeeze(0)
            labels = labels.squeeze(0)

            bs = 1
            images_train_three_domains_shape = np.shape(trains)
            total_batches_per_image = int(images_train_three_domains_shape[0] / bs)
            print("total number of batches >>>>>>>>", total_batches_per_image)
            batch_index = 0

            use_gpu = False
            if use_gpu:
                print('Using GPU')
                aggregated_results = torch.zeros(([self.num_classes] + img_size), dtype=torch.half)
                aggregated_nb_of_predictions = torch.zeros(([self.num_classes] + img_size), dtype=torch.half)
                add_for_nb_of_preds = torch.ones((self.patch_size, self.patch_size, self.patch_size), dtype=torch.half)
            else:
                print('Using CPU')
                aggregated_results = np.zeros(([self.num_classes] + img_size), dtype=np.half)
                aggregated_nb_of_predictions = np.zeros(([self.num_classes] + img_size), dtype=np.half)
                add_for_nb_of_preds = np.ones((self.patch_size, self.patch_size, self.patch_size), dtype=np.half)

            for batch in range(1, total_batches_per_image + 1):
                print(batch)
                curr_patch = patch_indices[batch_index]

                images_test, labels_test = trains[batch_index: batch_index + bs, :, :, :, :], labels[batch_index: batch_index + bs, :, :, :]

                images_test, labels_test = Variable(images_test, requires_grad=False).cuda(), Variable(labels_test, requires_grad=False).float().cuda()

                _, predictions = self.network(images_test, meta_step_size=0.001, meta_loss=None,
                                                        stop_gradient=False)

                sg = curr_patch[0]
                idx1_sg, idx2_sg = sg[0], sg[1]

                cr = curr_patch[1]
                idx1_cr,idx2_cr = cr[0], cr[1]

                ax = curr_patch[2]
                idx1_ax,idx2_ax = ax[0], ax[1]

                if use_gpu:
                    predictions_reshaped = torch.reshape(predictions.half(), (self.num_classes, self.patch_size, self.patch_size, self.patch_size))

                else:
                    predictions = predictions.cpu().data.numpy()
                    predicted_classes = np.argmax(predictions, axis=1)
                    predictions_reshaped = np.reshape(predictions, (self.num_classes, self.patch_size, self.patch_size, self.patch_size))

                aggregated_results[:, idx1_sg:idx2_sg, idx1_cr:idx2_cr, idx1_ax:idx2_ax] +=  predictions_reshaped # (num_classes, h,w,d)
                aggregated_nb_of_predictions[:, idx1_sg:idx2_sg, idx1_cr:idx2_cr, idx1_ax:idx2_ax] += add_for_nb_of_preds

                del predictions_reshaped

                batch_index += bs


            class_probabilities = aggregated_results / aggregated_nb_of_predictions
            predicted_classes = np.argmax(class_probabilities, axis=0)
            predicted_classes = np.float64(predicted_classes)

            # unpad prediction
            shape_lb = np.shape(predicted_classes)
            print('seeee', shape_lb)
            predicted_classes = predicted_classes[self.step_size_unpad: shape_lb[0]-self.step_size_unpad, self.step_size_unpad: shape_lb[1]-self.step_size_unpad, self.step_size_unpad: shape_lb[2]- self.step_size_unpad]

            # save the predicted labels
            filepath_name_pred = '/home/pulkit/code_tmi/output_miccai/verse/ubound/' + 'case_' + image_files[im] + '.nii.gz'
            save_nifti(predicted_classes, filepath_name_pred)

            # read the label for the corresponding image from the file
            _, label_data = read_nifti_miccai(image_path[im])

            print(np.shape(predicted_classes), np.shape(label_data))

            accuracy_val_full_image = compute_accuracy(predictions=predicted_classes, labels=label_data)
            accuracies_all_image.append(accuracy_val_full_image)
            print("---------- current image test accuracy by the normalization method ----------", accuracy_val_full_image)

            del label_data
            del predicted_classes

        print("---------- TO REPORT final average image test accuracy by the normalization method ----------", np.mean(accuracies_all_image))


###### call the model here
baseline = Model()

PATH = '/path/to/trained/model/trained_model.tar'
test_file = 'path/to/images/test_file.txt'

baseline.inference(PATH, test_file)
