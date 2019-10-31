###############################################################################
#
# Faster-RCNN is composed of 3 neural networks
#   Feature Network
#   - usually a well-known pre-trained image classifier such as VGG or ResNet50
#       minus a few layers
#   - to generate good features from the images
#   Region Proposal Network (RPN)
#   - usually a simple network with 3 convolutional layers
#   - to generate a number of bounding boxes called Region of interests (ROIs)
#       that has high probability of containing any object
#   Detection Network (RCNN network)
#   - takes input from both the feature network and RPN, and generates the
#       final class and bounding box
#
#
# based on the work by yhenon (https://github.com/yhenon/keras-frcnn/) and
# RockyXu66 (https://github.com/RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras),
# - converted to use tensorflow.keras
# - refactored to be used as a library, following tensorflow.keras Model API
###############################################################################

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add, Input, InputSpec, Dense, Activation, Dropout
from tensorflow.keras.layers import Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, TimeDistributed
from tensorflow.keras import optimizers
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.backend import categorical_crossentropy
import tensorflow.keras.utils as utils
import tensorflow.keras.backend as K

import numpy as np
import pandas as pd
import cv2
import time
import random
import math
import copy
import os
import sys
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imgaug as ia
from matplotlib import pyplot as plt

from numba import jit

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# tf.config.optimizer.set_jit(True)
DEBUG = False


class FRCNN():  # class FRCNN(tf.keras.Model):
    def __init__(
        self,
        base_net_type='resnet50', base_trainable=False,
        num_classes=10, input_shape=(None, None, 3),
        num_rois=256, num_anchors=9
    ):
        # super(FRCNN, self).__init__(name='frcnn')
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_rois = num_rois
        self.base_net_type = base_net_type

        # Checking of inputs for Feature Network (Base Net),
        # allow some flexibility in name of base_net
        base_net_type = base_net_type.lower()
        if ('resnet' in base_net_type):
            base_net_type = 'resnet50'
        elif ('vgg' in base_net_type):
            base_net_type = 'vgg'

        if (base_net_type != 'resnet50' and base_net_type != 'vgg'):
            print("Only resnet50 and vgg are currently supported as base models")
            raise ValueError

        elif (base_net_type == 'resnet50'):
            from tensorflow.keras.applications import ResNet50 as fn

        elif (base_net_type == 'vgg'):
            from tensorflow.keras.applications import VGG16 as fn

        img_input = Input(shape=input_shape)
        roi_input = Input(shape=(None, 4))

        # Define Feature Network
        # Assume we will always use pretrained weights for Feature Network for now
        base_net = fn(weights='imagenet', include_top=False, input_tensor=img_input)

        for layer in base_net.layers:
            layer.trainable = base_trainable
            layer._name = layer.name + "a"  # prevent duplicate layer name

        # For VGG, the last max pooling layer in VGGNet is also removed
        if (base_net_type == 'vgg'):
            # base_net.layers.pop() # does not work - https://github.com/tensorflow/tensorflow/issues/22479
            feature_network = base_net.layers[-2].output
            num_features = 512
        else:
            feature_network = base_net.outputs[0]
            num_features = 1024

        self.feature_network = feature_network

        # Define RPN, built upon the base layers
        rpn = _rpn(feature_network, num_anchors)
        classifier = _classifier(
            feature_network, roi_input, num_rois, nb_classes=num_classes,
            trainable=True, base_net_type=base_net_type)
        self.model_rpn = Model(img_input, rpn[:2])
        self.model_classifier = Model([img_input, roi_input], classifier)

        # this will be the model that holds both the RPN and the classifier, used to load/save weights for the models
        self.model_all = Model([img_input, roi_input], rpn[:2] + classifier)

        # Create models that will be used for predictions
        roi_input = Input(shape=(num_rois, 4))
        feature_map_input = Input(shape=(None, None, num_features))
        p_classifier = _classifier(
            feature_map_input, roi_input, num_rois, nb_classes=num_classes,
            trainable=True, base_net_type=base_net_type)
        self.predict_rpn = Model(img_input, rpn)
        self.predict_classifier = Model([feature_map_input, roi_input], p_classifier)

        # return model_all

    def summary(self, line_length=None, positions=None, print_fn=None):
        """Prints a string summary of the overall FRCNN network

        Arguments:
            line_length: Total length of printed lines
                (e.g. set this to adapt the display to different
                terminal window sizes).
            positions: Relative or absolute positions of log elements
                in each line. If not provided,
                defaults to `[.33, .55, .67, 1.]`.
            print_fn: Print function to use. Defaults to `print`.
                It will be called on each line of the summary.
                You can set it to a custom function
                in order to capture the string summary.
        Raises:
            ValueError: if `summary()` is called before the model is built.
        """
        return self.model_all.summary(line_length=line_length, positions=positions, print_fn=print_fn)

    def compile(
            self,
            optimizer=None,
            loss=None,
            **kwargs):
        """Configures the model for training.

        Arguments:
            optimizer: Array of String (name of optimizer), array of optimizer instance,
                String (name of optimizer) or optimizer instance
                See `tf.keras.optimizers`. If it is not an array, the same optimizer will
                be used for all submodels. Otherwise index 0-rpn, 1-classifier, 2-all.
                Set to None to use defaults
                Example: [optimizers.Adam(lr=1e-5), optimizers.Adam(lr=1e-5), 'sgd']

            loss: Array of String (name of objective function), array of objective function,
                String (name of objective function), objective function or
                `tf.losses.Loss` instance. See `tf.losses`. If it is not an array,
                the same loss will be used for all submodels. Otherwise, index
                0-rpn, 1-classifier, 2-all.
                If the model has multiple outputs, you can use a different loss on each output by
                passing a dictionary or a list of losses. The loss value that will
                be minimized by the model will then be the sum of all individual
                losses.
                Set to None to use defaults

            **kwargs: Any additional arguments.
        Raises:
            ValueError: In case of invalid arguments for
                `optimizer`, `loss`, `metrics` or `sample_weight_mode`.
        """

        # Allow user to override defaults
        if optimizer is not None:
            # Ideally optimizer settings should be specified individually
            if (isinstance(optimizer, list)):
                if (len(optimizer) != 3):
                    print("Length of list for optimizer should be 3")
                    raise ValueError
                else:
                    optimizer_rpn = optimizer[0]
                    optimizer_classifier = optimizer[1]
                    optimizer_all = optimizer[2]
            # Use same optimizer for all
            else:
                optimizer_rpn = optimizer
                optimizer_classifier = optimizer
                optimizer_all = optimizer
        # Use defaults for optimizers if not specified
        else:
            optimizer_rpn = optimizers.Adam(lr=1e-5)
            optimizer_classifier = optimizers.Adam(lr=1e-5)
            optimizer_all = 'sgd'

        if loss is not None:
            if (isinstance(loss, list)):
                if (len(loss) != 3):
                    print("Length of list for loss should be 3")
                    raise ValueError
                else:
                    loss_rpn = loss[0]
                    loss_classifier = loss[1]
                    loss_all = loss[2]
            # Use same loss function for all
            else:
                loss_rpn = loss
                loss_classifier = loss
        # Use defaults for loss if not specified
        else:
            loss_rpn = [rpn_loss_cls(self.num_anchors), rpn_loss_regr(self.num_anchors)]
            loss_classifier = [class_loss_cls, class_loss_regr(self.num_classes - 1)]
            loss_all = 'mae'

        self.model_rpn.compile(optimizer=optimizer_rpn, loss=loss_rpn)
        self.model_classifier.compile(
            optimizer=optimizer_classifier,
            loss=loss_classifier, metrics={'dense_class_{}'.format(self.num_classes): 'accuracy'})

        self.model_all.compile(optimizer=optimizer_all, loss=loss_all)

        self.predict_rpn.compile(optimizer='sgd', loss='mse')
        self.predict_classifier.compile(optimizer='sgd', loss='mse')

    def fit_generator(
            self,
            generator,
            steps_per_epoch=1000,
            epochs=1,
            verbose=1,
            initial_epoch=-1,

            class_mapping=None,
            target_size=-1,                # length of shorter size
            anchor_box_scales=[128, 256, 512],
            anchor_box_ratios=[[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]],
            std_scaling=4.0,                           # for scaling of standard deviation
            classifier_regr_std=[8.0, 8.0, 4.0, 4.0],   #
            classifier_min_overlap=0.1,               # repo values
            classifier_max_overlap=0.5,               # repo values
            rpn_stride=16,                              # stride at the RPN (this depends on the network configuration)

            model_path='./frcnn.hdf5',
            csv_path="./frcnn.csv"
    ):
        """Fits the model on data yielded batch-by-batch by FRCNNGenerator.
        Will automatically save model and csv to the specified paths
        model_path and csv_path respectively.
        If file at model_path exists, will automatically resume training
        if initial_epoch is set to -1. Otherwise, will prompt user to resume
        training

        Arguments:
            generator: Generator that was created via FRCNNGenerator
                The generator is expected to loop over its data
                indefinitely. An epoch finishes when `steps_per_epoch`
                batches have been seen by the model.
            steps_per_epoch: Total number of steps (batches of samples)
                to yield from `generator` before declaring one epoch
                finished and starting the next epoch.
            epochs: Integer, total number of iterations on the data.
            verbose: Verbosity mode. 0 = Silent, 1 = progress bar
            initial_epoch: Integer. Epoch at which to start training
                (useful for resuming a previous training run)
            model_path: Path for saving model hdf5. Also used to resume training
            csv_path: Path for saving training csv. Also used to resume training

            class_mapping: Class mapping based on training set. This is the third output from parseAnnotationFile()
            target_size: Integer. Shorter-side length. Used for image resizing based on the shorter length
        Returns:
            None
        Raises:
            ValueError: In case the generator yields data in an invalid format.
            """
        epoch_length = steps_per_epoch
        iter_num = 0

        losses = np.zeros((epoch_length, 5))
        rpn_accuracy_rpn_monitor = []
        rpn_accuracy_for_epoch = []

        best_loss = np.Inf

        # input validation
        if (class_mapping is None):
            print("class_mapping should not be None")
            raise ValueError
        elif (target_size < 0):
            print("target_size (shorter-side size) must be a positive integer")
            raise ValueError

        print()
        # let's check if model file exists
        if not os.path.isfile(model_path):

            print('Starting training')
            initial_epoch = 0

            # Create the record.csv file to record losses, acc and mAP
            record_df = pd.DataFrame(
                columns=[
                    'mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls',
                    'loss_rpn_regr', 'loss_class_cls', 'loss_class_regr', 'curr_loss', 'elapsed_time', 'mAP'])
        else:

            # if setting is not to continue training and file exists, confirm with user again,
            # before overwriting file, just in case
            if (initial_epoch != -1):
                ask = input('File %s exists. Continue training? [Y/N]' % (model_path))
                if (ask.lower() in ['y', 'yes', 'ya']):
                    initial_epoch = -1
                else:
                    print('Restarting training and overwriting %s and %s' % (model_path, csv_path))

            if (initial_epoch == -1):
                # If this is a continued training, load the trained model from before
                print('Continuing training based on previous trained model')
                print('Loading weights from {}'.format(model_path))
                self.model_rpn.load_weights(model_path, by_name=True)
                self.model_classifier.load_weights(model_path, by_name=True)

                record_df = pd.read_csv(csv_path)
                initial_epoch = len(record_df)
                # for debugging
                # r_mean_overlapping_bboxes = record_df['mean_overlapping_bboxes']
                # r_class_acc = record_df['class_acc']
                # r_loss_rpn_cls = record_df['loss_rpn_cls']
                # r_loss_rpn_regr = record_df['loss_rpn_regr']
                # r_loss_class_cls = record_df['loss_class_cls']
                # r_loss_class_regr = record_df['loss_class_regr']
                # r_elapsed_time = record_df['elapsed_time']
                # r_mAP = record_df['mAP']

                r_curr_loss = record_df['curr_loss']
                best_loss = np.min(r_curr_loss)

                if verbose:
                    print('Already trained %dK batches' % (len(record_df)))

        ####
        start_time = time.time()
        total_epoch = initial_epoch + epochs    # We might be resuming training, so we will start with initial_epoch
        for epoch_num in range(epochs):

            progbar = utils.Progbar(epoch_length)
            print('Epoch {}/{}'.format(initial_epoch + 1 + epoch_num, total_epoch))

            while True:
                try:
                    if len(rpn_accuracy_rpn_monitor) == epoch_length and verbose:
                        mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                        rpn_accuracy_rpn_monitor = []
                        if mean_overlapping_bboxes == 0:
                            print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

                    # Generate X (x_img) and label Y ([y_rpn_cls, y_rpn_regr])
                    X, Y, img_data, debug_img, debug_num_pos = next(generator)
                    if DEBUG:
                        print("DEBUG", img_data['filepath'])

                    # Train rpn model and get loss value [_, loss_rpn_cls, loss_rpn_regr]
                    loss_rpn = self.model_rpn.train_on_batch(X, Y)

                    # Get predicted rpn from rpn model [rpn_cls, rpn_regr]
                    P_rpn = self.model_rpn.predict_on_batch(X)

                    # R: bboxes (shape=(300,4))
                    # Convert rpn layer to roi bboxes
                    R = rpn_to_roi(
                        P_rpn[0], P_rpn[1],
                        std_scaling, anchor_box_ratios, anchor_box_scales, rpn_stride,
                        use_regr=True, overlap_thresh=0.7, max_boxes=300)

                    # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                    # X2: bboxes that iou > C.classifier_min_overlap for all gt bboxes in 300 non_max_suppression bboxes
                    # Y1: one hot code for bboxes from above => x_roi (X)
                    # Y2: corresponding labels and corresponding gt bboxes
                    X2, Y1, Y2, IouS = calc_iou(
                        R, img_data, [classifier_min_overlap, classifier_max_overlap],
                        target_size, rpn_stride, class_mapping, classifier_regr_std)

                    if DEBUG:
                        print("DEBUG calc_iou (inputs)", classifier_min_overlap, classifier_max_overlap, target_size, rpn_stride, class_mapping, classifier_regr_std)
                        print("DEBUG calc_iou", X2, Y1, Y2, IouS)

                    # If X2 is None means there are no matching bboxes
                    if X2 is None:
                        rpn_accuracy_rpn_monitor.append(0)
                        rpn_accuracy_for_epoch.append(0)
                        continue

                    # Find out the positive anchors and negative anchors
                    neg_samples = np.where(Y1[0, :, -1] == 1)
                    pos_samples = np.where(Y1[0, :, -1] == 0)

                    if len(neg_samples) > 0:
                        neg_samples = neg_samples[0]
                    else:
                        neg_samples = []

                    if len(pos_samples) > 0:
                        pos_samples = pos_samples[0]
                    else:
                        pos_samples = []

                    rpn_accuracy_rpn_monitor.append(len(pos_samples))
                    rpn_accuracy_for_epoch.append((len(pos_samples)))

                    if self.num_rois > 1:
                        # If number of positive anchors is larger than 4//2 = 2, randomly choose 2 pos samples
                        if len(pos_samples) < self.num_rois//2:
                            selected_pos_samples = pos_samples.tolist()
                        else:
                            selected_pos_samples = np.random.choice(pos_samples, self.num_rois//2, replace=False).tolist()

                        # Randomly choose (num_rois - num_pos) neg samples
                        try:
                            selected_neg_samples = np.random.choice(neg_samples, self.num_rois - len(selected_pos_samples), replace=False).tolist()
                        except ValueError:
                            try:
                                selected_neg_samples = np.random.choice(neg_samples, self.num_rois - len(selected_pos_samples), replace=True).tolist()
                            except Exception as e:
                                if DEBUG: print(e)
                                # The neg_samples is [[1 0 ]] only, therefore there's no negative sample
                                continue

                        # Save all the pos and neg samples in sel_samples
                        sel_samples = selected_pos_samples + selected_neg_samples
                    else:
                        # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                        selected_pos_samples = pos_samples.tolist()
                        selected_neg_samples = neg_samples.tolist()
                        if np.random.randint(0, 2):
                            sel_samples = random.choice(neg_samples)
                        else:
                            sel_samples = random.choice(pos_samples)

                    # training_data: [X, X2[:, sel_samples, :]]
                    # labels: [Y1[:, sel_samples, :], Y2[:, sel_samples, :]]
                    #  X                     => img_data resized image
                    #  X2[:, sel_samples, :] => num_rois (4 in here) bboxes which contains selected neg and pos
                    #  Y1[:, sel_samples, :] => one hot encode for num_rois bboxes which contains selected neg and pos
                    #  Y2[:, sel_samples, :] => labels and gt bboxes for num_rois bboxes which contains selected neg and pos
                    loss_class = self.model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

                    losses[iter_num, 0] = loss_rpn[1]
                    losses[iter_num, 1] = loss_rpn[2]

                    losses[iter_num, 2] = loss_class[1]
                    losses[iter_num, 3] = loss_class[2]
                    losses[iter_num, 4] = loss_class[3]

                    iter_num += 1

                    progbar.update(
                        iter_num, [
                            ('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                            ('final_cls', np.mean(losses[:iter_num, 2])), ('final_regr', np.mean(losses[:iter_num, 3]))
                        ])

                    if iter_num == epoch_length:
                        loss_rpn_cls = np.mean(losses[:, 0])
                        loss_rpn_regr = np.mean(losses[:, 1])
                        loss_class_cls = np.mean(losses[:, 2])
                        loss_class_regr = np.mean(losses[:, 3])
                        class_acc = np.mean(losses[:, 4])

                        mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                        rpn_accuracy_for_epoch = []

                        if verbose:
                            print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                            print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                            print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                            print('Loss RPN regression: {}'.format(loss_rpn_regr))
                            print('Loss Detector classifier: {}'.format(loss_class_cls))
                            print('Loss Detector regression: {}'.format(loss_class_regr))
                            print('Total loss: {}'.format(loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr))
                            print('Elapsed time: {}'.format(time.time() - start_time))
                            elapsed_time = (time.time()-start_time)/60

                        curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                        iter_num = 0
                        start_time = time.time()

                        if curr_loss < best_loss:
                            if verbose:
                                print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                            best_loss = curr_loss
                            self.model_all.save_weights(model_path)

                        new_row = {
                            'mean_overlapping_bboxes': round(mean_overlapping_bboxes, 3),
                            'class_acc': round(class_acc, 3),
                            'loss_rpn_cls': round(loss_rpn_cls, 3),
                            'loss_rpn_regr': round(loss_rpn_regr, 3),
                            'loss_class_cls': round(loss_class_cls, 3),
                            'loss_class_regr': round(loss_class_regr, 3),
                            'curr_loss': round(curr_loss, 3),
                            'elapsed_time': round(elapsed_time, 3),
                            'mAP': 0}

                        record_df = record_df.append(new_row, ignore_index=True)
                        record_df.to_csv(csv_path, index=0)

                        break

                except Exception as e:
                    print('Exception: {}'.format(e))
                    continue

        print('-- Training complete, exiting.')
        return None

    def load_config(
        self,
        anchor_box_scales=[128, 256, 512],
        anchor_box_ratios=[[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]],
        std_scaling=4.0,
        rpn_stride=16,                              # stride at the RPN (this depends on the network configuration)
        num_rois=32,
        target_size=600,
        img_channel_mean=[103.939, 116.779, 123.68],
        img_scaling_factor=1,
        classifier_regr_std=[8.0, 8.0, 4.0, 4.0],
    ):
        """Loads configuration settings for FRCNN model.
        These will be used for predictions

        Arguments:
            anchor_box_scales: Anchor box scales array
            anchor_box_ratios: Anchor box ratios array
            std_scaling: For scaling of standard deviation
            rpn_stride: RPN stride. This should be the same as what was passed into the generator
            num_rois: number of regions of interest to be used
            target_size: Integer. Shorter-side length. Used for image resizing based on the shorter length
            img_channel_mean: image channel-wise (RGB) mean to subtract for standardisation
            img_scaling_factor: scaling factor to divide by, for standardisation
            classifier_regr_std: For scaling of standard deviation for classifier regression for x,y,w,h

        Returns:
            None

        """
        self.anchor_box_scales = anchor_box_scales
        self.anchor_box_ratios = anchor_box_ratios
        self.std_scaling = std_scaling
        self.rpn_stride = rpn_stride
        self.im_size = target_size
        self.img_channel_mean = img_channel_mean
        self.img_scaling_factor = 1

        return None

    def load_weights(self, filepath):
        """Loads all layer weights, from an HDF5 file.

        Weights are loaded with 'by_name' true, meaning that weights are loaded into
        layers only if they share the same name. This assumes a single HDF5 file and
        consistent layer names

        If it is desired to load weights with 'by_name' is False, and load
        weights based on the network's topology, please access the individual embedded
        sub-models in this class. eg frcnn.model_rpn.load_weights(filepath, by_name=False)

        Arguments:
             filepath: String, path to the weights file to load. For weight files in
                TensorFlow format, this is the file prefix (the same as was passed
                to 'save_weights').
        Returns:
            None
        """
        if (not os.path.isfile(filepath)):
            raise FileNotFoundError('File does not exist: %s ' % filepath)

        self.model_rpn.load_weights(filepath, by_name=True)
        self.model_classifier.load_weights(filepath, by_name=True)

        self.predict_rpn.load_weights(filepath, by_name=True)
        self.predict_classifier.load_weights(filepath, by_name=True)
        return None

    def predict(
        self,
        x,                            #
        verbose=2,                    #
        class_mapping=None,
        bbox_threshold=0.7,
        overlap_thres=0.2
    ):   #
        """Generates output predictions for the input samples.
        Computation is done in batches.
        Arguments:
            x: Input samples. This should be a list of img data
                or a list of dict containing groundtruth bounding
                boxes (key=bboxes) and path of the image (key=filepath)
            verbose: Verbosity mode.
                0 = silent. 1 = print results. 2 = print results and show images
            class_mapping: Class mapping based on training set
            bbox_threshold: If box classification value is less than this, we will ignore that box
            overlap_thres: Non-maximum suppression setting. If overlap > overlap_thres, we will remove the box
        Returns:
            Numpy array(s) of predictions.
        """

        return self._loopSamplesAndPredictOrEvaluate(
            x, class_mapping,
            bbox_threshold, overlap_thresh=overlap_thres, verbose=verbose)

    def evaluate(
        self,
        x=None,
        verbose=2,
        class_mapping=None,
        overlap_thresh=0.5
    ):
        """Returns the mean average precision (mAP) for the model in test mode.
        Computation is done in batches.

        Arguments:
            x: Input samples. This should be a list of dict containing
                groundtruth bounding boxes (key=bboxes) and path of the image (key=filepath)
            verbose: Verbosity mode.
                0 = silent. 1 = print results. 2 = print results and show images
            class_mapping: Class mapping based on training set
            overlap_thres: Non-maximum suppression setting. If overlap > overlap_thres, we will remove the box

        Returns:
            List of mAPs
        Raises:
            ValueError: in case of invalid arguments.
        """

        return self._loopSamplesAndPredictOrEvaluate(
            x, class_mapping, overlap_thresh=overlap_thresh, verbose=verbose, mode='evaluate')

    # from profilehooks import profile
    # @profile
    def _loopSamplesAndPredictOrEvaluate(
        self, samples, class_mapping, bbox_threshold=None,
        overlap_thresh=0.5, verbose=1, mode='predict'
    ):

        visualise = (verbose > 1)

        from sklearn.metrics import average_precision_score
        # predicts = []
        T = {}
        P = {}
        # mAPs = []
        output = []

        i = 1
        isImgData = True

        if isinstance(samples[0], dict):
            isImgData = False

        # For evaluation of mAP, we will need the ground-truth bboxes
        if (mode == 'evaluate' and isImgData):
            print('For evaluate, please provide input as array of dict containing bboxes and filepath')
            raise ValueError

        if (class_mapping is None):
            print("class_mapping should not be None")
            raise ValueError

        # Switch key and value for class_mapping 'Person': 0 --> 0: 'Person'
        class_mapping = {v: k for k, v in class_mapping.items()}

        # Assign color to each
        class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

        def get_map(pred, gt, f):
            T = {}
            P = {}
            fx, fy = f

            for bbox in gt:
                bbox['bbox_matched'] = False

            pred_probs = np.array([s['prob'] for s in pred])
            box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

            for box_idx in box_idx_sorted_by_prob:
                pred_box = pred[box_idx]
                pred_class = pred_box['class']
                pred_x1 = pred_box['x1']
                pred_x2 = pred_box['x2']
                pred_y1 = pred_box['y1']
                pred_y2 = pred_box['y2']
                pred_prob = pred_box['prob']
                if pred_class not in P:
                    P[pred_class] = []
                    T[pred_class] = []
                P[pred_class].append(pred_prob)
                found_match = False

                for gt_box in gt:
                    gt_class = gt_box['class']
                    gt_x1 = gt_box['x1']/fx
                    gt_x2 = gt_box['x2']/fx
                    gt_y1 = gt_box['y1']/fy
                    gt_y2 = gt_box['y2']/fy
                    gt_seen = gt_box['bbox_matched']
                    if gt_class != pred_class:
                        continue
                    if gt_seen:
                        continue
                    iou_map = iou((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
                    if iou_map >= 0.5:
                        found_match = True
                        gt_box['bbox_matched'] = True
                        break
                    else:
                        continue

                T[pred_class].append(int(found_match))

            for gt_box in gt:
                if not gt_box['bbox_matched']:  # and not gt_box['difficult']:
                    if gt_box['class'] not in P:
                        P[gt_box['class']] = []
                        T[gt_box['class']] = []

                    T[gt_box['class']].append(1)
                    P[gt_box['class']].append(0)

            return T, P

        def calcPredictOutput():
            # Calculate real coordinates on original image and save coordinates, and (key and prob) separately
            (real_x1, real_y1, real_x2, real_y2) = _get_real_coordinates(ratio, x1, y1, x2, y2)
            all_pos.append((real_x1, real_y1, real_x2, real_y2))
            all_dets.append((key, 100*new_probs[jk]))

            if (visualise):
                cv2.rectangle(
                    img_original, (real_x1, real_y1), (real_x2, real_y2),
                    (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])), 4)
                textLabel = '{}: {}'.format(key, int(100*new_probs[jk]))

                # (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
                textOrg = (real_x1, real_y1-0)
                y = real_y1+10 if real_y1 < 10 else real_y1
                textOrg = (real_x1, y)
                cv2.putText(
                    img_original, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX,
                    0.5, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.putText(
                    img_original, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX,
                    0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        def calcEvalOutput():
            # Save coordinates, class and probability
            det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs[jk]}
            all_dets.append(det)

        calcOutput = calcPredictOutput if mode == 'predict' else calcEvalOutput   # check once, instead of every loop

        overall_st = time.time()
        for data in samples:
            if verbose and not isImgData:
                print('{}/{} - {}'.format(i, len(samples), data['filepath']))
            i = i + 1
            st = time.time()

            # convert image
            img_original = data if isImgData else cv2.imread(data['filepath'])
            img, ratio, fx, fy = _format_img(img_original, self.img_channel_mean, self.img_scaling_factor, self.im_size)
            img = np.transpose(img, (0, 2, 3, 1))

            # get output layer Y1, Y2 from the RPN and the feature maps F
            # Y1: y_rpn_cls, Y2: y_rpn_regr
            [Y1, Y2, F] = self.predict_rpn.predict(img)  #

            # Get bboxes by applying NMS
            # R.shape = (300, 4)
            R = rpn_to_roi(
                Y1, Y2, self.std_scaling, self.anchor_box_ratios, self.anchor_box_scales, self.rpn_stride,
                use_regr=True, overlap_thresh=0.7)

            # convert from (x1,y1,x2,y2) to (x,y,w,h)
            R[:, 2] -= R[:, 0]
            R[:, 3] -= R[:, 1]

            # apply the spatial pyramid pooling to the proposed regions
            bboxes = {}
            probs = {}

            for jk in range(R.shape[0]//self.num_rois + 1):
                ROIs = np.expand_dims(R[self.num_rois*jk:self.num_rois*(jk+1), :], axis=0)
                if ROIs.shape[1] == 0:
                    break

                if jk == R.shape[0]//self.num_rois:
                    # pad R
                    curr_shape = ROIs.shape
                    target_shape = (curr_shape[0], self.num_rois, curr_shape[2])
                    ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                    ROIs_padded[:, :curr_shape[1], :] = ROIs
                    ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                    ROIs = ROIs_padded

                [P_cls, P_regr] = self.predict_classifier.predict([F, ROIs])

                # Calculate bboxes coordinates on resized image
                for ii in range(P_cls.shape[1]):

                    # Ignore 'bg' class
                    if ((bbox_threshold is not None and np.max(P_cls[0, ii, :]) < bbox_threshold) or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1)):
                        continue

                    # Get class name
                    cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                    if cls_name not in bboxes:
                        bboxes[cls_name] = []
                        probs[cls_name] = []

                    (x, y, w, h) = ROIs[0, ii, :]

                    cls_num = np.argmax(P_cls[0, ii, :])
                    try:
                        (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                        tx /= self.classifier_regr_std[0]
                        ty /= self.classifier_regr_std[1]
                        tw /= self.classifier_regr_std[2]
                        th /= self.classifier_regr_std[3]
                        x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
                    except Exception as e:
                        if DEBUG: print(e)
                        pass
                    bboxes[cls_name].append([self.rpn_stride*x, self.rpn_stride*y, self.rpn_stride*(x+w), self.rpn_stride*(y+h)])
                    probs[cls_name].append(np.max(P_cls[0, ii, :]))

            all_dets = []
            all_pos = []
            for key in bboxes:
                bbox = np.array(bboxes[key])

                # Apply non-max-suppression on final bboxes to get the output bounding boxes
                new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=overlap_thresh)
                for jk in range(new_boxes.shape[0]):
                    (x1, y1, x2, y2) = new_boxes[jk, :]

                    # Upate all_dets and all_poos
                    calcOutput()

            if verbose:
                print('Elapsed time = {}'.format(time.time() - st))

            if mode == 'predict':
                if verbose:
                    print(all_dets)
                if visualise:
                    # plt.figure(figsize=(10,10))
                    plt.figure()
                    plt.grid()
                    plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
                    plt.show()
                output.append((all_dets, all_pos))    # store all predictions and their positions for each image
            else:

                t, p = get_map(all_dets, data['bboxes'], (fx, fy))
                for key in t.keys():
                    if key not in T:
                        T[key] = []
                        P[key] = []
                    T[key].extend(t[key])
                    P[key].extend(p[key])
                all_aps = []
                for key in T.keys():
                    ap = average_precision_score(T[key], P[key])
                    all_aps.append(ap)
                    if verbose:
                        print('{} AP: {}'.format(key, ap))
                if verbose:
                    print('mAP = {}'.format(np.nanmean(np.array(all_aps))))
                    print()
                output.append(np.nanmean(np.array(all_aps)))

        if verbose:
            print('Total elapsed time = {}'.format(time.time() - overall_st))
        output = np.asarray(output)
        return output


###############################################################################
def _get_real_coordinates(ratio, x1, y1, x2, y2):

    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2, real_y2)


def _format_img(img, img_channel_mean, img_scaling_factor, target_size):
    """ format image for prediction or mAP calculation. Resize original image to target_size
    Arguments:
        img: cv2 image
        img_channel_mean: image channel-wise (RGB) mean to subtract for standardisation
        img_scaling_factor: scaling factor to divide by, for standardisation
        target_size: shorter-side length. Used for image resizing based on the shorter length

    Returns:
        img: Scaled and normalized image with expanding dimension
        ratio: img_min_side / original min side eg img_min_side / width if width <= height
        fx: ratio for width scaling (original width / new width)
        fy: ratio for height scaling (original height/ new height)
    """

    """ resize image based on config """
    img_min_side = float(target_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side/width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side/height
        new_width = int(ratio * width)
        new_height = int(img_min_side)

    fx = width/float(new_width)
    fy = height/float(new_height)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    """ format image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= img_channel_mean[0]
    img[:, :, 1] -= img_channel_mean[1]
    img[:, :, 2] -= img_channel_mean[2]
    img /= img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img, ratio, fx, fy


def _get_img_output_length(width, height, base_net_type='resnet50'):
    b = base_net_type

    def get_output_length(input_length, b):
        if (b == 'resnet50'):
            # zero_pad
            input_length += 6
            # apply 4 strided convolutions
            filter_sizes = [7, 3, 1, 1]
            stride = 2
            for filter_size in filter_sizes:
                input_length = (input_length - filter_size + stride) // stride

            return input_length
        else:
            return input_length//16

    return get_output_length(width, b), get_output_length(height, b)


def _rpn(base_layers, num_anchors):
    # common layer fed to 2 layers
    # - x_class for classification (is object in bounding box?)
    # - x_regr for bounding box regression (ROIs)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)
    return [x_class, x_regr, base_layers]


def _classifier(base_layers, input_rois, num_rois, nb_classes=4, trainable=True, base_net_type='resnet50'):

    if (base_net_type == 'resnet50'):
        pooling_regions = 14
        input_shape = (num_rois, pooling_regions, pooling_regions, 1024)
        out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois], name='roi_pooling_conv')

        # out = _classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)
        trainable = True
        out = _conv_block_td(out_roi_pool, 3, [512, 512, 2048], stage=5, block='a', input_shape=input_shape, strides=(2, 2), trainable=trainable)
        out = _identity_block_td(out, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
        out = _identity_block_td(out, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)
        out = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(out)

        out = TimeDistributed(Flatten())(out)
    else:
        pooling_regions = 7
        input_shape = (num_rois, pooling_regions, pooling_regions, 512)
        out_roi_pool = RoiPoolingConv(pooling_regions, num_rois, name='roi_pooling_conv')([base_layers, input_rois])

        # flatten convolution layer and connect to 2 FC with dropout
        # print(out_roi_pool.shape)
        out = TimeDistributed(Flatten(name='flatten'), name='time_distributed')(out_roi_pool)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc1'), name='time_distributed_1')(out)
        out = TimeDistributed(Dropout(rate=0.5), name='time_distributed_2')(out)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc2'), name='time_distributed_3')(out)
        out = TimeDistributed(Dropout(rate=0.5), name='time_distributed_4')(out)

    # There are two output layer
    # out_class: softmax acivation function for classification of the class name of the object
    # out_regr: linear activation function for bboxes coordinates regression
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]


def _conv_block_td(input_tensor, kernel_size, filters, stage, block, input_shape, strides=(2, 2), trainable=True):

    # conv block time distributed

    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Conv2D(nb_filter1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), input_shape=input_shape, name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter3, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2c', trainable=trainable)(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    shortcut = TimeDistributed(Conv2D(nb_filter3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '1')(input_tensor)
    shortcut = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def _identity_block_td(input_tensor, kernel_size, filters, stage, block, trainable=True):

    # identity block time distributed

    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Conv2D(nb_filter1, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter2, (kernel_size, kernel_size), trainable=trainable, kernel_initializer='normal', padding='same'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter3, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2c')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x


###############################################################################
# Definition for custom layers
class RoiPoolingConv(tf.keras.layers.Layer):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, rows, cols, channels)`
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    '''
    def __init__(self, pool_size, num_rois, **kwargs):

        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):

        assert(len(x) == 2)

        # x[0] is image with shape (rows, cols, channels)
        img = x[0]

        # x[1] is roi with shape (num_rois,4) with ordering (x,y,w,h)
        rois = x[1]

        # input_shape = K.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):

            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            # Resized roi of the image to pooling size (7x7)
            rs = tf.image.resize_images(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = K.concatenate(outputs, axis=0)

        # Reshape to (1, num_rois, pool_size, pool_size, nb_channels)
        # Might be (1, 4, 7, 7, 3)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        # permute_dimensions is similar to transpose
        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))
        # print(final_output.shape)
        return final_output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FixedBatchNormalization(tf.keras.layers.Layer):

    def __init__(self, epsilon=1e-3, axis=-1,
                 weights=None, beta_init='zero', gamma_init='one',
                 gamma_regularizer=None, beta_regularizer=None, **kwargs):

        self.supports_masking = True
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.epsilon = epsilon
        self.axis = axis
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.initial_weights = weights
        super(FixedBatchNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)

        self.gamma = self.add_weight(shape=shape,
                                     initializer=self.gamma_init,
                                     regularizer=self.gamma_regularizer,
                                     name='{}_gamma'.format(self.name),
                                     trainable=False)
        self.beta = self.add_weight(shape=shape,
                                    initializer=self.beta_init,
                                    regularizer=self.beta_regularizer,
                                    name='{}_beta'.format(self.name),
                                    trainable=False)
        self.running_mean = self.add_weight(shape=shape, initializer='zero',
                                            name='{}_running_mean'.format(self.name),
                                            trainable=False)
        self.running_std = self.add_weight(shape=shape, initializer='one',
                                           name='{}_running_std'.format(self.name),
                                           trainable=False)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True

    def call(self, x, mask=None):

        assert self.built, 'Layer must be built before being called'
        input_shape = K.int_shape(x)

        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        if sorted(reduction_axes) == range(K.ndim(x))[:-1]:
            x_normed = K.batch_normalization(
                x, self.running_mean, self.running_std,
                self.beta, self.gamma,
                epsilon=self.epsilon)
        else:
            # need broadcasting
            broadcast_running_mean = K.reshape(self.running_mean, broadcast_shape)
            broadcast_running_std = K.reshape(self.running_std, broadcast_shape)
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            x_normed = K.batch_normalization(
                x, broadcast_running_mean, broadcast_running_std,
                broadcast_beta, broadcast_gamma,
                epsilon=self.epsilon)

        return x_normed

    def get_config(self):
        config = {'epsilon': self.epsilon,
                  'axis': self.axis,
                  'gamma_regularizer': self.gamma_regularizer.get_config() if self.gamma_regularizer else None,
                  'beta_regularizer': self.beta_regularizer.get_config() if self.beta_regularizer else None}
        base_config = super(FixedBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


###############################################################################
# Definitions for losses
lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0
lambda_cls_regr = 1.0
lambda_cls_class = 1.0
epsilon = 1e-4


def rpn_loss_regr(num_anchors):
    """Loss function for rpn regression
    Args:
        num_anchors: number of anchors (9 in here)
    Returns:
        Smooth L1 loss function
                           0.5*x*x (if x_abs < 1)
                           x_abx - 0.5 (otherwise)
    """
    def rpn_loss_regr_fixed_num(y_true, y_pred):

        # x is the difference between true value and predicted vaue
        x = y_true[:, :, :, 4 * num_anchors:] - y_pred

        # absolute value of x
        x_abs = K.abs(x)

        # If x_abs <= 1.0, x_bool = 1
        x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

        return lambda_rpn_regr * K.sum(
            y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])

    return rpn_loss_regr_fixed_num


def rpn_loss_cls(num_anchors):
    """Loss function for rpn classification
    Args:
        num_anchors: number of anchors (9 in here)
        y_true[:, :, :, :9]: [0,1,0,0,0,0,0,1,0] means only the second and the eighth box is valid which contains pos or neg anchor => isValid
        y_true[:, :, :, 9:]: [0,1,0,0,0,0,0,0,0] means the second box is pos and eighth box is negative
    Returns:
        lambda * sum((binary_crossentropy(isValid*y_pred,y_true))) / N
    """
    def rpn_loss_cls_fixed_num(y_true, y_pred):
        return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])

    return rpn_loss_cls_fixed_num


def class_loss_regr(num_classes):
    """Loss function for rpn regression
    Args:
        num_anchors: number of anchors (9 in here)
    Returns:
        Smooth L1 loss function
                           0.5*x*x (if x_abs < 1)
                           x_abx - 0.5 (otherwise)
    """
    def class_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, 4*num_classes:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        return lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])
    return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
    return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))


###############################################################################
# Definitions for roi related helpers
@jit(nopython=True)
def calc_iou(R, img_data, classifier_overlap, im_size, rpn_stride, class_mapping, classifier_regr_std):
    """Converts from (x1,y1,x2,y2) to (x,y,w,h) format
    """
    bboxes = img_data['bboxes']
    (width, height) = (img_data['width'], img_data['height'])
    # get image dimensions for resizing
    (resized_width, resized_height) = get_new_img_size(width, height, im_size)

    gta = np.zeros((len(bboxes), 4))

    for bbox_num, bbox in enumerate(bboxes):
        # get the GT box coordinates, and resize to account for image resizing
        gta[bbox_num, 0] = int(round(bbox['x1'] * (resized_width / float(width))/rpn_stride))
        gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_width / float(width))/rpn_stride))
        gta[bbox_num, 2] = int(round(bbox['y1'] * (resized_height / float(height))/rpn_stride))
        gta[bbox_num, 3] = int(round(bbox['y2'] * (resized_height / float(height))/rpn_stride))

    x_roi = []
    y_class_num = []
    y_class_regr_coords = []
    y_class_regr_label = []
    IoUs = []  # for debugging only

    # R.shape[0]: number of bboxes (=300 from non_max_suppression)
    for ix in range(R.shape[0]):
        (x1, y1, x2, y2) = R[ix, :]
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        best_iou = 0.0
        best_bbox = -1
        # Iterate through all the ground-truth bboxes to calculate the iou
        for bbox_num in range(len(bboxes)):
            curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1, y1, x2, y2])

            # Find out the corresponding ground-truth bbox_num with larget iou
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = bbox_num

        if best_iou < classifier_overlap[0]:
            continue
        else:
            w = x2 - x1
            h = y2 - y1
            x_roi.append([x1, y1, w, h])
            IoUs.append(best_iou)

            if classifier_overlap[0] <= best_iou < classifier_overlap[1]:
                # hard negative example
                cls_name = 'bg'
            elif classifier_overlap[1] <= best_iou:
                cls_name = bboxes[best_bbox]['class']
                cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0
                cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0

                cx = x1 + w / 2.0
                cy = y1 + h / 2.0

                tx = (cxg - cx) / float(w)
                ty = (cyg - cy) / float(h)
                tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / float(w))
                th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))
            else:
                print('roi = {}'.format(best_iou))
                raise RuntimeError

        class_num = class_mapping[cls_name]
        class_label = len(class_mapping) * [0]
        class_label[class_num] = 1
        y_class_num.append(copy.deepcopy(class_label))
        coords = [0] * 4 * (len(class_mapping) - 1)
        labels = [0] * 4 * (len(class_mapping) - 1)
        if cls_name != 'bg':
            label_pos = 4 * class_num
            sx, sy, sw, sh = classifier_regr_std
            coords[label_pos:4+label_pos] = [sx*tx, sy*ty, sw*tw, sh*th]
            labels[label_pos:4+label_pos] = [1, 1, 1, 1]
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))
        else:
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))

    if len(x_roi) == 0:
        return None, None, None, None

    # bboxes that iou > C.classifier_min_overlap for all gt bboxes in 300 non_max_suppression bboxes
    X = np.array(x_roi)
    # one hot code for bboxes from above => x_roi (X)
    Y1 = np.array(y_class_num)
    Y2 = np.concatenate([np.array(y_class_regr_label), np.array(y_class_regr_coords)], axis=1)

    return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs


def apply_regr(x, y, w, h, tx, ty, tw, th):
    try:
        cx = x + w/2.
        cy = y + h/2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy
        w1 = math.exp(tw) * w
        h1 = math.exp(th) * h
        x1 = cx1 - w1/2.
        y1 = cy1 - h1/2.
        x1 = int(round(x1))
        y1 = int(round(y1))
        w1 = int(round(w1))
        h1 = int(round(h1))

        return x1, y1, w1, h1

    except ValueError:
        return x, y, w, h
    except OverflowError:
        return x, y, w, h
    except Exception as e:
        print(e)
        return x, y, w, h


def apply_regr_np(X, T):
    try:
        x = X[0, :, :]
        y = X[1, :, :]
        w = X[2, :, :]
        h = X[3, :, :]

        tx = T[0, :, :]
        ty = T[1, :, :]
        tw = T[2, :, :]
        th = T[3, :, :]

        cx = x + w/2.
        cy = y + h/2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy

        w1 = np.exp(tw.astype(np.float64)) * w
        h1 = np.exp(th.astype(np.float64)) * h
        x1 = cx1 - w1/2.
        y1 = cy1 - h1/2.

        x1 = np.round(x1)
        y1 = np.round(y1)
        w1 = np.round(w1)
        h1 = np.round(h1)
        return np.stack([x1, y1, w1, h1])
    except Exception as e:
        print(e)
        return X


def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=300):
    # code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # calculate the areas
    area = (x2 - x1) * (y2 - y1)

    # sort the bounding boxes
    idxs = np.argsort(probs)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the intersection

        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int

        # find the union
        area_union = area[i] + area[idxs[:last]] - area_int

        # compute the ratio of overlap
        overlap = area_int/(area_union + 1e-6)

        # delete all indexes from the index list that are over the threshold
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break

    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick].astype("int")
    probs = probs[pick]
    return boxes, probs

# @jit(nopython=True)
def rpn_to_roi(rpn_layer, regr_layer, std_scaling, anchor_box_ratios, anchor_box_scales, rpn_stride, use_regr=True, max_boxes=300, overlap_thresh=0.9):

    regr_layer = regr_layer / std_scaling
    anchor_sizes = anchor_box_scales
    anchor_ratios = anchor_box_ratios
    assert rpn_layer.shape[0] == 1

    (rows, cols) = rpn_layer.shape[1:3]
    curr_layer = 0
    A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))

    for anchor_size in anchor_sizes:
        for anchor_ratio in anchor_ratios:

            anchor_x = (anchor_size * anchor_ratio[0])/rpn_stride
            anchor_y = (anchor_size * anchor_ratio[1])/rpn_stride

            regr = regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4]
            regr = np.transpose(regr, (2, 0, 1))

            X, Y = np.meshgrid(np.arange(cols), np. arange(rows))

            A[0, :, :, curr_layer] = X - anchor_x/2
            A[1, :, :, curr_layer] = Y - anchor_y/2
            A[2, :, :, curr_layer] = anchor_x
            A[3, :, :, curr_layer] = anchor_y

            if use_regr:
                A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], regr)

            # Avoid width and height exceeding 1
            A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer])

            # Convert (x, y , w, h) to (x1, y1, x2, y2)
            A[2, :, :, curr_layer] += A[0, :, :, curr_layer]
            A[3, :, :, curr_layer] += A[1, :, :, curr_layer]

            # Avoid bboxes drawn outside the feature map
            A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer])
            A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])
            A[2, :, :, curr_layer] = np.minimum(cols-1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.minimum(rows-1, A[3, :, :, curr_layer])

            curr_layer += 1

    all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))
    all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))

    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]

    # Find out the bboxes which is illegal and delete them from bboxes list
    idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))
    all_boxes = np.delete(all_boxes, idxs, 0)
    all_probs = np.delete(all_probs, idxs, 0)

    # Apply non_max_suppression
    # Only extract the bboxes. Don't need rpn probs in the later process
    result = non_max_suppression_fast(all_boxes, all_probs, overlap_thresh=overlap_thresh, max_boxes=max_boxes)[0]

    return result


###############################################################################
# Data generator and data augmentation
def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w*h


def iou(a, b):
    # a and b should be (x1,y1,x2,y2)

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)


def get_new_img_size(width, height, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = img_min_side
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = img_min_side

    return resized_width, resized_height


def augment(img_data, config, augment=True, visualise=False, verbose=0):
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    assert 'width' in img_data
    assert 'height' in img_data

    config = dotdict(config)

    img_data_aug = copy.deepcopy(img_data)
    img = cv2.imread(img_data_aug['filepath'])
    rows, cols, channels = img.shape
    images = np.zeros((1, rows, cols, channels), dtype=np.uint8)

    images[0] = img
    if augment:

        np.random.seed(config.seed)
        ia.seed(config.seed)

        if (config.translate_x):
            try:  # 1-D array-like or int
                tx = np.random.choice(config.translate_x)
                tx *= np.random.choice([-1, 1])
            except ValueError:  # floating point
                tx = np.random.uniform(-config.translate_x, config.translate_x)
        else:
            tx = 0

        if (config.translate_y):
            try:  # 1-D array-like or int
                ty = np.random.choice(config.translate_y)
                ty *= np.random.choice([-1, 1])
            except ValueError:  # floating point
                ty = np.random.uniform(-config.translate_y, config.translate_y)
        else:
            ty = 0

        bbox_list = [BoundingBox(x1=bbox['x1'], y1=bbox['y1'], x2=bbox['x2'], y2=bbox['y2']) for bbox in img_data_aug['bboxes']]
        bbs = BoundingBoxesOnImage(bbox_list, images[0].shape)

        seq = iaa.Sequential(
            [
                iaa.Fliplr(config.horizontal_flip),  # horizontal flips
                iaa.Flipud(config.vertical_flip),  # vertical flips
                iaa.Crop(percent=config.crop),  # random crops
                # Small gaussian blur with random sigma between 0 and 0.5.
                # But we only blur about 50% of all images.
                iaa.Sometimes(config.blur,
                              iaa.GaussianBlur(sigma=config.sigma)
                              ),
                # Strengthen or weaken the contrast in each image.
                # iaa.ContrastNormalization(config.contrast),
                # Add gaussian noise.
                # For 50% of all images, we sample the noise once per pixel.
                # For the other 50% of all images, we sample the noise per pixel AND
                # channel. This can change the color (not only brightness) of the
                # pixels.
                # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                # Make some images brighter and some darker.
                # In 20% of all cases, we sample the multiplier once per channel,
                # which can end up changing the color of the images.
                # iaa.Multiply((0.8, 1.2), per_channel=0.2),
                # Apply affine transformations to each image.
                # Scale/zoom them, translate/move them, rotate them and shear them.
                iaa.Sometimes(
                    0.5, iaa.Affine(
                        scale={"x": config.scale_x, "y": config.scale_y},
                        translate_percent={"x": tx, "y": ty},
                        rotate=config.rotate,
                        # shear=(-8, 8)
                    )
                )
            ], random_order=True)  # apply augmenters in random order

        img_aug, bbs_aug = seq(images=images, bounding_boxes=bbs)

        for i in range(len(bbs.bounding_boxes)):
            before = bbs.bounding_boxes[i]
            after = bbs_aug.bounding_boxes[i]
            if verbose:
                print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)"
                      % (i, before.x1, before.y1, before.x2, before.y2, after.x1, after.y1, after.x2, after.y2))
            img_data_aug['bboxes'][i]['x1'] = int(after.x1)
            img_data_aug['bboxes'][i]['y1'] = int(after.y1)
            img_data_aug['bboxes'][i]['x2'] = int(after.x2)
            img_data_aug['bboxes'][i]['y2'] = int(after.y2)

        # bbs.draw_on_image(img, size=100)
        # bbs_aug.draw_on_image(img_aug, size=100, color=[0, 0, 255])

    img_data_aug['width'] = images[0].shape[1]
    img_data_aug['height'] = images[0].shape[0]
    return img_data_aug, img_aug[0]


def calc_rpn(
    img_data, width, height, resized_width, resized_height, output_width, output_height,
    rpn_stride, anchor_sizes, anchor_ratios, rpn_min_overlap, rpn_max_overlap
):
    downscale = float(rpn_stride)
    num_anchors = len(anchor_sizes) * len(anchor_ratios)
    n_anchratios = len(anchor_ratios)

    # initialise empty output objectives
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
    y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

    num_bboxes = len(img_data['bboxes'])

    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
    best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int)
    best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
    best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
    best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

    # get the GT box coordinates, and resize to account for image resizing
    gta = np.zeros((num_bboxes, 4))
    for bbox_num, bbox in enumerate(img_data['bboxes']):
        # get the GT box coordinates, and resize to account for image resizing
        gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
        gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
        gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
        gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))

    # rpn ground truth

    for anchor_size_idx in range(len(anchor_sizes)):
        for anchor_ratio_idx in range(n_anchratios):
            anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
            anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]

            for ix in range(output_width):
                # x-coordinates of the current anchor box
                x1_anc = downscale * (ix + 0.5) - anchor_x / 2
                x2_anc = downscale * (ix + 0.5) + anchor_x / 2

                # ignore boxes that go across image boundaries
                if x1_anc < 0 or x2_anc > resized_width:
                    continue

                for jy in range(output_height):

                    # y-coordinates of the current anchor box
                    y1_anc = downscale * (jy + 0.5) - anchor_y / 2
                    y2_anc = downscale * (jy + 0.5) + anchor_y / 2

                    # ignore boxes that go across image boundaries
                    if y1_anc < 0 or y2_anc > resized_height:
                        continue

                    # bbox_type indicates whether an anchor should be a target
                    bbox_type = 'neg'

                    # this is the best IOU for the (x,y) coord and the current anchor
                    # note that this is different from the best IOU for a GT bbox
                    best_iou_for_loc = 0.0

                    for bbox_num in range(num_bboxes):

                        # get IOU of the current GT box and the current anchor box
                        curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1_anc, y1_anc, x2_anc, y2_anc])
                        # calculate the regression targets if they will be needed
                        if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > rpn_max_overlap:
                            cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
                            cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
                            cxa = (x1_anc + x2_anc)/2.0
                            cya = (y1_anc + y2_anc)/2.0

                            tx = (cx - cxa) / (x2_anc - x1_anc)
                            ty = (cy - cya) / (y2_anc - y1_anc)
                            tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
                            th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))

                        if img_data['bboxes'][bbox_num]['class'] != 'bg':

                            # all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
                            if curr_iou > best_iou_for_bbox[bbox_num]:
                                best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                                best_iou_for_bbox[bbox_num] = curr_iou
                                best_x_for_bbox[bbox_num, :] = [x1_anc, x2_anc, y1_anc, y2_anc]
                                best_dx_for_bbox[bbox_num, :] = [tx, ty, tw, th]

                            # we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
                            if curr_iou > rpn_max_overlap:
                                bbox_type = 'pos'
                                num_anchors_for_bbox[bbox_num] += 1
                                # we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
                                if curr_iou > best_iou_for_loc:
                                    best_iou_for_loc = curr_iou
                                    best_regr = (tx, ty, tw, th)

                            # if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
                            if rpn_min_overlap < curr_iou < rpn_max_overlap:
                                # gray zone between neg and pos
                                if bbox_type != 'pos':
                                    bbox_type = 'neutral'

                    # turn on or off outputs depending on IOUs
                    if bbox_type == 'neg':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'neutral':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'pos':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
                        y_rpn_regr[jy, ix, start:start+4] = best_regr

    # we ensure that every bbox has at least one positive RPN region

    for idx in range(num_anchors_for_bbox.shape[0]):
        if num_anchors_for_bbox[idx] == 0:
            # no box with an IOU greater than zero ...
            if best_anchor_for_bbox[idx, 0] == -1:
                continue
            y_is_box_valid[
                best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], best_anchor_for_bbox[idx, 2] + n_anchratios * best_anchor_for_bbox[idx, 3]] = 1
            y_rpn_overlap[
                best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], best_anchor_for_bbox[idx, 2] + n_anchratios * best_anchor_for_bbox[idx, 3]] = 1
            start = 4 * (best_anchor_for_bbox[idx, 2] + n_anchratios * best_anchor_for_bbox[idx, 3])
            y_rpn_regr[
                best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], start:start+4] = best_dx_for_bbox[idx, :]

    y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

    y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

    y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

    pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

    num_pos = len(pos_locs[0])

    # one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
    # regions. We also limit it to 256 regions.
    num_regions = 256

    if len(pos_locs[0]) > num_regions/2:
        val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2)
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
        num_pos = num_regions/2

    if len(neg_locs[0]) + num_pos > num_regions:
        val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
        y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

    return np.copy(y_rpn_cls), np.copy(y_rpn_regr), num_pos


###############################################################################
# Utilities
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


###############################################################################
# Public functions that will be revealed
###############################################################################
def preprocess_input(x_img):
    # Zero-center by mean pixel, and preprocess image
    img_channel_mean = [103.939, 116.779, 123.68]
    img_scaling_factor = 1

    x_img = x_img[:, :, (2, 1, 0)]  # BGR -> RGB
    x_img = x_img.astype(np.float32)
    x_img[:, :, 0] -= img_channel_mean[0]
    x_img[:, :, 1] -= img_channel_mean[1]
    x_img[:, :, 2] -= img_channel_mean[2]
    x_img /= img_scaling_factor

    x_img = np.transpose(x_img, (2, 0, 1))
    x_img = np.expand_dims(x_img, axis=0)
    x_img = np.transpose(x_img, (0, 2, 3, 1))
    return x_img


def FRCNNGenerator(
    all_img_data,
    mode='train',
    shuffle=True,

    horizontal_flip=False,       #
    vertical_flip=False,        #
    rotation_range=0,   #
    width_shift_range=0,
    height_shift_range=0,
    crop=0.0,
    blur=0.0,                # gaussian blur probability
    sigma=0.0,               # gaussian blur sigma
    scale_x=1.0,
    scale_y=1.0,
    seed=1,
    # img_channel_mean=[103.939, 116.779, 123.68],
    # img_scaling_factor=1,
    std_scaling=4,
    target_size=600,

    rpn_stride=16,
    anchor_box_scales=[128, 256, 512],
    anchor_box_ratios=[[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]],
    rpn_min_overlap=0.3,
    rpn_max_overlap=0.7,

    base_net_type='vgg',
    preprocessing_function=preprocess_input
):
    """ Generates batch of image data with real-time data augmentation for FRCNN model
    Yield the ground-truth anchors as Y (labels)

    Args:
        all_img_data: list(filepath, width, height, list(bboxes))
        mode: 'train' or 'test'; 'train' mode need augmentation.
        shuffle: Boolean. Whether to shuffle the data.

        horizontal_flip: Boolean. Randomly flip inputs horizontally.
        vertical_flip: Boolean. Randomly flip inputs vertically.
        rotation_range: Int or Tuple (min, max). Degree range for random rotations
        width_shift_range: Float, 1-D array-like or int
            - float: fraction of total width, if < 1, or pixels if >= 1.
            - 1-D array-like: random elements from the array.
            - int: integer number of pixels from interval
                `(-width_shift_range, +width_shift_range)`
            - With `width_shift_range=2` possible values
                are integers `[-1, 0, +1]`,
                same as with `width_shift_range=[-1, 0, +1]`,
                while with `width_shift_range=1.0` possible values are floats
                in the interval [-1.0, +1.0).
        height_shift_range: Float, 1-D array-like or int
            - float: fraction of total height, if < 1, or pixels if >= 1.
            - 1-D array-like: random elements from the array.
            - int: integer number of pixels from interval
                `(-height_shift_range, +height_shift_range)`
            - With `height_shift_range=2` possible values
                are integers `[-1, 0, +1]`,
                same as with `height_shift_range=[-1, 0, +1]`,
                while with `height_shift_range=1.0` possible values are floats
                in the interval [-1.0, +1.0).
        crop: The number of pixels to crop away (cut off) on each side of the image given *in percent* of the image height/width.  E.g. if this is set to 0.1, the augmenter will always crop away 10 percent of the image's height at the top, 10 percent of the width on the right, 10 percent of the height at the bottom and 10 percent of the width on the left.
        blur: Probability of augmentation with gaussian blur
        sigma: Sigma parameter for gaussian blur
        scale_x: Rescaling factor along x-axis
        scale_y: Rescaling factor along y-axis
        seed: Int (default: 1). Random seed.

        target_size: shorter-side length. Used for image resizing based on the shorter length
        std_scaling: For scaling of standard deviation
        target_size: Integer. Shorter-side length. Used for image resizing based on the shorter length.
        rpn_stride: Stride at the RPN. This depends on the network configuration. For VGG16, rpn_stride = 16.
        anchor_box_scales: Anchor box scales array.
        anchor_box_ratios: Anchor box ratios array.
        rpn_min_overlap: RPN minimum overlap. Anchor is labelled as “negative”
            if anchor has IoU with all ground-truth boxes < rpn_min_overlap.
        rpn_max_overlap: RPN maximum overlap. Anchor is labelled as “positive”
            if anchor has an IoU > rpn_max_overlap with any ground-truth box.
        base_net_type:

        preprocessing_function: If None, will do zero-center by mean pixel, else will execute function.

    Returns:
        x_img: image data after resized and scaling (smallest size = 300px)
        Y: [y_rpn_cls, y_rpn_regr]
        img_data_aug: augmented image data (original image with augmentation)
        debug_img: show image for debug
        num_pos: show number of positive anchors for debug
    """
    config = {
        'horizontal_flip': 0.5 if horizontal_flip else 0.0,
        'vertical_flip': 0.5 if vertical_flip else 0.0,
        'rotate': rotation_range if (type(rotation_range) == tuple and len(rotation_range) == 2) else (-abs(rotation_range), abs(rotation_range)),
        'translate_x': width_shift_range,
        'translate_y': height_shift_range,
        'crop': crop,
        'blur': blur,
        'sigma': sigma,

        # 'contrast': contrast,
        'scale_x': scale_x,
        'scale_y': scale_y,

        'seed': seed
    }

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(all_img_data)

    while True:
        for img_data in all_img_data:
            try:
                # read in image, and optionally add augmentation
                if mode == 'train':
                    img_data_aug, x_img = augment(img_data, config, augment=True)
                else:
                    img_data_aug, x_img = augment(img_data, config, augment=False)

                (width, height) = (img_data_aug['width'], img_data_aug['height'])
                (rows, cols, _) = x_img.shape

                assert cols == width
                assert rows == height

                # get image dimensions for resizing
                (resized_width, resized_height) = get_new_img_size(width, height, target_size)

                # resize the image so that smaller side has length = target_size
                x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
                debug_img = x_img.copy()
                try:
                    # calculate the output map size based on the network architecture
                    (output_width, output_height) = _get_img_output_length(resized_width, resized_height, base_net_type=base_net_type)

                    # calculate RPN
                    y_rpn_cls, y_rpn_regr, num_pos = calc_rpn(
                        img_data_aug, width, height, resized_width, resized_height, output_width, output_height,
                        rpn_stride, anchor_box_scales,
                        anchor_box_ratios, rpn_min_overlap, rpn_max_overlap)
                except Exception as e:
                    print(e)
                    continue

                # Preprocessing function
                if (preprocessing_function is not None):
                    x_img = preprocessing_function(x_img)

                y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= std_scaling
                y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
                y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

                yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug, debug_img, num_pos

            except Exception as e:
                print(e)
                continue


# Parser for annotations
def parseAnnotationFile(input_path, verbose=1, visualise=True, mode='simple', filteredList=None):
    """Parse the data from an annotation file in 'simple' format, or from a directory in ‘Pascal VOC’ format

    Args:
        input_path: annotation file path or directory
        verbose: Verbosity mode. 0 = silent. 1 = print out details of annotation file
        visualise: Boolean. If True, show distribution of classes in annotation file
        mode: 'simple' or 'voc'. Default mode is 'simple' where each line
            should contain filepath,x1,y1,x2,y2,class_name.
            Can also accept 'voc' format which is the Pascal VOC data set format
        filteredList: If specified, only extract classes in this list

    Returns:
        all_data: list(filepath, width, height, list(bboxes))
        classes_count: dict{key:class_name, value:count_num}
            e.g. {'Car': 2383, 'Mobile phone': 1108, 'Person': 3745}
        class_mapping: dict{key:class_name, value: idx}
            e.g. {'Car': 0, 'Mobile phone': 1, 'Person': 2}
    """

    st = time.time()
    if (mode == 'simple'):
        all_data, classes_count, class_mapping = parseAnnotationFileSimple(input_path, verbose, filteredList)
    else:
        all_data, classes_count, class_mapping = parseAnnotationFileVOC(input_path, verbose, filteredList)

    sorted_classes = sorted(classes_count.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    sorted_classes = list(filter(lambda c: c[1] > 0, sorted_classes))
    if (verbose):
        print()
        print()
        print('Classes in annotation file %s:' % (input_path))
        # print({v: k for k, v in class_mapping.items()})
        # print(list(class_mapping.keys()))
        print(sorted_classes)
        print('Total classes: ' + str(len(sorted_classes)))
        print('Spend %0.2f mins to load the data' % ((time.time()-st)/60))

    if (visualise):
        plt.figure(figsize=(8, 8))
        plt.title('Distribution of classes for %s' % (input_path))
        plt.bar([i[0] for i in sorted_classes], [i[1] for i in sorted_classes])
        plt.show()
        # plt.bar(list(classes_count.keys()), list(classes_count.values()) )

    return all_data, classes_count, class_mapping


def parseAnnotationFileSimple(input_path, verbose=1, filteredList=None):
    found_bg = False
    all_imgs = {}
    classes_count = {}
    class_mapping = {}
    i = 1

    with open(input_path, 'r') as f:

        if verbose:
            print('Parsing annotation files')
        for line in f:
            sys.stdout.write('\r'+'idx=' + str(i))
            i += 1

            line_split = line.strip().split(',')
            (filename, x1, y1, x2, y2, class_name) = line_split

            if (filteredList is not None and (class_name not in filteredList)):
                continue    # If not one of our classes of interest, we ignore

            filename = filename.replace('\\', '/')  # in case backslash is used, we will replace with forward slash instead
            if (not os.path.isfile(filename)):
                print("\n" + filename + " could not be read")
            else:

                if class_name not in classes_count:
                    classes_count[class_name] = 1
                else:
                    classes_count[class_name] += 1

                if class_name not in class_mapping:
                    if class_name == 'bg' and found_bg is False:
                        if verbose:
                            print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                        found_bg = True
                    class_mapping[class_name] = len(class_mapping)

                if filename not in all_imgs:
                    img = cv2.imread(filename)
                    (rows, cols) = img.shape[:2]

                    all_imgs[filename] = {
                        'filepath': filename,
                        'width': cols,
                        'height': rows,
                        'bboxes': []
                    }
                    # if np.random.randint(0,6) > 0:
                    #     all_imgs[filename]['imageset'] = 'trainval'
                    # else:
                    #     all_imgs[filename]['imageset'] = 'test'

                all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})

        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        # always include special class 'bg'
        else:
            classes_count['bg'] = 0
            class_mapping['bg'] = len(class_mapping)

        return all_data, classes_count, class_mapping


def parseAnnotationFileVOC(input_path, verbose=1, filteredList=None):
    import xml.etree.ElementTree as ET

    all_imgs = []
    classes_count = {}
    class_mapping = {}
    data_paths = [os.path.join(input_path, s) for s in ['VOC2012']]
    # data_paths = [os.path.join(input_path,s) for s in ['VOC2007', 'VOC2012']]

    if verbose:
        print('Parsing annotation files')

    for data_path in data_paths:
        annot_path = os.path.join(data_path, 'Annotations')
        imgs_path = os.path.join(data_path, 'JPEGImages')
        imgsets_path_trainval = os.path.join(data_path, 'ImageSets', 'Main', 'trainval.txt')
        imgsets_path_test = os.path.join(data_path, 'ImageSets', 'Main', 'test.txt')

        trainval_files = []
        test_files = []
        try:
            with open(imgsets_path_trainval) as f:
                for line in f:
                    trainval_files.append(line.strip() + '.jpg')
        except Exception as e:
            print(e)

        try:
            with open(imgsets_path_test) as f:
                for line in f:
                    test_files.append(line.strip() + '.jpg')
        except Exception as e:
            if data_path[-7:] == 'VOC2012':
                # this is expected, most pascal voc distibutions dont have the test.txt file
                pass
            else:
                print(e)

        annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]
        idx = 0
        for annot in annots:
            try:
                idx += 1
                sys.stdout.write('\r'+'idx=' + str(idx))
                et = ET.parse(annot)
                element = et.getroot()

                element_objs = element.findall('object')
                element_filename = element.find('filename').text
                element_width = int(element.find('size').find('width').text)
                element_height = int(element.find('size').find('height').text)

                if len(element_objs) > 0:
                    annotation_data = {'filepath': os.path.join(imgs_path, element_filename), 'width': element_width,
                                       'height': element_height, 'bboxes': []}

                    if element_filename in trainval_files:
                        annotation_data['imageset'] = 'trainval'
                    elif element_filename in test_files:
                        annotation_data['imageset'] = 'test'
                    else:
                        annotation_data['imageset'] = 'trainval'

                for element_obj in element_objs:
                    class_name = element_obj.find('name').text

                    if (filteredList is not None and (class_name not in filteredList)):
                        continue    # If not one of our classes of interest, we ignore

                    if class_name not in classes_count:
                        classes_count[class_name] = 1
                    else:
                        classes_count[class_name] += 1

                    if class_name not in class_mapping:
                        class_mapping[class_name] = len(class_mapping)

                    obj_bbox = element_obj.find('bndbox')
                    x1 = int(round(float(obj_bbox.find('xmin').text)))
                    y1 = int(round(float(obj_bbox.find('ymin').text)))
                    x2 = int(round(float(obj_bbox.find('xmax').text)))
                    y2 = int(round(float(obj_bbox.find('ymax').text)))
                    difficulty = int(element_obj.find('difficult').text) == 1
                    annotation_data['bboxes'].append(
                        {'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty})
                all_imgs.append(annotation_data)

                # if visualise:
                #     img = cv2.imread(annotation_data['filepath'])
                #     for bbox in annotation_data['bboxes']:
                #         cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox[
                #                       'x2'], bbox['y2']), (0, 0, 255))
                #     cv2.imshow('img', img)
                #     cv2.waitKey(0)
            except Exception as e:
                print(e)
                continue

        # always include special class 'bg'
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)

    return all_imgs, classes_count, class_mapping


# Inspect generator
def inspect(
    generator, target_size, rpn_stride=16, anchor_box_scales=[128, 256, 512],
    anchor_box_ratios=[[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]]
):
    """ Based on generator, prints details of image, ground-truth annotations, as well as positive anchors
    Args:
        generator: Generator that was created via FRCNNGenerator
        target_size: Target size of shorter side. This should be the same as what was passed into the generator
        rpn_stride: RPN stride. This should be the same as what was passed into the generator
        anchor_box_scales: Anchor box scales array. This should be the same as what was passed into the generator
        anchor_box_ratios: Anchor box ratios array. This should be the same as what was passed into the generator

    Returns:
        None
    """
    from matplotlib import pyplot as plt

    X, Y, image_data, debug_img, debug_num_pos = next(generator)
    print('Original image: height=%d width=%d' % (image_data['height'], image_data['width']))
    print('Resized image:  height=%d width=%d im_size=%d' % (X.shape[1], X.shape[2], target_size))
    print('Feature map size: height=%d width=%d rpn_stride=%d' % (Y[0].shape[1], Y[0].shape[2], rpn_stride))
    print(X.shape)
    print(str(len(Y))+" includes 'y_rpn_cls' and 'y_rpn_regr'")
    print('Shape of y_rpn_cls {}'.format(Y[0].shape))
    print('Shape of y_rpn_regr {}'.format(Y[1].shape))
    print(image_data)

    print('Number of positive anchors for this image: %d' % (debug_num_pos))
    if debug_num_pos == 0:
        gt_x1, gt_x2 = image_data['bboxes'][0]['x1']*(X.shape[2]/image_data['height']), image_data['bboxes'][0]['x2']*(X.shape[2]/image_data['height'])
        gt_y1, gt_y2 = image_data['bboxes'][0]['y1']*(X.shape[1]/image_data['width']), image_data['bboxes'][0]['y2']*(X.shape[1]/image_data['width'])
        gt_x1, gt_y1, gt_x2, gt_y2 = int(gt_x1), int(gt_y1), int(gt_x2), int(gt_y2)

        img = debug_img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        color = (0, 255, 0)
        cv2.putText(img, 'gt bbox', (gt_x1, gt_y1-5), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
        cv2.rectangle(img, (gt_x1, gt_y1), (gt_x2, gt_y2), color, 2)
        cv2.circle(img, (int((gt_x1+gt_x2)/2), int((gt_y1+gt_y2)/2)), 3, color, -1)

        plt.grid()
        plt.imshow(img)
        plt.show()
    else:
        cls = Y[0][0]
        pos_cls = np.where(cls == 1)
        print(pos_cls)
        regr = Y[1][0]
        pos_regr = np.where(regr == 1)
        print(pos_regr)
        print('y_rpn_cls for possible pos anchor: {}'.format(cls[pos_cls[0][0], pos_cls[1][0], :]))
        print('y_rpn_regr for positive anchor: {}'.format(regr[pos_regr[0][0], pos_regr[1][0], :]))

        gt_x1, gt_x2 = image_data['bboxes'][0]['x1']*(X.shape[2]/image_data['width']), image_data['bboxes'][0]['x2']*(X.shape[2]/image_data['width'])
        gt_y1, gt_y2 = image_data['bboxes'][0]['y1']*(X.shape[1]/image_data['height']), image_data['bboxes'][0]['y2']*(X.shape[1]/image_data['height'])
        gt_x1, gt_y1, gt_x2, gt_y2 = int(gt_x1), int(gt_y1), int(gt_x2), int(gt_y2)

        img = debug_img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        color = (0, 255, 0)
        #   cv2.putText(img, 'gt bbox', (gt_x1, gt_y1-5), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
        cv2.rectangle(img, (gt_x1, gt_y1), (gt_x2, gt_y2), color, 2)
        cv2.circle(img, (int((gt_x1+gt_x2)/2), int((gt_y1+gt_y2)/2)), 3, color, -1)

        # Add text
        textLabel = 'gt bbox'
        (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
        textOrg = (gt_x1, gt_y1+5)
        cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
        cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
        cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

        # Draw positive anchors according to the y_rpn_regr
        for i in range(debug_num_pos):

            color = (100+i*(155/4), 0, 100+i*(155/4))

            idx = pos_regr[2][i*4]/4
            anchor_size = anchor_box_scales[int(idx/3)]
            anchor_ratio = anchor_box_ratios[2-int((idx+1) % 3)]

            center = (pos_regr[1][i*4]*rpn_stride, pos_regr[0][i*4]*rpn_stride)
            print('Center position of positive anchor: ', center)
            cv2.circle(img, center, 3, color, -1)
            anc_w, anc_h = anchor_size*anchor_ratio[0], anchor_size*anchor_ratio[1]
            cv2.rectangle(img, (center[0]-int(anc_w/2), center[1]-int(anc_h/2)), (center[0]+int(anc_w/2), center[1]+int(anc_h/2)), color, 2)
    #         cv2.putText(img, 'pos anchor bbox '+str(i+1), (center[0]-int(anc_w/2), center[1]-int(anc_h/2)-5), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

    print('Green bboxes is ground-truth bbox. Others are positive anchors')
    plt.figure(figsize=(8, 8))
    plt.grid()
    plt.imshow(img)
    plt.show()


# Viewer for annotated image
def viewAnnotatedImage(annotation_file, query_image_path):
    """Views the annotated image based on an annotation file (in ‘simple’ format) and a query image path

    Args:
        annotation_file: annotation file path
        query_image_path: path of the image to check. eg 'data/train/image100.jpg'.
            This should correspond exactly with the annotation file's first column
    Returns:
        None
    """
    from matplotlib import pyplot as plt
    # import matplotlib.colors as mcolors

    annotations = pd.read_csv(annotation_file, sep=',', names=['image_name', 'x1', 'y1', 'x2', 'y2', 'Object_type'])
    class_mapping = annotations['Object_type'].unique()
    class_mapping = {class_mapping[i]: i for i in range(0, len(class_mapping))}
    num_classes = len(class_mapping)  # annotations['Object_type'].nunique()

    colorset = np.random.uniform(0, 255, size=(num_classes, 3))
    img = plt.imread(query_image_path)

    # maxVal = 255
    if (img.max() <= 1):
        colorset /= 255
        # maxVal = 1

    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    plt.imshow(img)

    windows_resize_image_path_file = query_image_path.replace('/', '\\')  # just in case annotation file is in windows directory format

    textArr = []

    # iterate over the image for different objects
    for _, r in annotations[(annotations.image_name == query_image_path) | (annotations.image_name == windows_resize_image_path_file)].iterrows():

        edgeColor = colorset[class_mapping[r.Object_type]]

        # ensure that our text is not out of image. Add to textArr, but don't draw text first
        y = r.y1-5
        if y < 10: y = r.y1+10
        textArr.append((r.Object_type, (r.x1, y), edgeColor))

        # draw bounding box
        cv2.rectangle(img, (r.x1, r.y1), (r.x2, r.y2), edgeColor, 2)

    # draw text last, so that they will not be obscured by the rectangles
    for t in textArr:
        # draw text twice, once in outline color with double thickness, and once in the text color. This enables text to always be seen
        # cv2.putText(img, t[0], t[1], cv2.FONT_HERSHEY_DUPLEX, 0.5, maxVal - t[2], 2, cv2.LINE_AA)

        # Calculate perceived luminance: https://www.w3.org/TR/AERT/#color-contrast
        # so that we can use a contrasting outline color.
        r, g, b = t[2]
        perceivedLum = (0.299*r + 0.587*g + 0.114*b)/255
        outlineColor = (0, 0, 0) if perceivedLum > 0.5 else (255, 255, 255)

        cv2.putText(img, t[0], t[1], cv2.FONT_HERSHEY_DUPLEX, 0.5, outlineColor, 2, cv2.LINE_AA)
        cv2.putText(img, t[0], t[1], cv2.FONT_HERSHEY_DUPLEX, 0.5, t[2], 1, cv2.LINE_AA)

    plt.grid()
    plt.imshow(img)
    plt.show()

    return None


def plotAccAndLoss(csv_path):
    from matplotlib import pyplot as plt

    record_df = pd.read_csv(csv_path)
    r_epochs = len(record_df)

    plt.figure(figsize=(15, 5))
    plt.subplot(4, 2, 1)
    plt.plot(np.arange(0, r_epochs), record_df['mean_overlapping_bboxes'], 'r')
    plt.title('mean_overlapping_bboxes')

    plt.subplot(4, 2, 2)
    plt.plot(np.arange(0, r_epochs), record_df['class_acc'], 'r')
    plt.title('class_acc')

    # plt.show()

    # plt.figure(figsize=(15,5))

    plt.subplot(4, 2, 3)
    plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_cls'], 'r')
    plt.title('loss_rpn_cls')

    plt.subplot(4, 2, 4)
    plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_regr'], 'r')
    plt.title('loss_rpn_regr')
    # plt.show()
    # plt.figure(figsize=(15,5))
    plt.subplot(4, 2, 5)
    plt.plot(np.arange(0, r_epochs), record_df['loss_class_cls'], 'r')
    plt.title('loss_class_cls')

    plt.subplot(4, 2, 6)
    plt.plot(np.arange(0, r_epochs), record_df['loss_class_regr'], 'r')
    plt.title('loss_class_regr')
    # plt.show()
    # plt.figure(figsize=(15,5))
    plt.subplot(4, 2, 7)
    plt.plot(np.arange(0, r_epochs), record_df['curr_loss'], 'r')
    plt.title('total_loss')

    plt.subplot(4, 2, 8)
    plt.plot(np.arange(0, r_epochs), record_df['elapsed_time'], 'r')
    plt.title('elapsed_time')

    plt.show()


def convertDataToImg(all_data, verbose=1):
    """Converts all_data from parseAnnotationFile into a list of img

    Args:
        all_data: first output from parseAnnotationFile.
            Format is list(filepath, width, height, list(bboxes))
        verbose: 0 or 1. Verbosity mode.
            0 = silent, 1 = print out details of conversion

    Returns:
        list of img array
    """

    if verbose:
        print('Retrieving images from filepaths')
        progbar = utils.Progbar(len(all_data))

        def readImg(i, name, opt):
            # print(i)
            progbar.update(i)
            return cv2.imread(name, opt)

        test_imgs = [readImg(i, all_data[i]['filepath'], cv2.IMREAD_UNCHANGED) for i in range(len(all_data))]
        progbar.update(len(all_data))  # update with last value after finishing list comprehension
        print('')
    else:
        test_imgs = [cv2.imread(d['filepath'], cv2.IMREAD_UNCHANGED) for d in all_data]

    return test_imgs
