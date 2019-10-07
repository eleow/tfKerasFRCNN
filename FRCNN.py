###############################################################################
#
# Faster-RCNN is composed of 3 neural networks
#   Feature Network
#   - usually a well-known pre-trained image classifier such as VGG or ResNet50,
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
# based on the work by yhenon (https://github.com/yhenon/keras-frcnn/)
# and RockyXu66 (https://github.com/RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras),
# - converted to use tensorflow.keras
# - refactored to be used as a library, following tensorflow.keras Model API
###############################################################################

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Add, Input, InputSpec, Dense, Activation, Dropout
from tensorflow.keras.layers import Flatten, BatchNormalization, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, LeakyReLU, TimeDistributed
from tensorflow.keras.initializers import he_normal

from tensorflow.keras import optimizers

import tensorflow.keras.utils as utils
import numpy as np
import pandas as pd
import cv2
import time
import random
import math
import copy
import os
import sys


DEBUG = False

# class FRCNN(tf.keras.Model):
class FRCNN():
    def __init__(self,
        base_net_type='resnet50', base_trainable=False,
        num_classes=10, input_shape=(None, None, 3),
        num_rois=32, num_anchors=9
    ):
        # super(FRCNN, self).__init__(name='frcnn')
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_rois = num_rois
        self.base_net_type = base_net_type

        # Checking of inputs for Feature Network (Base Net), allow some flexibility in name of base_net
        base_net_type = base_net_type.lower()
        if ('resnet' in base_net_type): base_net_type = 'resnet50'
        if ('vgg' in base_net_type): base_net_type = 'vgg'

        if (base_net_type  != 'resnet50' and base_net_type != 'vgg'):
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

        if (base_trainable == False):
            for layer in base_net.layers:
                layer.trainable = False
                layer._name = layer.name + "a" # prevent duplicate layer name

        # For VGG, the last max pooling layer in VGGNet is also removed
        if (base_net_type == 'vgg'):
            # base_net.layers.pop() # does not work - https://github.com/tensorflow/tensorflow/issues/22479
            feature_network = base_net.layers[-2].output
        else:
            feature_network = base_net.outputs[0]

        # Define RPN, built upon the base layers
        rpn = _rpn(feature_network, num_anchors)

        classifier = _classifier(feature_network, roi_input, num_rois, nb_classes=num_classes, trainable=True, base_net_type=base_net_type)
        self.model_rpn = Model(img_input, rpn[:2])
        self.model_classifier = Model([img_input, roi_input], classifier)


        # this will be the  model that holds both the RPN and the classifier, used to load/save weights for the models
        self.model_all = Model([img_input, roi_input], rpn[:2] + classifier)

        # return model_all

    def inspect(self, generator, target_size, rpn_stride=16, anchor_box_scales=[128,256,512], anchor_box_ratios=[[1,1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]]):
        """ Based on generator, prints details of image, ground-truth annotations, as well as positive anchors
        Args:
            generator: Generator that was created via frcnn.FRCNNGenerator
            target_size: Target size of shorter side. This should be the same as what was passed into the generator
            rpn_stride: RPN stride. This should be the same as what was passed into the generator
            anchor_box_scales: Anchor box scales array. This should be the same as what was passed into the generator
            anchor_box_ratios: Anchor box ratios array. This should be the same as what was passed into the generator

        Returns:
            None
        """
        from matplotlib import pyplot as plt

        X, Y, image_data, debug_img, debug_num_pos = next(generator)
        print('Original image: height=%d width=%d'%(image_data['height'], image_data['width']))
        print('Resized image:  height=%d width=%d im_size=%d'%(X.shape[1], X.shape[2], target_size))
        print('Feature map size: height=%d width=%d rpn_stride=%d'%(Y[0].shape[1], Y[0].shape[2], rpn_stride))
        print(X.shape)
        print(str(len(Y))+" includes 'y_rpn_cls' and 'y_rpn_regr'")
        print('Shape of y_rpn_cls {}'.format(Y[0].shape))
        print('Shape of y_rpn_regr {}'.format(Y[1].shape))
        print(image_data)

        print('Number of positive anchors for this image: %d' % (debug_num_pos))
        if debug_num_pos==0:
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
            pos_cls = np.where(cls==1)
            print(pos_cls)
            regr = Y[1][0]
            pos_regr = np.where(regr==1)
            print(pos_regr)
            print('y_rpn_cls for possible pos anchor: {}'.format(cls[pos_cls[0][0],pos_cls[1][0],:]))
            print('y_rpn_regr for positive anchor: {}'.format(regr[pos_regr[0][0],pos_regr[1][0],:]))

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
            (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,0.5,1)
            textOrg = (gt_x1, gt_y1+5)
            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
            cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
            cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

            # Draw positive anchors according to the y_rpn_regr
            for i in range(debug_num_pos):

                color = (100+i*(155/4), 0, 100+i*(155/4))

                idx = pos_regr[2][i*4]/4
                anchor_size = anchor_box_scales[int(idx/3)]
                anchor_ratio = anchor_box_ratios[2-int((idx+1)%3)]

                center = (pos_regr[1][i*4]*rpn_stride, pos_regr[0][i*4]*rpn_stride)
                print('Center position of positive anchor: ', center)
                cv2.circle(img, center, 3, color, -1)
                anc_w, anc_h = anchor_size*anchor_ratio[0], anchor_size*anchor_ratio[1]
                cv2.rectangle(img, (center[0]-int(anc_w/2), center[1]-int(anc_h/2)), (center[0]+int(anc_w/2), center[1]+int(anc_h/2)), color, 2)
        #         cv2.putText(img, 'pos anchor bbox '+str(i+1), (center[0]-int(anc_w/2), center[1]-int(anc_h/2)-5), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

        print('Green bboxes is ground-truth bbox. Others are positive anchors')
        plt.figure(figsize=(8,8))
        plt.grid()
        plt.imshow(img)
        plt.show()

    def summary(self):
        return self.model_all.summary()

    def compile(self,
            optimizer=None,
            loss=None,
            metrics=None,
            loss_weights=None,
            sample_weight_mode=None,
            weighted_metrics=None,
            target_tensors=None,
            distribute=None,

            **kwargs):

        # Allow user to override defaults
        if optimizer != None:
            optimizer_rpn = optimizer
            optimizer_classifier = optimizer
        else:
            optimizer_rpn=optimizers.Adam(lr=1e-5)
            optimizer_classifier=optimizers.Adam(lr=1e-5)

        if loss != None:
            loss_rpn = loss
            loss_classifier = loss
        else:
            loss_rpn = [rpn_loss_cls(self.num_anchors), rpn_loss_regr(self.num_anchors)]
            loss_classifier = [class_loss_cls, class_loss_regr(self.num_classes-1)]

        self.model_rpn.compile(optimizer=optimizer_rpn, loss=loss_rpn)
        self.model_classifier.compile(optimizer=optimizer_classifier,
            loss=loss_classifier, metrics={'dense_class_{}'.format(self.num_classes): 'accuracy'})
        self.model_all.compile(optimizer='sgd', loss='mae')

    def FRCNNGenerator(self, all_img_data,
        mode='train',
        shuffle=True,
        horizontal_flip=False,
        vertical_flip=False,
        rotation_range=0,
        img_channel_mean=[103.939, 116.779, 123.68],
        img_scaling_factor=1,
        std_scaling=4,
        target_size=600,

        rpn_stride=16,
        anchor_box_scales=[128,256,512],
        anchor_box_ratios=[[1,1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]],
        rpn_min_overlap = 0.3,
        rpn_max_overlap = 0.5,

        preprocessing_function=None):
        """ Yield the ground-truth anchors as Y (labels)
        Args:
            all_img_data: list(filepath, width, height, list(bboxes))
            horizontal_flip: Boolean. Randomly flip inputs horizontally.
            vertical_flip: Boolean. Randomly flip inputs vertically.
            rotation_range: Int. Degree range for random rotations (only 0 or 90 currently)
            target_size: shorter-side length. Used for image resizing based on the shorter length
            mode: 'train' or 'test'; 'train' mode need augmentation
            preprocessing_function: If None, will do zero-center by mean pixel, else will execute function.

        Returns:
            x_img: image data after resized and scaling (smallest size = 300px)
            Y: [y_rpn_cls, y_rpn_regr]
            img_data_aug: augmented image data (original image with augmentation)
            debug_img: show image for debug
            num_pos: show number of positive anchors for debug
        """
        config = {
            'horizontal_flip': horizontal_flip,
            'vertical_flip': vertical_flip,
            'rotation_range': rotation_range
        }

        if shuffle:
            np.random.seed(1)
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

                    # resize the image so that smaller side is length = 300px
                    x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
                    debug_img = x_img.copy()
                    try:
                        # calculate the output map size based on the network architecture
                        (output_width, output_height) = _get_img_output_length(resized_width, resized_height, base_net_type=self.base_net_type)

                        # # quick-fix
                        # output_width = 33
                        # output_height = 18

                        # calculate RPN
                        y_rpn_cls, y_rpn_regr, num_pos = calc_rpn(
                            img_data_aug, width, height, resized_width, resized_height, output_width, output_height,
                            rpn_stride, anchor_box_scales,
                            anchor_box_ratios, rpn_min_overlap, rpn_max_overlap)
                    except Exception as e:
                        print(e)
                        continue

                    # Zero-center by mean pixel, and preprocess image

                    if (preprocessing_function == None):
                        # Zero-center by mean pixel, and preprocess image
                        x_img = x_img[:,:, (2, 1, 0)]  # BGR -> RGB
                        x_img = x_img.astype(np.float32)
                        x_img[:, :, 0] -= img_channel_mean[0]
                        x_img[:, :, 1] -= img_channel_mean[1]
                        x_img[:, :, 2] -= img_channel_mean[2]
                        x_img /= img_scaling_factor

                        x_img = np.transpose(x_img, (2, 0, 1))
                        x_img = np.expand_dims(x_img, axis=0)
                        x_img = np.transpose(x_img, (0, 2, 3, 1))
                    else:
                        # Custom preprocessing function
                        x_img = preprocessing_function(x_img)

                    y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= std_scaling
                    y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))

                    # print("DEBUG - AAAAA")
                    # print(y_rpn_cls.shape)


                    y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

                    yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug, debug_img, num_pos

                except Exception as e:
                    print(e)
                    continue

    def fit(self,
        x=None, y=None, batch_size=None,
        epochs=1, verbose=1,
        callbacks=None,
        validation_split=0., validation_data=None,
        shuffle=True,
        class_weight=None, sample_weight=None,
        initial_epoch=0, steps_per_epoch=None,
        validation_steps=None, validation_freq=1,
        max_queue_size=10, workers=1, use_multiprocessing=False,
        **kwargs
    ):
        """Trains the model for a fixed number of epochs (iterations on a dataset).

        Arguments:
        x: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays
            (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
            (in case the model has multiple inputs).
          - A dict mapping input names to the corresponding array/tensors,
            if the model has named inputs.
          - A `tf.data` dataset. Should return a tuple
            of either `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
          - A generator or `keras.utils.Sequence` returning `(inputs, targets)`
            or `(inputs, targets, sample weights)`.
        y: Target data. Like the input data `x`,
          it could be either Numpy array(s) or TensorFlow tensor(s).
          It should be consistent with `x` (you cannot have Numpy inputs and
          tensor targets, or inversely). If `x` is a dataset, generator,
          or `keras.utils.Sequence` instance, `y` should
          not be specified (since targets will be obtained from `x`).
        batch_size: Integer or `None`.
            Number of samples per gradient update.
            If unspecified, `batch_size` will default to 32.
            Do not specify the `batch_size` if your data is in the
            form of symbolic tensors, datasets,
            generators, or `keras.utils.Sequence` instances (since they generate
            batches).
        epochs: Integer. Number of epochs to train the model.
            An epoch is an iteration over the entire `x` and `y`
            data provided.
            Note that in conjunction with `initial_epoch`,
            `epochs` is to be understood as "final epoch".
            The model is not trained for a number of iterations
            given by `epochs`, but merely until the epoch
            of index `epochs` is reached.
        verbose: 0, 1, or 2. Verbosity mode.
            0 = silent, 1 = progress bar, 2 = one line per epoch.
            Note that the progress bar is not particularly useful when
            logged to a file, so verbose=2 is recommended when not running
            interactively (eg, in a production environment).
        callbacks: List of `keras.callbacks.Callback` instances.
            List of callbacks to apply during training.
            See `tf.keras.callbacks`.
        validation_split: Float between 0 and 1.
            Fraction of the training data to be used as validation data.
            The model will set apart this fraction of the training data,
            will not train on it, and will evaluate
            the loss and any model metrics
            on this data at the end of each epoch.
            The validation data is selected from the last samples
            in the `x` and `y` data provided, before shuffling. This argument is
            not supported when `x` is a dataset, generator or
           `keras.utils.Sequence` instance.
        validation_data: Data on which to evaluate
            the loss and any model metrics at the end of each epoch.
            The model will not be trained on this data.
            `validation_data` will override `validation_split`.
            `validation_data` could be:
              - tuple `(x_val, y_val)` of Numpy arrays or tensors
              - tuple `(x_val, y_val, val_sample_weights)` of Numpy arrays
              - dataset
            For the first two cases, `batch_size` must be provided.
            For the last case, `validation_steps` must be provided.
        shuffle: Boolean (whether to shuffle the training data
            before each epoch) or str (for 'batch').
            'batch' is a special option for dealing with the
            limitations of HDF5 data; it shuffles in batch-sized chunks.
            Has no effect when `steps_per_epoch` is not `None`.
        class_weight: Optional dictionary mapping class indices (integers)
            to a weight (float) value, used for weighting the loss function
            (during training only).
            This can be useful to tell the model to
            "pay more attention" to samples from
            an under-represented class.
        sample_weight: Optional Numpy array of weights for
            the training samples, used for weighting the loss function
            (during training only). You can either pass a flat (1D)
            Numpy array with the same length as the input samples
            (1:1 mapping between weights and samples),
            or in the case of temporal data,
            you can pass a 2D array with shape
            `(samples, sequence_length)`,
            to apply a different weight to every timestep of every sample.
            In this case you should make sure to specify
            `sample_weight_mode="temporal"` in `compile()`. This argument is not
            supported when `x` is a dataset, generator, or
           `keras.utils.Sequence` instance, instead provide the sample_weights
            as the third element of `x`.
        initial_epoch: Integer.
            Epoch at which to start training
            (useful for resuming a previous training run).
        steps_per_epoch: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring one epoch finished and starting the
            next epoch. When training with input tensors such as
            TensorFlow data tensors, the default `None` is equal to
            the number of samples in your dataset divided by
            the batch size, or 1 if that cannot be determined. If x is a
            `tf.data` dataset, and 'steps_per_epoch'
            is None, the epoch will run until the input dataset is exhausted.
            This argument is not supported with array inputs.
        validation_steps: Only relevant if `validation_data` is provided and
            is a `tf.data` dataset. Total number of steps (batches of
            samples) to draw before stopping when performing validation
            at the end of every epoch. If validation_data is a `tf.data` dataset
            and 'validation_steps' is None, validation
            will run until the `validation_data` dataset is exhausted.
        validation_freq: Only relevant if validation data is provided. Integer
            or `collections_abc.Container` instance (e.g. list, tuple, etc.).
            If an integer, specifies how many training epochs to run before a
            new validation run is performed, e.g. `validation_freq=2` runs
            validation every 2 epochs. If a Container, specifies the epochs on
            which to run validation, e.g. `validation_freq=[1, 2, 10]` runs
            validation at the end of the 1st, 2nd, and 10th epochs.
        max_queue_size: Integer. Used for generator or `keras.utils.Sequence`
            input only. Maximum size for the generator queue.
            If unspecified, `max_queue_size` will default to 10.
        workers: Integer. Used for generator or `keras.utils.Sequence` input
            only. Maximum number of processes to spin up
            when using process-based threading. If unspecified, `workers`
            will default to 1. If 0, will execute the generator on the main
            thread.
        use_multiprocessing: Boolean. Used for generator or
            `keras.utils.Sequence` input only. If `True`, use process-based
            threading. If unspecified, `use_multiprocessing` will default to
            `False`. Note that because this implementation relies on
            multiprocessing, you should not pass non-picklable arguments to
            the generator as they can't be passed easily to children processes.
        **kwargs: Used for backwards compatibility.
    Returns:
        A `History` object. Its `History.history` attribute is
        a record of training loss values and metrics values
        at successive epochs, as well as validation loss values
        and validation metrics values (if applicable).
    Raises:
        RuntimeError: If the model was never compiled.
        ValueError: In case of mismatch between the provided input data
            and what the model expects.
        """
        #TODO

        return None

    def fit_generator(self,
        generator,                  #
        steps_per_epoch=None,       #
        epochs=1,                   # Yes
        verbose=1,                  # Yes
        callbacks=None,             #
        validation_data=None,       #
        validation_steps=None,      #
        validation_freq=1,          #
        class_weight=None,          #
        max_queue_size=10,          #
        workers=1,                  #
        use_multiprocessing=False,  #
        shuffle=True,               #
        initial_epoch=0,             # Yes
                                    #### customs
        class_mapping=None,
        target_size=-1,                # length of shorter size
        anchor_box_scales=[128,256,512],
        anchor_box_ratios=[[1,1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]],
        std_scaling= 4.0,                           # for scaling of standard deviation
        classifier_regr_std=[8.0, 8.0, 4.0, 4.0],   #
        classifier_min_overlap = 0.1,
        classifier_max_overlap = 0.5,
        rpn_stride=16,                              # stride at the RPN (this depends on the network configuration)

        model_path='./frcnn.hdf5',
        csv_path="./frcnn.csv",
        ):
        """Fits the model on data yielded batch-by-batch by a Python generator.
        The generator is run in parallel to the model, for efficiency.
        For instance, this allows you to do real-time data augmentation
        on images on CPU in parallel to training your model on GPU.
        The use of `keras.utils.Sequence` guarantees the ordering
        and guarantees the single use of every input per epoch when
        using `use_multiprocessing=True`.
        Arguments:
            generator: A generator or an instance of `Sequence`
            (`keras.utils.Sequence`)
                object in order to avoid duplicate data
                when using multiprocessing.
                The output of the generator must be either
                - a tuple `(inputs, targets)`
                - a tuple `(inputs, targets, sample_weights)`.
                This tuple (a single output of the generator) makes a single batch.
                Therefore, all arrays in this tuple must have the same length (equal
                to the size of this batch). Different batches may have different
                sizes.
                For example, the last batch of the epoch is commonly smaller than
                the
                others, if the size of the dataset is not divisible by the batch
                size.
                The generator is expected to loop over its data
                indefinitely. An epoch finishes when `steps_per_epoch`
                batches have been seen by the model.
            steps_per_epoch: Total number of steps (batches of samples)
                to yield from `generator` before declaring one epoch
                finished and starting the next epoch. It should typically
                be equal to the number of samples of your dataset
                divided by the batch size.
                Optional for `Sequence`: if unspecified, will use
                the `len(generator)` as a number of steps.
            epochs: Integer, total number of iterations on the data.
            verbose: Verbosity mode, 0, 1, or 2.
            callbacks: List of callbacks to be called during training.
            validation_data: This can be either
                - a generator for the validation data
                - a tuple (inputs, targets)
                - a tuple (inputs, targets, sample_weights).
            validation_steps: Only relevant if `validation_data`
                is a generator. Total number of steps (batches of samples)
                to yield from `generator` before stopping.
                Optional for `Sequence`: if unspecified, will use
                the `len(validation_data)` as a number of steps.
            validation_freq: Only relevant if validation data is provided. Integer
                or `collections_abc.Container` instance (e.g. list, tuple, etc.).
                If an integer, specifies how many training epochs to run before a
                new validation run is performed, e.g. `validation_freq=2` runs
                validation every 2 epochs. If a Container, specifies the epochs on
                which to run validation, e.g. `validation_freq=[1, 2, 10]` runs
                validation at the end of the 1st, 2nd, and 10th epochs.
            class_weight: Dictionary mapping class indices to a weight
                for the class.
            max_queue_size: Integer. Maximum size for the generator queue.
                If unspecified, `max_queue_size` will default to 10.
            workers: Integer. Maximum number of processes to spin up
                when using process-based threading.
                If unspecified, `workers` will default to 1. If 0, will
                execute the generator on the main thread.
            use_multiprocessing: Boolean.
                If `True`, use process-based threading.
                If unspecified, `use_multiprocessing` will default to `False`.
                Note that because this implementation relies on multiprocessing,
                you should not pass non-picklable arguments to the generator
                as they can't be passed easily to children processes.
            shuffle: Boolean. Whether to shuffle the order of the batches at
                the beginning of each epoch. Only used with instances
                of `Sequence` (`keras.utils.Sequence`).
                Has no effect when `steps_per_epoch` is not `None`.
            initial_epoch: Epoch at which to start training
                (useful for resuming a previous training run)
        Returns:
            A `History` object.
        Example:
        ```python
            def generate_arrays_from_file(path):
                while 1:
                    f = open(path)
                    for line in f:
                        # create numpy arrays of input data
                        # and labels, from each line in the file
                        x1, x2, y = process_line(line)
                        yield ({'input_1': x1, 'input_2': x2}, {'output': y})
                    f.close()
            model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                                steps_per_epoch=10000, epochs=10)
        ```
        Raises:
            ValueError: In case the generator yields data in an invalid format.
            """
        epoch_length = 1000
        iter_num = 0

        losses = np.zeros((epoch_length, 5))
        rpn_accuracy_rpn_monitor = []
        rpn_accuracy_for_epoch = []

        best_loss = np.Inf

        # input validation
        if (class_mapping == None):
            print("class_mapping should not be None")
            raise ValueError
        elif (target_size < 0):
            print("target_size (shorter-side size) must be a positive integer")
            raise ValueError

        print()
        # let's check if model file exists
        if not os.path.isfile(model_path):

            print('Starting training')

            # Create the record.csv file to record losses, acc and mAP
            record_df = pd.DataFrame(columns=['mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls', 'loss_rpn_regr', 'loss_class_cls', 'loss_class_regr', 'curr_loss', 'elapsed_time', 'mAP'])
        else:
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

            if verbose: print('Already trained %dK batches'% (len(record_df)))

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
        #                 print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
                        if mean_overlapping_bboxes == 0:
                            print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

                    # Generate X (x_img) and label Y ([y_rpn_cls, y_rpn_regr])
                    X, Y, img_data, debug_img, debug_num_pos = next(generator)
                    if DEBUG: print("DEBUG", img_data['filepath'])

                    # Train rpn model and get loss value [_, loss_rpn_cls, loss_rpn_regr]
                    loss_rpn = self.model_rpn.train_on_batch(X, Y)

                    # Get predicted rpn from rpn model [rpn_cls, rpn_regr]
                    P_rpn = self.model_rpn.predict_on_batch(X)

                    # R: bboxes (shape=(300,4))
                    # Convert rpn layer to roi bboxes
                    R = rpn_to_roi(P_rpn[0], P_rpn[1],
                        std_scaling, anchor_box_ratios, anchor_box_scales, rpn_stride,
                        use_regr=True, overlap_thresh=0.7, max_boxes=300)

                    # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                    # X2: bboxes that iou > C.classifier_min_overlap for all gt bboxes in 300 non_max_suppression bboxes
                    # Y1: one hot code for bboxes from above => x_roi (X)
                    # Y2: corresponding labels and corresponding gt bboxes
                    X2, Y1, Y2, IouS = calc_iou(R, img_data, [classifier_min_overlap, classifier_max_overlap], target_size, rpn_stride, class_mapping, classifier_regr_std)
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
                            except:
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

                    progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                            ('final_cls', np.mean(losses[:iter_num, 2])), ('final_regr', np.mean(losses[:iter_num, 3]))])

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
                                print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
                            best_loss = curr_loss
                            self.model_all.save_weights(model_path)

                        new_row = {'mean_overlapping_bboxes':round(mean_overlapping_bboxes, 3),
                                'class_acc':round(class_acc, 3),
                                'loss_rpn_cls':round(loss_rpn_cls, 3),
                                'loss_rpn_regr':round(loss_rpn_regr, 3),
                                'loss_class_cls':round(loss_class_cls, 3),
                                'loss_class_regr':round(loss_class_regr, 3),
                                'curr_loss':round(curr_loss, 3),
                                'elapsed_time':round(elapsed_time, 3),
                                'mAP': 0}

                        record_df = record_df.append(new_row, ignore_index=True)
                        record_df.to_csv(csv_path, index=0)

                        break

                except Exception as e:
                    print('Exception: {}'.format(e))
                    continue

        print('-- Training complete, exiting.')
        return None

def _get_img_output_length(width, height, base_net_type='resnet50'):
    b = base_net_type
    def get_output_length(input_length, b):
        if (b=='resnet50'):
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

    return get_output_length(width,b), get_output_length(height,b)


def _rpn(base_layers, num_anchors):
    # common layer fed to 2 layers
    # - x_class for classification (is object in bounding box?)
    # - x_regr for bounding box regression (ROIs)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)
    return [x_class, x_regr, base_layers]

def _classifier(base_layers, input_rois, num_rois, nb_classes = 4, trainable=True, base_net_type='resnet50'):

    if (base_net_type == 'resnet50'):
        pooling_regions = 14
        input_shape = (num_rois, pooling_regions, pooling_regions, 1024)
        out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

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
        out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

        # flatten convolution layer and connect to 2 FC with dropout
        print(out_roi_pool.shape)
        out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
        out = TimeDistributed(Dropout(0.5))(out)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
        out = TimeDistributed(Dropout(0.5))(out)

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

    x = TimeDistributed(Conv2D(nb_filter2, (kernel_size, kernel_size), trainable=trainable, kernel_initializer='normal',padding='same'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter3, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2c')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x


###############################################################################
# Definition for custom layers
import tensorflow.keras.backend as K
from tensorflow.keras import initializers, regularizers
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

#         self.dim_ordering = K.image_dim_ordering()
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

        input_shape = K.shape(img)

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
from tensorflow.keras.backend import categorical_crossentropy

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
    IoUs = [] # for debugging only

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
    Y2 = np.concatenate([np.array(y_class_regr_label),np.array(y_class_regr_coords)],axis=1)

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

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break

    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick].astype("int")
    probs = probs[pick]
    return boxes, probs

import time
def rpn_to_roi(rpn_layer, regr_layer, std_scaling, anchor_box_ratios, anchor_box_scales, rpn_stride, use_regr=True, max_boxes=300,overlap_thresh=0.9):

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

            X, Y = np.meshgrid(np.arange(cols),np. arange(rows))

            A[0, :, :, curr_layer] = X - anchor_x/2
            A[1, :, :, curr_layer] = Y - anchor_y/2
            A[2, :, :, curr_layer] = anchor_x
            A[3, :, :, curr_layer] = anchor_y

            if use_regr:
                A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], regr)

            A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer])
            A[2, :, :, curr_layer] += A[0, :, :, curr_layer]
            A[3, :, :, curr_layer] += A[1, :, :, curr_layer]

            A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer])
            A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])
            A[2, :, :, curr_layer] = np.minimum(cols-1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.minimum(rows-1, A[3, :, :, curr_layer])

            curr_layer += 1

    all_boxes = np.reshape(A.transpose((0, 3, 1,2)), (4, -1)).transpose((1, 0))
    all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))

    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]

    idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))

    all_boxes = np.delete(all_boxes, idxs, 0)
    all_probs = np.delete(all_probs, idxs, 0)

    result = non_max_suppression_fast(all_boxes, all_probs, overlap_thresh=overlap_thresh, max_boxes=max_boxes)[0]

    return result

###############################################################################
# Data generator and data augmentation
import numpy as np
import cv2
import random
import copy
import threading
import itertools

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

def augment(img_data, config, augment=True):
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    assert 'width' in img_data
    assert 'height' in img_data

    config = dotdict(config)

    img_data_aug = copy.deepcopy(img_data)
    img = cv2.imread(img_data_aug['filepath'])

    if augment:
        rows, cols = img.shape[:2]

        if config.horizontal_flip and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 1)
            for bbox in img_data_aug['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                bbox['x2'] = cols - x1
                bbox['x1'] = cols - x2

        if config.vertical_flip and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 0)
            for bbox in img_data_aug['bboxes']:
                y1 = bbox['y1']
                y2 = bbox['y2']
                bbox['y2'] = rows - y1
                bbox['y1'] = rows - y2

        if config.rotation_range == 90:
            angle = np.random.choice([0,90,180,270],1)[0]
            if angle == 270:
                img = np.transpose(img, (1,0,2))
                img = cv2.flip(img, 0)
            elif angle == 180:
                img = cv2.flip(img, -1)
            elif angle == 90:
                img = np.transpose(img, (1,0,2))
                img = cv2.flip(img, 1)
            elif angle == 0:
                pass

            for bbox in img_data_aug['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                y1 = bbox['y1']
                y2 = bbox['y2']
                if angle == 270:
                    bbox['x1'] = y1
                    bbox['x2'] = y2
                    bbox['y1'] = cols - x2
                    bbox['y2'] = cols - x1
                elif angle == 180:
                    bbox['x2'] = cols - x1
                    bbox['x1'] = cols - x2
                    bbox['y2'] = rows - y1
                    bbox['y1'] = rows - y2
                elif angle == 90:
                    bbox['x1'] = rows - y2
                    bbox['x2'] = rows - y1
                    bbox['y1'] = x1
                    bbox['y2'] = x2
                elif angle == 0:
                    pass

    img_data_aug['width'] = img.shape[1]
    img_data_aug['height'] = img.shape[0]
    return img_data_aug, img


# class SampleSelector:
#     def __init__(self, class_count):
#         # ignore classes that have zero samples
#         self.classes = [b for b in class_count.keys() if class_count[b] > 0]
#         self.class_cycle = itertools.cycle(self.classes)
#         self.curr_class = next(self.class_cycle)

#     def skip_sample_for_balanced_class(self, img_data):

#         class_in_img = False

#         for bbox in img_data['bboxes']:

#             cls_name = bbox['class']

#             if cls_name == self.curr_class:
#                 class_in_img = True
#                 self.curr_class = next(self.class_cycle)
#                 break

#         if class_in_img:
#             return False
#         else:
#             return True

def calc_rpn(img_data, width, height, resized_width, resized_height, output_width, output_height,
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
                                best_x_for_bbox[bbox_num,:] = [x1_anc, x2_anc, y1_anc, y2_anc]
                                best_dx_for_bbox[bbox_num,:] = [tx, ty, tw, th]

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
                best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
                best_anchor_for_bbox[idx,3]] = 1
            y_rpn_overlap[
                best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
                best_anchor_for_bbox[idx,3]] = 1
            start = 4 * (best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3])
            y_rpn_regr[
                best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], start:start+4] = best_dx_for_bbox[idx, :]

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
# Parser for annotations
def parseAnnotationFile(input_path, verbose=1, split=None):
    """Parse the data from annotation file (each line should contain filepath,x1,y1,x2,y2,class_name)

    Args:
        input_path: annotation file path
        verbose: 0, 1, or 2. Verbosity mode.
            0 = silent, 1 = print out details of annotation file

    Returns:
        all_data: list(filepath, width, height, list(bboxes))
        classes_count: dict{key:class_name, value:count_num}
            e.g. {'Car': 2383, 'Mobile phone': 1108, 'Person': 3745}
        class_mapping: dict{key:class_name, value: idx}
            e.g. {'Car': 0, 'Mobile phone': 1, 'Person': 2}
    """
    found_bg = False
    all_imgs = {}
    classes_count = {}
    class_mapping = {}
    visualise = True
    i = 1

    st = time.time()
    with open(input_path,'r') as f:

        if verbose: print('Parsing annotation files')

        for line in f:
            sys.stdout.write('\r'+'idx=' + str(i))
            i += 1


            line_split = line.strip().split(',')
            (filename,x1,y1,x2,y2,class_name) = line_split

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    if verbose: print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            if filename not in all_imgs:
                filename = filename.replace('\\', '/')  # in case backslash is used, we will replace with forward slash instead

                all_imgs[filename] = {}

                if (not os.path.isfile(filename)): print(filename + " could not be read")
                else:
                    img = cv2.imread(filename)
                    (rows,cols) = img.shape[:2]
                    all_imgs[filename]['filepath'] = filename
                    all_imgs[filename]['width'] = cols
                    all_imgs[filename]['height'] = rows
                    all_imgs[filename]['bboxes'] = []
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

        if (verbose):
            print()
            print('Spend %0.2f mins to load the data' % ((time.time()-st)/60) )

        return all_data, classes_count, class_mapping

# Viewer for annotated image
def viewAnnotatedImage(annotation_file, query_image_path, class_mapping, verbose=1, palette=None):
    """Views the annotated image based on an annotation file and a query image path

    Args:
        annotation_file: annotation file path
        query_image_path: path of the image to check. eg 'resize/train/image100.jpg'.
            This should correspond exactly with the annotation file's first column
        verbose: 0, 1, or 2. Verbosity mode.
            0 = silent, 1 = print out details of image and annotation file
        palette: choose between 'base', 'tableau', 'css' or None
            Colors for objects will be randomly picked from the palette if specified,
            If None, a random RGB color will be assigned
    Returns:
        None
    """
    from matplotlib import pyplot as plt
    import matplotlib.colors as mcolors

    annotations = pd.read_csv(annotation_file, sep = ',', names = ['image_name','x1','y1','x2','y2','Object_type'])
    num_classes = annotations['Object_type'].nunique()

    colorset = np.random.uniform(0, 255, size=(num_classes, 3))
    # if (palette != None):
    #     colors = None
    #     if (palette.lower() == 'tableau'):
    #         colors = list(mcolors.TABLEAU_COLORS)
    #     elif (palette.lower() == 'css'):
    #         colors= list(mcolors.CSS4_COLORS)
    #     elif (palette.lower() == 'base'):
    #         colors = list(mcolors.BASE_COLORS)
    #     else:
    #         print('Invalid palette. Default random colors will be used')

    #     if (colors != None):
    #         colorset = np.random.choice(colors, num_classes, replace=(num_classes > len(colors)))

    img = plt.imread(query_image_path)
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
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
        cv2.putText(img, t[0], t[1], cv2.FONT_HERSHEY_DUPLEX, 0.5, 255 - t[2], 2, cv2.LINE_AA)
        cv2.putText(img, t[0], t[1], cv2.FONT_HERSHEY_DUPLEX, 0.5, t[2], 1, cv2.LINE_AA)



    plt.grid()
    plt.imshow(img)
    plt.show()

    return None
