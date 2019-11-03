# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %% [markdown]
#  # Configuration

# %%
# get_ipython().run_line_magic('tensorflow_version', '1.x')
import math
baseModelName = "FRCNN"
base_net_type = 'vgg'   # either 'vgg' or 'resnet50'
modelName = baseModelName + "_" + base_net_type
model_path = modelName + ".hdf5"
csv_path = modelName + ".csv"

num_epochs = 40

im_size = 300                       # shorter-side length. Original is 600, half it to save training time
anchor_box_scales = [64,128,256]    # also half box_scales accordingly. Original is [128,256,512]
anchor_box_ratios = [[1,1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]] # anchor box ratios area == 1
num_rois = 256

# %% [markdown]
#  # Load data

# %%
parseAnnotation = False

# Parsing of data especially through Google Colab is slow, so we should save the results so that we do it once only
import pickle
annotation_train_path = './data/annotation_train.txt'

if parseAnnotation:
  from FRCNN import parseAnnotationFile
  classes_of_interest = ['bicycle', 'bus', 'car', 'motorbike', 'person']
  train_data, classes_count, class_mapping = parseAnnotationFile(annotation_train_path, mode='simple', filteredList=classes_of_interest)

  with open('./data/all_data.pickle', 'wb') as f2:
      pickle.dump((train_data, classes_count, class_mapping), f2)

else:
  # Load from pickle
  with open('./data/all_data.pickle', 'rb') as f_in:
      train_data, classes_count, class_mapping = pickle.load(f_in)
  
  for i in range(len(train_data)):
    train_data[i]['filepath'] = train_data[i]['filepath'].replace('\\', '/')

# %% [markdown]
#  ## Inspect annotation file with a sample image

# %%
from FRCNN import viewAnnotatedImage
viewAnnotatedImage('./data/annotation_train.txt', 'data/train/image1095.jpg')

# %% [markdown]
#  # Create and Train FRCNN model
# %% [markdown]
#  ## Create

# %%
from FRCNN import FRCNN
num_anchors = len(anchor_box_scales) * len(anchor_box_ratios)
frcnn = FRCNN(input_shape=(None,None,3), num_anchors=num_anchors, num_rois=num_rois, base_net_type=base_net_type, num_classes = len(classes_count))
frcnn.compile()

# %% [markdown]
#  ## Visualise

# %%
frcnn.model_rpn.summary()
#frcnn.summary()

# Plot structure of FRCNN
from tensorflow.keras.utils import plot_model
plot_model(frcnn.model_all, to_file=modelName+'.png', show_shapes=True, show_layer_names=False, rankdir='TB')

# %% [markdown]
#  ## Train

# %%
## create iterator
from FRCNN import FRCNNGenerator, inspect, preprocess_input
train_it = FRCNNGenerator(train_data,
    target_size=im_size,
    horizontal_flip=True, vertical_flip=False, rotation_range=5, 
    width_shift_range=0.2,
    shuffle=True, base_net_type=base_net_type,
    preprocessing_function=preprocess_input
)

inspect(train_it, im_size)


# %%
# train model - initial_epoch = -1 --> will automatically resume training if csv and model already exists
steps = 1000
frcnn.fit_generator(train_it, target_size = im_size, class_mapping = class_mapping, epochs=num_epochs, steps_per_epoch=steps,
    model_path=model_path, csv_path=csv_path, initial_epoch=-1)

# %% [markdown]
#  # Examine Performance
# %% [markdown]
#  # Test FRCNN model

# %%
# Load records of training, and view the accuracy and loss
from FRCNN import plotAccAndLoss
plotAccAndLoss('FRCNN_vgg.csv')


# %%
import math
parseAnnotation = False

# Parsing of data especially through Google Colab is slow, so we should save the results so that we do it once only
baseModelName = "FRCNN"
base_net_type = 'vgg'   # either 'vgg' or 'resnet50'
modelName = baseModelName + "_" + base_net_type
model_path = modelName + ".hdf5"

im_size = 300                       # shorter-side length. Original is 600, half it to save training time
anchor_box_scales = [64,128,256]    # also half box_scales accordingly. Original is [128,256,512]
anchor_box_ratios = [[1,1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]] # anchor box ratios area == 1
num_rois = 256
num_anchors = len(anchor_box_scales) * len(anchor_box_ratios)


import pickle
if parseAnnotation:
  # Load image information
  from FRCNN import parseAnnotationFile
  annotation_train_path = './data/annotation_train.txt'
  classes_of_interest = ['bicycle', 'bus', 'car', 'motorbike', 'person']
  train_data, classes_count, class_mapping = parseAnnotationFile(annotation_train_path, mode='simple', filteredList=classes_of_interest)

  annotation_test_path = './data/annotation_test.txt'
  test_data, _ , _ = parseAnnotationFile(annotation_test_path, mode='simple', filteredList=classes_of_interest)

  with open('./data/all_data.pickle', 'wb') as f2:
      pickle.dump((train_data, classes_count, class_mapping), f2)
  with open('./data/test_data.pickle', 'wb') as f2:
      pickle.dump(test_data, f2)

else:
  # Load from pickle
  with open('./data/all_data.pickle', 'rb') as f_in:
      train_data, classes_count, class_mapping = pickle.load(f_in)
  
  for i in range(len(train_data)):
    train_data[i]['filepath'] = train_data[i]['filepath'].replace('\\', '/')

  with open('./data/test_data.pickle', 'rb') as f_in:
      test_data = pickle.load(f_in)


# Create model and load trained weights (Note: class mapping and num_classes should be based on training set)
from FRCNN import FRCNN
frcnn_test = FRCNN(input_shape=(None,None,3), num_anchors=num_anchors, num_rois=num_rois, base_net_type=base_net_type, num_classes = len(classes_count))
frcnn_test.load_config(anchor_box_scales=anchor_box_scales, anchor_box_ratios=anchor_box_ratios, num_rois=num_rois, target_size=im_size)
frcnn_test.load_weights(model_path)
frcnn_test.compile()

# Load array of images
from FRCNN import convertDataToImg
test_imgs = convertDataToImg(test_data)


# %%
# Perform predictions
# predicts = frcnn_test.predict(test_data, class_mapping=class_mapping, verbose=2, bbox_threshold=0.5, overlap_thres=0.2)
predicts = frcnn_test.predict(test_imgs, class_mapping=class_mapping, verbose=2, bbox_threshold=0.5, overlap_thres=0.2)


# %%
import numpy as np
evaluate = frcnn_test.evaluate(test_data, class_mapping=class_mapping, verbose=2)


# %%



