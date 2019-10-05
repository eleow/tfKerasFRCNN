#%% [markdown]
# # If using Colab
#%%
# from google.colab import drive
# drive.mount('/content/drive')

# import os
# os.chdir('/content/drive/My Drive/Colab Notebooks/ISY5004')
# print(os.getcwd())

#%% [markdown]
# # Configuration
#%%
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
num_rois = 4

#%% [markdown]
# # Load data
#%%
from FRCNN import parseAnnotationFile
annotation_train_path = './annotation_train.txt'
train_imgs, classes_count, class_mapping = parseAnnotationFile(annotation_train_path)

#%% [markdown]
# ## Inspect annotation file with a sample image

#%%
from FRCNN import viewAnnotatedImage
viewAnnotatedImage(annotation_train_path, 'resize/train/image100.jpg', class_mapping)

#%% [markdown]
# # Create and Train FRCNN model

#%% [markdown]
# ## Create
#%%
from FRCNN import FRCNN
num_anchors = len(anchor_box_scales) * len(anchor_box_ratios)
frcnn = FRCNN(input_shape=(None,None,3), num_anchors=num_anchors, num_rois=num_rois, base_net_type=base_net_type, num_classes = len(classes_count))
frcnn.compile()

#%% [markdown]
# ## Visualise

#%%
frcnn.model_rpn.summary()
frcnn.summary()

# Plot structure of FRCNN
from tensorflow.keras.utils import plot_model
plot_model(frcnn.model_all, to_file=modelName+'.png', show_shapes=True, show_layer_names=False, rankdir='TB')

#%% [markdown]
# ## Train

#%%
## create iterator
train_it = frcnn.FRCNNGenerator(train_imgs,
    target_size= im_size, std_scaling=4,
    horizontal_flip=True, vertical_flip = False, rotation_range = 0, shuffle=False
)
# frcnn.inspect(train_it, im_size)

# train model - will automatically resume training if csv and model already exists
frcnn.fit_generator(train_it, target_size = im_size, class_mapping = class_mapping, epochs=num_epochs, model_path=model_path, csv_path=csv_path)


#%% [markdown]]
# # Test FRCNN model
#%%
# annotation_test_path = './annotation_test.txt'
# test_imgs, test_classes_count, test_class_mapping = parseAnnotationFile(annotation_test_path)
# test_it = frcnn.generator(test_imgs, target_size=im_size)
