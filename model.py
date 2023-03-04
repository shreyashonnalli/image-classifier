# Conv2D is spatial convolution over images
# MaxPooling2D is a max pooling operation for 2D spatial data
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
# sequential is good for when we have a single dataset and looking for single output
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.applications.resnet50 import ResNet50
import imghdr
import cv2
import tensorflow as tf
import os
import random
import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, load_img
from keras.models import Model
############################################################
# DATA COLLECTION AND CONFIG
############################################################

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# removing dodgy images
# open computer vision
# allows us to check extensions of images

# variable to data directory
data_dir = 'tonsildb'

# extensions of images
img_exts = ['jpeg', 'jpg', 'bmp', 'png']

# fnames = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir)]
# img_path = fnames[1]

'''
for image_class in os.listdir(data_dir):
    if image_class == '.DS_Store':
        continue
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))
'''

'''
i = 0
for batch in train_augmentation.flow_from_directory(directory=data_dir,
                                                    batch_size=32,
                                                    target_size=(256, 256),
                                                    color_mode='rgb',
                                                    save_to_dir='augmented',
                                                    save_prefix='aug'):
    i += 1
    if i > 31:
        break
'''


'''
train_generator = train_augmentation.flow_from_directory(
    directory=tonsil
)
'''

def print_rgb_channel_values(rgb_image):
    r, g, b = tf.split(rgb_image, num_or_size_splits=3, axis=-1)
    print(r)
    print(g)
    print(b)
    
    # Convert the tensors to numpy arrays
    r_numpy = r.numpy()
    g_numpy = g.numpy()
    b_numpy = b.numpy()

    # Print the shapes and ranges of the numpy arrays for the three channels
    print('Shape of R channel:', r_numpy.shape)
    print('Range of R channel: [{}, {}]'.format(r_numpy.min(), r_numpy.max()))
    print('Shape of G channel:', g_numpy.shape)
    print('Range of G channel: [{}, {}]'.format(g_numpy.min(), g_numpy.max()))
    print('Shape of B channel:', b_numpy.shape)
    print('Range of B channel: [{}, {}]'.format(b_numpy.min(), b_numpy.max()))


# building our data pipeline
data = tf.keras.utils.image_dataset_from_directory('tonsildb', seed=42)


def color_correct(image):
    image = tf.image.adjust_contrast(image, contrast_factor=1.5)
    image = tf.image.adjust_brightness(image, delta=0.2)
    image = tf.clip_by_value(image, 0, 255)
    # image = tf.cast(image, tf.float32)/255.0
    return image


def rgb_to_ycbcr(rgb_image):
    R, G, B = tf.split(rgb_image, num_or_size_splits=3, axis=-1)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.169 * R - 0.331 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.419 * G - 0.081 * B + 128
    ycbcr_image = tf.stack([Y, Cb, Cr], axis=3)
    ycbcr_image = tf.squeeze(ycbcr_image, axis=-1)
    return ycbcr_image


def color_mask_ycbcr(ycbcr_image):
    y, cb, cr = tf.split(ycbcr_image, num_or_size_splits=3, axis=-1)
    y_min = 92
    y_max = 145
    cb_min = 112
    cb_max = 142
    cr_min = 135
    cr_max = 185
    y_mask = tf.math.logical_and(y >= y_min, y <= y_max)
    cb_mask = tf.math.logical_and(cb >= cb_min, cb <= cb_max)
    cr_mask = tf.math.logical_and(cr >= cr_min, cr <= cr_max)
    color_mask = tf.math.logical_and(
        tf.math.logical_and(y_mask, cb_mask), cr_mask)
    color_mask = tf.cast(color_mask, dtype=tf.float32)
    masked_image = ycbcr_image * color_mask
    # masked_image = tf.cast(masked_image, tf.float32)/255.0
    return masked_image

def color_mask_rgb(rgb_image):
    print(rgb_image)
    #print_rgb_channel_values(rgb_image)
    r, g, b = tf.split(rgb_image, num_or_size_splits=3, axis=-1)
    r_min = 0
    r_max = 320
    g_min = 0
    g_max = 320
    b_min = 0
    b_max = 320
    r_mask = tf.math.logical_and(r >= r_min, r <= r_max)
    g_mask = tf.math.logical_and(g >= g_min, g <= g_max)
    b_mask = tf.math.logical_and(b >= b_min, b <= b_max)
    color_mask = tf.math.logical_and(
        tf.math.logical_and(r_mask, g_mask), b_mask)
    color_mask = tf.cast(color_mask, dtype=tf.float32)
    masked_image = rgb_image * color_mask
    # masked_image = tf.cast(masked_image, tf.float32)/255.0
    return masked_image


def pipeline(data):
    #data = data.map(lambda x, y: (color_correct(x), y))
    #data = data.map(lambda x, y: (color_mask_rgb(x), y))
    #data = data.map(lambda x, y: (rgb_to_ycbcr(x), y))
    #data = data.map(lambda x, y: (color_mask_ycbcr(x), y))
    return data


def pipeline_for_individial_image(image):
    image = color_correct(image=image)
    image = rgb_to_ycbcr(rgb_image=image)
    return image


data = pipeline(data)
# for batch in data.as_numpy_iterator():
#    print(batch)


# data = data.map(lambda x, y: (color_mask(x), y))

'''
data_masked = data.map(lambda x, y: (color_mask_rgb(x), x, y))
for color_corrected_batch, original_batch, label_batch in data_masked.take(1):
    for i in range(32):
        plt.subplot(4, 16, i*2 + 1)
        plt.imshow(original_batch[i]/255.0)
        plt.axis('off')
        plt.subplot(4, 16, i*2 + 2)
        plt.imshow(color_corrected_batch[i]/255.0)
        plt.axis('off')
    plt.show()
'''


'''
data_masked = data.map(lambda x, y: (color_mask_ycbcr(x), x, y))
for color_corrected_batch, original_batch, label_batch in data_masked.take(1):
    for i in range(32):
        plt.subplot(4, 16, i*2 + 1)
        plt.imshow(original_batch[i]/255.0)
        plt.axis('off')
        plt.subplot(4, 16, i*2 + 2)
        plt.imshow(color_corrected_batch[i]/255.0)
        plt.axis('off')
    plt.show()
'''
'''
data_ycbcr = data.map(lambda x, y: (rgb_to_ycbcr(x), x, y))
for color_corrected_batch, original_batch, label_batch in data_ycbcr.take(1):
    for i in range(32):
        plt.subplot(4, 16, i*2 + 1)
        plt.imshow(original_batch[i]/255.0)
        plt.axis('off')
        plt.subplot(4, 16, i*2 + 2)
        plt.imshow(color_corrected_batch[i]/255.0)
        plt.axis('off')
    plt.show()
'''

'''
color_corrected_data = data.map(lambda x, y: (color_correct(x), x, y))
for color_corrected_batch, original_batch, label_batch in color_corrected_data.take(1):
    for i in range(32):
        plt.subplot(4, 16, i*2 + 1)
        plt.imshow(original_batch[i]/255.0)
        plt.axis('off')
        plt.subplot(4, 16, i*2 + 2)
        plt.imshow(color_corrected_batch[i]/255.0)
        plt.axis('off')
    plt.show()
'''
'''
# this is allowing us to access our data pipeline
data_iterator = data.as_numpy_iterator()

batch = data_iterator.next()

# class 1 = tonsillitis, 0 = no tonsillitis
# images represented as numpy arrays
# print(batch[0].shape)
# print(batch[1].shape)


fig, ax = plt.subplots(ncols=4)
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
plt.show()
'''


def set_seed(seed: int = 20) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed()
############################################################
# PREPROCESSING THE DATA
############################################################
train_datagen = ImageDataGenerator(
    rotation_range=10,
    horizontal_flip=True,
    rescale=1./255
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)


def return_np_array_from_dataset(ds):
    images_list = []
    labels_list = []
    for images, labels in ds.unbatch():
        print(images)
        images_list.append(images.numpy())
        labels_list.append(labels.numpy())
    images_np = np.array(images_list)
    labels_np = np.array(labels_list)
    return images_np, labels_np


# scales data between 0 and 1

# splitting into training, test and validation set
# print(len(data))
train_size = int(len(data) * .7)
# print(train_size)
val_size = int(len(data) * .2)
# print(val_size)
# +1 to make sure total is 12 as that was total no. of batches
test_size = int(len(data) * .1) + 1
# print(test_size)


train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)
#print(train)
# train_images_numpy = np.array()
train_images_np, train_labels_np = return_np_array_from_dataset(train)
val_images_np, val_labels_np = return_np_array_from_dataset(val)
test_images_np, test_labels_np = return_np_array_from_dataset(test)

train_generator = train_datagen.flow(
    train_images_np,
    train_labels_np,
    shuffle=False,
    batch_size=32
)
val_generator = val_datagen.flow(
    val_images_np,
    val_labels_np,
    shuffle=False,
    batch_size=32
)
test_generator = test_datagen.flow(
    test_images_np,
    test_labels_np,
    shuffle=False,
    batch_size=32
)

'''
images_to_show, labels_to_show = next(test_generator)
fig, axes = plt.subplots(nrows=2, ncols=32//2)
for i, ax in enumerate(axes.flatten()):
    print(i, " ", ax)
    ax.imshow(images_to_show[i])
    ax.set_title(f"Label: {labels_to_show[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()
'''

############################################################
# BUILDING THE DEEP LEARNING MODEL
############################################################
'''
model = ResNet50(include_top=False, weights='imagenet',
                 input_shape=(256, 256, 3), classes=2, pooling='avg')
resnet_model = Sequential()
resnet_model.add(model)
for layer in resnet_model.layers:
    layer.trainable = False
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(1, activation='sigmoid'))
resnet_model.compile('adam', loss=tf.losses.BinaryCrossentropy(),
                     metrics=['accuracy'])
resnet_model.summary()
'''


model = Sequential()
# model.add(resnet_model)
# Conv2D(No. of filters, dimensions of filter, activation function, expected image size(only first time))
model.add(Conv2D(32, (3, 3), 1, activation='relu',
          padding="same", input_shape=(256, 256, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())


model.add(Conv2D(32, (3, 3), 1, activation='relu'))

model.add(Conv2D(32, (3, 3), 1, activation='relu', padding="same"))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
# model.summary()


############################################################
# FITTING AND TESTING THE MODEL
############################################################


hist = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    verbose=1
)

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
plt.plot(hist.history['accuracy'], color='green', label='accuracy')
plt.plot(hist.history['val_accuracy'],
         color='black', label='val_accuracy')
fig.suptitle('Loss & accuracy')
plt.legend(loc='upper left')
plt.show()

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for i in range(2):
    x, y = test_generator.next()
    y_pred = model.predict(x)
    y_hat = []
    for x in y_pred:
        y_hat.append(x[0])
    pre.update_state(y, y_hat)
    re.update_state(y, y_hat)
    acc.update_state(y, y_hat)

print("Precision: ", pre.result().numpy())
print("Recall: ", re.result().numpy())
print("Accuracy: ", acc.result().numpy())


'''
model = Model(inputs=model.inputs, outputs=model.layers[6].output)
model.summary()
img = tf.keras.utils.load_img('tonsildb/no_pharyngitis/0iu5yjuk34567.JPG')
img = tf.keras.utils.img_to_array(img)
img = tf.image.resize(img, size=[256, 256])
img /= 255.0
img = np.expand_dims(img, axis=0)
feature_map = model.predict(img)

square = 8
ix = 1
for _ in range(square):
    for _ in range(square):
        if ix == 32:
            break
        # specify subplot and turn of axis
        ax = plt.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
    # plot filter channel in grayscale
        plt.imshow(feature_map[0, :, :, ix-1])
        ix += 1
# show the figure
plt.show()
'''


############################################################
# SAVING THE MODEL
############################################################

# Save the model and serialising it into something we can store on disk
# reloaded using load_model
model.save(os.path.join('models', 'tonsil_detector.h5'))
