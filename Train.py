# Import the matplotlib so plot figures can be saved in the background and the necessary packages for the training
# process
import matplotlib
matplotlib.use("Agg")
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from Model import Model
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import xml.etree.ElementTree as ET
import random
import pickle
import cv2
import os

# Read the config.xml file
tree = ET.parse('config.xml')
root = tree.getroot()

# Initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 25
INIT_LR = 1e-4
BS = 500
IMAGE_DIMS = (50, 50, 3)

# Grab the image paths and randomly shuffle them, to minimize the probability of
# correlation between a picture and the, using the TRAIN Path, because the images will be augmented inside this script
print("[INFO] loading images from TRAIN path...")
imagePaths = sorted(list(paths.list_images(root[0].text)))
random.seed(42)
random.shuffle(imagePaths)

# Initialize the data and labels
data = []
labels = []

# Loop over the training input images, to get the images ant their label
for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)

	# Extract the class labels from the image path and update the
	# labels list
	l = imagePath.split(os.path.sep)[-2]
	labels.append(l)

# Scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {} images ({:.2f}MB)".format(len(imagePaths), data.nbytes / (1024 * 1000.0)))

# Binarize the labels using scikit-learn's special multi-label binarizer implementation
print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

# Loop over each of the possible class labels and show them, to see how the class has become after the binarization
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i + 1, label))

# Partition the data into training and testing splits using 70% of the data for training and the remaining 30%
# for testing, as a rule of thumb
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.3, random_state=42)

# Initialize the model using a sigmoid activation as the final layer in the network so we can perform multi-label
# classification
print("[INFO] compiling model for multi-label classification...")
model = Model.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0], depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
 	finalAct="sigmoid")

model.summary()

# Initialize the Adam optimizer
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# Compile the model using binary cross-entropy rather than categorical cross-entropy, even if we are doing
# multi-label classification. It's better to use binary crossentropy because that the goal here is to treat each
# output label as an independent Bernoulli distribution
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Construct the image generator for data augmentation
aug = ImageDataGenerator(
		rotation_range=0,						# Maximum range of rotation applied to the images
		width_shift_range=0,					# Range of the horizontal translation
		height_shift_range=0,					# Range of the vertical translation
		shear_range=0,							# Range of the shear transformation
		zoom_range=0,							# Maximum value of the zoom applied to the image
        horizontal_flip=False,					# Let's consider also the flipped image
        fill_mode='nearest')					# When new pixels are added, the value of the nearest pixel is copied

# Callback to early stop of the training procedure
#early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)

# Train the network
print("[INFO] training network using augmented data...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY),
						steps_per_epoch=len(trainX) // BS, epochs=EPOCHS, verbose=1)

# Save the model to disk, so you can use at a later time
print("[INFO] serializing network into a model file...")
model.save("Medical.model")

# Save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open("mlb.pickle", "wb")
f.write(pickle.dumps(mlb))
f.close()

# Plot the training loss and accuracy, using the validation data
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig("loss_plot.jpg")
