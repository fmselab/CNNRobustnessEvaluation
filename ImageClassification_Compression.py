import matplotlib
matplotlib.use("Agg")
from keras.preprocessing.image import img_to_array
import xml.etree.ElementTree as ET
from keras.models import load_model
import numpy as np
import imutils
from imutils import paths
import pickle
import cv2
import os
import Functions
import matplotlib.pyplot as plt

# Read the config.xml file
tree = ET.parse('config.xml')
root = tree.getroot()

# Variables to count each type of classification
true_positive = 0
false_negative = 0
true_negative = 0
false_positive = 0

# Array containing the information to be plotted
steps = []
accuracies = []

# Load the trained convolutional neural network and the multi-label binarizer (file produced during the training phase)
print("[INFO] loading trained network...")
model = load_model(root[3].text)
mlb = pickle.loads(open(root[4].text, "rb").read())

# Load the information on the parallel model, if it is enabled
if str(root[7].text) == "1":
    model_par = load_model(root[5].text)
    mlb_par = pickle.loads(open(root[6].text, "rb").read())

# Search the images in the test path
imagePaths = sorted(list(paths.list_images(root[1].text)))
total_img = len(imagePaths)
curr_img = 0

for step in Functions.frange(0, 100, 2.5):
    # Add the information about the step in the steps array
    steps.append(step)

    # Reset the value of the variables to count each type of classification
    curr_img = 0
    true_positive = 0
    false_negative = 0
    true_negative = 0
    false_positive = 0

    # Loop over the test images, to classify each of them
    for imagePath in imagePaths:
        curr_img = curr_img + 1

        # Load the image to be classified, and compress in JPG format
        if step != 0:
            image = cv2.imread(imagePath)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100-(step)]
            result, image = cv2.imencode('.jpg', image, encode_param)
            image = cv2.imdecode(image, 1)
        else:
            image = cv2.imread(imagePath)

        output = imutils.resize(image, width=50)

        # Pre-process the image for classification, in the same way of the pictures used for the training of the CNN
        image = cv2.resize(image, (50, 50))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # Classify the input image then find the indexes of the two class-labels with the *largest* probability
        print("[INFO] classifying image " + str(curr_img) + " of " + str(total_img) +
              " with the convolutional neural network... STEP: " + str(step))
        proba = model.predict(image)[0]
        idxs = np.argsort(proba)[::-1][:2]

        # Classify the input image with the parallel model
        if str(root[7].text) == "1":
            proba_par = model_par.predict(image)[0]
            idxs_par = np.argsort(proba_par)[::-1][:2]

        # Get the predicted label and the real one
        predicted_class = 0
        predicted_prob = 0
        for (label, p) in zip(mlb.classes_, proba):
            if float(p) > float(predicted_prob):
                predicted_class = label
                predicted_prob = p

        # Look also the classification produced by the parallel model
        if str(root[7].text) == "1":
            for (label, p) in zip(mlb_par.classes_, proba_par):
                if float(p) > float(predicted_prob):
                    predicted_class = label
                    predicted_prob = p

        real_label = imagePath.split(os.path.sep)[-2]

        # Classify the type of the classification
        if str(predicted_class) == "0" and str(real_label) == "0":
            true_negative = true_negative + 1
        elif str(predicted_class) == "1" and str(real_label) == "1":
            true_positive = true_positive + 1
        elif str(predicted_class) == "1" and str(real_label) == "0":
            false_positive = false_positive + 1
        else:
            false_negative = false_negative + 1

    # Plot the information
    accuracies.append(float((float(true_positive)+float(true_negative))/(float(true_positive)+float(true_negative)+
                                                                    float(false_negative)+float(false_positive))))
# Plot the training loss and accuracy, using the validation data
plt.style.use("ggplot")
plt.figure()
plt.plot(steps, accuracies)
#plt.title("Accuracy")
#plt.xlabel("Image alteration - Compression")
plt.ylabel("Accuracy")
plt.savefig("accuracy_plot_compression.jpg")

# Write the results in a file, to make it viewable also by SSH
root_out = ET.Element('Results')
robustness_xml = ET.SubElement(root_out, 'Robustness')
robustness_xml.text = str(Functions.compute_robustness(root[2].text, accuracies))
tree = ET.ElementTree(root_out)
tree.write("Results_compression.xml")

# Write all the points in a file, so the user can obtain the plot without repeating the image classification
f = open("Data_compression.csv", "w+")
for i in range(0, steps.__len__()):
    f.write(str(steps[i]) + "," + str(accuracies[i]) + "\r\n")
f.close()
