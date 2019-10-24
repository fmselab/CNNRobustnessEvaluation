from keras.preprocessing.image import img_to_array
import xml.etree.ElementTree as ET
from keras.models import load_model
import numpy as np
import imutils
from imutils import paths
import pickle
import cv2
import os

# Read the config.xml file
tree = ET.parse('config.xml')
root = tree.getroot()

# Variable to count each type of classification
true_positive = 0
false_negative = 0
true_negative = 0
false_positive = 0

# Load the trained convolutional neural network and the multi-label binarizer (file produced during the training phase)
print("[INFO] loading trained network...")
model = load_model(root[3].text)
mlb = pickle.loads(open(root[4].text, "rb").read())

# Search the images in the test path
imagePaths = sorted(list(paths.list_images(root[1].text)))
total_img = len(imagePaths)
curr_img = 0

# Loop over the test images, to classify each of them
for imagePath in imagePaths:
    curr_img = curr_img + 1

    # Load the image to be classified
    image = cv2.imread(imagePath)
    output = imutils.resize(image, width=50)

    # Pre-process the image for classification, in the same way of the pictures used for the training of the CNN
    image = cv2.resize(image, (50, 50))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # Classify the input image then find the indexes of the two class-labels with the *largest* probability
    print("[INFO] classifying image " + str(curr_img) + " of " + str(total_img) +
          " with the convolutional neural network...")
    proba = model.predict(image)[0]
    idxs = np.argsort(proba)[::-1][:2]

    # Get the predicted label and the real one
    predicted_class = 0
    predicted_prob = 0
    for (label, p) in zip(mlb.classes_, proba):
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

# Write the results in a file, to make it viewable also by SSH
root = ET.Element('Results')
true_positive_xml = ET.SubElement(root, 'TruePositive')
true_positive_xml.text = str(true_positive)
true_negative_xml = ET.SubElement(root, 'TrueNegative')
true_negative_xml.text = str(true_negative)
false_positive_xml = ET.SubElement(root, 'FalsePositive')
false_positive_xml.text = str(false_positive)
false_negative_xml = ET.SubElement(root, 'FalseNegative')
false_negative_xml.text = str(false_negative)
accuracy_xml = ET.SubElement(root, 'Accuracy')
accuracy_xml.text = str(float((float(true_positive)+float(true_negative))/(float(true_positive)+float(true_negative)+
                                                                float(false_negative)+float(false_positive))))
precision_xml = ET.SubElement(root, 'Precision')
precision_xml.text = str(float(true_positive)/(float(true_positive)+float(true_negative)))
recall_xml = ET.SubElement(root, 'Recall')
recall_xml.text = str(float(true_positive)/(float(true_positive)+float(false_negative)))
tree = ET.ElementTree(root)
tree.write("Results.xml")
