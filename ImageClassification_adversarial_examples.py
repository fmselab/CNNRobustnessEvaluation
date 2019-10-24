from keras.preprocessing.image import img_to_array
from skimage import measure
import xml.etree.ElementTree as ET
from keras.models import load_model
import numpy as np
from imutils import paths
import pickle
import cv2
import os
import foolbox


# Read the config.xml file
tree = ET.parse('config.xml')
root = tree.getroot()

# Variables to count each type of classification
true_positive = 0
false_negative = 0
true_negative = 0
false_positive = 0

# Array containing the information to be given as output
similarities = []
similarities_with_empty = []
image_names = []
correctly = []

# Load the trained convolutional neural network and the multi-label binarizer (file produced during the training phase)
print("[INFO] loading trained network...")
modelName = root[3].text
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


# Reset the value of the variables to count each type of classification
curr_img = 0
attackable = 0
non_attackable = 0
correctly_classified = 0

# Create the adversarial example for the image
preprocessing = (np.array([104, 116, 123]), 1)
fmodel = foolbox.models.KerasModel(model, bounds=(0, 255), preprocessing=preprocessing)

# Try the attack to the network
attack1 = foolbox.attacks.LBFGSAttack(model=fmodel, criterion=foolbox.criteria.TargetClassProbability(
    1, p=.51))
attack0 = foolbox.attacks.LBFGSAttack(model=fmodel, criterion=foolbox.criteria.TargetClassProbability(
    0, p=.51))

# Loop over the test images, to classify each of them
for imagePath in imagePaths:
    curr_img = curr_img + 1

    # Load the image to be classified and its dimensions
    image = cv2.imread(imagePath)
    image_names.append(imagePath)
    rows, cols, ch = image.shape

    real_label = imagePath.split(os.path.sep)[-2]

    image2 = cv2.resize(image, (50, 50))
    image2 = image2.astype("float") / 255.0
    image2 = img_to_array(image2)
    image2 = np.expand_dims(image2, axis=0)

    proba = model.predict(image2)[0]

    # Get the predicted label and the real one
    predicted_class = 0
    predicted_prob = 0
    for (label, p) in zip(mlb.classes_, proba):
        if float(p) > float(predicted_prob):
            predicted_class = label
            predicted_prob = p

    if str(predicted_class) == str(real_label):
        correctly_classified = correctly_classified + 1
        correctly.append("YES")
    else:
        correctly.append("NO")

    try:
        # Try the attack to the network
        if real_label=="1":
            adversarial = attack0(image, int(real_label))[:, :, ::-1]
        else:
            adversarial = attack1(image, int(real_label))[:, :, ::-1]

        if adversarial  is None:
            non_attackable = non_attackable + 1
            similarities_with_empty.append("")
        else:
            attackable = attackable + 1
            similarities.append(measure.compare_ssim(image[:,:,::-1], adversarial,
                                                     multichannel=True))
            similarities_with_empty.append(measure.compare_ssim(image[:,:,::-1], adversarial,
                                                     multichannel=True))
    except:
        non_attackable = non_attackable + 1
        similarities_with_empty.append("")

    if curr_img >= 200:
        break

    print("======================")
    print(curr_img)
    print("Attackable: " + str(attackable))
    print("Non attackable: " + str(non_attackable))
    if len(similarities) > 0:
        print("Similarity: " + str(similarities[len(similarities)-1]))
    print("======================")

# Compute the average similarity
print("---- AVERAGE SIMILARITY: " + str(np.mean(similarities)) + " ----")
print("---- SIMILARITY SUM: " + str(np.sum(similarities)) + " ----")
print("---- CORRECTLY CLASSIFIED: " + str(correctly_classified) + " ----")


# Write all the points in a file, so the user can obtain the plot without repeating the image classification
f = open("Similarities_" + str(modelName).split("\\")[-1] +".csv", "w+")
for i in range(0, similarities.__len__()):
    f.write(str(image_names[i]) + ";" + str(similarities_with_empty[i]).replace(".",",") + ";" + correctly[i] + "\n")
f.close()