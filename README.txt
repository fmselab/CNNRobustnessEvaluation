---DATASET ORGANIZATION---
The real label of the picture is the name of the folder that contains the images ("0" or "1")

---MODELS---
The folder "Training Results" contains all the models and all the mlb files to be used to classify the images. The files are produced by the training scripts contained in this repository.

---SCRIPTS---
* ImageClassification.py: classifies the images without apply any alteration, to evaluate the original performances of the original classifier C_0
* ImageClassification_adversarial_examples.py: evaluates the adversariability of the network
* ImageClassification_blur.py: classifies the images by applying different levels of blur alterations
* ImageClassification_brightness.py: classifies the images by applying different levels of brightness alterations
* ImageClassification_Compression.py: classifies the images by applying different levels of compression alterations
* ImageClassification_GN.py: classifies the images by applying different levels of gaussian noise alterations
* ImageClassification_horizontal_translation.py: classifies the images by applying different levels of horizontal translation
* ImageClassification_rotation.py: classifies the images by applying different levels of rotation
* ImageClassification_vertical_translation.py: classifies the images by applying different levels of vertical translation
* ImageClassification_zoom.py: classifies the images by applying different levels of zoom
* Model.py: contains the structure of the classifier
* PlotFromData.py: script that can be used to plot the variation of accuracy when an alteration is applied to the inputs
* Train.py: script that execute the training phase on the classifier C_0
* Train_Augmented.py: script that execute the training phase on the classifier C_DA
* Train_Limited_Augmented.py: script that execute the training phase on the classifier C_LDA
* Train_Limited_Parallel_Net.py: script that execute the training phase on the classifier C_LNP
* Train_Parallel_Net.py: script that execute the training phase on the classifier C_NP

---CONFIGURATION FILE---
The file config.xml contains the configuration information used by all the python scripts:
	* TrainingPath: is the path that contains the pictures to be used in the training and validation phase
	* TestPath: is the path that contains the pictures to be used in the test phase
	* Min_Accept_Accuracy: is the threshold theta
	* Model: is the main model that have to be used by the classifier
	* MLB: is the file that contains the classes to be used by the classifier
	* ParallelModel: is the parallel (optional) model that have to be used by the classifier
	* ParallelMLB: is the file that contains the classes to be used by the parallel (optional) classifier
	* EnableParallel: "1" if the network parallelization has to be enabled, "0" otherwhise
	* PlotName: is the path and the name of the plot that has to be produced by the PlotFromData.py script
	* OriginalClassifierDataFile: is the path of the CSV file, containing the accuracy values of the original reference classifier (C_0) to be plotted
	* ChangedClassifierDataFile: is the path of the CSV file, containing the accuracy values of the changed classifier to be plotted
	* OriginalClassifierLabel: is the label of the original reference classifier (C_0), to be shown on the plot
	* ChangedClassifierLabel: is the label of the changed classifier, to be shown on the plot
