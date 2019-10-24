import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import csv

# Read the config.xml file
tree = ET.parse('config.xml')
root = tree.getroot()

# We will put:
# + Inside the STEPS vector, the list of all the values of the steps
# + Inside the ACCURACIES the list of the accuracies of the analyzed classifier
# + Inside the ACCURACIES2 the list of the accuracies of the original classifier C_o
steps = []
steps2 = []
accuracies = []
accuracies2 = []

# Insert the data from the files into the vectors
if root[9].text != "" and root[9].text != None:
    with open(root[9].text) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            steps2.append(float(row[0]))
            accuracies2.append(float(row[1]))

with open(root[10].text) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        steps.append(float(row[0]))
        accuracies.append(float(row[1]))

# Draw the plot
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
fig = plt.figure(figsize=(6, 3))
fig.gca().set_ylim(0.68, 0.9)
plt.grid(color='black', linestyle='-', linewidth=0.2, which='both')
plt.plot(steps, accuracies, 'k-', label=root[12].text)
if accuracies2.__len__()>0:
    plt.plot(steps2, accuracies2, 'k--', label=root[11].text)
plt.ylabel("Accuracy", {'fontname': 'Arial', 'size': '14'})

plt.legend(loc='best', shadow=True, fontsize='x-large')

plt.savefig(root[8].text)
plt.show()

