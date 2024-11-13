import os

from matplotlib import pyplot as plt

from CNN import CNN

image_folder = "/Users/hyperspace/Desktop/PythonScripts/CNN-SkinCancerMNIST/Dataset/SkinCancerMNIST/AllImages"
csv_file = "/Users/hyperspace/Desktop/PythonScripts/CNN-SkinCancerMNIST/Dataset/SkinCancerMNIST/HAM10000_metadata.csv"

m_CNN = CNN(0.01, image_folder, csv_file, "/Users/hyperspace/Desktop/PythonScripts/CNN-SkinCancerMNIST/Final/SaveFiles")
m_CNN.__train__(55, 32, True)

history = m_CNN.__get_history__()

# Print the model results
# Graph and show the CNN values
# accuracy - the accuracy on the training dataset - evaluated during training
# val_accuracy - the accuracy on the testing dataset - evaluated after
history = dict(enumerate(history.flatten(), 1))[1]

plt.plot(history['accuracy'], label='accuracy')
plt.plot(history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()