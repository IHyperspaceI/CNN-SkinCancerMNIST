import numpy

from CNN import CNN

image_folder = "/Users/hyperspace/Desktop/PythonScripts/CNN-SkinCancerMNIST/Dataset/SkinCancerMNIST/AllImages"
csv_file = "/Users/hyperspace/Desktop/PythonScripts/CNN-SkinCancerMNIST/Dataset/SkinCancerMNIST/HAM10000_metadata.csv"

m_CNN = CNN(0.25, image_folder, csv_file, "/Users/hyperspace/Desktop/PythonScripts/CNN-SkinCancerMNIST/Final/SaveFiles")

class_names = numpy.array(m_CNN.__get_class_names__())

prediction = m_CNN.__predict__("/Users/hyperspace/Downloads/ISIC_0024306.jpg")

print("Certainties:")
i = 0
while i < len(prediction[0]):
	print(class_names[i] + ": " + str(prediction[0][i]))
	i += 1