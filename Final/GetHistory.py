import numpy
from matplotlib import pyplot as plt

from CNN import CNN

history_file = "/Users/hyperspace/Desktop/PythonScripts/CNN-SkinCancerMNIST/Final/SaveFiles/training_history.npy"
history = None

try:
	history = numpy.load(history_file, allow_pickle=True)
	history = dict(enumerate(history.flatten(), 1))
except Exception as e:
	print("No history!")
	print(e)

if history is not None:
	# Print the model results
	# Graph and show the CNN values
	# accuracy - the accuracy on the training dataset - evaluated during training
	# val_accuracy - the accuracy on the testing dataset - evaluated after
	plt.plot(history[1]['accuracy'], label='accuracy')
	plt.plot(history[1]['val_accuracy'], label='val_accuracy')
	plt.plot(history[1]['loss'], label='loss')
	plt.plot(history[1]['val_loss'], label='val_loss')
	plt.plot(history[1]['precision'], label='precision')
	plt.plot(history[1]['val_precision'], label='val_precision')
	plt.plot(history[1]['auc'], label='AUC')
	plt.plot(history[1]['val_auc'], label='val_AUC')
	plt.title = "Iteration 1"
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.ylim([0.5, 1])
	plt.legend(loc='lower right')
	plt.show()