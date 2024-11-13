import os

import keras
import matplotlib.pyplot as plt
import numpy
from keras import layers, models, losses
from keras.api.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

model = None
history = None
train_images, test_images, train_labels, test_labels = None, None, None, None

checkpoint_path = ""
save_file_directory = ""

# Define the image classes
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

class CNN:

    def __init__(self, test_split_size, image_folder, csv_file, save_file_path):
        print("Initializing...")

        global model
        global checkpoint_path
        global save_file_directory
        global train_images, test_images, train_labels, test_labels

        save_file_directory = save_file_path

        images, labels = CNN.__load_data__(self, image_folder, csv_file)

        # Set the training and testing dataset split
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_split_size, random_state=42, stratify=labels)


        # Build the model architecture
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='sigmoid'))

        model.summary()

        # Define some model parameters and compile it
        model.compile(optimizer='adam', loss=losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy', keras.metrics.AUC, keras.metrics.Precision, keras.metrics.SensitivityAtSpecificity(0.5)])

        # Define where the checkpoint files are stored
        checkpoint_path = save_file_path + "/cnn_cp.weights.h5"

        # Create a callback that saves the model's weights
        cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

        # Attempt to load the weights to train an existing model
        try:
            # Load the previously saved weights
            model.load_weights(checkpoint_path)
        except:
            print("No model saved!")

    # Load the dataset from the folder
    def __load_data__(self, image_folder, csv_file):
        # Check if the array save files exist, hopefully cutting down on computing time
        try:
            m_images = numpy.load(
                save_file_directory + "/image_array.npy")
            m_labels = numpy.load(
                save_file_directory + "/label_array.npy")

            return m_images, m_labels
        except Exception as e:
            print("Data not saved!\nSaving...\nE: " + str(e))

        # Open the CSV file
        with open(csv_file, 'r') as f:
            lines = f.readlines()

        # Initialize empty lists for images and labels
        m_images = []
        m_labels = []

        # Loop through each line (except header)
        for line in lines[1:]:  # Skip header row
            # Split the line based on a delimiter (e.g., comma)
            image_id = line.strip().split(',')[1]
            label = line.strip().split(',')[2]

            # Construct the image path
            image_path = os.path.join(image_folder, image_id + '.jpg')  # Adjust extension if needed

            # Load the image and preprocess (resize, normalize)
            image = keras.preprocessing.image.load_img(image_path, target_size=(32, 32))
            image = keras.preprocessing.image.img_to_array(image)
            image = image / 255.0  # Normalize  

            # Append image and label to respective lists
            m_images.append(image)
            m_labels.append(label)

        # Convert lists to numpy arrays
        m_images = numpy.array(m_images)
        m_labels = numpy.array(m_labels, dtype="str")

        # Encode the labels into integers as strings do not work with to_categorical
        le = LabelEncoder()
        le.fit(m_labels)
        encoded_labels = le.transform(m_labels)

        # Convert labels to categorical format if needed (depending on classification task)
        m_labels = keras.utils.to_categorical(encoded_labels, num_classes=10)  # Replace with actual number of classes

        # Save the arrays to file, cutting down on computing time
        numpy.save(save_file_directory + "/image_array.npy", m_images)
        numpy.save(save_file_directory + "/label_array.npy", m_labels)

        return m_images, m_labels

    def __train__(self, epochs, batch_size, save_weights):
        global history

        # Train the model
        history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(test_images, test_labels))

        # Test the model on the test section of the dataset
        test_loss, test_acc, test_auc, test_precision, test_specificity = model.evaluate(test_images, test_labels, verbose=2)

        # Save the weights to load later - failure protection
        if save_weights:
            model.save_weights(checkpoint_path)
            CNN.__save_history__(self, history.history)

    def __predict__(self, image_file):
        image = keras.preprocessing.image.load_img(image_file, target_size=(32, 32))
        image = keras.preprocessing.image.img_to_array(image)
        image = image / 255.0  # Normalize  
        image = keras.ops.expand_dims(image, 0)

        return model.predict(image)

    def __get_class_names__(self):
        return class_names

    def __get_history__(self):
        try:
            m_history = numpy.load(save_file_directory + "/training_history.npy", allow_pickle=True)
            return m_history
        except Exception as e:
            print("No history saved, returning current")
            print(e)
            return model.history.history

    def __save_history__(self, history):
        new_hist = numpy.array(history)

        try:
            m_history = numpy.load(save_file_directory + "/training_history.npy", allow_pickle=True)
            m_history = dict(enumerate(m_history.flatten(), 1))[1]
            new_hist = dict(enumerate(new_hist.flatten(), 1))[1]


            # print("m_history: " + str(m_history))
            # print("new_hist: " + str(new_hist))

            m_history["accuracy"].extend(new_hist['accuracy'])
            m_history["loss"].extend(new_hist['loss'])
            m_history["val_accuracy"].extend(new_hist['val_accuracy'])
            m_history["val_loss"].extend(new_hist['val_loss'])

            # hist_array = {**m_history, **new_hist}
            # print("hist_array: " + str(m_history))
        except Exception as e:
            print("No history saved!")
            print(e)
            m_history = new_hist

        numpy.save(save_file_directory + "/training_history.npy", m_history)