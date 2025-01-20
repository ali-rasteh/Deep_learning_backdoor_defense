import keras
import sys
import h5py
import numpy as np
from architecture import *

clean_data_filename = str(sys.argv[1])
poisoned_data_filename = str(sys.argv[2])
model_filename = str(sys.argv[3])

def data_loader(filepath):
    """
    Loads data from an HDF5 file and returns the data and labels.
    Args:
        filepath (str): The path to the HDF5 file containing the data.
    Returns:
        tuple: A tuple containing:
            - x_data (numpy.ndarray): The data array with shape (num_samples, height, width, channels).
            - y_data (numpy.ndarray): The labels array.
    """

    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def main():
    """
    Main function to evaluate the performance of a backdoor defense model.
    This function loads clean and poisoned test datasets, loads a pre-trained model,
    and evaluates the model's performance on both datasets. It prints the clean 
    classification accuracy and the attack success rate.
    The function performs the following steps:
    1. Loads the clean and poisoned test datasets using the `data_loader` function.
    2. Loads the pre-trained model using `keras.models.load_model`.
    3. Predicts the labels for the clean test dataset and calculates the clean classification accuracy.
    4. Predicts the labels for the poisoned test dataset and calculates the attack success rate.
    Variables:
    - clean_data_filename: Path to the clean test dataset file.
    - poisoned_data_filename: Path to the poisoned test dataset file.
    - model_filename: Path to the pre-trained model file.
    Prints:
    - Clean Classification accuracy: The accuracy of the model on the clean test dataset.
    - Attack Success Rate: The success rate of the backdoor attack on the poisoned test dataset.
    """

    cl_x_test, cl_y_test = data_loader(clean_data_filename)
    bd_x_test, bd_y_test = data_loader(poisoned_data_filename)

    bd_model = keras.models.load_model(model_filename)
    # bd_model = keras.models.load_model(model_filename, custom_objects={"ChannelPruningLayer": ChannelPruningLayer, "CompareAndSelectLayer": CompareAndSelectLayer})

    cl_label_p = np.argmax(bd_model.predict(cl_x_test), axis=1)
    clean_accuracy = np.mean(np.equal(cl_label_p, cl_y_test))*100
    print('Clean Classification accuracy:', clean_accuracy)
    
    bd_label_p = np.argmax(bd_model.predict(bd_x_test), axis=1)
    asr = np.mean(np.equal(bd_label_p, bd_y_test))*100
    print('Attack Success Rate:', asr)

if __name__ == '__main__':
    main()
