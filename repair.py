import keras
from keras.models import Model
import sys
import h5py
import numpy as np
from architecture import *

clean_valid_data_filename = str(sys.argv[1])
# clean_test_data_filename = str(sys.argv[2])
# poisoned_valid_data_filename = str(sys.argv[3])
# poisoned_test_data_filename = str(sys.argv[4])
model_filename = str(sys.argv[2])
X = [2,4,10]
pruned_models = []


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
    Main function to perform model pruning and evaluate the impact on clean validation accuracy.
    This function performs the following steps:
    1. Loads clean validation data.
    2. Loads a pre-trained backdoored model.
    3. Evaluates the base clean validation accuracy of the model.
    4. Identifies the channels in a specified pooling layer to prune based on average activations.
    5. Iteratively prunes channels and evaluates the clean validation accuracy after each pruning step.
    6. Saves pruned models and corresponding "GoodNet" models when a specified drop in clean validation accuracy is reached.
    Variables:
    - cl_x_valid, cl_y_valid: Clean validation data and labels.
    - bd_model: Loaded backdoored model.
    - clean_accuracy_base: Base clean validation accuracy of the backdoored model.
    - layer_to_prune: Name of the layer to prune.
    - pruning_layer: Name of the pruning layer.
    - num_channels: Number of channels in the layer to prune.
    - activations: Activations of the specified layer for the clean validation data.
    - average_activations: Average activations of the channels.
    - sorted_indices: Indices of channels sorted by average activations.
    - channel_ids: List of channel IDs to prune.
    - pruned_model: Model with pruned channels.
    - clean_accuracy: Clean validation accuracy after pruning.
    - pruned_models: List of pruned models saved.
    - good_model: "GoodNet" model combining the backdoored and pruned models.
    """

    cl_x_valid, cl_y_valid = data_loader(clean_valid_data_filename)
    # cl_x_test, cl_y_test = data_loader(clean_test_data_filename)
    # bd_x_valid, bd_y_valid = data_loader(poisoned_valid_data_filename)
    # bd_x_test, bd_y_test = data_loader(poisoned_test_data_filename)

    bd_model = keras.models.load_model(model_filename)

    cl_label_p = np.argmax(bd_model.predict(cl_x_valid), axis=1)
    clean_accuracy_base = np.mean(np.equal(cl_label_p, cl_y_valid))*100
    print('Base clean validation classification accuracy:', clean_accuracy_base)

    layer_to_prune = 'pool_3'
    pruning_layer = 'pruned_pool_3'
    num_channels = bd_model.get_layer(layer_to_prune).output_shape[-1]
    # print(bd_model.get_layer(layer_to_prune).output_shape)

    pool_3_layer_model = Model(inputs=bd_model.input,
                                 outputs=bd_model.get_layer(layer_to_prune).output)
    activations = pool_3_layer_model.predict(cl_x_valid)
    average_activations = np.mean(activations, axis=(0, 1, 2))
    sorted_indices = list(np.argsort(average_activations))
    # sorted_indices = list(np.argsort(average_activations)[::-1])
    print("Sorted indices of the average activations of the channels of pooling layer 3 over the validation set:\n",sorted_indices)


    channel_ids = []
    for step, channel_id in enumerate(sorted_indices):
        # Prune the channels
        channel_ids.append(channel_id)
        print("Pruned channel IDs at step {}:\n".format(step), channel_ids)
        pruned_model = Net_pruned(channel_ids)
        for layer in bd_model.layers:
            if layer.name != pruning_layer and layer.get_weights():
                pruned_model.get_layer(layer.name).set_weights(layer.get_weights())

        cl_label_p = np.argmax(pruned_model.predict(cl_x_valid), axis=1)
        clean_accuracy = np.mean(np.equal(cl_label_p, cl_y_valid))*100
        print('Clean validation classification accuracy on step {}: {}'.format(step, clean_accuracy))

        for i,x in enumerate(X):
            if(clean_accuracy_base-clean_accuracy>=x and len(pruned_models)<i+1):
                pruned_models.append(pruned_model)
                print("Reached at least {}% drop on clean validation accuracy, Saving models...".format(x))
                pruned_model.save('./models/PrunedNet_X-{}.keras'.format(x))
                good_model = Good_Net(bd_model, pruned_model)
                good_model.save('./models/GoodNet_X-{}.keras'.format(x))
    

if __name__ == '__main__':
    main()
