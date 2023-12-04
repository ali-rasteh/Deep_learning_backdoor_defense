```bash
├── data 
    └── cl
        └── valid.h5 // this is clean validation data used to design the defense
        └── test.h5  // this is clean test data used to evaluate the BadNet
    └── bd
        └── bd_valid.h5 // this is sunglasses poisoned validation data
        └── bd_test.h5  // this is sunglasses poisoned test data
├── models
    └── bd_net.h5
    └── bd_weights.h5
    └── PrunedNet_X-2.keras
    └── PrunedNet_X-4.keras
    └── PrunedNet_X-10.keras
    └── GoodNet_X-2.keras
    └── GoodNet_X-4.keras
    └── GoodNet_X-10.keras
├── architecture.py     // This is the architecture script consisting all the models and customized layers.
└── repair.py   // This is the script for pruning the given badnet using the clean validation data.
└── eval.py // this is the evaluation script.
```

## I. Dependencies
   1. Python 3.6.9
   2. Keras 2.3.1
   3. Numpy 1.16.3
   4. Matplotlib 2.2.2
   5. H5py 2.9.0
   6. TensorFlow-gpu 1.15.2

## II. Data
   1. Download the validation and test datasets from [here](https://drive.google.com/drive/folders/1Rs68uH8Xqa4j6UxG53wzD0uyI8347dSq?usp=sharing) and store them under `data/` directory.
   2. The clean validation data is used to repair the network and create the pruned and finally good networks.
   3. The clean and poisoned test data are used to evaluate the performance of the repaired models.

## III. Models
   1. bd_net.h5 is the given badnet that needs to be pruned.
   2. PrunedNet_X-2.keras, PrunedNet_X-4.keras, and PrunedNet_X-10.keras are the pruned networks that some of the channels in pool_3 layer is removed for them until the accuracy on the validation data dropped 2%, 4%, and 10% respectively.
   3. GoodNet_X-2.keras, GoodNet_X-4.keras, and GoodNet_X-10.keras are the created Goodnets (repaired badnets) that output the same class with the badnet and pruned networks if their output is same and output another added class (N+1) in the case that their output differ.

## IV. Evaluating the Backdoored Model
   1. For running the prunning on the given badnet run `repair.py` by:
      `python3 repair.py <clean validation data path> <badnet model path>`.
      E.g., `python3 repair.py data/cl/valid.h5 models/bd_net.h5`
   2. To evaluate the backdoored model, execute `eval.py` by running:  
      `python3 eval.py <clean test data path> <poisoned test data path> <model path>`.
      E.g., `python3 eval.py data/cl/test.h5 data/bd/bd_test.h5 models/GoodNet_X-2.keras`. This will output:
      Clean Classification accuracy: 95.74 %
      Attack Success Rate: 100.0 %

