# phys139-project
This repository contians the source code for Group F's PHYS 139 project: Recurrent Neural Network Decoder for Quantum Error
Correction in Surface Code. Large portions of the code are based on the work from an earlier project by Paul Baireuther et al. You can find their paper and both their NN and simulator code in resources.txt.

simple_model.ipynb contains the bulk of our work on this project. It is a TensorFlow keras model based on the architecture and parameters stated in the source paper, and with parts (such as the statistics calculation code) adapted from code from Baireuther's respository. However, the main network is entirely rewritten with the modern TensorFlow keras library rather than TensorFlow v1. It is also largely simplified for the scale of our project.

Relu_Model.ipynb contains only an alternate version of the model used for testing purposes.

circuit_model.py is the unmodified simulator script taken from https://github.com/baireuther/circuit_model, which is used to generate our dataset. The dataset is stored as database files in the db/ directory.

database_io.py contains an interface to read from the databases and easily generate batches for use in our neural networks. Most of the code is adapted from the original source code by Baireuther. simple_model.ipynb uses the class from this script.

plotting.py contains several utility functions for plotting only.

The unmodified source model from the paper can be found in the source-model/ directory. Since it uses libraries exclusive to an older version of TensorFlow, it is recommended to set up a virtual environment to run it if desired.
