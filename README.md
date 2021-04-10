# CS6910-A2
## Authors: Dhruvjyoti Bagadthey EE17B156, D Tony Fredrick EE17B154


1. Install the required libraries in your environment using this command:

`
pip install -r requirements.txt
`

2. To train a custom CNN model for image classification on the inaturalist dataset from scratch for Part A of the assignment, use the notebook: **Assignment_2_Part_A.ipynb**.

   a.  In this notebook, to train using the best values for hyperparameters obtained from our use of the wandb sweeps functionality, do not run cells in the section titled "Hyperparameter Search using WandB". Run all the other cells of the notebook to train the model. The final model will be trained on the full training set and evaluation is done on the test set.
   
   b. In order to run the hyperparameter search on your own, run the full notebook.
   
   
3. To train a neural network model for image classification on the inaturalist dataset using a pretrained model for Part B, use the notebook: **Assignment_2_CS6910_Part_B.ipynb**. Follow the instructions in the above point.

Note: Wherever you need to log to wandb, please remember to change the name of the entity and project in the corresponding line of code.

### Link to the project report:

https://wandb.ai/ee17b154tony/dl_assignment_2/reports/CS6910-Assignment-2-Report--Vmlldzo1OTMyMzk

### General Framework:

ALl our notebooks have been created in google colab with a GPU backend. We have used tensorflow and keras for defining, training and testing our model.

### Part A functions:

`
make_generators(train_batch_size, data_aug)
`
: This function makes the image data generators depending on the batch_size and data_augmentation. if data_augmentation = True then data augmentation is applied on the data.


### Part B functions:

`
make_generators(train_batch_size, data_aug)
`
: This function makes the image data generators depending on the batch_size and data_augmentation. if data_augmentation = True then data augmentation is applied on the data.

`
define_model(pretrained_model_name, activation_function_dense, fc_layer, dropout, pre_layer_train=None)
`
: This function loads a pretrained model specified by pretrained_model_name and removes its top dense layer. it then freezes all the layers for training and keeps only the last pre_layer_train trainable. pre_layer_train = None implies that all layers shall be frozen. The function then applends a flatten, a fully convolutional layer(fc_layer), a dropout and a softmax layer having 10 classes. the activation function and the number of neurons of the fc layer is specified by activation_function_dense and fc_layer respectively.

`
train_validate_test_model(train_batch_size, pre_train_model, data_aug, activation_function_dense, fc_layer, dropout, epochs, pre_layer_train=None)
`
: This function trains the inaturalist data by calling the make_generators function implicitly on a model returned by define_model(..) for an epochs number of epochs using an Adam optimizer with the learning rate set to 0.0001. It uses the categorical crossentropy function and uses accuracy as the metric. It plots train and validation loss and accuray vs epochs and prints the test accuracy obtained. It also utilises an early stopping callback using a patience parameter of 10.

`
train_validate_model_wandb()
`
: Trains, validates the model on the data and logs the accuracies and losses into wandb.
