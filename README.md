# CS6910-A2
## Authors: Dhruvjyoti Bagadthey EE17B156, D Tony Fredrick EE17B154


1. Install the required libraries in your environment using this command:

`
pip install -r requirements.txt
`

2. **IMPORTANT: To Check code correctness use the notebooks Assignment_2_Part_A_NO_WANDB.ipynb and Guided_Backpropagation_NO_WANDB.ipynb for part A and Assignment_2_CS6910_Part_B_no_wandb.ipynb ONLY**.

3. To train a custom CNN model for image classification on the inaturalist dataset from scratch for Part A of the assignment, use the notebook: **Assignment_2_Part_A.ipynb**.

   a.  In this notebook, to train using the best values for hyperparameters obtained from our use of the wandb sweeps functionality, do not run cells in the section titled "Hyperparameter Search using WandB". Run all the other cells of the notebook up to just before the section titled "Additional Tasks" to train the model. The final model will be trained on the full training set and evaluation is done on the test set.
   
   b. In order to run the hyperparameter search on your own, run the section "Hyperparameter Search using WandB".
   
   c. The additional tasks such as visualizing the filters in the first layer, the results of guided backpropagation, sample images with predictions from the test set, run the cells in the "Additional Tasks" section.
   
   d. Please note that for guided backpropagation, we have used Tensorflow V1, therefore restart the runtime before running that section.
  
4. To train a neural network model for image classification on the inaturalist dataset using a pretrained model for Part B, use the notebook: **Assignment_2_CS6910_Part_B.ipynb**. Follow the instructions in the above point (a and b).

Note: Wherever you need to log to wandb, please remember to change the name of the entity and project in the corresponding line of code.

### Link to the project report:

https://wandb.ai/ee17b154tony/dl_assignment_2/reports/CS6910-Assignment-2-Report--Vmlldzo1OTMyMzk

### General Framework:

ALl our notebooks have been created in Google Colab with a GPU backend. We have used TensorFlow and Keras for defining, training and testing our model.

### Part A functions:

`
make_generators(train_batch_size, data_aug)
`
: This function makes the image data generators depending on the batch_size and data_augmentation. if data_augmentation = True then data augmentation is applied on the data.

`
define_model(activation_function_conv, activation_function_dense, num_filters, shape_of_filters_conv, shape_of_filters_pool, batch_norm_use, fc_layer, dropout)
`
: Creates the CNN model with 5 convolution units (i.e. convolution-batch normalization-relu activation-max pooling units).

Here is the layout of the model created:

![image](https://user-images.githubusercontent.com/38160688/114278556-08298f00-9a4e-11eb-91ca-ba81daade66b.png)

`
train_validate_model(train_batch_size, data_aug, activation_function_conv, activation_function_dense, num_filters, shape_of_filters_conv, shape_of_filters_pool, batch_norm_use, fc_layer, dropout)
`
: This function is used to train the model on the full training set, after the best set of hyperparameters is known and passed as arguments. It returns the trained model and the history object which has details such as loss and accuracy for every epoch. The role of the hyperparameters passed in to the function as arguments is as follows:
   * _train_batch_size_: Number of training examples to be used in one mini-batch
   * _data_aug_: Takes a boolean value, determines whether data augmentation will be used while training the model.
   * _activation_function_conv_: Activation function for the convolution layers.
   * _activation_function_dense_: Activation function for the dense (fully connected layer).
   * _num_filters_: It is a list, where each element is the number of filters in a layer.
   * _shape_of_filters_conv_: A list of tuples, each tuple is the shape of the filters used in a convolutional layer.
   * _batch_norm_use_: A boolean value, determines whether or not batch normalization is used in the model.
   * _fc_layer_: The number of neurons in the fully connected layer.
   * _dropout_: Amount of dropout (fraction) used in the fully connected layer.

`
train_validate_model_wandb()
`
: This function is used for training and validation to find out the best combination of hyperparameter values with the help of WandB sweeps functionality.

##### Accuracy and Loss plots for the best model:

![image](https://user-images.githubusercontent.com/38160688/114279081-6e171600-9a50-11eb-8737-d262d088d4bd.png)

![image](https://user-images.githubusercontent.com/38160688/114279099-8d15a800-9a50-11eb-88aa-ca0c350f80ba.png)


##### Best model performance:
Training accuracy: 42.60426163673401 %

Test accuracy: 37.59999871253967 %


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
