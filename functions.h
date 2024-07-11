#include <math.h>
#include <stdlib.h> // For srand
#include <stdbool.h>
#include "graphics.h"

#define max_file_name_length 20
#define max_training_images 10000
#define input_size 784
#define n_classes 10
#define max_neurons_per_layer 1000
#define PI 3.14159265

static char main_folder_name[max_file_name_length];

static int input_images[max_training_images][input_size][input_size];
static int input_labels[max_training_images][n_classes];
static int training_images[max_training_images][input_size][input_size];
static int training_labels[max_training_images][n_classes];
static int testing_images[max_training_images][input_size][input_size];
static int testing_labels[max_training_images][n_classes];
static int validation_images[max_training_images][input_size][input_size];
static int validation_labels[max_training_images][n_classes];
static int map_training_images[max_training_images];//map that keeps the ordering of the training images
static int map_input_images[max_training_images]; //map that keeps the ordering of the input images
static float weights[4][max_neurons_per_layer][max_neurons_per_layer]; //weights for each layer
static int weights_dim[4][2]; //dimensions of the weights for each layer
static float biases[4][max_neurons_per_layer]; //biases for each layer
static float activations[4][max_neurons_per_layer];
static float outputs[4][max_neurons_per_layer];
static float dw[4][max_neurons_per_layer][max_neurons_per_layer]; //weight variation for each layer
static float db[4][max_neurons_per_layer]; //bias variation for each layer

static int number_of_inputs;
static int number_of_val_images;
static int number_of_train_images;
static int number_of_test_images;
static int neurons_per_layer[4];//number of neurons per each layer
static int neurons_output_layer=10; //the output layer has 10 neurons, one for each digit
static int neurons_input_layer=input_size*input_size; //the input layer has the same number of neurons as the number of pixels in the image
static int type_of_activation; //0 for sigmoid, 1 for ReLU
static int type_of_loss; //0 for Log-likelihood, 1 for mean squared error
static int type_of_initialization; //0 for random, 1 for gaussian
static int type_of_shuffling; //0 for no shuffling, 1 for shuffling
static float learning_rate; //the learning rate of the model
static int train_val_split; 
static int minibatch_size;
static int number_of_epochs;
static float threshold_error;
static float error_on_batch;
static float error_on_epoch;
float loss_on_example(int t_loss);

void permute(int *x, int dim);
void set_number_of_inputs(int n_inputs, int n_test_images);
void split_data();
float rand_normal(float mu, float sigma);
void define_network_structure(int npl1, int npl2, int npl3, int activation, 
int initialization);
void define_training_parameters(int n_epochs,float lr, int loss, int shuffling, float error);
float ReLU(float x);
float sigmoid(float x);
float softmax(float *x, int dim);
float ReLU_derivative(float x);
float sigmoid_derivative(float x);
void gaussian_layer_initialization(int n_layer);
void weight_initialization();
void layer_output(int *input, int layer_index, int activ_function);
void neuron_output(int neuron_index, int layer_index, int *input, int activ_function);
void forward_propagation(int input[input_size][input_size]);
int *linearize(int x[input_size][input_size]);
void learn_example(int input[input_size][input_size]);
void reset_dw_db();
void average_dw_db(int M);
void update_w_b();
void learn_batch(int batch_index);
void learn_epoch();
float log_likelihood(int *t, float *y, int dim);
float mean_squared_error(int *t, float *y, int dim);
void train_network();
float *get_probabilities(int input[input_size][input_size]);
int get_best_class(int input[n_classes]);
float accuracy(int *input_examples[input_size][input_size],int *input_labels[n_classes], int dim);

void set_folder_name(char *folder)
{
    //function to set the name of the base folder as a global variable
    int i;
    for (i=0; i<max_file_name_length; i++)
        main_folder_name[i] = folder[i];
}

void load_training_set()
{
    //load the data i use for training and validation
}

void load_test_set()
{
    //load the data i use for testing
}

void set_train_val(int n_minibatch, int n_train_val_split)
{
    minibatch_size = n_minibatch;
    train_val_split = n_train_val_split;
}

void set_number_of_inputs(int n_inputs, int n_test_images)
{
    number_of_test_images = n_test_images;
    number_of_inputs = n_inputs;
    //i initialize the map for the ordering of input images
    for (int i=0; i<number_of_inputs; i++)
    {
        map_input_images[i] = i;
    }
}

void define_network_structure(int npl1, int npl2, int npl3, int activation, int initialization)
{
    //neurons_per_layer[0] = neurons_input_layer;
    neurons_per_layer[0] = npl1;
    neurons_per_layer[1] = npl2;
    neurons_per_layer[2] = npl3;
    neurons_per_layer[3] = neurons_output_layer;
    type_of_activation = activation;
    type_of_initialization = initialization;
    //determine the weight matrices dimensions
    for (int i=0; i<4; i++)
    {
        weights_dim[i][0] = neurons_per_layer[i];
        if(i==0)
            weights_dim[i][1] = neurons_input_layer;
        else
            weights_dim[i][1] = neurons_per_layer[i-1];
    }
}

void define_training_parameters(int n_epochs,float lr, int loss, int shuffling, float error)
{
    type_of_loss = loss;
    type_of_shuffling = shuffling;
    learning_rate=lr;
    number_of_epochs = n_epochs;
    threshold_error=error;
}

void permute(int *x, int dim){
    //This function takes in input an array of dimension d and permutes its elements
    //I will use this to permute the elements of the maps that associate indexes to training images

    //I keep a map of the indexes that were already extracted
    //If an index is extracted a second time i select the closes non selected index
    int temp[dim];
    int is_extracted[dim];
    int new_index[dim];
    //I initialize the map to zero -> no index is extracted yt
    for (int i=0; i<dim; i++)
    {
        is_extracted[i] = 0;
        temp[i]=x[i];//i create a copy of x to avoid modifying the original array while reordering
    }
    //I generate the new_index array that define the new ordering of values
    for (int i=0; i<dim; i++)
    {   
        int index = rand()%dim;
        bool found = false;
        do{
            if(is_extracted[index]==0)
            {
                new_index[i] = index;
                is_extracted[index] = 1;
                found=true;
            }
            else
                index = (index+1)%dim;
        }while(!found);
    }
    for(int i=0; i<dim; i++)
    {
        temp[i] = x[new_index[i]];
    }
    for(int i=0; i<dim; i++)
    {
        x[i] = temp[i];
    }
}

/*void return_ith_batch(int i)
{
    //make the batches
    permute(map_training_images, number_of_train_images);

}*/

void split_data()
{
    //split the data in training and validation
    //I assume number of validation images < number of training images
    number_of_train_images = train_val_split*number_of_inputs;
    number_of_val_images = number_of_inputs - number_of_train_images;
    permute(map_input_images, number_of_inputs);
    for(int i = 0; i < number_of_train_images; i++) {
        for(int j = 0; j < input_size; j++) {
            for(int k = 0; k < input_size; k++) {
                if (i<number_of_val_images)
                    validation_images[i][j][k] = input_images[map_input_images[number_of_train_images+i]][j][k];
                training_images[i][j][k] = input_images[map_input_images[i]][j][k];
            }
        }
    }
}

float rand_normal(float mu, float sigma)
{
    float x, y, r; // generated by Box-Muller algorithm
    x = (float)rand()/RAND_MAX;
    y = (float)rand()/RAND_MAX;
    r = cos(2*PI*y)*sqrt(-2.*log(x));
    return mu + sigma*r;
}

void gaussian_layer_initialization(int n_layer)
{
    int dim1 = weights_dim[n_layer][0];
    int dim2 = weights_dim[n_layer][1];
    int n_weights=dim1*dim2;
    int n_neurons=neurons_per_layer[n_layer];//cause the input neurons don't do computations
    //initialize the weights of the layer with a gaussian distribution
    for (int i=0; i<dim1; i++)
    {
        for (int j=0; j<dim2; j++)
        {
            weights[n_layer][i][j] = rand_normal(0,1.0/n_weights);
        }
    }
    for (int i=0; i<n_neurons; i++) 
    {
        biases[n_layer][i] = rand_normal(0,1.0/n_weights);
    }
}

void weight_initialization()
{
    //initialize the weights
    switch (type_of_initialization)
    {
    case 1:
        for (int i=0; i<4; i++)
        {
            gaussian_layer_initialization(i);
        }
        break;
    default: //default is gaussian initialization
        for (int i=0; i<4; i++)
        {
            gaussian_layer_initialization(i);
        }
        break;
    }
}


//----------------------
//activation functions
float sigmoid(float x)
{
    return 1/(1+exp(-x));
}

float ReLU(float x)
{
    if (x>0)
        return x;
    else
        return 0;
}

float softmax(float *x, int dim)
{
    //softmax function
    float sum = 0;
    for (int i=0; i<dim; i++)
    {
        sum += exp(x[i]);
    }
    for (int i=0; i<dim; i++)
    {
        x[i] = exp(x[i])/sum;
    }
}
//--------------------------

//------------------------
//derivatives of activation functions
float sigmoid_derivative(float x)//check
{
    return sigmoid(x)*(1-sigmoid(x));
}
float ReLU_derivative(float x)
{
    if (x>0)
        return 1;
    else
        return 0;
}
//------------------------

int *linearize(int x[input_size][input_size])
{
    //takes an image and compress it to one dimension
    static int y[input_size*input_size];
    for (int i=0; i<input_size; i++)
    {
        for (int j=0; j<input_size; j++)
        {
            y[i*input_size+j] = x[i][j];
        }
    }
    return y;
}


//-----------------------
//inference
void neuron_output(int neuron_index, int layer_index, int *input, int activ_function)
{
    //gives the output of a single neuron
    int input_dimension=weights_dim[layer_index][0];
    activations[layer_index][neuron_index]=0;
    for (int i=0; i<input_dimension; i++)
    {
        activations[layer_index][neuron_index] += weights[layer_index][neuron_index][i]*input[i];
    }
    activations[layer_index][neuron_index] += biases[layer_index][neuron_index];
    switch (activ_function)
    {
    case 0://sigmoid
        outputs[layer_index][neuron_index] = sigmoid(activations[layer_index][neuron_index]);
        break;
    case 1://ReLU
        outputs[layer_index][neuron_index] = ReLU(activations[layer_index][neuron_index]);
        break;
    case 2://no activation
        outputs[layer_index][neuron_index] = activations[layer_index][neuron_index];
        break;
    default: //sigmoid
        outputs[layer_index][neuron_index] = sigmoid(activations[layer_index][neuron_index]);
        break;
    }
}

void layer_output(int *input, int layer_index, int activ_function)
{
    //compute the output of a single layer
    int n_neurons = neurons_per_layer[layer_index];
    for (int i=0; i<n_neurons; i++)
    {
        neuron_output(i, layer_index, input, activ_function);
    }
}

//i use this function only for inference. During training is called by learn_example
void forward_propagation(int input[input_size][input_size])
{
    int *input_linear = linearize(input);
    for (int i=0;i<4;i++)
    {
        if (i==0){
            layer_output(input_linear, i, type_of_activation);
        }
        else if(i==3)
        {
            layer_output(outputs[i-1], i, 2);
            softmax(outputs[i], neurons_output_layer);
            //the output of the final layer is passed through a softmax not a sigmoid
        }
        else
        {
            layer_output(outputs[i-1], i, type_of_activation);
            //the input for the ith layer is the output of the previous
        }
    }
    //apply softmax to the output layer and save results in softmax layer
}
//-----------------------

//-----------------------
//backpropagation

//Compute dw and db and add to global arrays
//If you want to update the global for the single example recall to reset the weights
void learn_example(int input[input_size][input_size])
{
    forward_propagation(input);
    error_on_batch +=loss_on_example(type_of_loss);
    int *input_linear = linearize(input);
    float deltas[4][max_neurons_per_layer];
    for (int l=3; l>=0; l--)//i start from the last layer 
    {
        for(int i=0; i<neurons_per_layer[l]; i++)
        {
            if (l==3){
                switch (type_of_loss)
                {
                case 0: //log likelihood
                    deltas[l][i] = training_labels[l][i]-outputs[l][i];
                    break;
                default: // log likelihood
                    deltas[l][i] = training_labels[l][i]-outputs[l][i];
                    break;
                }
            }
            else{
                //backpropagation step
                float weighted_sum = 0;
                for (int k=0; k<neurons_per_layer[l+1]; k++)
                {
                    weighted_sum += weights[l+1][i][k]*deltas[l+1][k];
                }
                float activ_derivative;
                switch (type_of_loss)
                {
                case 0: //log likelihood
                    activ_derivative = sigmoid_derivative(activations[l][i]);
                    break;
                default: // log likelihood
                    activ_derivative = sigmoid_derivative(activations[l][i]);
                    break;
                }
                deltas[l][i] = activ_derivative*weighted_sum;
            }
            if(l==0){
                for (int j=0; j<neurons_input_layer; j++)
                    dw[l][i][j] += learning_rate*deltas[l][i]*input_linear[j];
            }
            else{
                for (int j=0; j<neurons_per_layer[l-1]; j++)
                {
                    //compute the dw for the layer
                    dw[l][i][j] += learning_rate*deltas[l][i]*outputs[l][j];
                }
            }
            db[l][i] += learning_rate*deltas[l][i];
            //for the first layer the outputs are the linearized inputs
        }
    }
}

void reset_dw_db()
{   
    for (int l=0; l<4; l++)
    {
        int dim1 = weights_dim[l][0];
        int dim2 = weights_dim[l][1];
        int n_neurons=neurons_per_layer[l];//cause the input neurons don't do computations
        //initialize the weights of the layer with a gaussian distribution
        for (int i=0; i<dim1; i++)
            for (int j=0; j<dim2; j++) dw[l][i][j] = 0;
        for (int i=0; i<n_neurons; i++) db[l][i] = 0;
    }
}

void average_dw_db(int M)
{   
    //M is batch size if you are averaging on a mini-batch
    for (int l=0; l<4; l++)
    {
        int dim1 = weights_dim[l][0];
        int dim2 = weights_dim[l][1];
        int n_neurons=neurons_per_layer[l];//cause the input neurons don't do computations
        //initialize the weights of the layer with a gaussian distribution
        for (int i=0; i<dim1; i++)
            for (int j=0; j<dim2; j++) dw[l][i][j] /= M;
        for (int i=0; i<n_neurons; i++) db[l][i] /= M;
    }
}

void update_w_b(){
    for (int l=0; l<4; l++)
    {
        int dim1 = weights_dim[l][0];
        int dim2 = weights_dim[l][1];
        int n_neurons=neurons_per_layer[l];//cause the input neurons don't do computations
        //initialize the weights of the layer with a gaussian distribution
        for (int i=0; i<dim1; i++)
            for (int j=0; j<dim2; j++) weights[l][i][j] += dw[l][i][j];
        for (int i=0; i<n_neurons; i++) biases[l][i] += db[l][i];
    }
}

void learn_batch(int batch_index)
{
    reset_dw_db();
    error_on_batch=0;
    int start = batch_index*minibatch_size;
    int end = (batch_index+1)*minibatch_size;
    for (int i=start; i<end; i++)
    {
        int ind=i%number_of_train_images;//so that if the training dimension is not exactly 
        //a multiple of the batch size i have no error (i start taking again from the i=0)
        learn_example(training_images[map_training_images[i]]);
        // i didn't sort the images in training batches, i've only sorted the map
    }
    average_dw_db(minibatch_size);
    update_w_b();
    error_on_batch/=minibatch_size; //i sum all the errors in learn_example, i need to average
    //them at the end of the batch
}

void learn_epoch()
{
    error_on_epoch=0;
    int n_batches=number_of_train_images/minibatch_size;//in this way 2.8=2 -> i loose last batch
    //int n_batches=ceil(number_of_train_images/minibatch_size); //in this way i round the value
    //int n_batches=number_of_train_images/minibatch_size+1;// i always round up (but i take one more if divisible)
    for (int i=0;i<n_batches;i++){
        learn_batch(i);
        error_on_epoch+=error_on_batch;
    }
    error_on_epoch/=n_batches;
}

void train_network()
{
    //train the network
    //learn for a defined number of epochs or to the point at which error is very small
    int current_epoch=0;
    float current_error=0;
    do{
        learn_epoch();
        current_epoch++;
        current_error=error_on_epoch;
    }while(current_epoch<number_of_epochs && current_error>threshold_error);
}
//----------------------

//----------------------
//loss functions
float log_likelihood(int *t, float *y, int dim)
{
    float C=0;
    for (int i=0;i<dim;i++){
        C -= t[i]*log(y[i]); 
    }
    return C;
}

float mean_squared_error(int *t, float *y, int dim)
{
    float mse=0;
    for (int i=0;i<dim;i++){
        mse += 0.5 * (t[i]-y[i]) * (t[i]-y[i]); 
    }
    return mse;
}
//-----------------------


//---------------------------
//metrics
//this has to be applied after the forward pass function
float loss_on_example(int t_loss)
{
    //takes in input the type of loss function
    switch (t_loss){
        case 0: //log likelihood
            return log_likelihood(training_labels,outputs[3],neurons_output_layer);
            break;
        case 1: //mse
            return mean_squared_error(training_labels,outputs[3],neurons_output_layer);
            break;
        default: // log likelihood
            return log_likelihood(training_labels,outputs[3],neurons_output_layer);
            break;
    }
}

float accuracy(int *input_examples[input_size][input_size],int *input_labels[n_classes], int dim)
{
    int successes=0;
    int total=dim;
    for (int i=0;i<dim;i++){
        forward_propagation(input_examples[i]);
        if (get_best_class(outputs[3])==get_best_class(input_labels)){
            successes+=1;
        }
    }
    float accuracy=successes/total;
    return accuracy;
}
/*
void confusion_matrices(){
}

float precision()
{

}

float recall()
{

}*/

//----------------------------

void test_network()
{
    //test the network
}

void validate_network()
{
    //validate the network
}

void save_model()
{
    //save the model
}

//------------------------------------
//retrieval functions
//apply to outputs after a forward propagation
int get_best_class(int input[n_classes])
{
    int max = input[0]; // Assume first output is the max initially
    int ind=0; //index of max element
    for (int i = 1; i < n_classes; i++) {
        if (input[i] > max) {
            max = input[i]; // Update max if current element is greater
            ind=i;
        }
    }
    return ind;
}

float *get_probabilities(int input[input_size][input_size])
{
    forward_propagation(input);
    return outputs[3];
}
//-------------------------------------