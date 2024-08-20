//#include "graphics.h"
#include <math.h>
#include <stdlib.h> // For srand
#include <stdbool.h>
#include <stddef.h> // For NULL
#include <stdio.h>
#include <time.h>   // For time
#include <string.h>

#define max_file_name_length 20
#define max_training_images 60000
#define max_testing_images 10000
#define input_size 28
#define n_classes 10
#define max_neurons_per_layer 1000
#define PI 3.14159265
#define max_hidden_layers 3

static char main_folder_name[max_file_name_length];

static int input_images[max_training_images][input_size][input_size];//these will store the whole training set
static int input_labels[max_training_images][n_classes];
static int training_images[max_training_images][input_size][input_size];
static int training_labels[max_training_images][n_classes];
static int testing_images[max_testing_images][input_size][input_size];//this will store the whole testing set
static int testing_labels[max_testing_images][n_classes];
static int validation_images[max_training_images][input_size][input_size];
static int validation_labels[max_training_images][n_classes];
static int map_training_images[max_training_images];//map that keeps the ordering of the training images
static int map_input_images[max_training_images]; //map that keeps the ordering of the input images
static float weights[max_hidden_layers+1][max_neurons_per_layer][max_neurons_per_layer]; //weights for each layer
static int weights_dim[max_hidden_layers+1][2]; //dimensions of the weights for each layer
static float biases[max_hidden_layers+1][max_neurons_per_layer]; //biases for each layer
static float activations[max_hidden_layers+1][max_neurons_per_layer];
static float outputs[max_hidden_layers+1][max_neurons_per_layer];
static float dw[max_hidden_layers+1][max_neurons_per_layer][max_neurons_per_layer]; //weight variation for each layer
static float db[max_hidden_layers+1][max_neurons_per_layer]; //bias variation for each layer
static float w_momentum[max_hidden_layers+1][max_neurons_per_layer][max_neurons_per_layer]; //weight variation for each layer
static float b_momentum[max_hidden_layers+1][max_neurons_per_layer]; //bias variation for each layer


static int number_of_layers;
static int number_of_inputs;
static int number_of_val_images;
static int number_of_train_images;
static int number_of_test_images;
static int neurons_per_layer[max_hidden_layers+1];//number of neurons per each layer
static int neurons_output_layer=10; //the output layer has 10 neurons, one for each digit
static int neurons_input_layer=input_size*input_size; //the input layer has the same number of neurons as the number of pixels in the image
static int type_of_activation; //0 for sigmoid, 1 for ReLU
static int type_of_loss; //0 for Log-likelihood, 1 for mean squared error
static int type_of_initialization; //0 for random, 1 for gaussian
static int type_of_shuffling; //0 for no shuffling, 1 for shuffling
static float learning_rate; //the learning rate of the model
static float momentum;
static int type_of_optimization=0; //0 for SGD, 1 for momentum, 2 for nesterov
static float train_val_split; 
static int minibatch_size;
static int number_of_epochs;
static float threshold_error;
static float error_on_batch;
static float error_on_epoch;
static float error_on_validation;
typedef struct {
    //in confusion matrices i have gold as columns and predicted as rows   
    //in micro confusion matrices i have 1 as first and 0 as second
    int full_confusion_matrix[10][10];
    int micro_confusion_matrices[10][2][2];
    int micro_confusion_matrix[2][2];
    float overall_accuracy;
    float macro_precision;
    float macro_recall;
    float micro_recall;
    float micro_precision;
    float accuracies[10];
    float precisions[10];
    float recalls[10];
} Metrics;

float *int_to_float(float *y,int *x, int dim);
void printIntegerByteByByte(int number);
void read_magic_number(FILE *fptr);
void read_dimensions(FILE *fptr,int n_dim);
int flip_reading_order(int number);
void load_test_set();
void load_training_set();
void read_labels(int *labels, FILE *fptr, int dim);
void read_images(int (*images)[input_size][input_size], FILE *fptr, int dim);
void print_image(int image[input_size][input_size]);
void permute(int *x, int dim);
void set_number_of_inputs(int n_inputs, int n_test_images);
void split_data();
float rand_normal(float mu, float sigma);
void define_network_structure(int *npl,int n_layers, int activation, int initialization);//should move initialization to other function
void define_training_parameters(int n_epochs,float lr, int loss, int shuffling, float error, int opt, float mom);
float ReLU(float x);
float sigmoid(float x);
float softmax(float *x, int dim);
float softmax_stable(float *x, int dim);
float ReLU_derivative(float x);
float sigmoid_derivative(float x);
void gaussian_layer_initialization(int n_layer);
void weight_initialization();
void layer_output(float *input, int layer_index, int activ_function);
void neuron_output(int neuron_index, int layer_index, float *input, int activ_function);
void forward_propagation(int input[input_size][input_size]);
float *lin_and_norm(int x[input_size][input_size]);
float *lin_and_bin(int x[input_size][input_size]);
void learn_example(int input_index);
void reset_dw_db();
void reset_momentum();
void update_dw_db_prev();
void average_dw_db(int M);
void update_w_b();
void learn_batch(int batch_index);
void learn_epoch();
float log_likelihood(int *t, float *y, int dim);
float mean_squared_error(int *t, float *y, int dim);
void train_network();
float *get_probabilities(int input[input_size][input_size]);
int get_best_class(float input[n_classes]);
float loss_on_example(int *label,float *probabilities,int t_loss);
void testing_layer_initialization(int n_layer);
float optimizer_w(int l,int i, int j);
float optimizer_b(int l, int i);
void load_model(char *filename);
void save_NN(char *filename);
float loss_on_set(int (*labels)[n_classes],float (*probabilities)[n_classes], int dim, int t_loss);

float *int_to_float(float *y,int *x, int dim){
    for (int i=0; i<dim; i++){
        y[i] = (float)x[i];
    }
    return y;
}

void set_folder_name(char *folder)
{
    //function to set the name of the base folder as a global variable
    int i;
    for (i=0; i<max_file_name_length; i++)
        main_folder_name[i] = folder[i];
}

void printIntegerByteByByte(int number) {
    int size = sizeof(int);
    for (int i = 0; i < size; i++) {
        // Extract the lowest byte
        unsigned char byte = (number >> (i * 8)) & 0xFF; //1byte = 8bits = 2^8 = 256 FF=255
        printf("Byte %d: %02x\n", i, byte);
    }
}

void read_magic_number(FILE *fptr) {
    int magic_number;
    fread(&magic_number, sizeof(int), 1, fptr);
    //printIntegerByteByByte(flip_reading_order(magic_number));
}// 3rd byte is 0x08 -> unsigned byte according to https://github.com/cvdfoundation/mnist 

void read_dimensions(FILE *fptr,int n_dim) {
    int dim_sizes[n_dim];
    fread(dim_sizes, sizeof(int), n_dim, fptr);
    /*for (int i = 0; i < n_dim; i++) {
        printf("Dimension %d: %d\n", i, flip_reading_order(dim_sizes[i]));
    }*/
}

int flip_reading_order(int number){
    int size = sizeof(int);
    int flipped=0;
    unsigned char bytes[4];
    unsigned char flipped_bytes[4];
    for (int i = 0; i < size; i++) {
        bytes[i] = (number >> (i * 8)) & 0xFF; 
    }
    for (int i = 0; i < size; i++) {
        flipped_bytes[i] = bytes[size-i-1]; 
    }
    for (int i = 0; i < size; i++) {
        flipped += flipped_bytes[i] << (i * 8);
    }
    return flipped;
}

void read_labels(int *labels, FILE *fptr, int dim) {
    read_magic_number(fptr);
    read_dimensions(fptr,1);
    for (int i = 0; i < dim; i++) {
        unsigned char c;
        fread(&c, sizeof(unsigned char), 1, fptr);//labels are unsigned byte 
        labels[i] = c;
        //printf("%d \n",labels[i]);
    }
}

void read_images(int (*images)[input_size][input_size], FILE *fptr, int dim) {
    read_magic_number(fptr);
    read_dimensions(fptr,3);//number of elements, width and length
    for (int k=0;k<dim;k++){
        for (int i = 0; i <input_size; i++) {
            for (int j=0;j<input_size;j++){
                unsigned char c;
                fread(&c, sizeof(unsigned char), 1, fptr);
                images[k][i][j] = c;
            }
        }
    }
}

void print_image(int image[input_size][input_size]){
    for (int i = 0; i <input_size; i++) {
        for (int j=0;j<input_size;j++){
            if(image[i][j]>0){
                //printf("x");
                printf("%d ",image[i][j]);
            }
            else{
                printf("   ");
            }
        }
        printf("\n");
    }
}

void set_train_val(int n_minibatch, float n_train_val_split)
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

void define_network_structure(int *npl, int n_hidden_layers, int activation, int initialization)
{
    number_of_layers = n_hidden_layers+1;
    //neurons_per_layer[0] = neurons_input_layer;
    for (int i=0; i<n_hidden_layers; i++)
    {
        neurons_per_layer[i] = npl[i];
    }
    neurons_per_layer[n_hidden_layers] = neurons_output_layer;
    type_of_activation = activation;
    type_of_initialization = initialization;
    //determine the weight matrices dimensions
    for (int i=0; i<number_of_layers; i++)
    {
        weights_dim[i][0] = neurons_per_layer[i];
        if(i==0)
            weights_dim[i][1] = neurons_input_layer;
        else
            weights_dim[i][1] = neurons_per_layer[i-1];
    }
    reset_momentum(); //initialize the previous weights and biases to zero
}

void define_training_parameters(int n_epochs,float lr, int loss, int shuffling, float error, int opt, float mom)
{
    type_of_loss = loss;
    type_of_shuffling = shuffling;
    learning_rate=lr;
    number_of_epochs = n_epochs;
    threshold_error=error;
    type_of_optimization = opt;
    momentum=mom;
}

void load_training_set()
{
    //load the data i use for training and validation
    //load the data i use for testing
    FILE *fptr;
    /*char temporary[100];
    strcpy(temporary,main_folder_name);
    fptr = fopen(strcat(temporary,"/train-labels-idx1-ubyte"),"rb");*/
    fptr = fopen("input_folder/train-labels-idx1-ubyte","rb");
    if(fptr == NULL)
    {
        printf("Error opening file train labels!");   
        exit(1);             
    }
    int labels[number_of_inputs];
    read_labels(labels,fptr,number_of_inputs);
    fclose(fptr);
    //i put the labels in the 10 dimnesional form
    for (int i=0; i<number_of_inputs; i++)
    {
        for (int j=0; j<n_classes; j++)
        {
            if (j==labels[i])
                input_labels[i][j] = 1;
            else
                input_labels[i][j] = 0;
        }
    }
    FILE *fptr2;
    /*char temporary_2[100];
    strcpy(temporary_2,main_folder_name);
    fptr2 = fopen(strcat(temporary_2,"/train-images-idx3-ubyte"),"rb");*/
    fptr2 = fopen("input_folder/train-images-idx3-ubyte","rb");
    if(fptr2 == NULL)
    {
        printf("Error!");   
        exit(1);             
    }
    read_images(input_images,fptr2,number_of_inputs);
    fclose(fptr2);
}

void load_test_set()
{
    //load the data i use for testing
    FILE *fptr;
    /*char temporary[100];
    strcpy(temporary,main_folder_name);
    fptr = fopen(strcat(temporary,"/t10k-labels-idx1-ubyte"),"rb");*/
    fptr = fopen("input_folder/t10k-labels-idx1-ubyte","rb");
    if(fptr == NULL)
    {
        printf("Error opening file!");   
        exit(1);             
    }
    int labels[number_of_test_images];
    read_labels(labels,fptr,number_of_test_images);
    fclose(fptr);
    //i put the labels in the 10 dimnesional form
    for (int i=0; i<number_of_test_images; i++)
    {
        for (int j=0; j<n_classes; j++)
        {
            if (j==labels[i])
                testing_labels[i][j] = 1;
            else
                testing_labels[i][j] = 0;
        }
    }
    FILE *fptr2;
    /*char temporary_2[100];
    strcpy(temporary_2,main_folder_name);
    fptr2 = fopen(strcat(temporary_2,"/t10k-images-idx3-ubyte"),"rb");*/
    fptr2 = fopen("input_folder/t10k-images-idx3-ubyte","rb");
    if(fptr2 == NULL)
    {
        printf("Error!");   
        exit(1);             
    }
    read_images(testing_images,fptr2,number_of_test_images);
    fclose(fptr2);
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
    number_of_train_images = (1-train_val_split)*number_of_inputs;
    number_of_val_images = number_of_inputs - number_of_train_images;
    permute(map_input_images, number_of_inputs);
    for(int i = 0; i < number_of_train_images; i++) {
        map_training_images[i] = i;
        for(int j = 0; j < input_size; j++) {
            for(int k = 0; k < input_size; k++) {
                if (i<number_of_val_images)
                    validation_images[i][j][k] = input_images[map_input_images[number_of_train_images+i]][j][k];
                training_images[i][j][k] = input_images[map_input_images[i]][j][k];
            }
        }
        for (int j=0; j<n_classes; j++)
        {
            if (i<number_of_val_images)
                validation_labels[i][j] = input_labels[map_input_images[number_of_train_images+i]][j];
            training_labels[i][j] = input_labels[map_input_images[i]][j];
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
            weights[n_layer][i][j] = rand_normal(0,sqrt(1.0/weights_dim[n_layer][1]));
        }
    }
    for (int i=0; i<n_neurons; i++) 
    {
        biases[n_layer][i] = rand_normal(0,sqrt(1.0/weights_dim[n_layer][1]));
    }
}

void testing_layer_initialization(int n_layer){
    int dim1 = weights_dim[n_layer][0];
    int dim2 = weights_dim[n_layer][1];
    int n_weights=dim1*dim2;
    int n_neurons=neurons_per_layer[n_layer];//cause the input neurons don't do computations
    //initialize the weights of the layer with a gaussian distribution
    for (int i=0; i<dim1; i++)
    {
        for (int j=0; j<dim2; j++)
        {
            weights[n_layer][i][j] = (i*dim2+j+1)/10.0;
        }
    }
    for (int i=0; i<n_neurons; i++) 
    {
        biases[n_layer][i] = (i+1)/10.0;
    }
}

void weight_initialization()
{
    //initialize the weights
    //printf("%d \n", type_of_initialization);
    switch (type_of_initialization)
    {
    case 1:
        for (int i=0; i<number_of_layers; i++)
        {
            gaussian_layer_initialization(i);
        }
        break;
    case 5://testing initialization
        for (int i=0; i<number_of_layers; i++)
        {
            testing_layer_initialization(i);
        }
        break;
    default: //default is gaussian initialization
        for (int i=0; i<number_of_layers; i++)
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

float softmax_stable(float *x, int dim)
{
    int i;
    float sum, max;
    //I find the maximum
    for (i = 1, max = x[0]; i < dim; i++) {
        if (x[i] > max) {
            max = x[i];
        }
    }

    //I can subtract a constant exponent to each exponential term
    //cause i can collect it both at the numerator and denominator when i compute the output
    for (i = 0, sum = 0; i < dim; i++) {
        x[i] = exp(x[i] - max);
        sum += x[i];
    }

    for (i = 0; i < dim; i++) {
        x[i] /= sum;
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

float *lin_and_norm(int x[input_size][input_size])
{
    //takes an image and compress it to one dimension
    //also convert to float since the layers take float arrays
    static float y[input_size*input_size];
    for (int i=0; i<input_size; i++)
    {
        for (int j=0; j<input_size; j++)
        {
            y[i*input_size+j] = (float)x[i][j] / 255.0;
        }
    }
    return y;
}

float *lin_and_bin(int x[input_size][input_size]){
    //takes an image and compress it to one dimension
    //also encodes all values above a threshold to 1
    float threshold = 10;
    static float y[input_size*input_size];
    for (int i=0; i<input_size; i++)
    {
        for (int j=0; j<input_size; j++)
        {
            if (x[i][j]>threshold)
                y[i*input_size+j] = 1;
            else
                y[i*input_size+j] = 0;
        }
    }
    return y;
}
//-----------------------
//inference
void neuron_output(int neuron_index, int layer_index, float *input, int activ_function)
{
    //gives the output of a single neuron
    int input_dimension=weights_dim[layer_index][1];
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

void layer_output(float *input, int layer_index, int activ_function)
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
    float *input_linear = lin_and_norm(input);
    for (int i=0;i<number_of_layers;i++)
    {
        if (i==0){
            layer_output(input_linear, i, type_of_activation);
        }
        else if(i==number_of_layers-1)
        {
            layer_output(outputs[i-1], i, 2);
            softmax_stable(outputs[i], neurons_output_layer);
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
void learn_example(int index_of_example)
{
    forward_propagation(training_images[index_of_example]);
    error_on_batch += loss_on_example(training_labels[index_of_example],outputs[number_of_layers-1],type_of_loss);
    float *input_linear = lin_and_norm(training_images[index_of_example]);
    float deltas[number_of_layers][max_neurons_per_layer];
    for (int l=number_of_layers-1; l>=0; l--)//i start from the last layer 
    {
        for(int i=0; i<neurons_per_layer[l]; i++)
        {
            if (l==number_of_layers-1){
                switch (type_of_loss)
                {
                case 0: //log likelihood
                    deltas[l][i] = training_labels[index_of_example][i]-outputs[l][i];
                    break;
                case 1: //mean squared error
                    float sum_k=0;
                    for (int k=0; k<neurons_output_layer; k++)
                    {
                        sum_k += (training_labels[index_of_example][k]-outputs[l][k]);
                    }
                    deltas[l][i] =-outputs[l][i] * (sum_k+ (training_labels[index_of_example][i]-outputs[l][i]));
                    break;
                default: // log likelihood
                    deltas[l][i] = training_labels[index_of_example][i]-outputs[l][i];
                    break;
                }
            }
            else{
                //backpropagation step
                float weighted_sum = 0;
                for (int k=0; k<neurons_per_layer[l+1]; k++)
                {
                    if(type_of_optimization==2){
                        weighted_sum += (weights[l+1][i][k]+momentum*w_momentum[l+1][i][k])*deltas[l+1][k];
                    }
                    else{
                        weighted_sum += weights[l+1][i][k]*deltas[l+1][k];
                    }
                }
                float activ_derivative;
                switch (type_of_activation)
                {
                case 0: //sigmoid
                    activ_derivative = sigmoid_derivative(activations[l][i]);
                    break;
                case 1: //Relu
                    activ_derivative = ReLU_derivative(activations[l][i]);
                    break;
                default: 
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
                    /*if(l==number_of_layers-1){
                        printf("%.3f ",dw[l][1][2]);
                    }*/
                }
                /*if(l==number_of_layers-1){
                    printf("\n");
                }*/
            }
            db[l][i] += learning_rate*deltas[l][i];
            //for the first layer the outputs are the linearized inputs
        }
    }
    /*for(int i=0;i<neurons_output_layer;i++){
        printf("%.3f ",outputs[number_of_layers-1][i]);
    }
    printf("\n");*/
}

float optimizer_w(int l,int i, int j){
    w_momentum[l][i][j] = momentum*w_momentum[l][i][j]+dw[l][i][j];//dw a questo punto è solo 
    //-lambda*gradiente
    switch (type_of_optimization)
    {
    case 0://sgd
        return dw[l][i][j];
        break;
    case 1:
        return w_momentum[l][i][j];
        break;
    case 2://nesterov
        return w_momentum[l][i][j];
        break; 
    default: //sgd
        return dw[l][i][j];
        break;
    }
}

float optimizer_b(int l,int i){
    b_momentum[l][i] = momentum*b_momentum[l][i]+db[l][i];//dw a questo punto è solo 
    //-lambda*gradiente
    switch (type_of_optimization)
    {
    case 0://sgd
        return db[l][i];
        break;
    case 1:
        return b_momentum[l][i];
        break;
    case 2://nesterov
        return b_momentum[l][i];
        break; 
    default: //sgd
        return db[l][i];
        break;
    }
}

void reset_dw_db()
{   
    for (int l=0; l<number_of_layers; l++)
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

void reset_momentum()
{   
    for (int l=0; l<number_of_layers; l++)
    {
        int dim1 = weights_dim[l][0];
        int dim2 = weights_dim[l][1];
        int n_neurons=neurons_per_layer[l];//cause the input neurons don't do computations
        //initialize the weights of the layer with a gaussian distribution
        for (int i=0; i<dim1; i++)
            for (int j=0; j<dim2; j++) w_momentum[l][i][j] = 0;
        for (int i=0; i<n_neurons; i++) b_momentum[l][i] = 0;
    }
}

void average_dw_db(int M)
{   
    //M is batch size if you are averaging on a mini-batch
    for (int l=0; l<number_of_layers; l++)
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
    for (int l=0; l<number_of_layers; l++)
    {
        int dim1 = weights_dim[l][0];
        int dim2 = weights_dim[l][1];
        int n_neurons=neurons_per_layer[l];//cause the input neurons don't do computations
        //initialize the weights of the layer with a gaussian distribution
        for (int i=0; i<dim1; i++){
            for (int j=0; j<dim2; j++){
                weights[l][i][j] += optimizer_w(l,i,j);
            }
        }
        for (int i=0; i<n_neurons; i++){
            biases[l][i] += optimizer_b(l,i);
        }
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
        learn_example(map_training_images[i]);
        //printf("Example %d: ",i);
        /*for (int j=0;j<10;j++){
            printf("%.3f ",outputs[3][j]);
        }*/
        /*for (int j=0;j<neurons_per_layer[1];j++){
            printf("%.3f ",outputs[1][j]);
        }*/
        //print_image(training_images[map_training_images[i]]);
        //printf("\n");
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
    
    if(type_of_shuffling==1)
        permute(map_training_images, number_of_train_images);

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
        printf("Current epoch=%d, current error=%.3f\n",current_epoch,current_error);
    }while(current_epoch<number_of_epochs && current_error>threshold_error);
}
//----------------------

//----------------------
//loss functions
float log_likelihood(int *t, float *y, int dim)
{
    float C=0;
    float epsilon=1e-30;
    for (int i=0;i<dim;i++){
        C -= t[i]*log(y[i]+epsilon); 
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
float loss_on_example(int *label, float *probabilities,int t_loss)
{
    //takes in input the type of loss function
    switch (t_loss){
        case 0: //log likelihood
            return log_likelihood(label,probabilities,neurons_output_layer);
            break;
        case 1: //mse
            return mean_squared_error(label,probabilities,neurons_output_layer);
            break;
        default: // log likelihood
            return log_likelihood(label,probabilities,neurons_output_layer);
            break;
    }
}

float loss_on_set(int (*labels)[n_classes],float (*probabilities)[n_classes], int dim, int t_loss){
    float loss=0;
    for (int i=0;i<dim;i++){
        loss+=loss_on_example(labels[i],probabilities[i],t_loss);
    }
    return loss/dim;
}

void inference_on_set(int (*input_examples)[input_size][input_size], float (*probabilities)[10], int dim){
    for (int i=0;i<dim;i++){
        float *z=get_probabilities(input_examples[i]);
        for (int j=0;j<10;j++){
            probabilities[i][j]=z[j];
        }
    }

}

Metrics compute_metrics(float (*probabilities)[10],int (*input_labels)[n_classes], int dim){
    Metrics metrics;
    float epsilon=1e-30;
    //initialize metrics
    for (int i=0;i<10;i++){
        for (int j=0;j<10;j++){
            metrics.full_confusion_matrix[i][j]=0;
        }
        for (int j=0;j<2;j++){
            for (int k=0;k<2;k++){
                metrics.micro_confusion_matrix[j][k]=0;
            }
        }
    }
    //compute confusion matrices
    for(int i=0;i<dim;i++){
        int predicted_class=get_best_class(probabilities[i]);
        float labels[10];
        int gold_class=get_best_class(int_to_float(labels,input_labels[i],10));
        metrics.full_confusion_matrix[predicted_class][gold_class]+=1;
    }
    // For every class the 00 of the micro-confusion is the [class][class] entry
    //the 01 is the sum of the [class][not class] entries
    //the 10 is the sum of the [not class][class] entries
    //.. the total umber of entries should be dim
    for(int i=0;i<10;i++){
        metrics.micro_confusion_matrices[i][0][0]=metrics.full_confusion_matrix[i][i];
        metrics.micro_confusion_matrices[i][0][1]=0;
        metrics.micro_confusion_matrices[i][1][0]=0;
        for(int j=0;j<10;j++){
            if(j!=i){
                metrics.micro_confusion_matrices[i][0][1]+=metrics.full_confusion_matrix[i][j];
                metrics.micro_confusion_matrices[i][1][0]+=metrics.full_confusion_matrix[j][i];
            }
        }
        metrics.micro_confusion_matrices[i][1][1]=dim-metrics.micro_confusion_matrices[i][0][0]
        -metrics.micro_confusion_matrices[i][0][1]-metrics.micro_confusion_matrices[i][1][0];
    }
    metrics.overall_accuracy=0;
    metrics.macro_precision=0;
    metrics.macro_recall=0;
    for(int i=0;i<10;i++){
        //I didn't find a definition of accuracy per class
        //metrics.accuracies[i]=metrics.micro_confusion_matrices[i][0][0]/(float)(dim);
        metrics.overall_accuracy+=metrics.micro_confusion_matrices[i][0][0]/(float)dim;
        metrics.recalls[i]=metrics.micro_confusion_matrices[i][0][0]/
        (float)(metrics.micro_confusion_matrices[i][0][0]
        +metrics.micro_confusion_matrices[i][1][0] + epsilon);
        metrics.precisions[i]=metrics.micro_confusion_matrices[i][0][0]/
        (float)(metrics.micro_confusion_matrices[i][0][0]
        +metrics.micro_confusion_matrices[i][0][1] + epsilon);
        metrics.macro_recall+=metrics.recalls[i]/10;
        metrics.macro_precision+=metrics.precisions[i]/10;
        for (int j=0;j<2;j++){
            for (int k=0;k<2;k++){
                metrics.micro_confusion_matrix[j][k]+=metrics.micro_confusion_matrices[i][j][k];
            }
        }
    }
    metrics.micro_recall=metrics.micro_confusion_matrix[0][0]/
    (float)(metrics.micro_confusion_matrix[0][0]
    +metrics.micro_confusion_matrix[1][0]+epsilon);
    metrics.micro_precision=metrics.micro_confusion_matrix[0][0]/
    (float)(metrics.micro_confusion_matrix[0][0]+
    metrics.micro_confusion_matrix[0][1]+epsilon);
    return metrics;
}

//------------------------------------
//retrieval functions
//apply to outputs after a forward propagation
int get_best_class(float input[n_classes])
{
    float max = input[0]; // Assume first output is the max initially
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
    return outputs[number_of_layers-1];
}
//-------------------------------------


//----------------------------
void save_NN(char *filename) {
    /*char temporary[100];
    strcpy(temporary,main_folder_name);
    strcat(temporary,"/models/");
    //printf("%s\n",strcat(temporary,filename)); //input_folder/models/first_working_model
    FILE *file = fopen(strcat(temporary,filename), "w");*/
    char temporary[100]="models/";
    FILE *file = fopen(strcat(temporary,filename), "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }
    //save network structure
    fprintf(file, "%d\n", number_of_layers-1);//number of hidden layers
    for(int i=0;i<number_of_layers-1;i++){
        fprintf(file, "%d ", neurons_per_layer[i]);
    }
    fprintf(file, "\n");
    fprintf(file, "%d\n", type_of_activation);//activation function

    // Save dw
    for(int l=0;l<number_of_layers;l++){
        for (int i = 0; i < weights_dim[l][0]; i++) {
            for (int j = 0; j < weights_dim[l][1]; j++) {
                fprintf(file, "%f ", weights[l][i][j]);
            }
            fprintf(file, "\n");
        }
    }

    // Save db
    for (int l = 0; l < number_of_layers; l++) {
        for (int i = 0; i < neurons_per_layer[l]; i++) {
            fprintf(file, "%f ", biases[l][i]);
        }
        fprintf(file, "\n");
    }
    //Save training data
    fprintf(file, "%d\n", type_of_activation);//activation function
    fprintf(file, "%d\n", type_of_initialization);//initialization function
    fprintf(file, "%d\n", type_of_loss);//loss function
    fprintf(file, "%d\n", type_of_optimization);//optimization function
    fprintf(file, "%d\n", type_of_shuffling);//shuffling function
    fprintf(file, "%d\n", number_of_epochs);//number of epochs
    fprintf(file, "%f\n", learning_rate);//learning rate
    fprintf(file, "%f\n", threshold_error);//threshold error
    fprintf(file, "%f\n", momentum);//momentum
    fprintf(file, "%d\n", minibatch_size);
    fprintf(file, "%d\n", number_of_val_images);
    fprintf(file, "%d\n", number_of_train_images);
    fprintf(file, "%f\n", train_val_split);
    fprintf(file, "%f\n", error_on_validation);

    fclose(file);
}

void load_model(char *filename)
{
    /*char temporary[100];
    strcpy(temporary,main_folder_name);
    strcat(temporary,"/models/");*/
    char temporary[100]="models/";
    FILE *file = fopen(strcat(temporary,filename), "r");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }
    
    int n_hid_layers=0;
    int npl[n_hid_layers];
    int t_activ=0;
    fscanf(file, "%d", &n_hid_layers);//number of hidden layers
    for(int i=0;i<n_hid_layers;i++){
        fscanf(file, "%d", &npl[i]);
    }
    fscanf(file, "%d\n", &t_activ);//activation function

    define_network_structure(npl,n_hid_layers, t_activ, 0);

    // load dw
    for(int l=0;l<number_of_layers;l++){
        for (int i = 0; i < weights_dim[l][0]; i++) {
            for (int j = 0; j < weights_dim[l][1]; j++) {
                fscanf(file, "%f", &weights[l][i][j]);
            }
        }
    }

    // load db
    for (int l = 0; l < number_of_layers; l++) {
        for (int i = 0; i < neurons_per_layer[l]; i++) {
            fscanf(file, "%f", &biases[l][i]);
        }
    }

    fscanf(file, "%d", &type_of_activation);//activation function
    fscanf(file, "%d", &type_of_initialization);//initialization function
    fscanf(file, "%d", &type_of_loss);//loss function
    fscanf(file, "%d", &type_of_optimization);//optimization function
    fscanf(file, "%d", &type_of_shuffling);//shuffling function
    fscanf(file, "%d", &number_of_epochs);//number of epochs
    fscanf(file, "%f", &learning_rate);//learning rate
    fscanf(file, "%f", &threshold_error);//threshold error
    fscanf(file, "%f", &momentum);//momentum
    fscanf(file, "%d", &minibatch_size);
    fscanf(file, "%d", &number_of_val_images);
    fscanf(file, "%d", &number_of_train_images);
    fscanf(file, "%f", &train_val_split);
    fscanf(file, "%f", &error_on_validation);

    fclose(file);
}
