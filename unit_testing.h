//#include "functions.h"
#include <math.h>
#include <stdlib.h> // For srand
#include <stdbool.h>
#include <stddef.h> // For NULL
#include <stdio.h>
#include <time.h>   // For time
#include <string.h>
#include <assert.h>

void test_int_to_float()
{
    int x[3]={0,1,2};
    float y[3];
    int_to_float(y,x,3);
    assert(y[0] == 0.0 && y[1] == 1.0 && y[2] == 2.0);
}

void test_set_folder_name()
{
    char main_folder[max_file_name_length];
    set_folder_name("test_folder");
    assert(strcmp(main_folder_name, "test_folder") == 0);
}

void test_define_network_structure(){
    int x[3] = {3,6,8};
    define_network_structure(x,3,0, 1);
    assert(weights_dim[0][1] == 28*28 && weights_dim[0][0] == 3);
    assert(weights_dim[3][0] == 10 && weights_dim[3][1] == 8);
    assert(weights_dim[1][0]==6 && weights_dim[1][1]==3);
}

void test_read_labels(){
    set_folder_name("input_folder");
    int labels[10000];
    FILE *fptr;
    char temporary[100];
    strcpy(temporary,main_folder_name);
    fptr = fopen(strcat(temporary,"/t10k-labels-idx1-ubyte"),"rb");
    set_number_of_inputs(60000, 10000);
    //printf("%d \n", number_of_test_images);
    read_labels(labels,fptr,number_of_test_images);
    /*for (int i = 0; i < 1000; i++){
        printf("%d ",labels[i]);
    }*/
    //printf("%d \n", labels[9000]);
    assert(labels[9000] == 7 && labels[2]==1);

}

void test_load_test_set(){
    set_folder_name("input_folder");
    set_number_of_inputs(60000, 10000);
    //printf("%s \n", main_folder_name);
    load_test_set();
    /*print_image(testing_images[9000]);
    printf("%d \n", number_of_test_images);
    for(int i = 0; i < 10; i++)
    {
        printf("%d ", testing_labels[9000][i]);
    }*/
    assert(testing_labels[9000][6] == 0 && testing_labels[9000][7] == 1 && testing_labels[2][1]==1);
}

void test_load_training_set(){
    set_folder_name("input_folder");
    set_number_of_inputs(60000, 10000);
    //printf("%s \n", main_folder_name);
    load_training_set();
    //print_image(input_images[19000]);
    //printf("%d \n", number_of_test_images);
    /*for(int i = 0; i < 10; i++)
    {
        printf("%d ", input_labels[19000][i]);
    }*/
    assert(input_labels[19000][6] == 0 && input_labels[19000][8] == 1);
}

void test_permute(){
    int x[10];
    for (int i = 0; i < 10; i++){
        x[i] = i;
    }
    permute(x,10);
    int sum=0;
    for (int i = 0; i < 10; i++){
        sum+=x[i];
        //printf("%d ", x[i]);
    }
    assert(sum == 45);
}

void test_split_data(){
    set_number_of_inputs(100, 10000);//i load 100 examples
    //load_test_set();
    set_folder_name("input_folder");
    load_training_set();
    set_train_val(10, 0.3);
    for (int i=0; i<100; i++){
        training_images[i][0][0] = -1;
        validation_images[i][0][0] = -1;
        training_labels[i][0] = -1;
        validation_labels[i][0] = -1;
    }
    split_data();
    int len_train=0;
    int len_val=0;
    for (int i=0; i<100; i++){
        if(training_images[i][0][0] != -1){
            len_train++;
        }
        if(validation_images[i][0][0] != -1){
            len_val++;
        }
    }
    /*print_image(training_images[2]);
    print_image(validation_images[2]);
    for(int i = 0; i < 10; i++)
    {
        printf("%d ", training_labels[2][i]);
    }
    printf("%d\n", len_train);
    printf("%d\n", len_val);*/
    assert(len_train == 70 && len_val == 30);
}

void test_gaussian_layer_initialization(){
    int x[3] = {10,5,7};
    define_network_structure(x,3, 0, 0);
    gaussian_layer_initialization(0);
    float sum=0;
    for (int i=0; i<10; i++)
    {
        for (int j=0; j<28*28; j++)
        {
            //printf("%f ", weights[0][i][j]);
            sum+=weights[0][i][j];
            assert(weights[0][i][j] != 0);
        }
        //printf("\n");
    }
    //printf("SUM = %f",sum);
    gaussian_layer_initialization(2);
    assert(weights_dim[2][0]==7 && weights_dim[2][1]==5);
    sum=0;
    for (int i=0; i<7; i++)
    {
        for (int j=0; j<5; j++)
        {
            //printf("%f ", weights[2][i][j]);
            sum+=weights[2][i][j];
            assert(weights[2][i][j]!=0);
        }
        //printf("\n");
    }
    //printf("SUM = %f",sum);
    gaussian_layer_initialization(1);
    assert(weights_dim[1][0]==5 && weights_dim[1][1]==10);
    sum=0;
    for (int i=0; i<5; i++)
    {
        for (int j=0; j<10; j++)
        {
            //printf("%f ", weights[1][i][j]);
            sum+=weights[1][i][j];
            assert(weights[1][i][j]!=0);
        }
        //printf("\n");
    }
    gaussian_layer_initialization(3);
    assert(weights_dim[3][0]==10 && weights_dim[3][1]==7);
    sum=0;
    for (int i=0; i<10; i++)
    {
        for (int j=0; j<7; j++)
        {
            //printf("%f ", weights[1][i][j]);
            sum+=weights[3][i][j];
            assert(weights[3][i][j]!=0);
        }
        //printf("\n");
    }
}

void test_weight_initialization(){
    int x[3] = {10,5,7};
    define_network_structure(x,3, 0, 1);
    weight_initialization();
    printf("\n");
    for (int i=0;i<10;i++)
    {
        for (int j=0;j<7;j++)
        {
            //printf("%f ",weights[3][i][j]);
        }
        //printf("\n");
    }
    for (int i=0; i<4; i++)
    {
        //printf("%d \n",i);
        assert(weights[i][0][0] != 0);
        assert(weights[i][2][3] != 0);
    }
}

void test_softmax(){
    float x[3] = {1,2,3};
    softmax(x,3);
    float sum =0;
    for (int i=0;i<3;i++){
        sum+=x[i];
        //printf("%f ",x[i]);
    }
    assert(sum == 1);
}

void test_softmax_stable(){
    float x[3] = {1,2,3};
    float y[3] = {1,2,3};
    softmax(x,3);
    softmax_stable(y,3);
    //printf("%f %f %f\n",x[0],x[1],x[2]);
    assert(fabs(x[0] - y[0])<=0.0001);
    assert(fabs(x[1] - y[1])<=0.0001);
    assert(fabs(x[2] - y[2])<=0.0001);
}

void test_lin_and_norm(){
    int x[28][28];
    for (int i=0;i<28;i++){
        for (int j=0;j<28;j++){
            x[i][j] = i*28+j+1;
        }
    }   
    float *y=lin_and_norm(x);
    for (int i=0;i<28*28;i++){
        //printf("%f ",y[i]);
    }
    //printf("%f \n",y[0]*255);
    assert(fabs(y[0] - 1/255.0)<=0.0001);
    assert(fabs(y[1] - 2/255.0)<=0.0001);
    assert(fabs(y[28] - 29/255.0));
    assert(fabs(y[29] - 30/255.0));
}

//definisco due layer con due e tre neuroni rispettivamente
//definisco dei valori per i pesi che posso controllare
//definisco dei valori per i bias che posso controllare
//testo sia no activation function che sigmoid
void test_neuron_output(){
    int L[3] = {2,3,5};
    define_network_structure(L,3, 0, 1);
    weights_dim[1][0] = 3;
    weights_dim[1][1] = 2;
    weights[1][0][0] = 0.1;
    weights[1][0][1] = 0.2;
    weights[1][1][0] = 0.3;
    weights[1][1][1] = 0.4;
    weights[1][2][0] = 0.5;
    weights[1][2][1] = 0.6;
    biases[1][0] = 0.1;
    biases[1][1] = 0.2;
    biases[1][2] = 0.3;
    float x[2] = {0.8,0.4};
    int activation_funct=0;
    neuron_output(1, 1, x, activation_funct);
    //printf("activation: %f\n", activations[1][1]);
    //printf("output: %f\n", outputs[1][1]);
    if(activation_funct==0){
        assert(outputs[1][1] <= 0.65 && outputs[1][1] >= 0.64);
        assert(fabs(activations[1][1] - 0.6)<=0.0001);
    }
    else if(activation_funct==2){
        assert(fabs(activations[1][1] - 0.6)<=0.0001 && fabs(outputs[1][1] - 0.6)<=0.0001);
    }
}

void test_layer_output(){
    int L[3] = {2,3,5};
    define_network_structure(L,3, 0, 1);
    weights_dim[1][0] = 3;
    weights_dim[1][1] = 2;
    weights[1][0][0] = 0.1;
    weights[1][0][1] = 0.2;
    weights[1][1][0] = 0.3;
    weights[1][1][1] = 0.4;
    weights[1][2][0] = 0.5;
    weights[1][2][1] = 0.6;
    biases[1][0] = 0.1;
    biases[1][1] = 0.2;
    biases[1][2] = 0.3;
    float x[2] = {0.8,0.4};
    int activation_funct=0;
    neurons_per_layer[1] = 3;
    layer_output(x, 1, activation_funct);
    //printf("activation: %f\n", activations[1][0]);
    assert(fabs(outputs[1][0] - 0.56)<=0.01);
    assert(fabs(activations[1][0] - 0.26)<=0.0001);
    assert(fabs(outputs[1][1] - 0.64)<=0.01);
    assert(fabs(activations[1][1] - 0.6)<=0.0001);
    assert(fabs(outputs[1][2] - 0.72)<=0.01);
    assert(fabs(activations[1][2] - 0.94)<=0.0001);
}

void test_forward_propagation(){
    int L[3] = {2,3,2};
    define_network_structure(L,3, 0, 0);
    int x[28][28];
    for (int i=0;i<28;i++){
        for (int j=0;j<28;j++){
            x[i][j] = 0;
        }
    }
    x[0][1]=255.0;
    type_of_initialization=5;
    weight_initialization();
    weights[0][0][1] = 0.2;
    weights[0][1][1] = 0.5;
    //printf("%d \n",neurons_per_layer[0]);
    //printf("%f \n",biases[0][0]);
    forward_propagation(x);
    /*float *y = linearize(x);
    for (int i=0;i<28*28;i++){
        printf("%f ",y[i]);
    }*/
    //printf("%f \n",activations[3][4]);
    //printf("%f \n",outputs[3][4]);
    assert(fabs(outputs[0][1] - 0.668)<=0.01);
    assert(fabs(outputs[0][0] - 0.57)<=0.01);
    assert(fabs(activations[1][0] - 0.29)<=0.01);
    assert(fabs(outputs[1][2] - 0.72)<=0.02);
    assert(fabs(activations[2][1] - 1.19)<=0.01);
    assert(fabs(outputs[2][0] - 0.62)<=0.01);
    assert(fabs(activations[3][4] - 1.829)<=0.02);
    assert(fabs(outputs[3][4] - 0.0487)<=0.01);
    assert(fabs(outputs[3][0] - 0.0107)<=0.001);
    assert(fabs(outputs[3][9] - 0.322)<=0.01);
}

void test_loss_on_example(){
    int L[3] = {2,3,5};
    define_network_structure(L,3, 0, 1);
    int x[10] = {0,0,0,0,0,0,0,0,0,1};
    for (int i=0;i<10;i++){
        outputs[3][i] = 0.1;
    }
    outputs[3][9] = 0.9;
    float loss=loss_on_example(x,0);
    assert(fabs(loss - 0.10536)<=0.01);
}

void test_get_best_class(){
    float x[10]={1,2,4,2,3,1,2,3,4,5};
    int result=get_best_class(x);
    assert(result == 9);
    float z[10]={1,2,4,2,13,1,2,3,4,5};
    result=get_best_class(z);
    assert(result == 4);
    float y[10]={10,2,4,2,3,1,2,3,4,5};
    result=get_best_class(y);
    assert(result == 0);
}
// if i train on a single image but is always the same it should learn to output the correct label
//-> i can make a train loop on the single image and then test the output
//I should also check that the weights, deltas etc are not zero
void test_learn_example(){
    int L[3] = {10,6,4};
    define_network_structure(L,3, 0, 1);
    weight_initialization();
    set_folder_name("input_folder");
    set_number_of_inputs(10, 10);
    load_training_set();
    define_training_parameters(10,0.1, 0, 0, 0.00001,0,0.9);
    /*print_image(input_images[1]); //is a 0
    for(int i = 0; i < 10; i++)
    {
        printf("%d ", input_labels[1][i]);
    }*/
    for (int i=0;i<28;i++){
        for (int j=0;j<28;j++){
            training_images[0][i][j] = input_images[1][i][j];
        }
    }
    for(int i=0;i<10;i++){
        training_labels[0][i] = input_labels[1][i];
    }
    for (int i=0;i<50;i++){
        reset_dw_db();
        learn_example(0);
        update_w_b();
        float loss=loss_on_example(training_labels[0],type_of_loss);
        //printf("LOSS at %d: %f\n",i,loss);
        /*for (int j=0;j<10;j++){
            printf("%f ",outputs[3][j]);
            //printf("%f ",biases[3][j]);
        }
        printf("\n");*/
    }
    int result=get_best_class(outputs[3]);
    //printf("%d\n",result);
    assert(result == 0);
}

void test_learn_batch(){
    int L[1] = {64};
    define_network_structure(L,1, 0, 1);
    weight_initialization();
    set_folder_name("input_folder");
    set_number_of_inputs(100, 10);
    load_training_set();
    define_training_parameters(10,1, 0, 0, 0.00001,0,0.9);
    set_train_val(10, 0.5);
    split_data();
    float label[n_classes];
    for (int i=0;i<number_of_train_images;i++){
        printf("%d ",map_training_images[i]);
    }
    printf("\n");
    for (int i=0;i<number_of_train_images;i++){
        printf("%d ",get_best_class(int_to_float(label,training_labels[map_training_images[i]],n_classes)));
    }
    printf("\n");
    printf("SHUFFLING\n");
    permute(map_training_images,number_of_train_images);
    for (int i=0;i<number_of_train_images;i++){
        printf("%d ",map_training_images[i]);
    }
    printf("\n");
    for (int i=0;i<number_of_train_images;i++){
        printf("%d ",get_best_class(int_to_float(label,training_labels[map_training_images[i]],n_classes)));
    }
    printf("\n");
    printf("BATCH\n");
    int batch_index=2;
    int start = batch_index*minibatch_size;
    int end = (batch_index+1)*minibatch_size;
    for (int i=start; i<end; i++)
    {
        printf("%d ",get_best_class(int_to_float(label,training_labels[map_training_images[i]],n_classes)));
    }
    printf("\n");
    
    for (int i=0;i<number_of_epochs;i++){
        printf("EPOCH %d\n",i);
        learn_batch(2);
        printf("LOSS on batch at %d: %f \n",i,error_on_batch);
        /*for (int j=0;j<10;j++){
            printf("%f ",outputs[3][j]);
            //printf("%f ",biases[3][j]);
        }
        printf("\n");*/
    }
    for (int i=start;i<end;i++){
        forward_propagation(training_images[map_training_images[i]]);
        for (int j=0;j<10;j++){
            printf("%f ",outputs[number_of_layers-1][j]);
        }
        printf("%d ",get_best_class(outputs[number_of_layers-1]));
        printf("\n");
    }
    printf("\n");
}

void test_learn_epoch(){
    int L[1] = {32};
    define_network_structure(L,1, 0, 1);
    weight_initialization();
    set_folder_name("input_folder");
    set_number_of_inputs(10000, 10);
    load_training_set();
    define_training_parameters(10,1, 0, 1, 0.00001,0,0.9);
    set_train_val(100, 0.1);
    split_data();
    learn_epoch();
    int batch_index=5;
    int start = batch_index*minibatch_size;
    int end = (batch_index+1)*minibatch_size;
    float label[n_classes];
    for (int i=start; i<end; i++)
    {
        printf("%d ",get_best_class(int_to_float(label,training_labels[map_training_images[i]],n_classes)));
    }
    printf("\n");
    printf("Results\n");
    for (int i=start;i<end;i++){
        forward_propagation(training_images[map_training_images[i]]);
        for (int j=0;j<10;j++){
            printf("%f ",outputs[number_of_layers-1][j]);
        }
        printf("%d ",get_best_class(outputs[number_of_layers-1]));
        printf("\n");
    }
    printf("\n");
}

void test_training_loop(){
    int L[1] = {64};
    define_network_structure(L,1, 0, 1);
    weight_initialization();
    set_folder_name("input_folder");
    set_number_of_inputs(10000, 10);
    load_training_set();
    define_training_parameters(10,0.1, 0, 1, 0.00001,2,0.9);
    set_train_val(10, 0.1);
    split_data();
    train_network();
    int batch_index=5;
    int start = batch_index*minibatch_size;
    int end = (batch_index+1)*minibatch_size;
    float label[n_classes];
    printf("Results on ten random images\n");
    for (int i=0;i<20;i++){
        forward_propagation(training_images[map_training_images[i]]);
        for (int j=0;j<10;j++){
            printf("%f ",outputs[number_of_layers-1][j]);
        }
        printf("%d ",get_best_class(outputs[number_of_layers-1]));
        printf(" | %d ",get_best_class(int_to_float(label,training_labels[map_training_images[i]],n_classes)));
        printf("\n");
    }
}

void test_inference_on_set(){
    int L[1] = {64};
    define_network_structure(L,1, 0, 1);
    weight_initialization();
    set_folder_name("input_folder");
    set_number_of_inputs(10000, 10);
    load_training_set();
    define_training_parameters(10,0.1, 0, 1, 0.00001,2,0.9);
    set_train_val(10, 0.1);
    split_data();
    train_network();
    int batch_index=5;
    int start = batch_index*minibatch_size;
    int end = (batch_index+1)*minibatch_size;
    float label[n_classes];
    int test_images[20][28][28];
    int test_labels[20][10];
    for (int i=0;i<20;i++){
        for (int j=0;j<28;j++){
            for (int k=0;k<28;k++){
                test_images[i][j][k] = training_images[map_training_images[i]][j][k];
            }
        }
        forward_propagation(training_images[map_training_images[i]]);
        for (int j=0;j<10;j++){
            test_labels[i][j] = training_labels[map_training_images[i]][j];
            printf("%f ",outputs[number_of_layers-1][j]);
        }
        printf("%d ",get_best_class(outputs[number_of_layers-1]));
        printf(" | %d ",get_best_class(int_to_float(label,training_labels[map_training_images[i]],n_classes)));
        printf("\n");
    }
    printf("Testing inference on set\n");
    float probabilities[20][10];
    inference_on_set(test_images,probabilities,20);
    for (int i=0;i<20;i++){
        for (int j=0;j<10;j++){
            printf("%f ",probabilities[i][j]);
        }
        printf("\n");
    }
}

void test_compute_metrics(){
    float probabilities[10][10];
    int gold_standards[10][10];
    printf("Probabilities vs Gold standards\n");
    for (int i=0;i<10;i++){
        for (int j=0;j<10;j++){
            probabilities[i][j] = 0;
        }
        int ind = rand() % 3;
        printf("%d ",ind);
        probabilities[i][ind] = 1;
    }
    printf("\n");
    for (int i=0;i<10;i++){
        for (int j=0;j<10;j++){
            gold_standards[i][j] = 0;
        }
        int ind = rand() % 3;
        printf("%d ",ind);
        gold_standards[i][ind] = 1;
    }
    printf("\n");
    Metrics M=compute_metrics(probabilities,gold_standards,10);
    printf("precision scores\n");
    for (int i=0;i<3;i++){
        printf("%f ",M.precisions[i]);
    }
    printf("\n");
    printf("Recall scores\n");
    for (int i=0;i<3;i++){
        printf("%f ",M.recalls[i]);
    }
    printf("\n");
    printf("global scores\n");
    printf("Accuracy=%f Macro_recall=%f Macro_precision=%f Micro_recall=%f Micro_precision=%f",
    M.overall_accuracy,M.macro_recall,M.macro_precision,M.micro_recall,M.micro_precision);
    printf("\n");
    for(int i=0;i<2;i++){
        for(int j=0;j<2;j++){
            printf("%d ",M.micro_confusion_matrix[i][j]);
        }
        printf("\n");
    }
}