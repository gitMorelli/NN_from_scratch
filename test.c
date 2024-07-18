#include "functions.h"
#include "graphics.h"
#include "unit_testing.h"
#include <allegro5/allegro.h>
#include <allegro5/allegro_primitives.h>
#include <allegro5/allegro_font.h>
#include <stddef.h> // For NULL
#include <stdlib.h> // For srand
#include <stdio.h>
#include <stdbool.h>
#include <time.h>   // For time
#include <math.h>


int main() {
    unsigned int seed = 12345; // Example fixed seed
    srand(seed);
    //training_loop();
    //srand(time(NULL));

    //unit tests
    /*
    test_int_to_float();
    test_set_folder_name();
    test_define_network_structure();
    test_read_labels();
    test_load_test_set();
    test_permute();
    //test_load_training_set(); // i comment it because it takes a couple of seconds to run
    test_split_data();
    test_gaussian_layer_initialization();
    test_weight_initialization();
    test_softmax();
    test_lin_and_norm();
    test_neuron_output();
    test_layer_output();
    test_forward_propagation();
    test_loss_on_example();
    test_get_best_class();
    test_learn_example();
    test_learn_batch();
    test_learn_epoch();
    test_training_loop();
    test_training_loop();
    test_softmax_stable();
    test_training_loop();
    //test_accuracy();
    test_inference_on_set();
    test_compute_metrics();
    test_testing_loop();
    test_training_loop();*/
    test_training_graphics();
    return 0;
}