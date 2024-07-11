#include "functions.h"
#include "graphics.h"
#include <allegro5/allegro.h>
#include <allegro5/allegro_primitives.h>
#include <allegro5/allegro_font.h>
#include <stddef.h> // For NULL
#include <stdlib.h> // For srand
#include <stdio.h>
#include <stdbool.h>
#include <time.h>   // For time
#include <math.h>

int main(int argc, char *argv[])
{
    // i interpret the command line arguments
    // i expect as input only the path of the folder with the 
    //saved data (models, training images, ..)
    char main_folder[max_file_name_length];
    srand(time(NULL));
    strcpy(main_folder, "test_folder"); // default if no file is provided
    if (argc > 1)
        strcpy(main_folder, argv[1]); // specified name if name is provided
    set_folder_name(main_folder);
    main_loop();
    
    return 0;
}

void main_loop()
{
    // the program opens in the menu mode
    // in the menu you can select between train, test and interactive using buttons
    // from every mode you can return to the main menu pressing the menu button
    do {
        switch (INPUT) 
        {
            case '1': menu(); break;
            case '2': train(); break;
            case '3': test(); break;
            case '4': interactive(); break;
            default: break;
        }
    } while (INPUT != 0);
}

void menu()
{
    display_menu();
}

void train()
{
    //In this modality there is a selection region to determine the network structure
    //with a button you can confirm the network structure
    //with another button you start the training

    //load the data i use for training and validation
    load_training_set();
    display_training_interface();
}

void test()
{
    //In this modality there is a selection region to choose the model to test
    //with a button you can start the test

    //load the data i use for testing the model
    load_test_set();
    display_testing_interface();
}

void interactive()
{
    //In this modality you are shown with a region to draw numbers with the mouse
    //you are shown with a region to select the network to use
    //you can submit input with a button
    //when you submit the selected network predicts the result

    display_interactive_interface();
}