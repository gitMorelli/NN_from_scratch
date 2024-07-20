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

int main(int argc, char *argv[])
{
    /*char main_folder[max_file_name_length];
    srand(time(NULL));
    strcpy(main_folder, "input_folder"); // default if no file is provided
    if (argc > 1)
        strcpy(main_folder, argv[1]); // specified name if name is provided
    set_folder_name(main_folder);*/
    unsigned int seed = 1234; // Example fixed seed
    srand(seed);
    menu_loop();

    return 0;
}