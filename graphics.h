#include <allegro5/allegro.h>
#include <allegro5/allegro_primitives.h>
#include <allegro5/allegro_font.h>
#include <allegro5/allegro_ttf.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#define FPS 60
#define FRAME_TAU 60
#define NUM_VERTICES 4096
#define KEY_SEEN     1
#define KEY_RELEASED 2
#define WIDTH 1400
#define HEIGHT 800
#define MAX_BUTTONS 20
#define MAX_FILE_NAME_LENGTH 1000

static int INPUT; //stores the input of the user (which come from the mouse interaction with some buttons)
// quit=0, menu=1, train=2, test=3, interactive=4
static ALLEGRO_TIMER* timer;
static ALLEGRO_EVENT_QUEUE* queue;
static ALLEGRO_DISPLAY* disp;
static ALLEGRO_FONT* font;
static ALLEGRO_FONT* font_roboto;
static ALLEGRO_FONT* small_font;
static bool done;
static bool redraw;
static ALLEGRO_EVENT event;
static float x_graphics, y_graphics;
static unsigned char key[ALLEGRO_KEY_MAX];
static int width=WIDTH;
static int height=HEIGHT;
static float x_lim_sup=10;
static float y_lim_sup=-1;
static float x_lim_inf=0;
static float y_lim_inf=0;
static float x_pos_sup=-1;
static float y_pos_sup=-1;
static float x_pos_inf=-1;
static float y_pos_inf=-1;
struct Box{
    int x1;
    int y1;
    int x2;
    int y2;
    ALLEGRO_COLOR color;
    char text[1000];
    int text_position;
    ALLEGRO_COLOR text_color;
};
typedef struct{
    struct Box boxes[MAX_BUTTONS];
    int num_boxes;
} Boxes;

bool drawing = false; // Flag to track whether we are currently drawing
bool inference = false;
bool is_typing=false;
char typed_text[1000]={"->"};
char text_to_show[1000];

char valid_names[100][MAX_FILE_NAME_LENGTH];
char result_string[100];


void display_training_interface();
void display_testing_interface();
void must_init(bool test, const char *description);
void menu_initialization();
void generic_initialization();
void destroy_graphics();
bool is_point_inside_button();
void menu_loop();
void interactive_mode_initialization();
struct Box create_default_box();
void interactive_loop();
int read_pixel_from_display(ALLEGRO_BITMAP *backbuffer,int x, int y);
int process_drawing_region(int dim, int **drawing_matrix);
void training_loop();
void testing_loop();
void drawing_function(int x, int y, int drawing_x,int drawing_y, int r,int dim,int **drawing_matrix);
void upsample_image(int image[28][28], int dim, int upsampled_image[dim][dim]);
ALLEGRO_BITMAP *create_bitmap_from_image(int dim,int image[dim][dim]);
void display_next_image(int dim, int x, int y,int index);
void place_object_grid(int *x,int *y,int *w, int *h, int x_grid, int y_grid, 
int w_grid, int h_grid, float grid_size_x, float grid_size_y);
void place_object(int x,int y,float x_ratio, float y_ratio);
void draw_multiline_text(int text_position,ALLEGRO_COLOR color,ALLEGRO_FONT *font, float x, float y, float line_height, const char *text);
int read_input(float *x, float min, float max, int dim,char *text);
int is_valid_number(char *str);
void print_numbers(char *result, float *numbers, int n);
void draw_plot(struct Box container , float *y, float *x, int dim, ALLEGRO_COLOR color);
void plot_metrics(struct Box results_region,Metrics resulting_metrics);
int read_filename(char (*valid_names)[MAX_FILE_NAME_LENGTH], int dim,char *text);
int read_filenames_from_directory(char (*filenames)[MAX_FILE_NAME_LENGTH]);
void draw_matrix(int **drawing_matrix, int drawing_x,int drawing_y,int dim);
void show_typing_interface(struct Box input_box, char *text_to_show, char *typed_text);
void load_saved_model(int n_files,struct Box *box);
void get_keyboard_input();
void draw_box_elements(struct Box *boxes,int num_boxes);
void print_network_characteristics();
void init_input_box(struct Box *input_box);
Boxes initialize_testing_boxes();
Boxes initialize_interactive_boxes();
Boxes initialize_training_training_boxes();
Boxes initialize_training_input_fields(int num_input_fields);
Boxes initialize_training_input_boxes();

int read_filenames_from_directory(char (*filenames)[MAX_FILE_NAME_LENGTH]) {
    char *directoryPath;
    char buffer[1024]; // Adjust the size as necessary
    // Attempt to get the current working directory
    directoryPath = getcwd(buffer, sizeof(buffer));
    if (directoryPath == NULL) {
        perror("getcwd");
        return -1;
    }
    strcat(directoryPath,"/models");
    DIR *dir = opendir(directoryPath);
    if (dir == NULL) {
        perror("opendir");
        return -1;
    }
    struct dirent *ent;
    int count=0;
    while ((ent = readdir(dir)) != NULL) {
        // Skip "." and ".." entries
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
            continue;
        }

        // Construct the full path of the file
        char fullPath[1024];
        snprintf(fullPath, sizeof(fullPath), "%s/%s", directoryPath, ent->d_name);

        // Use stat() to get file status
        struct stat fileInfo;
        if (stat(fullPath, &fileInfo) == 0) {
            // Check if it's a regular file
            if (S_ISREG(fileInfo.st_mode)) {
                //printf("Regular file: %s\n", fullPath);
                strcpy(filenames[count],ent->d_name);
                count++;
            }
        } else {
            perror("stat");
        }
    }

    closedir(dir);
    return count;
}

int is_valid_number(char *str) {//check if string can be converted to float
    char *endptr;
    strtod(str, &endptr); // Use strtod to attempt to convert the string to a double
    if (endptr == str || *endptr != '\0') return 0; // Check if conversion was successful and entire string was consumed
    return 1;
}

void draw_multiline_text(int text_position,ALLEGRO_COLOR color,ALLEGRO_FONT *font, float x, float y, float line_height, const char *text) {
    char line[1000];
    for(int i=0,j=0;i<strlen(text);i++,j++){
        //printf("%d -> %s\n",i,line);
        if(text[i]=='\n' || text[i]=='\0'){
            al_draw_text(font, color, x, y, text_position, line); // Draw the last line
            memset(line, 0, sizeof(line)); // Clear the line buffer
            j=-1;
            y+=line_height;
        }
        else{
            line[j]=text[i];
        }
    }
}

int read_input(float *x, float min, float max, int dim,char *text){
    int count=0;
    char text_copy[1000];
    strcpy(text_copy,text);
    text_copy[0]=' ';//i remove the arrow character from the input string
    text_copy[1]=' ';
    char *token = strtok(text_copy, " ");
    while (token != NULL) {
        //printf("%s\n",token);
        if (is_valid_number(token)) {
            x[count] = atof(token); // Convert token to float and store it
            if(x[count]<min || x[count]>max){
                printf("Invalid input: %f is not in the range [%f, %f].\n", x[count], min, max);
                return 0; // Exit if an invalid number is found
            }
            count++;
        } else {
            printf("Invalid input: %s is not a valid number.\n", token);
            return 0; // Exit if an invalid number is found
        }
        token = strtok(NULL, " "); // Get next token
    }
    if (count != dim) {
        printf("Invalid input: Expected %d numbers, but got %d.\n", dim, count);
        return 0; // Exit if the number of inputs does not match the expected dimension
    }
    return 1;
}

int read_filename(char (*valid_names)[MAX_FILE_NAME_LENGTH], int dim,char *text){
    //return the index of the valid filename
    char text_copy[1000];
    for(int i=2;i<strlen(text);i++){
        text_copy[i-2]=text[i];
    }
    text_copy[strlen(text)-2]='\0';
    for(int i=0;i<dim;i++){
        //printf("%s %s %s\n",text, text_copy, valid_names[i]);
        if(strcmp(text_copy,valid_names[i])==0){
            return i;
        }
    }
    return -1;
}

void print_numbers(char *result, float *numbers, int n){
    for (int i = 0; i < n; i++) {
        char numStr[20]; // Temporary string for the current number
        sprintf(numStr, "%.2f", numbers[i]); // Convert number to string
        strcat(result, numStr); // Concatenate number string to result
        if (i < n - 1) {
            strcat(result, " "); // Add a space after the number, except for the last one
        }
    }
}

struct Box create_default_box() {
    struct Box newBox;
    newBox.x1 = 0; // Default x1 value
    newBox.y1 = 0; // Default y1 value
    newBox.x2 = 100; // Default x2 value
    newBox.y2 = 50; // Default y2 value
    strcpy(newBox.text, "Default Text"); // Default text
    newBox.text_color=al_map_rgb(255, 255, 255);
    newBox.color = al_map_rgb(255, 255, 255); // Default color
    newBox.text_position=1;
    return newBox;
}

bool is_point_inside_button(int x, int y, struct Box button) {
    return (x >= button.x1 && x <= button.x2 && y >= button.y1 && y <= button.y2);
}

void must_init(bool test, const char *description)
{
    if(test) return;

    printf("couldn't initialize %s\n", description);
    exit(1);
}

void generic_initialization()
{
    must_init(al_init(), "allegro");
    must_init(al_install_keyboard(), "keyboard");
    must_init(al_install_mouse(),"mouse");

    timer = al_create_timer(1.0 / 30.0);
    must_init(timer, "timer");

    queue = al_create_event_queue();
    must_init(queue, "queue");

    al_set_new_display_option(ALLEGRO_SAMPLE_BUFFERS, 1, ALLEGRO_SUGGEST);
    al_set_new_display_option(ALLEGRO_SAMPLES, 8, ALLEGRO_SUGGEST);
    al_set_new_bitmap_flags(ALLEGRO_MIN_LINEAR | ALLEGRO_MAG_LINEAR);

    disp = al_create_display(width, height);
    must_init(disp, "display");

    font = al_create_builtin_font();
    must_init(font, "font");

    al_init_ttf_addon();
    al_init_font_addon();

    // Load a TTF font with a specific size
    font_roboto = al_load_ttf_font("fonts/roboto-font/RobotoRegular-3m4L.ttf", 36, 0); // Example size 36
    must_init(font_roboto, "font");

    must_init(al_init_primitives_addon(), "primitives");

    al_register_event_source(queue, al_get_keyboard_event_source());
    al_register_event_source(queue, al_get_display_event_source(disp));
    al_register_event_source(queue, al_get_timer_event_source(timer));
    al_register_event_source(queue, al_get_mouse_event_source());

    x_graphics = 100;
    y_graphics = 100;

    memset(key, 0, sizeof(key));//inititalize all elements of the array to 0

    al_start_timer(timer);

    drawing = false; // Flag to track whether we are currently drawing
    inference = false;
    is_typing=false;
    done = false;
    redraw = true;

    valid_names[100][MAX_FILE_NAME_LENGTH];
    result_string[100];
    memset(result_string, 0, sizeof(result_string));


}

void destroy_graphics()
{
    al_destroy_font(font);
    al_destroy_display(disp);
    al_destroy_timer(timer);
    al_destroy_event_queue(queue);
}

void menu_loop(){
    generic_initialization();
    int x_blocks=width/5; //i divide in fifths and the buttons take the central fifth
    int y_blocks=height/10; //the three boxes will go in 
    int submenu=-1;//index to tell in which windows i need to go when menu is closed
    struct Box boxes[3];
    ALLEGRO_COLOR color[3]={al_map_rgb(255, 0, 0),al_map_rgb(0, 255, 0),al_map_rgb(0, 0, 255)};
    char text[3][15]={"TRAIN","TEST","INTERACTIVE"};
    for (int i=0;i<3;i++){
        boxes[i].x1=x_blocks*2;
        boxes[i].x2=x_blocks*3;
        boxes[i].y1=y_blocks*(1+i*3);
        boxes[i].y2=y_blocks*(3+i*3);
        boxes[i].color=color[i];
        strcpy(boxes[i].text,text[i]);
    }
    //printf("%s\n",boxes[0].text);
    while(1)
    {
        al_wait_for_event(queue, &event);

        switch(event.type)
        {
            case ALLEGRO_EVENT_TIMER:
                if(key[ALLEGRO_KEY_ESCAPE])
                    done = true;
                for(int i = 0; i < ALLEGRO_KEY_MAX; i++)
                    key[i] &= KEY_SEEN;
                redraw = true;
                break;
            case ALLEGRO_EVENT_MOUSE_BUTTON_DOWN:
                if (is_point_inside_button(event.mouse.x, event.mouse.y, boxes[0])) {
                    submenu=0;
                    done=true;
                }
                else if(is_point_inside_button(event.mouse.x, event.mouse.y, boxes[1])){
                    submenu=1;
                    done=true;
                }
                else if (is_point_inside_button(event.mouse.x, event.mouse.y, boxes[2])){
                    submenu=2;
                    done=true;
                }
                break; // Example action: break out of the loop
            case ALLEGRO_EVENT_KEY_DOWN:
                key[event.keyboard.keycode] = KEY_SEEN | KEY_RELEASED;
                //key_seen is 1=00000001
                //key_released is 2=00000010
                // seen|released=3=00000011
                break;
            case ALLEGRO_EVENT_KEY_UP:
                key[event.keyboard.keycode] &= KEY_RELEASED;
                break;

            case ALLEGRO_EVENT_DISPLAY_CLOSE:
                done = true;
                break;
        }

        if(done)
            break;

        if(redraw && al_is_event_queue_empty(queue)){
            al_clear_to_color(al_map_rgb(0, 0, 0));
            al_draw_textf(font, al_map_rgb(255, 255, 255), width/2, 10, 1, "MENU");
            for (int i=0;i<3;i++){
                al_draw_filled_rectangle(boxes[i].x1, boxes[i].y1, boxes[i].x2, boxes[i].y2, boxes[i].color);
                float center_x=(boxes[i].x1+boxes[i].x2)/2;
                float center_y=(boxes[i].y1+boxes[i].y2)/2;
                al_draw_textf(font, al_map_rgb(255, 255, 255), center_x, center_y, 1,"%s", boxes[i].text);
            }

            al_flip_display();

            redraw = false;
        }
    }
    destroy_graphics();
    switch (submenu)
    {
    case 0:
        training_loop();
        break;
    case 1:
        testing_loop();
        break;
    case 2:
        interactive_loop();
        break;
    
    default:
        break;
    }
}


void drawing_function(int x, int y, int drawing_x,int drawing_y, int r,int dim,int **drawing_matrix){
    al_draw_filled_rectangle(x-r, y-r, x+r, y+r, al_map_rgb(0, 0, 0));
    //al_draw_filled_circle(x, y,r, al_map_rgb(0, 0, 0));
    int x_new=x-drawing_x;
    int y_new=y-drawing_y;//x and y in the reference frame of the drawing area
    //consider that in matrices you have i=row and j=column while for pixels is opposite
    for(int i=y_new-r;i<=y_new+r;i++){
        for(int j=x_new-r;j<=x_new+r;j++){
            if(i>=0 && i<dim && j>=0 && j<dim){
                drawing_matrix[i][j]=255;
            }
        }
    }
}

void draw_matrix(int **drawing_matrix, int drawing_x,int drawing_y,int dim){
    for(int i=0;i<dim;i++){
        for(int j=0;j<dim;j++){
            if(drawing_matrix[i][j]==255){
                int x_new=i+drawing_y;
                int y_new=j+drawing_x;//x and y in the reference frame of the drawing area
                al_draw_filled_rectangle(y_new, x_new, y_new+1, x_new+1, al_map_rgb(0, 0, 0));
            }
        }
    }

}
//which elements i want in the window? A window where i can draw numbers with the mouse
//a window where i can select the network to use
void interactive_loop(){
    generic_initialization();
    float last_x = 0, last_y = 0; // Keep track of the last position
    int submenu=2;//index to tell in which windows i need to go when menu is closed
    int n_files=read_filenames_from_directory(valid_names);
    /*for(int i=0;i<n_files;i++){
        printf("%s\n",valid_names[i]);
    }*/

    Boxes graphic_elements=initialize_interactive_boxes();
    struct Box *boxes=graphic_elements.boxes;
    int num_boxes=graphic_elements.num_boxes;
    struct Box clear_button = boxes[0];
    struct Box NN_selection = boxes[1];
    struct Box NN_selection_result = boxes[2];
    struct Box result_box = boxes[3];
    struct Box drawing_area = boxes[4];
    struct Box submit_button = boxes[5];
    int dim_drawing=drawing_area.x2-drawing_area.x1;
    int **drawing_matrix;
    drawing_matrix = (int **)malloc(dim_drawing * sizeof(int *));
    for (int i = 0; i < dim_drawing; i++) {
        drawing_matrix[i] = (int *)malloc(dim_drawing * sizeof(int));
    }
    for (int i=0;i<dim_drawing;i++){
        for (int j=0;j<dim_drawing;j++){
            drawing_matrix[i][j]=0;
        }
    }
    
    
    struct Box input_box = create_default_box();
    init_input_box(&input_box);

    while(1){
        al_wait_for_event(queue, &event);

        switch(event.type)
        {
            case ALLEGRO_EVENT_TIMER:
                if(key[ALLEGRO_KEY_ESCAPE]){
                    submenu=-1;
                    done = true;
                }
                if(is_typing && key[ALLEGRO_KEY_ENTER]){
                    //int i_file=read_filename(valid_names, n_files,typed_text);
                    load_saved_model(n_files,&boxes[2]);
                }
                for(int i = 0; i < ALLEGRO_KEY_MAX; i++)
                    key[i] &= KEY_SEEN;
                //redraw = true;
                break;
            case ALLEGRO_EVENT_MOUSE_BUTTON_DOWN:
                if (is_point_inside_button(event.mouse.x, event.mouse.y, clear_button)) {
                    redraw = true;
                    //clear drawing matrix
                    for (int i=0;i<dim_drawing;i++){
                        for (int j=0;j<dim_drawing;j++){
                            drawing_matrix[i][j]=0;
                        }
                    }
                }
                if (is_point_inside_button(event.mouse.x, event.mouse.y, drawing_area)) {
                    //rintf("drawing\n");
                    drawing = true; // Start drawing
                    last_x = event.mouse.x; // Update last position
                    last_y = event.mouse.y;
                    redraw=true;
                }
                if (is_point_inside_button(event.mouse.x, event.mouse.y, submit_button)) {
                    // Button was clicked, perform an action
                    inference = true;
                    redraw=true;
                }
                if (is_point_inside_button(event.mouse.x, event.mouse.y, NN_selection)) {
                    is_typing=true;
                    char temporary[10000]="Insert the number of the file in the list \n (first is 0) to load the neural network:\n";
                    for(int i=0;i<n_files;i++){
                        strcat(temporary,valid_names[i]);
                        strcat(temporary,"\n");
                    }
                    strcpy(text_to_show,temporary);
                    redraw = true;
                }
                break;
            case ALLEGRO_EVENT_MOUSE_BUTTON_UP:
                drawing = false; // Stop drawing
                break;
            case ALLEGRO_EVENT_MOUSE_AXES:
                if (drawing) {
                    //printf("still drawing\n");
                    last_x = event.mouse.x;
                    last_y = event.mouse.y;
                }
                break;
            case ALLEGRO_EVENT_KEY_DOWN:
                key[event.keyboard.keycode] = KEY_SEEN | KEY_RELEASED;
                break;
            case ALLEGRO_EVENT_KEY_UP:
                key[event.keyboard.keycode] &= KEY_RELEASED;
                break;
            case ALLEGRO_EVENT_KEY_CHAR:
                if(is_typing){
                    get_keyboard_input();
                }
                break;
            case ALLEGRO_EVENT_DISPLAY_CLOSE:
                done = true;
                break;
        }

        if(done)
            break;
        if(drawing){
            //printf("drawing\n");
            drawing_function(last_x,last_y, drawing_area.x1,drawing_area.y1, 
            10, dim_drawing, drawing_matrix);//fill drawing matrix
        }
        if(drawing && al_is_event_queue_empty(queue)){
            al_flip_display();
        }
        if(redraw && al_is_event_queue_empty(queue)){
            //printf("aaaaah\n");
            if (inference){
                //printf("here 0\n");
                int guess=process_drawing_region(dim_drawing, drawing_matrix);
                //printf("guess=%d\n",guess);
                inference=false;
                sprintf(boxes[3].text,"RESULT: %d",guess);
            }
            al_clear_to_color(al_map_rgb(0, 0, 0));
            al_draw_textf(font, al_map_rgb(255, 255, 255), width/2, 10, 1, "INTERACTIVE");
            draw_box_elements(boxes,num_boxes);
            draw_matrix(drawing_matrix,drawing_area.x1,drawing_area.y1,dim_drawing);
            if(is_typing){
                show_typing_interface(input_box,text_to_show,typed_text);
            }
            al_flip_display();

            redraw = false;
        }
    }
    destroy_graphics();
    for (int i = 0; i < dim_drawing; i++) {
        free(drawing_matrix[i]);
    }
    free(drawing_matrix);
    switch (submenu)
    {
    case -1:// if escape is pressed i return to the menu
        menu_loop();
        break;
    default:
        break;
    }
}

void testing_loop(){
    generic_initialization();
    //set_folder_name("input_folder");
    set_number_of_inputs(60000, 10000);
    load_test_set();

    int n_files=read_filenames_from_directory(valid_names);
    int index=rand()%number_of_test_images;
    int submenu=1;//index to tell in which windows i need to go when menu is closed
    Metrics resulting_metrics;

    Boxes graphic_elements=initialize_testing_boxes();
    struct Box *boxes=graphic_elements.boxes;
    int num_boxes=graphic_elements.num_boxes;
    struct Box test_results_area=boxes[7];
    struct Box showing_area=boxes[6];
    struct Box result_box=boxes[5];
    struct Box start_test_box=boxes[4];
    struct Box NN_selection_result=boxes[3];
    struct Box NN_selection=boxes[2];
    struct Box submit_button=boxes[1];
    struct Box next_button=boxes[0];
    int drawing_dim=showing_area.x2-showing_area.x1;

    struct Box input_box = create_default_box();
    init_input_box(&input_box);

    while(1){
        al_wait_for_event(queue, &event);

        switch(event.type)
        {
            case ALLEGRO_EVENT_TIMER:
                if(key[ALLEGRO_KEY_ESCAPE]){
                    submenu=-1;
                    done = true;
                }
                if(is_typing && key[ALLEGRO_KEY_ENTER]){
                    //int i_file=read_filename(valid_names, n_files,typed_text);
                    load_saved_model(n_files,&boxes[3]);
                    //printf("2) %s \n", boxes[3].text);
                }
                for(int i = 0; i < ALLEGRO_KEY_MAX; i++)
                    key[i] &= KEY_SEEN;
                //redraw = true;
                break;
            case ALLEGRO_EVENT_MOUSE_BUTTON_DOWN:
                if (is_point_inside_button(event.mouse.x, event.mouse.y, start_test_box)) {
                    redraw = true;
                    float p_test[number_of_test_images][10];
                    inference_on_set(testing_images,p_test, number_of_test_images);
                    resulting_metrics=compute_metrics(p_test,testing_labels, number_of_test_images);
                }
                if (is_point_inside_button(event.mouse.x, event.mouse.y, next_button)) {
                    // Button was clicked, perform an action
                    index=rand()%number_of_test_images;
                    redraw = true;
                }
                if (is_point_inside_button(event.mouse.x, event.mouse.y, submit_button)) {
                    // Button was clicked, perform an action
                    inference = true;
                    redraw=true;
                }
                if (is_point_inside_button(event.mouse.x, event.mouse.y, NN_selection)) {
                    is_typing=true;
                    char temporary[10000]="Insert the number of the file in the list \n (first is 0) to load the neural network:\n";
                    for(int i=0;i<n_files;i++){
                        strcat(temporary,valid_names[i]);
                        strcat(temporary,"\n");
                    }
                    strcpy(text_to_show,temporary);
                    redraw = true;
                }
                break;
            case ALLEGRO_EVENT_KEY_DOWN:
                key[event.keyboard.keycode] = KEY_SEEN | KEY_RELEASED;
                break;
            case ALLEGRO_EVENT_KEY_UP:
                key[event.keyboard.keycode] &= KEY_RELEASED;
                break;
            case ALLEGRO_EVENT_KEY_CHAR:
                if(is_typing){
                    get_keyboard_input();
                }
                break;
            case ALLEGRO_EVENT_DISPLAY_CLOSE:
                done = true;
                break;
        }

        if(done)
            break;

        if(redraw && al_is_event_queue_empty(queue)){
            //printf("aaaaah\n");
            if (inference){
                //printf("here 0\n");
                forward_propagation(testing_images[index]);
                int guess=get_best_class(outputs[number_of_layers-1]);
                //printf("guess=%d\n",guess);
                inference=false;
                sprintf(boxes[5].text,"RESULT: %d",guess);
            }
            al_clear_to_color(al_map_rgb(0, 0, 0));
            al_draw_textf(font, al_map_rgb(255, 255, 255), width/2, 10, 1, "TESTING");
            draw_box_elements(boxes,num_boxes);
            display_next_image(drawing_dim,showing_area.x1,showing_area.y1, index);//extract one image at random and put as bitmap in the drawing region
            //extract one image at random and put as bitmap in the drawing region
            plot_metrics(test_results_area,resulting_metrics);
            if(is_typing){
                show_typing_interface(input_box,text_to_show,typed_text);
            }
            al_flip_display();

            redraw = false;
        }
    }
    destroy_graphics();
    switch (submenu)
    {
    case -1:// if escape is pressed i return to the menu
        menu_loop();
        break;
    default:
        break;
    }
}

void place_object(int x,int y,float x_ratio, float y_ratio){
    //you fix the postion of the upper left corner of the object relative
    //to the upper left corner of the window using the fraction of screen as coordinate
    x=width*x_ratio;
    y=height*y_ratio;
}

void place_object_grid(int *x,int *y,int *w, int *h, int x_grid, int y_grid, int w_grid, int h_grid, float grid_size_x, float grid_size_y){
    //grid size = 0.3 means i divide in thirds
    float x_size=width*grid_size_x;
    float y_size=height*grid_size_y;
    *x=x_size*x_grid;
    *y=y_size*y_grid;
    *w=x_size*w_grid;
    *h=y_size*h_grid;
}

void training_loop(){
    generic_initialization();

    //i initialize the boolean variables that describe the state of execution
    int submenu=0;//index to tell in which windows i need to go when menu is closed
    bool is_training=false;
    bool is_saveing=false;
    bool has_trained=0;
    int input_button=-1;
    float loss_train[1000];
    float loss_val[1000];
    float epochs[1000];
    int n_epochs=0;
    Metrics resulting_metrics;

    Boxes graphic_elements=initialize_training_input_boxes();
    struct Box *input_boxes=graphic_elements.boxes;
    int num_input_boxes=graphic_elements.num_boxes;
    int num_input_fields=num_input_boxes-1;
    struct Box n_layers_button=input_boxes[0];
    struct Box neurons_per_layer_button=input_boxes[1];
    struct Box activation_button=input_boxes[2];
    struct Box train_val_split_button=input_boxes[3];
    struct Box n_train_test_button=input_boxes[4];
    struct Box optimization_button=input_boxes[5];
    struct Box loss_button=input_boxes[6];
    struct Box learning_rate_button=input_boxes[7];
    struct Box momentum_button=input_boxes[8];
    struct Box shuffling_button=input_boxes[9];
    struct Box batch_size_button=input_boxes[10];
    struct Box epochs_button=input_boxes[11];
    struct Box reset_template_button=input_boxes[12];
    //printf("num_input_boxes=%d\n",num_input_boxes);

    //int num_input_fields=12;
    Boxes graphic_elements_2=initialize_training_input_fields(num_input_fields);
    struct Box *input_fields=graphic_elements_2.boxes;

    Boxes graphic_elements_3=initialize_training_training_boxes();
    struct Box *training_boxes=graphic_elements_3.boxes;
    int num_training_boxes=graphic_elements_3.num_boxes;
    struct Box start_button=training_boxes[0];
    struct Box stop_button=training_boxes[1];
    struct Box plot_region=training_boxes[2];
    struct Box results_region=training_boxes[3];
    struct Box save_button=training_boxes[4];

    //this defines the user interface that appears when i ask for input to the user
    struct Box input_box = create_default_box();
    init_input_box(&input_box);

    //When a button is pressed the cmd shows a prompt to insert the corresponding value
    //this stores the prompt
    char output_cmd[13][1000]={"Insert the number of layers\n",
    "Insert the number of neurons per layer\n using the format n1 n2 n3 .. nk for k layers\n",
    "Insert the activation function\n 0=sigmoid\n 1=ReLu\n",
    "Determine the fraction of the training set to use for validation\n",
    "Determine the number of training (max 60000) and test examples (max 10000). Write: n_train n_test\n",
    "Choose the optimization algorithm\n 0=SGD\n 1=SGD with momentum\n 2=Nesterov\n",
    "Choose the loss function\n 0=Cross entropy(log likelihood)\n 1=MSE\n",
    "Insert the learning rate\n",
    "Insert the momentum\n",
    "Choose if you want to shuffle the training set\n 0=No\n 1=Yes\n",
    "Insert the batch size\n",
    "Insert the number of training epochs\n",
    "You can choose some predefined configurations\n" 
    "0) Default -> 1 layer, 64 neurons, sigmoid, 0.1 validation, 60000 training,\n10000 test, Nesterov, log-likelihood, lr 0.1, p 0.9, shuffle, 32 batch, 20 epochs\n"
    "\n1) Small network -> "
    "\n2)Large network -> \n"};
    //These arrays enabels to deal with the logic of each button avoiding the need
    //to implement a different logic for each button
    int n_inputs[13]={1,-1,1,1,2,1,1,1,1,1,1,1,1};//is the number of inputs expected for the prompt of each buttton
    //-1 means that the number of inputs depends on other variables (eg. neurons per layer depends on n_layers)
    float min_input[13]={0  ,1    ,0,0.00001,1    ,0,0,0.00000001,0.0000001,0,1    ,1    ,0};
    float max_input[13]={100,28*28,1,0.99999,60000,2,1,100000    ,100000   ,1,60000,10000,2};//i put the same max and min for n_train and n_val -> fix
    int layers[max_hidden_layers]={64};//stores the neurons per layer
    float parameters[12]={1,0,0.1,60000,10000,2,0,0.1,0.9,1,32,20};//stores the input gathered by the user
    //they are ordered sequentially: n_inputs[0] values for first input, n_inputs[1] values for second input, ...
    int index_to_par[13];//stores the index of the first element of the input in the parameters array
    //eg the third button expects two inputs -> index_to_par[3] gives the position of the first input
    //in the parameters array
    int s=0;//this builds the index_to_par array
    for (int i=0;i<13;i++){
        if(n_inputs[i]==-1){
            index_to_par[i]=-1;
        }
        else{
            index_to_par[i]=s;
            s+=n_inputs[i];
        }
    }//should be {0,-1,1,2,3,5,6,7,8,9,10,11,-1};
    int type_output[13] ={0,0,2,0,0,3,2,0,0,2,0,0,3};//0=numeric, n>0=number of string
    //Once we insert the input for a button we want to display the selected option
    //if the input was numeric we print the number/s
    //if the input was a selection we print the corresponding string 
    //type output tells how many possible strings we should be able to display for each button
    char output_text[13][100]={"sigmoid","ReLu","SGD","SGD with momentum","Nesterov",
    "Cross entropy","MSE","no","yes"}; //the strings that can be displayed. The logic of the ordering is
    //the same as for the parameters array
    int index_to_text[13];//give the index of the first possible string to display in the output text array
    // for each button. Again the logic is that of the parameters array and the index_to_par array
    s=0;//this builds the index_to_text array
    for (int i=0;i<13;i++){
        if(type_output[i]==0){
            index_to_text[i]=-1;
        }
        else{
            index_to_text[i]=s;
            s+=type_output[i];
        }
    }//should be {0,-1,1,2,3,5,6,7,8,9,10,11,-1};

    //i define some default configurations that can be selected with the template reset button
    float default_configs[3][12] = {
        {1,0,0.1,60000,10000,2,0,0.1,0.9,1,32,20},
        {2,0,0.1,60000,10000,2,0,0.1,0.9,1,32,20},
        {1,1,0.1,60000,10000,2,0,0.1,0.9,1,32,20}//with relu
    };
    int default_n_layers[3][max_hidden_layers]=
    {
        {64},
        {32,32},
        {128}
    };

    while(1){
        al_wait_for_event(queue, &event);
        switch(event.type)
        {
            case ALLEGRO_EVENT_TIMER:
                if(key[ALLEGRO_KEY_ESCAPE]){
                    submenu=-1;
                    done = true;
                }
                if(is_typing && key[ALLEGRO_KEY_ENTER]){
                    //sleep(2);
                    if(!is_saveing){
                        int reading_result=0;
                        float x[10];
                        if(n_inputs[input_button]!=-1){
                            reading_result=read_input(x, min_input[input_button], max_input[input_button], 
                            n_inputs[input_button] ,typed_text);
                        }
                        else{
                            reading_result=read_input(x, min_input[input_button], max_input[input_button]
                            ,parameters[0],typed_text);//se ho cliccato il bottone
                            //per fissare i neuroni per layer mi aspetto n_layers=parameters[0] valori
                        }
                        //printf("%f %f %d %s\n",min_input[input_button], max_input[input_button],n_inputs[input_button],typed_text);
                        if(reading_result==0){
                            memset(typed_text, 0, sizeof(typed_text));
                            typed_text[0]='-';
                            typed_text[1]='>';
                        }
                        else{
                            if (input_button==num_input_boxes-1) {
                                int p;
                                p=x[0];
                                for (int i=0;i<num_input_boxes-1;i++){
                                    parameters[i]=default_configs[p][i];
                                }
                                for (int i=0;i<parameters[0];i++){
                                    layers[i]=default_n_layers[p][i];
                                }
                            }
                            else{
                                if(n_inputs[input_button]!=-1){
                                    for(int j=0;j<n_inputs[input_button];j++){
                                        parameters[index_to_par[input_button]+j]=x[j];
                                    }
                                }
                                else{
                                    for(int j=0;j<parameters[0];j++){
                                        layers[j]=x[j];
                                    }
                                }
                            }
                            is_typing=false;
                            memset(typed_text, 0, sizeof(typed_text));
                            typed_text[0]='-';
                            typed_text[1]='>';
                            memset(text_to_show, 0, sizeof(text_to_show));
                            //printf("typed text=%s\n",typed_text);
                        }
                    }
                    else{
                        for(int j=2;j<strlen(typed_text);j++){
                            result_string[j-2]=typed_text[j];
                        }
                        //set_folder_name(".");
                        save_NN(result_string);
                        is_saveing=false;
                        is_typing=false;
                        memset(typed_text, 0, sizeof(typed_text));
                        typed_text[0]='-';
                        typed_text[1]='>';
                        memset(text_to_show, 0, sizeof(text_to_show));
                        memset(result_string, 0, sizeof(result_string));
                    }
                    redraw=true;
                    //printf("doneeeee\n");
                }
                for(int i = 0; i < ALLEGRO_KEY_MAX; i++)
                    key[i] &= KEY_SEEN;
                //redraw = true;
                break;
            case ALLEGRO_EVENT_MOUSE_BUTTON_DOWN:
                if(!is_training){
                    if (is_point_inside_button(event.mouse.x, event.mouse.y, save_button)) {
                        is_typing=true;
                        is_saveing=true;
                        strcpy(text_to_show,"Insert the name of the file \n to save the neural network");
                        redraw = true;
                    }
                    for (int i=0;i<num_input_boxes;i++){
                        if (is_point_inside_button(event.mouse.x, event.mouse.y, input_boxes[i])) {
                            is_typing=true;
                            strcpy(text_to_show,output_cmd[i]);
                            redraw = true;
                            input_button=i;
                        }
                    }
                    if (is_point_inside_button(event.mouse.x, event.mouse.y, start_button)) {
                        is_training=true;
                        //printf("is_training=%d\n",is_training);
                        redraw = true;
                        //i set the neural network structure
                        //set at training start
                        //set_folder_name("input_folder");
                        set_number_of_inputs((int)parameters[3], (int)parameters[4]);
                        load_training_set();
                        define_training_parameters((int)parameters[11],parameters[7], (int)parameters[6], (int)parameters[9], 
                        0.001, (int)parameters[5], parameters[8]);
                        //i fix the minimum error to 0.0001
                        define_network_structure(layers, (int)parameters[0], (int)parameters[1], 1);//i fix initialization to gaussian
                        set_train_val((int)parameters[10], parameters[2]);
                        split_data();
                        weight_initialization();
                    }
                }
                else if (is_point_inside_button(event.mouse.x, event.mouse.y, stop_button)) {
                    is_training=false;
                    has_trained=true;
                }
                break;
            case ALLEGRO_EVENT_KEY_DOWN:
                key[event.keyboard.keycode] = KEY_SEEN | KEY_RELEASED;
                break;
            case ALLEGRO_EVENT_KEY_UP:
                key[event.keyboard.keycode] &= KEY_RELEASED;
                break;
            case ALLEGRO_EVENT_KEY_CHAR:
                if(is_typing){
                    get_keyboard_input();
                }
                break;
            case ALLEGRO_EVENT_DISPLAY_CLOSE:
                done = true;
                break;
        }
        if(done)
            break;
        //printf("%d",is_training);
        if(redraw && al_is_event_queue_empty(queue)){
            if(!is_training){
                for (int i=0;i<num_input_fields;i++){
                    if(type_output[i]==0){
                        if(n_inputs[i]==-1){
                            float float_layers[100];
                            //printf("n_layers=%d\n",n_layers);
                            for(int j=0;j<parameters[0];j++){
                                //printf("%d ",layers[j]);
                                float_layers[j]=(float)layers[j];
                            }
                            print_numbers(result_string, float_layers, parameters[0]);
                            sprintf(input_fields[i].text,"%s",result_string);
                            memset(result_string, 0, sizeof(result_string));
                        }
                        else{
                            float float_params[100];
                            for(int j=0;j<n_inputs[i];j++){
                                float_params[j]=parameters[index_to_par[i]+j];
                                //printf("%f ",float_params[j]);
                            }
                            print_numbers(result_string, float_params, n_inputs[i]);
                            //printf("%s\n",result_string);
                            //printf("\n");
                            sprintf(input_fields[i].text,"%s",result_string);
                            memset(result_string, 0, sizeof(result_string));
                        }
                    }
                    else{
                        sprintf(input_fields[i].text,"%s",output_text[index_to_text[i]+(int)parameters[index_to_par[i]]]);
                        //copilot ha capito da solo questa parte, incredibile
                        //essenzialmente per i bottoni con scelta categorica seleziono la stringa da output text
                        //l'indice è quello della prima stringa in output_text corrispondente al bottone + il valore della
                        //scelta (da 0 a scelte possibili-1)
                    }
                }
            }
            al_clear_to_color(al_map_rgb(0, 0, 0));
            al_draw_textf(font, al_map_rgb(255, 255, 255), width/2, 10, 1, "TRAINING");
            draw_box_elements(input_boxes,num_input_boxes);
            draw_box_elements(input_fields,num_input_fields);
            draw_box_elements(training_boxes,num_training_boxes);
            if(has_trained || is_training){
                draw_plot(plot_region,loss_train,epochs,n_epochs,al_map_rgb(0, 0, 255));
                draw_plot(plot_region,loss_val,epochs,n_epochs,al_map_rgb(255, 0, 0));
            }
            if(has_trained){
                plot_metrics(results_region,resulting_metrics);
            }
            if(is_typing){
                show_typing_interface(input_box,text_to_show,typed_text);
            }
            al_flip_display();
            redraw = false;
            printf("is drawing\n");
        }
        if(is_training && al_is_event_queue_empty(queue)){
            redraw=true;
            printf("learning epoch %d in progress\n",n_epochs);
            learn_epoch();
            epochs[n_epochs]=n_epochs;
            loss_train[n_epochs]=error_on_epoch;
            printf("loss on training set=%f\n",loss_train[n_epochs]);
            float probabilities[number_of_val_images][10];
            inference_on_set(validation_images,probabilities, number_of_val_images);
            loss_val[n_epochs]=loss_on_set(validation_labels,probabilities,number_of_val_images,type_of_loss);
            n_epochs++;
            if(n_epochs>=number_of_epochs){
                is_training=false;
                has_trained=true;
                float p_test[number_of_val_images][10];
                inference_on_set(validation_images,p_test, number_of_val_images);
                resulting_metrics=compute_metrics(probabilities,validation_labels, number_of_val_images);

            }
            printf("loss on validation set=%f\n",loss_val[n_epochs-1]);
            fflush(stdout);
        }
    }
    destroy_graphics();
    switch (submenu)
    {
    case -1:// if escape is pressed i return to the menu
        menu_loop();
        break;
    default:
        break;
    }
}

/*int read_pixel_from_display(ALLEGRO_BITMAP *backbuffer,int x, int y) {
    ALLEGRO_COLOR color = al_get_pixel(backbuffer, x, y); // Get the pixel color
    // Extract and print RGBA values
    float r, g, b, a;
    al_unmap_rgba_f(color, &r, &g, &b, &a);
    //printf("Pixel at (%d, %d): R=%.2f, G=%.2f, B=%.2f, A=%.2f\n", x, y, r, g, b, a);
    if(r==0 && g==0 && b==0){
        return 1;
    }
    else{
        return 0;
    }
}
*/

int process_drawing_region(int dim, int **drawing_matrix){
    //i determine the maximum value of the drawing matrix
    /*int max=drawing_matrix[0][0];
    for (int i=0;i<dim;i++){
        for (int j=0;j<dim;j++){
            if(drawing_matrix[i][j]>max){
                max=drawing_matrix[i][j];
            }
        }
    }*/
    // i scale the image down to 28x28 by averaging with a k=n-27 grid
    //n=dim
    int n=dim;
    int k=n-27;
    int image[28][28];
    /* Convolution makes no sense since there is too much white
    for(int sup_i=0;sup_i<n-k+1;sup_i++){
        for(int sup_j=0;sup_j<n-k+1;sup_j++){
            float sum=0;
            for (int i=0+sup_i;i<k+sup_i;i++){
                for (int j=0+sup_j;j<k+sup_j;j++){
                    sum+=drawing_matrix[i][j]/max*255;//i scale the values to range 0-255
                }
            }
            sum/=(float)(k*k);
            image[sup_i][sup_j]=sum;
        }
    }*/
    //prendo ogni pixel nell'immagine di arrivo. Calcolo la posizione del corrispondente
    //pixel nell'immagine di partenza. Faccio la media dei pixel in un intorno di quello di partenenza
    //o prendo il massimo
    int l_sup=3;//-> considero un qudrato di lato 5 attorno al pixel di arrivo per fare la media
    for(int i=0;i<28;i++){
        for(int j=0;j<28;j++){
            image[i][j]=0;
            int i_sup=(int)((float)i/28.0*(float)n);
            int j_sup=(int)((float)j/28.0*(float)n);
            for(int k=i_sup-l_sup;k<i_sup+l_sup+1;k++){
                for(int l=j_sup-l_sup;l<j_sup+l_sup+1;l++){
                    if(k>=0 && k<n && l>=0 && l<n){
                        /*if(drawing_matrix[k][l]>image[i][j]){//i take the max value
                            image[i][j]=drawing_matrix[k][l];
                        }*/
                        image[i][j]+=(int)((float)(drawing_matrix[k][l]/(float)((2*l_sup+1)*(2*l_sup+1)))); //i take the average
                    }
                }
            }
        }
    }
    forward_propagation(image);
    int guess=get_best_class(outputs[number_of_layers-1]);
    //print_image(image);
    //print_image(testing_images[0]);
    return guess;
}

void upsample_image(int image[28][28], int dim, int upsampled_image[dim][dim]){
    int l_inf=1;//-> considero un qudrato di lato 5 attorno al pixel di arrivo per fare la media
    for(int i=0;i<dim;i++){
        for(int j=0;j<dim;j++){
            upsampled_image[i][j]=0;
            int i_inf=(int)((float)i/(float)dim*28.0);
            int j_inf=(int)((float)j/(float)dim*28.0);
            for(int k=i_inf-l_inf;k<i_inf+l_inf;k++){
                for(int l=j_inf-l_inf;l<j_inf+l_inf;l++){
                    if(k>=0 && k<28 && l>=0 && l<28){
                        if(image[k][l]>upsampled_image[i][j]){//i take the max value
                            upsampled_image[i][j]=image[k][l];
                        }
                        //upsampled_image[i][j]+=(int)((float)(image[k][l]/(l_inf*l_inf+1))); //i take the average
                    }
                }
            }
        }
    }
    /*for(int i=0;i<dim;i++){
        for(int j=0;j<dim;j++){
            printf("%d ",upsampled_image[i][j]);
        }
        printf("\n");
    }*/
}

ALLEGRO_BITMAP *create_bitmap_from_image(int dim, int image[dim][dim]){
    ALLEGRO_BITMAP *bitmap = al_create_bitmap(dim, dim);
    al_set_target_bitmap(bitmap);

    // Clear the bitmap with a color
    al_clear_to_color(al_map_rgb(255, 0, 0));
    // Draw a circle on the bitmap
    //al_draw_circle(100, 100, 50, al_map_rgb(0, 255, 0), 5);

    // Reset the target bitmap to the backbuffer (display)
    ALLEGRO_LOCKED_REGION *locked = al_lock_bitmap(bitmap, ALLEGRO_PIXEL_FORMAT_ANY, ALLEGRO_LOCK_WRITEONLY);
    if (locked) {
        unsigned char *data = (unsigned char *)locked->data;
        int pitch = locked->pitch;
        int pixel_size = al_get_pixel_size(locked->format);
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                int reversed_pixel = 255 - image[i][j];
                // Calculate the position in the locked region's data
                unsigned char *pixel = data + i * pitch + j * pixel_size;
                pixel[0] = reversed_pixel; // Blue component
                pixel[1] = reversed_pixel;   // Green component
                pixel[2] = reversed_pixel;   // Red component
                if (pixel_size == 4) {
                    pixel[3] = 255; // Alpha component (if present)
                }
            }
        }
    }
    // Get the pixel data

    // Unlock the bitmap
    al_unlock_bitmap(bitmap);
    al_set_target_backbuffer(disp);
    return bitmap;
}

void display_next_image(int dim, int x, int y,int index){
    //i take the next image from the test set
    /*printf("This will be printed before the error occurs.\n");
    fflush(stdout);*/
    //printf("%d\n",index);
    int upsampled_image[dim][dim];
    /*int **upsampled_image=(int **)malloc(dim * sizeof(int *));
    for(int i = 0; i < dim; i++) {
        upsampled_image[i] = (int *)malloc(dim * sizeof(int));
    }*/
    upsample_image(testing_images[index],dim,upsampled_image);
    ALLEGRO_BITMAP *bitmap=create_bitmap_from_image(dim,upsampled_image);
    //al_set_target_bitmap(al_get_backbuffer(disp));
    al_draw_bitmap(bitmap, x, y , 0);

    /*for (int i = 0; i < dim; i++) {
        free(upsampled_image[i]);
    }
    free(upsampled_image);*/
}

void draw_plot(struct Box container , float *y, float *x, int dim, ALLEGRO_COLOR color){
    float spacing=10;
    int delta_epochs=10;
    float thickness=2;
    float label_spacing=5;
    float tick_dimension=1;
    if(x_pos_sup==-1){
        x_pos_sup=container.x2-spacing;
        x_pos_inf=container.x1+spacing;
        y_pos_sup=container.y1+spacing;
        y_pos_inf=container.y2-spacing;
        y_lim_sup=y[0];
    }
    //calcolo massimo minimo etc
    float max_x=x[0];
    float max_y=y[0];
    float min_y=y[0];
    for (int i=0;i<dim;i++){
        if(x[i]>max_x){
            max_x=x[i];
        }
        if(y[i]>max_y){
            max_y=y[i];
        }
        if(y[i]<min_y){
            min_y=y[i];
        }
    }
    if(max_x>x_lim_sup){
        x_lim_sup+=delta_epochs;
    }
    if(max_y>y_lim_sup){
        y_lim_sup=max_y;
    }
    float scale_y = (y_lim_inf-y_lim_sup)/(y_pos_inf-y_pos_sup); // Scale factor for better visualization
    float scale_x = (x_lim_sup-x_lim_inf)/(x_pos_sup-x_pos_inf); // Scale factor for better visualization
    //scale=dvalue/dposition_in_box
    for (int i = 1; i < dim; i++) {
        float x1=(x[i-1]-x_lim_inf)/scale_x+x_pos_inf;
        float x2=(x[i]-x_lim_inf)/scale_x+x_pos_inf;
        float y1=(y[i-1]-y_lim_inf)/scale_y+y_pos_inf;
        float y2=(y[i]-y_lim_inf)/scale_y+y_pos_inf;
        // Draw a line segment connecting (x1, y1) and (x2, y2)
        al_draw_line(x1, y1, x2, y2, color, thickness);
    }
    al_draw_line(x_pos_sup, y_pos_inf, x_pos_inf, y_pos_inf, color, thickness/2);
    al_draw_line(x_pos_inf, y_pos_inf, x_pos_inf, y_pos_sup, color, thickness/2);
    al_draw_line(x_pos_inf-tick_dimension, y_pos_inf, x_pos_inf+tick_dimension, y_pos_inf, color, thickness/2);
    al_draw_textf(font, al_map_rgb(0,0,0), x_pos_inf-label_spacing, y_pos_inf, 0, "%.3f" , y_lim_inf);
    al_draw_line(x_pos_inf-tick_dimension, y_pos_sup, x_pos_inf+tick_dimension, y_pos_sup, color, thickness/2);
    al_draw_textf(font, al_map_rgb(0,0,0), x_pos_inf-label_spacing, y_pos_sup, 0, "%.3f" , y_lim_sup);
    for(int i=0;i<x_lim_sup;i++){
        float x_pos=(x[i]-x_lim_inf)/scale_x+x_pos_inf;
        al_draw_line(x_pos, y_pos_inf-tick_dimension, x_pos, y_pos_inf+tick_dimension, color, thickness/2);
        al_draw_textf(font, al_map_rgb(0,0,0), x_pos, y_pos_inf+label_spacing, 0, "%d" , (int)x[i]);
    }
}

void plot_metrics(struct Box results_region,Metrics resulting_metrics){
    float border=5;
    float spacing=(results_region.x2-results_region.x1)/4.0;
    float x_coordinates[4];
    float y_spacing=(results_region.y2-results_region.y1)/11.0;
    float y_coordinates[n_classes+1];
    float global_metrics[5]={resulting_metrics.overall_accuracy,resulting_metrics.macro_precision,
    resulting_metrics.macro_recall,resulting_metrics.micro_recall,resulting_metrics.micro_precision};
    char global_metrics_names[5][20]={ "ACC","MP","MR","mr","mp"};
    char column_names[4][20]={"Class","Precision","Recall","Global metrics"};
    for(int i=0;i<4;i++){
        x_coordinates[i]=results_region.x1+border+spacing*i;
    }
    for(int i=0;i<n_classes+1;i++){
        y_coordinates[i]=results_region.y1+border+y_spacing*i;
    }
    for(int i=0;i<4;i++){
        al_draw_textf(font, al_map_rgb(0,0,0), x_coordinates[i], y_coordinates[0], 0, "%s" , column_names[i]);
    }
    for(int i=0;i<n_classes;i++){
        al_draw_textf(font, al_map_rgb(0,0,0), x_coordinates[0], y_coordinates[i+1], 0, "%d" , i);
        al_draw_textf(font, al_map_rgb(0,0,0), x_coordinates[1], y_coordinates[i+1], 0, "%.1f" , resulting_metrics.precisions[i]);
        al_draw_textf(font, al_map_rgb(0,0,0), x_coordinates[2], y_coordinates[i+1], 0, "%.1f" , resulting_metrics.recalls[i]);
        if(i<5){
            al_draw_textf(font, al_map_rgb(0,0,0), x_coordinates[3], y_coordinates[i+1], 0, 
            "%s: %.2f" , global_metrics_names[i],global_metrics[i]);
        }
    }
}

void show_typing_interface(struct Box input_box, char *text_to_show, char *typed_text){
    al_draw_filled_rectangle(input_box.x1, input_box.y1, input_box.x2, input_box.y2, input_box.color);
    float center_x=(input_box.x1+input_box.x2)/2;
    float center_y=(input_box.y1+input_box.y2)/2;
    float height_box=input_box.y2-input_box.y1;
    float spacing=height_box/10.0;
    //printf("text to show=%s\n",text_to_show);
    draw_multiline_text(input_box.text_position,input_box.text_color, font_roboto, input_box.x1+spacing , input_box.y1+spacing,height_box/10.0, text_to_show);
    //al_draw_textf(font, input_box.text_color, input_box.x1+10 , input_box.y1, input_box.text_position, "%s" , text_to_show);
    al_draw_textf(font_roboto, input_box.text_color, input_box.x1+spacing, input_box.y2-height_box/10.0, input_box.text_position, "%s" , typed_text);
}

void load_saved_model(int n_files,struct Box *box){
    float result_input[1];
    read_input(result_input, 0, n_files, 1,typed_text);
    int i_file=(int)result_input[0];
    //set_folder_name(".");
    //printf("%d\n",i_file);
    printf("Loading model %s\n",valid_names[i_file]);
    sprintf(box -> text,"%s",valid_names[i_file]);//boxes[3] is the result box
    //for the NN selection
    load_model(valid_names[i_file]);
    is_typing=false;
    memset(typed_text, 0, sizeof(typed_text));
    typed_text[0]='-';
    typed_text[1]='>';
    memset(text_to_show, 0, sizeof(text_to_show));
    redraw=true;
    //printf("1)%s \n", box.text);
}

void get_keyboard_input(){
    if (event.keyboard.unichar >= 32 && event.keyboard.unichar <= 126) {
        if (strlen(typed_text) < sizeof(typed_text) - 1) {
            int len = strlen(typed_text);
            typed_text[len] = (char)event.keyboard.unichar;
            typed_text[len + 1] = '\0';
            redraw = true;
        }
    } else if (event.keyboard.keycode == ALLEGRO_KEY_BACKSPACE) {
        int len = strlen(typed_text);
        if (len > 0) {
            typed_text[len - 1] = '\0';
            redraw = true;
        }
    }
}

void draw_box_elements(struct Box *boxes,int num_boxes){
    for (int i=0;i<num_boxes;i++){
        al_draw_filled_rectangle(boxes[i].x1, boxes[i].y1, boxes[i].x2, boxes[i].y2, boxes[i].color);
        float center_x=(boxes[i].x1+boxes[i].x2)/2;
        float center_y=(boxes[i].y1+boxes[i].y2)/2;
        al_draw_textf(font, boxes[i].text_color, center_x, center_y, boxes[i].text_position, "%s" ,boxes[i].text);
        //printf("%s\n" ,boxes[i].text);
        //printf("%s\n" ,result_box.text);
    }
}

void print_network_characteristics(){
    printf("number of epochs=%d\n",number_of_epochs);
    printf("number of training examples=%d\n",number_of_train_images);
    printf("number of validation examples=%d\n",number_of_val_images);
    printf("lr=%f\n",learning_rate);
    printf("momentum=%f\n",momentum);
    printf("batch size=%d\n",minibatch_size);
    printf("n_layers=%d\n",number_of_layers);
    for (int i=0;i<number_of_layers;i++){
        printf("neurons in layer %d=%d\n",i,neurons_per_layer[i]);
    }
    printf("activation function=%d\n",type_of_activation);
    printf("initialization=%d\n",type_of_initialization);
    printf("optimization=%d\n",type_of_optimization);
    printf("sample weight = %f\n",weights[0][0][0]);
    printf("shuffling = %d\n",type_of_shuffling);
    printf("loss function = %d\n",type_of_loss);
}

void init_input_box(struct Box *input_box){
    input_box->color=al_map_rgb(255, 255, 255);
    int x,y,w,h;
    place_object_grid(&x,&y,&w,&h,1,1,8,8,1/10.0, 1/10.0);
    input_box->x1=x;
    input_box->x2=x+w;
    input_box->y1=y;
    input_box->y2=y+h;
    input_box->text_color=al_map_rgb(0, 0, 0);
    input_box->text_position=0;
}

Boxes initialize_testing_boxes(){
    int x_blocks=width/6; //i divide in fifths and the buttons take the central fifth
    int y_blocks=height/6; //the three boxes will go in 
    int x_blocks_fine=width/12; //i divide in fifths and the buttons take the central fifth
    int y_blocks_fine=height/12; //the three boxes will go in 
    struct Box next_button = create_default_box();
    struct Box submit_button = create_default_box();
    struct Box NN_selection = create_default_box();
    struct Box NN_selection_result = create_default_box();
    struct Box start_test_box = create_default_box();
    struct Box result_box = create_default_box();
    struct Box showing_area = create_default_box();
    struct Box test_results_area = create_default_box();
    //i define the drawing region
    showing_area.color=al_map_rgb(255, 255, 255);
    showing_area.x1=x_blocks_fine*7;
    showing_area.x2=x_blocks_fine*10;
    showing_area.y1=y_blocks_fine;
    showing_area.y2=y_blocks+showing_area.x2-showing_area.x1;
    //I define the are where i will plot the results
    test_results_area.color=al_map_rgb(255, 255, 255);
    test_results_area.x1=x_blocks_fine*1;
    test_results_area.x2=x_blocks_fine*5;
    test_results_area.y1=y_blocks_fine*3;
    test_results_area.y2=y_blocks_fine*11;
    strcpy(test_results_area.text,"TEST RESULTS");
    //i define the next button
    next_button.color=al_map_rgba(0, 255, 0,0.6);
    next_button.x1=x_blocks_fine*9;
    next_button.x2=x_blocks_fine*10;
    next_button.y1=showing_area.y2+y_blocks_fine;
    next_button.y2=next_button.y1+y_blocks_fine;
    strcpy(next_button.text,"NEXT");
    //i define the submit button
    submit_button.color=al_map_rgba(255, 0, 0,0.6);
    submit_button.x1=x_blocks_fine*10;
    submit_button.x2=x_blocks_fine*11;
    submit_button.y1=showing_area.y2+y_blocks_fine;
    submit_button.y2=next_button.y1+y_blocks_fine;
    strcpy(submit_button.text,"SUBMIT");
    //i define the selection button
    NN_selection.color=al_map_rgba(0, 255, 0,1);
    NN_selection.x1=test_results_area.x1;
    NN_selection.x2=test_results_area.x1+2*x_blocks_fine;
    NN_selection.y1=y_blocks_fine;
    NN_selection.y2=y_blocks_fine*2;
    NN_selection_result.text_color=al_map_rgb(0, 0, 0);
    strcpy(NN_selection.text,"SELECT NN");
    //i define the selection button
    NN_selection_result.color=al_map_rgb(255, 255, 255);
    NN_selection_result.x1=NN_selection.x2;
    NN_selection_result.x2=NN_selection_result.x1+2*x_blocks_fine;
    NN_selection_result.y1=y_blocks_fine;
    NN_selection_result.y2=y_blocks_fine*2;
    NN_selection_result.text_color=al_map_rgb(0, 0, 0);
    strcpy(NN_selection_result.text," ");
    //i define the button to start execution of test on test set
    start_test_box.color=al_map_rgba(0, 255, 0,1);
    start_test_box.x1=x_blocks_fine*6;
    start_test_box.x2=submit_button.x2;
    start_test_box.y1=next_button.y2+y_blocks_fine;
    start_test_box.y2=start_test_box.y1+y_blocks_fine;
    strcpy(start_test_box.text,"START TEST");
    //i define the result region
    result_box.color=al_map_rgb(255, 255, 255);
    result_box.x1=x_blocks_fine*6;
    result_box.x2=x_blocks_fine*8;
    result_box.y1=showing_area.y2+y_blocks_fine;
    result_box.y2=result_box.y1+y_blocks_fine;
    result_box.text_position=2;//left aligned
    result_box.text_color=al_map_rgb(0, 0, 0);
    strcpy(result_box.text,"RESULT:");
    int num_boxes=8;
    struct Box boxes[MAX_BUTTONS]={next_button,submit_button,NN_selection,NN_selection_result,
    start_test_box,result_box,showing_area,test_results_area};
    Boxes result;
    result.num_boxes=num_boxes;
    for(int i=0;i<num_boxes;i++){
        result.boxes[i]=boxes[i];
    }
    return result;
}

Boxes initialize_interactive_boxes(){
    int x_blocks=width/6; //i divide in fifths and the buttons take the central fifth
    int y_blocks=height/6; //the three boxes will go in 
    int x_blocks_fine=width/12; //i divide in fifths and the buttons take the central fifth
    int y_blocks_fine=height/12; //the three boxes will go in 
    struct Box clear_button = create_default_box();
    struct Box NN_selection = create_default_box();
    struct Box NN_selection_result = create_default_box();
    struct Box result_box = create_default_box();
    struct Box drawing_area = create_default_box();
    struct Box submit_button = create_default_box();
    //i define the drawing region
    drawing_area.color=al_map_rgb(255, 255, 255);
    drawing_area.x1=x_blocks*2;
    drawing_area.x2=x_blocks*4;
    drawing_area.y1=y_blocks;
    drawing_area.y2=y_blocks+drawing_area.x2-drawing_area.x1;
    //i define the clear button
    clear_button.color=al_map_rgba(0, 255, 0,0.6);
    clear_button.x1=x_blocks_fine*9;
    clear_button.x2=x_blocks_fine*10;
    clear_button.y1=y_blocks_fine*4;
    clear_button.y2=y_blocks_fine*5;
    strcpy(clear_button.text,"CLEAR");
    //i define the selection button
    NN_selection.color=al_map_rgba(255, 255, 0,1);
    NN_selection.x1=x_blocks_fine*9;
    NN_selection.x2=x_blocks_fine*10;
    NN_selection.y1=y_blocks_fine*6;
    NN_selection.y2=y_blocks_fine*7;
    NN_selection.text_color=al_map_rgb(0, 0, 0);
    strcpy(NN_selection.text,"SELECT NN");
    //i define the selection button
    NN_selection_result.color=al_map_rgb(255, 255, 255);
    NN_selection_result.x1=x_blocks_fine*10;
    NN_selection_result.x2=x_blocks_fine*11;
    NN_selection_result.y1=y_blocks_fine*6;
    NN_selection_result.y2=y_blocks_fine*7;
    NN_selection_result.text_color=al_map_rgb(0, 0, 0);
    strcpy(NN_selection_result.text," ");
    //i define the result region
    result_box.color=al_map_rgb(255, 255, 255);
    result_box.x1=x_blocks_fine*5;
    result_box.x2=x_blocks_fine*7;
    result_box.y1=drawing_area.y2+y_blocks_fine;
    result_box.y2=result_box.y1+y_blocks_fine;
    result_box.text_position=2;//left aligned
    result_box.text_color=al_map_rgb(0, 0, 0);
    strcpy(result_box.text,"RESULT:");
    //i define the submit button
    submit_button.color=al_map_rgb(255, 0, 0);
    submit_button.x1=x_blocks_fine*10;
    submit_button.x2=x_blocks_fine*11;
    submit_button.y1=y_blocks_fine*4;
    submit_button.y2=y_blocks_fine*5;
    strcpy(submit_button.text,"SUBMIT");
    int num_boxes=6;
    struct Box boxes[MAX_BUTTONS]={clear_button,NN_selection,NN_selection_result,
    result_box,drawing_area,submit_button};
    Boxes result;
    result.num_boxes=num_boxes;
    for(int i=0;i<num_boxes;i++){
        result.boxes[i]=boxes[i];
    }
    return result;
}

Boxes initialize_training_input_boxes(){
    //i define the buttons needed to select the network structure
    char input_text[13][25]={"N LAYERS","NEURONS PER LAYER","ACTIVATION","TRAIN VAL SPLIT",
    "N TRAIN TEST","OPTIMIZATION","LOSS","LEARNING RATE","MOMENTUM","SHUFFLING",
    "BATCH SIZE","EPOCHS","RESET|TEMPLATE"};
    struct Box n_layers_button = create_default_box();
    struct Box neurons_per_layer_button = create_default_box();
    struct Box activation_button = create_default_box();
    struct Box train_val_split_button = create_default_box();
    struct Box n_train_test_button = create_default_box();
    struct Box optimization_button = create_default_box();
    struct Box loss_button = create_default_box();
    struct Box learning_rate_button = create_default_box();
    struct Box momentum_button = create_default_box();
    struct Box shuffling_button = create_default_box();
    struct Box batch_size_button = create_default_box();
    struct Box epochs_button = create_default_box();
    struct Box reset_template_button = create_default_box();
    int num_input_boxes=13;
    struct Box input_boxes[13]={n_layers_button,neurons_per_layer_button,
    activation_button, train_val_split_button,n_train_test_button,optimization_button,
    loss_button,learning_rate_button,momentum_button,shuffling_button,
    batch_size_button,epochs_button, reset_template_button};

    for (int i=0;i<num_input_boxes;i++){
        if(i==num_input_boxes-1){
            input_boxes[i].color=al_map_rgb(255, 0, 0);
        }
        else{
            input_boxes[i].color=al_map_rgb(0, 255, 0);
        }
        int x=0,y=0,w,h;
        place_object_grid(&x,&y,&w,&h,1,2*i+1,4,1,1/20.0, 1/(float)(num_input_boxes*2+1));
        //printf("x=%d,y=%d,w=%d,h=%d\n",x,y,w,h);
        input_boxes[i].x1=x;
        if(i==num_input_boxes-1){
            input_boxes[i].x2=x+w*2;
        }
        else{
            input_boxes[i].x2=x+w;
        }
        input_boxes[i].y1=y;
        input_boxes[i].y2=y+h;
        strcpy(input_boxes[i].text,input_text[i]);
    }
    Boxes result;
    result.num_boxes=num_input_boxes;
    for(int i=0;i<num_input_boxes;i++){
        result.boxes[i]=input_boxes[i];
    }
    return result;
}

Boxes initialize_training_input_fields(int num_input_fields){
    struct Box input_fields[MAX_BUTTONS];
    //I define the display options of the input buttons
    for (int i=0;i<num_input_fields;i++){
        input_fields[i]=create_default_box();
        input_fields[i].color=al_map_rgb(255, 255, 255);
        int x=0,y=0,w,h;
        place_object_grid(&x,&y,&w,&h,5,2*i+1,4,1,1/20.0, 1/(float)((num_input_fields+1)*2+1));
        input_fields[i].x1=x;
        input_fields[i].x2=x+w;
        input_fields[i].y1=y;
        input_fields[i].y2=y+h;
        input_fields[i].text_color=al_map_rgb(0, 0, 0);
    }
    Boxes result;
    result.num_boxes=num_input_fields;
    for(int i=0;i<num_input_fields;i++){
        result.boxes[i]=input_fields[i];
    }
    return result;
}

Boxes initialize_training_training_boxes(){
    //i define the other buttons and display regions to be shown
    struct Box start_button = create_default_box();
    struct Box stop_button = create_default_box();
    struct Box plot_region = create_default_box();
    struct Box results_region = create_default_box();
    struct Box save_button = create_default_box();
    char training_text[5][25]={
        "START","STOP","PLOT","RESULTS","SAVE"
    };
    int num_training_boxes=5;
    int x,y,w,h;
    float grid_size_x=1/10.0;
    float grid_size_y=1/20.0;
    //i define the start button
    start_button.color=al_map_rgb(0, 255, 0);
    place_object_grid(&x,&y,&w,&h,5,2,2,1,grid_size_x, grid_size_y);
    start_button.x1=x;
    start_button.x2=x+w;
    start_button.y1=y;
    start_button.y2=y+h;
    //i define the stop button
    stop_button.color=al_map_rgb(255, 0, 0);
    place_object_grid(&x,&y,&w,&h,7,2,2,1,grid_size_x, grid_size_y);
    stop_button.x1=x;
    stop_button.x2=x+w;
    stop_button.y1=y;
    stop_button.y2=y+h;
    //i define the plot region
    plot_region.color=al_map_rgb(255, 255, 255);
    place_object_grid(&x,&y,&w,&h,5,4,4,6,grid_size_x, grid_size_y);
    plot_region.x1=x;
    plot_region.x2=x+w;
    plot_region.y1=y;
    plot_region.y2=y+h;
    //i define the results region
    results_region.color=al_map_rgb(255, 255, 255);
    place_object_grid(&x,&y,&w,&h,5,11,4,5,grid_size_x, grid_size_y);
    results_region.x1=x;
    results_region.x2=x+w;
    results_region.y1=y;
    results_region.y2=y+h;
    //i define the save button
    save_button.color=al_map_rgb(0, 255, 0);
    place_object_grid(&x,&y,&w,&h,5,17,4,2,grid_size_x, grid_size_y);
    save_button.x1=x;
    save_button.x2=x+w;
    save_button.y1=y;
    save_button.y2=y+h;
    struct Box training_boxes[5]={start_button,stop_button,plot_region,results_region,save_button};
    //quando assegno in questo modo save_button, start_button etc non sono messi nell'array. Loro copie vengono
    //messe nell'array (lo vedi perchè se l'inizializzazione dell'array è prima -> le coordinate sono sbagliate)
    for (int i=0;i<num_training_boxes;i++){
        strcpy(training_boxes[i].text,training_text[i]);
        //printf("training boxes x1=%d,x2=%d,y1=%d,y2=%d\n",training_boxes[i].x1,training_boxes[i].x2,training_boxes[i].y1,training_boxes[i].y2);
    }
    Boxes result;
    result.num_boxes=num_training_boxes;
    for(int i=0;i<num_training_boxes;i++){
        result.boxes[i]=training_boxes[i];
    }
    return result;
}