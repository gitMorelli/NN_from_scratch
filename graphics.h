#include <allegro5/allegro.h>
#include <allegro5/allegro_primitives.h>
#include <allegro5/allegro_font.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FPS 60
#define FRAME_TAU 60
#define NUM_VERTICES 4096
#define KEY_SEEN     1
#define KEY_RELEASED 2
#define WIDTH 640
#define HEIGHT 480
#define MAX_BUTTONS 20

static int INPUT; //stores the input of the user (which come from the mouse interaction with some buttons)
// quit=0, menu=1, train=2, test=3, interactive=4
static ALLEGRO_TIMER* timer;
static ALLEGRO_EVENT_QUEUE* queue;
static ALLEGRO_DISPLAY* disp;
static ALLEGRO_FONT* font;
static bool done;
static bool redraw;
static ALLEGRO_EVENT event;
static float x_graphics, y_graphics;
static unsigned char key[ALLEGRO_KEY_MAX];
static int width=WIDTH;
static int height=HEIGHT;
static struct Box{
    int x1;
    int y1;
    int x2;
    int y2;
    ALLEGRO_COLOR color;
    char text[15];
    int text_position;
    ALLEGRO_COLOR text_color;
} ;

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
void process_drawing_region(int dim, int **drawing_matrix);
void training_loop();
void testing_loop();
void drawing_function(int x, int y, int drawing_x,int drawing_y, int r,int dim,int **drawing_matrix);
void upsample_image(int image[28][28], int dim, int upsampled_image[dim][dim]);
ALLEGRO_BITMAP *create_bitmap_from_image(int dim,int image[dim][dim]);
void display_next_image(int dim, int x, int y);
void place_object_grid(int *x,int *y,int *w, int *h, int x_grid, int y_grid, 
int w_grid, int h_grid, float grid_size_x, float grid_size_y);
void place_object(int x,int y,float x_ratio, float y_ratio);

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

    must_init(al_init_primitives_addon(), "primitives");

    al_register_event_source(queue, al_get_keyboard_event_source());
    al_register_event_source(queue, al_get_display_event_source(disp));
    al_register_event_source(queue, al_get_timer_event_source(timer));
    al_register_event_source(queue, al_get_mouse_event_source());

    done = false;
    redraw = true;

    x_graphics = 100;
    y_graphics = 100;

    memset(key, 0, sizeof(key));//inititalize all elements of the array to 0

    al_start_timer(timer);
}

void destroy_graphics()
{
    al_destroy_font(font);
    al_destroy_display(disp);
    al_destroy_timer(timer);
    al_destroy_event_queue(queue);
}

void menu_initialization()
{
    generic_initialization();
}

void menu_loop(){
    menu_initialization();
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
                al_draw_textf(font, al_map_rgb(255, 255, 255), center_x, center_y, 1, boxes[i].text);
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

void interactive_mode_initialization(){
    generic_initialization();
}

void drawing_function(int x, int y, int drawing_x,int drawing_y, int r,int dim,int **drawing_matrix){
    al_draw_filled_rectangle(x-r, y-r, x+r, y+r, al_map_rgb(0, 0, 0));
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
//which elements i want in the window? A window where i can draw numbers with the mouse
//a window where i can select the network to use
void interactive_loop(){
    interactive_mode_initialization();
    bool drawing = false; // Flag to track whether we are currently drawing
    bool inference = false;
    float last_x = 0, last_y = 0; // Keep track of the last position
    int x_blocks=width/6; //i divide in fifths and the buttons take the central fifth
    int y_blocks=height/6; //the three boxes will go in 
    int x_blocks_fine=width/12; //i divide in fifths and the buttons take the central fifth
    int y_blocks_fine=height/12; //the three boxes will go in 
    int submenu=2;//index to tell in which windows i need to go when menu is closed
    struct Box clear_button = create_default_box();
    struct Box NN_selection = create_default_box();
    struct Box result_box = create_default_box();
    struct Box drawing_area = create_default_box();
    struct Box submit_button = create_default_box();
    //i define the drawing region
    drawing_area.color=al_map_rgb(255, 255, 255);
    drawing_area.x1=x_blocks*2;
    drawing_area.x2=x_blocks*4;
    drawing_area.y1=y_blocks;
    drawing_area.y2=y_blocks+drawing_area.x2-drawing_area.x1;
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
    //i define the clear button
    clear_button.color=al_map_rgba(0, 255, 0,0.6);
    clear_button.x1=x_blocks_fine*9;
    clear_button.x2=x_blocks_fine*10;
    clear_button.y1=y_blocks_fine*4;
    clear_button.y2=y_blocks_fine*5;
    strcpy(clear_button.text,"CLEAR");
    //i define the selection button
    NN_selection.color=al_map_rgba(255, 255, 120,0.6);
    NN_selection.x1=x_blocks_fine*9;
    NN_selection.x2=x_blocks_fine*11;
    NN_selection.y1=y_blocks_fine*6;
    NN_selection.y2=y_blocks_fine*7;
    strcpy(NN_selection.text,"SELECT NN");
    //i define the result region
    result_box.color=al_map_rgb(255, 255, 255);
    result_box.x1=x_blocks_fine*5;
    result_box.x2=x_blocks_fine*7;
    result_box.y1=y_blocks_fine*9;
    result_box.y2=y_blocks_fine*10;
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
    const int num_boxes=5;
    struct Box boxes[MAX_BUTTONS]={clear_button,NN_selection,result_box,drawing_area,submit_button};
    while(1){
        al_wait_for_event(queue, &event);

        switch(event.type)
        {
            case ALLEGRO_EVENT_TIMER:
                if(key[ALLEGRO_KEY_ESCAPE]){
                    submenu=-1;
                    done = true;
                }
                for(int i = 0; i < ALLEGRO_KEY_MAX; i++)
                    key[i] &= KEY_SEEN;
                //redraw = true;
                break;
            case ALLEGRO_EVENT_MOUSE_BUTTON_DOWN:
                if (is_point_inside_button(event.mouse.x, event.mouse.y, clear_button)) {
                    // Button was clicked, perform an action
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
                }
                if (is_point_inside_button(event.mouse.x, event.mouse.y, submit_button)) {
                    // Button was clicked, perform an action
                    inference = true;
                }
                break;
            case ALLEGRO_EVENT_MOUSE_BUTTON_UP:
                drawing = false; // Stop drawing
                break;
            case ALLEGRO_EVENT_MOUSE_AXES:
                if (drawing) {
                    //printf("still drawing\n");
                    if (is_point_inside_button(event.mouse.x, event.mouse.y, drawing_area))
                        drawing_function(event.mouse.x, event.mouse.y, drawing_area.x1,drawing_area.y1, 
                        4, dim_drawing, drawing_matrix);
                        al_flip_display();
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

            case ALLEGRO_EVENT_DISPLAY_CLOSE:
                done = true;
                break;
        }

        if(done)
            break;

        if(redraw && al_is_event_queue_empty(queue)){
            //printf("aaaaah\n");
            al_clear_to_color(al_map_rgb(0, 0, 0));
            al_draw_textf(font, al_map_rgb(255, 255, 255), width/2, 10, 1, "INTERACTIVE");
            for (int i=0;i<num_boxes;i++){
                al_draw_filled_rectangle(boxes[i].x1, boxes[i].y1, boxes[i].x2, boxes[i].y2, boxes[i].color);
                float center_x=(boxes[i].x1+boxes[i].x2)/2;
                float center_y=(boxes[i].y1+boxes[i].y2)/2;
                al_draw_textf(font, boxes[i].text_color, center_x, center_y, boxes[i].text_position, "%s" ,boxes[i].text);
            }

            al_flip_display();

            redraw = false;
        }
        if (inference){
            //printf("here 0\n");
            process_drawing_region(dim_drawing, drawing_matrix);
            inference=false;
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

void testing_mode_initialization(){
    generic_initialization();
}

void testing_loop(){
    testing_mode_initialization();
    int x_blocks=width/6; //i divide in fifths and the buttons take the central fifth
    int y_blocks=height/6; //the three boxes will go in 
    int x_blocks_fine=width/12; //i divide in fifths and the buttons take the central fifth
    int y_blocks_fine=height/12; //the three boxes will go in 
    int submenu=1;//index to tell in which windows i need to go when menu is closed
    struct Box next_button = create_default_box();
    struct Box submit_button = create_default_box();
    struct Box NN_selection = create_default_box();
    struct Box start_test_box = create_default_box();
    struct Box result_box = create_default_box();
    struct Box showing_area = create_default_box();
    struct Box test_results_area = create_default_box();
    //i define the drawing region
    showing_area.color=al_map_rgb(255, 255, 255);
    showing_area.x1=x_blocks*2;
    showing_area.x2=x_blocks*4;
    showing_area.y1=y_blocks;
    showing_area.y2=y_blocks+showing_area.x2-showing_area.x1;
    int drawing_dim=showing_area.x2-showing_area.x1;
    //I define the are where i will plot the results
    test_results_area.color=al_map_rgb(255, 255, 255);
    test_results_area.x1=x_blocks_fine*1;
    test_results_area.x2=x_blocks_fine*3;
    test_results_area.y1=y_blocks;
    test_results_area.y2=showing_area.y2;
    strcpy(test_results_area.text,"TEST RESULTS");
    //i define the next button
    next_button.color=al_map_rgba(0, 255, 0,0.6);
    next_button.x1=x_blocks_fine*9;
    next_button.x2=x_blocks_fine*10;
    next_button.y1=y_blocks_fine*4;
    next_button.y2=y_blocks_fine*5;
    strcpy(next_button.text,"NEXT");
    //i define the submit button
    submit_button.color=al_map_rgba(255, 0, 0,0.6);
    submit_button.x1=x_blocks_fine*10;
    submit_button.x2=x_blocks_fine*11;
    submit_button.y1=y_blocks_fine*4;
    submit_button.y2=y_blocks_fine*5;
    strcpy(submit_button.text,"SUBMIT");
    //i define the selection button
    NN_selection.color=al_map_rgba(255, 255, 120,0.6);
    NN_selection.x1=x_blocks_fine*9;
    NN_selection.x2=x_blocks_fine*10;
    NN_selection.y1=y_blocks_fine*6;
    NN_selection.y2=y_blocks_fine*7;
    strcpy(NN_selection.text,"SELECT NN");
    //i define the button to start execution of test on test set
    start_test_box.color=al_map_rgba(255, 255, 120,0.6);
    start_test_box.x1=x_blocks_fine*10;
    start_test_box.x2=x_blocks_fine*11;
    start_test_box.y1=y_blocks_fine*6;
    start_test_box.y2=y_blocks_fine*7;
    strcpy(start_test_box.text,"START TEST");
    //i define the result region
    result_box.color=al_map_rgb(255, 255, 255);
    result_box.x1=x_blocks_fine*5;
    result_box.x2=x_blocks_fine*7;
    result_box.y1=y_blocks_fine*9;
    result_box.y2=y_blocks_fine*10;
    result_box.text_position=2;//left aligned
    result_box.text_color=al_map_rgb(0, 0, 0);
    strcpy(result_box.text,"RESULT:");
    const int num_boxes=7;
    struct Box boxes[MAX_BUTTONS]={next_button,submit_button,NN_selection,
    start_test_box,result_box,showing_area,test_results_area};
    while(1){
        al_wait_for_event(queue, &event);

        switch(event.type)
        {
            case ALLEGRO_EVENT_TIMER:
                if(key[ALLEGRO_KEY_ESCAPE]){
                    submenu=-1;
                    done = true;
                }
                for(int i = 0; i < ALLEGRO_KEY_MAX; i++)
                    key[i] &= KEY_SEEN;
                //redraw = true;
                break;
            case ALLEGRO_EVENT_MOUSE_BUTTON_DOWN:
                if (is_point_inside_button(event.mouse.x, event.mouse.y, next_button)) {
                    // Button was clicked, perform an action
                    redraw = true;
                }
                break;
            case ALLEGRO_EVENT_KEY_DOWN:
                key[event.keyboard.keycode] = KEY_SEEN | KEY_RELEASED;
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
            //printf("aaaaah\n");
            al_clear_to_color(al_map_rgb(0, 0, 0));
            al_draw_textf(font, al_map_rgb(255, 255, 255), width/2, 10, 1, "TESTING");
            for (int i=0;i<num_boxes;i++){
                al_draw_filled_rectangle(boxes[i].x1, boxes[i].y1, boxes[i].x2, boxes[i].y2, boxes[i].color);
                float center_x=(boxes[i].x1+boxes[i].x2)/2;
                float center_y=(boxes[i].y1+boxes[i].y2)/2;
                al_draw_textf(font, boxes[i].text_color, center_x, center_y, boxes[i].text_position,"%s" , boxes[i].text);
            }
            display_next_image(drawing_dim,showing_area.x1,showing_area.y1);//extract one image at random and put as bitmap in the drawing region
            //extract one image at random and put as bitmap in the drawing region
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

void training_mode_initialization(){
    generic_initialization();
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
    training_mode_initialization();

    //i initialize the boolean variables that describe the state of execution
    int submenu=0;//index to tell in which windows i need to go when menu is closed
    bool done=false;
    bool is_training=false;

    //i define the buttons needed to select the network structure
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
    struct Box input_boxes[MAX_BUTTONS]={n_layers_button,neurons_per_layer_button,
    activation_button, train_val_split_button,n_train_test_button,optimization_button,
    loss_button,learning_rate_button,momentum_button,shuffling_button,
    batch_size_button,epochs_button, reset_template_button};
    struct Box input_fields[MAX_BUTTONS];
    int num_input_fields=num_input_boxes-1;
    char input_text[13][25]={"N LAYERS","NEURONS PER LAYER","ACTIVATION","TRAIN VAL SPLIT",
    "N TRAIN TEST","OPTIMIZATION","LOSS","LEARNING RATE","MOMENTUM","SHUFFLING",
    "BATCH SIZE","EPOCHS","RESET|TEMPLATE"};

    //When a button is pressed the cmd shows a prompt to insert the corresponding value
    //this stores the prompt
    char output_cmd[13][1000]={"Insert the number of layers\n",
    "Insert the number of neurons per layer using the format n1 n2 n3 .. nk for k layers\n",
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
    "0=Default -> 1 layer, 64 neurons, sigmoid, 0.1 validation, 60000 training, 10000 test, Nesterov, log-likelihood, lr 0.1, p 0.9, shuffle, 32 batch, 20 epochs\n"
    "\n 1=Small network -> "
    "\n 2=Large network -> "};
    //These arrays enabels to deal with the logic of each button avoiding the need
    //to implement a different logic for each button
    int n_layers=1;
    int n_inputs[13]={1,-1,1,1,2,1,1,1,1,1,1,1,1};//is the number of inputs expected for the prompt of each buttton
    //-1 means that the number of inputs depends on other variables (eg. neurons per layer depends on n_layers)
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
    char output_text[13][1000]={"sigmoid","ReLu","SGD","SGD with momentum","Nesterov",
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
        {1,0,0.1,60000,10000,2,0,0.1,0.9,1,32,20}
    };
    int default_n_layers[3][max_hidden_layers]=
    {
        {64},
        {32,32},
        {128}
    };

    //I define the display options of the input buttons
    for (int i=0;i<num_input_boxes;i++){
        if(i==num_input_boxes-1){
            input_boxes[i].color=al_map_rgb(255, 0, 0);
        }
        else{
            input_boxes[i].color=al_map_rgb(0, 255, 0);
        }
        int x=0,y=0,w,h;
        place_object_grid(&x,&y,&w,&h,1,2*i+1,1,1,1/10.0, 1/(float)(num_input_boxes*2+1));
        //printf("x=%d,y=%d,w=%d,h=%d\n",x,y,w,h);
        input_boxes[i].x1=x;
        if(i==num_input_boxes-1){
            input_boxes[i].x2=x+w*3;
        }
        else{
            input_boxes[i].x2=x+w;
        }
        input_boxes[i].y1=y;
        input_boxes[i].y2=y+h;
        strcpy(input_boxes[i].text,input_text[i]);
    }
    for (int i=0;i<num_input_fields;i++){
        input_fields[i]=create_default_box();
        input_fields[i].color=al_map_rgb(255, 255, 255);
        int x=0,y=0,w,h;
        place_object_grid(&x,&y,&w,&h,3,2*i+1,1,1,1/10.0, 1/(float)(num_input_boxes*2+1));
        input_fields[i].x1=x;
        input_fields[i].x2=x+w;
        input_fields[i].y1=y;
        input_fields[i].y2=y+h;
    }

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

    
    while(1){
        al_wait_for_event(queue, &event);
        switch(event.type)
        {
            case ALLEGRO_EVENT_TIMER:
                if(key[ALLEGRO_KEY_ESCAPE]){
                    submenu=-1;
                    done = true;
                }
                for(int i = 0; i < ALLEGRO_KEY_MAX; i++)
                    key[i] &= KEY_SEEN;
                //redraw = true;
                break;
            case ALLEGRO_EVENT_MOUSE_BUTTON_DOWN:
                if(!is_training){
                    for (int i=0;i<num_input_boxes-1;i++){
                        if (is_point_inside_button(event.mouse.x, event.mouse.y, training_boxes[i])) {
                            // Button was clicked, perform an action
                            printf("%s",output_cmd[i]);
                            if(n_inputs[i]!=-1){
                                for(int j=0;j<n_inputs[i];j++){
                                    fscanf(stdin,"%f",&parameters[index_to_par[i]+j]);
                                }
                            }
                            else{
                                for(int j=0;j<n_layers;j++){
                                    fscanf(stdin,"%d",&layers[j]);
                                }
                            }
                            redraw = true;
                        }
                    }
                    if (is_point_inside_button(event.mouse.x, event.mouse.y, training_boxes[num_input_boxes-1])) {
                        int p;
                        fscanf(stdin,"%f",&p);
                        for (int i=0;i<num_input_boxes-1;i++){
                            parameters[i]=default_configs[p][i];
                        }
                    }
                }
                break;
            case ALLEGRO_EVENT_KEY_DOWN:
                key[event.keyboard.keycode] = KEY_SEEN | KEY_RELEASED;
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
            if(!is_training){
                //i set the neural network structure
                define_training_parameters((int)parameters[11],parameters[7], (int)parameters[6], (int)parameters[9], 
                0.001, (int)parameters[5], parameters[8]);
                //i fix the minimum error to 0.0001
                define_network_structure(layers, (int)parameters[0], (int)parameters[1], 0);//i fix initialization to gaussian
                set_number_of_inputs((int)parameters[3], (int)parameters[4]);
                set_train_val((int)parameters[10], parameters[2]);
                for (int i=0;i<num_input_fields;i++){
                    if(type_output[i]==0){
                        sprintf(input_fields[i].text,"%f",parameters[index_to_par[i]]);
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
            for (int i=0;i<num_input_boxes;i++){
                al_draw_filled_rectangle(input_boxes[i].x1, input_boxes[i].y1, input_boxes[i].x2, input_boxes[i].y2, input_boxes[i].color);
                float center_x=(input_boxes[i].x1+input_boxes[i].x2)/2;
                float center_y=(input_boxes[i].y1+input_boxes[i].y2)/2;
                al_draw_textf(font, input_boxes[i].text_color, center_x, center_y, input_boxes[i].text_position,"%s" , input_boxes[i].text);
            }
            for (int i=0;i<num_input_fields;i++){
                al_draw_filled_rectangle(input_fields[i].x1, input_fields[i].y1, input_fields[i].x2, input_fields[i].y2, input_fields[i].color);
                float center_x=(input_fields[i].x1+input_fields[i].x2)/2;
                float center_y=(input_fields[i].y1+input_fields[i].y2)/2;
                al_draw_textf(font, input_fields[i].text_color, center_x, center_y, input_fields[i].text_position,"%s" , input_fields[i].text);
            }
            for (int i=0;i<num_training_boxes;i++){
                al_draw_filled_rectangle(training_boxes[i].x1, training_boxes[i].y1, training_boxes[i].x2, training_boxes[i].y2, training_boxes[i].color);
                float center_x=(training_boxes[i].x1+training_boxes[i].x2)/2;
                float center_y=(training_boxes[i].y1+training_boxes[i].y2)/2;
                al_draw_textf(font, training_boxes[i].text_color, center_x, center_y, training_boxes[i].text_position, "%s" , training_boxes[i].text);
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

void process_drawing_region(int dim, int **drawing_matrix){
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
    int l_sup=5;//-> considero un qudrato di lato 5 attorno al pixel di arrivo per fare la media
    for(int i=0;i<28;i++){
        for(int j=0;j<28;j++){
            image[i][j]=0;
            int i_sup=(int)((float)i/28.0*(float)n);
            int j_sup=(int)((float)j/28.0*(float)n);
            for(int k=i_sup-l_sup;k<i_sup+l_sup;k++){
                for(int l=j_sup-l_sup;l<j_sup+l_sup;l++){
                    if(k>=0 && k<n && l>=0 && l<n){
                        /*if(drawing_matrix[k][l]>image[i][j]){//i take the max value
                            image[i][j]=drawing_matrix[k][l];
                        }*/
                        image[i][j]+=(int)((float)(drawing_matrix[k][l]/(l_sup*l_sup+1))); //i take the average
                    }
                }
            }
        }
    }
    //print_image(image);
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

void display_next_image(int dim, int x, int y){
    //i take the next image from the test set
    printf("This will be printed before the error occurs.\n");
    fflush(stdout);
    int index=rand()%number_of_test_images;
    printf("%d\n",index);
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

