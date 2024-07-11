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

//which elements i want in the window? A window where i can draw numbers with the mouse
//a window where i can select the network to use
void interactive_loop(){
    interactive_mode_initialization();
    bool drawing = false; // Flag to track whether we are currently drawing
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
    //i define the drawing region
    drawing_area.color=al_map_rgb(255, 255, 255);
    drawing_area.x1=x_blocks*2;
    drawing_area.x2=x_blocks*4;
    drawing_area.y1=y_blocks*2;
    drawing_area.y2=y_blocks*4;
    //i define the clear button
    clear_button.color=al_map_rgba(0, 255, 0,0.6);
    clear_button.x1=x_blocks_fine*9;
    clear_button.x2=x_blocks_fine*11;
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
    const int num_boxes=4;
    struct Box boxes[MAX_BUTTONS]={clear_button,NN_selection,result_box,drawing_area};
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
                }
                if (is_point_inside_button(event.mouse.x, event.mouse.y, drawing_area)) {
                    printf("drawing\n");
                    drawing = true; // Start drawing
                    last_x = event.mouse.x; // Update last position
                    last_y = event.mouse.y;
                }
                break;
            case ALLEGRO_EVENT_MOUSE_BUTTON_UP:
                drawing = false; // Stop drawing
                break;
            case ALLEGRO_EVENT_MOUSE_AXES:
                if (drawing) {
                    printf("still drawing\n");
                    if (is_point_inside_button(event.mouse.x, event.mouse.y, drawing_area))
                        al_draw_line(last_x, last_y, event.mouse.x, event.mouse.y, al_map_rgb(0, 0, 0), 1);
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
                al_draw_textf(font, boxes[i].text_color, center_x, center_y, boxes[i].text_position, boxes[i].text);
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

void testing_mode_initialization(){
    generic_initialization();
}

void testing_loop(){
    testing_mode_initialization();
    bool drawing = false; // Flag to track whether we are currently drawing
    float last_x = 0, last_y = 0; // Keep track of the last position
    int x_blocks=width/6; //i divide in fifths and the buttons take the central fifth
    int y_blocks=height/6; //the three boxes will go in 
    int x_blocks_fine=width/12; //i divide in fifths and the buttons take the central fifth
    int y_blocks_fine=height/12; //the three boxes will go in 
    int submenu=1;//index to tell in which windows i need to go when menu is closed
    struct Box clear_button = create_default_box();
    struct Box NN_selection = create_default_box();
    struct Box result_box = create_default_box();
    struct Box drawing_area = create_default_box();
    //i define the drawing region
    drawing_area.color=al_map_rgb(255, 255, 255);
    drawing_area.x1=x_blocks*2;
    drawing_area.x2=x_blocks*4;
    drawing_area.y1=y_blocks*2;
    drawing_area.y2=y_blocks*4;
    //i define the clear button
    clear_button.color=al_map_rgba(0, 255, 0,0.6);
    clear_button.x1=x_blocks_fine*9;
    clear_button.x2=x_blocks_fine*11;
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
    const int num_boxes=4;
    struct Box boxes[MAX_BUTTONS]={clear_button,NN_selection,result_box,drawing_area};
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
                }
                if (is_point_inside_button(event.mouse.x, event.mouse.y, drawing_area)) {
                    printf("drawing\n");
                    drawing = true; // Start drawing
                    last_x = event.mouse.x; // Update last position
                    last_y = event.mouse.y;
                }
                break;
            case ALLEGRO_EVENT_MOUSE_BUTTON_UP:
                drawing = false; // Stop drawing
                break;
            case ALLEGRO_EVENT_MOUSE_AXES:
                if (drawing) {
                    printf("still drawing\n");
                    if (is_point_inside_button(event.mouse.x, event.mouse.y, drawing_area))
                        al_draw_line(last_x, last_y, event.mouse.x, event.mouse.y, al_map_rgb(0, 0, 0), 1);
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
            al_draw_textf(font, al_map_rgb(255, 255, 255), width/2, 10, 1, "TESTING");
            for (int i=0;i<num_boxes;i++){
                al_draw_filled_rectangle(boxes[i].x1, boxes[i].y1, boxes[i].x2, boxes[i].y2, boxes[i].color);
                float center_x=(boxes[i].x1+boxes[i].x2)/2;
                float center_y=(boxes[i].y1+boxes[i].y2)/2;
                al_draw_textf(font, boxes[i].text_color, center_x, center_y, boxes[i].text_position, boxes[i].text);
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

void training_mode_initialization(){
    generic_initialization();
}

void training_loop(){
    training_mode_initialization();
    bool drawing = false; // Flag to track whether we are currently drawing
    float last_x = 0, last_y = 0; // Keep track of the last position
    int x_blocks=width/6; //i divide in fifths and the buttons take the central fifth
    int y_blocks=height/6; //the three boxes will go in 
    int x_blocks_fine=width/12; //i divide in fifths and the buttons take the central fifth
    int y_blocks_fine=height/12; //the three boxes will go in 
    int submenu=0;//index to tell in which windows i need to go when menu is closed
    struct Box clear_button = create_default_box();
    struct Box NN_selection = create_default_box();
    struct Box result_box = create_default_box();
    struct Box drawing_area = create_default_box();
    //i define the drawing region
    drawing_area.color=al_map_rgb(255, 255, 255);
    drawing_area.x1=x_blocks*2;
    drawing_area.x2=x_blocks*4;
    drawing_area.y1=y_blocks*2;
    drawing_area.y2=y_blocks*4;
    //i define the clear button
    clear_button.color=al_map_rgba(0, 255, 0,0.6);
    clear_button.x1=x_blocks_fine*9;
    clear_button.x2=x_blocks_fine*11;
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
    const int num_boxes=4;
    struct Box boxes[MAX_BUTTONS]={clear_button,NN_selection,result_box,drawing_area};
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
                }
                if (is_point_inside_button(event.mouse.x, event.mouse.y, drawing_area)) {
                    printf("drawing\n");
                    drawing = true; // Start drawing
                    last_x = event.mouse.x; // Update last position
                    last_y = event.mouse.y;
                }
                break;
            case ALLEGRO_EVENT_MOUSE_BUTTON_UP:
                drawing = false; // Stop drawing
                break;
            case ALLEGRO_EVENT_MOUSE_AXES:
                if (drawing) {
                    printf("still drawing\n");
                    if (is_point_inside_button(event.mouse.x, event.mouse.y, drawing_area))
                        al_draw_line(last_x, last_y, event.mouse.x, event.mouse.y, al_map_rgb(0, 0, 0), 1);
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
            al_draw_textf(font, al_map_rgb(255, 255, 255), width/2, 10, 1, "TRAINING");
            for (int i=0;i<num_boxes;i++){
                al_draw_filled_rectangle(boxes[i].x1, boxes[i].y1, boxes[i].x2, boxes[i].y2, boxes[i].color);
                float center_x=(boxes[i].x1+boxes[i].x2)/2;
                float center_y=(boxes[i].y1+boxes[i].y2)/2;
                al_draw_textf(font, boxes[i].text_color, center_x, center_y, boxes[i].text_position, boxes[i].text);
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

