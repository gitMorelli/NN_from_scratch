//#include "functions.h"
//#include "graphics.h"
#include <allegro5/allegro.h>
#include <allegro5/allegro_primitives.h>
#include <allegro5/allegro_font.h>
#include <stddef.h> // For NULL
#include <stdlib.h> // For srand
#include <stdio.h>
#include <stdbool.h>
#include <time.h>   // For time
#include <math.h>

#include <stdio.h>
#define image_size 28
static int images[10000][28][28];
void printIntegerByteByByte(int number);
void read_magic_number(FILE *fptr);
void read_dimensions(FILE *fptr,int n_dim);
int flip_reading_order(int number);

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
    printIntegerByteByByte(flip_reading_order(magic_number));
}// 3rd byte is 0x08 -> unsigned byte according to https://github.com/cvdfoundation/mnist 

void read_dimensions(FILE *fptr,int n_dim) {
    int dim_sizes[n_dim];
    fread(dim_sizes, sizeof(int), n_dim, fptr);
    for (int i = 0; i < n_dim; i++) {
        printf("Dimension %d: %d\n", i, flip_reading_order(dim_sizes[i]));
    }
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
        //unsigned char c;
        fread(&labels[i], sizeof(unsigned char), 1, fptr);//labels are unsigned byte 
        //labels[i] = c;
    }
}

void read_images(int (*images)[image_size][image_size], FILE *fptr, int dim) {
    read_magic_number(fptr);
    read_dimensions(fptr,3);//number of elements, width and length
    for (int k=0;k<dim;k++){
        for (int i = 0; i <image_size; i++) {
            for (int j=0;j<image_size;j++){
                unsigned char c;
                fread(&c, sizeof(unsigned char), 1, fptr);
                images[k][i][j] = c;
            }
        }
    }
}

void print_image(int image[image_size][image_size]){
    for (int i = 0; i <image_size; i++) {
        for (int j=0;j<image_size;j++){
            if(image[i][j]>0){
                printf("x");
            }
            else{
                printf(" ");
            }
        }
        printf("\n");
    }
}

int main() {
    // Your code here
    //generic_initialization();
    //menu_loop();
    //interactive_loop();
    FILE *fptr;
    fptr = fopen("input_folder/t10k-labels-idx1-ubyte","rb");
    if(fptr == NULL)
    {
        printf("Error!");   
        exit(1);             
    }
    int labels[10000];
    read_labels(labels,fptr,10000);
    printf("first label: %d\n",labels[20]);
    fclose(fptr);
    FILE *fptr2;
    fptr2 = fopen("input_folder/t10k-images-idx3-ubyte","rb");
    if(fptr2 == NULL)
    {
        printf("Error!");   
        exit(1);             
    }
    read_images(images,fptr2,10000);
    print_image(images[20]);
    fclose(fptr2);
    return 0;
}