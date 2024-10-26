#include <stdio.h>

// initialize c array values to 0
void init_arr(float* arr, int size) {
    for (int i = 0; i < size; ++i) {
        arr[i] = 0;
    }
}

// initialize c array with incremental values from 0 to size-1
void arange_arr(float* arr, int size) {
    for (int i = 0; i < size; ++i) {
        arr[i] = i;
    }
}

// print the given c array
void print_arr(float* arr, int size) {
    for (int i = 0; i < size; ++i) {
        if (i > 0) {
            printf(", ");
        }
        printf("%0.2f", arr[i]);
    }
    printf("\n");
}

