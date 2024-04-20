#ifndef UTILS_H
#define UTILS_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
struct Params{
    int rows;
    int cols;
}param = {.rows = 150, .cols = 4};

typedef struct{
    double** matrix;
    double** matrix_norm;
    char** names;
    int* label;
}Data;
double** init_float_2D(int rows, int cols, double val){
    double** matrix = malloc(rows * sizeof(double));
    for (int i = 0; i < rows; i++){
        
        matrix[i] = (double*) malloc(cols * sizeof(double));
        for (int j=0; j < cols; j++){
            matrix[i][j] = val;
        } 
    }
    return matrix;
}
double* init_float_1D(int size){
    double* vector = malloc(size * sizeof(double));
    return vector;
}
char** init_string_1D(){
    char** vector  = malloc(param.rows * sizeof(char));
    for (int i = 0; i < param.rows; i++){
        vector[i] = (char*)malloc(100 * sizeof(char));
    }
    return vector;
}

void show_1D(double* vector, int size){
    for (int i = 0; i < size; i++){
        printf("%f ",vector[i]);
    }
    printf("\n");
} 
 void show_2D(double** matrix, int rows, int cols){
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
        printf("%f ",matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
} 
void free_float_2D(double** matrix, int size){
    for (int i = 0; i < size; i++) free(matrix[i]);
    free(matrix);
}

void show_string_1D(char** vector){
    for (int i = 0; i < param.rows; i++){
        printf("%s ",vector[i]);
    }
    printf("\n");
}
void free_string_1D(char** vector){
    for (int i = 0; i < param.rows; i++) free(vector[i]);
    free(vector);
}
Data * charge_database(char* fileName, Data *data){
    int     lineSize = 100;
    char*   ligne = (char*)malloc(lineSize*sizeof(char));
    char*   chaine;   
    int     compterLine=0, compterColonne;
    FILE* fichier = fopen(fileName, "r");
    if (fichier != NULL){
        while ( fgets( ligne, lineSize, fichier) != NULL ){
            compterColonne = 1;
            chaine=strtok(ligne, ",");
            data->matrix[compterLine][0] = atof(chaine);
            while(compterColonne < param.cols){
                chaine=strtok(NULL, ",");
                data->matrix[compterLine][compterColonne] = atof(chaine);
                compterColonne++;
            }
            chaine=strtok(NULL, "\n");
            data->names[compterLine]  = strdup(chaine);
            if(strcmp(strdup(chaine), "Iris-setosa") == 0) data->label[compterLine] = 1;
            if(strcmp(strdup(chaine), "Iris-versicolor") == 0) data->label[compterLine] = 2;
            if(strcmp(strdup(chaine), "Iris-virginica") == 0) data->label[compterLine] = 3;
           compterLine++;  
    } 
    fclose( fichier ) ;
    }
    return data;
} 

double vector_norm(double* vector, int size){
    double norm = 0;
    for(int i= 0; i < size; i++) norm += pow(vector[i], 2);
    norm = sqrt(norm);
    return norm;
}

double* unit_vector(double* vector, int size){
    double* vec = malloc(size * sizeof(double));
    double norm = vector_norm(vector, size);
    for(int i= 0; i < size; i++) vec[i] = vector[i] / norm; 
    return vec;
}
void norm_matrix(Data* data, int rows, int cols){
    for(int i= 0; i < rows; i++){
        data->matrix_norm[i] = unit_vector(data->matrix[i], cols);
    }
} 
double* sum_vector(double* vec_1, double* vec_2){
    for(int i= 0; i < param.rows; i++){
        vec_1[i] = vec_1[i] + vec_2[i];
    }
    return vec_1;
} 
#endif