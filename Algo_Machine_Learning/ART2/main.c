#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include "utils.h"
#include "art2.h"
/* 
                                Execution:
                                >make
                                >./exec
 */

int main(){
    //Init dataset
    Data* data = (Data*) malloc(sizeof(Data));
    data->matrix = init_float_2D(param.rows, param.cols, 0.0);
    data->matrix_norm = init_float_2D(param.rows, param.cols, 0.0);
    data->names = init_string_1D();
    data->label = (int*)malloc(param.rows * sizeof(int));
    
    //Load data
    data = charge_database("iris.data", data);
    //Normalize the dataset
    norm_matrix(data, param.rows, param.cols); 

    /* show_2D(data->matrix, param.rows, param.cols);
    show_2D(data->matrix_norm, param.rows, param.cols);
    show_string_1D(data->names);  */
    
    //Init neural network and bias
    Network *net = (Network*) malloc(sizeof(Network));
    initNetwork(net); 

    //Show bais before learning
    printf("Bottom up\n");
    show_2D(net->bottom_up, net->nb_cluster, net->nb_feature);
    printf("Top down\n");
    show_2D(net->top_down, net->nb_feature, net->nb_cluster);

    //Start learning
    //double inputs[2][5] = {{0.2,0.7,0.1,0.5,0.4},{0.1,0.8,0.2,0.7,0.1}};
    for(int i = 0; i < param.rows; i++){
        double * input = data->matrix_norm[i];
        propagation_to_F1(net, input);
        propagation_to_F2(net);
        learning(net, input);
        show_2D(net->top_down, net->nb_feature, net->nb_cluster); 
    } 
    ////Show bais after learning
    printf("Bottom up\n");
    show_2D(net->bottom_up, net->nb_cluster, net->nb_feature);
    printf("Top down\n");
    show_2D(net->top_down, net->nb_feature, net->nb_cluster); 
    

    //Free memory
    free_float_2D(data->matrix, param.rows);
    free_float_2D(data->matrix_norm, param.rows);
    free_string_1D(data->names); free(data->label); 

    free(net->S); free(net->X); free(net->Y); 
    free(net->U); free(net->V); free(net->W); free(net->P);
    free(net->Q); free(net->R); free(net->T);
    free_float_2D(net->bottom_up, net->nb_cluster);
    free_float_2D(net->top_down, net->nb_feature); 
    return 0;
}
