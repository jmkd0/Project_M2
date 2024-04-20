#ifndef ART2_H
#define ART2_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

typedef struct {
    int nb_feature, nb_cluster, winner;
    double *S, *X, *Y;
    double a, b, c, d, e;
    double vigil, theta, alpha;
    double *U, *V, *W, *P, *Q, *R, *T;
    double **bottom_up, **top_down;
	
}Network;



void initNetwork(Network *net){
    net->nb_feature = param.cols;//5 (n)
    net->nb_cluster = 3;//(m)
    net->a = 10.0;
    net->b = 10.0;
    net->c = 0.1;
    net->d = 0.9;
    net->e = 0.00001;
    net->vigil = 0.9;
    net->theta = 0.2;//1.0 / sqrt(net->nb_feature);//learning rate
    net->winner = 0;

    net->W =  init_float_1D(net->nb_feature);
	net->X =  init_float_1D(net->nb_feature);
	net->V =  init_float_1D(net->nb_feature);
	net->U =  init_float_1D(net->nb_feature);
	net->P =  init_float_1D(net->nb_feature);
	net->Q =  init_float_1D(net->nb_feature);
	net->R =  init_float_1D(net->nb_feature);
	net->T =  init_float_1D(net->nb_feature);

    double val = 0.5 / ((1-net->d) * sqrt((double)net->nb_feature)); 
    net->bottom_up = init_float_2D(net->nb_cluster, net->nb_feature, val);
    net->top_down = init_float_2D(net->nb_feature, net->nb_cluster, 0.0);
}
double threshold(double val, double theta){
    if (val > theta) return val ; 
    else return 0.0; 
}
/*x, u, q ont presque la meme formule :
- Xi = Wi / e +||w||
- Qi = Pi / e + ||P|| 
- Ui = Vi / e +||V||
*/
double* compute_X_U_Q(double * sublayer_dst , double * sublayer_src , double parameter, int size){
	double norm = 0.0;
	norm = vector_norm(sublayer_src, size);
	for(int i=0; i<size; i++)
		sublayer_dst[i] = sublayer_src[i] / (norm + parameter);

    return sublayer_dst;
}

//Ri = (Ui + (c * Pi))/ (e + ||U|| + (c * ||P||))
void compute_R(Network * net){
	double normP = vector_norm(net->P, net->nb_feature);
	double normU = vector_norm(net->U, net->nb_feature);
	for(int i=0;i<net->nb_feature;i++)
		net->R[i] = (net->U[i] + (net->c * net->P[i])) / (net->e + normU + (net->c * normP));
}
//Wi = Ii+a.Ui  I==> input vector (vec)
void compute_W(Network* net, double *vec){
    for (int i=0; i < net->nb_feature; i++)
    net->W[i] = vec[i] + net->a * net->U[i];
}
//Vi = F(Xi)+ b * F(Qi) 
void compute_V(Network* net){
    for (int i=0; i < net->nb_feature; i++)
    net->V[i] = threshold(net->X[i], net->theta) + net->b * threshold(net->Q[i], net->theta);
}
//Pi= Ui + d * ZiJ  J==> l'indice du gagnat  Z == top_down 
void compute_P(Network * net, double * invec){
	for(int i=0; i<net->nb_feature; i++)
		net->P[i] = net->U[i] + (net->d * net->top_down[i][net->winner]);
}
void propagation_to_F1(Network * net, double * input){
	//compute W
	compute_W(net, input);
	//compute X
	net->X = compute_X_U_Q(net->X, net->W, net->e, net->nb_feature);
	//compute V
	compute_V(net);
	//compute U
	net->U = compute_X_U_Q(net->U, net->V, net->e, net->nb_feature);
	//compute P
	compute_P(net, input);
	//compute Q
	net->Q = compute_X_U_Q(net->Q, net->P, net->e, net->nb_feature);	
}
//la sortie du F2 est Tj = ∑(Pi * Zji)    Zji == bottom_up
void propagation_to_F2(Network * net){
	for (int j = 0; j < net->nb_cluster; j++)
        for (int i = 0; i < net->nb_feature; i++)
            net->T[j] += net->bottom_up[j][i] * net->P[i]; 
}
//find the winner
int find_winner(double * array, int nb_feature){
	int winner = 0;
	for(int i=0; i < nb_feature; i++)
		if(array[winner] < array[i])
			winner = i;
	return winner;
}
//Lerning
void learning(Network * net, double * input){
	int reset = 1, compter = 0;
	double norm = 0.0;
	while(reset && compter < net->nb_cluster){
        
		net->winner = find_winner(net->T, net->nb_feature);
		printf("winner is : %d\n", net->winner);
		//compute U
		net->U = compute_X_U_Q(net->U, net->V, net->e, net->nb_feature);
		//compute P
		compute_P(net, input);
		//compute R
		compute_R(net);
		//compute of R
		norm = vector_norm(net->R, net->nb_feature);
		if(norm >= net->vigil){
			propagation_to_F1(net, input);
			for (int i = 0; i < net->nb_feature; i++)
			{
				net->top_down[i][net->winner] = net->U[i]/ (1-net->d);
        		net->bottom_up[net->winner][i] = net->U[i]/ (1-net->d);
    		}
    		reset = 0;
		}
		else{
			net->T[net->winner] = -1;
			reset = 1;
		}
		compter += 1;
	}
}
#endif