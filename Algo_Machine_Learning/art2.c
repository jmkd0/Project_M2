#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include <ctype.h>
#include <math.h>
#include <assert.h>

/*
les paramètres du réseau :

n --> int 
	la taille de notre vecteur d'entree

m --> int 
	le nombre de classe 
	nombre de neurone dans la couche F2

a,b poids fixes dans la couche F1. Ne doit pas être nul.
	a = 10
	b = 10

c poids fixe utilisé dans le test de réinitialisation
	c = 0.1

rho : float
	Vigilance parametere

theta : float
	Suppression paramatere
	e.g. theta = 1 / sqrt(n) 
	ici j'ai pris un montant indiqué sur l'exemple de skapura 
*/


struct parameters
{
    int n,m;
    double a,b,c,d,e,theta,rho; 
};
typedef struct parameters parameters;

struct network
{
    int n,m,winner;
    double a,b,c,d,e,theta,rho; 
    double **top_down, **bottom_up, *w,*x,*v,*u,*p,*q,*r,*t;
};
typedef struct network network;


network * initialisation(parameters data , double * invec)
{
	int i, j;
	network * net = (network *) malloc(sizeof(network));

	net->n = data.n;
	net->m = data.m;
	net->a = data.a;
	net->b = data.b;
	net->c = data.c;
	net->d = data.d;
	net->e = data.e;
	net->theta = data.theta;
	net->rho = data.rho;
	net->winner = 1;
	net->w =  calloc(net->n, sizeof(double));
	net->x =  calloc(net->n, sizeof(double));
	net->v =  calloc(net->n, sizeof(double));
	net->u =  calloc(net->n, sizeof(double));
	net->p =  calloc(net->n, sizeof(double));
	net->q =  calloc(net->n, sizeof(double));
	net->r =  calloc(net->n, sizeof(double));
	net->t =  calloc(net->n, sizeof(double));

	//On alloue de la memoire à notre matrice top_down au debut il est initialiser a 0
	net->top_down = calloc(net->n , sizeof(double *));
	for (i=0;i<net->n;i++)
		net->top_down[i]=  calloc(net->m, sizeof(double));

	//On alloue de la memoire à notre matrice bottom_up 
	net->bottom_up = calloc(net->m , sizeof(double *));
	for (i=0;i<net->m;i++)
		net->bottom_up[i]=  calloc(net->n, sizeof(double));
	//On itilitalise notre matrice la meme valeur obtenue en appliquant la formule :
	// 0.5/(1 - d)√N    N==> taille de notre vecteur en entree 

	for (i=0;i<net->m;i++)
		for(j=0;j<net->n;j++)
			net->bottom_up[i][j]=0.5/((1.0 - net->d)*sqrt((double)net->n)) ;

	//on affiche nous deux matrices top_down et botton_up 
	printf("Top Down :\n"); 
	for (i = 0; i < net->n; i++){
		for (j = 0; j < net->m; j++){
			printf("%f ",net->top_down[i][j]);
		}
		printf("\n");
	}
	printf("\nBottom Up :\n");
	for (i = 0; i < net->m; i++){
		for (j = 0; j < net->n; j++)
			printf("%f ",net->bottom_up[i][j]);
		printf("\n");
	}
	return net;
}
/* la fonction suivante nous permet de calculer la norme d'un vecteur
 on urilise cette fonction dans les calcules des sublayers */ 
double normalisation(double * array, int size)
{
	double sum = 0.0;
	for(int i=0;i<size;i++)
		sum += pow(array[i],2);
	return sqrt(sum);
}
void show_1D(double* vector, int size){
    for (int i = 0; i < size; i++){
        printf("%f ",vector[i]);
    }
    printf("\n");
} 
/*la fonction sera utilisée pour calculer le sublayer V
Cette fonction traite tout signal qui est inférieur à thêta 
comme du bruit et le supprime (le met à zéro).*/
double _F(double value_array , double theta)
{
	if(value_array  > theta)
		return value_array;
	return 0;
}

/*x, u, q ont presque la meme formule :
- Xi = Wi / e +||w||
- Qi = Pi / e + ||P|| 
- Ui = Vi / e +||V||
*/
void calculate_sublayers_X_U_Q(double * sublayer_dst , double * sublayer_src , double parameter, int size)
{
	double norm = 0.0;
	norm = normalisation(sublayer_src,size);
	for(int i=0; i<size; i++)
		sublayer_dst[i] = sublayer_src[i] / (norm + parameter);
}

//Ri = (Ui + (c * Pi))/ (e + ||U|| + (c * ||P||))
void calculate_sublayer_R(network * net)
{
	double normP = normalisation(net->p,net->n);
	double normU = normalisation(net->u,net->n);
	for(int i=0;i<net->n;i++)
		net->r[i] = (net->u[i] + (net->c * net->p[i])) / (net->e + normU + (net->c * normP));
}

//Wi=Ii+a.Ui  I==> le vecteur en entree 
void calculate_sublayer_W(network * net, double * invec)
{
	for(int i=0;i<net->n;i++)
		net->w[i] = invec[i] + (net->a * net->u[i]);
}
//Vi = F(Xi)+ b * F(Qi) 
void calculate_sublayer_V(network * net)
{
	for (int i = 0; i < net->n; i++)
        net->v[i] = _F(net->x[i], net->theta) + net->b * _F(net->q[i], net->theta);
}

//Pi= Ui + d * ZiJ  J==> l'indice du gagnat  Z == top_down 
void calculate_sublayer_P(network * net, double * invec)
{
	for(int i=0; i<net->n; i++)
		net->p[i] = net->u[i] + (net->d * net->top_down[i][net->winner]);
}

void prop_to_F1(network * net, double * invec)
{
	//claculer W
	calculate_sublayer_W(net,invec);
	//calculer X
	calculate_sublayers_X_U_Q(net->x, net->w, net->e, net->n);
	//calculer V
	calculate_sublayer_V(net);
	//calculer U
	calculate_sublayers_X_U_Q(net->u, net->v, net->e, net->n);
	//calculer P
	calculate_sublayer_P(net,invec);
	//calculer Q
	calculate_sublayers_X_U_Q(net->q, net->p, net->e, net->n);	
}

//la sortie du F2 est Tj = ∑(Pi * Zji)    Zji == bottom_up
void prop_to_F2(network * net)
{
	for (int j = 0; j < net->m; j++)
        for (int i = 0; i < net->n; i++)
            net->t[j] += net->bottom_up[j][i] * net->p[i]; 
}

//la fonction suivante nous permet de trouver le gagnat 
int index_winner(double * array, int size)
{
	show_1D(array, size);
	int winner = 0;
	for(int i=0; i<size; i++)
		if(array[winner] < array[i])
			winner = i;
	return winner;
}

void learning(network * net, double * invec)
{
	int reset = 1, compter = 0;
	double norm = 0.0;
	while(reset && compter < net->m)
	{
		net->winner = index_winner(net->t,net->n);
		printf("winner is : %d\n", net->winner);
		//calculer U
		calculate_sublayers_X_U_Q(net->u, net->v, net->e, net->n);
		//calculer P
		calculate_sublayer_P(net,invec);
		//calculer R
		calculate_sublayer_R(net);
		//calculer la norme du vecteur R
		norm = normalisation(net->r,net->n);
		if(norm >= net->rho)
		{
			prop_to_F1(net,invec);
			for (int i = 0; i < net->n; i++)
			{
				net->top_down[i][net->winner] = net->u[i]/ (1-net->d);
        		net->bottom_up[net->winner][i] = net->u[i]/ (1-net->d);
    		}
    		reset = 0;
		}
		else
		{
			net->t[net->winner] = -1;
			reset = 1;
		}
		compter += 1;
	}

	printf("Top Down :\n"); 
	for (int i = 0; i < net->n; i++){
		for (int j = 0; j < net->m; j++){
			printf("%f ",net->top_down[i][j]);
		}
		printf("\n");
	}
	printf("\n\nBottom Up :\n");
	for (int i = 0; i < net->m; i++){
		for (int j = 0; j < net->n; j++)
			printf("%f ",net->bottom_up[i][j]);
		printf("\n");
	}
}

int main (){
	parameters data; 
	data.n = 5;
	data.m = 6;
	data.a = 10.0;
	data.b = 10.0;
	data.c = 0.1;
	data.d = 0.9;
	data.e = 0.00001;
	data.theta = 0.2;
	data.rho = 0.9;
	double invec[5] = {0.2,0.7,0.1,0.5,0.4};

	network * net = initialisation(data,invec);

	prop_to_F1(net,invec);
	//prop_to_F1(net,invec);
	prop_to_F2(net);
	learning(net,invec);

return 0;

}