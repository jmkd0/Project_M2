#include <stdlib.h>
#include <stdio.h>
#include <string.h> 

int Trouve(char* str, char c){
    int pos = 0, len = strlen(str);
    while(pos < len){
        if(str[pos] == c)
            return 1;
        else
            pos += 1;
    }
    return 0;
}
int NbVoyelles(char* str){
    char* voyelles = "ayeouis";
    int nb = 0;
    for(int i = 0; i < strlen(str); i ++){
        if(Trouve(voyelles, str[i]))
        nb += 1;
    }
    return nb;
}
int main(){
    char* str = "mange avec moi!";
    int nb = NbVoyelles(str);
    printf("Nombre de voyelles: %d\n", nb);
    return 0;
}