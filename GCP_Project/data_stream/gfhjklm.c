#include <stdlib.h>
#include <stdio.h>
#include <string.h> 
char Mid(char* str, int pos){
    return str[pos];
}
int Trouve(char* str, char c){
    int pos = 0, len = strlen(str);
    while(pos < len && str[pos] != c){
        pos += 1;
    }
    return pos;
}
int main(){
    char bla[100], cod[100];
    char alpha[26] = {'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'};
    int i, pos;
    char let, cod_char;
    printf("Ecrire la variable à coder: ");
    scanf("%s", bla);
    for(i = 0; i < strlen(bla); i++){
        let = Mid(bla, i);
        if(let != 'Z'){
            pos = Trouve(alpha, let);
            cod_char = Mid(alpha, pos + 1);
            cod[i] = cod_char;
        }else
        cod[i] = 'A';
    }
    printf("Phrase initiale: %s\n", bla);
    for(i = 0; i < strlen(bla); i++) 
    bla[i] = cod[i];
    
    printf("La phrase codée: %s\n", bla);
    return 0;
}