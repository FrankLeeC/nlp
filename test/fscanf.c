#include<stdio.h>
#include<stdlib.h>

int main() {
    FILE* f = fopen("./test.txt", "r");
    if (NULL == f) {
        printf("%s\n", "no such file!");
        return -2;
    }
    char a;
    fscanf(f, "%s", &a);
    printf("---%s\n", a);
    fclose(f);
    return 0;
}