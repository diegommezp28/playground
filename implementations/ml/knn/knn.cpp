#include <stdio.h>
#include "knn.h"

inline __attribute__((always_inline)) void knn_2()
{
    int i = 2 + 3;
    printf("knn_2 %d\n", i);
}

// int main()
// {
//     knn();
//     knn_2();
//     return 0;
// }