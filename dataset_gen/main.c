// github.com/jcwml
// gcc main.c -lm -Ofast -o main
#include <stdio.h>
#include <sys/file.h>
#include <stdint.h>
#include <unistd.h>

#pragma GCC diagnostic ignored "-Wunused-result"

#define DATASET_SIZE 3333333
#define VECTOR_RANGE 10000000.f
#define DATASET_HIGH_QUALITY 1

#if DATASET_HIGH_QUALITY == 1
float randf()
{
    static const float RECIP_FLOAT_UINT64_MAX = 2.f/(float)UINT64_MAX;
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    uint64_t s = 0;
    read(f, &s, sizeof(uint64_t));
    close(f);
    return (((float)s) * RECIP_FLOAT_UINT64_MAX)-1.f;
}
#else
int urand_int()
{
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    int s = 0;
    read(f, &s, sizeof(int));
    close(f);
    return s;
}
int srandfq = 6543;
float randf()
{
    // https://www.musicdsp.org/en/latest/Other/273-fast-float-random-numbers.html
    // moc.liamg@seir.kinimod
    srandfq *= 16807;
    return (float)srandfq * 4.6566129e-010f; // -1 to 1
    // return (float)(srandfq & 0x7FFFFFFF) * 4.6566129e-010f; // 0-1
}
#endif

int main()
{
#if DATASET_HIGH_QUALITY == 0
    srandfq = urand_int();
#endif

    FILE* f = fopen("../dataset.dat", "w");
    if(f != NULL)
    {
        for(size_t i = 0; i < DATASET_SIZE; i++)
        {
            const float x = randf()*VECTOR_RANGE;
            const float y = randf()*VECTOR_RANGE;
            const float z = randf()*VECTOR_RANGE;
            
            if(fwrite(&x, 1, sizeof(float), f) < sizeof(float))
                printf("write error\n");
            if(fwrite(&y, 1, sizeof(float), f) < sizeof(float))
                printf("write error\n");
            if(fwrite(&z, 1, sizeof(float), f) < sizeof(float))
                printf("write error\n");
        }
        fclose(f);
    }

    f = fopen("../testset.dat", "w");
    if(f != NULL)
    {
        for(size_t i = 0; i < DATASET_SIZE; i++)
        {
            const float x = randf()*VECTOR_RANGE;
            const float y = randf()*VECTOR_RANGE;
            const float z = randf()*VECTOR_RANGE;
            
            if(fwrite(&x, 1, sizeof(float), f) < sizeof(float))
                printf("write error\n");
            if(fwrite(&y, 1, sizeof(float), f) < sizeof(float))
                printf("write error\n");
            if(fwrite(&z, 1, sizeof(float), f) < sizeof(float))
                printf("write error\n");
        }
        fclose(f);
    }

    return 0;
}
