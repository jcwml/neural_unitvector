
// github.com/jcwml
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <sys/time.h>

#include <x86intrin.h>

#define NUM_ITERATIONS 1000000000

uint64_t microtime()
{
    struct timeval tv;
    struct timezone tz;
    memset(&tz, 0, sizeof(struct timezone));
    gettimeofday(&tv, &tz);
    return 1000000 * tv.tv_sec + tv.tv_usec;
}

// https://en.wikipedia.org/wiki/Fast_inverse_square_root
float InvSqrt(float x)
{
    float xhalf = 0.5f * x;
    int i = *(int*)&x;            // store floating-point bits in integer
    i = 0x5f3759df - (i >> 1);    // initial guess for Newton's method
    x = *(float*)&i;              // convert new bits into float
    x = x*(1.5f - xhalf*x*x);     // One round of Newton's method
    return x;
}

float nx, ny, nz;
void norm(float x, float y, float z)
{
    const float len = 1.f/sqrtf(x*x + y*y + z*z);
    nx = x * len;
    ny = y * len;
    nz = z * len;
}
float nx1, ny1, nz1;
void norm_inv(float x, float y, float z)
{
    const float len = InvSqrt(x*x + y*y + z*z);
    nx1 = x * len;
    ny1 = y * len;
    nz1 = z * len;
}
float nx2, ny2, nz2;
void norm_intrin(float x, float y, float z)
{
    const float len = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x*x + y*y + z*z)));
    nx2 = x * len;
    ny2 = y * len;
    nz2 = z * len;
}

// tanh_adam_0_16_128_3333333_6_[0.90]
const float nv0[] = {-0.39879575,-0.48412916,0.450909, /* bias */ 0.25869402,0.27027094,-0.06547201,0.4244762, /* bias */ -0.3684898,0.64823955,-0.45263678,-0.29404083, /* bias */ 0.29431447,-0.37444034,0.35139787,-0.37529132, /* bias */ 0.24774797,0.09204272,-1.0231098,0.14390624, /* bias */ 0.0003230142,-0.17254636,-0.32850498,0.291209, /* bias */ 0.1735064,-0.4576754,-0.76062876,-0.43811232, /* bias */ -0.638915,-0.4367048,-0.253564,-0.0193794, /* bias */ -0.056036152,-0.2655969,-0.059854627,0.1378513, /* bias */ 0.24290895,0.62050456,-0.9127042,0.1934067, /* bias */ 0.0,0.41116375,-0.11640373,0.44771346, /* bias */ 2.4103256e-06,0.3375145,-0.48310238,0.38121364, /* bias */ 0.12730102,-0.29908195,0.52429354,-0.28219193, /* bias */ 1.0750377,-0.11793279,0.6446215,-0.44602153, /* bias */ 0.7545985,0.13627762,-0.45721877,0.21879677, /* bias */ 0.0,0.20851132,0.48091263,0.9139804, /* bias */ -0.92144644};
const float nv1[] = {-0.04304739,-0.040557653,0.06273674,-0.062496893,0.030465797,0.03307364,0.06333267,-0.047295507,-0.06078761,0.01841037,-0.087970056,0.034147475,-0.09294898,-0.57730436,-0.2817628,0.04295613, /* bias */ -0.008993808,-0.058301527,0.06213507,0.08108177,-0.020443933,-0.029178683,-0.12463052,-0.19275494,-0.36155888,-0.021774864,-0.021767603,0.08170835,-0.3155085,0.28161997,0.03290635,-0.043577507,-0.2719435, /* bias */ 0.24550056,0.049762726,-0.02716966,0.061025236,-0.040225115,0.0017402614,-0.10296145,0.023425946,-0.098629765,0.0071707135,0.016772764,-0.026235979,-0.012678294,-0.2866619,-0.005575278,-0.0024562972,0.08665302, /* bias */ 0.028602168};
float nx3, ny3, nz3;
void norm_neural(float x, float y, float z)
{
    float h[16];
    for(int i = 0; i < 16; i++)
    {
        const int j = i*4;
        h[i] = (nv0[j] * x) + (nv0[j+1] * y) + (nv0[j+2] * z) + nv0[j+3];
    }
    float o[3];
    for(int i = 0; i < 3; i++)
    {
        const int j = i*17;
        for(int k = 0; k < 17; k++)
            o[i] += (nv0[j+k] * x);
        o[i] += nv0[j+17];
    }

    nx3 = o[0];
    ny3 = o[1];
    nz3 = o[2];
}

float dist(float x1, float y1, float z1, float x2, float y2, float z2)
{
    const float xm = (x1 - x2);
    const float ym = (y1 - y2);
    const float zm = (z1 - z2);
    return sqrtf(xm*xm + ym*ym + zm*zm);
}

int srandfq = 6543;
float randf()
{
    // https://www.musicdsp.org/en/latest/Other/273-fast-float-random-numbers.html
    // moc.liamg@seir.kinimod
    srandfq *= 16807;
    return (float)(srandfq & 0x7FFFFFFF) * 4.6566129e-010f;
}

int main()
{
    float antioptim = 0.f;
    uint64_t stm, st, stf, stmf;

    ///

    printf("Speed Test\n");

    ///

    stm = microtime();
    st  = __rdtsc();

    for(uint i = 0; i < NUM_ITERATIONS; i++)
    {
        norm_neural(randf()*10000000, randf()*10000000, randf()*10000000);
        antioptim += nx3+ny3+nz3;
    }

    stf  = __rdtsc()-st;
    stmf = microtime()-stm;

    printf(":: norm_neural() :: %'lu μs, %'lu Cycles\n", stmf, stf);

    ///

    stm = microtime();
    st  = __rdtsc();

    for(uint i = 0; i < NUM_ITERATIONS; i++)
    {
        norm(randf()*10000000, randf()*10000000, randf()*10000000);
        antioptim += nx+ny+nz;
    }

    stf  = __rdtsc()-st;
    stmf = microtime()-stm;

    printf(":: norm()        :: %'lu μs, %'lu Cycles\n", stmf, stf);

    ///

    stm = microtime();
    st  = __rdtsc();

    for(uint i = 0; i < NUM_ITERATIONS; i++)
    {
        norm_inv(randf()*10000000, randf()*10000000, randf()*10000000);
        antioptim += nx1+ny1+nz1;
    }

    stf  = __rdtsc()-st;
    stmf = microtime()-stm;

    printf(":: norm_inv()    :: %'lu μs, %'lu Cycles\n", stmf, stf);

    ///

    stm = microtime();
    st  = __rdtsc();

    for(uint i = 0; i < NUM_ITERATIONS; i++)
    {
        norm_intrin(randf()*10000000, randf()*10000000, randf()*10000000);
        antioptim += nx2+ny2+nz2;
    }

    stf  = __rdtsc()-st;
    stmf = microtime()-stm;

    printf(":: norm_intrin() :: %'lu μs, %'lu Cycles\n", stmf, stf);

    ///

    printf("\nAccuracy Test\n");

    ///

    float accuracy_inv = 0.f;
    float accuracy_intrin = 0.f;
    float accuracy_neural = 0.f;
    for(uint i = 0; i < NUM_ITERATIONS; i++)
    {
        norm(randf()*10000000, randf()*10000000, randf()*10000000);
        norm_inv(randf()*10000000, randf()*10000000, randf()*10000000);
        norm_intrin(randf()*10000000, randf()*10000000, randf()*10000000);
        norm_neural(randf()*10000000, randf()*10000000, randf()*10000000);

        // we will assume norm() is the most accurate
        accuracy_inv += dist(nx, ny, nz, nx1, ny1, nz1);
        accuracy_intrin += dist(nx, ny, nz, nx2, ny2, nz2);
        accuracy_neural += dist(nx, ny, nz, nx3, ny3, nz3);
    }
    accuracy_inv /= NUM_ITERATIONS;
    accuracy_intrin /= NUM_ITERATIONS;
    accuracy_neural /= NUM_ITERATIONS;

    printf("InvSqrt: %.3f\nIntrinsic: %.3f\nNeural: %.3f\n", accuracy_inv, accuracy_intrin, accuracy_neural);

    ///

    printf("\n%f\n", antioptim);
    return 0;
}
