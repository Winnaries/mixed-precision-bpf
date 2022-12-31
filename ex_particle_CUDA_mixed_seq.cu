#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <float.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define BLOCK_X 16
#define BLOCK_Y 16
#define PI 3.1415926535897932

/**
@var M value for Linear Congruential Generator (LCG); use GCC's value
 */
long M = INT_MAX;
/**
@var A value for LCG
 */
int A = 1103515245;
/**
@var C value for LCG
 */
int C = 12345;

/*****************************
 *GET_TIME
 *returns a long int representing the time
 *****************************/
long long get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

double elapsed_time(long long start_time, long long end_time)
{
    return (double)(end_time - start_time) / (1000 * 1000);
}

/*****************************
 * checkError
 * Checks for CUDA errors and prints them to the screen to help with
 * debugging of CUDA related programming
 *****************************/
void cudaCheck(cudaError e)
{
    if (e != cudaSuccess)
    {
        printf("\nCUDA error: %s\n", cudaGetErrorString(e));
        exit(1);
    }
}

/****************************
CDF CALCULATE
CALCULATES CDF
param1 CDF
param2 weights
param3 nParticles
 *****************************/
__device__ void cdfCalc(double *CDF, double *weights, int nParticles)
{
    int x;
    CDF[0] = weights[0];
    for (x = 1; x < nParticles; x++)
    {
        CDF[x] = weights[x] + CDF[x - 1];
    }
}

/**
 * Generates a uniformly distributed random number using the provided seed and GCC's settings for the Linear Congruential Generator (LCG)
 * @see http://en.wikipedia.org/wiki/Linear_congruential_generator
 * @note This function is thread-safe
 * @param seed The seed array
 * @param index The specific index of the seed to be advanced
 * @return a uniformly distributed number [0, 1)
 */
double randu(int *seed, int index)
{
    int num = A * seed[index] + C;
    seed[index] = num % M;
    return fabs(seed[index] / ((double)M));
}

/**
 * Generates a normally distributed random number using the Box-Muller transformation
 * @note This function is thread-safe
 * @param seed The seed array
 * @param index The specific index of the seed to be advanced
 * @return a double representing random number generated using the Box-Muller algorithm
 * @see http://en.wikipedia.org/wiki/Normal_distribution, section computing value for normal random distribution
 */
double randn(int *seed, int index)
{
    /*Box-Muller algorithm*/
    double u = randu(seed, index);
    double v = randu(seed, index);
    double cosine = cos(2 * PI * v);
    double rt = -2 * log(u);
    return sqrt(rt) * cosine;
}

template<typename T>
__device__ T deviceRound(T value) 
{
    int newValue = (int)value; 
    return value - (T)newValue < ((T)0.5) 
        ? newValue 
        : newValue + 1; 
}

/**
 * Takes in a double and returns an integer that approximates to that double
 * @return if the mantissa < .5 => return value < input value; else return value > input value
 */
double roundDouble(double value)
{
    int newValue = (int)(value);
    if (value - newValue < .5)
        return newValue;
    else
        return newValue++;
}

/**
 * Set values of the 3D array to a newValue if that value is equal to the testValue
 * @param testValue The value to be replaced
 * @param newValue The value to replace testValue with
 * @param array3D The image vector
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 */
void setIf(int testValue, int newValue, unsigned char *array3D, int *dimX, int *dimY, int *dimZ)
{
    int x, y, z;
    for (x = 0; x < *dimX; x++)
    {
        for (y = 0; y < *dimY; y++)
        {
            for (z = 0; z < *dimZ; z++)
            {
                if (array3D[x * *dimY * *dimZ + y * *dimZ + z] == testValue)
                    array3D[x * *dimY * *dimZ + y * *dimZ + z] = newValue;
            }
        }
    }
}

/**
 * Sets values of 3D matrix using randomly generated numbers from a normal distribution
 * @param array3D The video to be modified
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 * @param seed The seed array
 */
void addNoise(unsigned char *array3D, int *dimX, int *dimY, int *dimZ, int *seed)
{
    int x, y, z;
    for (x = 0; x < *dimX; x++)
    {
        for (y = 0; y < *dimY; y++)
        {
            for (z = 0; z < *dimZ; z++)
            {
                array3D[x * *dimY * *dimZ + y * *dimZ + z] = array3D[x * *dimY * *dimZ + y * *dimZ + z] + (unsigned char)(5 * randn(seed, 0));
            }
        }
    }
}

/**
 * Fills a radius x radius matrix representing the disk
 * @param disk The pointer to the disk to be made
 * @param radius  The radius of the disk to be made
 */
void strelDisk(int *disk, int radius)
{
    int diameter = radius * 2 - 1;
    int x, y;
    for (x = 0; x < diameter; x++)
    {
        for (y = 0; y < diameter; y++)
        {
            double distance = sqrt(pow((double)(x - radius + 1), 2) + pow((double)(y - radius + 1), 2));
            if (distance < radius)
                disk[x * diameter + y] = 1;
        }
    }
}

/**
 * Dilates the provided video
 * @param matrix The video to be dilated
 * @param posX The x location of the pixel to be dilated
 * @param posY The y location of the pixel to be dilated
 * @param poxZ The z location of the pixel to be dilated
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 * @param error The error radius
 */
void dilate_matrix(unsigned char *matrix, int posX, int posY, int posZ, int dimX, int dimY, int dimZ, int error)
{
    int startX = posX - error;
    while (startX < 0)
        startX++;
    int startY = posY - error;
    while (startY < 0)
        startY++;
    int endX = posX + error;
    while (endX > dimX)
        endX--;
    int endY = posY + error;
    while (endY > dimY)
        endY--;
    int x, y;
    for (x = startX; x < endX; x++)
    {
        for (y = startY; y < endY; y++)
        {
            double distance = sqrt(pow((double)(x - posX), 2) + pow((double)(y - posY), 2));
            if (distance < error)
                matrix[x * dimY * dimZ + y * dimZ + posZ] = 1;
        }
    }
}

/**
 * Dilates the target matrix using the radius as a guide
 * @param matrix The reference matrix
 * @param dimX The x dimension of the video
 * @param dimY The y dimension of the video
 * @param dimZ The z dimension of the video
 * @param error The error radius to be dilated
 * @param newMatrix The target matrix
 */
void imdilate_disk(unsigned char *matrix, int dimX, int dimY, int dimZ, int error, unsigned char *newMatrix)
{
    int x, y, z;
    for (z = 0; z < dimZ; z++)
    {
        for (x = 0; x < dimX; x++)
        {
            for (y = 0; y < dimY; y++)
            {
                if (matrix[x * dimY * dimZ + y * dimZ + z] == 1)
                {
                    dilate_matrix(newMatrix, x, y, z, dimX, dimY, dimZ, error);
                }
            }
        }
    }
}

/**
 * Fills a 2D array describing the offsets of the disk object
 * @param se The disk object
 * @param numOnes The number of ones in the disk
 * @param neighbors The array that will contain the offsets
 * @param radius The radius used for dilation
 */
void getneighbors(int *se, int numOnes, int *neighbors, int radius)
{
    int x, y;
    int neighY = 0;
    int center = radius - 1;
    int diameter = radius * 2 - 1;
    for (x = 0; x < diameter; x++)
    {
        for (y = 0; y < diameter; y++)
        {
            if (se[x * diameter + y])
            {
                neighbors[neighY * 2] = (int)(y - center);
                neighbors[neighY * 2 + 1] = (int)(x - center);
                neighY++;
            }
        }
    }
}

/**
 * The synthetic video sequence we will work with here is composed of a
 * single moving object, circular in shape (fixed radius)
 * The motion here is a linear motion
 * the foreground intensity and the background intensity is known
 * the image is corrupted with zero mean Gaussian noise
 * @param I The video itself
 * @param IszX The x dimension of the video
 * @param IszY The y dimension of the video
 * @param Nfr The number of frames of the video
 * @param seed The seed array used for number generation
 */
void videoSequence(unsigned char *I, int IszX, int IszY, int Nfr, int *seed)
{
    int k;
    int max_size = IszX * IszY * Nfr;
    /*get object centers*/
    int x0 = (int)roundDouble(IszY / 2.0);
    int y0 = (int)roundDouble(IszX / 2.0);
    I[x0 * IszY * Nfr + y0 * Nfr + 0] = 1;

    FILE *fp = fopen("true_path.txt", "w");
    fprintf(fp, "TRUE MOVEMENT: \n");

    /*move point*/
    int xk, yk, pos;
    for (k = 1; k < Nfr; k++)
    {
        xk = abs(x0 + (k - 1));
        yk = abs(y0 - 2 * (k - 1));
        fprintf(fp, ". <%d, %d>\n", xk, yk);
        pos = yk * IszY * Nfr + xk * Nfr + k;
        if (pos >= max_size)
            pos = 0;
        I[pos] = 1;
    }

    fprintf(fp, "end\n");

    /*dilate matrix*/
    unsigned char *newMatrix = (unsigned char *)malloc(sizeof(unsigned char) * IszX * IszY * Nfr);
    imdilate_disk(I, IszX, IszY, Nfr, 5, newMatrix);
    int x, y;
    for (x = 0; x < IszX; x++)
    {
        for (y = 0; y < IszY; y++)
        {
            for (k = 0; k < Nfr; k++)
            {
                I[x * IszY * Nfr + y * Nfr + k] = newMatrix[x * IszY * Nfr + y * Nfr + k];
            }
        }
    }
    free(newMatrix);

    /*define background, add noise*/
    setIf(0, 100, I, &IszX, &IszY, &Nfr);
    setIf(1, 228, I, &IszX, &IszY, &Nfr);
    /*add noise*/
    addNoise(I, &IszX, &IszY, &Nfr, seed);
}

/**
 * A set of inline overload forwarders.
 * Gsqrt short for generic square root.
 */
__device__ inline nv_half Gsqrt(nv_half x) { return hsqrt(x); }
__device__ inline float Gsqrt(float x) { return sqrt(x); }
__device__ inline double Gsqrt(double x) { return sqrt(x); }

/**
 * A set of inline overload forwarders.
 * Gexp short for generic square root.
 */
__device__ inline nv_half Gexp(nv_half x) { return hexp(x); }
__device__ inline float Gexp(float x) { return exp(x); }
__device__ inline double Gexp(double x) { return exp(x); }

/**
 * Initialize cuRAND state for random number generation.
 * Every threads received the same seed, but different sequence.
 * @param states The curandState array for each threads.
 * @param nStates The number of states
 * @param seed The generator seed
 */
__global__ void curandSetupKernel(curandState *states, int nStates, int seed)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nStates)
        curand_init(seed, idx, 0, &states[idx]);
}

/**
 * Propagate all particles to the next steps.
 * Sampling from true distribution and all model
 * parameters are assumed to be known.
 * @param states the cuRAND state arrays for each threads
 * @param X the X coordinates of current particles
 * @param Y the Y coordinates of current particles
 * @param Ax the X coordinates of the selected ancestors
 * @param Ay the Y coordinates of the selected ancestors
 */
template <typename T>
__global__ void propagationKernel(curandState *states, T *X, T *Y, T *Ax, T *Ay, int nParticles)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    curandState localState = states[idx];

    T sigmaX = 5.0;
    T sigmaY = 2.0;
    T muX = 1.0;
    T muY = 2.0;

    if (idx < nParticles)
    {
        X[idx] = Ax[idx] + muX + sigmaX * (T)curand_normal(&localState);
        Y[idx] = Ay[idx] - muY + sigmaY * (T)curand_normal(&localState);
    }

    states[idx] = localState;
}

/**
 * Evaluate the likelihood of each particles
 * giving the observation at frame T.
 * @param X 
 * @param Y
 * @param nParticles
 * @param objXy
 * @param countOnes
 * @param likelihood
 * @param I
 * @param IszY
 * @param Nfr
 * @param maxSize
 */
template <typename T>
__global__ void likelihoodKernel(T *X, T *Y, int nParticles, int *objXy, int countOnes, T *likelihood, unsigned char *I, int IszY, int Nfr, int maxSize, int frIdx)
{
    int k, i, indX, indY, idx = blockDim.x * blockIdx.x + threadIdx.x;
    T fg, bg; 

    if (idx < nParticles)
    {
        T localLL = 0.0; 

        for (k = 0; k < countOnes; k++) {
            /**
             * BREAKING: In the original version, 
             * the objXy is y-first, but here I changed it
            */
            indX = (int)deviceRound(X[idx]) + objXy[k * 2];
            indY = (int)deviceRound(Y[idx]) + objXy[k * 2 + 1];
            i = abs(indY * IszY * Nfr + indX * Nfr + frIdx); 
            i = i < maxSize ? i : 0;
            bg = (T)(I[i] - 100) / Gsqrt((T)countOnes * (T)50.0);
            fg = (T)(I[i] - 228) / Gsqrt((T)countOnes * (T)50.0);
            localLL += bg * bg - fg * fg;
        }

        likelihood[idx] = localLL;
        // printf("%d -> %f\n", frIdx, likelihood[idx]);
    }
}

/**
 * Recalculate particle weights and normalize them. 
 * This function operate on higher-precision type 
 * to preserve numerical stability. 
 * @note This function requires double precision
 * @param likelihood The particles likelihood
 * @param weights The particles weights
 * @param sum The reference to sum on global mem
 * @param nParticles The number of particles
*/
template <typename T>
__global__ void weightingKernel(T *likelihood, T *weights, T *cdf, double *sum, int nParticles)
{
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x; 
    int localIdx = threadIdx.x; 
    double unnormalized; 

    extern __shared__ double buffer[]; 

    if (globalIdx < nParticles) {
        unnormalized = (double)weights[globalIdx] * Gexp((double)likelihood[globalIdx] + 100.0);
        buffer[localIdx] = unnormalized; 
    }

    __syncthreads(); 

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            buffer[localIdx] += buffer[localIdx + s]; 
        __syncthreads();
    }

    if (localIdx == 0) {
        atomicAdd(sum, buffer[0]); 
    }

    __syncthreads(); 

    weights[globalIdx] = (T)(unnormalized / (*sum)); 

    if (globalIdx == 0) {
        int x; 
        cdf[0] = weights[0]; 
        for (x = 1; x < nParticles; x++) {
            cdf[x] = weights[x] + cdf[x - 1]; 
        }
    }
        
}

/**
 * Resamples particles from a set of selected 
 * ancestors using a systematic resmapling scheme. 
 * @param states cuRAND states for each threads
 * @param X The X coordinate of current particles
 * @param Y The Y coordinate of current particles
 * @param Ax The X coordinate of ancestors for next gen
 * @param Ax The Y coordinate of ancestors for next gen
 * @param cdf The CDF of normalized weights to resample from
 * @param u The uniform random variable needed for resampling
 * @param nParticles The number of particles in the simulation
*/
template <typename T>
__global__ void resamplingKernel(curandState *states, T *X, T *Y, T *Ax, T *Ay, T *weights, T *cdf, T *u, int nParticles) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    if (idx == 0) {
        curandState localState = states[idx]; 
        u[0] = (T)curand_uniform(&localState) / (T)nParticles; 
        states[idx] = localState; 
    }

    __syncthreads(); 

    if (idx != 0 && idx < nParticles) 
    {
        u[idx] = u[0] + (T)idx / (T)nParticles;
    }

    if (idx < nParticles)
    {
        int ancestor = -1; 
        int x; 

        for (x = 0; x < nParticles; x++) {
            if (cdf[x] >= u[idx]) {
                ancestor = x; 
                break; 
            }
        }

        if (ancestor == -1) {
            ancestor = nParticles - 1; 
        }

        Ax[idx] = X[ancestor]; 
        Ay[idx] = Y[ancestor]; 
        weights[idx] = (T)1 / (T)nParticles; 
    }
}

/**
 * The implementation of the particle filter using OpenMP for many frames
 * @see http://openmp.org/wp/
 * @note This function is designed to work with a video of several frames. In addition, it references a provided MATLAB function which takes the video, the objxy matrix and the x and y arrays as arguments and returns the likelihoods
 * @param I The video to be run
 * @param IszX The x dimension of the video
 * @param IszY The y dimension of the video
 * @param Nfr The number of frames
 * @param seed The seed array used for random number generation
 * @param nParticles The number of particles to be used
 */
template <typename T>
void particleFilter(unsigned char *I, int IszX, int IszY, int Nfr, int *seed, int nParticles)
{
    curandState *states;
    int threadsPerBlocks = 128;
    int numBlocks = (nParticles - 1) / threadsPerBlocks + 1;
    int totalThreads = numBlocks * threadsPerBlocks;

    /**
     * Initial setup for the particle filter.
     */
    int maxSize = IszX * IszY * Nfr;
    T xe = (T)roundDouble(IszY / 2.0);
    T ye = (T)roundDouble(IszX / 2.0);

    /**
     * Target object template needed for
     * likelihood calculation
     */
    int radius = 5;
    int diameter = radius * 2 - 1;
    int *disk = (int *)malloc(diameter * diameter * sizeof(int));
    strelDisk(disk, radius);

    /**
     * Object template preprocessing.
     * Counting entries with one.
     */
    int x, y;
    int countOnes = 0;
    for (x = 0; x < diameter; x++)
    {
        for (y = 0; y < diameter; y++)
        {
            if (disk[x * diameter + y] == 1)
                countOnes++;
        }
    }

    /**
     * Collecting indices around the center
     * of object to test against.
     */
    int *objxy = (int *)malloc(countOnes * 2 * sizeof(int));
    getneighbors(disk, countOnes, objxy, radius);

    /**
     * Variables allocations ranging from
     * ancestorsXY, currentXY, uniformRV,
     * weights, likelihood, indices along with
     * device copies.
     */
    T *Ax = (T *)malloc(nParticles * sizeof(T));
    T *Ay = (T *)malloc(nParticles * sizeof(T));
    T *X = (T *)malloc(nParticles * sizeof(T));
    T *Y = (T *)malloc(nParticles * sizeof(T));
    T *U = (T *)malloc(nParticles * sizeof(T));

    T *cdf = (T *)malloc(nParticles * sizeof(T));
    T *weights = (T *)malloc(nParticles * sizeof(T));
    T *likelihood = (T *)malloc(nParticles * sizeof(T));

    T *deviceAx, *deviceAy, *deviceX, *deviceY, *deviceU;
    T *deviceLikelihood, *deviceCdf, *deviceWeights;
    double *deviceSum;

    int *deviceObjXy;
    unsigned char *deviceI;

    // Uniformly initialize weights
    for (x = 0; x < nParticles; x++)
    {
        weights[x] = (T)(1 / (double)nParticles);
    }

    // All particles begins at the center
    for (x = 0; x < nParticles; x++)
    {
        Ax[x] = (T)xe;
        Ay[x] = (T)ye;
    }

    /**
     * Memory allocations for various
     * variables listed above.
     */
    cudaCheck(cudaMalloc(&deviceAx, nParticles * sizeof(T)));
    cudaCheck(cudaMalloc(&deviceAy, nParticles * sizeof(T)));
    cudaCheck(cudaMalloc(&deviceX, nParticles * sizeof(T)));
    cudaCheck(cudaMalloc(&deviceY, nParticles * sizeof(T)));
    cudaCheck(cudaMalloc(&deviceU, nParticles * sizeof(T)));

    cudaCheck(cudaMalloc(&deviceSum, sizeof(double)));
    cudaCheck(cudaMalloc(&deviceCdf, nParticles * sizeof(T)));
    cudaCheck(cudaMalloc(&deviceWeights, nParticles * sizeof(T)));
    cudaCheck(cudaMalloc(&deviceObjXy, 2 * countOnes * sizeof(int)));
    cudaCheck(cudaMalloc(&deviceI, IszX * IszY * Nfr * sizeof(unsigned char)));
    cudaCheck(cudaMalloc(&states, totalThreads * sizeof(curandState)));

    cudaCheck(cudaMalloc(&deviceLikelihood, nParticles * sizeof(T)));
    cudaCheck(cudaMemset(deviceLikelihood, 0, nParticles * sizeof(T)));

    /**
     * Memory copy from host to device
     * for various variables.
     */
    int k, r;
    long long sendStart = get_time();
    cudaCheck(cudaMemcpy(deviceI, I, IszX * IszY * Nfr * sizeof(unsigned char), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(deviceObjXy, objxy, 2 * countOnes * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(deviceWeights, weights, nParticles * sizeof(T), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(deviceAx, Ax, nParticles * sizeof(T), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(deviceAy, Ay, nParticles * sizeof(T), cudaMemcpyHostToDevice));
    long long sendEnd = get_time();

    curandSetupKernel<<<numBlocks, threadsPerBlocks>>>(states, totalThreads, 1234);

#ifdef TRACE
    FILE *fp = fopen("mixed_precision_trace.txt", "w"); 
#endif

    
    int sharedMemSize = threadsPerBlocks * sizeof(double); 
    for (r = 1; r < Nfr; r++)
    {
        propagationKernel<<<numBlocks, threadsPerBlocks>>>(states, deviceX, deviceY, deviceAx, deviceAy, nParticles);
        likelihoodKernel<<<numBlocks, threadsPerBlocks>>>(deviceX, deviceY, nParticles, deviceObjXy, countOnes, deviceLikelihood, deviceI, IszY, Nfr, maxSize, r);
        weightingKernel<<<numBlocks, threadsPerBlocks, sharedMemSize>>>(deviceLikelihood, deviceWeights, deviceCdf, deviceSum, nParticles);
#ifdef TRACE
        cudaCheck(cudaMemcpy(likelihood, deviceLikelihood, nParticles * sizeof(T), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(weights, deviceWeights, nParticles * sizeof(T), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(cdf, deviceCdf, nParticles * sizeof(T), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(Ax, deviceAx, nParticles * sizeof(T), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(Ay, deviceAy, nParticles * sizeof(T), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(X, deviceX, nParticles * sizeof(T), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(Y, deviceY, nParticles * sizeof(T), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(U, deviceU, nParticles * sizeof(T), cudaMemcpyDeviceToHost));

        // fprintf(fp, "\nFRAME %d\n", r); 
        // for (k = 0; k < nParticles; k++)
        // {
        //     fprintf(fp, "<%5.2f, %5.2f> || <%5.2f, %5.2f> ~ %10.4f ~ %10.4f ^+ %10.4f ~ %10.4f \n",
        //             (double)Ax[k], (double)Ay[k], (double) X[k], (double)Y[k], 
        //             (double)likelihood[k], (double)weights[k], 
        //             (double)cdf[k], (double)U[k]);
        // }

        double xHat = 0.0; 
        double yHat = 0.0; 

        for (k = 0; k < nParticles; k++) {
            xHat += (double)weights[k] * (double)X[k];
            yHat += (double)weights[k] * (double)Y[k];
        }

        fprintf(fp, "END FRAME %d ~ Predicted Position = <%f,%f>\n", 
                r, xHat, yHat);
#endif
        resamplingKernel<<<numBlocks, threadsPerBlocks>>>(states, deviceX, deviceY, deviceAx, deviceAy, deviceWeights, deviceCdf, deviceU, nParticles);

    }

#ifdef TRACE
    fclose(fp);
#endif

    cudaDeviceSynchronize();

    long long backTime = get_time();
    cudaCheck(cudaMemcpy(likelihood, deviceLikelihood, nParticles * sizeof(T), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(weights, deviceWeights, nParticles * sizeof(T), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(cdf, deviceCdf, nParticles * sizeof(T), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(X, deviceX, nParticles * sizeof(T), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(Y, deviceY, nParticles * sizeof(T), cudaMemcpyDeviceToHost));
    long long freeTime = get_time();

    printf("TIME TO SEND H2D: %lf\n", elapsed_time(sendStart, sendEnd));
    printf("GPU Execution: %lf\n", elapsed_time(sendEnd, backTime));
    printf("TIME TO SEND D2H: %lf\n", elapsed_time(backTime, freeTime));

    /**
     * Memory allocations for various
     * variables listed above.
     */
    cudaCheck(cudaFree(deviceAx));
    cudaCheck(cudaFree(deviceAy));
    cudaCheck(cudaFree(deviceX));
    cudaCheck(cudaFree(deviceY));
    cudaCheck(cudaFree(deviceU));

    cudaCheck(cudaFree(deviceCdf));
    cudaCheck(cudaFree(deviceWeights));
    cudaCheck(cudaFree(deviceObjXy));
    cudaCheck(cudaFree(deviceSum));
    cudaCheck(cudaFree(deviceI));
    cudaCheck(cudaFree(deviceLikelihood));
    cudaCheck(cudaFree(states));

    cudaCheck(cudaDeviceSynchronize()); 
}

int main(int argc, char *argv[])
{

    const char *usage = "double.out -x <dimX> -y <dimY> -z <Nfr> -np <nParticles>";
    // check number of arguments
    if (argc != 9)
    {
        printf("%s\n", usage);
        return 0;
    }
    // check args deliminators
    if (strcmp(argv[1], "-x") || strcmp(argv[3], "-y") || strcmp(argv[5], "-z") || strcmp(argv[7], "-np"))
    {
        printf("%s\n", usage);
        return 0;
    }

    int IszX, IszY, Nfr, nParticles;

    // converting a string to a integer
    if (sscanf(argv[2], "%d", &IszX) == EOF)
    {
        printf("ERROR: dimX input is incorrect");
        return 0;
    }

    if (IszX <= 0)
    {
        printf("dimX must be > 0\n");
        return 0;
    }

    // converting a string to a integer
    if (sscanf(argv[4], "%d", &IszY) == EOF)
    {
        printf("ERROR: dimY input is incorrect");
        return 0;
    }

    if (IszY <= 0)
    {
        printf("dimY must be > 0\n");
        return 0;
    }

    // converting a string to a integer
    if (sscanf(argv[6], "%d", &Nfr) == EOF)
    {
        printf("ERROR: Number of frames input is incorrect");
        return 0;
    }

    if (Nfr <= 0)
    {
        printf("number of frames must be > 0\n");
        return 0;
    }

    // converting a string to a integer
    if (sscanf(argv[8], "%d", &nParticles) == EOF)
    {
        printf("ERROR: Number of particles input is incorrect");
        return 0;
    }

    if (nParticles <= 0)
    {
        printf("Number of particles must be > 0\n");
        return 0;
    }

    // establish seed
    int *seed = (int *)malloc(sizeof(int) * nParticles);
    int i;
    for (i = 0; i < nParticles; i++)
        seed[i] = time(0) * i;
    // malloc matrix
    unsigned char *I = (unsigned char *)malloc(sizeof(unsigned char) * IszX * IszY * Nfr);
    long long start = get_time();
    // call video sequence
    videoSequence(I, IszX, IszY, Nfr, seed);
    long long endVideoSequence = get_time();
    printf("VIDEO SEQUENCE TOOK %f\n", elapsed_time(start, endVideoSequence));
    // call particle filter
    particleFilter<half>(I, IszX, IszY, Nfr, seed, nParticles);
    long long endParticleFilter = get_time();
    printf("PARTICLE FILTER TOOK %f\n", elapsed_time(endVideoSequence, endParticleFilter));
    printf("ENTIRE PROGRAM TOOK %f\n", elapsed_time(start, endParticleFilter));

    free(seed);
    free(I);
    return 0;
}
