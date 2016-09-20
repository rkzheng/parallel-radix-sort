#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>

#define BLOCK_SIZE 512
#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }


typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}



//Split based on each bit
__global__ void split(unsigned int*in_d, unsigned int *out_d, unsigned int in_size,int bit_shift) {
    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
    int bit = 0;
    if (index < in_size) {
        bit = in_d[index] & (1 << bit_shift);// get the value on each bit
        bit = (bit > 0) ? 1 : 0;
        out_d[index] = 1 - bit;
    }

}

__global__ void exclusiveScan(unsigned int *out, unsigned int* in, unsigned int*sum, unsigned int inputSize) {
    __shared__ unsigned int temp[2 * BLOCK_SIZE];
    int start = 2 * blockIdx.x * blockDim.x;
    int tx = threadIdx.x;
    int index = 0;
    if (start + tx < inputSize) {
        temp[tx] = in[start + tx];
    } else {
        temp[tx] = 0;
    }
    if (start + tx + blockDim.x < inputSize) {
        temp[tx + blockDim.x] = in[start + tx + blockDim.x];
    } else {
        temp[tx + blockDim.x] = 0;
    }

    __syncthreads();
    // up-sweep phase
    int stride = 1;
    while(stride <= blockDim.x) {
        index = (tx + 1) * 2 * stride - 1;
        if (index < (2 * blockDim.x)) {
              temp[index] += temp[index - stride];
        }
        stride *= 2;
        __syncthreads();
    }
    // first store the reduction sum in sum array
    // make it zero since it is exclusive scan
    if (tx == 0) {
        // sum array contains the prefix sum of each
        // 2*blockDim blocks of element..
        if (sum != NULL) { 
            sum[blockIdx.x] = temp[2*blockDim.x - 1];
        }
        temp[2*blockDim.x - 1] = 0; 
    }
    //wait for thread zero to write
    __syncthreads();
 
    stride = blockDim.x;
    index = 0;
    unsigned int var = 0;
    while(stride > 0) {
        index = (2 * stride * (tx + 1)) - 1;
        if (index < 2 * blockDim.x) {
            var = temp[index];
            temp[index] += temp[index - stride];
            temp[index-stride] = var;
        }
        stride >>= 1;
        __syncthreads();
    }

    // write the temp array to output
    if (start + tx < inputSize) {
        out[start + tx] = temp[tx];   
    }
    if(start + tx + blockDim.x < inputSize) {
        out[start + tx + blockDim.x] = temp[tx + blockDim.x];    
    }
}
// merge the scan blocks
__global__ void mergeScanBlocks(unsigned int *sum, unsigned int* output, int opSize) {
    int index = 2 * blockDim.x * blockIdx.x + threadIdx.x;
    if (index < opSize) {
        output[index] += sum[blockIdx.x]; 
    }
    if (index + blockDim.x < opSize) {
        output[index + blockDim.x] += sum[blockIdx.x];    
    } 

}

void preScan(unsigned int *out, unsigned int *in, unsigned int in_size)
{

    
    unsigned int numBlocks1 = in_size / BLOCK_SIZE;
    if (in_size % BLOCK_SIZE) numBlocks1++;
   
    int numBlocks2 = numBlocks1 / 2;
    if(numBlocks1 % 2) numBlocks2++;
    dim3 dimThreadBlock;
    dimThreadBlock.x = BLOCK_SIZE;
    dimThreadBlock.y = 1;
    dimThreadBlock.z = 1;

    dim3 dimGrid;
    dimGrid.x = numBlocks2;
    dimGrid.y = 1;
    dimGrid.z = 1;

    unsigned int*sumArr_d = NULL;
    if (in_size > (2*BLOCK_SIZE)) {
        // we need the sum auxilarry  array only if numblocks2 > 1
        cudaMalloc((void**)&sumArr_d, numBlocks2 * sizeof(unsigned int));
       cudaCheckError();
    }
    exclusiveScan<<<dimGrid, dimThreadBlock>>>(out, in, sumArr_d, in_size);
    cudaDeviceSynchronize();
    cudaCheckError();
    if (in_size <= (2*BLOCK_SIZE)) {
        // out has proper exclusive scan. return
        return;
    } else {
        // now we need to perform exclusive scan on the auxilliary sum array
        unsigned int *sumArr_scan_d;
        cudaMalloc((void**)&sumArr_scan_d, numBlocks2 * sizeof(unsigned int));
        cudaCheckError();
        preScan(sumArr_scan_d, sumArr_d, numBlocks2);
        // sumAdd_scan_d now contains the exclusive scan op of individual blocks
        // now just do a one-one addition of blocks
        mergeScanBlocks<<<dimGrid,dimThreadBlock>>>(sumArr_scan_d, out, in_size);
        cudaDeviceSynchronize();
        cudaCheckError();
		 cudaFree(sumArr_d); 
		 cudaFree(sumArr_scan_d); 
	 }
 }
//Define the destination index
__global__ void indexDefine(unsigned int *in_d, unsigned int *rev_bit_d, 
unsigned int in_size, unsigned int last_input) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    int total_falses = in_d[in_size - 1] + last_input;
    __syncthreads();
    if (index < in_size) {
        if (rev_bit_d[index] == 0) {
            int val = in_d[index];
            in_d[index] = index + 1 - val + total_falses;
        }
    }

}
//Scatter input using in_d address
__global__ void scatterElements(unsigned int *in_d, unsigned int *index_d, unsigned int *out_d, unsigned int in_size) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < in_size) {
        unsigned int val = index_d[index];
        if (val < in_size) {
            out_d[val] = in_d[index];
        }
    }

}

 void radix_sort(unsigned int *in_d, unsigned int *out_d, unsigned int *out_scan_d, unsigned int *in_h,unsigned int *out_scan_h, int num_elements) {
    
	
    unsigned int *temp;
    dim3 dimThreadBlock;
    dimThreadBlock.x = BLOCK_SIZE;
    dimThreadBlock.y = 1;
    dimThreadBlock.z = 1;

    dim3 dimGrid;
    dimGrid.x =(int)(ceil(num_elements/(1.0 * dimThreadBlock.x)));
    dimGrid.y = 1;
    dimGrid.z = 1; 
	
    for (int i =0;i<32;i++) {
	
        split<<<dimGrid, dimThreadBlock>>>(in_d,out_d,num_elements,i);
        cudaDeviceSynchronize();
        cudaCheckError();
        preScan(out_scan_d, out_d, num_elements);
        cudaDeviceSynchronize();
        cudaCheckError();
        indexDefine<<<dimGrid, dimThreadBlock>>>(out_scan_d, out_d, num_elements, in_h[num_elements - 1]);
        cudaDeviceSynchronize();
        cudaCheckError();
        scatterElements<<<dimGrid, dimThreadBlock>>>(in_d, out_scan_d, out_d, num_elements);
        cudaDeviceSynchronize();
        cudaCheckError();
		
        // swap pointers
        temp = in_d;
        in_d = out_d;
        out_d = temp;
    }
}

int compare(const void *a, const void *b) {
    int a1 = *((unsigned int*)a);
    int b1 = *((unsigned int*)b);
    if (a1 == b1) return 0;
    else if (a1 < b1) return -1; 
    else return 1;
}

int main(){
	
	Timer timer;
	
    unsigned int *in_h;
    unsigned int *out_h;
    unsigned int *out_d;
    unsigned int *in_d;
    unsigned int *out_scan_d;
    unsigned int num_elements = 1000000;
   
    in_h = (unsigned int*) malloc(num_elements*sizeof(unsigned int));
    out_h = (unsigned int*) malloc(num_elements*sizeof(unsigned int));
	unsigned int *out_scan_h = (unsigned int *)malloc(num_elements * sizeof(unsigned int));
	
    cudaMalloc((void**)&in_d, num_elements * sizeof(unsigned int));
    cudaCheckError();
    cudaMalloc((void**)&out_d, num_elements * sizeof(unsigned int));
    cudaCheckError();
    cudaMalloc((void**)&out_scan_d, num_elements * sizeof(unsigned int ));
    cudaCheckError();

    cudaDeviceSynchronize();
	
    //init array
    for(int i = 0;i < num_elements;i++) {
        in_h[i] = num_elements - 1 - i;
    }
  // Copy host variables to device ------------------------------------------
    cudaMemcpy(in_d, in_h, num_elements * sizeof(unsigned int),
        cudaMemcpyHostToDevice);
    cudaCheckError();

    cudaDeviceSynchronize();
    // Launch kernel ----------------------------------------------------------
	startTime(&timer);
		
    radix_sort(in_d, out_d, out_scan_d, in_h, out_scan_h, num_elements);
    cudaDeviceSynchronize();
    cudaCheckError();
	
	stopTime(&timer); printf("GPU Sort time: %f s\n", elapsedTime(timer));
	cudaCheckError();
	
  
	// Copy device variables from host ----------------------------------------
	
    cudaMemcpy(out_h, out_d, num_elements * sizeof(unsigned int),
        cudaMemcpyDeviceToHost);
	cudaCheckError();
	
    cudaDeviceSynchronize();
	
   // Verify correctness -----------------------------------------------------
    qsort(in_h, num_elements, sizeof(unsigned int),compare);
    int flag = 0;
    for (int i = 0;i < num_elements;i++) {
        if (in_h[i] != out_h[i]) {
            flag = 1;
            break; 
        }
    }
    if (flag == 1) {
        printf("test failed\n");
    } else
        printf("test passed\n");
    // Free memory 
    cudaFree(in_d);
    cudaFree(out_scan_d);
    cudaFree(out_d);
    free(in_h); 
    free(out_h);

    return 0;
}


