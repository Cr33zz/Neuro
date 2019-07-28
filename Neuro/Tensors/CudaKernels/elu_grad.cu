extern "C" {
    __global__ void Run(int n, float* __restrict output, float* __restrict outputGradient, float* __restrict result, float alpha)
    {
		int i = blockIdx.x*blockDim.x + threadIdx.x;
		if (i < n)
            result[i] = (output[i] > 0 ? 1 : (output[i] + alpha)) * outputGradient[i];
	}
}