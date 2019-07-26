extern "C" {
    __global__ void Run(int n, float* __restrict input, float* __restrict result, float alpha)
    {
		int i = blockIdx.x*blockDim.x + threadIdx.x;
		if (i < n)
            result[i] = input[i] > 0 ? input[i] : alpha * (exp(input[i]) - 1);
	}
}