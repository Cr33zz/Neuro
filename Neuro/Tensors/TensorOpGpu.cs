using System;
using Cudafy;
using Cudafy.Atomics;
using Cudafy.Host;
using Cudafy.Translator;

namespace Neuro.Tensors
{
    internal class TensorOpGpu : TensorOpMultiCpu
    {
        internal TensorOpGpu()
        {
            //CudafyTranslator.GenerateDebug = true;
            Module = CudafyTranslator.Cudafy();
            Gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
            Gpu.LoadModule(Module);
        }

        public override void Add(Tensor t1, Tensor t2, Tensor result)
        {
            int threadsRequired = result.Length;
            double[] devT1 = Gpu.CopyToDevice(t1.Values);
            double[] devT2 = Gpu.CopyToDevice(t2.Values);
            double[] devResult = Gpu.Allocate(result.Values);

            Gpu.Launch(GetBlocksNum(threadsRequired), THREADS_PER_BLOCK).GpuAdd(devT1, devT2, devResult);
            Gpu.Synchronize();

            Gpu.CopyFromDevice(devResult, result.Values);
            Gpu.FreeAll();
        }

        public override void Sub(Tensor t1, Tensor t2, Tensor result)
        {
            int threadsRequired = result.Length;
            double[] devT1 = Gpu.CopyToDevice(t1.Values);
            double[] devT2 = Gpu.CopyToDevice(t2.Values);
            double[] devResult = Gpu.Allocate(result.Values);

            Gpu.Launch(GetBlocksNum(threadsRequired), THREADS_PER_BLOCK).GpuSub(devT1, devT2, devResult);
            Gpu.Synchronize();

            Gpu.CopyFromDevice(devResult, result.Values);
            Gpu.FreeAll();
        }

        public override void Mul(Tensor t1, Tensor t2, Tensor result)
        {
            int threadsRequired = result.Batches * t1.Depth * t1.Height * t2.Width;
            CudaShape[] shapes = new [] { new CudaShape(t1.Shape), new CudaShape(t2.Shape), new CudaShape(result.Shape) };

            double[] devT1 = Gpu.CopyToDevice(t1.Values);
            double[] devT2 = Gpu.CopyToDevice(t2.Values);
            double[] devResult = Gpu.Allocate(result.Values);
            CudaShape[] devShapes = Gpu.CopyToDevice(shapes);

            Gpu.Launch(GetBlocksNum(threadsRequired), THREADS_PER_BLOCK).GpuMul(devT1, devT2, devResult, devShapes);
            Gpu.Synchronize();

            Gpu.CopyFromDevice(devResult, result.Values);
            Gpu.FreeAll();
        }

        public override void Conv2D(Tensor t, Tensor kernels, int stride, int paddingX, int paddingY, Tensor result)
        {
            int threadsRequired = t.Batches * kernels.Batches * result.Width * result.Height;
            CudaShape[] shapes = new[] { new CudaShape(t.Shape), new CudaShape(kernels.Shape), new CudaShape(result.Shape) };

            double[] devT = Gpu.CopyToDevice(t.Values);
            double[] devKernels = Gpu.CopyToDevice(kernels.Values);
            double[] devResult = Gpu.Allocate(result.Values);
            CudaShape[] devShapes = Gpu.CopyToDevice(shapes);

            Gpu.Launch(GetBlocksNum(threadsRequired), THREADS_PER_BLOCK).GpuConv2D(devT, devKernels, devResult, devShapes, paddingX, paddingY, stride);
            Gpu.Synchronize();

            Gpu.CopyFromDevice(devResult, result.Values);
            Gpu.FreeAll();
        }

        public override void Conv2DInputGradient(Tensor gradient, Tensor rotKernels, int stride, int paddingX, int paddingY, Tensor inputGradients)
        {
            CudaShape[] shapes = new[] { new CudaShape(gradient.Shape),
                                         new CudaShape(rotKernels.Shape),
                                         new CudaShape(inputGradients.Shape),
                                         new CudaShape(rotKernels.Width, rotKernels.Height, 1, rotKernels.Batches) };

            double[] devGradient = Gpu.CopyToDevice(gradient.Values);
            double[] devRotKernels = Gpu.CopyToDevice(rotKernels.Values);
            CudaShape[] devShapes = Gpu.CopyToDevice(shapes);

            int threadsRequiredPerResultElem = rotKernels.Batches * rotKernels.Height * rotKernels.Width;
            double[,] resultPartials = new double[inputGradients.Length, GetBlocksNum(threadsRequiredPerResultElem)];
            double[,] devResultPartials = Gpu.Allocate(resultPartials);

            // simulate
            //GpuConv2DInputGradient(GetSimulatedThread(blockSize, new dim3(bx, by, bz), new dim3(tx, ty, tz)), gradient.Values, rotKernels.Values, resultPartials, shapes, paddingX, paddingY, stride);

            Gpu.Launch(new dim3(inputGradients.Length, GetBlocksNum(threadsRequiredPerResultElem)), THREADS_PER_BLOCK).GpuConv2DInputGradient(devGradient, devRotKernels, devResultPartials, devShapes, paddingX, paddingY, stride);
            Gpu.Synchronize();

            Gpu.CopyFromDevice(devResultPartials, resultPartials);

            Gpu.FreeAll();

            for (int k = 0; k < resultPartials.GetLength(0); ++k)
            for (int partialId = 0; partialId < resultPartials.GetLength(1); ++partialId)
                inputGradients.Values[k] += resultPartials[k, partialId];
        }

        public override void Conv2DKernelsGradient(Tensor input, Tensor gradient, int stride, int paddingX, int paddingY, Tensor kernelsGradient)
        {
            CudaShape[] shapes = new[] { new CudaShape(input.Shape),
                                         new CudaShape(kernelsGradient.Shape),
                                         new CudaShape(gradient.Shape),
                                         new CudaShape(kernelsGradient.Shape),
                                         new CudaShape(gradient.Width, gradient.Height, 1, gradient.Batches) };

            double[] devGradient = Gpu.CopyToDevice(gradient.Values);
            CudaShape[] devShapes = Gpu.CopyToDevice(shapes);

            int threadsRequiredPerResultElem = gradient.Batches * gradient.Height * gradient.Width;
            double[,] resultPartials = new double[kernelsGradient.Length, GetBlocksNum(threadsRequiredPerResultElem)];

            double[] devInput = Gpu.CopyToDevice(input.Values);
            double[,] devResultPartials = Gpu.Allocate(resultPartials);

            // simulate
            //GpuConv2DKernelsGradient(GetSimulatedThread(blockSize, new dim3(bx, by, bz), new dim3(tx, ty, tz)), input.Values, gradient.Values, kernelsGradientPartials, shapes, paddingX, paddingY, stride);

            Gpu.Launch(new dim3(kernelsGradient.Length, GetBlocksNum(threadsRequiredPerResultElem)), THREADS_PER_BLOCK).GpuConv2DKernelsGradient(devInput, devGradient, devResultPartials, devShapes, paddingX, paddingY, stride);
            Gpu.Synchronize();

            Gpu.CopyFromDevice(devResultPartials, resultPartials);

            Gpu.FreeAll();

            for (int k = 0; k < resultPartials.GetLength(0); ++k)
            for (int partialId = 0; partialId < resultPartials.GetLength(1); ++partialId)
                kernelsGradient.Values[k] += resultPartials[k, partialId];
        }

        // this is only for testing purposes
        private GThread GetSimulatedThread(dim3 blockDim, dim3 blockId, dim3 threadId)
        {
            return new GThread(threadId.x, threadId.y, new GBlock(new GGrid(new dim3(0)), blockDim, blockId.x, blockId.y));
        }

        private int GetBlocksNum(int threadsRequired)
        {
            return (int)Math.Ceiling(threadsRequired / (double)THREADS_PER_BLOCK);
        }

        private const int THREADS_PER_BLOCK = 256;

        private CudafyModule Module;
        private GPGPU Gpu;

        [Cudafy]
        private struct CudaShape
        {
            [CudafyIgnore]
            public CudaShape(Shape shape)
            : this(shape.Width, shape.Height, shape.Depth, shape.Batches)
            {
            }

            public CudaShape(int width, int height = 1, int depth = 1, int batches = 1)
            {
                Width = width;
                Height = height;
                Depth = depth;
                Batches = batches;
                Dim0 = width;
                Dim0Dim1 = Dim0 * height;
                Dim0Dim1Dim2 = Dim0Dim1 * depth;
            }

            public int GetIndex(int w, int h = 1, int d = 1, int n = 1)
            {
                return Dim0Dim1Dim2 * n + Dim0Dim1 * d + Dim0 * h + w;
            }

            public int TryGetIndex(int w, int h = 1, int d = 1, int n = 1)
            {
                if (h < 0 || h >= Height || w < 0 || w >= Width || d < 0 || d >= Depth)
                    return -1;

                return GetIndex(w, h, d, n);
            }

            public int GetWidth(int index)
            {
                return index % Width;
            }

            public int GetHeight(int index)
            {
                return (index / Dim0) % Height;
            }

            public int GetDepth(int index)
            {
                return (index / Dim0Dim1) % Depth;
            }

            public int GetBatch(int index)
            {
                return index / Dim0Dim1Dim2;
            }

            public int Width;
            public int Height;
            public int Depth;
            public int Batches;
            public int Dim0;
            public int Dim0Dim1;
            public int Dim0Dim1Dim2;
        }

        [Cudafy]
        private static void GpuAdd(GThread thread, double[] t1, double[] t2, double[] result)
        {
            int id = (thread.blockDim.x * thread.blockIdx.x) + thread.threadIdx.x;

            if (id >= result.Length)
                return;

            result[id] = t1[id] + t2[id % t2.Length];
        }

        [Cudafy]
        private static void GpuSub(GThread thread, double[] t1, double[] t2, double[] result)
        {
            int id = (thread.blockDim.x * thread.blockIdx.x) + thread.threadIdx.x;

            if (id >= result.Length)
                return;

            result[id] = t1[id] - t2[id % t2.Length];
        }

        [Cudafy]
        private static void GpuMul(GThread thread, double[] t1, double[] t2, double[] result, CudaShape[] shapes)
        {
            int id = (thread.blockDim.x * thread.blockIdx.x) + thread.threadIdx.x;

            if (id >= result.Length)
                return;

            int n = shapes[2].GetBatch(id);
            int d = shapes[2].GetDepth(id);
            int h = shapes[2].GetHeight(id);
            int w = shapes[2].GetWidth(id);

            for (int i = 0; i < shapes[0].Width; ++i)
                result[id] += t1[shapes[0].GetIndex(i, h, d, Math.Min(n, shapes[0].Batches - 1))] * t2[shapes[1].GetIndex(w, i, d, Math.Min(n, shapes[1].Batches - 1))];
        }

        [Cudafy]
        private static void GpuConv2D(GThread thread, double[] t, double[] kernels, double[] result, CudaShape[] shapes, int paddingX, int paddingY, int stride)
        {
            int id = (thread.blockDim.x * thread.blockIdx.x) + thread.threadIdx.x;

            if (id >= result.Length)
                return;

            int n = shapes[2].GetBatch(id);
            int outD = shapes[2].GetDepth(id);
            int outH = shapes[2].GetHeight(id);
            int outW = shapes[2].GetWidth(id);

            int h = -paddingY + stride * outH;
            int w = -paddingX + stride * outW;

            double val = 0;

            for (int kernelD = 0; kernelD < shapes[1].Depth; ++kernelD)
            for (int kernelH = 0; kernelH < shapes[1].Height; ++kernelH)
            for (int kernelW = 0; kernelW < shapes[1].Width; ++kernelW)
            {
                int tIndex = shapes[0].TryGetIndex(w + kernelW, h + kernelH, kernelD, n);

                if (tIndex >= 0)
                    val += t[tIndex] * kernels[shapes[1].GetIndex(kernelW, kernelH, kernelD, outD)];
            }

            result[shapes[2].GetIndex(outW, outH, outD, n)] = val;
        }

        [Cudafy]
        private static void GpuConv2DInputGradient(GThread thread, double[] gradient, double[] rotKernels, double[,] resultPartials, CudaShape[] shapes, int paddingX, int paddingY, int stride)
        {
            /*
            for (int n = 0; n < gradients.Batches; ++n)
            for (int outW = 0, w = -paddingX; outW < inputGradients.Width; w += stride, ++outW)
            for (int outH = 0, h = -paddingY; outH < inputGradients.Height; h += stride, ++outH)            
            for (int outD = 0; outD < inputGradients.Depth; ++outD)
            {
                for (int kernelN = 0; kernelN < rotKernels.Batches; ++kernelN)
                for (int kernelH = 0; kernelH < rotKernels.Height; ++kernelH)
                for (int kernelW = 0; kernelW < rotKernels.Width; ++kernelW)
                    inputGradients[outW, outH, outD, n] += gradients.TryGet(0, w + kernelW, h + kernelH, kernelN, n) * rotKernels[kernelW, kernelH, outD, kernelN];
            }
            */

            // this shared memory will store partial sums that later on will be reduced
            double[] sdata = thread.AllocateShared<double>("sdata", THREADS_PER_BLOCK);

            int resultElemId = thread.blockIdx.x;
            int tid = thread.threadIdx.x;
            int id = (thread.blockDim.x * thread.blockIdx.y) + thread.threadIdx.x;

            int threadsRequiredPerResultElem = shapes[1].Batches * shapes[1].Height * shapes[1].Width;

            int outN = shapes[2].GetBatch(resultElemId);
            int outD = shapes[2].GetDepth(resultElemId);
            int outH = shapes[2].GetHeight(resultElemId);
            int outW = shapes[2].GetWidth(resultElemId);

            int kernelN = shapes[3].GetBatch(id);
            int kernelH = shapes[3].GetHeight(id);
            int kernelW = shapes[3].GetWidth(id);

            int h = -paddingY + stride * outH;
            int w = -paddingX + stride * outW;

            double temp = 0;
            if (id < threadsRequiredPerResultElem)
            {
                int gradientIndex = shapes[0].TryGetIndex(w + kernelW, h + kernelH, kernelN, outN);
                if (gradientIndex >= 0)
                    temp = gradient[gradientIndex] * rotKernels[shapes[1].GetIndex(kernelW, kernelH, outD, kernelN)];
            }
            sdata[tid] = temp;

            thread.SyncThreads();

            int i = thread.blockDim.x / 2;
            while (i != 0)
            {
                if (tid < i)
                    sdata[tid] += sdata[tid + i];
                thread.SyncThreads();
                i /= 2;
            }

            if (tid == 0)
                resultPartials[thread.blockIdx.x, thread.blockIdx.y] = sdata[0];
        }

        [Cudafy]
        private static void GpuConv2DKernelsGradient(GThread thread, double[] input, double[] gradient, double[,] resultPartials, CudaShape[] shapes, int paddingX, int paddingY, int stride)
        {
            /*
            for (int kernelD = 0; kernelD < kernels.Depth; ++kernelD)
            for (int kernelH = 0; kernelH < kernels.Height; ++kernelH)
            for (int kernelW = 0; kernelW < kernels.Width; ++kernelW)
            for (int kernelN = 0; kernelN < kernels.Batches; ++kernelN)
            {
                for (int n = 0; n < gradient.Batches; ++n)
                for (int h = -paddingY, outH = 0; outH < gradient.Height; h += stride, ++outH)
                for (int w = -paddingX, outW = 0; outW < gradient.Width; w += stride, ++outW)
                {
                    double grad = gradient[outW, outH, kernelN, n];
                    double kernGradVal = input.TryGet(0, w + kernelW, h + kernelH, kernelD, n) * grad;
                    kernelsGradient[kernelW, kernelH, kernelD, kernelN] += kernGradVal;
                }
            }
            */

            // this shared memory will store partial sums that later on will be reduced
            double[] sdata = thread.AllocateShared<double>("sdata", THREADS_PER_BLOCK);

            int resultElemId = thread.blockIdx.x;
            int tid = thread.threadIdx.x;
            int id = (thread.blockDim.x * thread.blockIdx.y) + thread.threadIdx.x;

            int threadsRequiredPerResultElem = shapes[4].Batches * shapes[4].Height * shapes[4].Width;

            int kernelN = shapes[1].GetBatch(resultElemId);
            int kernelD = shapes[1].GetDepth(resultElemId);
            int kernelH = shapes[1].GetHeight(resultElemId);
            int kernelW = shapes[1].GetWidth(resultElemId);
            int n = shapes[4].GetBatch(id);
            int outH = shapes[4].GetHeight(id);
            int outW = shapes[4].GetWidth(id);

            int h = -paddingY + stride * outH;
            int w = -paddingX + stride * outW;

            double temp = 0;
            if (id < threadsRequiredPerResultElem)
            {
                int inputIndex = shapes[0].TryGetIndex(w + kernelW, h + kernelH, kernelD, n);
                if (inputIndex >= 0)
                    temp = input[inputIndex] * gradient[shapes[2].GetIndex(outW, outH, kernelN, n)];

                //if (resultElemId == 0)
                //    Console.WriteLine("tid=%d - %f", id, temp);
            }
            sdata[tid] = temp;

            thread.SyncThreads();

            int i = thread.blockDim.x / 2;
            while (i != 0)
            {
                if (tid < i)
                    sdata[tid] += sdata[tid + i];
                thread.SyncThreads();
                i /= 2;
            }

            if (tid == 0)
            {
                //if (resultElemId == 0)
                //    Console.WriteLine("gridDim.x=%d gridDim.y=%d blockDim.x=%d blockDim.y=%d", thread.gridDim.x, thread.gridDim.y, thread.blockDim.x, thread.blockDim.y);
                resultPartials[thread.blockIdx.x, thread.blockIdx.y] = sdata[0];
            }
        }
    }
}
