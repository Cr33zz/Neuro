using System;
using System.Collections.Concurrent;
using System.Threading.Tasks;

namespace Neuro.Tensors
{
    internal class TensorOpMultiCpu : TensorOpCpu
    {
        public override void Add(Tensor t1, Tensor t2, Tensor result)
        {
            if (t2.BatchSize == t1.BatchSize)
            {
                var rangePartitioner = Partitioner.Create(0, t1.Values.Length);
                Parallel.ForEach(rangePartitioner, range =>
                {
                    for (int i = range.Item1; i < range.Item2; ++i)
                        result.Values[i] = t1.Values[i] + t2.Values[i];
                });
                return;
            }

            var rangePartitioner2 = Partitioner.Create(0, t1.BatchLength);

            for (int n = 0; n < t1.BatchSize; ++n)
            {
                Parallel.ForEach(rangePartitioner2, range =>
                {
                    for (int i = range.Item1, idx = n * t1.BatchLength + range.Item1; i < range.Item2; ++i, ++idx)
                        result.Values[idx] = t1.Values[idx] + t2.Values[i];
                });
            }
        }

        public override void Sub(Tensor t1, Tensor t2, Tensor result)
        {
            if (t2.BatchSize == t1.BatchSize)
            {
                var rangePartitioner = Partitioner.Create(0, t1.Values.Length);
                Parallel.ForEach(rangePartitioner, range =>
                {
                    for (int i = range.Item1; i < range.Item2; ++i)
                        result.Values[i] = t1.Values[i] - t2.Values[i];
                });
                return;
            }

            var rangePartitioner2 = Partitioner.Create(0, t1.BatchLength);

            for (int n = 0; n < t1.BatchSize; ++n)
            {
                Parallel.ForEach(rangePartitioner2, range =>
                {
                    for (int i = range.Item1, idx = n * t1.BatchLength + range.Item1; i < range.Item2; ++i, ++idx)
                        result.Values[idx] = t1.Values[idx] - t2.Values[i];
                });
            }
        }

        public override void Mul(Tensor t1, Tensor t2, Tensor result)
        {
            Parallel.For(0, result.BatchSize, n =>
            {
                Parallel.For(0, t1.Depth, d => {
                for (int h = 0; h < t1.Height; ++h)
                for (int w = 0; w < t2.Width; ++w)
                for (int i = 0; i < t1.Width; ++i)
                    result[w, h, d, n] += t1.Get(i, h, d, Math.Min(n, t1.BatchSize - 1)) *
                                          t2.Get(w, i, d, Math.Min(n, t2.BatchSize - 1));
                });
            });
        }

        public override void MulElem(Tensor t1, Tensor t2, Tensor result)
        {
            var rangePartitioner = Partitioner.Create(0, t1.Values.Length);
            Parallel.ForEach(rangePartitioner, range =>
            {
                for (int i = range.Item1; i < range.Item2; ++i)
                    result.Values[i] = t1.Values[i] * t2.Values[i];
            });
        }

        public override void Conv2D(Tensor t, Tensor kernels, int stride, int paddingX, int paddingY, Tensor result)
        {
            Parallel.For(0, t.BatchSize, n =>
            {
                Parallel.For(0, kernels.BatchSize, outD => {
                for (int h = -paddingY, outH = 0; outH < result.Height; h += stride, ++outH)
                for (int w = -paddingX, outW = 0; outW < result.Width; w += stride, ++outW)
                {
                    double val = 0;

                    for (int kernelD = 0; kernelD < kernels.Depth; ++kernelD)
                    for (int kernelH = 0; kernelH < kernels.Height; ++kernelH)
                    for (int kernelW = 0; kernelW < kernels.Width; ++kernelW)
                        val += t.TryGet(0, w + kernelW, h + kernelH, kernelD, n) *
                               kernels[kernelW, kernelH, kernelD, outD];

                    result[outW, outH, outD, n] = val;
                }});
            });
        }

        public override void Conv2DInputGradient(Tensor gradients, Tensor rotKernels, int stride, int paddingX, int paddingY, Tensor inputGradients)
        {
            Parallel.For(0, gradients.BatchSize, n =>
            {
                for (int outH = 0, h = -paddingY; outH < inputGradients.Height; h += stride, ++outH)
                for (int outW = 0, w = -paddingX; outW < inputGradients.Width; w += stride, ++outW)
                Parallel.For(0, inputGradients.Depth, outD =>
                {
                    for (int kernelN = 0; kernelN < rotKernels.BatchSize; ++kernelN)
                    for (int kernelH = 0; kernelH < rotKernels.Height; ++kernelH)
                    for (int kernelW = 0; kernelW < rotKernels.Width; ++kernelW)
                    {
                        inputGradients[outW, outH, outD, n] += gradients.TryGet(0, w + kernelW, h + kernelH, kernelN, n) * rotKernels[kernelW, kernelH, outD, kernelN];
                    }
                });
            });
        }

        public override void Conv2DKernelsGradient(Tensor input, Tensor gradient, int stride, int paddingX, int paddingY, Tensor kernelsGradient)
        {
            for (int n = 0; n < gradient.BatchSize; ++n)
            {
                Parallel.For(0, kernelsGradient.BatchSize, outD =>
                {
                    for (int h = -paddingY, outH = 0; outH < gradient.Height; h += stride, ++outH)
                    for (int w = -paddingX, outW = 0; outW < gradient.Width; w += stride, ++outW)
                    {
                        double grad = gradient[outW, outH, outD, n];

                        for (int kernelD = 0; kernelD < kernelsGradient.Depth; ++kernelD)
                        for (int kernelH = 0; kernelH < kernelsGradient.Height; ++kernelH)
                        for (int kernelW = 0; kernelW < kernelsGradient.Width; ++kernelW)
                        {
                            double kernGradVal = input.TryGet(0, w + kernelW, h + kernelH, kernelD, n) * grad;
                            kernelsGradient[kernelW, kernelH, kernelD, outD] += kernGradVal;
                        }
                    }
                });
            }
        }

        public override void Pool(Tensor t, int filterSize, int stride, Tensor.PoolType type, int paddingX, int paddingY, Tensor result)
        {
            Parallel.For(0, t.BatchSize, outN => 
            {
                Parallel.For(0, t.Depth, outD =>
                {
                    for (int outH = 0, h = -paddingY; outH < result.Height; h += stride, ++outH)
                    for (int outW = 0, w = -paddingX; outW < result.Width; w += stride, ++outW)
                    {
                        if (type == Tensor.PoolType.Max)
                        {
                            double value = double.MinValue;

                            for (int poolY = 0; poolY < filterSize; ++poolY)
                            for (int poolX = 0; poolX < filterSize; ++poolX)
                            {
                                value = Math.Max(value, t.TryGet(double.MinValue, w + poolX, h + poolY, outD, outN));
                            }

                            result[outW, outH, outD, outN] = value;
                        }
                        else if (type == Tensor.PoolType.Avg)
                        {
                            double sum = 0;
                            for (int poolY = 0; poolY < filterSize; ++poolY)
                            for (int poolX = 0; poolX < filterSize; ++poolX)
                                sum += t.TryGet(0, w + poolX, h + poolY, outD, outN);

                            result[outW, outH, outD, outN] = sum / (filterSize * filterSize);
                        }
                    }
                });
            });
        }

        public override void PoolGradient(Tensor output, Tensor input, Tensor outputGradient, int filterSize, int stride, Tensor.PoolType type, int paddingX, int paddingY, Tensor result)
        {
            Parallel.For(0, output.BatchSize, outN =>
            {
                Parallel.For(0, output.Depth, outD =>
                {
                    for (int outH = 0, h = -paddingY; outH < output.Height; ++outH, h += stride)
                    for (int outW = 0, w = -paddingX; outW < output.Width; ++outW, w += stride)
                    {
                        if (type == Tensor.PoolType.Max)
                        {
                            // use 1 for all elements equal to max value in each pooled matrix and 0 for all others
                            for (int poolH = 0; poolH < filterSize; ++poolH)
                            for (int poolW = 0; poolW < filterSize; ++poolW)
                            {
                                double value = input.TryGet(Double.MinValue, w + poolW, h + poolH, outD, outN);
                                result.TrySet(value == output[outW, outH, outD, outN] ? outputGradient.Get(outW, outH, outD, outN) : 0, w + poolW, h + poolH, outD, outN);
                            }
                        }
                        else if (type == Tensor.PoolType.Avg)
                        {
                            double filterElementsNum = filterSize * filterSize;

                            for (int poolH = 0; poolH < filterSize; ++poolH)
                            for (int poolW = 0; poolW < filterSize; ++poolW)
                            {
                                result.TrySet(outputGradient[outW, outH, outD, outN] / filterElementsNum, w + poolW, h + poolH, outD, outN);
                            }
                        }
                    }
                });
            });
        }
    }
}
