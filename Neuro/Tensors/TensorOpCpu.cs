﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuro.Tensors
{
    internal class TensorOpCpu
    {
        public virtual void Add(Tensor t1, Tensor t2, Tensor result)
        {
            if (t2.BatchSize == t1.BatchSize)
            {
                for (int i = 0; i < t1.Values.Length; ++i)
                    result.Values[i] = t1.Values[i] + t2.Values[i];
                return;
            }

            for (int n = 0; n < t1.BatchSize; ++n)
                for (int i = 0, idx = n * t1.BatchLength; i < t1.BatchLength; ++i, ++idx)
                    result.Values[idx] = t1.Values[idx] + t2.Values[i];
        }

        public virtual void Sub(Tensor t1, Tensor t2, Tensor result)
        {
            if (t2.BatchSize == t1.BatchSize)
            {
                for (int i = 0; i < t1.Values.Length; ++i)
                    result.Values[i] = t1.Values[i] - t2.Values[i];
                return;
            }

            for (int n = 0; n < t1.BatchSize; ++n)
            for (int i = 0, idx = n * t1.BatchLength; i < t1.BatchLength; ++i, ++idx)
                result.Values[idx] = t1.Values[idx] - t2.Values[i];
        }

        public virtual void Mul(Tensor t1, Tensor t2, Tensor result)
        {
            //for (int n = 0; n < result.BatchSize; ++n)
            //for (int d = 0; d < t1.Depth; ++d)
            //for (int h = 0; h < t1.Height; ++h)
            //for (int w = 0; w < t2.Width; ++w)
            //for (int i = 0; i < t1.Width; ++i)
            //    result[w, h, d, n] += t1[i, h, d, Math.Min(n, t1.BatchSize - 1)] * t2[w, i, d, Math.Min(n, t2.BatchSize - 1)];

            //const int BLOCK_SIZE = 32 / sizeof(double);
            int N = t1.Height;
            int M = t2.Width;
            int K = t1.Width;

            for (int n = 0; n < result.BatchSize; ++n)
            {
                int t1N = Math.Min(n, t1.BatchSize - 1);
                int t2N = Math.Min(n, t2.BatchSize - 1);

                for (int d = 0; d < t1.Depth; ++d)
                for (int i = 0; i < N; ++i)
                for (int j = 0; j < M; ++j)
                for (int k = 0; k < K; ++k)
                    result[j, i, d, n] += t1[k, i, d, t1N] * t2[j, k, d, t2N];
            }

            //for (int n = 0; n < result.BatchSize; ++n)
            //for (int d = 0; d < t1.Depth; ++d)
            //{
            //    int t1N = Math.Min(n, t1.BatchSize - 1);
            //    int t2N = Math.Min(n, t2.BatchSize - 1);

            //    for (int jj = 0; jj < M; jj += BLOCK_SIZE)
            //    {
            //        int jMax = Math.Min(jj + BLOCK_SIZE, M);

            //        for (int kk = 0; kk < K; kk += BLOCK_SIZE)
            //        {
            //            int kMax = Math.Min(kk + BLOCK_SIZE, K);

            //            for (int i = 0; i < N; ++i)
            //            for (int j = jj; j < jMax; ++j)
            //            for (int k = kk; k < kMax; ++k)
            //                result[j, i, d, n] += t1[k, i, d, t1N] * t2[j, k, d, t2N];
            //        }
            //    }
            //}
        }

        public virtual void MulElem(Tensor t1, Tensor t2, Tensor result)
        {
            for (int i = 0; i < t1.Values.Length; ++i)
                result.Values[i] = t1.Values[i] * t2.Values[i];
        }

        public virtual void Map(Tensor t, Func<double, double> func, Tensor result)
        {
            for (int i = 0; i < t.Values.Length; ++i)
                result.Values[i] = func(t.Values[i]);
        }

        public virtual void Conv2D(Tensor t, Tensor kernels, int stride, int paddingX, int paddingY, Tensor result)
        {
            for (int n = 0; n < t.BatchSize; ++n)
            {
                for (int outD = 0; outD < kernels.BatchSize; ++outD)
                for (int h = -paddingY, outH = 0; outH < result.Height; h += stride, ++outH)
                for (int w = -paddingX, outW = 0; outW < result.Width; w += stride, ++outW)
                {
                    double val = 0;

                    for (int kernelD = 0; kernelD < kernels.Depth; ++kernelD)
                    for (int kernelH = 0; kernelH < kernels.Height; ++kernelH)
                    for (int kernelW = 0; kernelW < kernels.Width; ++kernelW)
                        val += t.TryGet(0, w + kernelW, h + kernelH, kernelD, n) * kernels[kernelW, kernelH, kernelD, outD];

                    result[outW, outH, outD, n] = val;
                }
            }
        }

        public virtual void Conv2DInputGradient(Tensor gradients, Tensor rotKernels, int stride, int paddingX, int paddingY, Tensor inputGradients)
        {
            for (int n = 0; n < gradients.BatchSize; ++n)
            {
                for (int outH = 0, h = -paddingY; outH < inputGradients.Height; h += stride, ++outH)
                for (int outW = 0, w = -paddingX; outW < inputGradients.Width; w += stride, ++outW)
                for (int outD = 0; outD < inputGradients.Depth; ++outD)
                {
                    for (int kernelN = 0; kernelN < rotKernels.BatchSize; ++kernelN)
                    for (int kernelH = 0; kernelH < rotKernels.Height; ++kernelH)
                    for (int kernelW = 0; kernelW < rotKernels.Width; ++kernelW)
                    {
                        inputGradients[outW, outH, outD, n] += gradients.TryGet(0, w + kernelW, h + kernelH, kernelN, n) * rotKernels[kernelW, kernelH, outD, kernelN];
                    }
                }
            }
        }

        public virtual void Conv2DKernelsGradient(Tensor input, Tensor gradient, int stride, int paddingX, int paddingY, Tensor kernelsGradient)
        {
            for (int kernelD = 0; kernelD < kernelsGradient.Depth; ++kernelD)
            for (int kernelH = 0; kernelH < kernelsGradient.Height; ++kernelH)
            for (int kernelW = 0; kernelW < kernelsGradient.Width; ++kernelW)
            for (int kernelN = 0; kernelN < kernelsGradient.BatchSize; ++kernelN)
            {
                for (int outN = 0; outN < gradient.BatchSize; ++outN)
                for (int h = -paddingY, outH = 0; outH < gradient.Height; h += stride, ++outH)
                for (int w = -paddingX, outW = 0; outW < gradient.Width; w += stride, ++outW)
                {
                    double grad = gradient[outW, outH, kernelN, outN];
                    double kernGradVal = input.TryGet(0, w + kernelW, h + kernelH, kernelD, outN) * grad;
                    kernelsGradient[kernelW, kernelH, kernelD, kernelN] += kernGradVal;

                    //if (kernelsGradient.Shape.GetIndex(kernelW, kernelH, kernelD, kernelN) == 0)
                    //{
                    //    Trace.WriteLine($"cid={outN * output.Height * output.Width + outH * output.Width + outW} - {kernGradVal}");
                    //}
                }
            }
        }

        public virtual void Conv2DGradient_old(Tensor input, Tensor kernels, Tensor outputGradient, int stride, int paddingX, int paddingY, Tensor inputGradient, Tensor kernelsGradient)
        {
            for (var n = 0; n < input.BatchSize; n++)
            {
                for (var depth = 0; depth < outputGradient.Depth; depth++)
                {
                    var y = -paddingY;
                    for (var ay = 0; ay < outputGradient.Height; y += stride, ay++)
                    {
                        var x = -paddingX;
                        for (var ax = 0; ax < outputGradient.Width; x += stride, ax++)
                        {
                            // convolve centered at this particular location
                            var chainGradient = outputGradient.Get(ax, ay, depth, n);

                            // gradient from above, from chain rule
                            for (var fy = 0; fy < kernels.Height; fy++)
                            {
                                var oy = y + fy; // coordinates in the original input array coordinates
                                for (var fx = 0; fx < kernels.Width; fx++)
                                {
                                    var ox = x + fx;
                                    if (oy >= 0 && oy < input.Height && ox >= 0 && ox < input.Width)
                                    {
                                        for (var fd = 0; fd < kernels.Depth; fd++)
                                        {
                                            kernelsGradient.Set(kernelsGradient.Get(fx, fy, fd, depth) + input.Get(ox, oy, fd, n) * chainGradient, fx, fy, fd, depth);
                                            inputGradient.Set(inputGradient.Get(ox, oy, fd, n) + kernels.Get(fx, fy, fd, depth) * chainGradient, ox, oy, fd, n);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            //for (int n = 0; n < output.Batches; ++n)
            //{
            //    for (int outD = 0; outD < kernels.Batches; ++outD)
            //    for (int outH = 0, h = -paddingY; outH < output.Height; h += stride, ++outH)
            //    for (int outW = 0, w = -paddingX; outW < output.Width; w += stride, ++outW)
            //    {
            //        double grad = gradient[outW, outH, outD, n];

            //        for (int kernelD = 0; kernelD < kernels.Depth; ++kernelD)
            //        for (int kernelH = 0; kernelH < kernels.Height; ++kernelH)
            //        for (int kernelW = 0; kernelW < kernels.Width; ++kernelW)
            //        {
            //            double inputGradVal = kernels[kernelW, kernelH, kernelD, outD] * grad;
            //            inputGradient.TrySet(inputGradient.TryGet(0, w + kernelW, h + kernelH, kernelD, n) + inputGradVal, w + kernelW, h + kernelH, kernelD, n);

            //            double kernGradVal = input.TryGet(0, w + kernelW, h + kernelH, kernelD, n) * grad;
            //            kernelsGradient[kernelW, kernelH, kernelD, outD] += kernGradVal;
            //        }
            //    }
            //}
        }

        public virtual void Pool(Tensor t, int filterSize, int stride, Tensor.PoolType type, int paddingX, int paddingY, Tensor result)
        {
            for (int outN = 0; outN < t.BatchSize; ++outN)
            for (int outD = 0; outD < t.Depth; ++outD)
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
        }

        public virtual void PoolGradient(Tensor output, Tensor input, Tensor outputGradient, int filterSize, int stride, Tensor.PoolType type, int paddingX, int paddingY, Tensor result)
        {
            for (int outN = 0; outN < output.BatchSize; ++outN)
            for (int outD = 0; outD < output.Depth; ++outD)
            for (int outH = 0, h = -paddingY; outH < output.Height; ++outH, h += stride)
            for (int outW = 0, w = -paddingX; outW < output.Width; ++outW, w += stride)
            {
                if (type == Tensor.PoolType.Max)
                {
                    for (int poolH = 0; poolH < filterSize; ++poolH)
                    for (int poolW = 0; poolW < filterSize; ++poolW)
                    {
                        double value = input.TryGet(Double.MinValue, w + poolW, h + poolH, outD, outN);
                        result.TrySet(value == output[outW, outH, outD, outN] ? outputGradient[outW, outH, outD, outN] : 0, w + poolW, h + poolH, outD, outN);
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
        }
    }
}
