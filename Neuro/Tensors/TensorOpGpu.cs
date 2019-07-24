using System;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaBlas;
using ManagedCuda.CudaDNN;

namespace Neuro.Tensors
{
    internal class TensorOpGpu : TensorOpMultiCpu
    {
        static TensorOpGpu()
        {
            _CudaContext = new CudaContext(0, true);
            _CudaBlasHandle = new CudaBlas();

            var props = _CudaContext.GetDeviceInfo();
            //this.DefaultBlockCount = props.MultiProcessorCount * 32;
            //this.DefaultThreadsPerBlock = props.MaxThreadsPerBlock;
            //this.WarpSize = props.WarpSize;

            _CudaStream = new CudaStream();
            _CudnnContext = new CudaDNNContext();            
        }

        public override void Add(float alpha, Tensor t1, float beta, Tensor t2, Tensor result)
        {
            t1.CopyToDevice();
            t2.CopyToDevice();
            result.CopyToDevice();

            if (t2.BatchSize == t1.BatchSize)
            {
                _CudaBlasHandle.Geam(Operation.NonTranspose, Operation.NonTranspose, 
                                     t1.Length, 1, 
                                     alpha, 
                                     t1.GpuData.DeviceVar, t1.Length, 
                                     t2.GpuData.DeviceVar, t2.Length, 
                                     beta, 
                                     result.GpuData.DeviceVar, result.Length);
                return;
            }

            for (int n = 0; n < t1.BatchSize; ++n)
            {
                _CudaBlasHandle.Geam(Operation.NonTranspose, Operation.NonTranspose, 
                                     t1.BatchLength, 1, 
                                     alpha,
                                     new CudaDeviceVariable<float>(t1.GpuData.DeviceVar.DevicePointer + n * t1.BatchLength * sizeof(float)), t1.BatchLength,
                                     t2.GpuData.DeviceVar, t2.BatchLength, 
                                     beta,
                                     new CudaDeviceVariable<float>(result.GpuData.DeviceVar.DevicePointer + n * result.BatchLength * sizeof(float)), result.BatchLength);
            }
        }

        public override void Mul(bool transposeT1, bool transposeT2, Tensor t1, Tensor t2, Tensor result)
        {
            t1.CopyToDevice();
            t2.CopyToDevice();
            result.CopyToDevice();

            var m = t1.Height;
            var n = t2.Width;
            var k = t1.Width;

            //treat depth as batch
            int batches = t1.Depth * t1.BatchSize;

            for (int b = 0; b < batches; ++b)
            {
                _CudaBlasHandle.Gemm(transposeT2 ? Operation.Transpose : Operation.NonTranspose,
                                     transposeT1 ? Operation.Transpose : Operation.NonTranspose, 
                                     n, m, k,  // trick to convert row major to column major
                                     1.0f,
                                     new CudaDeviceVariable<float>(t2.GpuData.DeviceVar.DevicePointer + b * t2.Shape.Dim0Dim1 * sizeof(float)), n,
                                     new CudaDeviceVariable<float>(t1.GpuData.DeviceVar.DevicePointer + b * t1.Shape.Dim0Dim1 * sizeof(float)), k,
                                     0.0f,
                                     new CudaDeviceVariable<float>(result.GpuData.DeviceVar.DevicePointer + b * result.Shape.Dim0Dim1 * sizeof(float)), n);
            }

            //CUdeviceptr[] aArray = new CUdeviceptr[batches];
            //CUdeviceptr[] bArray = new CUdeviceptr[batches];
            //CUdeviceptr[] cArray = new CUdeviceptr[batches];

            //for (int b = 0; b < batches; ++b)
            //{
            //    aArray[b] = t1.GpuData.DeviceVar.DevicePointer + b * t1.Shape.Dim0Dim1 * sizeof(float);
            //    bArray[b] = t2.GpuData.DeviceVar.DevicePointer + b * t2.Shape.Dim0Dim1 * sizeof(float);
            //    cArray[b] = result.GpuData.DeviceVar.DevicePointer + b * result.Shape.Dim0Dim1 * sizeof(float);
            //}

            //var dev_aArray = new CudaDeviceVariable<CUdeviceptr>(batches * 4);
            //dev_aArray.CopyToDevice(aArray);
            //var dev_bArray = new CudaDeviceVariable<CUdeviceptr>(batches * 4);
            //dev_bArray.CopyToDevice(bArray);
            //var dev_cArray = new CudaDeviceVariable<CUdeviceptr>(batches * 4);
            //dev_cArray.CopyToDevice(cArray);

            //_CudaBlasHandle.GemmBatched(transposeT2 ? Operation.Transpose : Operation.NonTranspose, 
            //                            transposeT1 ? Operation.Transpose : Operation.NonTranspose, 
            //                            n, m, k, 
            //                            1.0f, 
            //                            dev_bArray, n, 
            //                            dev_aArray, k, 
            //                            0.0f, 
            //                            dev_cArray, n, 
            //                            batches);

            //dev_aArray.Dispose();
            //dev_bArray.Dispose();
            //dev_cArray.Dispose();
        }

        public override void Transpose(Tensor t, Tensor result)
        {
            t.CopyToDevice();
            result.CopyToDevice();

            var m = t.Height;
            var n = t.Width;

            //treat depth as batch
            int batches = t.Depth * t.BatchSize;

            for (int b = 0; b < batches; ++b)
            {
                var tPtr = new CudaDeviceVariable<float>(t.GpuData.DeviceVar.DevicePointer + b * t.Shape.Dim0Dim1 * sizeof(float));

                _CudaBlasHandle.Geam(Operation.Transpose, 
                                     Operation.NonTranspose, m, n,  // trick to convert row major to column major
                                     1.0f,
                                     tPtr, n,
                                     tPtr, m, 
                                     0.0f,
                                     new CudaDeviceVariable<float>(result.GpuData.DeviceVar.DevicePointer + b * result.Shape.Dim0Dim1 * sizeof(float)), m);
            }
        }

        public override void Conv2D(Tensor t, Tensor kernels, int stride, Tensor.PaddingType padding, Tensor result)
        {
            int outputWidth = 0, outputHeight = 0, paddingX = 0, paddingY = 0;
            Tensor.GetPaddingParams(padding, t.Width, t.Height, kernels.Width, kernels.Height, stride, out outputHeight, out outputWidth, out paddingX, out paddingY);

            t.CopyToDevice();
            kernels.CopyToDevice();
            result.CopyToDevice();

            using (var convolutionDesc = new ConvolutionDescriptor())
            using (var tDesc = new TensorDescriptor())
            using (var kernelsDesc = new FilterDescriptor())
            using (var resultDesc = new TensorDescriptor())
            {
                convolutionDesc.SetConvolution2dDescriptor(paddingY, paddingX, stride, stride, 1, 1, cudnnConvolutionMode.CrossCorrelation, cudnnDataType.Float);
                tDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, t.Shape.Dimensions[3], t.Shape.Dimensions[2], t.Shape.Dimensions[1], t.Shape.Dimensions[0]);
                kernelsDesc.SetFilter4dDescriptor(cudnnDataType.Float, cudnnTensorFormat.NCHW, kernels.Shape.Dimensions[3], kernels.Shape.Dimensions[2], kernels.Shape.Dimensions[1], kernels.Shape.Dimensions[0]);
                resultDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, result.Shape.Dimensions[3], result.Shape.Dimensions[2], result.Shape.Dimensions[1], result.Shape.Dimensions[0]);

                var algo = _CudnnContext.GetConvolutionForwardAlgorithm(tDesc, kernelsDesc, convolutionDesc, resultDesc, cudnnConvolutionFwdPreference.PreferFastest, IntPtr.Zero);

                var workspaceSize = _CudnnContext.GetConvolutionForwardWorkspaceSize(tDesc, kernelsDesc, convolutionDesc, resultDesc, algo);
                workspaceSize = workspaceSize == 0 ? new SizeT(1) : workspaceSize;

                if (result.GpuData.ConvWorkspace == null || result.GpuData.ConvWorkspace.Size != workspaceSize)
                    result.GpuData.ConvWorkspace = new CudaDeviceVariable<byte>(workspaceSize);

                _CudnnContext.ConvolutionForward(1.0f, tDesc, t.GpuData.DeviceVar, kernelsDesc, kernels.GpuData.DeviceVar, convolutionDesc, algo, result.GpuData.ConvWorkspace, 0.0f, resultDesc, result.GpuData.DeviceVar);
            }            
        }

        public override void Conv2DInputGradient(Tensor gradient, Tensor kernels, int stride, Tensor.PaddingType padding, Tensor inputGradients)
        {
            int outputWidth = 0, outputHeight = 0, paddingX = 0, paddingY = 0;
            Tensor.GetPaddingParams(padding, gradient.Width, gradient.Height, kernels.Width, kernels.Height, stride, out outputHeight, out outputWidth, out paddingX, out paddingY);

            gradient.CopyToDevice();
            kernels.CopyToDevice();
            inputGradients.CopyToDevice();

            using (var convolutionDesc = new ConvolutionDescriptor())
            using (var gradientDesc = new TensorDescriptor())
            using (var kernelsDesc = new FilterDescriptor())
            using (var inputGradientsDesc = new TensorDescriptor())
            {
                convolutionDesc.SetConvolution2dDescriptor(paddingY, paddingX, stride, stride, 1, 1, cudnnConvolutionMode.CrossCorrelation, cudnnDataType.Float);
                gradientDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, gradient.Shape.Dimensions[3], gradient.Shape.Dimensions[2], gradient.Shape.Dimensions[1], gradient.Shape.Dimensions[0]);
                kernelsDesc.SetFilter4dDescriptor(cudnnDataType.Float, cudnnTensorFormat.NCHW, kernels.Shape.Dimensions[3], kernels.Shape.Dimensions[2], kernels.Shape.Dimensions[1], kernels.Shape.Dimensions[0]);
                inputGradientsDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, inputGradients.Shape.Dimensions[3], inputGradients.Shape.Dimensions[2], inputGradients.Shape.Dimensions[1], inputGradients.Shape.Dimensions[0]);

                var algo = _CudnnContext.GetConvolutionBackwardDataAlgorithm(kernelsDesc, gradientDesc, convolutionDesc, inputGradientsDesc, cudnnConvolutionBwdDataPreference.PreferFastest, IntPtr.Zero);
                var workspaceSize = _CudnnContext.GetConvolutionBackwardDataWorkspaceSize(kernelsDesc, gradientDesc, convolutionDesc, inputGradientsDesc, algo);
                workspaceSize = workspaceSize == 0 ? new SizeT(1) : workspaceSize;

                if (inputGradients.GpuData.ConvBackWorkspace == null || inputGradients.GpuData.ConvBackWorkspace.Size != workspaceSize)
                    inputGradients.GpuData.ConvBackWorkspace = new CudaDeviceVariable<byte>(workspaceSize);

                _CudnnContext.ConvolutionBackwardData(1.0f, kernelsDesc, kernels.GpuData.DeviceVar, gradientDesc, gradient.GpuData.DeviceVar, convolutionDesc, algo, inputGradients.GpuData.ConvBackWorkspace, 0.0f, inputGradientsDesc, inputGradients.GpuData.DeviceVar);
            }
        }

        public override void Conv2DKernelsGradient(Tensor input, Tensor gradient, int stride, Tensor.PaddingType padding, Tensor kernelsGradient)
        {
            int outputWidth = 0, outputHeight = 0, paddingX = 0, paddingY = 0;
            Tensor.GetPaddingParams(padding, input.Width, input.Height, kernelsGradient.Width, kernelsGradient.Height, stride, out outputHeight, out outputWidth, out paddingX, out paddingY);

            gradient.CopyToDevice();
            input.CopyToDevice();
            kernelsGradient.CopyToDevice();

            using (var convolutionDesc = new ConvolutionDescriptor())
            using (var gradientDesc = new TensorDescriptor())
            using (var inputDesc = new TensorDescriptor())
            using (var kernelsGradientsDesc = new FilterDescriptor())
            {
                convolutionDesc.SetConvolution2dDescriptor(paddingY, paddingX, stride, stride, 1, 1, cudnnConvolutionMode.CrossCorrelation, cudnnDataType.Float);
                gradientDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, gradient.Shape.Dimensions[3], gradient.Shape.Dimensions[2], gradient.Shape.Dimensions[1], gradient.Shape.Dimensions[0]);
                inputDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, input.Shape.Dimensions[3], input.Shape.Dimensions[2], input.Shape.Dimensions[1], input.Shape.Dimensions[0]);
                kernelsGradientsDesc.SetFilter4dDescriptor(cudnnDataType.Float, cudnnTensorFormat.NCHW, kernelsGradient.Shape.Dimensions[3], kernelsGradient.Shape.Dimensions[2], kernelsGradient.Shape.Dimensions[1], kernelsGradient.Shape.Dimensions[0]);

                var algo = _CudnnContext.GetConvolutionBackwardFilterAlgorithm(inputDesc, gradientDesc, convolutionDesc, kernelsGradientsDesc, cudnnConvolutionBwdFilterPreference.PreferFastest, IntPtr.Zero);
                var workspaceSize = _CudnnContext.GetConvolutionBackwardFilterWorkspaceSize(inputDesc, gradientDesc, convolutionDesc, kernelsGradientsDesc, algo);
                workspaceSize = workspaceSize == 0 ? new SizeT(1) : workspaceSize;

                if (kernelsGradient.GpuData.ConvBackKernelWorkspace == null || kernelsGradient.GpuData.ConvBackKernelWorkspace.Size != workspaceSize)
                    kernelsGradient.GpuData.ConvBackKernelWorkspace = new CudaDeviceVariable<byte>(workspaceSize);

                _CudnnContext.ConvolutionBackwardFilter(1.0f, inputDesc, input.GpuData.DeviceVar, gradientDesc, gradient.GpuData.DeviceVar, convolutionDesc, algo, kernelsGradient.GpuData.ConvBackKernelWorkspace, 0.0f, kernelsGradientsDesc, kernelsGradient.GpuData.DeviceVar);
            }
        }

        private cudnnPoolingMode TensorPoolTypeToCuDNNPoolType(Tensor.PoolType type)
        {
            if (type == Tensor.PoolType.Max)
                return cudnnPoolingMode.Max;
            return cudnnPoolingMode.AverageCountIncludePadding;
        }

        public override void Pool(Tensor t, int filterSize, int stride, Tensor.PoolType type, int paddingX, int paddingY, Tensor result)
        {
            t.CopyToDevice();
            result.CopyToDevice();

            using (var poolingDesc = new PoolingDescriptor())
            using (var tDesc = new TensorDescriptor())
            using (var resultDesc = new TensorDescriptor())
            {
                poolingDesc.SetPooling2dDescriptor(TensorPoolTypeToCuDNNPoolType(type), cudnnNanPropagation.NotPropagateNan, filterSize, filterSize, paddingX, paddingY, stride, stride);
                tDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, t.Shape.Dimensions[3], t.Shape.Dimensions[2], t.Shape.Dimensions[1], t.Shape.Dimensions[0]);
                resultDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, result.Shape.Dimensions[3], result.Shape.Dimensions[2], result.Shape.Dimensions[1], result.Shape.Dimensions[0]);

                _CudnnContext.PoolingForward(poolingDesc, 1.0f, tDesc, t.GpuData.DeviceVar, 0.0f, resultDesc, result.GpuData.DeviceVar);
            }
        }

        public override void PoolGradient(Tensor output, Tensor input, Tensor outputGradient, int filterSize, int stride, Tensor.PoolType type, int paddingX, int paddingY, Tensor result)
        {
            output.CopyToDevice();
            input.CopyToDevice();
            outputGradient.CopyToDevice();
            result.CopyToDevice();

            using (var poolingDesc = new PoolingDescriptor())
            using (var outputDesc = new TensorDescriptor())
            using (var inputDesc = new TensorDescriptor())
            using (var outputGradientDesc = new TensorDescriptor())
            using (var resultDesc = new TensorDescriptor())
            {
                poolingDesc.SetPooling2dDescriptor(TensorPoolTypeToCuDNNPoolType(type), cudnnNanPropagation.NotPropagateNan, filterSize, filterSize, paddingX, paddingY, stride, stride);
                outputDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, output.Shape.Dimensions[3], output.Shape.Dimensions[2], output.Shape.Dimensions[1], output.Shape.Dimensions[0]);
                inputDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, input.Shape.Dimensions[3], input.Shape.Dimensions[2], input.Shape.Dimensions[1], input.Shape.Dimensions[0]);
                outputGradientDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, outputGradient.Shape.Dimensions[3], outputGradient.Shape.Dimensions[2], outputGradient.Shape.Dimensions[1], outputGradient.Shape.Dimensions[0]);
                resultDesc.SetTensor4dDescriptor(cudnnTensorFormat.NCHW, cudnnDataType.Float, result.Shape.Dimensions[3], result.Shape.Dimensions[2], result.Shape.Dimensions[1], result.Shape.Dimensions[0]);

                _CudnnContext.PoolingBackward(poolingDesc, 1.0f, outputDesc, output.GpuData.DeviceVar, outputGradientDesc, outputGradient.GpuData.DeviceVar, inputDesc, input.GpuData.DeviceVar, 0.0f, resultDesc, result.GpuData.DeviceVar);
            }
        }

        private static CudaContext _CudaContext;
        private static CudaStream _CudaStream;
        private static CudaBlas _CudaBlasHandle;
        private static CudaDNNContext _CudnnContext;
    }
}
