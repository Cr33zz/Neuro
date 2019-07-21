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

        //public override void Add(Tensor t1, Tensor t2, Tensor result)
        //{
        //    int threadsRequired = result.Length;
        //    float[] devT1 = Gpu.CopyToDevice(t1.Values);
        //    float[] devT2 = Gpu.CopyToDevice(t2.Values);
        //    float[] devResult = Gpu.Allocate(result.Values);

        //    Gpu.Launch(GetBlocksNum(threadsRequired), THREADS_PER_BLOCK).GpuAdd(devT1, devT2, devResult);
        //    Gpu.Synchronize();

        //    Gpu.CopyFromDevice(devResult, result.Values);
        //    Gpu.FreeAll();
        //}

        //public override void Sub(Tensor t1, Tensor t2, Tensor result)
        //{
        //    int threadsRequired = result.Length;
        //    float[] devT1 = Gpu.CopyToDevice(t1.Values);
        //    float[] devT2 = Gpu.CopyToDevice(t2.Values);
        //    float[] devResult = Gpu.Allocate(result.Values);

        //    Gpu.Launch(GetBlocksNum(threadsRequired), THREADS_PER_BLOCK).GpuSub(devT1, devT2, devResult);
        //    Gpu.Synchronize();

        //    Gpu.CopyFromDevice(devResult, result.Values);
        //    Gpu.FreeAll();
        //}

        //public override void Mul(Tensor t1, Tensor t2, Tensor result)
        //{
        //    int threadsRequired = result.BatchSize * t1.Depth * t1.Height * t2.Width;
        //    GpuShape[] shapes = new [] { new GpuShape(t1.Shape), new GpuShape(t2.Shape), new GpuShape(result.Shape) };

        //    float[] devT1 = Gpu.CopyToDevice(t1.Values);
        //    float[] devT2 = Gpu.CopyToDevice(t2.Values);
        //    float[] devResult = Gpu.Allocate(result.Values);
        //    GpuShape[] devShapes = Gpu.CopyToDevice(shapes);

        //    Gpu.Launch(GetBlocksNum(threadsRequired), THREADS_PER_BLOCK).GpuMul(devT1, devT2, devResult, devShapes);
        //    Gpu.Synchronize();

        //    Gpu.CopyFromDevice(devResult, result.Values);
        //    Gpu.FreeAll();
        //}

        private static CudaContext _CudaContext;
        private static CudaStream _CudaStream;
        private static CudaBlas _CudaBlasHandle;
        private static CudaDNNContext _CudnnContext;

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
    }
}
