using System;
using System.Collections.Generic;
using ManagedCuda;
using ManagedCuda.NVRTC;
using ManagedCuda.BasicTypes;
using System.IO;
using System.Linq;
using ManagedCuda.VectorTypes;

namespace Neuro.Tensors
{
    internal partial class TensorOpGpu : TensorOpMultiCpu
    {
        private class KernelLoader
        {
            public KernelLoader()
            {
                CudaDevProps = TensorOpGpu._CudaContext.GetDeviceInfo();
                LoadKernelsRecursive(AppDomain.CurrentDomain.BaseDirectory);
            }

            private void AddKernel(string name, CudaKernel kernel)
            {
                Kernels[name] = kernel;
            }

            private void LoadKernelsRecursive(string dir)
            {
                foreach (string d in Directory.GetDirectories(dir))
                {
                    foreach (string f in Directory.GetFiles(d, "*.cu"))
                        LoadKernel(Path.GetFileNameWithoutExtension(f), f);
                    LoadKernelsRecursive(d);
                }
            }

            public void LoadKernel(string name, string path)
            {
                var result = LoadKernel(path, out var kernel);
                if (result == nvrtcResult.Success)
                    AddKernel(name, kernel);
            }

            private nvrtcResult LoadKernel(string kernelSourceFile, out CudaKernel kernel)
            {
                nvrtcResult result;
                kernel = null;

                using (var compiler = new CudaRuntimeCompiler(File.ReadAllText(kernelSourceFile), Path.GetFileName(kernelSourceFile)))
                {
                    try
                    {
                        compiler.Compile(new string[0]);
                        result = nvrtcResult.Success;
                    }
                    catch (NVRTCException ex)
                    {
                        result = ex.NVRTCError;
                    }

                    var outputFileWithoutExt = Path.Combine(Path.GetDirectoryName(kernelSourceFile), Path.GetFileNameWithoutExtension(kernelSourceFile));
                    File.WriteAllText(outputFileWithoutExt + ".ptx.log", compiler.GetLogAsString());

                    if (result == nvrtcResult.Success)
                    {
                        var ptx = compiler.GetPTX();
                        kernel = _CudaContext.LoadKernelFatBin(ptx, "Run");
                        File.WriteAllBytes(outputFileWithoutExt + ".ptx", ptx);
                    }
                }
                return result;
            }

            public void RunKernel(string kernelName, Tensor input, Tensor output, params object[] extraParameters)
            {
                if (Kernels.TryGetValue(kernelName, out var kernel))
                    RunKernel(kernel, input, output, extraParameters);
                else
                    throw new ArgumentException($"Kernel '{kernelName}' not found");
            }

            public void RunKernel(string kernelName, Tensor input1, Tensor input2, Tensor output, params object[] extraParameters)
            {
                if (Kernels.TryGetValue(kernelName, out var kernel))
                    RunKernel(kernel, input1, input2, output, extraParameters);
                else
                    throw new ArgumentException($"Kernel '{kernelName}' not found");
            }

            private void RunKernel(CudaKernel kernel, Tensor input1, Tensor input2, Tensor output, params object[] extraParameters)
            {
                input1.CopyToDevice();
                input2.CopyToDevice();
                output.CopyToDevice();

                var parameters = new object[] { input1.GpuData.DeviceVar.DevicePointer, input2.GpuData.DeviceVar.DevicePointer, output.GpuData.DeviceVar.DevicePointer };
                if (extraParameters != null)
                    parameters = parameters.Concat(extraParameters).ToArray();

                RunKernel(kernel, output.Length, parameters);
            }

            private void RunKernel(CudaKernel kernel, Tensor input, Tensor output, params object[] extraParameters)
            {
                input.CopyToDevice();
                output.CopyToDevice();

                var parameters = new object[] { input.GpuData.DeviceVar.DevicePointer, output.GpuData.DeviceVar.DevicePointer };
                if (extraParameters != null)
                    parameters = parameters.Concat(extraParameters).ToArray();

                RunKernel(kernel, output.Length, parameters);
            }

            private void RunKernel(CudaKernel kernel, int count, object[] parameters)
            {
                int threadsPerBlock = CudaDevProps.MaxThreadsPerBlock;
                int blockCount = GetBlocksNum(count);

                if (count <= CudaDevProps.MaxThreadsPerBlock)
                {
                    blockCount = 1;
                    threadsPerBlock = count;
                }                

                kernel.BlockDimensions = new dim3(threadsPerBlock, 1, 1);
                kernel.GridDimensions = new dim3(blockCount, 1, 1);

                var finalParams = parameters.ToList();
                finalParams.Insert(0, count);
                kernel.Run(finalParams.ToArray());
            }

            private int GetBlocksNum(int count)
            {
                return (int)Math.Ceiling(count / (float)CudaDevProps.MaxThreadsPerBlock);
            }

            private readonly Dictionary<string, CudaKernel> Kernels = new Dictionary<string, CudaKernel>();
            private CudaDeviceProperties CudaDevProps;
        }
    }
}
