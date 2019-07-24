using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neuro.Tensors;

namespace Neuro.Tests
{
    [TestClass]
    public class TensorOpGpuTests
    {
        [TestMethod]
        public void Mul_CompareWithCpuResult()
        {
            Tensor t1 = new Tensor(new Shape(40, 30, 10, 32)); t1.FillWithRand(12);
            Tensor t2 = new Tensor(new Shape(35, 40, 10, 32)); t2.FillWithRand(1);

            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor r = t1.Mul(t2);

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor r2 = t1.Mul(t2);

            Assert.IsTrue(r.Equals(r2, 1e-4f));
        }

        [TestMethod]
        public void MulTranspose_CompareWithCpuResult()
        {
            Tensor t1 = new Tensor(new Shape(3, 2)); t1.FillWithRange(1);
            Tensor t2 = new Tensor(new Shape(3, 1)); t2.FillWithRange(1);

            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor r = t1.Mul(true, t2);

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor r2 = t1.Mul(true, t2);

            Assert.IsTrue(r.Equals(r2, 1e-4f));
        }

        [TestMethod]
        public void Transpose_CompareWithCpuResult()
        {
            Tensor t = new Tensor(new Shape(30, 30, 10, 32)); t.FillWithRand(12);

            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor r = t.Transposed();

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor r2 = t.Transposed();

            Assert.IsTrue(r.Equals(r2, 1e-4f));
        }

        [TestMethod]
        public void Add_1Batch_CompareWithCpuResult()
        {
            Tensor t1 = new Tensor(new Shape(82, 92, 30, 3)); t1.FillWithRand();
            Tensor t2 = new Tensor(new Shape(82, 92, 30, 1)); t2.FillWithRand();

            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor r = t1.Add(t2);

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor r2 = t1.Add(t2);

            Assert.IsTrue(r.Equals(r2));
        }

        [TestMethod]
        public void Add_SameBatches_CompareWithCpuResult()
        {
            Tensor t1 = new Tensor(new Shape(82, 92, 30, 3)); t1.FillWithRand();
            Tensor t2 = new Tensor(new Shape(82, 92, 30, 3)); t2.FillWithRand();

            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor r = t1.Add(t2);

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor r2 = t1.Add(t2);

            Assert.IsTrue(r.Equals(r2));
        }

        [TestMethod]
        public void Sub_1Batch_CompareWithCpuResult()
        {
            Tensor t1 = new Tensor(new Shape(82, 92, 30, 3)); t1.FillWithRand();
            Tensor t2 = new Tensor(new Shape(82, 92, 30, 1)); t2.FillWithRand();

            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor r = t1.Sub(t2);

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor r2 = t1.Sub(t2);

            Assert.IsTrue(r.Equals(r2));
        }

        [TestMethod]
        public void Sub_SameBatches_CompareWithCpuResult()
        {
            Tensor t1 = new Tensor(new Shape(82, 92, 30, 3)); t1.FillWithRand();
            Tensor t2 = new Tensor(new Shape(82, 92, 30, 3)); t2.FillWithRand();

            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor r = t1.Sub(t2);

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor r2 = t1.Sub(t2);

            Assert.IsTrue(r.Equals(r2));
        }

        [TestMethod]
        public void Conv2D_Valid_CompareWithCpuResult()
        {
            Tensor t = new Tensor(new Shape(26, 26, 32, 3)); t.FillWithRand();
            Tensor kernals = new Tensor(new Shape(3, 3, 32, 64)); kernals.FillWithRand();

            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor r = t.Conv2D(kernals, 1, Tensor.PaddingType.Valid);

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor r2 = t.Conv2D(kernals, 1, Tensor.PaddingType.Valid);

            Assert.IsTrue(r.Equals(r2, 1e-4f));
        }

        [TestMethod]
        public void Conv2DInputGradient_CompareWithCpuResult()
        {
            Tensor output = new Tensor(new Shape(24, 24, 64, 3)); output.FillWithRand();
            Tensor input = new Tensor(new Shape(26, 26, 32, 3)); input.FillWithRand();
            Tensor kernels = new Tensor(new Shape(3, 3, 32, 64)); kernels.FillWithRand();
            Tensor gradient = new Tensor(output); gradient.FillWithRand();

            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor inputGradient = new Tensor(input);
            Tensor.Conv2DInputsGradient(gradient, kernels, 1, Tensor.PaddingType.Valid, inputGradient);

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor inputGradient2 = new Tensor(input);
            Tensor.Conv2DInputsGradient(gradient, kernels, 1, Tensor.PaddingType.Valid, inputGradient2);

            Assert.IsTrue(inputGradient.Equals(inputGradient2, 1e-4f));
        }

        [TestMethod]
        public void Conv2DKernelsGradient_CompareWithCpuResult()
        {
            Tensor output = new Tensor(new Shape(24, 24, 64, 3)); output.FillWithRand();
            Tensor input = new Tensor(new Shape(26, 26, 32, 3)); input.FillWithRand();
            Tensor kernels = new Tensor(new Shape(3, 3, 32, 64)); kernels.FillWithRand();
            Tensor gradient = new Tensor(output); gradient.FillWithRand();

            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor kernelsGradient = new Tensor(kernels);
            Tensor.Conv2DKernelsGradient(input, gradient, 1, Tensor.PaddingType.Valid, kernelsGradient);

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor kernelsGradient2 = new Tensor(kernels);
            Tensor.Conv2DKernelsGradient(input, gradient, 1, Tensor.PaddingType.Valid, kernelsGradient2);

            Assert.IsTrue(kernelsGradient.Equals(kernelsGradient2, 1e-4f));
        }

        [TestMethod]
        public void Pool_Max_Valid_CompareWithCpuResult()
        {
            Tensor t = new Tensor(new Shape(27, 27, 20, 3)); t.FillWithRand();

            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor r = t.Pool(3, 2, Tensor.PoolType.Max, Tensor.PaddingType.Valid);

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor r2 = t.Pool(3, 2, Tensor.PoolType.Max, Tensor.PaddingType.Valid);

            Assert.IsTrue(r.Equals(r2));
        }

        [TestMethod]
        public void Pool_Avg_Valid_CompareWithCpuResult()
        {
            Tensor t = new Tensor(new Shape(27, 27, 20, 3)); t.FillWithRand();

            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor r = t.Pool(3, 2, Tensor.PoolType.Avg, Tensor.PaddingType.Valid);

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor r2 = t.Pool(3, 2, Tensor.PoolType.Avg, Tensor.PaddingType.Valid);

            Assert.IsTrue(r.Equals(r2));
        }

        [TestMethod]
        public void PoolGradient_Max_Valid_CompareWithCpuResult()
        {
            const int FILTER_SIZE = 3;
            const int STRIDE = 2;
            Tensor input = new Tensor(new Shape(27, 27, 20, 3)); input.FillWithRand();
            Tensor output = input.Pool(FILTER_SIZE, STRIDE, Tensor.PoolType.Max, Tensor.PaddingType.Valid);
            Tensor outputGradient = new Tensor(output.Shape); outputGradient.FillWithRand();

            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor r = new Tensor(input.Shape);
            Tensor.PoolGradient(output, input, outputGradient, FILTER_SIZE, STRIDE, Tensor.PoolType.Max, Tensor.PaddingType.Valid, r);

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor r2 = new Tensor(input.Shape);
            Tensor.PoolGradient(output, input, outputGradient, FILTER_SIZE, STRIDE, Tensor.PoolType.Max, Tensor.PaddingType.Valid, r2);

            Assert.IsTrue(r.Equals(r2));
        }

        [TestMethod]
        public void PoolGradient_Avg_Valid_CompareWithCpuResult()
        {
            const int FILTER_SIZE = 3;
            const int STRIDE = 2;
            Tensor input = new Tensor(new Shape(27, 27, 20, 3)); input.FillWithRand();
            Tensor output = input.Pool(FILTER_SIZE, STRIDE, Tensor.PoolType.Avg, Tensor.PaddingType.Valid);
            Tensor outputGradient = new Tensor(output.Shape); outputGradient.FillWithRand();

            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor r = new Tensor(input.Shape);
            Tensor.PoolGradient(output, input, outputGradient, FILTER_SIZE, STRIDE, Tensor.PoolType.Avg, Tensor.PaddingType.Valid, r);

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor r2 = new Tensor(input.Shape);
            Tensor.PoolGradient(output, input, outputGradient, FILTER_SIZE, STRIDE, Tensor.PoolType.Avg, Tensor.PaddingType.Valid, r2);

            Assert.IsTrue(r.Equals(r2));
        }
    }
}
