using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neuro.Tensors;

namespace Neuro.Tests
{
    [TestClass]
    public class TensorOpGpuTests
    {
        [TestMethod]
        public void Mult_CompareWithCpuResult()
        {
            Tensor t1 = new Tensor(new Shape(82, 40, 30, 3)); t1.FillWithRand();
            Tensor t2 = new Tensor(new Shape(40, 82, 30)); t2.FillWithRand();

            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor r = t1.Mul(t2);

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor r2 = t1.Mul(t2);

            Assert.IsTrue(r.Equals(r2, 1e-4f));
        }

        [TestMethod]
        public void Add_1Batch_CompareWithCpuResult()
        {
            Tensor t1 = new Tensor(new Shape(82, 921, 30, 3)); t1.FillWithRand();
            Tensor t2 = new Tensor(new Shape(82, 921, 30, 1)); t2.FillWithRand();

            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor r = t1.Add(t2);

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor r2 = t1.Add(t2);

            Assert.IsTrue(r.Equals(r2));
        }

        [TestMethod]
        public void Add_SameBatches_CompareWithCpuResult()
        {
            Tensor t1 = new Tensor(new Shape(82, 921, 30, 3)); t1.FillWithRand();
            Tensor t2 = new Tensor(new Shape(82, 921, 30, 3)); t2.FillWithRand();

            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor r = t1.Add(t2);

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor r2 = t1.Add(t2);

            Assert.IsTrue(r.Equals(r2));
        }

        [TestMethod]
        public void Sub_1Batch_CompareWithCpuResult()
        {
            Tensor t1 = new Tensor(new Shape(82, 921, 30, 3)); t1.FillWithRand();
            Tensor t2 = new Tensor(new Shape(82, 921, 30, 1)); t2.FillWithRand();

            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor r = t1.Sub(t2);

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor r2 = t1.Sub(t2);

            Assert.IsTrue(r.Equals(r2));
        }

        [TestMethod]
        public void Sub_SameBatches_CompareWithCpuResult()
        {
            Tensor t1 = new Tensor(new Shape(82, 921, 30, 3)); t1.FillWithRand();
            Tensor t2 = new Tensor(new Shape(82, 921, 30, 3)); t2.FillWithRand();

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
            Tensor.Conv2DInputsGradient(gradient, kernels, 1, inputGradient);

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor inputGradient2 = new Tensor(input);
            Tensor.Conv2DInputsGradient(gradient, kernels, 1, inputGradient2);

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
            Tensor input = new Tensor(new Shape(27, 27, 20, 3)); input.FillWithRand();
            Tensor output = input.Pool(3, 2, Tensor.PoolType.Max, Tensor.PaddingType.Valid);
            Tensor outputGradient = new Tensor(output.Shape); outputGradient.FillWithRand();

            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor r = new Tensor(input.Shape);
            Tensor.PoolGradient(output, input, outputGradient, 3, 2, Tensor.PoolType.Max, Tensor.PaddingType.Valid, r);

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor r2 = new Tensor(input.Shape);
            Tensor.PoolGradient(output, input, outputGradient, 3, 2, Tensor.PoolType.Max, Tensor.PaddingType.Valid, r2);

            Assert.IsTrue(r.Equals(r2));
        }

        [TestMethod]
        public void PoolGradient_Avg_Valid_CompareWithCpuResult()
        {
            Tensor input = new Tensor(new Shape(27, 27, 20, 3)); input.FillWithRand();
            Tensor output = input.Pool(3, 2, Tensor.PoolType.Avg, Tensor.PaddingType.Valid);
            Tensor outputGradient = new Tensor(output.Shape); outputGradient.FillWithRand();

            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor r = new Tensor(input.Shape);
            Tensor.PoolGradient(output, input, outputGradient, 3, 2, Tensor.PoolType.Avg, Tensor.PaddingType.Valid, r);

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor r2 = new Tensor(input.Shape);
            Tensor.PoolGradient(output, input, outputGradient, 3, 2, Tensor.PoolType.Avg, Tensor.PaddingType.Valid, r2);

            Assert.IsTrue(r.Equals(r2));
        }
    }
}
