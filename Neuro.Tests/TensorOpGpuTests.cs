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
            Tensor t1 = new Tensor(new Shape(82, 40, 30, 3)).Randomize();
            Tensor t2 = new Tensor(new Shape(40, 82, 30)).Randomize();

            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor r = t1.Mul(t2);

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor r2 = t1.Mul(t2);

            Assert.IsTrue(r.Equals(r2));
        }

        [TestMethod]
        public void Add_SingleBatch_CompareWithCpuResult()
        {
            Tensor t1 = new Tensor(new Shape(82, 921, 30, 3)).Randomize();
            Tensor t2 = new Tensor(new Shape(82, 921, 30, 1)).Randomize();

            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor r = t1.Add(t2);

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor r2 = t1.Add(t2);

            Assert.IsTrue(r.Equals(r2));
        }

        [TestMethod]
        public void Add_SameBatches_CompareWithCpuResult()
        {
            Tensor t1 = new Tensor(new Shape(82, 921, 30, 3)).Randomize();
            Tensor t2 = new Tensor(new Shape(82, 921, 30, 3)).Randomize();

            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor r = t1.Add(t2);

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor r2 = t1.Add(t2);

            Assert.IsTrue(r.Equals(r2));
        }

        [TestMethod]
        public void Sub_SingleBatch_CompareWithCpuResult()
        {
            Tensor t1 = new Tensor(new Shape(82, 921, 30, 3)).Randomize();
            Tensor t2 = new Tensor(new Shape(82, 921, 30, 1)).Randomize();

            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor r = t1.Sub(t2);

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor r2 = t1.Sub(t2);

            Assert.IsTrue(r.Equals(r2));
        }

        [TestMethod]
        public void Sub_SameBatches_CompareWithCpuResult()
        {
            Tensor t1 = new Tensor(new Shape(82, 921, 30, 3)).Randomize();
            Tensor t2 = new Tensor(new Shape(82, 921, 30, 3)).Randomize();

            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor r = t1.Sub(t2);

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor r2 = t1.Sub(t2);

            Assert.IsTrue(r.Equals(r2));
        }

        [TestMethod]
        public void Conv2D_CompareWithCpuResult()
        {
            Tensor t = new Tensor(new Shape(26, 26, 32, 3)).Randomize();
            Tensor kernals = new Tensor(new Shape(3, 3, 32, 64)).Randomize();

            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor r = t.Conv2D(kernals, 1, Tensor.PaddingType.Valid);

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor r2 = t.Conv2D(kernals, 1, Tensor.PaddingType.Valid);

            Assert.IsTrue(r.Equals(r2));
        }

        [TestMethod]
        public void Conv2DInputGradient_CompareWithCpuResult()
        {
            Tensor output = new Tensor(new Shape(24, 24, 64, 3)).Randomize();
            Tensor input = new Tensor(new Shape(26, 26, 32, 3)).Randomize();
            Tensor kernels = new Tensor(new Shape(3, 3, 32, 64)).Randomize();
            Tensor gradient = new Tensor(output).Randomize();

            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor inputGradient = new Tensor(input);
            Tensor.Conv2DInputsGradient(gradient, kernels, 1, inputGradient);

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor inputGradient2 = new Tensor(input);
            Tensor.Conv2DInputsGradient(gradient, kernels, 1, inputGradient2);

            Assert.IsTrue(inputGradient.Equals(inputGradient2));
        }

        [TestMethod]
        public void Conv2DKernelsGradient_CompareWithCpuResult()
        {
            Tensor output = new Tensor(new Shape(24, 24, 64, 3)).Randomize();
            Tensor input = new Tensor(new Shape(26, 26, 32, 3)).Randomize();
            Tensor kernels = new Tensor(new Shape(3, 3, 32, 64)).Randomize();
            Tensor gradient = new Tensor(output).Randomize();

            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor kernelsGradient = new Tensor(kernels);
            Tensor.Conv2DKernelsGradient(output, input, gradient, 1, Tensor.PaddingType.Valid, kernelsGradient);

            Tensor.SetOpMode(Tensor.OpMode.GPU);
            Tensor kernelsGradient2 = new Tensor(kernels);
            Tensor.Conv2DKernelsGradient(output, input, gradient, 1, Tensor.PaddingType.Valid, kernelsGradient2);

            Assert.IsTrue(kernelsGradient.Equals(kernelsGradient2));
        }
    }
}
