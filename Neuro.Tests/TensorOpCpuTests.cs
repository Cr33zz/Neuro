using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neuro.Tensors;

namespace Neuro.Tests
{
    [TestClass]
    public class TensorOpCpuTests
    {
        [TestMethod]
        public void Mult_1Batch()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            Tensor t1 = new Tensor(new Shape(4,2,2));
            t1.FillWithRange(0);
            Tensor t2 = new Tensor(new Shape(2,4,2));
            t2.FillWithRange(0);

            Tensor r = t1.Mul(t2);
            Tensor correct = new Tensor(new double[]{28,34,76,98,428,466,604,658}, new Shape(2,2,2));

            Assert.IsTrue(r.Equals(correct));
        }

        [TestMethod]
        public void Mult_2Batches_1Batch()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            Tensor t1 = new Tensor(new Shape(4, 2, 2, 2));
            t1.FillWithRange(0);
            Tensor t2 = new Tensor(new Shape(2, 4, 2));
            t2.FillWithRange(0);

            Tensor r = t1.Mul(t2);
            Tensor correct = new Tensor(new double[] { 28, 34, 76, 98, 428, 466, 604, 658, 220, 290, 268, 354, 1132, 1234, 1308, 1426 }, new Shape(2, 2, 2, 2));

            Assert.IsTrue(r.Equals(correct));
        }

        [TestMethod]
        public void Mult_2Batches_2Batches()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            Tensor t1 = new Tensor(new Shape(4, 2, 2, 2));
            t1.FillWithRange(0);
            Tensor t2 = new Tensor(new Shape(2, 4, 2, 2));
            t2.FillWithRange(0);

            Tensor r = t1.Mul(t2);
            Tensor correct = new Tensor(new double[] { 28, 34, 76, 98, 428, 466, 604, 658, 1340, 1410, 1644, 1730, 2764, 2866, 3196, 3314 }, new Shape(2, 2, 2, 2));

            Assert.IsTrue(r.Equals(correct));
        }

        [TestMethod]
        public void Conv2D_Valid_1Kernel_1Batch()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            Tensor t1 = new Tensor(new Shape(6, 6, 2));
            t1.FillWithRange(0);
            Tensor t2 = new Tensor(new Shape(3, 3, 2));
            t2.FillWithRange(0);

            Tensor r = t1.Conv2D(t2, 1, Tensor.PaddingType.Valid);
            Tensor correct = new Tensor(new double[] { 5511, 5664, 5817, 5970, 6429, 6582, 6735, 6888, 7347, 7500, 7653, 7806, 8265, 8418, 8571, 8724 }, new Shape(4, 4, 1));

            Assert.IsTrue(r.Equals(correct));
        }

        [TestMethod]
        public void Conv2D_Valid_3Kernels_1Batch()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            Tensor t1 = new Tensor(new Shape(6, 6, 2));
            t1.FillWithRange(0);
            Tensor t2 = new Tensor(new Shape(3, 3, 2, 3));
            t2.FillWithRange(0);

            Tensor r = t1.Conv2D(t2, 1, Tensor.PaddingType.Valid);
            Tensor correct = new Tensor(new double[] { 5511, 5664, 5817, 5970, 6429, 6582, 6735, 6888, 7347, 7500, 7653, 7806, 8265, 8418, 8571, 8724, 13611, 14088, 14565, 15042, 16473, 16950, 17427, 17904, 19335, 19812, 20289, 20766, 22197, 22674, 23151, 23628, 21711, 22512, 23313, 24114, 26517, 27318, 28119, 28920, 31323, 32124, 32925, 33726, 36129, 36930, 37731, 38532 }, new Shape(4, 4, 3));

            Assert.IsTrue(r.Equals(correct));
        }

        [TestMethod]
        public void Conv2D_Valid_2Kernels_2Batches()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            Tensor t1 = new Tensor(new Shape(6, 6, 2, 2));
            t1.FillWithRange(0);
            Tensor t2 = new Tensor(new Shape(3, 3, 2, 2));
            t2.FillWithRange(0);

            Tensor r = t1.Conv2D(t2, 1, Tensor.PaddingType.Valid);
            Tensor correct = new Tensor(new double[] { 5511, 5664, 5817, 5970, 6429, 6582, 6735, 6888, 7347, 7500, 7653, 7806, 8265, 8418, 8571, 8724, 13611, 14088, 14565, 15042, 16473, 16950, 17427, 17904, 19335, 19812, 20289, 20766, 22197, 22674, 23151, 23628, 16527, 16680, 16833, 16986, 17445, 17598, 17751, 17904, 18363, 18516, 18669, 18822, 19281, 19434, 19587, 19740, 47955, 48432, 48909, 49386, 50817, 51294, 51771, 52248, 53679, 54156, 54633, 55110, 56541, 57018, 57495, 57972 }, new Shape(4, 4, 2, 2));

            Assert.IsTrue(r.Equals(correct));
        }

        [TestMethod]
        public void Conv2D_Same_1Kernel_1Batch()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            Tensor t1 = new Tensor(new Shape(6, 6, 2));
            t1.FillWithRange(0);
            Tensor t2 = new Tensor(new Shape(3, 3, 2));
            t2.FillWithRange(0);

            Tensor r = t1.Conv2D(t2, 1, Tensor.PaddingType.Same);
            Tensor correct = new Tensor(new double[] { 2492, 3674, 3794, 3914, 4034, 2624, 3765, 5511, 5664, 5817, 5970, 3855, 4413, 6429, 6582, 6735, 6888, 4431, 5061, 7347, 7500, 7653, 7806, 5007, 5709, 8265, 8418, 8571, 8724, 5583, 3416, 4898, 4982, 5066, 5150, 3260 }, new Shape(6, 6, 1));

            Assert.IsTrue(r.Equals(correct));
        }

        [TestMethod]
        public void Conv2D_Full_1Kernel_1Batch()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            Tensor t1 = new Tensor(new Shape(6, 6, 2));
            t1.FillWithRange(0);
            Tensor t2 = new Tensor(new Shape(3, 3, 2));
            t2.FillWithRange(0);

            Tensor r = t1.Conv2D(t2, 1, Tensor.PaddingType.Full);
            Tensor correct = new Tensor(new double[] { 612, 1213, 1801, 1870, 1939, 2008, 1315, 645, 1266, 2492, 3674, 3794, 3914, 4034, 2624, 1278, 1926, 3765, 5511, 5664, 5817, 5970, 3855, 1863, 2268, 4413, 6429, 6582, 6735, 6888, 4431, 2133, 2610, 5061, 7347, 7500, 7653, 7806, 5007, 2403, 2952, 5709, 8265, 8418, 8571, 8724, 5583, 2673, 1782, 3416, 4898, 4982, 5066, 5150, 3260, 1542, 786, 1489, 2107, 2140, 2173, 2206, 1375, 639 }, new Shape(8, 8, 1));

            Assert.IsTrue(r.Equals(correct));
        }

        [TestMethod]
        public void Pool_Max_Valid_1Batch_Stride2()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            Tensor t1 = new Tensor(new Shape(6, 6));
            t1.FillWithRange(0);
            
            Tensor r = t1.Pool(2, 2, Tensor.PoolType.Max, Tensor.PaddingType.Valid);
            Tensor correct = new Tensor(new double[] { 7, 9, 11, 19, 21, 23, 31, 33, 35 }, new Shape(3, 3, 1));

            Assert.IsTrue(r.Equals(correct));
        }

        [TestMethod]
        public void Pool_Max_Valid_2Batches_Stride2()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            Tensor t1 = new Tensor(new Shape(6, 6, 1, 2)); t1.FillWithRange(0);

            Tensor r = t1.Pool(2, 2, Tensor.PoolType.Max, Tensor.PaddingType.Valid);
            Tensor correct = new Tensor(new double[] { 7, 9, 11, 19, 21, 23, 31, 33, 35, 43, 45, 47, 55, 57, 59, 67, 69, 71 }, new Shape(3, 3, 1, 2));

            Assert.IsTrue(r.Equals(correct));
        }

        [TestMethod]
        public void Pool_Avg_Valid_2Batches_Stride2()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            Tensor t1 = new Tensor(new Shape(6, 6, 1, 2)); t1.FillWithRange(0);

            Tensor r = t1.Pool(2, 2, Tensor.PoolType.Avg, Tensor.PaddingType.Valid);
            Tensor correct = new Tensor(new double[] { 3.5, 5.5, 7.5, 15.5, 17.5, 19.5, 27.5, 29.5, 31.5, 39.5, 41.5, 43.5, 51.5, 53.5, 55.5, 63.5, 65.5, 67.5 }, new Shape(3, 3, 1, 2));

            Assert.IsTrue(r.Equals(correct));
        }

        [TestMethod]
        public void PoolGradient_Max_Valid_2Batches_Stride2()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            Tensor input = new Tensor(new Shape(6, 6, 1, 2)); input.FillWithRange(0);
            Tensor output = input.Pool(2, 2, Tensor.PoolType.Max, Tensor.PaddingType.Valid);
            Tensor gradient = new Tensor(output.Shape); gradient.FillWithRange(1);
            Tensor result = new Tensor(input.Shape);

            Tensor.PoolGradient(output, input, gradient, 2, 2, Tensor.PoolType.Max, Tensor.PaddingType.Valid, result);
            Tensor correct = new Tensor(new double[] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 5, 0, 6, 0, 0, 0, 0, 0, 0, 0, 7, 0, 8, 0, 9, 0, 0, 0, 0, 0, 0, 0, 10, 0, 11, 0, 12, 0, 0, 0, 0, 0, 0, 0, 13, 0, 14, 0, 15, 0, 0, 0, 0, 0, 0, 0, 16, 0, 17, 0, 18 }, new Shape(6, 6, 1, 2));

            Assert.IsTrue(result.Equals(correct));
        }

        [TestMethod]
        public void PoolGradient_Avg_Valid_2Batches_Stride2()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            Tensor input = new Tensor(new Shape(6, 6, 1, 2)); input.FillWithRange(0);
            Tensor output = input.Pool(2, 2, Tensor.PoolType.Avg, Tensor.PaddingType.Valid);
            Tensor gradient = new Tensor(output.Shape); gradient.FillWithRange(1);
            Tensor result = new Tensor(input.Shape);

            Tensor.PoolGradient(output, input, gradient, 2, 2, Tensor.PoolType.Avg, Tensor.PaddingType.Valid, result);
            Tensor correct = new Tensor(new double[] { 0, 0.0714285714285714, 0.181818181818182, 0.272727272727273, 0.4, 0.5, 0.428571428571429, 0.5, 0.727272727272727, 0.818181818181818, 1, 1.1, 0.774193548387097, 0.838709677419355, 1, 1.07142857142857, 1.23076923076923, 1.30769230769231, 1.16129032258065, 1.2258064516129, 1.42857142857143, 1.5, 1.69230769230769, 1.76923076923077, 1.52727272727273, 1.59090909090909, 1.76271186440678, 1.83050847457627, 2, 2.07142857142857, 1.90909090909091, 1.97272727272727, 2.16949152542373, 2.23728813559322, 2.42857142857143, 2.5, 2.27848101265823, 2.34177215189873, 2.51807228915663, 2.58433734939759, 2.75862068965517, 2.82758620689655, 2.65822784810127, 2.72151898734177, 2.91566265060241, 2.98192771084337, 3.17241379310345, 3.24137931034483, 3.02912621359223, 3.09223300970874, 3.27102803738318, 3.33644859813084, 3.51351351351351, 3.58108108108108, 3.40776699029126, 3.47087378640777, 3.66355140186916, 3.72897196261682, 3.91891891891892, 3.98648648648649, 3.77952755905512, 3.84251968503937, 4.02290076335878, 4.08778625954198, 4.26666666666667, 4.33333333333333, 4.15748031496063, 4.22047244094488, 4.41221374045802, 4.47709923664122, 4.66666666666667, 4.73333333333333 }, new Shape(6, 6, 1, 2));

            Assert.IsTrue(result.Equals(correct));
        }
    }
}

