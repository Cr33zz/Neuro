using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.ComTypes;
using Neuro.Tensors;

namespace Neuro.Tests
{
    [TestClass]
    public class TensorTests
    {
        [TestMethod]
        public void Add_SameBatchSize()
        {
            var t1 = new Tensor(new Shape(2, 3, 4, 5)); t1.FillWithRange(1);
            var t2 = new Tensor(new Shape(2, 3, 4, 5)); t2.FillWithRange(2, 2);
            var result = new Tensor(t1.Shape);

            t1.Add(t2, result);
            for (int i = 0; i < t1.Shape.Length; ++i)
                Assert.AreEqual(result.GetFlat(i), t1.GetFlat(i) + t2.GetFlat(i % t2.Shape.Length));
        }

        [TestMethod]
        public void Add_5Batches_1Batch()
        {
            var t1 = new Tensor(new Shape(2, 3, 4, 5)); t1.FillWithRange(1);
            var t2 = new Tensor(new Shape(2, 3, 4, 1)); t2.FillWithRange(2, 2);
            var result = new Tensor(t1.Shape);

            t1.Add(t2, result);
            for (int i = 0; i < t1.Shape.Length; ++i)
                Assert.AreEqual(result.GetFlat(i), t1.GetFlat(i) + t2.GetFlat(i % t2.Shape.Length), 1e-5);
        }

        [TestMethod]
        public void Add_Scalar()
        {
            var t = new Tensor(new Shape(2, 3, 4, 5)); t.FillWithRange(1);
            var result = new Tensor(t.Shape);

            t.Add(2, result);
            for (int i = 0; i < t.Shape.Length; ++i)
                Assert.AreEqual(result.GetFlat(i), t.GetFlat(i) + 2, 1e-4);
        }

        [TestMethod]
        public void Sub_SameBatchSize()
        {
            var t1 = new Tensor(new Shape(2, 3, 4, 5)); t1.FillWithRange(1);
            var t2 = new Tensor(new Shape(2, 3, 4, 5)); t2.FillWithRange(2, 2);
            var result = new Tensor(t1.Shape);

            t1.Sub(t2, result);
            for (int i = 0; i < t1.Shape.Length; ++i)
                Assert.AreEqual(result.GetFlat(i), t1.GetFlat(i) - t2.GetFlat(i % t2.Shape.Length), 1e-5);
        }

        [TestMethod]
        public void Sub_5Batches_1Batch()
        {
            var t1 = new Tensor(new Shape(2, 3, 4, 5)); t1.FillWithRange(1);
            var t2 = new Tensor(new Shape(2, 3, 4, 1)); t2.FillWithRange(2, 2);
            var result = new Tensor(t1.Shape);

            t1.Sub(t2, result);
            for (int i = 0; i < t1.Shape.Length; ++i)
                Assert.AreEqual(result.GetFlat(i), t1.GetFlat(i) - t2.GetFlat(i % t2.Shape.Length), 1e-5);
        }

        [TestMethod]
        public void Sub_Scalar()
        {
            var t = new Tensor(new Shape(2, 3, 4, 5)); t.FillWithRange(1);
            var result = new Tensor(t.Shape);

            t.Sub(2, result);
            for (int i = 0; i < t.Shape.Length; ++i)
                Assert.AreEqual(result.GetFlat(i), t.GetFlat(i) - 2, 1e-4);
        }

        [TestMethod]
        public void Div()
        {
            var t1 = new Tensor(new Shape(2, 3, 4, 5)); t1.FillWithRange(2, 2);
            var t2 = new Tensor(new Shape(2, 3, 4, 5)); t2.FillWithRange(1);
            var result = new Tensor(t1.Shape);

            t1.Div(t2, result);
            for (int i = 0; i < t1.Shape.Length; ++i)
                Assert.AreEqual(result.GetFlat(i), 2, 1e-4);
        }

        [TestMethod]
        public void Div_Scalar()
        {
            var t = new Tensor(new Shape(2, 3, 4, 5)); t.FillWithRange(2, 2);
            var result = new Tensor(t.Shape);

            t.Div(2, result);
            for (int i = 0; i < t.Shape.Length; ++i)
                Assert.AreEqual(result.GetFlat(i), t.GetFlat(i) / 2, 1e-4);
        }

        [TestMethod]
        public void Mul_Scalar()
        {
            var t = new Tensor(new Shape(2, 3, 4, 5)); t.FillWithRange(2, 2);
            var result = new Tensor(t.Shape);

            t.Mul(2, result);
            for (int i = 0; i < t.Shape.Length; ++i)
                Assert.AreEqual(result.GetFlat(i), t.GetFlat(i) * 2, 1e-4);
        }

        [TestMethod]
        public void MulElem()
        {
            var t1 = new Tensor(new Shape(2, 3, 4, 5)); t1.FillWithRand();
            var t2 = new Tensor(new Shape(2, 3, 4, 5)); t2.FillWithRand();
            var result = new Tensor(t1.Shape);

            t1.MulElem(t2, result);
            for (int i = 0; i < t1.Shape.Length; ++i)
                Assert.AreEqual(result.GetFlat(i), t1.GetFlat(i) * t2.GetFlat(i), 1e-5);
        }

        [TestMethod]
        public void Mul_1Batch()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            Tensor t1 = new Tensor(new Shape(4, 2, 2));
            t1.FillWithRange(0);
            Tensor t2 = new Tensor(new Shape(2, 4, 2));
            t2.FillWithRange(0);

            Tensor r = t1.Mul(t2);
            Tensor correct = new Tensor(new float[] { 28, 34, 76, 98, 428, 466, 604, 658 }, new Shape(2, 2, 2));

            Assert.IsTrue(r.Equals(correct));
        }

        [TestMethod]
        public void Mul_1Batch_2D()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            Tensor t1 = new Tensor(new Shape(4, 2));
            t1.FillWithRange(0);
            Tensor t2 = new Tensor(new Shape(2, 4));
            t2.FillWithRange(0);

            Tensor r = t1.Mul(t2);
            Tensor correct = new Tensor(new float[] { 28, 34, 76, 98 }, new Shape(2, 2));

            Assert.IsTrue(r.Equals(correct));
        }

        [TestMethod]
        public void Mul_2Batches_1Batch()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            Tensor t1 = new Tensor(new Shape(4, 2, 2, 2));
            t1.FillWithRange(0);
            Tensor t2 = new Tensor(new Shape(2, 4, 2));
            t2.FillWithRange(0);

            Tensor r = t1.Mul(t2);
            Tensor correct = new Tensor(new float[] { 28, 34, 76, 98, 428, 466, 604, 658, 220, 290, 268, 354, 1132, 1234, 1308, 1426 }, new Shape(2, 2, 2, 2));

            Assert.IsTrue(r.Equals(correct));
        }

        [TestMethod]
        public void Mul_2Batches_2Batches()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            Tensor t1 = new Tensor(new Shape(4, 2, 2, 2));
            t1.FillWithRange(0);
            Tensor t2 = new Tensor(new Shape(2, 4, 2, 2));
            t2.FillWithRange(0);

            Tensor r = t1.Mul(t2);
            Tensor correct = new Tensor(new float[] { 28, 34, 76, 98, 428, 466, 604, 658, 1340, 1410, 1644, 1730, 2764, 2866, 3196, 3314 }, new Shape(2, 2, 2, 2));

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
            Tensor correct = new Tensor(new float[] { 5511, 5664, 5817, 5970, 6429, 6582, 6735, 6888, 7347, 7500, 7653, 7806, 8265, 8418, 8571, 8724 }, new Shape(4, 4, 1));

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
            Tensor correct = new Tensor(new float[] { 5511, 5664, 5817, 5970, 6429, 6582, 6735, 6888, 7347, 7500, 7653, 7806, 8265, 8418, 8571, 8724, 13611, 14088, 14565, 15042, 16473, 16950, 17427, 17904, 19335, 19812, 20289, 20766, 22197, 22674, 23151, 23628, 21711, 22512, 23313, 24114, 26517, 27318, 28119, 28920, 31323, 32124, 32925, 33726, 36129, 36930, 37731, 38532 }, new Shape(4, 4, 3));

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
            Tensor correct = new Tensor(new float[] { 5511, 5664, 5817, 5970, 6429, 6582, 6735, 6888, 7347, 7500, 7653, 7806, 8265, 8418, 8571, 8724, 13611, 14088, 14565, 15042, 16473, 16950, 17427, 17904, 19335, 19812, 20289, 20766, 22197, 22674, 23151, 23628, 16527, 16680, 16833, 16986, 17445, 17598, 17751, 17904, 18363, 18516, 18669, 18822, 19281, 19434, 19587, 19740, 47955, 48432, 48909, 49386, 50817, 51294, 51771, 52248, 53679, 54156, 54633, 55110, 56541, 57018, 57495, 57972 }, new Shape(4, 4, 2, 2));

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
            Tensor correct = new Tensor(new float[] { 2492, 3674, 3794, 3914, 4034, 2624, 3765, 5511, 5664, 5817, 5970, 3855, 4413, 6429, 6582, 6735, 6888, 4431, 5061, 7347, 7500, 7653, 7806, 5007, 5709, 8265, 8418, 8571, 8724, 5583, 3416, 4898, 4982, 5066, 5150, 3260 }, new Shape(6, 6, 1));

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
            Tensor correct = new Tensor(new float[] { 612, 1213, 1801, 1870, 1939, 2008, 1315, 645, 1266, 2492, 3674, 3794, 3914, 4034, 2624, 1278, 1926, 3765, 5511, 5664, 5817, 5970, 3855, 1863, 2268, 4413, 6429, 6582, 6735, 6888, 4431, 2133, 2610, 5061, 7347, 7500, 7653, 7806, 5007, 2403, 2952, 5709, 8265, 8418, 8571, 8724, 5583, 2673, 1782, 3416, 4898, 4982, 5066, 5150, 3260, 1542, 786, 1489, 2107, 2140, 2173, 2206, 1375, 639 }, new Shape(8, 8, 1));

            Assert.IsTrue(r.Equals(correct));
        }

        [TestMethod]
        public void Conv2DInputGradient_3Batches_10Kernels()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor output = new Tensor(new Shape(24, 24, 10, 3)); output.FillWithRand();
            Tensor input = new Tensor(new Shape(26, 26, 32, 3)); input.FillWithRand();
            Tensor kernels = new Tensor(new Shape(3, 3, 32, 10)); kernels.FillWithRand();
            Tensor gradient = new Tensor(output); gradient.FillWithRand();

            Tensor inputGradient = new Tensor(input);
            Tensor.Conv2DInputsGradient(gradient, kernels, 1, inputGradient);

            Tensor inputGradient2 = new Tensor(input);
            Tensor kernelsGradient = new Tensor(kernels);
            Tensor.Conv2DGradient_old(input, kernels, gradient, 1, Tensor.PaddingType.Valid, inputGradient2, kernelsGradient);

            Assert.IsTrue(inputGradient.Equals(inputGradient2));
        }

        [TestMethod]
        public void Conv2DKernelsGradient_3Batches_10Kernels()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);
            Tensor output = new Tensor(new Shape(24, 24, 10, 3)); output.FillWithRand();
            Tensor input = new Tensor(new Shape(26, 26, 32, 3)); input.FillWithRand();
            Tensor kernels = new Tensor(new Shape(3, 3, 32, 10)); kernels.FillWithRand();
            Tensor gradient = new Tensor(output); gradient.FillWithRand();

            Tensor kernelsGradient = new Tensor(kernels);
            Tensor.Conv2DKernelsGradient(input, gradient, 1, Tensor.PaddingType.Valid, kernelsGradient);

            Tensor inputGradient = new Tensor(input);
            Tensor kernelsGradient2 = new Tensor(kernels);
            Tensor.Conv2DGradient_old(input, kernels, gradient, 1, Tensor.PaddingType.Valid, inputGradient, kernelsGradient2);

            Assert.IsTrue(kernelsGradient.Equals(kernelsGradient2));
        }

        [TestMethod]
        public void Pool_Max_Valid_1Batch_Stride2()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            Tensor t1 = new Tensor(new Shape(6, 6));
            t1.FillWithRange(0);

            Tensor r = t1.Pool(2, 2, Tensor.PoolType.Max, Tensor.PaddingType.Valid);
            Tensor correct = new Tensor(new float[] { 7, 9, 11, 19, 21, 23, 31, 33, 35 }, new Shape(3, 3, 1));

            Assert.IsTrue(r.Equals(correct));
        }

        [TestMethod]
        public void Pool_Max_Valid_2Batches_Stride2()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            Tensor t1 = new Tensor(new Shape(6, 6, 1, 2)); t1.FillWithRange(0);

            Tensor r = t1.Pool(2, 2, Tensor.PoolType.Max, Tensor.PaddingType.Valid);
            Tensor correct = new Tensor(new float[] { 7, 9, 11, 19, 21, 23, 31, 33, 35, 43, 45, 47, 55, 57, 59, 67, 69, 71 }, new Shape(3, 3, 1, 2));

            Assert.IsTrue(r.Equals(correct));
        }

        [TestMethod]
        public void Pool_Avg_Valid_2Batches_Stride2()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            Tensor t1 = new Tensor(new Shape(6, 6, 1, 2)); t1.FillWithRange(0);

            Tensor r = t1.Pool(2, 2, Tensor.PoolType.Avg, Tensor.PaddingType.Valid);
            Tensor correct = new Tensor(new float[] { 3.5f, 5.5f, 7.5f, 15.5f, 17.5f, 19.5f, 27.5f, 29.5f, 31.5f, 39.5f, 41.5f, 43.5f, 51.5f, 53.5f, 55.5f, 63.5f, 65.5f, 67.5f }, new Shape(3, 3, 1, 2));

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
            Tensor correct = new Tensor(new float[] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 5, 0, 6, 0, 0, 0, 0, 0, 0, 0, 7, 0, 8, 0, 9, 0, 0, 0, 0, 0, 0, 0, 10, 0, 11, 0, 12, 0, 0, 0, 0, 0, 0, 0, 13, 0, 14, 0, 15, 0, 0, 0, 0, 0, 0, 0, 16, 0, 17, 0, 18 }, new Shape(6, 6, 1, 2));

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
            Tensor correct = new Tensor(new float[] { 0.25f, 0.25f, 0.5f, 0.5f, 0.75f, 0.75f, 0.25f, 0.25f, 0.5f, 0.5f, 0.75f, 0.75f, 1, 1, 1.25f, 1.25f, 1.5f, 1.5f, 1, 1, 1.25f, 1.25f, 1.5f, 1.5f, 1.75f, 1.75f, 2, 2, 2.25f, 2.25f, 1.75f, 1.75f, 2, 2, 2.25f, 2.25f, 2.5f, 2.5f, 2.75f, 2.75f, 3, 3, 2.5f, 2.5f, 2.75f, 2.75f, 3, 3, 3.25f, 3.25f, 3.5f, 3.5f, 3.75f, 3.75f, 3.25f, 3.25f, 3.5f, 3.5f, 3.75f, 3.75f, 4, 4, 4.25f, 4.25f, 4.5f, 4.5f, 4, 4, 4.25f, 4.25f, 4.5f, 4.5f }, new Shape(6, 6, 1, 2));

            Assert.IsTrue(result.Equals(correct));
        }

        [TestMethod]
        public void Clip_Max()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            var t = new Tensor(new Shape(2, 3, 4, 5)); t.FillWithRange(t.Shape.Length, 0.5f);
            var result = t.Clipped(-0.1f, 0.1f);

            for (int i = 0; i < t.Shape.Length; ++i)
                Assert.AreEqual(result.GetFlat(i), 0.1, 1e-7);
        }

        [TestMethod]
        public void Clip_Min()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            var t = new Tensor(new Shape(2, 3, 4, 5)); t.FillWithRange(-t.Shape.Length, 0.5f);
            var result = t.Clipped(-0.1f, 0.1f);

            for (int i = 0; i < t.Shape.Length; ++i)
                Assert.AreEqual(result.GetFlat(i), -0.1f, 1e-7f);
        }

        [TestMethod]
        public void Negated()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            var t = new Tensor(new Shape(2, 3, 4, 5)); t.FillWithRand();
            var result = t.Negated();

            for (int i = 0; i < t.Shape.Length; ++i)
                Assert.AreEqual(result.GetFlat(i), -t.GetFlat(i), 1e-7);
        }

        [TestMethod]
        public void Sum_Per_Batch()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            var t = new Tensor(new Shape(2, 2, 1, 3)); t.FillWithRange(1);
            var sums = new float[] { 10, 26, 42 };

            for (int i = 0; i < t.BatchSize; ++i)
                Assert.AreEqual(t.Sum(i), sums[i], 1e-7);
        }

        [TestMethod]
        public void Sum()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            var t = new Tensor(new Shape(2, 2, 1, 3)); t.FillWithRange(1);

            Assert.AreEqual(t.Sum(), 78, 1e-7);
        }

        [TestMethod]
        public void SumBatches()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            var t = new Tensor(new Shape(2, 2, 1, 4)); t.FillWithRange(1);
            var result = t.SumBatches();
            var correct = new Tensor(new float[] { 28, 32, 36, 40 }, new Shape(2, 2, 1, 1));

            Assert.IsTrue(result.Equals(correct));
        }

        [TestMethod]
        public void Avg_Per_Batch()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            var t = new Tensor(new Shape(2, 2, 1, 3)); t.FillWithRange(1);
            var averages = new float[] { 2.5f, 6.5f, 10.5f };

            for (int i = 0; i < t.BatchSize; ++i)
                Assert.AreEqual(t.Avg(i), averages[i], 1e-7);
        }

        [TestMethod]
        public void Avg()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            var t = new Tensor(new Shape(2, 2, 1, 3)); t.FillWithRange(1);

            Assert.AreEqual(t.Avg(), 6.5, 1e-7);
        }

        [TestMethod]
        public void AvgBatches()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            var t = new Tensor(new Shape(2, 2, 1, 4)); t.FillWithRange(1);
            var result = t.AvgBatches();
            var correct = new Tensor(new float[] { 7, 8, 9, 10 }, new Shape(2, 2, 1, 1));

            Assert.IsTrue(result.Equals(correct));
        }

        [TestMethod]
        public void Resized()
        {
            var t = new Tensor(new float[] { 1, 2, 3, 4 }, new Shape(2, 1, 1, 2));
            var result = t.Resized(1, 3);

            var correct = new Tensor(new float[] { 1, 2, 1, 3, 4, 3 }, new Shape(1, 3, 1, 2));

            Assert.IsTrue(result.Equals(correct));
        }
        

        [TestMethod]
        public void ArgMax()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            var t = new Tensor(new float[] { -20, 1, 5, 5, 6, -1, 3, 4, 2, 1, 16, 5, 3, 1, 10, 11 }, new Shape(2, 2, 1, 4));

            Assert.AreEqual(t.ArgMax(), 10);
        }

        [TestMethod]
        public void ArgMax_Per_Batch()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            var t = new Tensor(new float[] { -20, 1, 5, 5, 6, -1, 3, 4, 2, 1, 16, 5, 3, 1, 10, 11 }, new Shape(2, 2, 1, 4));
            var maxes = new int[] { 2, 0, 2, 3 };

            for (int i = 0; i < t.BatchSize; ++i)
                Assert.AreEqual(t.ArgMax(i), maxes[i]);
        }

        [TestMethod]
        public void CopyBatchTo()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            var t = new Tensor(new Shape(2, 2, 1, 4)); t.FillWithRange(1);
            var result = new Tensor(new Shape(2, 2, 1, 1));
            t.CopyBatchTo(1, 0, result);
            var correct = new Tensor(result.Shape); correct.FillWithRange(5);

            Assert.IsTrue(result.Equals(correct));
        }

        [TestMethod]
        public void Merge_Into_Batch()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            List<Tensor> tensors = new List<Tensor>();

            for (int i = 0; i < 5; ++i)
            {
                var t = new Tensor(new Shape(2,3,4));
                t.FillWithRand();
                tensors.Add(t);
            }

            var result = Tensor.MergeIntoBatch(tensors);

            for (int i = 0; i < tensors.Count; ++i)
                Assert.IsTrue(result.GetBatch(i).Equals(tensors[i]));
        }

        [TestMethod]
        public void Map()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            var t = new Tensor(new Shape(2, 3, 4, 5)); t.FillWithRand();
            var result = t.Map(x => x * 2);

            for (int i = 0; i < t.Shape.Length; ++i)
                Assert.AreEqual(result.GetFlat(i), 2 * t.GetFlat(i), 1e-7);
        }

        [TestMethod]
        public void Map_With_Other()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            var t = new Tensor(new Shape(2, 3, 4, 5)); t.FillWithRand();
            var other = new Tensor(new Shape(2, 3, 4, 5)); other.FillWithRand();
            var result = t.Map((x, x2) => x * x2, other);

            for (int i = 0; i < t.Shape.Length; ++i)
                Assert.AreEqual(result.GetFlat(i), t.GetFlat(i) * other.GetFlat(i), 1e-7);
        }

        [TestMethod]
        public void Concat()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            var t1 = new Tensor(new Shape(2, 2, 1, 2)); t1.FillWithRange();
            var t2 = new Tensor(new Shape(2, 2, 1, 2)); t2.FillWithRange(8);
            var inputs = new[] {t1, t2};
            var result = new Tensor(new Shape(1, inputs.Select(x => x.BatchLength).Sum(), 1, 2));

            Tensor.Concat(inputs, result);

            var correct = new Tensor(new float[] {0,1,2,3,8,9,10,11,4,5,6,7,12,13,14,15}, result.Shape);

            Assert.IsTrue(result.Equals(correct));
        }

        [TestMethod]
        public void Split()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);

            var t1 = new Tensor(new Shape(2, 2, 1, 2));
            var t2 = new Tensor(new Shape(2, 2, 1, 2));
            var inputs = new[] { t1, t2 };
            var concated = new Tensor(new float[] { 0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15 }, new Shape(1, inputs.Select(x => x.BatchLength).Sum(), 1, 2));

            concated.Split(inputs);

            var correct1 = new Tensor(new Shape(2, 2, 1, 2)); correct1.FillWithRange();
            var correct2 = new Tensor(new Shape(2, 2, 1, 2)); correct2.FillWithRange(8);
            var correctInputs = new[] { correct1, correct2 };

            for (int i = 0; i < inputs.Length; ++i)
                Assert.IsTrue(inputs[i].Equals(correctInputs[i]));
        }

        [TestMethod]
        public void Serialize_Deserialize()
        {
            string tempFilename = "tensor_tmp.txt";

            var t = new Tensor(new Shape(5, 4, 3, 2));
            t.FillWithRand();
            using (BinaryWriter writer = new BinaryWriter(File.Open(tempFilename, FileMode.Create)))
            {
                t.Serialize(writer);
            }

            using (BinaryReader reader = new BinaryReader(File.Open(tempFilename, FileMode.Open)))
            {
                Assert.IsTrue(t.Equals(Tensor.Deserialize(reader)));
            }

            File.Delete(tempFilename);
        }
    }
}
