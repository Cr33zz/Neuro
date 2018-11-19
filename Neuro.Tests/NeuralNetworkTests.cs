using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neuro.Layers;
using Neuro.Optimizers;
using Neuro.Tensors;

namespace Neuro.Tests
{
    [TestClass]
    public class NeuralNetworkTests
    {
        [TestMethod]
        public void Two_Dense_Layers_Batch_10()
        {
            TestDenseNetwork(2, 100, 10, 100);
        }

        [TestMethod]
        public void Two_Dense_Layers_Batch_1()
        {
            TestDenseNetwork(2, 5, 1, 500);
        }

        [TestMethod]
        public void Single_Dense_Layer_Batch_10()
        {
            TestDenseLayer(3, 2, 100, 10, 50);
        }

        [TestMethod]
        public void Single_Dense_Layer_Batch_1()
        {
            TestDenseLayer(3, 2, 100, 1, 50);
        }

        [TestMethod]
        public void Single_Dense_Layer_Stochastic()
        {
            TestDenseLayer(3, 2, 100, -1, 300);
        }

        [TestMethod]
        public void Single_Convolution_Layer_Batch_10_VS1()
        {
            TestConvolutionLayer(new Shape(9, 9, 2), 3, 4, 1, 100, 10, 15, ConvValidStride1);
        }

        [TestMethod]
        public void Single_Convolution_Layer_Batch_1_VS1()
        {
            TestConvolutionLayer(new Shape(9, 9, 2), 3, 4, 1, 100, 1, 10, ConvValidStride1);
        }

        [TestMethod]
        public void Single_Convolution_Layer_Stochastic_VS1()
        {
            TestConvolutionLayer(new Shape(9, 9, 2), 3, 4, 1, 100, -1, 20, ConvValidStride1);
        }

        [TestMethod]
        public void Single_Convolution_Layer_Batch_10_VS2()
        {
            TestConvolutionLayer(new Shape(9, 9, 2), 3, 4, 2, 100, 10, 15, ConvValidStride2);
        }

        [TestMethod]
        public void Single_Convolution_Layer_Batch_10_VS3()
        {
            TestConvolutionLayer(new Shape(9, 9, 2), 3, 4, 3, 100, 10, 15, ConvValidStride3);
        }

        [TestMethod]
        public void Batching_No_Reminder()
        {
            var tData = GenerateTrainingData(100, new Shape(1, 3), new Tensor(new Shape(3, 2)), MatMult);

            var trainingBatches = Neuro.Tools.MergeData(tData, 10);

            Assert.AreEqual(trainingBatches[0].Input.BatchSize, 10);
            Assert.AreEqual(trainingBatches[0].Output.BatchSize, 10);
        }

        private void TestDenseLayer(int inputs, int outputs, int samples, int batchSize, int epochs)
        {
            var net = new NeuralNetwork("dense_test", 7);
            net.AddLayer(new Dense(inputs, outputs, Activation.Linear) { KernelInitializer = new Initializers.Constant(1) });

            var expectedWeights = new Tensor(new[] { 1.1, 0.1, -1.3, 0.2, -0.9, 0.7 }, new Shape(3, 2));
            var tData = GenerateTrainingData(samples, net.LastLayer.InputShape, expectedWeights, MatMult);

            net.Optimize(new SGD(0.07), Loss.MeanSquareError);
            net.Fit(tData, batchSize, epochs, null, 0, Track.TrainError);

            var learnedParams = net.LastLayer.GetParameters();

            for (int i = 0; i < expectedWeights.Length; ++i)
                Assert.AreEqual(learnedParams.GetFlat(i), expectedWeights.GetFlat(i), 1e-2);
        }

        private void TestDenseNetwork(int inputs, int samples, int batchSize, int epochs)
        {
            var net = new NeuralNetwork("deep_dense_test", 7);
            net.AddLayer(new Dense(inputs, 12, Activation.Sigmoid) { KernelInitializer = new Initializers.Constant(1) });
            net.AddLayer(new Dense(net.LastLayer, 12, Activation.Sigmoid) { KernelInitializer = new Initializers.Constant(1) });
            net.AddLayer(new Dense(net.LastLayer, inputs, Activation.Linear) { KernelInitializer = new Initializers.Constant(1) });

            List<Data> tData = new List<Data>();

            for (int i = 0; i < samples; ++i)
            {
                var input = new Tensor(net.Layer(0).InputShape);
                input.FillWithRand();
                var output = new Tensor(net.Layer(0).InputShape);
                output.FillWithRand();
                tData.Add(new Data() { Input = input, Output = output});
            }

            net.Optimize(new SGD(0.05), Loss.MeanSquareError);
            net.Fit(tData, batchSize, epochs, null, 0, Track.TrainError);

            for (int i = 0; i < tData.Count; ++i)
                Assert.IsTrue(tData[i].Output.Equals(net.Predict(tData[i].Input), 0.001));
        }

        private void TestConvolutionLayer(Shape inputShape, int kernelSize, int kernelsNum, int stride, int samples, int batchSize, int epochs, TrainDataFunc convFunc)
        {
            var net = new NeuralNetwork("convolution_test", 7);
            net.AddLayer(new Convolution(inputShape, kernelSize, kernelsNum, stride, Activation.Linear) { KernelInitializer = new Initializers.Constant(1) });

            var expectedKernels = new Tensor(new Shape(kernelSize, kernelSize, inputShape.Depth, kernelsNum));
            expectedKernels.FillWithRand(17);

            var tData = GenerateTrainingData(samples, net.LastLayer.InputShape, expectedKernels, convFunc);
            
            net.Optimize(new SGD(), Loss.MeanSquareError);
            net.Fit(tData, batchSize, epochs, null, 0, Track.TrainError);

            var learnedParams = net.LastLayer.GetParameters();

            for (int i = 0; i < expectedKernels.Length; ++i)
                Assert.AreEqual(learnedParams.GetFlat(i), expectedKernels.GetFlat(i), 1e-2);
        }

        private delegate Tensor TrainDataFunc(Tensor input, Tensor expectedParams);
        private delegate Tensor TrainDataFuncNoParams(Tensor input);

        private Tensor MatMult(Tensor input, Tensor expectedParams)
        {
            return expectedParams.Mul(input);
        }

        //private Tensor TestFunc1(Tensor input)
        //{
        //    var o = new Tensor(input.Shape);

        //    return ;
        //}

        private Tensor ConvValidStride1(Tensor input, Tensor expectedParams)
        {
            return input.Conv2D(expectedParams, 1, Tensor.PaddingType.Valid);
        }

        private Tensor ConvValidStride2(Tensor input, Tensor expectedParams)
        {
            return input.Conv2D(expectedParams, 2, Tensor.PaddingType.Valid);
        }

        private Tensor ConvValidStride3(Tensor input, Tensor expectedParams)
        {
            return input.Conv2D(expectedParams, 3, Tensor.PaddingType.Valid);
        }

        private List<Data> GenerateTrainingData(int samples, Shape inShape, Tensor expectedParams, TrainDataFunc f)
        {
            List<Data> trainingData = new List<Data>();

            for (int i = 0; i < samples; ++i)
            {
                var input = new Tensor(inShape);
                input.FillWithRand(3 * i);
                trainingData.Add(new Data() { Input = input , Output = f(input, expectedParams) });
            }

            return trainingData;
        }
    }
}
