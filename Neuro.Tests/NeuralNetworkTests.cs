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
        public void Dense_Network_BS10()
        {
            TestDenseNetwork(2, 50, 10, 100);
        }

        [TestMethod]
        public void Dense_Network_FullBatch()
        {
            TestDenseNetwork(2, 50, -1, 100);
        }

        [TestMethod]
        public void Fit_Batched_Tensors()
        {
            NeuralNetwork net = CreateFitTestNet();
            
            var expectedWeights = new Tensor(new[] { 1.1f, 0.1f, -1.3f, 0.2f, -0.9f, 0.7f }, new Shape(3, 2));
            var tData = GenerateTrainingData(50, net.LastLayer.InputShape, expectedWeights, MatMult);

            var inputs = new Tensor(new Shape(net.Layer(0).InputShape.Width, net.Layer(0).InputShape.Height, net.Layer(0).InputShape.Depth, tData.Count));
            var outputs = new Tensor(new Shape(net.LastLayer.OutputShape.Width, net.LastLayer.OutputShape.Height, net.LastLayer.OutputShape.Depth, tData.Count));
            for (int i = 0; i < tData.Count; ++i)
            {
                tData[i].Input.CopyBatchTo(0, i, inputs);
                tData[i].Output.CopyBatchTo(0, i, outputs);
            }

            net.Fit(inputs, outputs, -1, 300, 0, Track.Nothing);

            var paramsAndGrads = net.LastLayer.GetParametersAndGradients();

            for (int i = 0; i < expectedWeights.Length; ++i)
                Assert.AreEqual(paramsAndGrads[0].Parameters.GetFlat(i), expectedWeights.GetFlat(i), 1e-2);
        }

        [TestMethod]
        public void Fit_Batched_Data()
        {
            NeuralNetwork net = CreateFitTestNet();

            var expectedWeights = new Tensor(new[] { 1.1f, 0.1f, -1.3f, 0.2f, -0.9f, 0.7f }, new Shape(3, 2));
            var tempData = GenerateTrainingData(50, net.LastLayer.InputShape, expectedWeights, MatMult);

            var inputs = new Tensor(new Shape(net.Layer(0).InputShape.Width, net.Layer(0).InputShape.Height, net.Layer(0).InputShape.Depth, tempData.Count));
            var outputs = new Tensor(new Shape(net.LastLayer.OutputShape.Width, net.LastLayer.OutputShape.Height, net.LastLayer.OutputShape.Depth, tempData.Count));
            for (int i = 0; i < tempData.Count; ++i)
            {
                tempData[i].Input.CopyBatchTo(0, i, inputs);
                tempData[i].Output.CopyBatchTo(0, i, outputs);
            }

            var tData = new List<Data> { new Data() { Input = inputs, Output = outputs } };

            net.Fit(tData, -1, 300, null, 0, Track.Nothing);

            var paramsAndGrads = net.LastLayer.GetParametersAndGradients();

            for (int i = 0; i < expectedWeights.Length; ++i)
                Assert.AreEqual(paramsAndGrads[0].Parameters.GetFlat(i), expectedWeights.GetFlat(i), 1e-2);
        }

        private NeuralNetwork CreateFitTestNet()
        {
            var net = new NeuralNetwork("fit_test", 7);
            net.AddLayer(new Dense(3, 2, Activation.Linear) { KernelInitializer = new Initializers.Constant(1), UseBias = false });
            net.Optimize(new SGD(0.07f), Loss.MeanSquareError);
            return net;
        }

        [TestMethod]
        public void Dense_Network_BS1()
        {
            TestDenseNetwork(2, 50, 1, 500);
        }

        [TestMethod]
        public void Single_Dense_Layer_BS10()
        {
            TestDenseLayer(3, 2, 100, 10, 50);
        }

        [TestMethod]
        public void Single_Dense_Layer_BS1()
        {
            TestDenseLayer(3, 2, 100, 1, 50);
        }

        [TestMethod]
        public void Single_Dense_Layer_FullBatch()
        {
            TestDenseLayer(3, 2, 100, -1, 300);
        }

        [TestMethod]
        public void Single_Convolution_Layer_BS10_VS1()
        {
            TestConvolutionLayer(new Shape(9, 9, 2), 3, 4, 1, 100, 10, 15, ConvValidStride1);
        }

        [TestMethod]
        public void Single_Convolution_Layer_BS1_VS1()
        {
            TestConvolutionLayer(new Shape(9, 9, 2), 3, 4, 1, 100, 1, 10, ConvValidStride1);
        }

        [TestMethod]
        public void Single_Convolution_Layer_FullBatch_VS1()
        {
            TestConvolutionLayer(new Shape(9, 9, 2), 3, 4, 1, 100, -1, 20, ConvValidStride1);
        }

        [TestMethod]
        public void Single_Convolution_Layer_BS10_VS2()
        {
            TestConvolutionLayer(new Shape(9, 9, 2), 3, 4, 2, 100, 10, 15, ConvValidStride2);
        }

        [TestMethod]
        public void Single_Convolution_Layer_BS10_VS3()
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
            net.AddLayer(new Dense(inputs, outputs, Activation.Linear) { KernelInitializer = new Initializers.Constant(1), UseBias = false });

            var expectedWeights = new Tensor(new[] { 1.1f, 0.1f, -1.3f, 0.2f, -0.9f, 0.7f }, new Shape(3, 2));
            var tData = GenerateTrainingData(samples, net.LastLayer.InputShape, expectedWeights, MatMult);

            net.Optimize(new SGD(0.07f), Loss.MeanSquareError);
            net.Fit(tData, batchSize, epochs, null, 2, Track.TrainError);

            var paramsAndGrads = net.LastLayer.GetParametersAndGradients();

            for (int i = 0; i < expectedWeights.Length; ++i)
                Assert.AreEqual(paramsAndGrads[0].Parameters.GetFlat(i), expectedWeights.GetFlat(i), 1e-2);
        }

        private void TestDenseNetwork(int inputs, int samples, int batchSize, int epochs)
        {
            var net = new NeuralNetwork("deep_dense_test", 7);
            net.AddLayer(new Dense(inputs, 5, Activation.Sigmoid));
            net.AddLayer(new Dense(net.LastLayer, 4, Activation.Sigmoid));
            net.AddLayer(new Dense(net.LastLayer, inputs, Activation.Linear));

            List<Data> tData = new List<Data>();
            for (int i = 0; i < samples; ++i)
            {
                var input = new Tensor(net.Layer(0).InputShape);
                input.FillWithRand(10 * i, -2, 2);
                tData.Add(new Data() { Input = input, Output = input.Add(1) });
            }

            net.Optimize(new SGD(), Loss.MeanSquareError);
            net.Fit(tData, batchSize, epochs, null, 2, Track.TrainError);

            for (int i = 0; i < tData.Count; ++i)
                Assert.IsTrue(tData[i].Output.Equals(net.Predict(tData[i].Input), 0.01f));
        }

        private void TestConvolutionLayer(Shape inputShape, int kernelSize, int kernelsNum, int stride, int samples, int batchSize, int epochs, TrainDataFunc convFunc)
        {
            var net = new NeuralNetwork("convolution_test", 7);
            net.AddLayer(new Convolution(inputShape, kernelSize, kernelsNum, stride, Activation.Linear) { KernelInitializer = new Initializers.Constant(1) });

            var expectedKernels = new Tensor(new Shape(kernelSize, kernelSize, inputShape.Depth, kernelsNum));
            expectedKernels.FillWithRand(17);

            var tData = GenerateTrainingData(samples, net.LastLayer.InputShape, expectedKernels, convFunc);
            
            net.Optimize(new SGD(), Loss.MeanSquareError);
            net.Fit(tData, batchSize, epochs, null, 0, Track.Nothing);

            var paramsAndGrads = net.LastLayer.GetParametersAndGradients();

            for (int i = 0; i < expectedKernels.Length; ++i)
                Assert.AreEqual(paramsAndGrads[0].Parameters.GetFlat(i), expectedKernels.GetFlat(i), 1e-2);
        }

        private delegate Tensor TrainDataFunc(Tensor input, Tensor expectedParams);
        private delegate Tensor TrainDataFuncNoParams(Tensor input);

        private Tensor MatMult(Tensor input, Tensor expectedParams)
        {
            return expectedParams.Mul(input);
        }

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
