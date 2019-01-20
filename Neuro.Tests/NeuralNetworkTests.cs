using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neuro.Layers;
using Neuro.Models;
using Neuro.Optimizers;
using Neuro.Tensors;

namespace Neuro.Tests
{
    [TestClass]
    public class NeuralNetworkTests
    {
        [TestMethod]
        public void Dense_Network_BS1()
        {
            TestDenseNetwork(2, 50, 1, 50);
        }

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
            Sequential seqModel = net.Model as Sequential;

            var expectedWeights = new Tensor(new[] { 1.1f, 0.1f, -1.3f, 0.2f, -0.9f, 0.7f }, new Shape(3, 2));
            var tData = GenerateTrainingData(50, seqModel.LastLayer.InputShapes[0], expectedWeights, MatMult);

            var inputs = new Tensor(new Shape(seqModel.Layer(0).InputShape.Width, seqModel.Layer(0).InputShape.Height, seqModel.Layer(0).InputShape.Depth, tData.Count));
            var outputs = new Tensor(new Shape(seqModel.LastLayer.OutputShape.Width, seqModel.LastLayer.OutputShape.Height, seqModel.LastLayer.OutputShape.Depth, tData.Count));
            for (int i = 0; i < tData.Count; ++i)
            {
                tData[i].Input.CopyBatchTo(0, i, inputs);
                tData[i].Output.CopyBatchTo(0, i, outputs);
            }

            net.FitBatched(inputs, outputs, 300, 0, Track.Nothing);

            var paramsAndGrads = seqModel.LastLayer.GetParametersAndGradients();

            for (int i = 0; i < expectedWeights.Length; ++i)
                Assert.AreEqual(paramsAndGrads[0].Parameters.GetFlat(i), expectedWeights.GetFlat(i), 1e-2);
        }

        [TestMethod]
        public void Fit_Batched_Data()
        {
            NeuralNetwork net = CreateFitTestNet();
            Sequential seqModel = net.Model as Sequential;

            var expectedWeights = new Tensor(new[] { 1.1f, 0.1f, -1.3f, 0.2f, -0.9f, 0.7f }, new Shape(3, 2));
            var tempData = GenerateTrainingData(50, seqModel.LastLayer.InputShapes[0], expectedWeights, MatMult);

            var inputs = new Tensor(new Shape(seqModel.Layer(0).InputShapes[0].Width, seqModel.Layer(0).InputShapes[0].Height, seqModel.Layer(0).InputShapes[0].Depth, tempData.Count));
            var outputs = new Tensor(new Shape(seqModel.LastLayer.OutputShape.Width, seqModel.LastLayer.OutputShape.Height, seqModel.LastLayer.OutputShape.Depth, tempData.Count));
            for (int i = 0; i < tempData.Count; ++i)
            {
                tempData[i].Inputs[0].CopyBatchTo(0, i, inputs);
                tempData[i].Outputs[0].CopyBatchTo(0, i, outputs);
            }

            var tData = new List<Data> { new Data(inputs, outputs) };

            net.Fit(tData, -1, 300, null, 0, Track.Nothing);

            var paramsAndGrads = seqModel.LastLayer.GetParametersAndGradients();

            for (int i = 0; i < expectedWeights.Length; ++i)
                Assert.AreEqual(paramsAndGrads[0].Parameters.GetFlat(i), expectedWeights.GetFlat(i), 1e-2);
        }

        private NeuralNetwork CreateFitTestNet()
        {
            var net = new NeuralNetwork("fit_test", 7);
            var model = new Sequential();
            model.AddLayer(new Dense(3, 2, Activation.Linear) { KernelInitializer = new Initializers.Constant(1), UseBias = false });
            net.Model = model;
            net.Optimize(new SGD(0.07f), Loss.MeanSquareError);
            return net;
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

        [TestMethod]
        public void CopyParameters()
        {
            var net = new NeuralNetwork("test");
            var model = new Sequential();
            model.AddLayer(new Dense(2, 3, Activation.Linear));
            model.AddLayer(new Dense(3, 3, Activation.Linear));
            net.Model = model;

            var net2 = net.Clone();
            net.CopyParametersTo(net2);

            var netParams = net2.GetParametersAndGradients();
            var net2Params = net2.GetParametersAndGradients();

            for (int i = 0; i < netParams.Count; ++i)
                Assert.IsTrue(netParams[i].Parameters.Equals(net2Params[i].Parameters));
        }

        [TestMethod]
        public void SoftCopyParameters()
        {
            var net = new NeuralNetwork("test");
            var model = new Sequential();
            model.AddLayer(new Dense(2, 3, Activation.Linear));
            model.AddLayer(new Dense(3, 3, Activation.Linear));
            net.Model = model;

            var net2 = net.Clone();

            net.SoftCopyParametersTo(net2, 0.1f);

            var netParams = net2.GetParametersAndGradients();
            var net2Params = net2.GetParametersAndGradients();

            for (int i = 0; i < netParams.Count; ++i)
                Assert.IsTrue(netParams[i].Parameters.Equals(net2Params[i].Parameters));
        }

        private void TestDenseLayer(int inputs, int outputs, int samples, int batchSize, int epochs)
        {
            var net = new NeuralNetwork("dense_test", 7);
            var model = new Sequential();
            model.AddLayer(new Dense(inputs, outputs, Activation.Linear) { KernelInitializer = new Initializers.Constant(1), UseBias = false });
            net.Model = model;

            var expectedWeights = new Tensor(new[] { 1.1f, 0.1f, -1.3f, 0.2f, -0.9f, 0.7f }, new Shape(3, 2));
            var tData = GenerateTrainingData(samples, model.LastLayer.InputShape, expectedWeights, MatMult);

            net.Optimize(new SGD(0.07f), Loss.MeanSquareError);
            net.Fit(tData, batchSize, epochs, null, 2, Track.TrainError);

            var paramsAndGrads = model.LastLayer.GetParametersAndGradients();

            for (int i = 0; i < expectedWeights.Length; ++i)
                Assert.AreEqual(paramsAndGrads[0].Parameters.GetFlat(i), expectedWeights.GetFlat(i), 1e-2);
        }

        private void TestDenseNetwork(int inputs, int samples, int batchSize, int epochs)
        {
            var net = new NeuralNetwork("deep_dense_test", 7);
            var model = new Sequential();
            model.AddLayer(new Dense(inputs, 5, Activation.Linear));
            model.AddLayer(new Dense(model.LastLayer, 4, Activation.Linear));
            model.AddLayer(new Dense(model.LastLayer, inputs, Activation.Linear));
            net.Model = model;


            List<Data> tData = new List<Data>();
            for (int i = 0; i < samples; ++i)
            {
                var input = new Tensor(model.Layer(0).InputShape);
                input.FillWithRand(10 * i, -2, 2);
                tData.Add(new Data(input, input.Mul(1.7f)));
            }

            net.Optimize(new SGD(0.02f), Loss.MeanSquareError);
            net.Fit(tData, batchSize, epochs, null, 2, Track.TrainError);

            for (int i = 0; i < tData.Count; ++i)
                Assert.IsTrue(tData[i].Output.Equals(net.Predict(tData[i].Input)[0], 0.01f));
        }

        private void TestConvolutionLayer(Shape inputShape, int kernelSize, int kernelsNum, int stride, int samples, int batchSize, int epochs, TrainDataFunc convFunc)
        {
            var net = new NeuralNetwork("convolution_test", 7);
            var model = new Sequential();
            model.AddLayer(new Convolution(inputShape, kernelSize, kernelsNum, stride, Activation.Linear) { KernelInitializer = new Initializers.Constant(1) });
            net.Model = model;

            var expectedKernels = new Tensor(new Shape(kernelSize, kernelSize, inputShape.Depth, kernelsNum));
            expectedKernels.FillWithRand(17);

            var tData = GenerateTrainingData(samples, model.LastLayer.InputShape, expectedKernels, convFunc);
            
            net.Optimize(new SGD(0.02f), Loss.MeanSquareError);
            net.Fit(tData, batchSize, epochs, null, 0, Track.Nothing);

            var paramsAndGrads = model.LastLayer.GetParametersAndGradients();

            for (int i = 0; i < expectedKernels.Length; ++i)
                Assert.AreEqual(paramsAndGrads[0].Parameters.GetFlat(i), expectedKernels.GetFlat(i), 1e-2);
        }

        [TestMethod]
        public void Streams_1Input_2Outputs_SimpleSplit()
        {
            var input1 = new Dense(2, 2, Activation.Sigmoid) { Name = "input1" }; ;
            var upperStream1 = new Dense(input1, 2, Activation.Linear) { Name = "upperStream1" }; ;
            var lowerStream1 = new Dense(input1, 2, Activation.Linear) { Name = "lowerStream1" };

            var net = new NeuralNetwork("test");
            net.Model = new Flow(new[] { input1 }, new[] { upperStream1, lowerStream1 });

            net.Optimize(new SGD(0.05f), Loss.MeanSquareError);

            var input = new Tensor(new float[] { 0, 1 }, new Shape(1, 2));
            var outputs = new[] { new Tensor(new float[] { 0, 1 }, new Shape(1, 2)),
                                  new Tensor(new float[] { 1, 2 }, new Shape(1, 2)) };
            var trainingData = new List<Data> { new Data(new[] { input }, outputs) };

            net.Fit(trainingData, 1, 100, null, 0, Track.Nothing, false);

            var prediction = net.Predict(input);
            Assert.IsTrue(prediction[0].Equals(outputs[0], 0.01f));
            Assert.IsTrue(prediction[1].Equals(outputs[1], 0.01f));
        }

        [TestMethod]
        public void Streams_2Inputs_1Output_SimpleConcat()
        {
            LayerBase mainInput = new Dense(2, 2, Activation.Linear) { Name = "main_input" };
            LayerBase auxInput = new Input(new Shape(1, 2)) { Name = "aux_input" };
            LayerBase concat = new Concat(new []{ mainInput, auxInput }) { Name = "concat" };

            var net = new NeuralNetwork("test");
            net.Model = new Flow(new[] { mainInput, auxInput }, new[] { concat });

            net.Optimize(new SGD(0.05f), Loss.MeanSquareError);

            var inputs = new[] { new Tensor(new float[] { 0, 1 }, new Shape(1, 2)),
                                 new Tensor(new float[] { 1, 2 }, new Shape(1, 2)) };
            var output = new Tensor(new float[] { 1, 2, 1, 2 }, new Shape(1, 4));
            var trainingData = new List<Data> { new Data(inputs, new []{output}) };

            net.Fit(trainingData, 1, 100, null, 0, Track.Nothing, false);

            var prediction = net.Predict(inputs);
            Assert.IsTrue(prediction[0].Equals(output, 0.01f));
        }

        [TestMethod]
        public void Streams_2Inputs_1Output_AvgMerge()
        {
            LayerBase input1 = new Dense(2, 2, Activation.Linear) { Name = "input1" };
            LayerBase input2 = new Dense(2, 2, Activation.Linear) { Name = "input2" };
            LayerBase avgMerge = new Merge(new[] { input1, input2 }, Merge.Mode.Avg) { Name = "avg_merge" };

            var net = new NeuralNetwork("test");
            net.Model = new Flow(new[] { input1, input2 }, new[] { avgMerge });

            net.Optimize(new SGD(0.05f), Loss.MeanSquareError);

            var inputs = new[] { new Tensor(new float[] { 0, 1 }, new Shape(1, 2)),
                                 new Tensor(new float[] { 1, 2 }, new Shape(1, 2)) };
            var output = new Tensor(new float[] { 2, 4 }, new Shape(1, 2));
            var trainingData = new List<Data> { new Data(inputs, new[] { output }) };

            net.Fit(trainingData, 1, 100, null, 0, Track.Nothing, false);

            var prediction = net.Predict(inputs);
            Assert.IsTrue(prediction[0].Equals(output, 0.01f));
        }

        [TestMethod]
        public void Streams_2Inputs_1Output_MinMerge()
        {
            LayerBase input1 = new Dense(2, 2, Activation.Linear) { Name = "input1" };
            LayerBase input2 = new Dense(2, 2, Activation.Linear) { Name = "input2" };
            LayerBase merge = new Merge(new[] { input1, input2 }, Merge.Mode.Min) { Name = "min_merge" };

            var net = new NeuralNetwork("test");
            net.Model = new Flow(new[] { input1, input2 }, new[] { merge });

            net.Optimize(new SGD(0.05f), Loss.MeanSquareError);

            var inputs = new[] { new Tensor(new float[] { 0, 1 }, new Shape(1, 2)),
                new Tensor(new float[] { 1, 2 }, new Shape(1, 2)) };
            var output = new Tensor(new float[] { 2, 4 }, new Shape(1, 2));
            var trainingData = new List<Data> { new Data(inputs, new[] { output }) };

            net.Fit(trainingData, 1, 100, null, 0, Track.Nothing, false);

            var prediction = net.Predict(inputs);
            Assert.IsTrue(prediction[0].Equals(output, 0.01f));
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
                trainingData.Add(new Data(input , f(input, expectedParams)));
            }

            return trainingData;
        }
    }
}
