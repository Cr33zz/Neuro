using System;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Xml;
using Neuro.Tensors;

namespace Neuro
{
    [Flags]
    public enum Track
    {
        Nothing = 0,
        TrainError = 1 << 0,
        TestError = 1 << 1,
        TrainAccuracy = 1 << 2,
        TestAccuracy = 1 << 3,
        All = -1
    }
    
    // Best explanation I found are Neural Network series from Coding Train on YT
    public class NeuralNetwork
    {
        public NeuralNetwork(string name, int seed = 0)
        {
            Name = name;
            if (seed > 0)
            {
                Seed = seed;
                Tools.Rng = new Random(seed);
            }
        }

        public string Name;
        public string FilePrefix { get { return Name.ToLower().Replace(" ", "_"); } }

        private List<Layers.LayerBase> Layers = new List<Layers.LayerBase>();

        public Layers.LayerBase Layer(int i)
        {
            return Layers[i];
        }

        public Layers.LayerBase LastLayer() 
        {
            return Layers.Last();
        }

        public void AddLayer(Layers.LayerBase layer)
        {
            layer.Init();
            Layers.Add(layer);
        }

        public Tensor FeedForward(Tensor input)
        {
            if (NeuralNetwork.DebugMode)
                Trace.WriteLine($"Input:\n{input}\n");

            for (int l = 0; l < Layers.Count; l++)
                Layers[l].FeedForward(l == 0 ? input : Layers[l - 1].Output);

            return Layers.Last().Output;
        }

        private void BackProp(Tensor delta)
        {
            if (NeuralNetwork.DebugMode)
                Trace.WriteLine($"Errors gradient:\n{delta}\n");

            for (int l = Layers.Count - 1; l >= 0; --l)
                delta = Layers[l].BackProp(delta);
        }

        private void UpdateParameters(int trainingSamples)
        {
            for (int l = 0; l < Layers.Count; ++l)
                Layers[l].UpdateParameters(trainingSamples);
        }

        public void Optimize(Optimizers.OptimizerBase optimizer, LossFunc loss)
        {
            Error = loss;
            Optimizer = optimizer;

            for (int l = 0; l < Layers.Count; ++l)
                Layers[l].Optimizer = optimizer.Clone();
        }

        public void Fit(List<Tensor> inputs, List<Tensor> outputs, int batchSize = -1, int epochs = 1, bool verbose = true, Track trackFlags = Track.TrainError | Track.TestAccuracy)
        {
            List<Data> trainingData = new List<Data>();
            for (int i = 0; i < inputs.Count; ++i)
                trainingData.Add(new Data() { Input = inputs[i], Output = outputs[i] });

            Fit(trainingData, batchSize, epochs, null, verbose, trackFlags);
        }

        // Training method, when batch size is -1 the whole training set is used for single gradient descent step (in other words, batch size equals to training set size)
        public void Fit(List<Data> trainingData, int batchSize = -1, int epochs = 1, List<Data> validationData = null, bool verbose = true, Track trackFlags = Track.TrainError | Track.TestAccuracy)
        {
            LogLines.Clear();

            bool trainingDataAlreadyBatched = trainingData[0].Input.BatchSize > 1;
            for (int i = 0; i < trainingData.Count; ++i)
            {
                Data d = trainingData[i];
                Debug.Assert(d.Input.BatchSize == d.Output.BatchSize, $"Training data set contains mismatched number if input and output batches for data at index {i}!");
                Debug.Assert(d.Input.BatchSize == trainingData[0].Input.BatchSize, "Training data set contains batches of different size!");
            }

            if (batchSize < 0)
                batchSize = trainingData.Count;

            string outFilename = $"{FilePrefix}_training_data_{Optimizer.GetType().Name.ToLower()}_b{batchSize}{(Seed > 0 ? ("_seed" + Seed) : "")}_{Tensor.CurrentOpMode}";
            var chartGen = new ChartGenerator($"{outFilename}.png", $"{Name} [{Error.GetType().Name}, {Optimizer}, BatchSize={batchSize}]\nSeed={(Seed > 0 ? Seed.ToString() : "None")}, TensorMode={Tensor.CurrentOpMode}", "Epoch");

            if (trackFlags.HasFlag(Track.TrainError))
                chartGen.AddSeries((int)Track.TrainError, "Error on train data\n(left Y axis)", Color.DarkRed);
            if (trackFlags.HasFlag(Track.TestError))
                chartGen.AddSeries((int)Track.TestError, "Error on test data\n(left Y axis)", Color.IndianRed);
            if (trackFlags.HasFlag(Track.TrainAccuracy))
                chartGen.AddSeries((int)Track.TrainAccuracy, "Accuracy on train data\n(right Y axis)", Color.DarkBlue, true);
            if (trackFlags.HasFlag(Track.TestAccuracy))
                chartGen.AddSeries((int)Track.TestAccuracy, "Accuracy on test\n(right Y axis)", Color.CornflowerBlue, true);

            var lastLayer = Layers.Last();
            Shape inputShape = Layers[0].InputShape;
            int outputsNum = lastLayer.OutputShape.Length;

            int batchesNum = trainingData.Count / batchSize;
            int trainingSamples = trainingData.Count;

            AccuracyFunc accuracyFunc = Tools.CategoricalClassificationEquality;
            if (outputsNum == 1) accuracyFunc = Tools.BinaryClassificationEquality;

            Stopwatch trainTimer = new Stopwatch();

            for (int e = 1; e <= epochs; ++e)
            {
                string output;

                LogLine($"Epoch {e}/{epochs}");

                // no point shuffling stuff when we have single batch
                if (batchesNum > 1 && !trainingDataAlreadyBatched)
                    trainingData.Shuffle();

                List<Data> batchedTrainingData = trainingDataAlreadyBatched ? trainingData : Tools.MergeData(trainingData, batchSize);

                double trainTotalError = 0;
                int trainHits = 0;

                trainTimer.Restart();

                for (int b = 0; b < batchedTrainingData.Count; ++b)
                {
                    // this will be equal to batch size; however, the last batch size may be different if there is a reminder of training data by batch size division
                    int samples = batchedTrainingData[b].Input.BatchSize;
                    GradientDescentStep(batchedTrainingData[b], samples, accuracyFunc, ref trainTotalError, ref trainHits);

                    if (verbose)
                    {
                        output = Tools.GetProgressString(b * batchSize + samples, trainingSamples);
                        Console.Write(output);
                        Console.Write(new string('\b', output.Length));
                    }
                }

                trainTimer.Stop();

                output = Tools.GetProgressString(trainingSamples, trainingSamples);

                if (verbose)
                    Console.Write(output);

                chartGen.AddData(e, trainTotalError / trainingSamples, (int)Track.TrainError);
                chartGen.AddData(e, (double)trainHits / trainingSamples, (int)Track.TrainAccuracy);

                double testTotalError = 0;

                if (validationData != null)
                {
                    int validationSamples = validationData.Count * validationData[0].Input.BatchSize;
                    double testHits = 0;

                    for (int i = 0; i < validationData.Count; ++i)
                    {
                        FeedForward(validationData[i].Input);
                        Tensor loss = new Tensor(lastLayer.Output.Shape);
                        Error.Compute(validationData[i].Output, lastLayer.Output, loss);
                        testTotalError += loss.Sum() / outputsNum;
                        testHits += accuracyFunc(validationData[i].Output, lastLayer.Output);

                        if (verbose)
                        {
                            string progress = " - validating: " + Math.Round(i / (double)validationData.Count * 100) + "%";
                            Console.Write(progress);
                            Console.Write(new string('\b', progress.Length));
                        }
                    }

                    chartGen.AddData(e, testTotalError / validationSamples, (int)Track.TestError);
                    chartGen.AddData(e, (double)testHits / validationSamples, (int)Track.TestAccuracy);
                }

                if (verbose)
                {
                    string s = $" - loss: {Math.Round(trainTotalError / trainingSamples, 6)}";
                    if (trackFlags.HasFlag(Track.TrainAccuracy))
                        s += $" - acc: {Math.Round((double)trainHits / trainingSamples * 100, 4)}%";
                    s += $" - eta: {trainTimer.Elapsed}";
                    LogLine(s);
                }

                chartGen.Save();
                File.WriteAllLines($"{outFilename}_log.txt", LogLines);
            }
        }

        // This is vectorized gradient descent
        private void GradientDescentStep(Data trainingData, int samplesInTrainingData, AccuracyFunc accuracyFunc, ref double trainError, ref int trainHits)
        {
            var lastLayer = Layers.Last();

            FeedForward(trainingData.Input);
            Tensor loss = new Tensor(lastLayer.Output.Shape);
            Error.Compute(trainingData.Output, lastLayer.Output, loss);
            trainError += loss.Sum() / lastLayer.OutputShape.Length;
            trainHits += accuracyFunc(trainingData.Output, lastLayer.Output);
            Error.Derivative(trainingData.Output, lastLayer.Output, loss);
            BackProp(loss);
            UpdateParameters(samplesInTrainingData);
        }

        private void LogLine(string text)
        {
            LogLines.Add(text);
            Console.WriteLine(text);
        }

        public string Summary()
        {
            int totalParams = 0;
            string output = "_________________________________________________________________\n";
            output += "Layer Type                   Output Shape              Param #\n";
            output += "=================================================================\n";

            foreach (var layer in Layers)
            {
                totalParams += layer.GetParamsNum();
                output += $"{layer.GetType().Name.PadRight(29)}"+ $"({layer.OutputShape.Width}, {layer.OutputShape.Height}, {layer.OutputShape.Depth})".PadRight(26) + $"{layer.GetParamsNum()}\n";
                output += "_________________________________________________________________\n";
            }

            output += $"Total params: {totalParams}";

            return output;
        }

        public void SaveStateXml(string filename = "")
        {
            XmlDocument doc = new XmlDocument();
            XmlElement modelElem = doc.CreateElement("Sequential");

            for (int l = 0; l < Layers.Count; l++)
            {
                XmlElement layerElem = doc.CreateElement(Layers[l].GetType().Name);
                Layers[l].SerializeParameters(layerElem);
                modelElem.AppendChild(layerElem);
            }

            doc.AppendChild(modelElem);
            doc.Save(filename.Length == 0 ? $"{FilePrefix}.xml" : filename);
        }

        public void LoadStateXml(string filename = "")
        {
            XmlDocument doc = new XmlDocument();
            doc.Load(filename.Length == 0 ? $"{FilePrefix}.xml" : filename);
            XmlElement modelElem = doc.FirstChild as XmlElement;

            for (int l = 0; l < Layers.Count; l++)
            {
                XmlElement layerElem = modelElem.ChildNodes.Item(l) as XmlElement;
                Layers[l].DeserializeParameters(layerElem);
            }
        }

        public static bool DebugMode = false;
        private delegate int AccuracyFunc(Tensor targetOutput, Tensor output);
        private LossFunc Error = Loss.MeanSquareError;
        private Optimizers.OptimizerBase Optimizer;
        private int Seed;
        private List<string> LogLines = new List<string>();
    }
}
