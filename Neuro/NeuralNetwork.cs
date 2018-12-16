using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Xml;
using Neuro.Layers;
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

        public NeuralNetwork Clone()
        {
            var clone = new NeuralNetwork(Name, Seed);
            clone.Model = Model.Clone();
            clone.Optimizer = Optimizer;
            clone.LossFuncs = LossFuncs;
            return clone;
        }

        //public void CopyParametersTo(NeuralNetwork target)
        //{
        //    for (int i = 0; i < Layers.Count; ++i)
        //        Layers[i].CopyParametersTo(target.Layers[i]);
        //}

        //// Tau specifies the percentage of copied parameters to be applied on a target network, when less than 1 target's network
        //// parameters will be updated as follows: this_parameters * tau + target_parameters * (1 - tau)
        //public void SoftCopyParametersTo(NeuralNetwork target, float tau)
        //{
        //    if (tau > 1 || tau <= 0) throw new Exception("Tau has to be a value from range (0, 1>.");
        //    for (int i = 0; i < Layers.Count; ++i)
        //        Layers[i].CopyParametersTo(target.Layers[i], tau);
        //}

        public string Name;

        public string FilePrefix
        {
            get { return Name.ToLower().Replace(" ", "_"); }
        }

        public Tensor[] Predict(Tensor[] inputs)
        {
            Model.FeedForward(inputs);
            return Model.GetOutputs();
        }

        private void FeedForward(Tensor[] inputs)
        {
            Model.FeedForward(inputs);
        }

        // There is single entry in deltas for every output layer of this network
        private void BackProp(Tensor[] deltas)
        {
            Model.BackProp(deltas);
        }

        public List<ParametersAndGradients> GetParametersAndGradients()
        {
            return Model.GetParametersAndGradients();
        }
        
        public void Optimize(Optimizers.OptimizerBase optimizer, LossFunc loss)
        {
            Optimizer = optimizer;
            Model.Optimize();

            LossFuncs = new LossFunc[Model.GetOutputLayersCount()];
            for (int i = 0; i < LossFuncs.Length; ++i)
                LossFuncs[i] = loss;
        }

        public void Optimize(Optimizers.OptimizerBase optimizer, Dictionary<string, LossFunc> losses)
        {
            Optimizer = optimizer;
            Model.Optimize();

            LossFuncs = new LossFunc[Model.GetOutputLayersCount()];
            int i = 0;
            foreach (var outLayer in Model.GetOutputLayers())
                LossFuncs[i++] = losses[outLayer.Name];
        }

        public void Fit(Tensor[] inputs, Tensor[] outputs, int batchSize = -1, int epochs = 1, int verbose = 1, Track trackFlags = Track.TrainError | Track.TestAccuracy, bool shuffle = true)
        {
            if (inputs[0].BatchSize != outputs[0].BatchSize) throw new Exception($"Mismatched input and output batch size.");

            List<Data> trainingData = new List<Data>();

            if (batchSize > 0 && batchSize != inputs[0].BatchSize)
            {
                // we have to split input and output tensors into datas so they can be shuffled later on
                for (int i = 0; i < inputs[0].BatchSize; ++i)
                    trainingData.Add(new Data() { Inputs = inputs.Select(x => x.GetBatch(i)).ToArray(), Outputs = outputs.Select(x => x.GetBatch(i)).ToArray() });
            }
            else
                trainingData.Add(new Data() { Inputs = inputs, Outputs = outputs });

            Fit(trainingData, batchSize, epochs, null, verbose, trackFlags, shuffle);
        }

        public void Fit(List<Tensor[]> inputs, List<Tensor[]> outputs, int batchSize = -1, int epochs = 1, int verbose = 1, Track trackFlags = Track.TrainError | Track.TestAccuracy, bool shuffle = true)
        {
            if (inputs.Count != outputs.Count) throw new Exception($"Mismatched number of inputs and outputs.");

            List<Data> trainingData = new List<Data>();
            for (int i = 0; i < inputs.Count; ++i)
            {
                if (inputs[i][0].BatchSize != 1 && inputs.Count > 1) throw new Exception($"Input tensor at index {i} has multiple batches in it, this is not supported!");
                if (outputs[i][0].BatchSize != 1 && outputs.Count > 1) throw new Exception($"Output tensor at index {i} has multiple batches in it, this is not supported!");
                trainingData.Add(new Data() { Inputs = inputs[i], Outputs = outputs[i] });
            }

            Fit(trainingData, batchSize, epochs, null, verbose, trackFlags, shuffle);
        }

        // Training method, when batch size is -1 the whole training set is used for single gradient descent step (in other words, batch size equals to training set size)
        public void Fit(List<Data> trainingData, int batchSize = -1, int epochs = 1, List<Data> validationData = null, int verbose = 1, Track trackFlags = Track.TrainError | Track.TestAccuracy, bool shuffle = true)
        {
            int inputsBatchSize = trainingData[0].Inputs[0].BatchSize;
            bool trainingDataAlreadyBatched = inputsBatchSize > 1;

            for (int i = 0; i < trainingData.Count; ++i)
            {
                Data d = trainingData[i];
                //Debug.Assert(d.Inputs.BatchSize == d.Outputs.BatchSize, $"Training data set contains mismatched number if input and output batches for data at index {i}!");
                //Debug.Assert(d.Inputs.BatchSize == trainingData[0].Inputs.BatchSize, "Training data set contains batches of different size!");
            }

            if (batchSize < 0)
                batchSize = trainingDataAlreadyBatched ? trainingData[0].Inputs[0].BatchSize : trainingData.Count;

            string outFilename = $"{FilePrefix}_training_data_{Optimizer.GetType().Name.ToLower()}_b{batchSize}{(Seed > 0 ? ("_seed" + Seed) : "")}_{Tensor.CurrentOpMode}";
            ChartGenerator chartGen = null;
            if (trackFlags != Track.Nothing)
                chartGen = new ChartGenerator($"{outFilename}", $"{Name}\nloss=[{string.Join(",", LossFuncs.Select(x => x.GetType().Name))}] optimizer={Optimizer} batch_size={batchSize}\nseed={(Seed > 0 ? Seed.ToString() : "None")} tensor_mode={Tensor.CurrentOpMode}", "Epoch");

            if (trackFlags.HasFlag(Track.TrainError))
                chartGen.AddSeries((int)Track.TrainError, "Error on train data\n(left Y axis)", Color.DarkRed);
            if (trackFlags.HasFlag(Track.TestError))
                chartGen.AddSeries((int)Track.TestError, "Error on test data\n(left Y axis)", Color.IndianRed);
            if (trackFlags.HasFlag(Track.TrainAccuracy))
                chartGen.AddSeries((int)Track.TrainAccuracy, "Accuracy on train data\n(right Y axis)", Color.DarkBlue, true);
            if (trackFlags.HasFlag(Track.TestAccuracy))
                chartGen.AddSeries((int)Track.TestAccuracy, "Accuracy on test\n(right Y axis)", Color.CornflowerBlue, true);

            //var lastLayer = Layers.Last();
            //int outputsNum = lastLayer.OutputShape.Length;

            int batchesNum = trainingDataAlreadyBatched ? trainingData.Count : (trainingData.Count / batchSize);
            int totalTrainingSamples = trainingData.Count * inputsBatchSize;

            AccuracyFunc accuracyFunc = Tools.AccNone;

            //if (trackFlags.HasFlag(Track.TrainAccuracy) || trackFlags.HasFlag(Track.TestAccuracy))
            //{
            //    if (outputsNum == 1)
            //        accuracyFunc = Tools.AccBinaryClassificationEquality;
            //    else
            //        accuracyFunc = Tools.AccCategoricalClassificationEquality;
            //}

            Stopwatch trainTimer = new Stopwatch();

            for (int e = 1; e <= epochs; ++e)
            {
                string output;

                if (verbose > 0)
                    LogLine($"Epoch {e}/{epochs}");

                // no point shuffling stuff when we have single batch
                if (batchesNum > 1 && shuffle)
                    trainingData.Shuffle();

                List<Data> batchedTrainingData = trainingDataAlreadyBatched ? trainingData : Tools.MergeData(trainingData, batchSize);

                float trainTotalError = 0;
                int trainHits = 0;

                trainTimer.Restart();

                for (int b = 0; b < batchedTrainingData.Count; ++b)
                {
                    // this will be equal to batch size; however, the last batch size may be different if there is a reminder of training data by batch size division
                    int samples = batchedTrainingData[b].Inputs[0].BatchSize;
                    GradientDescentStep(batchedTrainingData[b], samples, accuracyFunc, ref trainTotalError, ref trainHits);

                    if (verbose == 2)
                    {
                        output = Tools.GetProgressString(b * batchSize + samples, totalTrainingSamples);
                        Console.Write(output);
                        Console.Write(new string('\b', output.Length));
                    }
                }

                trainTimer.Stop();

                if (verbose == 2)
                {
                    output = Tools.GetProgressString(totalTrainingSamples, totalTrainingSamples);
                    LogLine(output);
                }

                float trainError = trainTotalError / totalTrainingSamples;

                chartGen?.AddData(e, trainError, (int)Track.TrainError);
                chartGen?.AddData(e, (float)trainHits / totalTrainingSamples, (int)Track.TrainAccuracy);

                if (verbose > 0)
                {
                    string s = $" - loss: {Math.Round(trainError, 4)}";
                    if (trackFlags.HasFlag(Track.TrainAccuracy))
                        s += $" - acc: {Math.Round((float)trainHits / totalTrainingSamples * 100, 4)}%";
                    s += " - eta: " + trainTimer.Elapsed.ToString(@"mm\:ss\.ffff");

                    LogLine(s);
                }

                float testTotalError = 0;

                if (validationData != null)
                {
                    int validationSamples = validationData.Count * validationData[0].Inputs[0].BatchSize;
                    float testHits = 0;

                    for (int n = 0; n < validationData.Count; ++n)
                    {
                        FeedForward(validationData[n].Inputs);
                        var outputs = Model.GetOutputs();
                        Tensor[] losses = new Tensor[outputs.Length];
                        for (int i = 0; i < outputs.Length; ++i)
                        {
                            LossFuncs[i].Compute(validationData[n].Outputs[i], outputs[i], losses[i]);
                            testTotalError += losses[i].Sum() / outputs[i].BatchLength;
                            testHits += accuracyFunc(validationData[n].Outputs[i], outputs[i]);
                        }

                        if (verbose == 2)
                        {
                            string progress = " - validating: " + Math.Round(n / (float)validationData.Count * 100) + "%";
                            Console.Write(progress);
                            Console.Write(new string('\b', progress.Length));
                        }
                    }

                    chartGen?.AddData(e, testTotalError / validationSamples, (int)Track.TestError);
                    chartGen?.AddData(e, (float)testHits / validationSamples, (int)Track.TestAccuracy);
                }

                if ((ChartSaveInterval > 0 && (e % ChartSaveInterval == 0)) || e == epochs)
                    chartGen?.Save();
            }

            if (verbose > 0)
                File.WriteAllLines($"{outFilename}_log.txt", LogLines);
        }

        // This is vectorized gradient descent
        private void GradientDescentStep(Data trainingData, int samplesInTrainingData, AccuracyFunc accuracyFunc, ref float trainError, ref int trainHits)
        {
            FeedForward(trainingData.Inputs);
            var outputs = Model.GetOutputs();
            Tensor[] losses = new Tensor[outputs.Length];
            for (int i = 0; i < outputs.Length; ++i)
            {
                LossFuncs[i].Compute(trainingData.Outputs[i], outputs[i], losses[i]);
                trainError += losses[i].Sum() / outputs[i].BatchLength;
                trainHits += accuracyFunc(trainingData.Outputs[i], outputs[i]);
                LossFuncs[i].Derivative(trainingData.Outputs[i], outputs[i], losses[i]);
            }
            BackProp(losses);
            Optimizer.Step(GetParametersAndGradients(), samplesInTrainingData);
        }

        private void LogLine(string text)
        {
            LogLines.Add(text);
            Console.WriteLine(text);
        }

        public string Summary()
        {
            return Model.Summary();
        }

        public void SaveStateXml(string filename = "")
        {
            Model.SaveStateXml(filename);
        }

        public void LoadStateXml(string filename = "")
        {
            Model.LoadStateXml(filename);
        }

        public int ChartSaveInterval = 20;
        public static bool DebugMode = false;        
        private LossFunc[] LossFuncs;
        private Optimizers.OptimizerBase Optimizer;
        public Models.ModelBase Model;
        private int Seed;
        private delegate int AccuracyFunc(Tensor targetOutput, Tensor output);
        private List<string> LogLines = new List<string>();
    }

    public class ParametersAndGradients
    {
        public Tensor Parameters;
        public Tensor Gradients;
    }
}
