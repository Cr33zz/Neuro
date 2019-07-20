using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.IO;
using Neuro.Models;
using Tensorflow;

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

        //public NeuralNetwork Clone()
        //{
        //    var clone = new NeuralNetwork(Name, Seed);
        //    clone.Model = Model.Clone();
        //    clone.Optimizer = Optimizer;
        //    clone.LossFuncs = LossFuncs;
        //    return clone;
        //}

        //public void ForceInitLayers()
        //{
        //    foreach (var layer in Model.GetLayers())
        //        layer.Init();
        //}

        public void CopyParametersTo(NeuralNetwork target)
        {
            //foreach (var layersPair in Model.GetLayers().Zip(target.Model.GetLayers(), (l1, l2) => new [] {l1, l2}))
            //    layersPair[0].CopyParametersTo(layersPair[1]);
        }

        // Tau specifies the percentage of copied parameters to be applied on a target network, when less than 1 target's network
        // parameters will be updated as follows: this_parameters * tau + target_parameters * (1 - tau)
        public void SoftCopyParametersTo(NeuralNetwork target, float tau)
        {
            //if (tau > 1 || tau <= 0) throw new Exception("Tau has to be a value from range (0, 1>.");
            //foreach (var layersPair in Model.GetLayers().Zip(target.Model.GetLayers(), (l1, l2) => new[] { l1, l2 }))
            //    layersPair[0].CopyParametersTo(layersPair[1], tau);
        }

        public string Name;

        public string FilePrefix
        {
            get { return Name.ToLower().Replace(" ", "_"); }
        }

        public List<Array> Predict(List<Array> inputs)
        {
            return Model.Predict(inputs);
        }

        public List<Array> Predict(Array input)
        {
            return Predict(new List<Array>{ input });
        }

        public void Optimize(Optimizers.OptimizerBase optimizer, Loss loss)
        {
            Optimizer = optimizer;

            Losses = new Loss[Model.GetOutputLayersCount()];
            for (int i = 0; i < Losses.Length; ++i)
                Losses[i] = loss;

            Model.Optimize(Optimizer, Losses);
        }

        public void Optimize(Optimizers.OptimizerBase optimizer, Dictionary<string, Loss> lossDict)
        {
            Optimizer = optimizer;

#if VALIDATION_ENABLED
            if (lossDict.Count != Model.GetOutputLayersCount()) throw new Exception($"Mismatched number of loss functions ({lossDict.Count}) and output layers ({Model.GetOutputLayersCount()})!");
#endif

            Losses = new Loss[Model.GetOutputLayersCount()];
            int i = 0;
            foreach (var outLayer in Model.GetOutputLayers())
            {
                Losses[i++] = lossDict[outLayer.Name];
            }

            Model.Optimize(Optimizer, Losses);
        }
        
        public void Fit(Array input, Array output, int batchSize = -1, int epochs = 1, int verbose = 1, Track trackFlags = Track.TrainError | Track.TestAccuracy, bool shuffle = true)
        {
            Fit(new List<Array>{ input }, new List<Array>{ output}, batchSize, epochs, null, verbose, trackFlags, shuffle);
        }

        // Training method, when batch size is -1 the whole training set is used for single gradient descent step (in other words, batch size equals to training set size)
        public void Fit(List<Array> inputs, List<Array> outputs, int batchSize = -1, int epochs = 1, List<Array> validationData = null, int verbose = 1, Track trackFlags = Track.TrainError | Track.TestAccuracy, bool shuffle = true)
        {
            int samplesNum = inputs[0].GetLength(0);

            if (batchSize < 0)
                batchSize = samplesNum;

            string outFilename = $"{FilePrefix}_training_data_{Optimizer.GetType().Name.ToLower()}_b{batchSize}{(Seed > 0 ? ("_seed" + Seed) : "")}";
            ChartGenerator chartGen = null;
            if (trackFlags != Track.Nothing)
                chartGen = new ChartGenerator($"{outFilename}", $"{Name}\nloss=[{string.Join(",", Losses.Select(x => x.GetType().Name))}] optimizer={Optimizer} batch_size={batchSize}\nseed={(Seed > 0 ? Seed.ToString() : "None")}", "Epoch");

            if (trackFlags.HasFlag(Track.TrainError))
                chartGen.AddSeries((int)Track.TrainError, "Error on train data\n(left Y axis)", Color.DarkRed);
            if (trackFlags.HasFlag(Track.TestError))
                chartGen.AddSeries((int)Track.TestError, "Error on test data\n(left Y axis)", Color.IndianRed);
            if (trackFlags.HasFlag(Track.TrainAccuracy))
                chartGen.AddSeries((int)Track.TrainAccuracy, "Accuracy on train data\n(right Y axis)", Color.DarkBlue, true);
            if (trackFlags.HasFlag(Track.TestAccuracy))
                chartGen.AddSeries((int)Track.TestAccuracy, "Accuracy on test\n(right Y axis)", Color.CornflowerBlue, true);

            //            if (AccuracyFuncs == null && (trackFlags.HasFlag(Track.TrainAccuracy) || trackFlags.HasFlag(Track.TestAccuracy)))
            //            {
            //                AccuracyFuncs = new AccuracyFunc[outputLayersCount];

            //                for (int i = 0; i < outputLayersCount; ++i)
            //                {
            //                    if (Model.GetOutputLayers().ElementAt(i).OutputShape.Length == 1)
            //                        AccuracyFuncs[i] = Tools.AccBinaryClassificationEquality;
            //                    else
            //                        AccuracyFuncs[i] = Tools.AccCategoricalClassificationEquality;
            //                }
            //            }

            Stopwatch trainTimer = new Stopwatch();

            for (int e = 1; e <= epochs; ++e)
            {
                string output;

                if (verbose > 0)
                    LogLine($"Epoch {e}/{epochs}");

                int[] indices = Enumerable.Range(0, samplesNum).ToArray();

                // no point shuffling stuff when we have single batch
                if (samplesNum > 1 && shuffle)
                    indices.Shuffle();

                int batchesNum = (int)Math.Ceiling(samplesNum / (double)batchSize);
                var batchesIndices = Enumerable.Range(0, batchesNum).Select(i => (i * batchSize, Math.Min((i + 1) * batchSize, samplesNum))).ToList();

                float trainTotalError = 0;
                int trainHits = 0;

                trainTimer.Restart();

                for (int b = 0; b < batchesNum; ++b)
                {
                    var (batchStart, batchEnd) = batchesIndices[b];
                    int[] batchIndices = indices.Get(batchStart, batchEnd);

                    List<Array> inputsBatch = GenerateBatch(inputs, batchIndices);
                    List<Array> outputsBatch = GenerateBatch(outputs, batchIndices);

                    TrainStep(inputsBatch, outputsBatch, ref trainTotalError, ref trainHits);

                    if (verbose == 2)
                    {
                        output = Tools.GetProgressString(b * batchSize + inputsBatch[0].GetLength(0), samplesNum);
                        Console.Write(output);
                        Console.Write(new string('\b', output.Length));
                    }
                }

                trainTimer.Stop();

                if (verbose == 2)
                {
                    output = Tools.GetProgressString(samplesNum, samplesNum);
                    LogLine(output);
                }

                float trainError = trainTotalError / samplesNum;

                chartGen?.AddData(e, trainError, (int)Track.TrainError);
                chartGen?.AddData(e, (float)trainHits / samplesNum / Model.GetOutputLayersCount(), (int)Track.TrainAccuracy);

                if (verbose > 0)
                {
                    string s = $" - loss: {Math.Round(trainError, 4)}";
                    if (trackFlags.HasFlag(Track.TrainAccuracy))
                        s += $" - acc: {Math.Round((float)trainHits / samplesNum * 100, 4)}%";
                    s += " - eta: " + trainTimer.Elapsed.ToString(@"mm\:ss\.ffff");

                    LogLine(s);
                }

                //float testTotalError = 0;

                //if (validationData != null)
                //{
                //    int validationSamples = validationData.Count * validationData[0].Inputs[0].BatchSize;
                //    float testHits = 0;

                //    for (int n = 0; n < validationData.Count; ++n)
                //    {
                //        FeedForward(validationData[n].Inputs);
                //        var outputs = Model.GetOutputs();
                //        Tensorflow.Tensor[] losses = new Tensorflow.Tensor[outputs.Length];
                //        for (int i = 0; i < outputLayersCount; ++i)
                //        {
                //            LossFuncs[i].Compute(validationData[n].Outputs[i], outputs[i], losses[i]);
                //            testTotalError += losses[i].Sum() / outputs[i].BatchLength;
                //            testHits += AccuracyFuncs[i](validationData[n].Outputs[i], outputs[i]);
                //        }

                //        if (verbose == 2)
                //        {
                //            string progress = " - validating: " + Math.Round(n / (float)validationData.Count * 100) + "%";
                //            Console.Write(progress);
                //            Console.Write(new string('\b', progress.Length));
                //        }
                //    }

                //    chartGen?.AddData(e, testTotalError / validationSamples, (int)Track.TestError);
                //    chartGen?.AddData(e, (float)testHits / validationSamples / outputLayersCount, (int)Track.TestAccuracy);
                //}

                if ((ChartSaveInterval > 0 && (e % ChartSaveInterval == 0)) || e == epochs)
                    chartGen?.Save();
            }

            if (verbose > 0)
                File.WriteAllLines($"{outFilename}_log.txt", LogLines);
        }

        // This is vectorized gradient descent
        private void TrainStep(List<Array> inputs, List<Array> outputs, ref float trainError, ref int trainHits)
        {
            var results = Model.Trainer.Train(inputs, outputs);
            trainError += results[Model.Metrics["loss"].Item2].GetValue<float>() * inputs[0].GetLength(0); //returned loss is mean and we care about the sum so we need to multiply that by number of batches
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

        private List<Array> GenerateBatch(List<Array> inputs, int[] batchIndices)
        {
            var result = new List<Array>();

            for (int i = 0; i < inputs.Count; ++i)
            {
                result.Add(inputs[i].GetEx(0, batchIndices));
            }

            return result;
        }

        //public void SaveStateXml(string filename = "")
        //{
        //    Model.SaveStateXml(filename.Length == 0 ? $"{FilePrefix}.xml" : filename);
        //}

        //public void LoadStateXml(string filename = "")
        //{
        //    Model.LoadStateXml(filename.Length == 0 ? $"{FilePrefix}.xml" : filename);
        //}

        public int ChartSaveInterval = 20;
        public static bool DebugMode = false;
        private Loss[] Losses;
        private Optimizers.OptimizerBase Optimizer;
        private int Seed;
        private List<string> LogLines = new List<string>();
        public Flow Model;
    }
}
