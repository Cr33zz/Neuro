﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Neuro.Tensors;

namespace Neuro.Layers
{
    public class Merge : LayerBase
    {
        public enum Mode
        {
            Sum,
            Avg,
            Max,
            Min
        }

        public Merge(LayerBase[] inputLayers, Mode mergeMode)
            : base(inputLayers, inputLayers[0].OutputShape)
        {
            MergeMode = mergeMode;
        }

        public Merge(Shape[] inputShapes, Mode mergeMode)
            : base(inputShapes, inputShapes[0])
        {
            MergeMode = mergeMode;
        }

        public override LayerBase Clone()
        {
            return new Merge(InputShapes, MergeMode);
        }

        protected override void FeedForwardInternal()
        {
            switch (MergeMode)
            {
                case Mode.Avg:
                    Tensor.MergeAvg(Inputs, Output);
                    break;
                case Mode.Max:
                    Tensor.MergeMax(Inputs, Output);
                    break;
                case Mode.Min:
                    Tensor.MergeMin(Inputs, Output);
                    break;
                case Mode.Sum:
                    Tensor.MergeSum(Inputs, Output);
                    break;
            }
        }

        protected override void BackPropInternal(Tensor outputGradient)
        {
            switch (MergeMode)
            {
                case Mode.Avg:
                    Tensor.MergeAvgGradient(Output, Inputs, outputGradient, InputsGradient);
                    break;
                case Mode.Max:
                case Mode.Min:
                    Tensor.MergeMinMaxGradient(Output, Inputs, outputGradient, InputsGradient);
                    break;
                case Mode.Sum:
                    Tensor.MergeSumGradient(Output, Inputs, outputGradient, InputsGradient);
                    break;
            }
        }

        public readonly Mode MergeMode;
    }
}
