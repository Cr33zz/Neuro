using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Xml;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace Neuro.Tensors
{
    public class Tensor
    {
        public enum OpMode
        {
            CPU,
            MultiCPU,
            GPU
        }

        public enum Location
        {
            Host,
            Device
        }

        static Tensor()
        {
            SetOpMode(OpMode.MultiCPU);
        }

        public Tensor(Shape shape)
        {
            Shape = shape;
            Values = new float[shape.Length];
        }

        public Tensor(float[] values, Shape shape)
        {
            Debug.Assert(values.Length == shape.Length, $"Invalid array size {values.Length}. Expected {shape.Length}.");
            Shape = shape;
            Values = (float[])values.Clone();
        }

        public Tensor(Tensor t)
        {
            t.CopyToHost();
            Shape = Shape.From(t.Shape.Dimensions);
            Values = (float[])t.Values.Clone();
        }

        public Tensor(string bmpFile, bool grayScale)
        {
            using (var bmp = new Bitmap(bmpFile))
            {
                Shape = new Shape(bmp.Width, bmp.Height, grayScale ? 1 : 3);
                Values = new float[Shape.Length];

                for (int h = 0; h < bmp.Height; ++h)
                {
                    for (int w = 0; w < bmp.Width; ++w)
                    {
                        Color c = bmp.GetPixel(w, h);
                        if (grayScale)
                            Set((c.R * 0.3f + c.G * 0.59f + c.B * 0.11f) / 255.0f, w, h);
                        else
                        {
                            Set(c.R / 255.0f, w, h, 0);
                            Set(c.G / 255.0f, w, h, 1);
                            Set(c.B / 255.0f, w, h, 2);
                        }
                    }
                }
            }
        }

        public static void SetOpMode(OpMode mode)
        {
            CurrentOpMode = mode;

            switch (mode)
            {
            case OpMode.CPU:
                Op = new TensorOpCpu();
                return;
            case OpMode.MultiCPU:
                Op = new TensorOpMultiCpu();
                return;
            case OpMode.GPU:
                Op = new TensorOpGpu();
                return;
            }
        }

        public Tensor Clone()
        {
            return new Tensor(this);
        }

        public static OpMode CurrentOpMode { get; private set; }

        public int Width => Shape.Width;

        public int Height => Shape.Height;

        public int Depth => Shape.Depth;

        public int BatchSize => Shape.BatchSize;

        public int BatchLength => Shape.Width * Shape.Height * Shape.Depth;

        public int Length => Values.Length;

        public float[] GetValues()
        {
            CopyToHost();
            return (float[])Values.Clone();
        }

        public Bitmap ToBitmap()
        {
            Debug.Assert(BatchSize == 1);

            Bitmap output = new Bitmap(Width, Height);
            bool grayScale = (Depth == 1);

            for (int d = 0; d < Depth; ++d)
            for (int h = 0; h < Height; ++h)
            for (int w = 0; w < Width; ++w)
                output.SetPixel(w, h, grayScale ? Color.FromArgb((int)(Get(w, h) * 255), (int)(Get(w, h) * 255), (int)(Get(w, h) * 255))
                                                : Color.FromArgb((int)(Get(w, h) * 255), (int)(Get(w, h, 1) * 255), (int)(Get(w, h, 2) * 255)));

            return output;
        }

        public void FillWithRand(int seed = -1, float min = -1, float max = 1)
        {
            CurrentLocation = Location.Host;
            Random rng = seed > 0 ? new Random(seed) : Rng;
            
            for (int i = 0; i < Values.Length; ++i)
                Values[i] = min + (max - min) * (float)rng.NextDouble();
        }

        public void FillWithRange(float start = 0, float increment = 1)
        {
            CurrentLocation = Location.Host;
            for (int i = 0; i < Values.Length; ++i)
                Values[i] = start + i * increment;
        }

        public void FillWithValue(float value)
        {
            CurrentLocation = Location.Host;
            for (int i = 0; i < Values.Length; ++i)
                Values[i] = value;
        }

        public void Zero()
        {
            CurrentLocation = Location.Host;
            Array.Clear(Values, 0, Values.Length);
        }

        private void Mul(bool transposeT, Tensor t, Tensor result)
        {
            Debug.Assert((!transposeT && Width == t.Height) || (transposeT && Width == t.Width));
            Debug.Assert(t.Depth == Depth);

            Op.Mul(false, transposeT, this, t, result);
        }

        private Tensor Mul(bool transposeT, Tensor t)
        {
            Tensor result = new Tensor(new Shape(transposeT ? t.Shape.Height : t.Shape.Width, Height, Depth, BatchSize));
            Mul(transposeT, t, result);
            return result;
        }

        public void Mul(Tensor t, Tensor result)
        {
            Mul(false, t, result);
        }

        public Tensor Mul(Tensor t)
        {
            return Mul(false, t);
        }

        // Element-wise multiplication
        public virtual void MulElem(Tensor t, Tensor result)
        {
            Debug.Assert(SameDimensionsExceptBatches(t));
            Debug.Assert(t.BatchSize == result.BatchSize);

            Op.MulElem(this, t, result);
        }

        public Tensor MulElem(Tensor t)
        {
            Tensor result = new Tensor(Shape);
            MulElem(t, result);
            return result;
        }

        public virtual void Mul(float v, Tensor result)
        {
            CopyToHost();
            result.CurrentLocation = Tensor.Location.Host;

            for (int i = 0; i < Values.Length; ++i)
                result.Values[i] = Values[i] * v;
        }

        public Tensor Mul(float v)
        {
            Tensor result = new Tensor(Shape);
            Mul(v, result);
            return result;
        }

        public virtual void Div(Tensor t, Tensor result)
        {
            CopyToHost();
            result.CurrentLocation = Tensor.Location.Host;

            Debug.Assert(SameDimensionsExceptBatches(t));
            Debug.Assert(t.BatchSize == result.BatchSize);

            for (int i = 0; i < Values.Length; ++i)
                result.Values[i] = Values[i] / t.Values[i];
        }

        public Tensor Div(Tensor t)
        {
            Tensor result = new Tensor(Shape);
            Div(t, result);
            return result;
        }

        public virtual void Div(float v, Tensor result)
        {
            CopyToHost();
            result.CurrentLocation = Tensor.Location.Host;

            for (int i = 0; i < Values.Length; ++i)
                result.Values[i] = Values[i] / v;
        }

        public Tensor Div(float v)
        {
            Tensor result = new Tensor(Shape);
            Div(v, result);
            return result;
        }

        public virtual void Add(float alpha, float beta, Tensor t, Tensor result)
        {
            Debug.Assert(SameDimensionsExceptBatches(t));
            Debug.Assert(t.BatchSize == result.BatchSize || t.BatchSize == 1);

            Op.Add(alpha, this, beta, t, result);
        }

        public virtual void Add(Tensor t, Tensor result)
        {
            Add(1, 1, t, result);
        }

        public Tensor Add(Tensor t)
        {
            Tensor result = new Tensor(Shape);
            Add(t, result);
            return result;
        }

        public Tensor Add(float alpha, float beta, Tensor t)
        {
            Tensor result = new Tensor(Shape);
            Add(alpha, beta, t, result);
            return result;
        }

        public virtual void Add(float v, Tensor result)
        {
            CopyToHost();
            for (int i = 0; i < Values.Length; ++i)
                result.Values[i] = Values[i] + v;
        }

        public Tensor Add(float v)
        {
            Tensor result = new Tensor(Shape);
            Add(v, result);
            return result;
        }

        public virtual void Sub(Tensor t, Tensor result)
        {
            Debug.Assert(SameDimensionsExceptBatches(t));
            Debug.Assert(t.BatchSize == result.BatchSize || t.BatchSize == 1);

            Op.Sub(this, t, result);
        }

        public Tensor Sub(Tensor t)
        {
            Tensor result = new Tensor(Shape);
            Sub(t, result);
            return result;
        }

        public virtual void Sub(float v, Tensor result)
        {
            CopyToHost();
            for (int i = 0; i < Values.Length; ++i)
                result.Values[i] = Values[i] - v;
        }

        public Tensor Sub(float v)
        {
            Tensor result = new Tensor(Shape);
            Sub(v, result);
            return result;
        }

        public void Negated(Tensor result)
        {
            for (int i = 0; i < Values.Length; ++i)
                result.Values[i] = -Values[i];
        }

        public Tensor Negated()
        {
            Tensor result = new Tensor(Shape);
            Negated(result);
            return result;
        }

        public void Clipped(float min, float max, Tensor result)
        {
            Map(x => Tools.Clip(x, min, max), result);
        }

        public Tensor Clipped(float min, float max)
        {
            Tensor result = new Tensor(Shape);
            Clipped(min, max, result);
            return result;
        }

        public Tensor DiagFlat()
        {
            Tensor result = new Tensor(new Shape(BatchLength, BatchLength, 1, BatchSize));

            int batchLen = BatchLength;

            for (int b = 0; b < BatchSize; ++b)
            for (int i = 0; i < batchLen; ++i)
                result[i, i, 0, b] = Values[b * batchLen + i];

            return result;
        }

        public void Map(Func<float, float> func, Tensor result)
        {
            Op.Map(func, this, result);
        }

        public Tensor Map(Func<float, float> func)
        {
            Tensor result = new Tensor(Shape);
            Map(func, result);
            return result;
        }

        public void Map(Func<float, float, float> func, Tensor other, Tensor result)
        {
            Op.Map(func, this, other, result);
        }

        public Tensor Map(Func<float, float, float> func, Tensor other)
        {
            Tensor result = new Tensor(Shape);
            Map(func, other, result);
            return result;
        }

        public Tensor SumBatches()
        {
            Tensor result = new Tensor(new Shape(Shape.Width, Shape.Height, Shape.Depth, 1));
            Op.SumBatches(this, result);
            return result;
        }

        public float Sum(int batch = -1)
        {
            CopyToHost();
            if (batch < 0)
                return Values.Sum();

            int batchLen = BatchLength;
            float sum = 0;

            for (int i = 0, idx = batch * batchLen; i < batchLen; ++i, ++idx)
                sum += Values[idx];

            return sum;
        }

        public Tensor SumPerBatch()
        {
            CopyToHost();
            Tensor result = new Tensor(new Shape(1, 1, 1, Shape.BatchSize));

            for (int n = 0; n < BatchSize; ++n)
                result.Values[n] = Sum(n);

            return result;
        }


        public Tensor AvgBatches()
        {
            CopyToHost();
            Tensor result = SumBatches();

            int batchLen = BatchLength;

            for (int n = 0; n < batchLen; ++n)
                result.Values[n] /= BatchSize;

            return result;
        }

        public float Avg(int batch = -1)
        {
            return Sum(batch) / (batch < 0 ? Length : BatchLength);
        }

        public Tensor AvgPerBatch()
        {
            Tensor result = SumPerBatch();

            int batchLen = BatchLength;

            for (int n = 0; n < BatchSize; ++n)
                result.Values[n] /= batchLen;

            return result;
        }

        public float Max(int batch = -1)
        {
            int maxIndex;
            return GetMaxData(batch, out maxIndex);
        }

        public Tensor MaxPerBatch()
        {
            var result = new Tensor(new Shape(1, 1, 1, Shape.BatchSize));
            for (int n = 0; n < BatchSize; ++n)
                result[0, 0, 0, n] = Max(n);
            return result;
        }

        public static Tensor MergeIntoBatch(List<Tensor> tensors)
        {
            if (tensors.Count == 0)
                throw new Exception("List cannot be empty.");
            
            Tensor output = new Tensor(new Shape(tensors[0].Width, tensors[0].Height, tensors[0].Depth, tensors.Count));
            
            for (int n = 0; n < tensors.Count; ++n)
            {
                Tensor t = tensors[n];
                t.CopyToHost();
                Array.Copy(t.Values, 0, output.Values, t.Length * n, t.Values.Length);
            }

            return output;
        }

        // In case number of tensors is smaller than forcedDepth, first tensor will be repeated to account for missing tensors
        public static Tensor MergeIntoDepth(List<Tensor> tensors, int forcedDepth = 0)
		{
			if (tensors.Count == 0)
				throw new Exception("List cannot be empty.");

			Tensor output = new Tensor(new Shape(tensors[0].Width, tensors[0].Height, Math.Max(tensors.Count, forcedDepth)));

            Tensor t = tensors[0];
            t.CopyToHost();

            int t0_copies = forcedDepth > 0 ? forcedDepth - tensors.Count : 0;

            for (int n = 0; n < t0_copies; ++n)
            {
                Array.Copy(t.Values, 0, output.Values, t.Length * n, t.Values.Length);
            }

			for (int n = t0_copies; n < output.Depth; ++n)
			{
				t = tensors[n - t0_copies];
                t.CopyToHost();
				Array.Copy(t.Values, 0, output.Values, t.Length * n, t.Values.Length);
			}

			return output;
		}

		// This operation will concatenate elements of all input tensors separately for each batch
		public static void Concat(Tensor[] inputs, Tensor result)
        {
            for (int b = 0; b < result.BatchSize; ++b)
            {
                int elementsCopied = 0;
                for (int i = 0; i < inputs.Length; ++i)
                {
                    inputs[i].CopyToHost();
                    Array.Copy(inputs[i].Values, b * inputs[i].BatchLength, result.Values, b * result.BatchLength + elementsCopied, inputs[i].BatchLength);
                    elementsCopied += inputs[i].BatchLength;
                }
            }
        }

        // This is reverse Concat operation
        public void Split(Tensor[] outputs)
        {
            CopyToHost();
            for (int b = 0; b < BatchSize; ++b)
            {
                int elementsCopied = 0;
                for (int i = 0; i < outputs.Length; ++i)
                {
                    outputs[i].CopyToHost();
                    Array.Copy(Values, b * BatchLength + elementsCopied, outputs[i].Values, b * outputs[i].BatchLength, outputs[i].BatchLength);
                    elementsCopied += outputs[i].BatchLength;
                }
            }
        }

        public static void MergeMin(Tensor[] inputs, Tensor result)
        {
            inputs[0].CopyTo(result);
            for (int i = 1; i < inputs.Length; ++i)
            for (int j = 0; j < result.Length; ++j)
                result.Values[j] = result.Values[j] > inputs[i].Values[j] ? inputs[i].Values[j] : result.Values[j];
        }

        public static void MergeMax(Tensor[] inputs, Tensor result)
        {
            inputs[0].CopyTo(result);
            for (int i = 1; i < inputs.Length; ++i)
            for (int j = 0; j < result.Length; ++j)
                result.Values[j] = result.Values[j] < inputs[i].Values[j] ? inputs[i].Values[j] : result.Values[j];
        }

        public static void MergeSum(Tensor[] inputs, Tensor result)
        {
            result.Zero();
            for (int i = 0; i < inputs.Length; ++i)
            for (int j = 0; j < result.Length; ++j)
                result.Values[j] += inputs[i].Values[j];
        }

        public static void MergeAvg(Tensor[] inputs, Tensor result)
        {
            MergeSum(inputs, result);
            result.Div(inputs.Length, result);
        }

        public static void MergeMinMaxGradient(Tensor output, Tensor[] inputs, Tensor outputGradient, Tensor[] results)
        {
            for (int i = 0; i < inputs.Length; ++i)
            {
                results[i].Zero();
                for (int j = 0; j < output.Length; ++j)
                    results[i].Values[j] = inputs[i].Values[j] == output.Values[j] ? outputGradient.Values[j] : 0;
            }
        }

        public static void MergeSumGradient(Tensor output, Tensor[] inputs, Tensor outputGradient, Tensor[] results)
        {
            for (int i = 0; i < inputs.Length; ++i)
                outputGradient.CopyTo(results[i]);
        }

        public static void MergeAvgGradient(Tensor output, Tensor[] inputs, Tensor outputGradient, Tensor[] results)
        {
            MergeSumGradient(output, inputs, outputGradient, results);
            for (int i = 0; i < results.Length; ++i)
                results[i].Div(results.Length, results[i]);
        }

        public void Normalized(Tensor result)
        {
            float sum = Sum();
            Map(x => x / sum, result);
        }

        // ArgMax will return local index inside given batch if batch is not -1
        public int ArgMax(int batch = -1)
        {
            int maxIndex;
            GetMaxData(batch, out maxIndex);
            return maxIndex;
        }

        public Tensor ArgMaxPerBatch()
        {
            var result = new Tensor(new Shape(1, 1, 1, Shape.BatchSize));
            for (int n = 0; n < BatchSize; ++n)
                result[0, 0, 0, n] = ArgMax(n);
            return result;
        }

        public Tensor Transposed()
        {
            Tensor result = new Tensor(new Shape(Height, Width, Depth, BatchSize));
            Transpose(result);

            return result;
        }

        public void Transpose(Tensor result)
        {
            Op.Transpose(this, result);
        }

        // Generates a new tensor with given dimensions and populate it with this tensor's values in index order. 
        // One of dimensions can be -1, in that case it will be calculated based on remaining dimensions.
        public Tensor Reshaped(Shape shape)
        {
            return new Tensor(Values, Shape.Reshaped(new[] { shape.Width, shape.Height, shape.Depth, shape.BatchSize }));
        }

        public void Reshape(Shape shape)
        {
            Shape = Shape.Reshaped(new[] { shape.Width, shape.Height, shape.Depth, shape.BatchSize });
        }

        // Create new tensor with different batch length and use current tensors values to fill the new tensor.
        // Number of batches will be the same as in source tensor.
        public Tensor Resized(int width, int height = 1, int depth = 1)
        {
            int newBatchLength = width * height * depth;
            var result = new Tensor(new Shape(width, height, depth, Shape.BatchSize));
            for (int n = 0; n < BatchSize; ++n)
            for (int i = 0, idx = n * newBatchLength; i < newBatchLength; ++i, ++idx)
                result.Values[idx] = Values[n * BatchLength + i % BatchLength];
            return result;
        }

        public Tensor FlattenHoriz()
        {
            return Reshaped(Shape.Reshaped(new[] { Shape.Auto, 1, 1, Shape.BatchSize }));
        }

        public Tensor FlattenVert()
        {
            return Reshaped(Shape.Reshaped(new[] { 1, Shape.Auto, 1, Shape.BatchSize }));
        }

        public void Rotated180(Tensor result)
        {
            Debug.Assert(SameDimensionsExceptBatches(result));

            for (int n = 0; n < BatchSize; ++n)
            for (int d = 0; d < Depth; ++d)
            for (int h = Height - 1; h >= 0; --h)
            for (int w = Width - 1; w >= 0; --w)
                result.Set(Get(Width - w - 1, Height - h - 1, d, n), w, h, d, n);
        }

        public Tensor Rotated180()
        {
            Tensor result = new Tensor(Shape);
            Rotated180(result);
            return result;
        }

        public enum PaddingType
        {
            Valid, // output matrix's size will be decreased (depending on kernel size)
            Same,  // output matrix's size will be the same (except for depth) as input matrix
            Full,  // output matrix's size will be increased (depending on kernel size)
        }

        public void Conv2D(Tensor kernels, int stride, PaddingType padding, Tensor result)
        {
            Debug.Assert(Depth == kernels.Depth);

            Op.Conv2D(this, kernels, stride, padding, result);
        }

        public Tensor Conv2D(Tensor kernels, int stride, PaddingType padding)
        {
            int outputWidth = 0, outputHeight = 0, paddingX = 0, paddingY = 0;
            GetPaddingParams(padding, Width, Height, kernels.Width, kernels.Height, stride, out outputHeight, out outputWidth, out paddingX, out paddingY);

            Tensor result = new Tensor(new Shape(outputWidth, outputHeight, kernels.BatchSize, BatchSize));
            Conv2D(kernels, stride, padding, result);
            return result;
        }

        public static void Conv2DInputsGradient(Tensor gradient, Tensor kernels, int stride, PaddingType padding, Tensor inputsGradient)
        {
            inputsGradient.Zero();                        
            Op.Conv2DInputGradient(gradient, kernels, stride, padding, inputsGradient);
        }

        public static void Conv2DKernelsGradient(Tensor input, Tensor gradient, int stride, PaddingType padding, Tensor kernelsGradient)
        {
            kernelsGradient.Zero();            
            Op.Conv2DKernelsGradient(input, gradient, stride, padding, kernelsGradient);
        }

        public static void Conv2DGradient_old(Tensor input, Tensor kernels, Tensor gradient, int stride, PaddingType padding, Tensor inputGradient, Tensor kernelsGradient)
        {
            inputGradient.Zero();
            kernelsGradient.Zero();
            int outputWidth = 0, outputHeight = 0, paddingX = 0, paddingY = 0;
            GetPaddingParams(padding, input.Width, input.Height, kernels.Width, kernels.Height, stride, out outputHeight, out outputWidth, out paddingX, out paddingY);
            Op.Conv2DGradient_old(input, kernels, gradient, stride, paddingX, paddingY, inputGradient, kernelsGradient);
        }

        public enum PoolType
        {
            Max,
            Avg
        }

        public void Pool(int filterSize, int stride, PoolType type, PaddingType padding, Tensor result)
        {
            int outWidth = 0, outHeight = 0, paddingX = 0, paddingY = 0;
            GetPaddingParams(padding, Width, Height, filterSize, filterSize, stride, out outHeight, out outWidth, out paddingX, out paddingY);

            Debug.Assert(result.Width == outWidth);
            Debug.Assert(result.Height == outHeight);
            Debug.Assert(result.BatchSize == BatchSize);

            Op.Pool(this, filterSize, stride, type, paddingX, paddingY, result);
        }

        public Tensor Pool(int filterSize, int stride, PoolType type, PaddingType padding)
        {
            int outWidth = 0, outHeight = 0, paddingX = 0, paddingY = 0;
            GetPaddingParams(padding, Width, Height, filterSize, filterSize, stride, out outHeight, out outWidth, out paddingX, out paddingY);

            Tensor result = new Tensor(new Shape(outWidth, outHeight, Depth, BatchSize));
            Pool(filterSize, stride, type, padding, result);

            return result;
        }

        // Assuming result matrix is of the dimensions of input to pooling layer
        public static void PoolGradient(Tensor output, Tensor input, Tensor outputGradient, int filterSize, int stride, PoolType type, PaddingType padding, Tensor result)
        {
            Debug.Assert(output.SameDimensionsExceptBatches(outputGradient));

            int outWidth = 0, outHeight = 0, paddingX = 0, paddingY = 0;
            GetPaddingParams(padding, result.Width, result.Height, filterSize, filterSize, stride, out outHeight, out outWidth, out paddingX, out paddingY);

            Op.PoolGradient(output, input, outputGradient, filterSize, stride, type, paddingX, paddingY, result);
        }

        public override string ToString()
        {
            string s = "";
            for (int n = 0; n < BatchSize; ++n)
            {
                s += "{\n  ";
                for (int d = 0; d < Depth; ++d)
                {
                    s += "{\n    ";
                    for (int h = 0; h < Height; ++h)
                    {
                        s += "{ ";
                        for (int w = 0; w < Width; ++w)
                        {
                            s += Get(w, h, d, n) + (w == Width - 1 ? "" : ", ");
                        }
                        s += " }" + (h == Height - 1 ? "\n  " : ",\n    ");
                    }
                    s += "}" + (d == Depth - 1 ? "\n" : ",\n  ");
                }
                s += "}" + (n < BatchSize - 1 ? "\n" : "");
            }

            return s;
        }

        public bool SameDimensionsExceptBatches(Tensor t)
        {
            return Width == t.Width && Height == t.Height && Depth == t.Depth;
        }

        internal static void GetPaddingParams(PaddingType type, int width, int height, int kernelWidth, int kernelHeight, int stride, out int outHeight, out int outWidth, out int paddingX, out int paddingY)
        {
            if (type == PaddingType.Valid)
            {
                outWidth = (int)Math.Floor((width - kernelWidth) / (float)stride + 1);
                outHeight = (int)Math.Floor((height - kernelHeight) / (float)stride + 1);
                paddingX = 0;
                paddingY = 0;
            }
            else if (type == PaddingType.Same)
            {
                outWidth = width / stride;
                outHeight = height / stride;
                paddingX = (int)Math.Floor((float)kernelWidth / 2);
                paddingY = (int)Math.Floor((float)kernelHeight / 2);
            }
            else //if (type == ConvType.Full)
            {
                outWidth = (width + (kernelWidth - 1)) / stride;
                outHeight = (height + (kernelHeight - 1)) / stride;
                paddingX = kernelWidth - 1;
                paddingY = kernelHeight - 1;
            }
        }

        internal void Serialize(XmlElement parentElem, string name)
        {
            XmlElement elem = parentElem.OwnerDocument.CreateElement(name);
            XmlAttribute shapeAttrib = parentElem.OwnerDocument.CreateAttribute("shape");
            shapeAttrib.Value = string.Join(",", Shape.Dimensions);
            elem.Attributes.Append(shapeAttrib);
            elem.InnerText = string.Join(",", Values);
            //elem.InnerText = $"\n{this.ToString()}\n";
            parentElem.AppendChild(elem);
        }

        internal void Deserialize(XmlElement elem)
        {
            Shape = Shape.From(elem.GetAttribute("shape").Split(',').Select(w => int.Parse(w)).ToArray());
            Values = elem.InnerText.Split(',').Select(w => float.Parse(w)).ToArray();
        }

        public void Serialize(BinaryWriter writer)
        {
            Shape.Serialize(writer);
            writer.Write(Values.Length);
            foreach (var val in Values)
                writer.Write(val);
        }

        public static Tensor Deserialize(BinaryReader reader)
        {
            var t = new Tensor(Shape.Deserialize(reader));
            int valuesCount = reader.ReadInt32();
            t.Values = new float[valuesCount];
            for (int i = 0; i < valuesCount; ++i)
                t.Values[i] = reader.ReadSingle();
            return t;
        }

        public float GetFlat(int i)
        {
            CopyToHost();
            return Values[i];
        }

        public float Get(int w, int h = 0, int d = 0, int n = 0)
        {
            CopyToHost();
            return Values[Shape.GetIndex(w, h, d, n)];
        }

        public float this[int w, int h = 0, int d = 0, int n = 0]
        {
            get { return Get(w, h, d, n); }
            set { Set(value, w, h, d, n); }
        }

        public float TryGet(float def, int w, int h = 0, int d = 0, int n = 0)
        {
            if (h < 0 || h >= Height || w < 0 || w >= Width || d < 0 || d >= Depth)
                return def;

            return Get(w, h, d, n);
        }

        public void SetFlat(float value, int i)
        {
            CopyToHost();
            Values[i] = value;
        }

        public void Set(float value, int w, int h = 0, int d = 0, int n = 0)
        {
            CopyToHost();
            Values[Shape.GetIndex(w, h, d, n)] = value;
        }

        public void TrySet(float value, int w, int h = 0, int d = 0, int n = 0)
        {
            if (h < 0 || h >= Height || w < 0 || w >= Width || d < 0 || d >= Depth || n < 0 || n > BatchSize)
                return;

            Set(value, w, h, d, n);
        }

        public void CopyTo(Tensor result, float tau = float.NaN)
        {
            CopyToHost();
            if (Shape.Length != result.Shape.Length) throw new Exception("Incompatible tensors.");

            if (float.IsNaN(tau))
                Array.Copy(Values, result.Values, Length);
            else
                Map((v1, v2) => v1 * tau + v2 * (1 - tau), result, result);
        }

        public void CopyBatchTo(int batchId, int targetBatchId, Tensor result)
        {
            CopyToHost();
            result.CurrentLocation = Tensor.Location.Host;
            if (Shape.Width != result.Shape.Width || Shape.Height != result.Shape.Height || Shape.Depth != result.Shape.Depth) throw new Exception("Incompatible tensors.");

            Array.Copy(Values, batchId * Shape.Dim0Dim1Dim2, result.Values, targetBatchId * Shape.Dim0Dim1Dim2, Shape.Dim0Dim1Dim2);
        }

        public void CopyDepthTo(int depthId, int batchId, int targetDepthId, int targetBatchId, Tensor result)
        {
            CopyToHost();
            result.CurrentLocation = Tensor.Location.Host;
            if (Shape.Width != result.Shape.Width || Shape.Height != result.Shape.Height) throw new Exception("Incompatible tensors.");

            Array.Copy(Values, batchId * Shape.Dim0Dim1Dim2 + depthId * Shape.Dim0Dim1, result.Values, targetBatchId * Shape.Dim0Dim1Dim2 + targetDepthId * Shape.Dim0Dim1, Shape.Dim0Dim1);
        }

        public Tensor GetBatch(int batchId)
        {
            Tensor result = new Tensor(new Shape(Width, Height, Depth));
            CopyBatchTo(batchId, 0, result);
            return result;
        }

        public Tensor GetDepth(int depthId, int batchId = 0)
        {
            Tensor result = new Tensor(new Shape(Width, Height));
            CopyDepthTo(depthId, batchId, 0, 0, result);
            return result;
        }

        public bool Equals(Tensor other, float epsilon = 0.00001f)
        {
            CopyToHost();
            other.CopyToHost();

            if (other == null)
                return false;

            //Debug.Assert(Values.Length == other.Values.Length, "Comparing tensors with different number of elements!");
            if (Values.Length != other.Values.Length)
                return false;

            if (epsilon == 0)
                return Values.SequenceEqual(other.Values);

            for (int i = 0; i < Values.Length; ++i)
                if (Math.Abs(Values[i] - other.Values[i]) > epsilon)
                     return false;

            return true;
        }

        private float GetMaxData(int batch, out int maxIndex)
        {
            CopyToHost();
            maxIndex = -1;
            float maxValue = float.MinValue;

            if (batch < 0)
            {
                for (int i = 0; i < Values.Length; ++i)
                    if (Values[i] > maxValue)
                    {
                        maxValue = Values[i];
                        maxIndex = i;
                    }
            }
            else
            {
                int batchLen = BatchLength;

                for (int i = 0, idx = batch * batchLen; i < batchLen; ++i, ++idx)
                    if(Values[idx] > maxValue)
                    {
                        maxValue = Values[idx];
                        maxIndex = i;
                    }
            }

            return maxValue;
        }

        public void Elu(float alpha, Tensor result)
        {
            Op.Elu(this, alpha, result);
        }

        public static void EluGradient(Tensor output, Tensor outputGradient, float alpha, Tensor result)
        {
            Op.EluGradient(output, outputGradient, alpha, result);
        }

        public Shape Shape { get; private set; }

        private static TensorOpCpu Op;
        private static Random Rng = new Random();

        internal class GPUData : IDisposable
        {
            public CudaDeviceVariable<float> DeviceVar;
            public CudaDeviceVariable<byte> ConvWorkspace;
            public CudaDeviceVariable<byte> ConvBackWorkspace;
            public CudaDeviceVariable<byte> ConvBackKernelWorkspace;

            ~GPUData()
            {
                Dispose();
            }

            public void Dispose()
            {
                DeviceVar?.Dispose(); DeviceVar = null;
                ConvWorkspace?.Dispose(); ConvWorkspace = null;
                ConvBackWorkspace?.Dispose(); ConvBackWorkspace = null;
                ConvBackKernelWorkspace?.Dispose(); ConvBackKernelWorkspace = null;
            }
        }

        public void CopyToDevice()
        {
            if (CurrentLocation == Location.Device)
                return;

            GpuData.DeviceVar = GpuData.DeviceVar ?? new CudaDeviceVariable<float>(Shape.Length);
            GpuData.DeviceVar.CopyToDevice(Values);
            CurrentLocation = Location.Device;
        }

        public void CopyToHost()
        {
            if (CurrentLocation == Location.Host)
                return;

            GpuData.DeviceVar.CopyToHost(Values);
            CurrentLocation = Location.Host;
        }

        internal GPUData GpuData = new GPUData();
        internal Location CurrentLocation = Location.Host;
        internal float[] Values;
    }
}
