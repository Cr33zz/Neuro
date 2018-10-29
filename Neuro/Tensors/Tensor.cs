using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Xml;

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

        public Tensor(Shape shape)
        {
            Shape = shape;
            Values = new double[shape.Length];
        }

        public Tensor(double[] values, Shape shape)
        {
            Debug.Assert(values.Length == shape.Length, $"Invalid array size {values.Length}. Expected {shape.Length}.");
            Shape = shape;
            Values = (double[])values.Clone();
        }

        public Tensor(Tensor t)
        {
            Shape = Shape.From(t.Shape.Dimensions);
            Values = (double[])t.Values.Clone();
        }

        public Tensor(string bmpFile, bool grayScale)
        {
            using (var bmp = new Bitmap(bmpFile))
            {
                Shape = new Shape(bmp.Width, bmp.Height, grayScale ? 1 : 3);
                Values = new double[Shape.Length];

                for (int h = 0; h < bmp.Height; ++h)
                {
                    for (int w = 0; w < bmp.Width; ++w)
                    {
                        Color c = bmp.GetPixel(w, h);
                        if (grayScale)
                            Set((c.R * 0.3 + c.G * 0.59 + c.B * 0.11) / 255.0, w, h);
                        else
                        {
                            Set(c.R / 255.0, w, h, 0);
                            Set(c.G / 255.0, w, h, 1);
                            Set(c.B / 255.0, w, h, 2);
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

        public int Batches => Shape.Batches;

        public int BatchLength => Shape.Width * Shape.Height * Shape.Depth;

        public int Length => Values.Length;

        public double[] GetValues()
        {
            return (double[])Values.Clone();
        }

        public Bitmap ToBitmap()
        {
            Debug.Assert(Batches == 1);

            Bitmap output = new Bitmap(Width, Height);
            bool grayScale = (Depth == 1);

            for (int d = 0; d < Depth; ++d)
            for (int h = 0; h < Height; ++h)
            for (int w = 0; w < Width; ++w)
                output.SetPixel(w, h, grayScale ? Color.FromArgb((int)(Get(w, h) * 255), (int)(Get(w, h) * 255), (int)(Get(w, h) * 255))
                                                : Color.FromArgb((int)(Get(w, h) * 255), (int)(Get(w, h, 1) * 255), (int)(Get(w, h, 2) * 255)));

            return output;
        }

        public void FillWithRand(int seed = 0)
        {
            Random rng = seed > 0 ? new Random(seed) : new Random();
            for (int i = 0; i < Values.Length; ++i)
                Values[i] = rng.NextDouble();
        }

        public void FillWithRange(int start = 0)
        {
            for (int i = 0; i < Values.Length; ++i)
                Values[i] = start + i;
        }

        public void Zero()
        {
            Array.Clear(Values, 0, Values.Length);
        }

        public virtual void Mul(Tensor t, Tensor result)
        {
            Debug.Assert(Width == t.Height);
            Debug.Assert(t.Depth == Depth);

            result.Zero();
            Op.Mul(this, t, result);
        }

        public Tensor Mul(Tensor t)
        {
            Tensor result = new Tensor(new Shape(t.Shape.Width, Height, Depth, Batches));
            Mul(t, result);
            return result;
        }

        // Element-wise multiplication
        public virtual void MulElem(Tensor t, Tensor result)
        {
            Debug.Assert(SameDimensions(t));
            Op.MulElem(this, t, result);
        }

        public Tensor MulElem(Tensor t)
        {
            Tensor result = new Tensor(Shape);
            MulElem(t, result);
            return result;
        }

        public virtual void Mul(double v, Tensor result)
        {
            for (int i = 0; i < Values.Length; ++i)
                result.Values[i] = Values[i] * v;
        }

        public Tensor Mul(double v)
        {
            Tensor result = new Tensor(Shape);
            Mul(v, result);
            return result;
        }

        public virtual void Div(Tensor t, Tensor result)
        {
            Debug.Assert(SameDimensions(t));

            for (int i = 0; i < Values.Length; ++i)
                result.Values[i] = Values[i] / t.Values[i];
        }

        public Tensor Div(Tensor t)
        {
            Tensor result = new Tensor(Shape);
            Div(t, result);
            return result;
        }

        public virtual void Div(double v, Tensor result)
        {
            double invV = 1 / v;
            for (int i = 0; i < Values.Length; ++i)
                result.Values[i] = Values[i] * invV;
        }

        public Tensor Div(double v)
        {
            Tensor result = new Tensor(Shape);
            Div(v, result);
            return result;
        }

        public virtual void Add(Tensor t, Tensor result)
        {
            Debug.Assert(SameDimensions(t));
            Op.Add(this, t, result);
        }

        public Tensor Add(Tensor t)
        {
            Tensor result = new Tensor(Shape);
            Add(t, result);
            return result;
        }

        public virtual void Add(double v, Tensor result)
        {
            for (int i = 0; i < Values.Length; ++i)
                result.Values[i] = Values[i] + v;
        }

        public Tensor Add(double v)
        {
            Tensor result = new Tensor(Shape);
            Add(v, result);
            return result;
        }

        public virtual void Sub(Tensor t, Tensor result)
        {
            Debug.Assert(SameDimensions(t));
            Op.Sub(this, t, result);
        }

        public Tensor Sub(Tensor t)
        {
            Tensor result = new Tensor(Shape);
            Sub(t, result);
            return result;
        }

        public virtual void Sub(double v, Tensor result)
        {
            for (int i = 0; i < Values.Length; ++i)
                result.Values[i] = Values[i] - v;
        }

        public Tensor Sub(double v)
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

        public virtual void Map(Func<double, double> func, Tensor result)
        {
            for (int i = 0; i < Values.Length; ++i)
                result.Values[i] = func(Values[i]);
        }

        public Tensor Map(Func<double, double> func)
        {
            Tensor result = new Tensor(Shape);
            Map(func, result);
            return result;
        }

        public Tensor SumBatches()
        {
            Tensor result = new Tensor(new Shape(Shape.Width, Shape.Height, Shape.Depth, 1));

            for (int n = 0; n < Batches; ++n)
            for (int i = 0, idx = n * BatchLength; i < BatchLength; ++i, ++idx)
                    result.Values[i] += Values[idx];

            return result;
        }

        public double Sum(int batch = -1)
        {
            if (batch < 0)
                return Values.Sum();

            double sum = 0;

            for (int i = 0, idx = batch * BatchLength; i < BatchLength; ++i, ++idx)
                sum += Values[idx];

            return sum;
        }

        public double Max(int batch = -1)
        {
            if (batch < 0)
                return Values.Max();

            double max = double.MinValue;
            
            for (int i = 0, idx = batch * BatchLength; i < BatchLength; ++i, ++idx)
                max = Math.Max(max, Values[idx]);

            return max;
        }

        public static Tensor Merge(List<Tensor> list)
        {
            Tensor output = new Tensor(new Shape(list[0].Width, list[0].Height, list[0].Depth, list.Count));

            for (int n = 0; n < list.Count; ++n)
            {
                Tensor t = list[n];
                Array.Copy(t.Values, 0, output.Values, t.Length * n, t.Values.Length);
            }

            return output;
        }

        public void Normalized(Tensor result)
        {
            double sum = Sum();
            Map(x => x / sum, result);
        }

        public int ArgMax(int batch = -1)
        {
            double max = Max(batch);

            if (batch < 0)
            {
                for (int i = 0; i < Values.Length; ++i)
                    if (Values[i] == max)
                        return i;
            }
            else
            {
                for (int i = 0, idx = batch * BatchLength; i < BatchLength; ++i, ++idx)
                    if (Values[idx] == max)
                        return idx;
            }

            return -1;
        }

        public virtual Tensor Transposed()
        {
            Tensor result = new Tensor(new Shape(Height, Width, Depth, Batches));
            for (int n = 0; n < Batches; ++n)
            for (int d = 0; d < Depth; ++d)
            for (int h = 0; h < Height; ++h)
            for (int w = 0; w < Width; ++w)
                result[h, w, d, n] = this[w, h, d, n];

            return result;
        }

        // Generates a new matrix with given dimensions and populate it with this matrix values in index order. 
        // One of dimensions can be -1, in that case it will be calculated based on remaining dimensions.
        public Tensor Reshaped(Shape shape)
        {
            Debug.Assert(shape.Length == Shape.Length);
            return new Tensor(Values, shape);
        }

        public void Reshape(Shape shape)
        {
            Debug.Assert(shape.Length == Shape.Length);
            Shape = shape;
        }

        public Tensor FlattenHoriz()
        {
            return Reshaped(Shape.Reshaped(new[] { -1, 1, 1, Shape.Batches }));
        }

        public Tensor FlattenVert()
        {
            return Reshaped(Shape.Reshaped(new[] { 1, -1, 1, Shape.Batches }));
        }

        public void Rotated180(Tensor result)
        {
            Debug.Assert(SameDimensions(result));

            for (int n = 0; n < Batches; ++n)
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

            int outputWidth = 0, outputHeight = 0, paddingX = 0, paddingY = 0;
            GetPaddingParams(padding, Width, Height, kernels.Width, kernels.Height, stride, out outputHeight, out outputWidth, out paddingX, out paddingY);

            Op.Conv2D(this, kernels, stride, paddingX, paddingY, result);
        }

        public Tensor Conv2D(Tensor kernels, int stride, PaddingType padding)
        {
            int outputWidth = 0, outputHeight = 0, paddingX = 0, paddingY = 0;
            GetPaddingParams(padding, Width, Height, kernels.Width, kernels.Height, stride, out outputHeight, out outputWidth, out paddingX, out paddingY);

            Tensor result = new Tensor(new Shape(outputWidth, outputHeight, kernels.Batches, Batches));
            Conv2D(kernels, stride, padding, result);
            return result;
        }

        public static void Conv2DInputsGradient(Tensor gradient, Tensor kernels, int stride, Tensor inputsGradient)
        {
            inputsGradient.Zero();
            Tensor rotKernels = kernels.Rotated180();

            int outputWidth = 0, outputHeight = 0, paddingX = 0, paddingY = 0;
            GetPaddingParams(PaddingType.Full, gradient.Width, gradient.Height, kernels.Width, kernels.Height, stride, out outputHeight, out outputWidth, out paddingX, out paddingY);
            Op.Conv2DInputGradient(gradient, rotKernels, stride, paddingX, paddingY, inputsGradient);
        }

        public static void Conv2DKernelsGradient(Tensor output, Tensor input, Tensor gradient, int stride, PaddingType padding, Tensor kernelsGradient)
        {
            kernelsGradient.Zero();
            int outputWidth = 0, outputHeight = 0, paddingX = 0, paddingY = 0;
            GetPaddingParams(padding, input.Width, input.Height, kernelsGradient.Width, kernelsGradient.Height, stride, out outputHeight, out outputWidth, out paddingX, out paddingY);
            Op.Conv2DKernelsGradient(output, input, gradient, stride, paddingX, paddingY, kernelsGradient);
        }

        public void Conv2DGradient_old(Tensor input, Tensor kernels, Tensor gradient, int stride, PaddingType padding, Tensor inputGradient, Tensor kernelsGradient)
        {
            inputGradient.Zero();
            kernelsGradient.Zero();
            int outputWidth = 0, outputHeight = 0, paddingX = 0, paddingY = 0;
            GetPaddingParams(padding, input.Width, input.Height, kernels.Width, kernels.Height, stride, out outputHeight, out outputWidth, out paddingX, out paddingY);
            Op.Conv2DGradient_old(this, input, kernels, gradient, stride, paddingX, paddingY, inputGradient, kernelsGradient);
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
            Debug.Assert(result.Batches == Batches);

            Op.Pool(this, filterSize, stride, type, paddingX, paddingY, result);
        }

        public Tensor Pool(int filterSize, int stride, PoolType type, PaddingType padding)
        {
            int outWidth = 0, outHeight = 0, paddingX = 0, paddingY = 0;
            GetPaddingParams(padding, Width, Height, filterSize, filterSize, stride, out outHeight, out outWidth, out paddingX, out paddingY);

            Tensor result = new Tensor(new Shape(outWidth, outHeight, Depth, Batches));
            Pool(filterSize, stride, type, padding, result);

            return result;
        }

        // Assuming result matrix is of the dimensions of input to pooling layer
        public static void PoolGradient(Tensor output, Tensor input, Tensor gradient, int filterSize, int stride, PoolType type, PaddingType padding, Tensor result)
        {
            Debug.Assert(output.SameDimensions(gradient));

            int outWidth = 0, outHeight = 0, paddingX = 0, paddingY = 0;
            GetPaddingParams(padding, result.Width, result.Height, filterSize, filterSize, stride, out outHeight, out outWidth, out paddingX, out paddingY);

            Op.PoolGradient(output, input, gradient, filterSize, stride, type, paddingX, paddingY, result);
        }

        public override string ToString()
        {
            string s = "";
            for (int n = 0; n < Batches; ++n)
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
                s += "}" + (n < Batches - 1 ? "\n" : "");
            }

            return s;
        }

        public bool SameDimensions(Tensor t)
        {
            return Width == t.Width && Height == t.Height && Depth == t.Depth;
        }

        internal static void GetPaddingParams(PaddingType type, int width, int height, int kernelWidth, int kernelHeight, int stride, out int outHeight, out int outWidth, out int paddingX, out int paddingY)
        {
            if (type == PaddingType.Valid)
            {
                outWidth = (int)Math.Floor((width - kernelWidth) / (double)stride + 1);
                outHeight = (int)Math.Floor((height - kernelHeight) / (double)stride + 1);
                paddingX = 0;
                paddingY = 0;
            }
            else if (type == PaddingType.Same)
            {
                outWidth = width / stride;
                outHeight = height / stride;
                paddingX = (int)Math.Floor((double)kernelWidth / 2);
                paddingY = (int)Math.Floor((double)kernelHeight / 2);
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
            Values = elem.InnerText.Split(',').Select(w => double.Parse(w)).ToArray();
        }

        public double GetFlat(int i)
        {
            return Values[i];
        }

        public double Get(int w, int h = 0, int d = 0, int n = 0)
        {
            return Values[Shape.GetIndex(w, h, d, n)];
        }

        public double this[int w, int h = 0, int d = 0, int n = 0]
        {
            get { return Get(w, h, d, n); }
            set { Set(value, w, h, d, n); }
        }

        public double TryGet(double def, int w, int h = 0, int d = 0, int n = 0)
        {
            if (h < 0 || h >= Height || w < 0 || w >= Width || d < 0 || d >= Depth)
                return def;

            return Get(w, h, d, n);
        }

        public void SetFlat(double value, int i)
        {
            Values[i] = value;
        }

        public void Set(double value, int w, int h = 0, int d = 0, int n = 0)
        {
            Values[Shape.GetIndex(w, h, d, n)] = value;
        }

        public void TrySet(double value, int w, int h = 0, int d = 0, int n = 0)
        {
            if (h < 0 || h >= Height || w < 0 || w >= Width || d < 0 || d >= Depth || n < 0 || n > Batches)
                return;

            Set(value, w, h, d, n);
        }

        public void CopyTo(Tensor result)
        {
            Debug.Assert(Shape.Length == result.Shape.Length);
            Array.Copy(Values, result.Values, Length);
        }

        public void CopyBatchTo(int n, int resultN, Tensor result)
        {
            Debug.Assert(Shape.Width == result.Shape.Width && Shape.Height == result.Shape.Height && Shape.Depth == result.Shape.Depth);
            int valuesPerBatch = Shape.Width * Shape.Height * Shape.Depth;
            Array.Copy(Values, n * valuesPerBatch, result.Values, resultN * valuesPerBatch, valuesPerBatch);
        }

        public void CopyBatchFrom(int n, int inputN, Tensor input)
        {
            Debug.Assert(Shape.Width == input.Shape.Width && Shape.Height == input.Shape.Height && Shape.Depth == input.Shape.Depth);
            int valuesPerBatch = Shape.Width * Shape.Height * Shape.Depth;
            Array.Copy(input.Values, inputN * valuesPerBatch, Values, n * valuesPerBatch, valuesPerBatch);
        }

        public bool Equals(Tensor other, double epsilon = 0.0000001)
        {
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

        public Shape Shape { get; private set; }

        private static TensorOpCpu Op = new TensorOpGpu();

        internal double[] Values;
    }
}
