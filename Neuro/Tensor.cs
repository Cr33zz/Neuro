using System;
using TensorFlow;

namespace Neuro
{
    public class Tensor
    {
        public Tensor(float value)
        {
            _Tensor = new TFTensor(value);
            Shape = new TFShape();
        }

        public Tensor(Array values)
        {
            _Tensor = new TFTensor(values);
            Shape = Backend.ToShape(values);
        }

        public Tensor(TFOutput output)
        {
            Output = output;
            Shape = Backend.ToShape(output);
        }

        public Tensor(TFTensor tensor)
        {
            _Tensor = tensor;
            Shape = Backend.ToShape(tensor);
        }

        public T GetValue<T>()
        {
            return (T)_Tensor.GetValue();
        }

        public string Name => Output.Operation.Name;

        public TFTensor _Tensor;
        public TFOutput Output;
        public TFShape Shape;

        public override string ToString()
        {
            return $"Tensor '{Output.Operation.Name}_{Output.Index}' shape={Shape.Str()}";
        }

        public static Tensor operator *(float a, Tensor b)
        {
            return Backend.Mul(a, b);
        }

        public static Tensor operator *(Tensor a, float b)
        {
            return Backend.Mul(a, b);
        }

        public static Tensor operator *(Tensor a, Tensor b)
        {
            return Backend.Mul(a, b);
        }

        public static Tensor operator /(float a, Tensor b)
        {
            return Backend.Div(a, b);
        }

        public static Tensor operator /(Tensor a, float b)
        {
            return Backend.Div(a, b);
        }

        public static Tensor operator /(Tensor a, Tensor b)
        {
            return Backend.Div(a, b);
        }

        public static Tensor operator +(float a, Tensor b)
        {
            return Backend.Add(a, b);
        }

        public static Tensor operator +(Tensor a, float b)
        {
            return Backend.Add(a, b);
        }

        public static Tensor operator +(Tensor a, Tensor b)
        {
            return Backend.Add(a, b);
        }

        public static Tensor operator -(float a, Tensor b)
        {
            return Backend.Sub(a, b);
        }

        public static Tensor operator -(Tensor a, float b)
        {
            return Backend.Sub(a, b);
        }

        public static Tensor operator -(Tensor a, Tensor b)
        {
            return Backend.Sub(a, b);
        }
    }
}
