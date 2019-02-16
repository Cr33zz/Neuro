using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;

namespace Neuro
{
    public class Tensor
    {
        public Tensor(float value)
        {
            _Tensor = new TFTensor(value);
            Shape = new TFShape(1);
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

        public Tensor(TFTensor tensor, TFShape shape)
        {
            _Tensor = tensor;
            Shape = shape;
        }

        public TFTensor _Tensor;
        public TFOutput Output;
        public TFShape Shape;

        //public static implicit operator TFOutput(Tensor t)
        //{
        //    return t.Output;
        //}

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
