using System;
using Tensorflow;

namespace Neuro
{
    //public class Tensor
    //{
    //    public Tensor(float value)
    //    {
    //        _Tensor = new Tensorflow.Tensor(value);
    //        Shape = new TFShape();
    //    }

    //    public Tensor(Array values)
    //    {
    //        _Tensor = new Tensorflow.Tensor(values);
    //        Shape = tf.ToShape(values);
    //    }

    //    public Tensor(Tensorflow.TF_Output output)
    //    {
    //        Output = output;
    //        Shape = tf.ToShape(output);
    //    }

    //    public Tensor(Tensorflow.Tensor tensor)
    //    {
    //        _Tensor = tensor;
    //        Shape = tf.ToShape(tensor);
    //    }

    //    public T GetValue<T>()
    //    {
    //        return (T)_Tensor.GetValue();
    //    }

    //    public string Name => Output.Operation.Name;

    //    public Tensorflow.Tensor _Tensor;
    //    public Tensorflow.TF_Output Output;
    //    public Tensorflow.TensorShape Shape;

    //    public override string ToString()
    //    {
    //        return $"Tensor '{Output.Operation.Name}_{Output.Index}' shape={Shape.Str()}";
    //    }

    //    public static Tensor operator *(float a, Tensor b)
    //    {
    //        return tf.Mul(a, b);
    //    }

    //    public static Tensor operator *(Tensor a, float b)
    //    {
    //        return tf.Mul(a, b);
    //    }

    //    public static Tensor operator *(Tensor a, Tensor b)
    //    {
    //        return tf.Mul(a, b);
    //    }

    //    public static Tensor operator /(float a, Tensor b)
    //    {
    //        return tf.Div(a, b);
    //    }

    //    public static Tensor operator /(Tensor a, float b)
    //    {
    //        return tf.Div(a, b);
    //    }

    //    public static Tensor operator /(Tensor a, Tensor b)
    //    {
    //        return tf.Div(a, b);
    //    }

    //    public static Tensor operator +(float a, Tensor b)
    //    {
    //        return tf.Add(a, b);
    //    }

    //    public static Tensor operator +(Tensor a, float b)
    //    {
    //        return tf.Add(a, b);
    //    }

    //    public static Tensor operator +(Tensor a, Tensor b)
    //    {
    //        return tf.Add(a, b);
    //    }

    //    public static Tensor operator -(float a, Tensor b)
    //    {
    //        return tf.Sub(a, b);
    //    }

    //    public static Tensor operator -(Tensor a, float b)
    //    {
    //        return tf.Sub(a, b);
    //    }

    //    public static Tensor operator -(Tensor a, Tensor b)
    //    {
    //        return tf.Sub(a, b);
    //    }
    //}
}
