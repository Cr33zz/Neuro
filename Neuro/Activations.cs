using System;
using Tensorflow;

namespace Neuro
{
    public abstract class ActivationFunc
    {
        public abstract Tensor Build(Tensor input);
    }

    public static class Activation
    {
        public static ActivationFunc Linear = new Linear();
        public static ActivationFunc Sigmoid = new Sigmoid();
        public static ActivationFunc Tanh = new Tanh();
        public static ActivationFunc ReLU = new ReLU();
        public static ActivationFunc ELU = new ELU();
        public static ActivationFunc Softmax = new Softmax();
    }

    public class Linear : ActivationFunc
    {
        public override Tensor Build(Tensor input)
        {
            return input;
        }
    }

    public class Sigmoid : ActivationFunc
    {
        public override Tensor Build(Tensor input)
        {
            return tf.sigmoid(input);
        }
    }

    public class Tanh : ActivationFunc
    {
        public override Tensor Build(Tensor input)
        {
            return tf.tanh(input);
        }
    }

    public class ReLU : ActivationFunc
    {
        public override Tensor Build(Tensor input)
        {
            return tf.nn.relu(input);
        }
    }

    public class ELU : ActivationFunc
    {
        public override Tensor Build(Tensor input)
        {
            return null;// tf.nn.(input);
        }
    }

    public class Softmax : ActivationFunc
    {
        public override Tensor Build(Tensor input)
        {
            return tf.nn.softmax(input);
        }
    }
}
