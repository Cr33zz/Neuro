using System;

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
            return Backend.Sigmoid(input);
        }
    }

    public class Tanh : ActivationFunc
    {
        public override Tensor Build(Tensor input)
        {
            return Backend.Tanh(input);
        }
    }

    public class ReLU : ActivationFunc
    {
        public override Tensor Build(Tensor input)
        {
            return Backend.Relu(input);
        }
    }

    public class ELU : ActivationFunc
    {
        public override Tensor Build(Tensor input)
        {
            return Backend.Elu(input);
        }
    }

    public class Softmax : ActivationFunc
    {
        public override Tensor Build(Tensor input)
        {
            return Backend.Softmax(input);
        }
    }
}
