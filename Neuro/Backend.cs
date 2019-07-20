using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow;

namespace Neuro
{
    public static class Backend
    {
        public enum PaddingType
        {
            Valid,
            Same,
            Full
        }

        public enum DataFormatType
        {
            ChannelsFirst,
            ChannelsLast,
        }

        /*static Backend()
        {
            Graph = new Tensorflow.Graph();
            Session = new Tensorflow.Session(Graph);
        }

        public static Tensor Const<T>(T value, string name = null)
        {
            return new Tensor(tf.constant(new Tensorflow.Tensor((dynamic)value), name));
        }

        public static Tensor Placeholder(int[] shape = null, string name = null)
        {
            var tfShape = ToShape(shape);
            return new Tensor(Graph.Placeholder(Tensorflow.DataType.DtFloat, tfShape, name));
        }

        //public static Tensor Variable(Tensor tensor, string name)
        //{
        //    var v = new Tensor(Graph.VariableV2(tensor.Shape, Tensorflow.DataType.DtFloat, operName: name));

        //    Tensorflow.TF_Output init = (tensor._Tensor != null) ? Graph.Const(tensor._Tensor, Tensorflow.DataType.DtFloat) : tensor.Output;

        //    init = Graph.Print(init, new[] { init }, $"initializing {v.Name}");
        //    Graph.AddInitVariable(Graph.Assign(v.Output, init).Operation);
        //    return v;
        //}

        public static RefVariable Variable(Array array, string name)
        {
            var v = new Tensor(tf.Variable(array, Tensorflow.DataType.DtFloat, perName: name));
            return v;
        }

        public static Tensor Zeros(int[] shape, string name)
        {
            Array zeros = Array.CreateInstance(typeof(float), shape);
            return Variable(zeros, name);
        }

        public static Tensor Identity(Tensor x, string name = null)
        {
            return new Tensor(Graph.Identity(x.Output, name));
        }

        public static Tensor Dot(Tensor x, Tensor y)
        {
            return new Tensor(tf.matmul(x.Output, y.Output));
        }

        public static Tensor Mul<T>(T a, Tensor b)
        {
            return Mul(Const(a), b);
        }

        public static Tensor Mul(Tensor a, Tensor b)
        {
            return new Tensor(tf.multiply(a.Output, b.Output));
        }

        public static Tensor Mul<T>(Tensor a, T b)
        {
            return Mul(a, Const(b));
        }

        public static Tensor Div<T>(T a, Tensor b)
        {
            return Div(Const(a), b);
        }

        public static Tensor Div(Tensor a, Tensor b)
        {
            return new Tensor(Graph.Div(a.Output, b.Output));
        }

        public static Tensor Div<T>(Tensor a, T b)
        {
            return Div(a, Const(b));
        }

        public static Tensor Add<T>(T a, Tensor b)
        {
            return Add(Const(a), b);
        }

        public static Tensor Add(Tensor a, Tensor b)
        {
            return new Tensor(tf.add(a.Output, b.Output));
        }

        public static Tensor Add<T>(Tensor a, T b)
        {
            return Add(a, Const(b));
        }

        public static Tensor Sub<T>(T a, Tensor b)
        {
            return Sub(Const(a), b);
        }

        public static Tensor Sub(Tensor a, Tensor b)
        {
            return new Tensor(Graph.Sub(a.Output, b.Output));
        }

        public static Tensor Sub<T>(Tensor a, T b)
        {
            return Sub(a, Const(b));
        }

        public static Tensor Transpose(Tensor x, int[] perm)
        {
            return new Tensor(tf.transpose(x.Output, Const(perm).Output));
        }

        public static Tensor BatchFlatten(Tensor x)
        {
            Tensorflow.TF_Output shape = tf.sx.;
            Tensorflow.TF_Output dim = tf.Graph.Prod(Graph.Slice(shape, Graph.Const(1), Graph.Rank(shape)), reduction_indices: Graph.ReduceDims(shape));
            return new Tensor(Graph.Reshape(x.Output, Graph.Stack(new Tensorflow.TF_Output[] { Graph.Const(-1), dim })));
        }

        // Data is expected to be in NHWC format, for example [4, 84, 84, 3] is 4 batches of 84x84 3 depth each
        public static Tensor Conv2D(Tensor x, Tensor kernel, int[] strides, PaddingType padding)
        {
            return new Tensor(Graph.Conv2D(x.Output, kernel.Output, strides.Select(i => (long)i).ToArray(), padding.ToString().ToUpper(), data_format: "NHWC"));
        }

        public static object Eval(Tensor t)
        {
            try
            {
                TFOperation[] ops = Graph.GetGlobalVariablesInitializer();
                if (ops.Length > 0)
                    Session.Run(new Tensorflow.TF_Output[] { }, new Tensorflow.Tensor[] { }, new Tensorflow.TF_Output[] { }, ops);
            }
            catch
            {
            }

            Tensorflow.Tensor[] result = Session.Run(new Tensorflow.TF_Output[] { }, new Tensorflow.Tensor[] { }, new[] { t.Output });

            if (result.Length == 1)
                return result[0].GetValue();

            return result[0].GetValue(true);
        }

        public static Tensor RandomUniform(int[] shape, float minVal = 0, float maxVal = 1, int? seed = null)
        {
            using (WithScope("random_uniform"))
            {
                return new Tensor(Graph._RandomUniform(ToShape(shape), minVal, maxval: maxVal, seed: seed));
            }
        }

        public static Tensor RandomNormal(int[] shape, float mean = 0, float stddev = 1, int? seed = null)
        {
            using (WithScope("random_normal"))
            {
                return new Tensor(Graph._RandomNormal(ToShape(shape), mean, stddev, seed: seed));
            }
        }

        public static Tensor ReduceSum(Tensor x, int? axis = null)
        {
            if (axis.HasValue)
                return new Tensor(Graph.ReduceSum(x.Output, axis: Const(axis.Value).Output));
            return new Tensor(Graph.ReduceSum(x.Output));
        }

        public static Tensor ClipByValue(Tensor x, float min, float max)
        {
            return new Tensor(Graph.ClipByValue(x.Output, Const(min).Output, Const(max).Output));
        }

        public static Tensor Neg(Tensor x)
        {
            return new Tensor(Graph.Neg(x.Output));
        }

        public static Tensor Log(Tensor x)
        {
            return new Tensor(Graph.Log(x.Output));
        }

        public static Tensor Mean(Tensor x, int axis = -1, bool keepDims = false)
        {
            return new Tensor(Graph.ReduceMean(x.Output, Const(axis).Output, keepDims));
        }

        public static Tensor Square(Tensor x)
        {
            return new Tensor(Graph.Square(x.Output));
        }

        public static Tensor Pow(Tensor x, Tensor p)
        {
            return new Tensor(Graph.Pow(x.Output, p.Output));
        }

        public static Tensor Sqrt(Tensor x)
        {
            return new Tensor(Graph.Sqrt(x.Output));
        }

        public static Tensor Assign(Tensor x, Tensor newX)
        {
            return new Tensor(Graph.Assign(x.Output, newX.Output));
        }

        public static Tensor AssignAdd(Tensor x, Array increment)
        {
            return new Tensor(Graph.AssignAdd(x.Output, Const(increment).Output));
        }

        public static Tensor AssignAdd(Tensor x, float increment)
        {
            return new Tensor(Graph.AssignAdd(x.Output, Const(increment).Output));
        }

        public static Tensor Print(Tensor x, string message = null)
        {
            return new Tensor(Graph.Print(x.Output, new[] { x.Output }, message));
        }

        public static List<Tensor> Gradients(Tensor loss, List<Tensor> param)
        {
            Tensorflow.TF_Output[] grads = Graph.AddGradients(new []{ loss.Output }, param.Select(x => x.Output).ToArray());

            List<Tensor> result = new List<Tensor>();
            for (int i = 0; i < grads.Length; i++)
                result.Add(new Tensor(grads[i]));
            return result;
        }

        public static Tensor Elu(Tensor x)
        {
            return new Tensor(Graph.Elu(x.Output));
        }

        public static Tensor Tanh(Tensor x)
        {
            return new Tensor(Graph.Tanh(x.Output));
        }

        public static Tensor Softmax(Tensor x)
        {
            return new Tensor(Graph.Softmax(x.Output));
        }

        public static Tensor Sigmoid(Tensor x)
        {
            return new Tensor(Graph.Sigmoid(x.Output));
        }

        public static Tensor Relu(Tensor x)
        {
            return new Tensor(Graph.Relu(x.Output));
        }

        public static TFScope WithScope(string name)
        {
            return Graph.WithScope(name);
        }

        public static TFShape ToShape(int?[] shape)
        {
            return new TFShape(shape.Select(x => x.HasValue ? (long)x.Value : -1).ToArray());
        }

        public static TFShape ToShape(int[] shape)
        {
            return new TFShape(shape.Select(x => (long)x).ToArray());
        }

        public static TFShape ToShape(Array array)
        {
            return ToShape(array.GetShape());
        }

        public static TFShape ToShape(Tensorflow.TF_Output output)
        {
            return Graph.GetTensorShape(output);
        }

        public static TFShape ToShape(Tensorflow.Tensor tensor)
        {
            return new TFShape(tensor.Shape);
        }

        public static Tensorflow.Graph Graph;
        public static Tensorflow.Session Session;*/
    }
}
