using System;
using System.Collections.Generic;
using System.Linq;
using TensorFlow;

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

        static Backend()
        {
            Graph = new TFGraph();
            Session = new TFSession(Graph);
        }

        public static Tensor Const<T>(T value)
        {
            return new Tensor(Graph.Const(new TFTensor((dynamic)value), TFDataType.Float));
        }

        public static Tensor Placeholder(int[] shape = null, string name = null)
        {
            var tfShape = ToShape(shape);
            return new Tensor(Graph.Placeholder(TFDataType.Float, tfShape, name));
        }

        public static Tensor Variable(Tensor tensor)
        {
            var v = new Tensor(Graph.VariableV2(tensor.Shape, TFDataType.Float));

            TFOutput init = (tensor._Tensor != null) ? Graph.Const(tensor._Tensor, TFDataType.Float) :
                                                       tensor.Output;

            Graph.AddInitVariable(Graph.AssignVariableOp(v.Output, init));
            return v;
        }

        public static Tensor Variable(Array array)
        {
            var v = new Tensor(Graph.VariableV2(ToShape(array), TFDataType.Float));
            Graph.AddInitVariable(Graph.AssignVariableOp(v.Output, Graph.Const(new TFTensor(array), TFDataType.Float)));
            return v;
        }

        public static Tensor Zeros(int[] shape)
        {
            Array zeros = Array.CreateInstance(typeof(float), shape);
            return Variable(zeros);
        }

        public static Tensor Dot(Tensor x, Tensor y)
        {
            return new Tensor(Graph.MatMul(x.Output, y.Output));
        }

        public static Tensor Mul<T>(T a, Tensor b)
        {
            return Mul(Const(a), b);
        }

        public static Tensor Mul(Tensor a, Tensor b)
        {
            return new Tensor(Graph.Mul(a.Output, b.Output));
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
            return new Tensor(Graph.Add(a.Output, b.Output));
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
            return new Tensor(Graph.Transpose(x.Output, Const(perm).Output));
        }

        public static Tensor BatchFlatten(Tensor x)
        {
            TFOutput shape = Graph.Shape(x.Output);
            TFOutput dim = Graph.Prod(Graph.Slice(shape, Graph.Const(1), Graph.Rank(shape)), reduction_indices: Graph.ReduceDims(shape));
            return new Tensor(Graph.Reshape(x.Output, Graph.Stack(new TFOutput[] { Graph.Const(-1), dim })));
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
                    Session.Run(new TFOutput[] { }, new TFTensor[] { }, new TFOutput[] { }, ops);
            }
            catch
            {
            }

            TFTensor[] result = Session.Run(new TFOutput[] { }, new TFTensor[] { }, new[] { t.Output });

            if (result.Length == 1)
                return result[0].GetValue();

            return result[0].GetValue(true);
        }

        public static Tensor RandomUniform(int[] shape, float minVal = 0, float maxVal = 1, int? seed = null)
        {
            using (WithScope("random_uniform"))
            {
                TFOutput u = Graph.RandomUniform(ToShape(shape), minVal, maxval: maxVal, seed: seed);
                return new Tensor(Graph.Add(Graph.Mul(u, Const(maxVal - minVal).Output), Const(minVal).Output));
            }
        }

        public static Tensor TruncatedNormal(int[] shape, int? seed = null)
        {
            throw new NotImplementedException();
            //using (WithScope("random_normal"))
            //{
            //    return new Tensor(Graph.TruncatedNormal(ToShape(shape), TFDataType.Float, seed: seed));
            //}
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

        public static Tensor Mean(Tensor x, int? axis = null, bool keepDims = false)
        {
            if (axis.HasValue)
                return new Tensor(Graph.ReduceMean(x.Output, axis: Const(axis.Value).Output, keep_dims: keepDims));
            return new Tensor(Graph.ReduceMean(x.Output, keep_dims: keepDims));
        }

        public static Tensor Square(Tensor x)
        {
            return new Tensor(Graph.Square(x.Output));
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
            TFOutput[] grads = Graph.AddGradients(new []{ loss.Output }, param.Select(x => x.Output).ToArray());

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
            long[] shape = new long[array.Rank];
            for (int i = array.Rank - 1; i <= 0; --i)
                shape[i] = array.GetLength(i);
            return new TFShape(shape);
        }

        public static TFShape ToShape(TFOutput output)
        {
            return Graph.GetTensorShape(output);
        }

        private static TFGraph Graph;
        private static TFSession Session;
    }
}
