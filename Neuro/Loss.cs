using System;
using Tensorflow;

namespace Neuro
{
    public abstract class Loss
    {
        public abstract Tensor Build(Tensor targetOutput, Tensor output);
    }

    // This function can be used for any output being probability distribution (i.e. softmaxed)
    // https://gombru.github.io/2018/05/23/cross_entropy_loss/
    public class CategoricalCrossEntropy : Loss
    {
        public override Tensor Build(Tensor targetOutput, Tensor output)
        {
            using (tf.name_scope("categorical_cross_entropy"))
            {
                var axis = output.shape.Get(-1);
                var clippedOutput = tf._clip_by_value(output, tf.constant(Tools._EPSILON), tf.constant(1.0f - Tools._EPSILON));
                return tf.negative(tf.reduce_sum(tf.multiply(targetOutput, tf.log(clippedOutput)), axis));
            }
        }
    }

    // This function is also known as binary cross entropy and can be used for any sigmoided or softmaxed output (doesn't have to be probability distribution)
    public class CrossEntropy : Loss
    {
        public override Tensor Build(Tensor targetOutput, Tensor output)
        {
            using (tf.name_scope("cross_entropy"))
            {
                var axis = output.shape.Get(-1);
                var clippedOutput = tf._clip_by_value(output, tf.constant(Tools._EPSILON), tf.constant(1.0f - Tools._EPSILON));
                return tf.sub(tf.negative(tf.reduce_sum(tf.multiply(targetOutput, tf.log(clippedOutput)), axis)), 
                              tf.negative(tf.reduce_sum(tf.multiply(tf.sub(tf.constant(1.0f), targetOutput), tf.log(tf.sub(tf.constant(1.0f), clippedOutput))), axis)));
            }
        }
    }

    public class MeanSquareError : Loss
    {
        public override Tensor Build(Tensor targetOutput, Tensor output)
        {
            using (tf.name_scope("mean_square_error"))
                return tf.reduce_mean(tf.square(output - targetOutput));
        }
    }

    /*public class Huber : LossFunc
    {
        public Huber(float delta)
        {
            Delta = delta;
        }

        public override void Compute(Tensorflow.Tensor targetOutput, Tensorflow.Tensor output, Tensorflow.Tensor result)
        {
            targetOutput.Map((yTrue, y) => { float a = y - yTrue; return Math.Abs(a) <= Delta ? (0.5f * a * a) : (Delta * (float)Math.Abs(a) - 0.5f * Delta * Delta); }, output, result);
        }

        public override void Derivative(Tensorflow.Tensor targetOutput, Tensorflow.Tensor output, Tensorflow.Tensor result)
        {
            targetOutput.Map((yTrue, y) => { float a = y - yTrue; return Math.Abs(a) <= Delta ? a : (Delta * Tools.Sign(a)); }, output, result);
        }

        public readonly float Delta;
    }*/
}
