using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;

namespace Neuro
{
	public static class Ext
	{
        public static T Get<T>(this T[] array, int i)
        {
            return i < 0 ? array[i + array.Length] : array[i];
        }

        public static T[] Get<T>(this T[] array, int start, int end)
        {
            T[] result = (T[])Array.CreateInstance(typeof(T), end - start + 1);
            for (int i = start, n = 0; i < end; ++i, ++n)
                result[n] = array[i];
            return result;
        }

        public static int Product(this int[] array)
        {
            int result = 1;
            for (int i = 0; i < array.Length; ++i)
                result *= array[i];
            return result;
        }

        public static float Product(this float[] array)
        {
            float result = 1;
            for (int i = 0; i < array.Length; ++i)
                result *= array[i];
            return result;
        }

        public static int Sum(this int[] array)
        {
            int result = 0;
            for (int i = 0; i < array.Length; ++i)
                result += array[i];
            return result;
        }

        public static float Sum(this float[] array)
        {
            float result = 0;
            for (int i = 0; i < array.Length; ++i)
                result += array[i];
            return result;
        }

        // Returns range(0, rank(x)) if axis is null
        public static TFOutput ReduceDims(this TFGraph g, TFOutput input, TFOutput? axis = null)
        {
            if (axis.HasValue)
                return axis.Value;

            long[] shape = g.GetTensorShape(input).ToArray();
            if (shape.Length >= 0)
            {
                var array = new int[shape.Length];
                for (int i = 0; i < array.Length; i++)
                    array[i] = i;

                return g.Const(array, TFDataType.Int32);
            }
            return g.Range(g.Const(0), g.Const(shape.Length), g.Const(1));
        }
    }
}
