using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;

namespace Neuro
{
	public static class Ext
	{
        public static string Str(this object obj)
        {
            if (obj == null)
                return "null";

            if (obj is IEnumerable)
            {
                var l = new List<string>();
                foreach (object o in (IEnumerable)obj)
                    l.Add(o.Str());

                return "[" + String.Join(", ", l.ToArray()) + "]";
            }
            if (obj is IDictionary)
            {
                var dict = obj as IDictionary;
                var l = new List<string>();
                foreach (object k in dict.Keys)
                    l.Add($"{k.Str()}: {dict[k].Str()}");

                return "{" + String.Join(", ", l.ToArray()) + "}";
            }

            return obj.ToString();
        }

        public static int[] GetShape(this Array array)
        {
            int[] shape = new int[array.Rank];
            for (int i = array.Rank - 1; i >= 0; --i)
                shape[i] = array.GetLength(i);
            return shape;
        }

        public static T[] RemoveAt<T>(this T[] array, int index)
        {
            T[] result = new T[array.Length - 1];
            for (int i = 0; i < index; ++i)
                result[i] = array[i];
            for (int i = index; i < result.Length; ++i)
                result[i] = array[i + 1];
            return result;
        }

        public static T Get<T>(this T[] array, int i)
        {
            return i < 0 ? array[i + array.Length] : array[i];
        }

        public static T[] Get<T>(this T[] array, int start, int end)
        {
            T[] result = new T[end - start];
            for (int i = start, n = 0; i < end; ++i, ++n)
                result[n] = array[i];
            return result;
        }

        public static Array GetEx(this Array source, int dimension, int[] indices)
        {
            int[] lengths = source.GetShape();
            lengths[dimension] = indices.Length;

            Type type = source.GetType().GetElementType();
            Array result = Array.CreateInstance(type, lengths);

            for (int i = 0; i < indices.Length; i++)
                SetEx(result, dimension, i, GetEx(source, dimension, indices[i]));

            return result;
        }

        public static Array GetEx(this Array source, int dimension, int index)
        {
            return GetEx(source, dimension, index, index + 1);
        }

        public static Array GetEx(this Array array, int dimension, int start, int end)
        {
            if (dimension != 0)
                throw new NotImplementedException();

            int[] length = array.GetShape();
            length = length.RemoveAt(dimension);
            int rows = end - start;
            if (length.Length == 0)
                length = new [] { rows };

            Type type = array.GetType().GetElementType();
            Array result = Array.CreateInstance(type, length);
            int rowSize = array.Length / array.GetLength(dimension);
#pragma warning disable CS0618 // Type or member is obsolete
            System.Buffer.BlockCopy(array, start * rowSize * Marshal.SizeOf(type), result, 0, rows * rowSize * Marshal.SizeOf(type));
#pragma warning restore CS0618 // Type or member is obsolete
            return result;
        }

        public static void SetEx(this Array destination, int dimension, int index, Array value)
        {
            SetEx(destination, dimension, index, index + 1, value);
        }

        public static void SetEx(this Array destination, int dimension, int start, int end, Array value)
        {
            if (dimension != 0)
                throw new NotImplementedException();

            Type type = destination.GetType().GetElementType();
            int rowSize = destination.Length / destination.GetLength(0);
            int length = end - start;
#pragma warning disable CS0618 // Type or member is obsolete
            System.Buffer.BlockCopy(value, 0, destination, start * rowSize * Marshal.SizeOf(type), length * rowSize * Marshal.SizeOf(type));
#pragma warning restore CS0618 // Type or member is obsolete
        }

        public static int Product(this IEnumerable<int> array)
        {
            int result = 1;
            foreach (var v in array)
                result *= v;
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
    }
}
