using System;
using System.Collections.Generic;
using System.Linq;
using Mono.CSharp;

namespace Neuro
{
    public partial class np
    {
        public class Storage
        {
            public Storage()
            {
                Values = new float[1];
                Shape = new Shape();
            }

            public Storage(System.Array values)
            {
                Allocate(values);
                SetData(ToFloatArray(values));
            }

            public Storage(Shape shape)
            {
                Allocate(shape);
            }

            public void Allocate(Shape shape)
            {
                Shape = shape;
                Values = new float[Shape.Size];
            }

            public void Allocate(System.Array values)
            {
                int[] dims = new int[values.Rank];
                for (int i = 0; i < dims.Length; ++i)
                    dims[i] = values.GetLength(i);

                Allocate(new Shape(dims));
            }

            public float[] GetData()
            {
                return Values;
            }

            public float[] CloneData()
            {
                return (float[])Values.Clone();
            }

            public float GetData(params int[] indexes)
            {
                float element;

                if (indexes.Length == Shape.NDim)
                    element = Values[Shape.GetIndexInShape(indexes)];
                else if (Shape.Dimensions.Length == 0)
	                element = Values[0];
				else if (Shape.Dimensions.Last() == 1)
					element = Values[Shape.GetIndexInShape(indexes)];
                else if (indexes.Length == 1)
                    element = Values[indexes[0]];
                else
                    throw new Exception("indexes must be equal to number of dimension.");
                return element;
            }

            public void SetData(float[] values)
            {
                Values = values;
            }

            public void SetData(float value, params int[] indexes)
            {
                Values[Shape.GetIndexInShape(indexes)] = value;
            }
            
            public void Reshape(params int[] dimensions)
            {
                Shape = new Shape(dimensions);                
            }

            public object Clone()
            {
                var clone = new Storage();
                clone.Allocate(new Shape(Shape.Dimensions));
                clone.SetData((float[])Values.Clone());
                return clone;
            }

            public static float[] ToFloatArray(System.Array array)
            {
                var newValues = new List<float>();
                var dimensionSizes = Enumerable.Range(0, array.Rank).Select(i => array.GetLength(i)).ToArray();
                ToFloatArrayRecursive(dimensionSizes, newValues, new int[] { }, array);
                return newValues.ToArray();
            }

            // special thanks to https://stackoverflow.com/a/47148145
            private static void ToFloatArrayRecursive(int[] dimensionSizes, List<float> outArray, int[] externalCoordinates, System.Array masterArray)
            {
                if (dimensionSizes.Length == 1)
                {
                    for (int i = 0; i < dimensionSizes[0]; i++)
                    {
                        var globalCoordinates = externalCoordinates.Concat(new[] {i}).ToArray();
                        outArray.Add(Convert.ToSingle(masterArray.GetValue(globalCoordinates)));
                    }
                }
                else
                {
                    for (int i = 0; i < dimensionSizes[0]; i++)
                        ToFloatArrayRecursive(dimensionSizes.Skip(1).ToArray(), outArray, externalCoordinates.Concat(new[] { i }).ToArray(), masterArray);
                }
            }

            public Shape Shape { get; private set; }
            private float[] Values;
        }
    }
}
