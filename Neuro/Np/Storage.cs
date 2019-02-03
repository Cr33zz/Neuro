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
                Shape = new Shape(1);
                TensorLayout = 1;
            }

            public Storage(System.Array values)
            {
                Allocate(values, 1);
                SetData(ToFloatArray(values));
            }

            public Storage(Shape shape)
            {
                Allocate(shape, 1);
            }

            public void Allocate(Shape shape, int tensorOrder = 1)
            {
                Shape = shape;
                Shape.ChangeTensorLayout(tensorOrder);
                Values = new float[Shape.Size];
                TensorLayout = tensorOrder;
            }

            public void Allocate(System.Array values, int tensorOrder = 1)
            {
                int[] dims = new int[values.Rank];
                for (int i = 0; i < dims.Length; ++i)
                    dims[i] = values.GetLength(i);

                Allocate(new Shape(dims), tensorOrder);
            }

            public Storage GetColumWiseStorage()
            {
                if (TensorLayout != 2)
                    ChangeRowToColumnLayout();

                return this;
            }
            public Storage GetRowWiseStorage()
            {
                if (TensorLayout != 1)
                    ChangeColumnToRowLayout();

                return this;
            }

            public float[] GetData()
            {
                return Values;
            }

            public float[] CloneData()
            {
                return (float[])Values.Clone();
            }

            public T[] GetData<T>()
            {
                return (Values as T[]);
            }
            
            public T[] CloneData<T>()
            {
                var clone = CloneData();
                return (clone as T[]);
            }
            
            public float GetData(params int[] indexes)
            {
                float element;
                if (indexes.Length == Shape.NDim)
                    element = Values[Shape.GetIndexInShape(indexes)];
                else if (Shape.Dimensions.Last() == 1)
                    element = Values[Shape.GetIndexInShape(indexes)];
                else if (indexes.Length == 1)
                {
                    element = Values[indexes[0]];
                }
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
            
            public void SetNewShape(params int[] dimensions)
            {
                Shape = new Shape(dimensions);
            }

            public void Reshape(params int[] dimensions)
            {
                if (TensorLayout == 2)
                {
                    Shape = new Shape(dimensions);
                }
                else
                {
                    ChangeTensorLayout(2);
                    Shape = new Shape(dimensions);
                    Shape.ChangeTensorLayout(2);
                    ChangeTensorLayout(1);
                }
            }

            public object Clone()
            {
                var clone = new Storage();
                clone.Allocate(new Shape(Shape.Dimensions), TensorLayout);
                clone.SetData((float[])Values.Clone());
                return clone;
            }

            public void ChangeTensorLayout(int layout)
            {
                if (layout != TensorLayout)
                    if (TensorLayout == 1)
                        ChangeRowToColumnLayout();
                    else
                        ChangeColumnToRowLayout();
            }

            protected void ChangeRowToColumnLayout()
            {
                if (Shape.NDim == 1)
                {

                }
                else
                {
                    var puffer = new float[Values.Length];

                    var pufferShape = new Shape(Shape.Dimensions);
                    pufferShape.ChangeTensorLayout(2);

                    for (int idx = 0; idx < Values.Length; idx++)
                        puffer[pufferShape.GetIndexInShape(Shape.GetDimIndexOutShape(idx))] = Values[idx];

                    Values = puffer;
                }

                Shape.ChangeTensorLayout(2);
                TensorLayout = 2;
            }
            protected void ChangeColumnToRowLayout()
            {
                if (Shape.NDim == 1)
                {

                }
                else
                {
                    var puffer = new float[Values.Length];

                    var pufferShape = new Shape(Shape.Dimensions);
                    pufferShape.ChangeTensorLayout(1);

                    for (int idx = 0; idx < Values.Length; idx++)
                        puffer[pufferShape.GetIndexInShape(Shape.GetDimIndexOutShape(idx))] = Values[idx];

                    Values = puffer;
                }

                Shape.ChangeTensorLayout(1);
                TensorLayout = 1;
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
            public int TensorLayout { get; private set; }
            private float[] Values;
        }
    }
}
