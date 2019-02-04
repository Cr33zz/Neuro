using System;
using System.Collections.Generic;
using System.Linq;

namespace Neuro
{
    public partial class np
    {
        public class Shape
        {
            public Shape(params int[] dims)
            {
                ReShape(dims);
            }

            public Shape(IEnumerable<int> shape)
                : this(shape.ToArray())
            {
            }

            public int[] Dimensions { get; private set; }
            public int[] DimOffset { get; private set; }
            public int Size { get; private set; }
            // Row (0) or column (1) wise order
            public int TensorLayout { get; private set; }
            public int NDim => Dimensions.Length;

            public (int, int) BiShape => Dimensions.Length == 2 ? (Dimensions[0], Dimensions[1]) : (0, 0);
            public (int, int, int) TriShape => Dimensions.Length == 3 ? (Dimensions[0], Dimensions[1], Dimensions[2]) : (0, 0, 0);

            public static implicit operator int(Shape shape) => shape.Size;
            public static implicit operator (int, int) (Shape shape) => shape.Dimensions.Length == 2 ? (shape.Dimensions[0], shape.Dimensions[1]) : (0, 0);
            public static implicit operator (int, int, int) (Shape shape) => shape.Dimensions.Length == 3 ? (shape.Dimensions[0], shape.Dimensions[1], shape.Dimensions[2]) : (0, 0, 0);
            public static implicit operator int[] (Shape shape) => shape.Dimensions;
            public static implicit operator Shape(int[] dims) => new Shape(dims);
            public static implicit operator Shape(int dim) => new Shape(dim);

            public int GetIndexInShape(params int[] select)
            {
                int idx = 0;
                for (int i = 0; i < select.Length; i++)
                {
                    idx += DimOffset[i] * (select[i] < 0 ? Dimensions[i] + select[i] : select[i]);
                }

                return idx;
            }

            public int[] GetDimIndexOutShape(int select)
            {
                int[] dimIndexes = null;
                if (this.DimOffset.Length == 1)
                    dimIndexes = new[] {select};
                else if (TensorLayout == 1)
                {
                    int counter = select;
                    dimIndexes = new int[DimOffset.Length];

                    for (int idx = DimOffset.Length - 1; idx > -1; idx--)
                    {
                        dimIndexes[idx] = counter / DimOffset[idx];
                        counter -= dimIndexes[idx] * DimOffset[idx];
                    }
                }
                else
                {
                    int counter = select;
                    dimIndexes = new int[DimOffset.Length];

                    for (int idx = 0; idx < DimOffset.Length; idx++)
                    {
                        dimIndexes[idx] = counter / DimOffset[idx];
                        counter -= dimIndexes[idx] * DimOffset[idx];
                    }
                }

                return dimIndexes;
            }

            public void ChangeTensorLayout(int layout)
            {
                DimOffset = new int[Dimensions.Length];

                layout = (layout == 0) ? 1 : layout;

                TensorLayout = layout;
                SetDimOffset();
            }

            public void ReShape(params int[] dims)
            {
                Dimensions = dims;
                DimOffset = new int[Dimensions.Length];
                TensorLayout = 1;

                Size = 1;

                foreach (int dimSize in dims)
                    Size *= dimSize;

                SetDimOffset();
            }

            protected void SetDimOffset()
            {
                if (Dimensions.Length == 0)
                {

                }
                else
                {
                    if (TensorLayout == 1)
                    {
                        DimOffset[0] = 1;

                        for (int idx = 1; idx < DimOffset.Length; idx++)
                            DimOffset[idx] = DimOffset[idx - 1] * Dimensions[idx - 1];
                    }
                    else if (TensorLayout == 2)
                    {
                        DimOffset[DimOffset.Length - 1] = 1;
                        for (int idx = DimOffset.Length - 1; idx >= 1; idx--)
                            DimOffset[idx - 1] = DimOffset[idx] * Dimensions[idx];
                    }
                }
            }

            public static bool operator ==(Shape a, Shape b)
            {
                if (b is null)
                    return false;
                return a.Dimensions.SequenceEqual(b.Dimensions);
            }

            public static bool operator !=(Shape a, Shape b)
            {
                return !(a == b);
            }

            public override bool Equals(object obj)
            {
                if (obj is Shape s)
                    return this == s;
                return false;
            }

            public override int GetHashCode()
            {
                return base.GetHashCode();
            }

            public override string ToString()
            {
                return "(" + String.Join(", ", Dimensions) + ")";
            }
        }
    }
}
