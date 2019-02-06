using System;

namespace Neuro
{
    public partial class np
    {
        public partial class Array : ICloneable
        {
            public Array(float val = 0)
            {
                Storage = new Storage();
				Storage.SetData(val, 0);
            }

            public Array(System.Array values)
            {
                Storage = new Storage(values);
            }

            public Array(Shape shape)
            {
                Storage = new Storage(shape);
            }

            public override bool Equals(object obj)
            {
                if (obj is Array array)
                {
                    if (array.Storage.Shape == Storage.Shape && array.Data() == Data())
                        return true;
                }

                return false;
            }

            public override int GetHashCode()
            {
                var result = 1337;
                result = (result * 397) ^ NDim;
                result = (result * 397) ^ Size;
                return result;
            }

            public object Clone()
            {
                var clone = new Array();
                clone.Storage.Allocate(Storage.Shape, Storage.TensorLayout);
                clone.Storage.SetData(Storage.CloneData());
                return clone;
            }

            public float this[params int[] select]
            {
                get { return Storage.GetData(select); }
                set { Storage.SetData(value, select); }
            }

            public int[] Shape => Storage.Shape.Dimensions;
            public int[] Dims => Storage.Shape.Dimensions;
            public int[] Strides => Storage.Shape.DimOffset;
            public Shape GetShape() => Storage.Shape;
            public int NDim => Storage.Shape.NDim;
            public int Size => Storage.Shape.Size;
            public float[] Data() => Storage.GetData();
            public float[] CloneData() => Storage.CloneData();

            public Storage Storage { get; private set; }
        }
    }
}
