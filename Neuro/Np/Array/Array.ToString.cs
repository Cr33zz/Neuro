namespace Neuro
{
    public partial class np
    {
        public partial class Array
        {
            public override string ToString()
            {
                string s = "";
                ToStringRecursive(s, Shape, 0, Max().ToString().Length);                
                return s;
            }

            private void ToStringRecursive(string str, int[] indices, int axisIndex, int valuePad)
            {
                bool lastAxis = axisIndex == Shape.Length - 1;
                int axisLen = Shape[axisIndex];

                for (int i = 0; i < axisLen; ++i)
                {
                    if (lastAxis)
                    {
                        str += this[indices].ToString().PadLeft(valuePad) + (i == axisLen - 1 ? "" : ", ");
                    }
                    else
                    {
                        str += "[".PadLeft(axisIndex + 1, ' ');
                    }

                    indices[axisIndex] = i;
                }
            }
        }
    }
}
