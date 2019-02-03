using System;
using System.Linq;

namespace Neuro
{
    public partial class np
    {
        public partial class Array
        {
            public float Max(int axis = -1)
            {
                if (axis != -1)
                    throw new NotImplementedException();

                return Data().Max();
            }
        }
    }
}
