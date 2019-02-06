using System;
using System.Diagnostics;

namespace Neuro.PerfTests
{
    class NpArrayPerfTests
    {
        static void Main(string[] args)
        {
            //var x = np.array(new float[] {1, 2, 3, 4});            
            var x = new np.Array(new[,,]{{{1,2},{3,4}},{ { 5, 6 }, { 7, 8 } } });
            var y = np.array(new[,] {{1, 0}, {0, 1}});

            Trace.WriteLine(x.dot(y));
        }
    }
}
