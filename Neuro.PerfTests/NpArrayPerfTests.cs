using System;
using System.Diagnostics;

namespace Neuro.PerfTests
{
    class NpArrayPerfTests
    {
        static void Main(string[] args)
        {
	        var s = new np.Shape(3, 3, 3, 3);
			var t = new np.Array(s);
			Trace.WriteLine(string.Join(",", t.Strides));
            var x = new np.Array(new float[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } });
            Trace.WriteLine(string.Join(",", x.Strides));
			var y = np.array(new float[,] { { 1, 0 }, { 0, 1 } });
            var z = np.array(new float[] { 1, 0 });
            var v = np.array(2);

			//Trace.WriteLine(x.dot(v));
			//Trace.WriteLine(x.T);
			Trace.WriteLine(x + y);
			Trace.WriteLine(string.Join(",", x.T.Strides));
		}
    }
}
