using System;
using System.Diagnostics;

namespace Neuro.PerfTests
{
    class NpArrayPerfTests
    {
        static void Main(string[] args)
        {
            //var x = np.array(new float[] {1, 2, 3, 4});            
            var x = new np.Array(new [,,,]{{{{1.45,2,1},                
                              {3,4,1}},
                              {{1,7,1},                
                              {3,49.6,1}},
                              {{1,2,1},                
                              {3,4,1}},
                              {{1,7,1},                
                              {3,49,1}}},
                              {{{1,2,1},                
                              {3,4,1}},
                              {{1,7,1},                
                              {3,49,1}},
                              {{1,2,1},                
                              {3,4,1}},
                              {{1,7,1},                
                              {3,49,1}}}});
            Trace.WriteLine(x.GetShape());
            //Trace.WriteLine(x);
            var y = np.zeros(2, 3, 4);
            Trace.WriteLine(y.GetShape());
            
        }
    }
}
