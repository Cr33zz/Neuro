namespace Neuro
{
    public partial class np
    {
        public static Array zeros(params int[] dims) => new Array(new Shape(dims));

        public static Array ones(params int[] dims)
        {
            var result = new Array(new Shape(dims));
            var dataArr = result.Data();
            for (int i = 0; i < dataArr.Length; ++i)
                dataArr[i] = 1;
            return result;
        }

        public static Array array(System.Array values)
        {
            return new Array(values);
        }

		public static Array array(float val)
		{
			return new Array(val);
		}

		// Returns a reshaped copy of a.
		public static Array reshape(Array a, params int[] dims)
        {
            var result = (Array)a.Clone();
            return result.Reshape(dims);
        }

        //Return a contiguous flattened array.
        public static Array ravel(Array a)
        {
            var result = (Array)a.Clone();
            return result.Ravel();
        }
    }
}
