﻿using System;
using System.Globalization;
using System.Linq;

namespace Neuro
{
    public partial class np
    {
        public partial class Array
        {
            public override string ToString()
            {
                return ToStringRecursive(this, new int[]{}, " ", 75, new Dragon4FloatFormatter(Data()));
            }

            private class Dragon4FloatFormatter
            {
	            public Dragon4FloatFormatter(float[] a)
	            {
		            for (int i = 0; i < a.Length; ++i)
		            {
			            string[] s = a[i].ToString("0.######", CultureInfo.InvariantCulture).Split('.');
			            PadLeft = Math.Max(PadLeft, s[0].Length);
			            PadRight = Math.Max(PadRight, s.Length > 1 ? s[1].Length : 0);
					}
	            }

	            public string FormatValue(float v)
	            {
					string[] s = v.ToString("0.######", CultureInfo.InvariantCulture).Split('.');
					return s[0].PadLeft(PadLeft) + '.' + (s.Length > 1 ? s[1] : "").PadRight(PadRight);
	            }

	            private int PadLeft;
	            private int PadRight;
            }

			private static string ToStringRecursive(Array a, int[] index, string hanging_indent, int curr_width, Dragon4FloatFormatter formatter)
			{
				string separator = " ";

				int axis = index.Length;
				int axes_left = a.NDim - axis;

				if (axes_left == 0)
					return formatter.FormatValue(a[index]);

				// when recursing, add a space to align with the [ added, and reduce the
				// length of the line by 1
				string next_hanging_indent = hanging_indent + ' ';
				int next_width = curr_width - 1;


				int a_len = a.Shape[axis];
				int leading_items = 0;
				int trailing_items = a_len;

				// stringify the array with the hanging indent on the first line too
				string s = "";

				// last axis (rows) - wrap elements if they would not fit on one line
				if (axes_left == 1)
				{
					// the length up until the beginning of the separator / bracket
					int elem_width = curr_width - 1;

					string line = hanging_indent;
					string word = "";

					for (int i = 0; i < leading_items; ++i)
					{
						word = ToStringRecursive(a, index.Concat(new[] {i}).ToArray(), next_hanging_indent, next_width, formatter);
						(s, line) = ExtendLine(s, line, word, elem_width, hanging_indent);
						line += separator;
					}

					for (int i = trailing_items; i > 1; --i)
					{
						word = ToStringRecursive(a, index.Concat(new[] {-i}).ToArray(), next_hanging_indent, next_width, formatter);
						(s, line) = ExtendLine(s, line, word, elem_width, hanging_indent);
						line += separator;
					}

					word = ToStringRecursive(a, index.Concat(new[] {-1}).ToArray(), next_hanging_indent, next_width, formatter);
					(s, line) = ExtendLine(s, line, word, elem_width, hanging_indent);

					s += line;
				}
				// other axes - insert newlines between rows
				else
				{
					string line_sep = separator;
					for (int i = 0; i < axes_left - 1; ++i)
						line_sep += '\n';
					string nested = "";

					for (int i = 0; i < leading_items; ++i)
					{
						nested = ToStringRecursive(a, index.Concat(new[] { i }).ToArray(), next_hanging_indent, next_width, formatter);
						s += hanging_indent + nested + line_sep;
					}

					for (int i = trailing_items; i > 1; --i)
					{
						nested = ToStringRecursive(a, index.Concat(new[] { -i }).ToArray(), next_hanging_indent, next_width, formatter);
						s += hanging_indent + nested + line_sep;
					}

					nested = ToStringRecursive(a, index.Concat(new[] { -1 }).ToArray(), next_hanging_indent, next_width, formatter);

					s += hanging_indent + nested;
				}

				// remove the hanging indent, and wrap in []
				s = '[' + s.Substring(hanging_indent.Length) + ']';

				return s;
			}

			private static (string, string) ExtendLine(string s, string line, string word, int line_width, string next_line_prefix)
			{
				bool needs_wrap = (line.Length + word.Length) > line_width;
				if (line.Length <= next_line_prefix.Length)
					needs_wrap = false;

				if (needs_wrap)
				{
					s += line.TrimEnd() + "\n";
					line = next_line_prefix;
				}

				line += word;
				return (s, line);
			}
        }
    }
}