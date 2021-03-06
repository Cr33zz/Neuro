﻿using System;
using System.Drawing;
using System.Linq;
using System.Windows.Forms.DataVisualization.Charting;

namespace Neuro
{
    public class ChartGenerator
    {
        public ChartGenerator(string outputFileName, string title = "", string xAxisLabel = "")
        {
            OutputFileName = $"{outputFileName}_{DateTime.Now.ToString("ddMMyyy_HHmm")}";
            ChartArea.AxisX.Title = xAxisLabel;
            ChartArea.AxisX.TitleFont = new Font(Chart.Font.Name, 11);
            ChartArea.AxisX.LabelStyle.Format = "#";
            ChartArea.AxisX.IsStartedFromZero = false;
            ChartArea.AxisY2.IsStartedFromZero = false;
            ChartArea.AxisY.IsStartedFromZero = false;
            Chart.ChartAreas.Add(ChartArea);
            Chart.Legends.Add(Legend);
            Legend.Font = new Font(Chart.Font.Name, 11);
            Chart.Width = 1000;
            Chart.Height = 600;
            Chart.Titles.Add(new Title(title, Docking.Top, new Font(Chart.Font.Name, 13), Color.Black));
        }

        public void AddSeries(int id, string label, Color color, bool useSecondaryAxis = false)
        {
            Series s = new Series(id.ToString());
            s.ChartType = SeriesChartType.Line;
            s.BorderWidth = 2;
            s.Color = color;
            s.LegendText = label;
            s.Legend = "leg";
            s.IsVisibleInLegend = true;
            Chart.Series.Add(s);

            if (useSecondaryAxis)
            {
                ChartArea.AxisY2.Enabled = AxisEnabled.True;
                s.YAxisType = AxisType.Secondary;
            }
        }

        public void AddData(float x, float h, int seriesId)
        {
            if (Chart.Series.IndexOf(seriesId.ToString()) == -1)
                return;

            Chart.Series[seriesId.ToString()].Points.AddXY(x, h);
            DataMinX = Math.Min(DataMinX, x);
            DataMaxX = Math.Max(DataMaxX, x);
        }

        public void Save()
        {
            ChartArea.AxisX.Minimum = DataMinX;
            ChartArea.AxisX.Maximum = DataMaxX;
            Chart.SaveImage($"{OutputFileName}.png", ChartImageFormat.Png);
        }

        public readonly string OutputFileName;
        private Chart Chart = new Chart();
        private ChartArea ChartArea = new ChartArea();
        private Legend Legend = new Legend("leg");
        private float DataMinX = float.MaxValue;
        private float DataMaxX = float.MinValue;
    }
}
