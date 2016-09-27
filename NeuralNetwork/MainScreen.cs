using OxyPlot;
using OxyPlot.Series;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Forms;

namespace NeuralNetwork
{
    public partial class MainScreen : Form
    {
        private static string URL = Path.GetDirectoryName(Path.GetDirectoryName(System.IO.Directory.GetCurrentDirectory())) + "\\DigitsSample\\";

        Network.Network network;

        public MainScreen()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            network = null;
            network = new Network.Network(double.Parse(textBox_learning.Text), double.Parse(textBox_momentum.Text), 30, int.Parse(textBox_hidden.Text), 10);
            trainNetwork(URL);

            plot();
            label_emq.Text = "" + Math.Round(network.ErrorList.Last(), 4);
        }

        private void plot()
        {
            var model = new PlotModel { Title = "" };
            LineSeries line = new LineSeries { Smooth = true };

            for (int i = 0; i < int.Parse(textBox_epoch.Text); i++)
                line.Points.Add(new DataPoint((i + 1), network.ErrorList.ElementAt(i)));

            model.Series.Add(line);
            plot1.Model = model;
        }

        private void printOutput()
        {
            textBox_output.Clear();
            textBox_output.AppendText("Neuron Output:" + "\n");
            for (int i = 0; i < network.GetOutputs().Count; i++)
                textBox_output.AppendText(GetOutputAsString(i) + " as " + Math.Round(network.GetOutputs().ElementAt(i), 8) + "\n");

            label_emq.Text = "" + Math.Round(network.ErrorList.Last(), 8);
        }

        private void trainNetwork(string trainLocation)
        {
            double[][] inputs = new double[][]{
            fileToArray(trainLocation + "A.txt"),
            fileToArray(trainLocation + "B.txt"),
            fileToArray(trainLocation + "C.txt"),
            fileToArray(trainLocation + "D.txt"),
            fileToArray(trainLocation + "E.txt"),
            fileToArray(trainLocation + "F.txt"),
            fileToArray(trainLocation + "G.txt"),
            fileToArray(trainLocation + "H.txt"),
            fileToArray(trainLocation + "I.txt"),
            fileToArray(trainLocation + "J.txt")
        };

            double[][] expectedOutputs = new double[][]{
            new double[] {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            new double[] {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
            new double[] {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
            new double[] {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
            new double[] {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
            new double[] {0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
            new double[] {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
            new double[] {0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
            new double[] {0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
            new double[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        };

            network.train(int.Parse(textBox_epoch.Text), inputs, expectedOutputs);

        }

        private double[] fileToArray(string path)
        {
            double[] digit = new double[30];
            int digitCurrentIndex = 0;

            if (!File.Exists(path))
                throw new FileNotFoundException(path);

            System.IO.StreamReader file = new System.IO.StreamReader(path);
            string line = file.ReadLine();

            while (line != null)
            {
                // Processa a linha
                double[] parsedLine = parseLine(line);

                foreach (double element in parsedLine)
                {
                    digit[digitCurrentIndex] = element;
                    digitCurrentIndex++;
                }

                line = file.ReadLine();
            }

            return digit;
        }

        private double[] parseLine(String line)
        {
            double[] digitLine = new double[5];
            string[] terms = line.Split(' ');

            for (int index = 0; index < terms.Length; index++)
            {
                if (terms[index].Equals("X"))
                    digitLine[index] = 1;
                else
                    digitLine[index] = 0;
            }

            return digitLine;
        }

        private void button2_Click(object sender, EventArgs e)
        {
            network.test(GetCheckBoxSelection());

            printOutput();

            List<double> output = network.GetOutputs();
            double max = Double.MinValue;
            int response = -1;

            for (int i = 0; i < output.Count; i++)
            {
                if (output.ElementAt(i) > max)
                {
                    max = output.ElementAt(i);
                    response = i;
                }
            }

            label_response.Text = GetOutputAsString(response);
        }

        private void button_clear_Click(object sender, EventArgs e)
        {
            foreach (CheckBox chkBox in groupBox2.Controls)
            {
                chkBox.Checked = false;
            }

            label_response.Text = "";
        }


        #region CheckBox
        private string GetOutputAsString(int outputResponse)
        {
            switch (outputResponse)
            {
                case 0:
                    return "A";
                case 1:
                    return "B";
                case 2:
                    return "C";
                case 3:
                    return "D";
                case 4:
                    return "E";
                case 5:
                    return "F";
                case 6:
                    return "G";
                case 7:
                    return "H";
                case 8:
                    return "I";
                case 9:
                    return "J";
                default:
                    return "NaN";
            }
        }

        private double[] GetCheckBoxSelection()
        {
            double[] checkboxInput = new double[]
{
                checkBox1.Checked ? 1 : 0,
                checkBox2.Checked ? 1 : 0,
                checkBox3.Checked ? 1 : 0,
                checkBox4.Checked ? 1 : 0,
                checkBox5.Checked ? 1 : 0,
                checkBox6.Checked ? 1 : 0,
                checkBox7.Checked ? 1 : 0,
                checkBox8.Checked ? 1 : 0,
                checkBox9.Checked ? 1 : 0,
                checkBox10.Checked ? 1 : 0,
                checkBox11.Checked ? 1 : 0,
                checkBox12.Checked ? 1 : 0,
                checkBox13.Checked ? 1 : 0,
                checkBox14.Checked ? 1 : 0,
                checkBox15.Checked ? 1 : 0,
                checkBox16.Checked ? 1 : 0,
                checkBox17.Checked ? 1 : 0,
                checkBox18.Checked ? 1 : 0,
                checkBox19.Checked ? 1 : 0,
                checkBox20.Checked ? 1 : 0,
                checkBox21.Checked ? 1 : 0,
                checkBox22.Checked ? 1 : 0,
                checkBox23.Checked ? 1 : 0,
                checkBox24.Checked ? 1 : 0,
                checkBox25.Checked ? 1 : 0,
                checkBox26.Checked ? 1 : 0,
                checkBox27.Checked ? 1 : 0,
                checkBox28.Checked ? 1 : 0,
                checkBox29.Checked ? 1 : 0,
                checkBox30.Checked ? 1 : 0
};
            return checkboxInput;
        }

        private void changeCheckBoxColor(CheckBox chk)
        {
            if (chk.Checked)
                chk.BackColor = Color.Red;
            else
                chk.BackColor = Control.DefaultBackColor;
        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox2_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox3_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox4_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox5_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox6_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox7_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox8_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox9_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox10_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox11_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox12_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox13_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox14_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox15_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox16_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox17_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox18_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox19_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox20_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox21_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox22_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox23_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox24_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox25_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox26_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox27_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox28_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox29_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        private void checkBox30_CheckedChanged(object sender, EventArgs e)
        {
            changeCheckBoxColor(sender as CheckBox);
        }

        #endregion

        private void toolStripButton1_Click(object sender, EventArgs e)
        {
            FolderBrowserDialog dialog = new FolderBrowserDialog();
            if (dialog.ShowDialog() == DialogResult.OK)
            {
                URL = dialog.SelectedPath;
            }

        }
    }
}
