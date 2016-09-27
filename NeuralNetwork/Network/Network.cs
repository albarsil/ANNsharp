using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Network
{
    public class Network
    {
        private static readonly double DEFAULT_LEARNING_RATE = 0.2;
        public static readonly Random Random = new Random();

        private double learningRate = DEFAULT_LEARNING_RATE;
        private Layer inputLayer;
        private Layer hiddenLayer;
        private Layer outputLayer;
        private double momentum;
        public List<double> ErrorList { get; set; }

        public Network(int seed, double learningRate, double momentum, int inputSize, int hiddenNeurons, int outputSize)
        {
            this.learningRate = learningRate;
            this.momentum = momentum;
            this.mountNetwork(inputSize, outputSize, hiddenNeurons);
            ErrorList = new List<double>();
        }

        public Network(double learningRate, double momentum, int inputSize, int hiddenNeurons, int outputSize)
        {
            this.learningRate = learningRate;
            this.momentum = momentum;
            this.mountNetwork(inputSize, outputSize, hiddenNeurons);
            ErrorList = new List<double>();
        }

        public double getMomentum()
        {
            return this.momentum;
        }

        public void setMomentum(double momentum)
        {
            this.momentum = momentum;
        }

        public void train(int epochs, double[][] inputs, double[][] expectedOutputs)
        {

            int printEpochs = epochs;
            ErrorList.Clear();

            while (epochs > 0)
            {
                double error = 0;
                for (int indexSamples = 0; indexSamples < inputs.Length; indexSamples++)
                {
                    double trainingError = train(inputs[indexSamples], expectedOutputs[indexSamples]);

                    error += trainingError;

                    backpropagate(expectedOutputs[indexSamples]);
                }

                ErrorList.Add((error / inputs.Length));
                epochs--;
            }
        }

        private void printInterval(double[] interval, double[] expected)
        {
            Console.WriteLine("Sample elements [");
            for (int sampleIndex = 0; sampleIndex < interval.Length; sampleIndex++)
                if (sampleIndex == (interval.Length - 1))
                    Console.WriteLine(interval[sampleIndex]);
                else
                    Console.WriteLine(interval[sampleIndex] + ", ");
            Console.WriteLine("]");

            Console.WriteLine("\t");

            Console.WriteLine(" Expected [");
            for (int sampleIndex = 0; sampleIndex < expected.Length; sampleIndex++)
                if (sampleIndex == (expected.Length - 1))
                    Console.WriteLine(expected[sampleIndex]);
                else
                    Console.WriteLine(expected[sampleIndex] + ", ");
            Console.WriteLine("]");
        }

        public double[] test(double[] inputs)
        {
            for (int inputNeurons = 0; inputNeurons < inputLayer.getNeurons().Length; inputNeurons++)
                inputLayer.getNeurons()[inputNeurons].setOutput(inputs[inputNeurons]);

            double[] hiddenNeuronOutput = new double[hiddenLayer.getNeurons().Length];

            for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < hiddenLayer.getNeurons().Length; hiddenNeuronIndex++)
            {
                hiddenLayer.getNeurons()[hiddenNeuronIndex].activeNeuron(inputs);
                hiddenNeuronOutput[hiddenNeuronIndex] = hiddenLayer.getNeurons()[hiddenNeuronIndex].getOutput();
            }

            for (int outputNeuronIndex = 0; outputNeuronIndex < outputLayer.getNeurons().Length; outputNeuronIndex++)
                outputLayer.getNeurons()[outputNeuronIndex].activeNeuron(hiddenNeuronOutput);

            double[] response = new double[outputLayer.getNeurons().Length];

            for (int outputNeuronIndex = 0; outputNeuronIndex < outputLayer.getNeurons().Length; outputNeuronIndex++)
                response[outputNeuronIndex] = outputLayer.getNeurons()[outputNeuronIndex].getOutput();

            return response;
        }

        private double train(double[] inputs, double[] expectedOutputs)
        {
            for (int inputNeurons = 0; inputNeurons < inputLayer.getNeurons().Length; inputNeurons++)
                inputLayer.getNeurons()[inputNeurons].setOutput(inputs[inputNeurons]);

            double[] hiddenNeuronOutput = new double[hiddenLayer.getNeurons().Length];

            for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < hiddenLayer.getNeurons().Length; hiddenNeuronIndex++)
            {
                hiddenLayer.getNeurons()[hiddenNeuronIndex].activeNeuron(inputs);
                hiddenNeuronOutput[hiddenNeuronIndex] = hiddenLayer.getNeurons()[hiddenNeuronIndex].getOutput();
            }

            for (int outputNeuronIndex = 0; outputNeuronIndex < outputLayer.getNeurons().Length; outputNeuronIndex++)
                outputLayer.getNeurons()[outputNeuronIndex].activeNeuron(hiddenNeuronOutput);

            double error = 0;

            for (int outputsErrorIndex = 0; outputsErrorIndex < outputLayer.getNeurons().Length; outputsErrorIndex++)
                error += 0.5 * ((expectedOutputs[outputsErrorIndex] - outputLayer.getNeurons()[outputsErrorIndex].getOutput()) * (expectedOutputs[outputsErrorIndex] - outputLayer.getNeurons()[outputsErrorIndex].getOutput()));

            return error;
        }

        private void backpropagate(double[] expectedOutputs)
        {
            calculateDeltaForOutput(expectedOutputs);
            calculateDeltaForHidden(expectedOutputs);
            calculateHiddenWeights();
            calculateOutputWeights();
        }

        private void calculateDeltaForOutput(double[] expectedOutputs)
        {
            for (int outputIndex = 0; outputIndex < outputLayer.getNeurons().Length; outputIndex++)

                outputLayer.getNeurons()[outputIndex].setDelta(
                        (expectedOutputs[outputIndex] - outputLayer.getNeurons()[outputIndex].getOutput())
                        * outputLayer.getNeurons()[outputIndex].getOutput()
                        * (1 - outputLayer.getNeurons()[outputIndex].getOutput())
                        );
        }

        private void calculateDeltaForHidden(double[] expectedOutputs)
        {

            for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < hiddenLayer.getNeurons().Length; hiddenNeuronIndex++)
            {
                double weightSomatory = 0;

                for (int outputNeuronIndex = 0; outputNeuronIndex < outputLayer.getNeurons().Length; outputNeuronIndex++)
                {
                    weightSomatory += hiddenLayer.getNeurons()[hiddenNeuronIndex].getOutputConnections()[outputNeuronIndex].getWeight() * outputLayer.getNeurons()[outputNeuronIndex].getDelta();
                }

                double delta = hiddenLayer.getNeurons()[hiddenNeuronIndex].getOutput()
                        * (1 - hiddenLayer.getNeurons()[hiddenNeuronIndex].getOutput())
                        * weightSomatory;

                hiddenLayer.getNeurons()[hiddenNeuronIndex].setDelta(delta);
            }
        }

        private void calculateHiddenWeights()
        {
            for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < hiddenLayer.getNeurons().Length; hiddenNeuronIndex++)
            {
                double deltaBias = learningRate * hiddenLayer.getNeurons()[hiddenNeuronIndex].getDelta() + momentum * hiddenLayer.getNeurons()[hiddenNeuronIndex].getBiasDelta();
                hiddenLayer.getNeurons()[hiddenNeuronIndex].setBiasDelta(deltaBias);

                hiddenLayer.getNeurons()[hiddenNeuronIndex].setBias(hiddenLayer.getNeurons()[hiddenNeuronIndex].getBias() + deltaBias);

                for (int inputNeuronIndex = 0; inputNeuronIndex < inputLayer.getNeurons().Length; inputNeuronIndex++)
                {
                    double deltaConnection = 
                        learningRate * inputLayer.getNeurons()[inputNeuronIndex].getOutput() * hiddenLayer.getNeurons()[hiddenNeuronIndex].getDelta()
                        + momentum * inputLayer.getNeurons()[inputNeuronIndex].getOutputConnections()[hiddenNeuronIndex].getDelta();

                    inputLayer.getNeurons()[inputNeuronIndex].getOutputConnections()[hiddenNeuronIndex].setDelta(deltaConnection);

                    inputLayer.getNeurons()[inputNeuronIndex].getOutputConnections()[hiddenNeuronIndex].setWeight(inputLayer.getNeurons()[inputNeuronIndex].getOutputConnections()[hiddenNeuronIndex].getWeight() + deltaConnection);
                }
            }
        }

        private void calculateOutputWeights()
        {
            for (int outputNeuronIndex = 0; outputNeuronIndex < outputLayer.getNeurons().Length; outputNeuronIndex++)
            {
                double deltaBias = learningRate * outputLayer.getNeurons()[outputNeuronIndex].getDelta() + momentum * outputLayer.getNeurons()[outputNeuronIndex].getBiasDelta();
                outputLayer.getNeurons()[outputNeuronIndex].setBiasDelta(deltaBias);

                outputLayer.getNeurons()[outputNeuronIndex].setBias(outputLayer.getNeurons()[outputNeuronIndex].getBias() + deltaBias);

                for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < hiddenLayer.getNeurons().Length; hiddenNeuronIndex++)
                {

                    double deltaConnection = learningRate * hiddenLayer.getNeurons()[hiddenNeuronIndex].getOutput() * outputLayer.getNeurons()[outputNeuronIndex].getDelta() 
                        + momentum * hiddenLayer.getNeurons()[hiddenNeuronIndex].getOutputConnections()[outputNeuronIndex].getDelta();

                    hiddenLayer.getNeurons()[hiddenNeuronIndex].getOutputConnections()[outputNeuronIndex].setDelta(deltaConnection);

                    hiddenLayer.getNeurons()[hiddenNeuronIndex].getOutputConnections()[outputNeuronIndex].setWeight(hiddenLayer.getNeurons()[hiddenNeuronIndex].getOutputConnections()[outputNeuronIndex].getWeight() + deltaConnection);
                }
            }
        }

        public List<double> GetOutputs()
        {
            List<double> output = new List<double>();
            for (int outputNeuronIndex = 0; outputNeuronIndex < outputLayer.getNeurons().Length; outputNeuronIndex++)
                output.Add(outputLayer.getNeurons()[outputNeuronIndex].getOutput());

            return output;
        }

        private void mountNetwork(int inputNeuronSize, int outputNeuronSize, int hiddenNeuronSize)
        {
            this.hiddenLayer = new Layer(hiddenNeuronSize, outputNeuronSize);
            this.outputLayer = new Layer(outputNeuronSize);
            this.inputLayer = new Layer(inputNeuronSize, hiddenNeuronSize);

            for (int i = 0; i < inputNeuronSize; i++)
                inputLayer.getNeurons()[i].setInputConnections(null);

            // Configura as conexões de entrada para a camada da rede oculta
            configureNeuronConnections(inputLayer, hiddenLayer);

            // Configura as conexões da camada oculta para a camada de saída

            configureNeuronConnections(hiddenLayer, outputLayer);
        }

        private void configureNeuronConnections(Layer previousLayer, Layer nextLayer)
        {
            // Percorre todos os neuronios da proxima camada
            for (int nextLayerNeuronIndex = 0; nextLayerNeuronIndex < nextLayer.getNeurons().Length; nextLayerNeuronIndex++)
            {
                // Cria uma lista temporaria para armazenar as conexões do neuronio no indice nextLayerNeuronIndex da proxima camada
                List<Connection> tempConnections = new List<Connection>();

                // Percorre os neuronios da camada anterior
                for (int previousLayerNeuronIndex = 0; previousLayerNeuronIndex < previousLayer.getNeurons().Length; previousLayerNeuronIndex++)
                {
                    // Pega a conexão que aponta para o neuronio da proxima camada através da posição do neuronio da proxima camada
                    //PreviousNeuron.getConnections() -> Connections[] -> Connections[nextLayerNeuronIndex]
                    Connection temp = previousLayer.getNeurons()[previousLayerNeuronIndex].getOutputConnections()[nextLayerNeuronIndex];
                    tempConnections.Add(temp);
                }
                // Adiciona as conexões que apontam pra este neuronio
                nextLayer.getNeurons()[nextLayerNeuronIndex].setInputConnections(tempConnections.ToArray<Connection>());
            }
        }

        public double getLearningRate()
        {
            return learningRate;
        }

        public void setLearningRate(double learningRate)
        {
            this.learningRate = learningRate;
        }

        public Layer getInputLayer()
        {
            return inputLayer;
        }

        public void setInputLayer(Layer inputLayer)
        {
            this.inputLayer = inputLayer;
        }

        public Layer getHiddenLayer()
        {
            return hiddenLayer;
        }

        public void setHiddenLayer(Layer hiddenLayer)
        {
            this.hiddenLayer = hiddenLayer;
        }

        public Layer getOutputLayer()
        {
            return outputLayer;
        }

        public void setOutputLayer(Layer outputLayer)
        {
            this.outputLayer = outputLayer;
        }

        public void printNetworkState()
        {
            Console.WriteLine("Neurons");
            Console.WriteLine("\t" + "\t|\t" + "Id" + "\t");
            Console.WriteLine("\t" + "\t|\t" + "Bias");
            Console.WriteLine("\t" + "\t|\t" + "Delta");
            Console.WriteLine("\t" + "\t|\t" + "Output");


            for (int i = 0; i < inputLayer.getNeurons().Length; i++)
            {
                Console.WriteLine();
                Console.WriteLine("\t" + "\t|\t" + "input - " + (i + 1));
                Console.WriteLine("\t" + "\t|\t" + Math.Round(new Decimal(inputLayer.getNeurons()[i].getBias()), 4));
                Console.WriteLine("\t" + "\t|\t" + Math.Round(new Decimal(inputLayer.getNeurons()[i].getDelta()), 4));
                Console.WriteLine("\t" + "\t|\t" + Math.Round(new Decimal(inputLayer.getNeurons()[i].getOutput()), 4));

                for (int k = 0; k < inputLayer.getNeurons()[i].getOutputConnections().Length; k++)
                {
                    Console.WriteLine();
                    Console.WriteLine("\t" + "\t|\t" + "conOut - I " + (k + 1));
                    Console.WriteLine("\t" + "\t|\t" + Math.Round(new Decimal(inputLayer.getNeurons()[i].getOutputConnections()[k].getWeight()), 4));
                }
            }

            for (int neuronIndex = 0; neuronIndex < hiddenLayer.getNeurons().Length; neuronIndex++)
            {

                for (int k = 0; k < hiddenLayer.getNeurons()[neuronIndex].getInputConnections().Length; k++)
                {
                    Console.WriteLine();
                    Console.WriteLine("\t" + "\t|\t" + "conIn - H " + (k + 1));
                    Console.WriteLine("\t" + "\t|\t" + Math.Round(new Decimal(hiddenLayer.getNeurons()[neuronIndex].getInputConnections()[k].getWeight()), 4));
                }

                Console.WriteLine();
                Console.WriteLine("\t" + "\t|\t" + "hidden - " + (neuronIndex + 1));
                Console.WriteLine("\t" + "\t|\t" + Math.Round(new Decimal(hiddenLayer.getNeurons()[neuronIndex].getBias()), 4));
                Console.WriteLine("\t" + "\t|\t" + Math.Round(new Decimal(hiddenLayer.getNeurons()[neuronIndex].getDelta()), 4));
                Console.WriteLine("\t" + "\t|\t" + Math.Round(new Decimal(hiddenLayer.getNeurons()[neuronIndex].getOutput()), 4));

                for (int k = 0; k < hiddenLayer.getNeurons()[neuronIndex].getOutputConnections().Length; k++)
                {
                    Console.WriteLine();
                    Console.WriteLine("\t" + "\t|\t" + "conOut - H " + (k + 1));
                    Console.WriteLine("\t" + "\t|\t" + Math.Round(new Decimal(hiddenLayer.getNeurons()[neuronIndex].getOutputConnections()[k].getWeight()), 4));
                }
            }

            for (int i = 0; i < outputLayer.getNeurons().Length; i++)
            {

                for (int k = 0; k < outputLayer.getNeurons()[i].getInputConnections().Length; k++)
                {
                    Console.WriteLine();
                    Console.WriteLine("\t" + "\t|\t" + "conIn - O " + (k + 1));
                    Console.WriteLine("\t" + "\t|\t" + Math.Round(new Decimal(outputLayer.getNeurons()[i].getInputConnections()[k].getWeight()), 4));
                }

                Console.WriteLine();
                Console.WriteLine("\t" + "\t|\t" + "output - " + (i + 1));
                Console.WriteLine("\t" + "\t|\t" + Math.Round(new Decimal(outputLayer.getNeurons()[i].getBias()), 4));
                Console.WriteLine("\t" + "\t|\t" + Math.Round(new Decimal(outputLayer.getNeurons()[i].getDelta()), 4));
                Console.WriteLine("\t" + "\t|\t" + Math.Round(new Decimal(outputLayer.getNeurons()[i].getOutput()), 4));
            }

            Console.WriteLine();
            Console.WriteLine();
        }
    }
}
