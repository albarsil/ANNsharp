using System;

namespace NeuralNetwork.Network
{
    public class Neuron
    {
        private double bias;
        private double biasDelta;
        private Connection[] inputConnections;
        private Connection[] outputConnections;
        private double delta;
        private double output;

        public Neuron(Connection[] inputs)
        {
            this.bias = 2 * (Network.Random.NextDouble() - 0.5) * 0.5;
            this.inputConnections = inputs;
        }

        public Neuron(Connection[] inputs, Connection[] outputs)
        {
            this.bias = 2 * (Network.Random.NextDouble() - 0.5) * 0.5;
            this.inputConnections = inputs;
            this.outputConnections = outputs;
        }

        public Connection[] getOutputConnections()
        {
            return outputConnections;
        }

        public void setOutputConnections(Connection[] outputConnections)
        {
            this.outputConnections = outputConnections;
        }

        public double getBias()
        {
            return bias;
        }

        public void setBias(double bias)
        {
            this.bias = bias;
        }

        public Connection[] getInputConnections()
        {
            return inputConnections;
        }

        public void setInputConnections(Connection[] inputConnections)
        {
            this.inputConnections = inputConnections;
        }

        public double getDelta()
        {
            return delta;
        }

        public void setDelta(double delta)
        {
            this.delta = delta;
        }

        public void activeNeuron(double[] previousLayerOutputs)
        {

            double net = 0;

            for (int i = 0; i < inputConnections.Length; i++)
                net += (previousLayerOutputs[i] * inputConnections[i].getWeight());

            output = 1 / (1 + Math.Exp(-(net + bias)));
        }

        public double getOutput()
        {
            return output;
        }

        public void setOutput(double output)
        {
            this.output = output;
        }

        public double getBiasDelta()
        {
            return biasDelta;
        }

        public void setBiasDelta(double biasDelta)
        {
            this.biasDelta = biasDelta;
        }
    }
}
