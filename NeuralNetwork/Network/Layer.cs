namespace NeuralNetwork.Network
{
    public class Layer
    {
        private Neuron[] neurons;

        public Layer(int neurons)
        {
            this.neurons = new Neuron[neurons];

            for (int index = 0; index < neurons; index++)
            {
                this.neurons[index] = new Neuron(generateRandomConnections(neurons));
            }
        }

        public Layer(int neurons, int neuronsNextLayer)
        {
            this.neurons = new Neuron[neurons];
            for (int index = 0; index < neurons; index++)
            {
                this.neurons[index] = new Neuron(generateRandomConnections(neurons), generateRandomConnections(neuronsNextLayer));
            }
        }

        private Connection[] generateRandomConnections(int size)
        {
            Connection[] conn = new Connection[size];

            for (int index = 0; index < size; index++)
                conn[index] = new Connection();

            return conn;
        }

        public Layer(Neuron[] neurons)
        {
            this.neurons = neurons;
        }

        public Layer(double[] inputs, Neuron[] neurons)
        {
            this.neurons = neurons;
        }

        public Neuron[] getNeurons()
        {
            return neurons;
        }

        public void setNeurons(Neuron[] neurons)
        {
            this.neurons = neurons;
        }
    }
}
