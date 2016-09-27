namespace NeuralNetwork.Network
{
    public class Connection
    {
        private double weight;
        private double delta;

        public Connection()
        {
            this.weight = 2 * (Network.Random.NextDouble() - 0.5) * 0.5;
        }

        public Connection(double weight)
        {
            this.weight = weight;
        }

        public double getWeight()
        {
            return weight;
        }

        public void setWeight(double weight)
        {
            this.weight = weight;
        }

        public double getDelta()
        {
            return this.delta;
        }
        public void setDelta(double delta)
        {
            this.delta = delta;
        }
    }
}
