namespace AI
{
    public class Linear
    {
        public Tensor Weights;
        public Tensor Biases;

        public Linear(int inputSize, int outputSize)
        {
            Biases = new Tensor()
        }

        public Tensor Forward(Tensor input)
        {
            return Tensor.MatrixMult(weights, input) + biases;
        }
    }
}
