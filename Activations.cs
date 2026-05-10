namespace AI
{
    public static class Activations
    {
        public enum ActivationTypes { ReLU, ELU, Linear, LeakyReLu, Sigmoid }

        public static float ReLU(float x)
            => x >= 0 ? x : 0;

        public static float ReLUDerivative(float x)
            => x > 0 ? 1 : 0;

        public static float ELU(float x)
            => x >= 0 ? x : (float)Math.Exp(x) - 1;

        public static float ELUDerivative(float x)
            => x > 0 ? 1 : (float)Math.Exp(x);

        public static float Linear(float x)
            => x;

        public static float LinearDerivative(float x)
            => 1;

        public static float LeakyReLU(float x)
            => x >= 0 ? x : 0.01f * x;

        public static float LeakyReLUDerivative(float x)
            => x > 0 ? 1 : 0.01f;

        public static float Sigmoid(float x)
            => (float)(1 / (1 + Math.Exp(-x)));

        public static float SigmoidDerivative(float x)
            => Sigmoid(x) * (1 - Sigmoid(x));

        public static Func<float, float> GetActivationFunction(ActivationTypes activationType)
        {
            switch (activationType)
            {
                case ActivationTypes.Linear:
                    return Linear;
                case ActivationTypes.ReLU:
                    return ReLU;
                case ActivationTypes.ELU:
                    return ELU;
                case ActivationTypes.LeakyReLu:
                    return LeakyReLU;
                case ActivationTypes.Sigmoid:
                    return Sigmoid;
            }

            throw new Exception($"Activation function of type {activationType} is not yet implemented");
        }

        public static Func<float, float> GetActivationDerivative(ActivationTypes activationType)
        {
            switch (activationType)
            {
                case ActivationTypes.Linear:
                    return LinearDerivative;
                case ActivationTypes.ReLU:
                    return ReLUDerivative;
                case ActivationTypes.ELU:
                    return ELUDerivative;
                case ActivationTypes.LeakyReLu:
                    return LeakyReLUDerivative;
                case ActivationTypes.Sigmoid:
                    return SigmoidDerivative;
            }

            throw new Exception($"Derivative of Activation function of type {activationType} is not yet implemented");
        }
    }
}
