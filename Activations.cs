namespace AI
{
    public enum ActivationType { ReLU, ELU, Linear, LeakyReLU, Sigmoid }
    public static class Activations
    {
        public static void ReLU(float[] x, float[] result)
        {
            for (int i = 0; i < x.Length; i++)
                result[i] = x[i] > 0 ? x[i] : 0;
        }

        public static void ReLUDerivative(float[] x, float[] result)
        {
            for (int i = 0; i < x.Length; i++)
                result[i] = x[i] > 0 ? 1 : 0;
        }

        public static void ELU(float[] x, float[] result)
        {
            for (int i = 0; i < x.Length; i++)
                result[i] = x[i] >= 0 ? x[i] : (float)Math.Exp(x[i]) - 1;
        }

        public static void ELUDerivative(float[] x, float[] result)
        {
            for (int i = 0; i < x.Length; i++)
                result[i] = x[i] > 0 ? 1 : (float)Math.Exp(x[i]);
        }

        public static void Linear(float[] x, float[] result)
        { }

        public static void LinearDerivative(float[] x, float[] result)
        {
            for (int i = 0; i < x.Length; i++)
                result[i] = 1;
        }

        public static void LeakyReLU(float[] x, float[] result)
        {
            for (int i = 0; i < x.Length; i++)
                result[i] = x[i] >= 0 ? x[i] : 0.01f * x[i];
        }

        public static void LeakyReLUDerivative(float[] x, float[] result)
        {
            for (int i = 0; i < x.Length; i++)
                result[i] = x[i] > 0 ? 1 : 0.01f;
        }

        public static void Sigmoid(float[] x, float[] result)
        {
            for (int i = 0; i < x.Length; i++)
                result[i] = (float)(1 / (1 + Math.Exp(-x[i])));
        }

        public static void SigmoidDerivative(float[] x, float[] result)
        {
            for (int i = 0; i < x.Length; i++)
            {
                float s = (float)(1 / (1 + Math.Exp(-x[i])));
                result[i] = s * (1 - s);
            }
        }

        public static Action<float[], float[]> GetActivationFunction(ActivationType activationType)
        {
            switch (activationType)
            {
                case ActivationType.Linear:
                    return Linear;
                case ActivationType.ReLU:
                    return ReLU;
                case ActivationType.ELU:
                    return ELU;
                case ActivationType.LeakyReLU:
                    return LeakyReLU;
                case ActivationType.Sigmoid:
                    return Sigmoid;
            }

            throw new Exception($"Activation function of type {activationType} is not yet implemented");
        }

        public static Action<float[], float[]> GetActivationDerivative(ActivationType activationType)
        {
            switch (activationType)
            {
                case ActivationType.Linear:
                    return LinearDerivative;
                case ActivationType.ReLU:
                    return ReLUDerivative;
                case ActivationType.ELU:
                    return ELUDerivative;
                case ActivationType.LeakyReLU:
                    return LeakyReLUDerivative;
                case ActivationType.Sigmoid:
                    return SigmoidDerivative;
            }

            throw new Exception($"Derivative of Activation function of type {activationType} is not yet implemented");
        }
    }
}
