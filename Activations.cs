namespace AI
{
    public enum ActivationTypes { ReLU, ELU, Linear, LeakyReLU, Sigmoid }
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

        public static Action<float[], float[]> GetActivationFunction(ActivationTypes activationType)
        {
            switch (activationType)
            {
                case ActivationTypes.Linear:
                    return Linear;
                case ActivationTypes.ReLU:
                    return ReLU;
                case ActivationTypes.ELU:
                    return ELU;
                case ActivationTypes.LeakyReLU:
                    return LeakyReLU;
                case ActivationTypes.Sigmoid:
                    return Sigmoid;
            }

            throw new Exception($"Activation function of type {activationType} is not yet implemented");
        }

        public static Action<float[], float[]> GetActivationDerivative(ActivationTypes activationType)
        {
            switch (activationType)
            {
                case ActivationTypes.Linear:
                    return LinearDerivative;
                case ActivationTypes.ReLU:
                    return ReLUDerivative;
                case ActivationTypes.ELU:
                    return ELUDerivative;
                case ActivationTypes.LeakyReLU:
                    return LeakyReLUDerivative;
                case ActivationTypes.Sigmoid:
                    return SigmoidDerivative;
            }

            throw new Exception($"Derivative of Activation function of type {activationType} is not yet implemented");
        }
    }
}
