namespace AI
{
    public enum LossFunctionType
    {
        MeanSquaredError,
    }

    public static class LossFunctions
    {
        public static Action<float[], float[], float[]> GetLossFunction(LossFunctionType type)
        {
            switch (type)
            {
                case LossFunctionType.MeanSquaredError:
                    return MeanSquaredError;
                default:
                    throw new Exception("Loss function not implemented");
            }
        }

        public static void MeanSquaredError(float[] output, float[] target, float[] loss)
        {
            for (int i = 0; i < output.Length; i++)
                loss[i] = (target[i] - output[i]) * (target[i] - output[i]);
        }
    }
}
