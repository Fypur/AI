namespace AI
{
    public class TensorOppDivide : TensorOpp
    {
        public Tensor Numerator;
        public Tensor Denominator;
        public Tensor Result;

        public TensorOppDivide(Tensor Numerator, Tensor Denominator, Tensor Result)
        {
            this.Numerator = Numerator;
            this.Denominator = Denominator;
            this.Result = Result;
        }

        public override void Backward()
        {
            for (int i = 0; i < Numerator.Grad.Length; i++)
            {
                Numerator.Grad[i] += 1 / Denominator.Data[i] * Result.Grad[i];
                Denominator.Grad[i] += -Numerator.Data[i] / (Denominator.Data[i] * Denominator.Data[i]) * Result.Grad[i];
            }

            Numerator.CreatedFrom.Backward();
            Denominator.CreatedFrom.Backward();
        }
    }
}
