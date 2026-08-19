namespace AI
{
    public class TensorOppMult : TensorOpp
    {
        public Tensor A;
        public Tensor B;
        public float Bf;
        public Tensor Result;

        public TensorOppMult(Tensor A, Tensor B, Tensor Result)
        {
            this.A = A;
            this.B = B;
            this.Result = Result;
        }

        public TensorOppMult(Tensor A, float B, Tensor Result)
        {
            this.A = A;
            Bf = B;
            this.Result = Result;
        }

        public override void Backward()
        {
            if (B != null)
            {
                for (int i = 0; i < A.Grad.Length; i++)
                {
                    A.Grad[i] += B.Data[i] * Result.Grad[i];
                    B.Grad[i] += A.Data[i] * Result.Grad[i];
                }

                A.CreatedFrom.Backward();
                B.CreatedFrom.Backward();
            }
            else
            {
                for (int i = 0; i < A.Grad.Length; i++)
                    A.Grad[i] += Bf * Result.Grad[i];
            }
        }
    }
}
