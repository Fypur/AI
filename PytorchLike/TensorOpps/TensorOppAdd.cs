namespace AI
{
    public class TensorOppAdd : TensorOpp
    {
        public Tensor A;
        public Tensor B;
        public Tensor Result;

        public TensorOppAdd(Tensor A, Tensor B, Tensor Result)
        {
            this.A = A;
            this.B = B;
            this.Result = Result;
        }

        public override void Backward()
        {
            for (int i = 0; i < A.Grad.Length; i++)
            {
                A.Grad[i] += Result.Grad[i];
                B.Grad[i] += Result.Grad[i];
            }

            A.CreatedFrom.Backward();
            B.CreatedFrom.Backward();
        }
    }
}
