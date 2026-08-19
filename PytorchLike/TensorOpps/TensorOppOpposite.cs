namespace AI
{
    public class TensorOppOpposite : TensorOpp
    {
        public Tensor A;
        public Tensor Result;

        public TensorOppOpposite(Tensor A, Tensor Result)
        {
            this.A = A;
            this.Result = Result;
        }

        public override void Backward()
        {
            for (int i = 0; i < A.Grad.Length; i++)
                A.Grad[i] -= Result.Grad[i];

            A.CreatedFrom.Backward();
        }
    }
}
