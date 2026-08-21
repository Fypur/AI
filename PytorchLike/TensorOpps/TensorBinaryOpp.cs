namespace AI
{
    public abstract class TensorBinaryOpp : TensorOpp
    {
        public Tensor A;
        public Tensor B;
        public Tensor Result;

        public bool IsBroadcasted;

        public TensorBinaryOpp(Tensor A, Tensor B, Tensor Result)
        {
            this.A = A;
            this.B = B;
            this.Result = Result;

            IsBroadcasted =
        }

        public override void Backward()
        {

        }
    }
}
