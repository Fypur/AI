using System.Numerics.Tensors;

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
            TensorPrimitives.Subtract(A.Grad, Result.Grad, A.Grad);

            A.CreatedFrom.Backward();
        }
    }
}
