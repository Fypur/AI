using System.Numerics.Tensors;

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
            TensorPrimitives.Add(A.Grad, Result.Grad, A.Grad);
            TensorPrimitives.Add(B.Grad, Result.Grad, B.Grad);

            A.CreatedFrom.Backward();
            B.CreatedFrom.Backward();
        }
    }
}
