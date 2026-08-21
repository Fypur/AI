using System.Numerics.Tensors;

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
                TensorPrimitives.MultiplyAdd(Result.Grad, B.Data, A.Grad, A.Grad);
                TensorPrimitives.MultiplyAdd(Result.Grad, A.Data, B.Grad, B.Grad);

                A.CreatedFrom.Backward();
                B.CreatedFrom.Backward();
            }
            else
            {
                TensorPrimitives.MultiplyAdd(Result.Grad, Bf, A.Grad, A.Grad);
            }
        }
    }
}
