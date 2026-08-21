using System.Numerics.Tensors;

namespace AI
{
    public class TensorOppMatrixMult : TensorOpp
    {
        public Tensor A;
        public Tensor B;
        public Tensor Result;

        public int NumMatrixMult;

        public TensorOppMatrixMult(Tensor A, Tensor B, Tensor Result, int numMatrixMult)
        {
            this.A = A;
            this.B = B;
            this.Result = Result;

            NumMatrixMult = numMatrixMult;
        }

        public override void Backward()
        {
            int numMatrixMult = 1;

            for (int i = 0; i < A.Shape.Length - 2; i++)
                numMatrixMult *= A.Shape[i];

            //Matrix mult backprop
            int aRows = A.Shape[A.Shape.Length - 2];
            int aCols = A.Shape[A.Shape.Length - 1];
            int bCols = B.Shape[B.Shape.Length - 1];

            int aMatrixSize = aRows * aCols;
            int bMatrixSize = aCols * bCols;
            int tMatrixSize = aRows * bCols;

            //Actual matrix mult code
            for (int i = 0; i < numMatrixMult; i++)
            {
                int aOffset = i * aMatrixSize;
                int bOffset = i * bMatrixSize;
                int tOffset = i * tMatrixSize;

                for (int r = 0; r < aRows; r++)
                {
                    for (int c = 0; c < aCols; c++)
                    {
                        float aValue = A.Data[aOffset + r * aCols + c];
                        Span<float> resultRow = new Span<float>(Result.Grad, tOffset + r * bCols, bCols);
                        ReadOnlySpan<float> bRow = new ReadOnlySpan<float>(B.Data, bOffset + c * bCols, bCols);

                        A.Grad[aOffset + r * aCols + c] += TensorPrimitives.Dot(bRow, resultRow);

                        Span<float> bGradRow = new Span<float>(B.Grad, bOffset + c * bCols, bCols);

                        TensorPrimitives.MultiplyAdd(resultRow, aValue, bGradRow, bGradRow);
                    }
                }
            }

            A.CreatedFrom.Backward();
            B.CreatedFrom.Backward();
        }
    }
}
