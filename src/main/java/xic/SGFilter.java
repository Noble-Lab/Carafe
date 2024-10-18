package main.java.xic;


import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class SGFilter {
    public static final RealMatrix sg_factor_rm = MatrixUtils.createRealMatrix(new double[][]{{-21.0}, {14.0}, {39.0}, {54.0}, {59.0}, {54.0}, {39.0}, {14.0}, {-21.0}});
    public static float factor = (float)(1.0/231.0);

    public static RealMatrix paddedSavitzkyGolaySmooth3(double [][] x) {
        RealMatrix zeros = MatrixUtils.createRealMatrix(x.length,8+x[0].length);
        zeros.setSubMatrix(x,0,4);
        long j = zeros.getColumnDimension() - 4;
        RealMatrix y = MatrixUtils.createRealMatrix(x.length,x[0].length);
        for (int i = 4; i < j; i++) {
            y.setColumnMatrix(i - 4,zeros.getSubMatrix(0,x.length-1,i-4,i+4).multiply(sg_factor_rm).scalarMultiply(1.0/231.0));
        }

        return y;
    }


}

