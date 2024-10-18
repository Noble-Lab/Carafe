package main.java.xic;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class SGFilter5points {

    public static final RealMatrix sg_factor_rm = MatrixUtils.createRealMatrix(new double[][]{{-3.0}, {12.0}, {17.0}, {12.0}, {-3.0}});
    public static float factor = (float)(1.0/231.0);


    public static RealMatrix paddedSavitzkyGolaySmooth3(double [][] x) {
        RealMatrix zeros = MatrixUtils.createRealMatrix(x.length,4+x[0].length);
        zeros.setSubMatrix(x,0,2);
        long j = zeros.getColumnDimension() - 2;
        RealMatrix y = MatrixUtils.createRealMatrix(x.length,x[0].length);
        for (int i = 2; i < j; i++) {
            y.setColumnMatrix(i - 2,zeros.getSubMatrix(0,x.length-1,i-2,i+2).multiply(sg_factor_rm).scalarMultiply(1.0/35.0));
        }

        return y;
    }

}

