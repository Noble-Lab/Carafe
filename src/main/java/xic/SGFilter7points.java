package main.java.xic;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class SGFilter7points {

    public static final RealMatrix sg_factor_rm = MatrixUtils.createRealMatrix(new double[][]{{-2.0}, {3.0}, {6.0}, {7.0}, {6.0}, {3.0}, {-2.0}});

    public static float factor = (float)(1.0/231.0);


    public static RealMatrix paddedSavitzkyGolaySmooth3(double [][] x) {
        RealMatrix zeros = MatrixUtils.createRealMatrix(x.length,6+x[0].length);
        zeros.setSubMatrix(x,0,3);
        long j = zeros.getColumnDimension() - 3;
        RealMatrix y = MatrixUtils.createRealMatrix(x.length,x[0].length);
        for (int i = 3; i < j; i++) {
            y.setColumnMatrix(i - 3,zeros.getSubMatrix(0,x.length-1,i-3,i+3).multiply(sg_factor_rm).scalarMultiply(1.0/21.0));
        }

        return y;
    }


}

