package main.java.xic;


import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class SGFilter3points {

    public static float factor = (float)(1.0/231.0);

    public static RealMatrix paddedSavitzkyGolaySmooth3(double [][] x) {
        RealMatrix zeros = MatrixUtils.createRealMatrix(x);
        int j = zeros.getColumnDimension();
        RealMatrix y = MatrixUtils.createRealMatrix(x.length,x[0].length);
        y.setColumnMatrix(0,zeros.getColumnMatrix(0).scalarMultiply(2.0/3.0).add(zeros.getColumnMatrix(1).scalarMultiply(1.0/3.0)));
        y.setColumnMatrix(j-1,zeros.getColumnMatrix(j-1).scalarMultiply(2.0/3.0).add(zeros.getColumnMatrix(j-2).scalarMultiply(1.0/3.0)));
        for (int i = 1; i <= (j-2); i++) {
            y.setColumnMatrix(i,zeros.getColumnMatrix(i).scalarMultiply(0.5).add(zeros.getColumnMatrix(i-1).scalarMultiply(0.25)).add(zeros.getColumnMatrix(i+1).scalarMultiply(0.25)));
        }
        return y;
    }

}

