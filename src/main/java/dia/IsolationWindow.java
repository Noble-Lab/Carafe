package main.java.dia;

public class IsolationWindow {

    public double target_mz = 0;
    public double mz_lower = 0;
    public double mz_upper = 0;

    public String id;

    public IsolationWindow(double mz1,double mz2){
        this.mz_lower = mz1;
        this.mz_upper = mz2;
        this.id = generate_id(mz1,mz2);
    }

    public static String generate_id(double mz1, double mz2){
        return Math.round(mz1*10.0)+"_"+Math.round(mz2*10);
    }

}
