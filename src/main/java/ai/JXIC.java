package main.java.ai;

public class JXIC {

    public double[][] smoothed_fragment_intensities = new double[1][1];
    public double[][] raw_fragment_intensities = new double[1][1];
    public double [] xic_rt_values = new double[1];
    public double [] fragment_ion_mzs = new double[1];
    public double [] fragment_ion_cors = new double[1];
    public int [] fragment_ion_skewness = new int[1];
    public String id = "";
    public double rt_start = -1;
    public double rt_end = -1;
    public double rt_apex = -1;
    public String peptide = "";
    public int charge = -1;
    public String modification = "";

}
