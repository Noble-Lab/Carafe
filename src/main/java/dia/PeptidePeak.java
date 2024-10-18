package main.java.dia;

import java.util.List;

public class PeptidePeak {

    public double apex_rt;
    public long apex_index;
    public long boundary_left_index;
    public long boundary_right_index;
    public double min_smoothed_intensity;
    public double boundary_left_rt;
    public double boundary_right_rt;
    public double [] cor_to_best_ion = new double[0];
    public List<Double> fragment_ions_mz;



}
