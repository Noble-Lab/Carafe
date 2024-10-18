package main.java.ai;

public class JPeakGroup {

    /**
     * The row index of this peak group/peptide form
     */
    int id;
    String peptide;
    double mz;
    double charge;
    double apex_rt;
    double rt_start;
    double rt_end;
    int n_shared_peaks= 0;
}
