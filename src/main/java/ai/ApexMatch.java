package main.java.ai;

public class ApexMatch {

    public int ms2index;
    public String isolation_window;

    /**
     * The Apex RT matched
     */
    public double apex_rt;
    /**
     * The difference between the apex RT and the RT to match
     */
    public double delta_rt;

    public int scan_number;

    /**
     * Isolation window of the matched MS2 scan. Carried on the match (rather than looked up
     * from a shared index at write time) so a multi-file finetune can use a separate per-run
     * index for each PSM.
     */
    public double isolation_mz_start;
    public double isolation_mz_end;
}
