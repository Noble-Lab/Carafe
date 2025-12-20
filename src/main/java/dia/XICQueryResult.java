package main.java.dia;

import java.util.ArrayList;
import java.util.List;

/**
 * This class is used for storing data extracted from TIMS-TOF data using the rust library.
 */
public class XICQueryResult {

    /**
     * The unique identifier of the PSM query: an unique integer ID for each PSM
     */
    public int id;

    /**
     * The ion mobility value
     */
    public double mobility_ook0;

    public double rt_seconds;

    /**
     * A list of precursor ions: different isotopes
     */
    public double [] precursor_mzs;

    /**
     * A 2D matrix of precursor intensities: each row corresponds to a precursor ion, each column corresponds to a retention time point
     */
    public double [][] precursor_intensities;

    /**
     * A list of fragment ions
     */
    public double [] fragment_mzs;
    public String [] fragment_labels;
    /**
     * A 2D matrix of fragment intensities: each row corresponds to a fragment ion, each column corresponds to a retention time point
     */
    public double [][] fragment_intensities;
    public double [] retention_time_results_seconds;
}
