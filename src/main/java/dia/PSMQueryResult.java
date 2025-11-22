package main.java.dia;

import java.util.ArrayList;
import java.util.List;

/**
 * This class is used for storing data extracted from TIMS-TOF data using the rust library.
 */
public class PSMQueryResult {

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
    public List<Double> precursor_mzs = new ArrayList<>();
    public List<Double> precursor_intensities = new ArrayList<>();

    /**
     * A list of fragment ions
     */
    public List<Double> fragment_mzs = new ArrayList<>();
    public List<Double> fragment_intensities = new ArrayList<>();
}
