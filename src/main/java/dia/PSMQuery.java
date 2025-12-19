package main.java.dia;

import java.util.ArrayList;
import java.util.List;

/**
 * This class is used for extracting data from TIMS-TOF data using the rust library.
 */
public class PSMQuery {

    /**
     * The unique identifier of the PSM query: an unique integer ID for each PSM
     */
    public int id;

    /**
     * The ion mobility value
     */
    public double mobility;

    public double rt_seconds;

    /**
     * The precursor m/z value
     */
    public double precursor;

    /**
     * The precursor charge state
     */
    public int precursor_charge;

    /**
     * A list of precursor isotopes, such as 0, 1, 2, etc.
     */
    public List<Integer> precursor_isotopes = new ArrayList<>();

    /**
     * A list of fragment ions
     */
    public List<Double> fragments = new ArrayList<>();

    /**
     * A list of fragment ion labels. The format follows the <a href="https://github.com/HUPO-PSI/mzPAF">mzPAF</a> standard.
     */
    public List<String> fragment_labels = new ArrayList<>();
}
