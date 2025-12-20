package main.java.dia;

import java.util.ArrayList;
import java.util.List;

import com.alibaba.fastjson2.annotation.JSONField;

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

    public double precursor_mz;

    public int precursor_charge;

    public double [] precursor_intensities;

    public int [] precursor_labels;

    /**
     * A list of fragment ions
     */
    public double [] fragment_mzs;

    public double [] fragment_intensities;
}
