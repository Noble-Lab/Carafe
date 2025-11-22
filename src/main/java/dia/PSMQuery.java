package main.java.dia;

import com.alibaba.fastjson.annotation.JSONField;

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
     * A list of precursor ions: different isotopes
     */
    public List<Double> precursors = new ArrayList<>();

    /**
     * A list of fragment ions
     */
    public List<Double> fragments = new ArrayList<>();
}
