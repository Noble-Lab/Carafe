package main.java.ai;

import java.util.ArrayList;

public class PeptideRT {

    String peptide;
    double rt;
    double rt_norm;
    String modification;

    ArrayList<Double> rts = new ArrayList<>();
    ArrayList<Double> scores = new ArrayList<>();

    /**
     * Two column modification, such as ["Oxidation@M;Carbamidomethyl@C", "17;6"]
     */
    String[] mods;

    double rt_min;
    double rt_max;

}
