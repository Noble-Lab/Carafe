package main.java.ai;

import java.util.ArrayList;

public class PeptideCCS {

    String peptide;
    int charge;
    String modification;
    double ccs;

    ArrayList<Double> ccs_values = new ArrayList<>();
    ArrayList<Double> scores = new ArrayList<>();


    /**
     * Two column modification, such as ["Oxidation@M;Carbamidomethyl@C", "17;6"]
     */
    String[] mods;
}
