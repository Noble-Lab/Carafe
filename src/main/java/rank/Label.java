package main.java.rank;

class Label {

    public int n_pos = 0;
    public int n_neg = 0;
    public boolean fully_detected = false;

    /**
     * A flag to indicate if the peptide a in the pair is detected or not.
     * It must be false in default
     */
    public boolean a_detected = false;

    /**
     * A flag to indicate if the peptide b in the pair is detected or not.
     * It must be false in default
     */
    public boolean b_detected = false;
}
