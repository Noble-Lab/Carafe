package main.java.dia;

import com.compomics.util.experiment.mass_spectrometry.spectra.Spectrum;

public class LibSpectrum {

    public Spectrum spectrum = new Spectrum();
    public int pepID;
    public int charge;
    public double mass;
    public double rt;
    public int scan;

    /**
     * The fragment ion number, such as for b1, the number is 1.
     */
    public int [] ion_numbers;

    /**
     * The ion types of the fragment ions: b or y
     */
    public String [] ion_types;
}
