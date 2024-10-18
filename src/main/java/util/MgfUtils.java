package main.java.util;

import com.compomics.util.experiment.mass_spectrometry.spectra.Spectrum;

public class MgfUtils{

    public static  String asMgf(Spectrum spectrum, String spectrumTitle,int charge, String scan_number){

        double intensity = spectrum.getPrecursor().intensity;
        double mz = spectrum.getPrecursor().mz;
        double[] mzArray = spectrum.mz;
        double[] intensityArray = spectrum.intensity;

        StringBuilder stringBuilder = new StringBuilder();

        stringBuilder.append("BEGIN IONS\n");
        stringBuilder.append("TITLE=").append(spectrumTitle).append("\n");
        stringBuilder.append("PEPMASS=").append(mz).append(" ").append(intensity).append("\n");
        if(spectrum.getPrecursor().rt >= 0.0) {
            double rt = spectrum.getPrecursor().rt;
            stringBuilder.append("RTINSECONDS=").append(rt).append("\n");
        }
        stringBuilder.append("CHARGE=").append(charge).append("+\n");
        if(scan_number != null){
            stringBuilder.append("SCANS=").append(scan_number).append("\n");
        }
        for(int i=0;i<mzArray.length;i++){
            if(intensityArray[i] > 0.000000000001) {
                stringBuilder.append(String.format("%.4f", mzArray[i])).append(" ").append(String.format("%.2f", intensityArray[i])).append("\n");
            }
        }
        stringBuilder.append("END IONS\n");
        return(stringBuilder.toString());

    }

}

