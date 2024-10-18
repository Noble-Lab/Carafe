package main.java.dia;

import java.util.Comparator;

public class LPeakComparatorMax2Min implements
        Comparator<LPeak> {
    @Override
    public int compare(LPeak a, LPeak b) {
        return Double.compare(b.intensity,a.intensity);
    }
}
