package main.java.input;

import com.compomics.util.experiment.biology.modifications.Modification;

public class CPTM {

    public int pos = -1;
    public Modification modification;
    public CPTM(int position, Modification ptm){
        this.pos = position;
        this.modification = ptm;
    }
}
