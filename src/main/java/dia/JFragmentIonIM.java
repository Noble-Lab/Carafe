package main.java.dia;

public class JFragmentIonIM{

    public static CCSDIAMeta meta = new CCSDIAMeta();

    public float mz;
    public float intensity;
    public float rt;
    public int scan;
    public float im;

    public JFragmentIonIM(float mz, float intensity, float im, int scan) {
        this.mz = mz;
        this.intensity = intensity;
        // this.rt = rt;
        this.im = im;
        this.scan = scan;
    }

    public long get_frag_mz_bin(){
        return meta.get_fragment_ion_mz_bin_index(this.mz);
    }

    public int get_scan(){
        return scan;
    }

}
