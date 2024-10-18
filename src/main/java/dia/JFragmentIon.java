package main.java.dia;

public class JFragmentIon {

    public static DIAMeta meta = new DIAMeta();

    public float mz;
    public float intensity;
    public float rt;
    public int scan;

    public JFragmentIon(float mz,float intensity, int scan){
        this.mz = mz;
        this.intensity = intensity;
        // this.rt = rt;
        this.scan = scan;
    }

    public long get_frag_mz_bin(){
        return meta.get_fragment_ion_mz_bin_index(this.mz);
    }

    public int get_scan(){
        return scan;
    }

}
