package main.java.dia;

import com.jerolba.carpet.CarpetReader;
import main.java.input.CParameter;
import org.eclipse.collections.impl.list.mutable.primitive.FloatArrayList;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;


public class CCSDIAMeta{

    public record MS2Spectra(//FloatArrayList mz_values,
                             //FloatArrayList intensities,
                             int index,
                             List<Float> mz_values,
                             List<Float> intensities,
                             double precursor_rt,
                             double collision_energy,
                             double isolation_mz,
                             double isolation_width) {
    }

    /**
     * Only for loading isolation window information
     * @param isolation_mz
     * @param isolation_width
     */
    public record IsolationWindowRecord(double isolation_mz,
                                        double isolation_width){
    }

    public int ms_level = 2;
    public double fragment_ion_mz_bin_size = 0.05;
    public double fragment_ion_mz_min = Double.MAX_VALUE;
    public double fragment_ion_mz_max = 0;
    public double precursor_ion_mz_min = Double.MAX_VALUE;
    public double precursor_ion_mz_max = 0;
    public double nce = 0;
    public double rt_min = Double.MAX_VALUE;
    public double rt_max = 0;

    public double min_fragment_ion_intensity = Double.MAX_VALUE;

    public RTUnit rt_unit = RTUnit.second;

    public enum RTUnit {
        minute("RT in minute"),
        second("RT in second");

        /**
         * The description.
         */
        public final String description;

        /**
         * Constructor.
         *
         * @param description the description
         */
        RTUnit(String description) {
            this.description = description;
        }
    }

    // public TreeMap<Integer, IScan> num2scanMap = new TreeMap<>();
    public HashMap<String,IsolationWindow> isolationWindowMap = new HashMap<>();
    public HashMap<String,Integer> isolationWindow2n_scan = new HashMap<>();
    public double isolation_win_mz_max = -1;

    public void load_ms_data(String ms_file){
        //TODO
    }

    /**
     * The default instrument type: timsTOF
     * @return
     */
    public String get_ms_instrument(){
        // https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo
        HashMap<String,String> instrumentInfo = new HashMap<>();
        instrumentInfo.put("timsTOF", "timsTOF");
        return instrumentInfo.get("timsTOF");
    }

    public void get_ms_run_meta_data(String ms_file) throws IOException {
        int total_spectra = 0;
        double mz_start = 0;
        double mz_end = 0;
        String isoWinID;

        CarpetReader<MS2Spectra> reader = new CarpetReader<>(new File(ms_file), MS2Spectra.class);
        double isolation_mz;
        double isolation_width;
        double rt;
        for (MS2Spectra spectrum: reader) {
            total_spectra++;
            isolation_mz = spectrum.isolation_mz;
            isolation_width = spectrum.isolation_width;

            //double mz_lower = spectrum.mz_values.min();
            //double mz_upper = spectrum.mz_values.max();
            double mz_lower = -1;
            double mz_upper = -1;
            try {
                if (!spectrum.mz_values.isEmpty()) {
                    mz_lower = Collections.min(spectrum.mz_values);
                    mz_upper = Collections.max(spectrum.mz_values);

                    min_fragment_ion_intensity = Math.min(min_fragment_ion_intensity, Collections.min(spectrum.intensities));

                    if (this.fragment_ion_mz_min > mz_lower) {
                        this.fragment_ion_mz_min = mz_lower;
                    }
                    if (this.fragment_ion_mz_max < mz_upper) {
                        this.fragment_ion_mz_max = mz_upper;
                    }
                } else {
                    System.out.println("Warning: Empty mz_values list in spectrum:"+spectrum.index);
                }
            } catch (NoSuchElementException e) {
                System.out.println("Error: index = " + spectrum.index);
                System.out.println("Error: " + e.getMessage());
            }

            if(this.rt_unit == RTUnit.second){
                rt = spectrum.precursor_rt/60.0;
            }else{
                rt = spectrum.precursor_rt;
            }

            if(this.rt_min > rt){
                this.rt_min = rt;
            }
            if(this.rt_max < rt){
                this.rt_max = rt;
            }

            if(this.ms_level == 2) {
                double [] iso_win_range = get_isolation_window(isolation_mz,isolation_width);
                mz_start = iso_win_range[0];
                mz_end =  iso_win_range[1];
                isoWinID = IsolationWindow.generate_id(mz_start,mz_end);
                if(this.precursor_ion_mz_max < mz_end){
                    this.precursor_ion_mz_max = mz_end;
                }
                if(this.precursor_ion_mz_min > mz_start){
                    this.precursor_ion_mz_min = mz_start;
                }
                if(this.nce==0){
                    this.nce = CParameter.NCE;
                }

            }else{
                isoWinID = "0";
                if(mz_lower >0){
                    mz_start = mz_lower;
                    mz_end = mz_upper;
                }
            }

            if(this.ms_level == 2) {
                if(this.isolation_win_mz_max >0 &&
                        (mz_end - mz_start) > this.isolation_win_mz_max){
                    System.out.println("Ignore - the isolation window m/z range is too large: "+mz_start+"-"+mz_end);
                    continue;
                }

            }

            if(!this.isolationWindowMap.containsKey(isoWinID)){
                this.isolationWindowMap.put(isoWinID,new IsolationWindow(mz_start,mz_end));
            }

            if(!isolationWindow2n_scan.containsKey(isoWinID)){
                isolationWindow2n_scan.put(isoWinID,0);
            }
            isolationWindow2n_scan.put(isoWinID,isolationWindow2n_scan.get(isoWinID)+1);

        }

        System.out.println("Total MS/MS spectra:"+total_spectra);
        this.rt_max = this.rt_max + 0.1;
        for(String isoWin: this.isolationWindowMap.keySet()){
            // System.out.println("Valid isolation window: "+isoWin + " -> "+isolationWindowMap.get(isoWin).mz_lower+" - "+isolationWindowMap.get(isoWin).mz_upper);
        }

        // Sort the HashMap by the attribute of the IsolationWindow (in this case, 'attribute')
        Map<String, IsolationWindow> sortedIWMap = isolationWindowMap.entrySet()
                .stream()
                // Sort by attribute of IsolationWindow
                .sorted(Map.Entry.comparingByValue(Comparator.comparingDouble(iw -> iw.mz_lower)))
                // Collect the sorted entries into a LinkedHashMap to preserve order
                .collect(Collectors.toMap(
                        Map.Entry::getKey,
                        Map.Entry::getValue,
                        (e1, e2) -> e1,
                        LinkedHashMap::new
                ));

        sortedIWMap.forEach((isoWin, value) -> System.out.println("Valid isolation window: "+isoWin + " -> " + value.mz_lower + " - " + value.mz_upper));


        if(this.ms_level == 2) {
            System.out.println("Fragment ion m/z range: " + this.fragment_ion_mz_min + "-" + this.fragment_ion_mz_max);
            System.out.println("Precursor ion m/z range: " + this.precursor_ion_mz_min + "-" + this.precursor_ion_mz_max);
        }
        System.out.println("Retention time range: " + this.rt_min + "-" + this.rt_max);
        System.out.println("Min fragment ion intensity: "+this.min_fragment_ion_intensity);
    }

    public long get_fragment_ion_mz_bin_index(double mz){
        return Math.round((mz - this.fragment_ion_mz_min) / this.fragment_ion_mz_bin_size);
    }

    public static double[] get_isolation_window(double isolation_mz, double isolation_width){
        return new double[]{isolation_mz - isolation_width/2.0, isolation_mz + isolation_width/2.0};
    }

}
