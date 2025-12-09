package main.java.dia;

import com.jerolba.carpet.CarpetReader;
import main.java.input.CParameter;
import main.java.util.Cloger;

import java.io.File;
import java.io.IOException;
import java.sql.*;
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

    /**
     * A map from m/z to collision energy
     */
    public HashMap<Integer,Double> mz2ce = new HashMap<>();

    /**
     * The step size for m/z in "mz2ce" (0.1 m/z units, represented as integer 10)
     */
    private final int ce_map_mz_step_size = 10; // 0.1 m/z

    public int ms_level = 2;
    public double fragment_ion_mz_bin_size = 0.05;
    public double fragment_ion_mz_min = Double.MAX_VALUE;
    public double fragment_ion_mz_max = 0;
    public double precursor_ion_mz_min = Double.MAX_VALUE;
    public double precursor_ion_mz_max = 0;
    public double nce = 0;
    public double rt_min = Double.MAX_VALUE;
    public double rt_max = 0;
    public String ms_instrument = "";

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

    /**
     * Extract MS meta data using SQL query directly on TIMS-TOF data.
     * @param ms_file The input TIMS-TOF data file (.d format)
     */
    public void get_ms_run_meta_data_using_sql(String ms_file) {

        File sql_File = new File(ms_file + File.separator + "analysis.tdf");
        if(sql_File.isFile()){
            Cloger.getInstance().logger.info("Extract MS meta information from file: " + sql_File.getAbsolutePath());
        }else{
            Cloger.getInstance().logger.error("Error: Cannot find the TIMS-TOF SQL database file: " + sql_File.getAbsolutePath());
            System.exit(1);
        }

        try {
            Connection connection = DriverManager.getConnection("jdbc:sqlite:" + sql_File.getAbsolutePath());

            // for fragment ion m/z range
            PreparedStatement pstmt = connection.prepareStatement("SELECT Value FROM GlobalMetadata WHERE Key='MzAcqRangeLower'");
            ResultSet min_fragment_mz_res = pstmt.executeQuery();
            double min_fragment_mz = min_fragment_mz_res.getDouble(1);

            pstmt = connection.prepareStatement("SELECT Value FROM GlobalMetadata WHERE Key='MzAcqRangeUpper'");
            ResultSet max_fragment_mz_res = pstmt.executeQuery();
            double max_fragment_mz = max_fragment_mz_res.getDouble(1);

            // for precursor ion m/z range
            pstmt = connection.prepareStatement("SELECT MIN(IsolationMz - IsolationWidth/2) FROM DiaFrameMsMsWindows");
            ResultSet min_precursor_mz_res = pstmt.executeQuery();
            double min_precursor_mz = min_precursor_mz_res.getDouble(1);

            pstmt = connection.prepareStatement("SELECT MAX(IsolationMz + IsolationWidth/2) FROM DiaFrameMsMsWindows");
            ResultSet max_precursor_mz_res = pstmt.executeQuery();
            double max_precursor_mz = max_precursor_mz_res.getDouble(1);

            pstmt = connection.prepareStatement("SELECT MIN(Time) FROM Frames");
            ResultSet min_rt_res = pstmt.executeQuery();
            double min_rt_tmp = min_rt_res.getDouble(1)/60.0;

            pstmt = connection.prepareStatement("SELECT MAX(Time) FROM Frames");
            ResultSet max_rt_res = pstmt.executeQuery();
            double max_rt_tmp = max_rt_res.getDouble(1)/60.0;

            // Instrument name
            pstmt = connection.prepareStatement("SELECT Value FROM GlobalMetadata WHERE Key='InstrumentName'");
            ResultSet instrument_res = pstmt.executeQuery();
            ms_instrument = instrument_res.getString(1);

            // extract isolation window information
            pstmt = connection.prepareStatement("SELECT IsolationMz, IsolationWidth, CollisionEnergy FROM DiaFrameMsMsWindows ORDER BY IsolationMz ASC");
            ResultSet iso_win_res = pstmt.executeQuery();
            while(iso_win_res.next()){
                double isolation_mz = iso_win_res.getDouble("IsolationMz");
                double isolation_width = iso_win_res.getDouble("IsolationWidth");
                double ce = iso_win_res.getDouble("CollisionEnergy");
                double [] iso_win_range = get_isolation_window(isolation_mz,isolation_width);
                String isoWinID = IsolationWindow.generate_id(iso_win_range[0],iso_win_range[1]);
                if(!this.isolationWindowMap.containsKey(isoWinID)){
                    this.isolationWindowMap.put(isoWinID,new IsolationWindow(iso_win_range[0],iso_win_range[1]));
                    this.isolationWindowMap.get(isoWinID).ce = ce;
                }
                System.out.println("Isolation window: "+isoWinID+" -> "+iso_win_range[0]+" - "+iso_win_range[1]+", CE: "+ce);
            }


            pstmt.close();
            connection.close();

            System.out.println("Meta information extracted directly from the input file:");
            System.out.println("Fragment ion m/z range: " + min_fragment_mz + "-" + max_fragment_mz);
            System.out.println("Precursor ion m/z range: " + min_precursor_mz + "-" + max_precursor_mz);
            System.out.println("Retention time range: " + min_rt_tmp + "-" + max_rt_tmp);
            System.out.println("MS instrument: " + ms_instrument);
            this.min_fragment_ion_intensity = 0.0;
            System.out.println("Min fragment ion intensity: "+this.min_fragment_ion_intensity);

            if (this.fragment_ion_mz_min > min_fragment_mz) {
                this.fragment_ion_mz_min = min_fragment_mz;
            }
            if (this.fragment_ion_mz_max < max_fragment_mz) {
                this.fragment_ion_mz_max = max_fragment_mz;
            }

            if(this.rt_min > min_rt_tmp){
                this.rt_min = min_rt_tmp;
            }
            if(this.rt_max < max_rt_tmp){
                this.rt_max = max_rt_tmp;
            }

            if(this.precursor_ion_mz_max < max_precursor_mz){
                this.precursor_ion_mz_max = max_precursor_mz;
            }
            if(this.precursor_ion_mz_min > min_precursor_mz){
                this.precursor_ion_mz_min = min_precursor_mz;
            }
            if(this.nce==0) {
                this.nce = CParameter.NCE;
            }

        } catch (SQLException e) {
            throw new RuntimeException(e);
        }

        this.rt_max = this.rt_max + 0.1;

        System.out.println("Fragment ion m/z range: " + this.fragment_ion_mz_min + "-" + this.fragment_ion_mz_max);
        System.out.println("Precursor ion m/z range: " + this.precursor_ion_mz_min + "-" + this.precursor_ion_mz_max);
        System.out.println("Retention time range: " + this.rt_min + "-" + this.rt_max);
        System.out.println("Min fragment ion intensity: "+this.min_fragment_ion_intensity);
    }

    public long get_fragment_ion_mz_bin_index(double mz){
        return Math.round((mz - this.fragment_ion_mz_min) / this.fragment_ion_mz_bin_size);
    }

    public static double[] get_isolation_window(double isolation_mz, double isolation_width){
        return new double[]{isolation_mz - isolation_width/2.0, isolation_mz + isolation_width/2.0};
    }

    /**
     * Generate a map from m/z to collision energy
     */
    public void generate_mz2ce_map(){
        // iterate isolationWindowMap
        for(String isoWinID: this.isolationWindowMap.keySet()){
            IsolationWindow isoWin = this.isolationWindowMap.get(isoWinID);
            // for each isolation window, consider all values from mz_lower + 0.2 to mz_upper - 0.2 with step size 0.1
            // convert to integer by rounding with multiplying 10
            int mz_lower_int = (int)Math.round((isoWin.mz_lower + 0.2) * ce_map_mz_step_size);
            int mz_upper_int = (int)Math.round((isoWin.mz_upper - 0.2) * ce_map_mz_step_size);
            double ce =  this.isolationWindowMap.get(isoWinID).ce;
            for(int mz_int = mz_lower_int; mz_int <= mz_upper_int; mz_int +=1){
                this.mz2ce.put(mz_int,ce);
            }
        }
    }

    public double get_ce_for_mz(double mz){
        int mz_int = (int)Math.round(mz * ce_map_mz_step_size);
        if(this.mz2ce.containsKey(mz_int)){
            return this.mz2ce.get(mz_int);
        }else{
            // check isolationWindowMap
            double ce = -1;
            for(String isoWinID: this.isolationWindowMap.keySet()){
                IsolationWindow isoWin = this.isolationWindowMap.get(isoWinID);
                if(mz >= isoWin.mz_lower && mz <= isoWin.mz_upper){
                    ce = isoWin.ce;
                    break;
                }
            }
            if(ce <= 0){
                Cloger.getInstance().logger.info("Use default CE for m/z: "+mz + " -> "+ CParameter.NCE);
                // if mz is less than the min mz in the map, use the CE of the lowest mz window
                double min_mz_lower = Double.MAX_VALUE;
                double max_mz_upper = 0;
                double ce_min = CParameter.NCE;
                double ce_max = CParameter.NCE;
                for(String isoWinID: this.isolationWindowMap.keySet()){
                    IsolationWindow isoWin = this.isolationWindowMap.get(isoWinID);
                    if(isoWin.mz_lower < min_mz_lower){
                        min_mz_lower = isoWin.mz_lower;
                        ce_min = isoWin.ce;
                    }
                    if(isoWin.mz_upper > max_mz_upper) {
                        max_mz_upper = isoWin.mz_upper;
                        ce_max = isoWin.ce;
                    }
                }
                if(mz <= min_mz_lower){
                    ce = ce_min;
                }else if(mz >= max_mz_upper){
                    ce = ce_max;
                }else{
                    ce = CParameter.NCE;
                }
            }
            return ce;
        }
    }

}
