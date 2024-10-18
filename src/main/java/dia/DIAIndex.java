package main.java.dia;

import com.compomics.util.experiment.mass_spectrometry.spectra.Precursor;
import com.compomics.util.experiment.mass_spectrometry.spectra.Spectrum;
import com.google.common.math.Quantiles;
import main.java.ai.PeptideMatch;
import main.java.util.Cloger;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.util.FastMath;
import umich.ms.datatypes.scan.IScan;

import java.util.*;
import java.util.concurrent.ConcurrentMap;

import static java.util.stream.Collectors.*;

public class DIAIndex {


    public DIAMeta meta = new DIAMeta();
    public int ms_level = 2;
    public double fragment_ion_intensity_threshold = 0.0;
    public HashMap<String, ConcurrentMap<Long, ArrayList<JFragmentIon>>> frag_ion_index = new HashMap<>();
    public HashMap<String, HashMap<Integer,Integer>> isolation_win2scan2index = new HashMap<>();
    public HashMap<String, HashMap<Integer,Integer>> isolation_win2index2scan = new HashMap<>();
    public HashMap<String, HashMap<Integer,Double>> isolation_win2scan2rt = new HashMap<>();
    public HashSet<String> target_isolation_wins = new HashSet<>();
    public HashMap<Integer, Spectrum> scan2spectrum = new HashMap<>();
    public int min_scan_for_peak = 3;
    public int sg_smoothing_data_points = 5;


    public int get_index_by_scan(String isolation_win, int scan){
        return this.isolation_win2scan2index.get(isolation_win).get(scan);
    }

    public int get_scan_by_index(String isolation_win, int index){
        return this.isolation_win2index2scan.get(isolation_win).get(index);
    }

    public double get_rt_by_scan(String isolation_win, int scan){
        return this.isolation_win2scan2rt.get(isolation_win).get(scan);
    }

    private void add_spectrum(IScan scan){
        Precursor precursor = new Precursor(scan.getRt(),scan.getPrecursor().getMzTarget(),new int[]{2});
        Spectrum spectrum = new Spectrum(precursor,scan.getSpectrum().getMZs(),scan.getSpectrum().getIntensities());
        this.scan2spectrum.put(scan.getNum(),spectrum);
    }

    public Spectrum get_spectrum_by_scan(int scan_number){
        return this.scan2spectrum.get(scan_number);
    }

    public void index(){

        Set<Map.Entry<Integer, IScan>> num2scanEntries = this.meta.num2scanMap.entrySet();
        //HashMap<String, ArrayList<IScan>> spectrum_batch = new HashMap<>(CParameter.batch_size_query_spectra);
        HashMap<String,ArrayList<IScan>> spectrum_batch = new HashMap<>();
        int spectrum_batch_current_size = 0;
        int total_spectra = 0;
        double mz_start;
        double mz_end;
        HashMap<String,Integer> isolation_win2cur_index = new HashMap<>();
        for (Map.Entry<Integer, IScan> next : num2scanEntries) {
            IScan scan = next.getValue();
            if (scan.getSpectrum() != null) {
                if (scan.getMsLevel() == this.ms_level) {
                    total_spectra++;
                    String isoWinID;
                    if(this.ms_level==2) {
                        mz_start = scan.getPrecursor().getMzRangeStart();
                        mz_end = scan.getPrecursor().getMzRangeEnd();
                        isoWinID = IsolationWindow.generate_id(mz_start, mz_end);
                    }else{
                        isoWinID = "0";
                    }
                    if(this.target_isolation_wins.contains(isoWinID)) {
                        if (!spectrum_batch.containsKey(isoWinID)) {
                            spectrum_batch.put(isoWinID, new ArrayList<>());
                        }
                        spectrum_batch.get(isoWinID).add(scan);


                        if(this.ms_level==2) {
                            this.add_spectrum(scan);
                        }

                        if(!isolation_win2cur_index.containsKey(isoWinID)){
                            isolation_win2cur_index.put(isoWinID, -1);
                        }
                        isolation_win2cur_index.put(isoWinID,isolation_win2cur_index.get(isoWinID)+1);
                        if(!this.isolation_win2scan2index.containsKey(isoWinID)){
                            this.isolation_win2scan2index.put(isoWinID, new HashMap<>());
                        }
                        this.isolation_win2scan2index.get(isoWinID).put(scan.getNum(),isolation_win2cur_index.get(isoWinID));

                        if(!this.isolation_win2index2scan.containsKey(isoWinID)){
                            this.isolation_win2index2scan.put(isoWinID, new HashMap<>());
                        }
                        this.isolation_win2index2scan.get(isoWinID).put(this.isolation_win2scan2index.get(isoWinID).get(scan.getNum()),scan.getNum());

                        if(!this.isolation_win2scan2rt.containsKey(isoWinID)){
                            this.isolation_win2scan2rt.put(isoWinID, new HashMap<>());
                        }
                        this.isolation_win2scan2rt.get(isoWinID).put(scan.getNum(),scan.getRt());

                    }
                }
            }
        }

        JFragmentIon.meta = meta;
        for(String isoWinID: target_isolation_wins) {
            // System.out.print("Total spectra for "+isoWinID+": ");
            // System.out.println(spectrum_batch.get(isoWinID).size());
            frag_ion_index.put(isoWinID,spectrum_batch.get(isoWinID).parallelStream()
                    .map(scan -> this.generate_frag_ion_index_for_one_spectrum(scan, this.fragment_ion_intensity_threshold))
                    .flatMap(Collection::stream)
                    .unordered()
                    .collect(groupingByConcurrent(JFragmentIon::get_frag_mz_bin, toCollection((ArrayList::new)))));

            frag_ion_index.get(isoWinID).keySet().parallelStream().forEach(i -> frag_ion_index.get(isoWinID).get(i).sort(comparator_rt_for_fragment_ion_from_min2max));
        }
        // frag_ion_index.forEach((key, value) -> System.out.println(key + " => " + value.size()));

        for(String isoWin: frag_ion_index.keySet()){
            // System.out.println(isoWin);
        }
    }

    public final Comparator<JFragmentIon> comparator_rt_for_fragment_ion_from_min2max = Comparator.comparingDouble(JFragmentIon::get_scan);

    public ArrayList<JFragmentIon> generate_frag_ion_index_for_one_spectrum(IScan scan, double intensity_threshold) {
        double [] mz = scan.getSpectrum().getMZs();
        double [] intensity = scan.getSpectrum().getIntensities();
        double max_intensity = StatUtils.max(intensity);
        double min_intensity_cutoff = max_intensity * intensity_threshold;
        int scan_number = scan.getNum();
        ArrayList<JFragmentIon> ion_index = new ArrayList<>(mz.length);
        for (int i=0;i<mz.length;i++) {
            if(intensity[i] >= min_intensity_cutoff) {
                JFragmentIon fragmentIon = new JFragmentIon((float) mz[i], (float) intensity[i], scan_number);
                fragmentIon.rt = scan.getRt().floatValue();
                ion_index.add(fragmentIon);
            }
        }
        return ion_index;
    }

    public double[] detect_best_ion(RealMatrix x, int index_start, int index_end, int index_apex, PeptideMatch pMatch){
        RealMatrix x_peak = x.getSubMatrix(0,x.getRowDimension()-1,index_start,index_end);

        double []max_intensity_list = x.getColumn(index_apex);
        double max_intensity = StatUtils.max(max_intensity_list);
        // double []cor = new double[x_peak.getRowDimension()];
        int n_ions = x_peak.getRowDimension();
        double [][] cor_matrix = new double[n_ions][n_ions];
        double left_boundary_median_intensity = Quantiles.median().compute(x.getColumn(index_start))*1.5;
        double right_boundary_median_intensity = Quantiles.median().compute(x.getColumn(index_end))*1.5;

        int [] skewed_peaks = new int[n_ions];
        for(int i=0;i<n_ions;i++){
            cor_matrix[i][i] = 1;
            for(int j=0;j<n_ions;j++){
                if(i!=j) {
                    cor_matrix[i][j] = new PearsonsCorrelation().correlation(x_peak.getRow(i), x_peak.getRow(j));
                    if (Double.isNaN(cor_matrix[i][j]) || cor_matrix[i][j] < 0) {
                        cor_matrix[i][j] = 0;
                    }
                    cor_matrix[j][i] = cor_matrix[i][j];
                }
            }

            boolean left_boundary_skewed;
            boolean right_boundary_skewed;
            if(max_intensity_list[i] >= 0.5 * max_intensity){
                left_boundary_skewed = x.getRow(i)[index_start] > left_boundary_median_intensity && x.getRow(i)[index_start] > 0.10 * max_intensity_list[i];
                right_boundary_skewed = x.getRow(i)[index_end] > right_boundary_median_intensity && x.getRow(i)[index_end] > 0.10 * max_intensity_list[i];
            }else{
                left_boundary_skewed = x.getRow(i)[index_start] > left_boundary_median_intensity && x.getRow(i)[index_start] > 0.25 * max_intensity_list[i];
                right_boundary_skewed = x.getRow(i)[index_end] > right_boundary_median_intensity && x.getRow(i)[index_end] > 0.25 * max_intensity_list[i];
            }

            if(left_boundary_skewed || right_boundary_skewed){
                if(left_boundary_skewed && right_boundary_skewed){
                    skewed_peaks[i] = 2;
                }else {
                    skewed_peaks[i] = 1;
                }
            }
        }

        pMatch.skewed_peaks = skewed_peaks;

        double max_cor = -1;
        double cur_cor = -1;
        int best_i = 0;
        for(int i=0;i<n_ions;i++) {
            if(skewed_peaks[i] >= 1){
                continue;
            }
            cur_cor = weighted_mean(cor_matrix[i], max_intensity_list, false);
            if(max_cor < cur_cor){
                max_cor = cur_cor;
                best_i = i;
            }
        }
        return cor_matrix[best_i];
    }

    private double weighted_mean(double []x, double []weights, boolean log_transform){
        double sum = 0;
        double sum_weights = 0;
        double w;
        for(int i=0;i<x.length;i++){
            if(log_transform){
                w = FastMath.log(weights[i]+1);
                sum = sum + x[i] * w;
                sum_weights = sum_weights + w;
            }else {
                sum = sum + x[i] * weights[i];
                sum_weights = sum_weights + weights[i];
            }
        }
        return sum/sum_weights;
    }

}
