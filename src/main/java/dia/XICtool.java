package main.java.dia;

import com.google.common.math.Quantiles;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.util.FastMath;
import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.commons.math3.stat.StatUtils;


public class XICtool {

    /**
     * Key: pepID (Integer value), value: hashmap (key: charge, value: peptide peak group)
     */
    public ConcurrentHashMap<Integer,HashMap<Integer,ArrayList<PeptidePeak>>> peptides = new ConcurrentHashMap<>();
    public float peak_boundary_low_intensity_limit = 0.1f;


    public void refine_peak(RealMatrix x, PeptidePeak peak, double [] fragment_ion_correlations, double min_cor, boolean debug){
        // only use fragment ions with correlation >= min_cor, e.g., >= 0.75
        if(peak.boundary_right_index - peak.boundary_left_index + 1 >=5) {
            ArrayList<Integer> cor_index = new ArrayList<>(fragment_ion_correlations.length);
            for (int i = 0; i < fragment_ion_correlations.length; i++) {
                if (fragment_ion_correlations[i] >= min_cor) {
                    cor_index.add(i);
                }
            }
            if (cor_index.size()>=3) {
                int n_scans = (int) (peak.boundary_right_index - peak.boundary_left_index + 1);
                int[] scan_index = new int[n_scans];
                int k = 0;
                for (long i = peak.boundary_left_index; i <= peak.boundary_right_index; i++) {
                    scan_index[k] = (int) i;
                    k = k + 1;
                }
                RealMatrix new_x = x.getSubMatrix(cor_index.stream().mapToInt(i -> i).toArray(), scan_index);
                double[] median_peaks = new double[n_scans];
                if (cor_index.size() == 1) {
                    median_peaks = new_x.getRow(0);
                } else {
                    for (int i = 0; i < n_scans; i++) {
                        // median_peaks[i] = StatUtils.percentile(new_x.getColumn(i), 50);
                        median_peaks[i] = Quantiles.median().compute(new_x.getColumn(i));
                    }
                }
                PeptidePeak new_peak = find_max_peak(median_peaks);

                if ((new_peak.boundary_right_index - new_peak.boundary_left_index + 1) >= 2) {
                    int left_index = (int) peak.boundary_left_index;
                    peak.boundary_left_index = left_index + new_peak.boundary_left_index;
                    peak.boundary_right_index = left_index + new_peak.boundary_right_index;
                    peak.apex_index = left_index + new_peak.apex_index;
                    peak.min_smoothed_intensity = new_peak.min_smoothed_intensity;
                } else {
                    // for debugging
                    if(false){
                        System.out.println("1:" + peak.boundary_left_index + "\t" + peak.apex_index + "\t" + peak.boundary_right_index);
                        int left_index = (int) peak.boundary_left_index;
                        long boundary_left_index = left_index + new_peak.boundary_left_index;
                        long boundary_right_index = left_index + new_peak.boundary_right_index;
                        long apex_index = left_index + new_peak.apex_index;
                        System.out.println("2:" + boundary_left_index + "\t" + apex_index + "\t" + boundary_right_index);
                        String file = "1-" + peak.boundary_left_index + "_" + peak.apex_index + "_" + peak.boundary_right_index + "_2-" + boundary_left_index + "_" + apex_index + "_" + boundary_right_index + ".txt";
                        try {
                            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(file));

                            for(int i=0;i<x.getRowDimension();i++){

                                    bufferedWriter.write(fragment_ion_correlations[i]+"\t"+StringUtils.join(x.getRow(i),'\t')+"\n");

                            }

                            bufferedWriter.close();
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    }

                }
            }
        }
    }

    public double[] detect_best_ion(RealMatrix x, int index_start, int index_end, int index_apex){
        RealMatrix x_peak = x.getSubMatrix(0,x.getRowDimension()-1,index_start,index_end);
        double []max_intensity_list = x.getColumn(index_apex);
        // double []cor = new double[x_peak.getRowDimension()];
        // The number of ions in the matrix
        int n_ions = x_peak.getRowDimension();
        double [][] cor_matrix = new double[n_ions][n_ions];
        //double []max_intensity_list = new double[n_ions];
        for(int i=0;i<n_ions;i++){
            cor_matrix[i][i] = 1;
            // max_intensity_list[i] = StatUtils.max(x_peak.getRow(i))+1;
            for(int j=0;j<n_ions;j++){
                if(i!=j) {
                    cor_matrix[i][j] = new PearsonsCorrelation().correlation(x_peak.getRow(i), x_peak.getRow(j));
                    if (Double.isNaN(cor_matrix[i][j]) || cor_matrix[i][j] < 0) {
                        cor_matrix[i][j] = 0;
                    }
                    cor_matrix[j][i] = cor_matrix[i][j];
                }
            }
        }

        double max_cor = -1;
        double cur_cor = -1;
        int best_i = 0;
        for(int i=0;i<n_ions;i++) {
            cur_cor = weighted_mean(cor_matrix[i], max_intensity_list, false);
            if(max_cor < cur_cor){
                max_cor = cur_cor;
                best_i = i;
            }
        }
        // from small to large
        //Arrays.sort(cor_matrix[best_i]);
        return cor_matrix[best_i];
    }

    public double weighted_mean(double []x, double []weights, boolean log_transform){
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

    public PeptidePeak find_max_peak(double [] median_peaks){
        long max_index = 0;
        double max_int = 0;
        for(int i=0;i<median_peaks.length;i++){
            if(max_int < median_peaks[i]){
                max_int = median_peaks[i];
                max_index = i;
            }
        }

        PeptidePeak peak = new PeptidePeak();
        peak.apex_index = max_index;
        double int_limit = this.peak_boundary_low_intensity_limit * max_int;

        if (max_index == 0L) {
            // max_index = Math.round(median_peaks.length / 2.0);
            peak.boundary_left_index = 0;
        }else{
            // for left index
            double last_int = max_int;
            boolean boundary_found = false;
            int drop_n = 0;
            for (int i = (int) max_index - 1; i >= 0; i--) {
                if (median_peaks[i] < last_int) {
                    last_int = median_peaks[i];
                }else if(last_int > int_limit){
                    drop_n = drop_n + 1;
                    if(drop_n>=2){
                        peak.boundary_left_index = i + 1;
                        boundary_found = true;
                        break;
                    }else{
                        last_int = median_peaks[i];
                    }
                } else {
                    peak.boundary_left_index = i + 1;
                    boundary_found = true;
                    break;
                }
            }
            if(!boundary_found){
                peak.boundary_left_index = 0;
            }

            long left_index = peak.boundary_left_index;
            long min_intensity_index = peak.boundary_left_index;
            double min_intensity = max_int;
            for (int i = (int) max_index - 1; i >= left_index; i--) {
                if(median_peaks[i] < min_intensity){
                    min_intensity = median_peaks[i];
                    min_intensity_index = i;
                }
            }
            peak.boundary_left_index = min_intensity_index;
        }

        if (max_index == median_peaks.length - 1) {
            // max_index = Math.round(median_peaks.length / 2.0);
            peak.boundary_right_index = median_peaks.length - 1;
        }else{
            // for right index
            double last_int = max_int;
            int drop_n = 0;
            boolean boundary_found = false;
            for (int i = (int) max_index + 1; i < median_peaks.length; i++) {
                if (median_peaks[i] < last_int) {
                    //System.out.println(i+"\t"+median_peaks[i]+"\t"+last_int+"\t"+int_limit);
                    last_int = median_peaks[i];
                }else if(last_int > int_limit){
                    drop_n = drop_n + 1;
                    if(drop_n>=2){
                        peak.boundary_right_index = i - 1;
                        boundary_found = true;
                        break;
                    }else{
                        last_int = median_peaks[i];
                    }
                } else {
                    peak.boundary_right_index = i - 1;
                    boundary_found = true;
                    break;
                }
            }
            if(!boundary_found){
                peak.boundary_right_index = median_peaks.length - 1;
            }

            long right_index = peak.boundary_right_index;
            long min_intensity_index = peak.boundary_right_index;
            double min_intensity = max_int;
            for (int i = (int) max_index + 1; i <=right_index; i++) {
                if(median_peaks[i] < min_intensity){
                    min_intensity = median_peaks[i];
                    min_intensity_index = i;
                }
            }
            peak.boundary_right_index = min_intensity_index;
        }

        peak.min_smoothed_intensity = StatUtils.min(median_peaks, (int) peak.boundary_left_index, (int) (peak.boundary_right_index-peak.boundary_left_index+1L));
        peak.min_smoothed_intensity = Math.max(peak.min_smoothed_intensity,0);

        return peak;
    }

    public PeptidePeak find_max_peak(double [] median_peaks, int apex_index){
        long max_index = apex_index;
        double max_int = median_peaks[apex_index];

        PeptidePeak peak = new PeptidePeak();
        peak.apex_index = max_index;
        double int_limit = this.peak_boundary_low_intensity_limit * max_int;

        if (max_index == 0L) {
            // max_index = Math.round(median_peaks.length / 2.0);
            peak.boundary_left_index = 0;
        }else{
            // for left index
            double last_int = max_int;
            boolean boundary_found = false;
            int drop_n = 0;
            for (int i = (int) max_index - 1; i >= 0; i--) {
                if (median_peaks[i] < last_int) {
                    last_int = median_peaks[i];
                }else if(last_int > int_limit){
                    drop_n = drop_n + 1;
                    if(drop_n>=2){
                        peak.boundary_left_index = i + 1;
                        boundary_found = true;
                        break;
                    }else{
                        last_int = median_peaks[i];
                    }
                } else {
                    peak.boundary_left_index = i + 1;
                    boundary_found = true;
                    break;
                }
            }
            if(!boundary_found){
                peak.boundary_left_index = 0;
            }

            long left_index = peak.boundary_left_index;
            long min_intensity_index = peak.boundary_left_index;
            double min_intensity = max_int;
            for (int i = (int) max_index - 1; i >= left_index; i--) {
                if(median_peaks[i] < min_intensity){
                    min_intensity = median_peaks[i];
                    min_intensity_index = i;
                }
            }
            peak.boundary_left_index = min_intensity_index;
        }

        if (max_index == median_peaks.length - 1) {
            // max_index = Math.round(median_peaks.length / 2.0);
            peak.boundary_right_index = median_peaks.length - 1;
        }else{
            // for right index
            double last_int = max_int;
            int drop_n = 0;
            boolean boundary_found = false;
            for (int i = (int) max_index + 1; i < median_peaks.length; i++) {
                if (median_peaks[i] < last_int) {
                    //System.out.println(i+"\t"+median_peaks[i]+"\t"+last_int+"\t"+int_limit);
                    last_int = median_peaks[i];
                }else if(last_int > int_limit){
                    drop_n = drop_n + 1;
                    if(drop_n>=2){
                        peak.boundary_right_index = i - 1;
                        boundary_found = true;
                        break;
                    }else{
                        last_int = median_peaks[i];
                    }
                } else {
                    peak.boundary_right_index = i - 1;
                    boundary_found = true;
                    break;
                }
            }
            if(!boundary_found){
                peak.boundary_right_index = median_peaks.length - 1;
            }

            long right_index = peak.boundary_right_index;
            long min_intensity_index = peak.boundary_right_index;
            double min_intensity = max_int;
            for (int i = (int) max_index + 1; i <=right_index; i++) {
                if(median_peaks[i] < min_intensity){
                    min_intensity = median_peaks[i];
                    min_intensity_index = i;
                }
            }
            peak.boundary_right_index = min_intensity_index;
        }

        peak.min_smoothed_intensity = StatUtils.min(median_peaks, (int) peak.boundary_left_index, (int) (peak.boundary_right_index-peak.boundary_left_index+1L));
        peak.min_smoothed_intensity = Math.max(peak.min_smoothed_intensity,0);

        return peak;
    }


}
