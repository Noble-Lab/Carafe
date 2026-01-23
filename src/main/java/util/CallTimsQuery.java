package main.java.util;

import org.apache.commons.lang3.StringUtils;

import main.java.input.CParameter;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static main.java.ai.AIGear.get_jar_path;

/**
 * This class is used to call the rust library timsquery to extract spectra from
 * TIMS-TOF data.
 */
public class CallTimsQuery {

    /**
     * The unit of retention time: min or sec
     */
    public String rt_unit = "min";

    /**
     * The retention time window size
     */
    public double rt_win = 5 / 60.0; // in minutes

    public double mobility = 3.0;

    public double quad = 0.1;

    public double tol = 15.0;
    public String tolu = "ppm";

    public double itol = 15.0;
    public String itolu = "ppm";
    public double itol_shift = 0.0;

    /**
     * Get the platform-specific timsquery binary path.
     * The binaries are located in: bin/timsquery/<platform>/
     * - Windows: bin/timsquery/windows/timsquery_cli.exe
     * - Linux: bin/timsquery/linux/timsquery_cli
     * - macOS: bin/timsquery/macos/timsquery_cli
     */
    private String getTimsQueryBinaryPath() {
        String osName = System.getProperty("os.name").toLowerCase();
        String platformFolder;
        String binaryName;
        boolean isWindows = osName.contains("win");

        if (isWindows) {
            platformFolder = "windows";
            binaryName = "timsquery_cli.exe";
        } else if (osName.contains("mac") || osName.contains("darwin")) {
            platformFolder = "macos";
            binaryName = "timsquery_cli";
        } else {
            // Default to Linux for other Unix-like systems
            platformFolder = "linux";
            binaryName = "timsquery_cli";
        }

        // Try jar path first: bin/timsquery/<platform>/<binary>
        String timsQueryBin = get_jar_path() + File.separator + "bin" + File.separator + "timsquery"
                + File.separator + platformFolder + File.separator + binaryName;
        File f = new File(timsQueryBin);
        if (f.exists()) {
            // Set executable permission on Linux/macOS
            if (!isWindows && !f.canExecute()) {
                boolean success = f.setExecutable(true);
                if (success) {
                    Cloger.getInstance().logger.info("Set executable permission for: " + timsQueryBin);
                } else {
                    Cloger.getInstance().logger.warn("Failed to set executable permission for: " + timsQueryBin);
                }
            }
            Cloger.getInstance().logger.info("timsquery found at: " + timsQueryBin);
            return timsQueryBin;
        }

        // Try current directory as fallback
        timsQueryBin = "bin" + File.separator + "timsquery" + File.separator + platformFolder
                + File.separator + binaryName;
        f = new File(timsQueryBin);
        if (f.exists()) {
            // Set executable permission on Linux/macOS
            if (!isWindows && !f.canExecute()) {
                boolean success = f.setExecutable(true);
                if (success) {
                    Cloger.getInstance().logger.info("Set executable permission for: " + timsQueryBin);
                } else {
                    Cloger.getInstance().logger.warn("Failed to set executable permission for: " + timsQueryBin);
                }
            }
            timsQueryBin = f.getAbsolutePath();
            Cloger.getInstance().logger.info("timsquery found at: " + timsQueryBin);
            return timsQueryBin;
        }

        Cloger.getInstance().logger.error("TIMSQUERY binary not found! Expected at: bin/timsquery/"
                + platformFolder + "/" + binaryName);
        return null;
    }

    public void run_ms2_spectra_query(String ms_file, String psm_query_file, String out_file) {
        String tims_query_bin = getTimsQueryBinaryPath();
        if (tims_query_bin == null) {
            return;
        }

        // get the folder of out_file
        File OF = new File(psm_query_file);
        String out_dir = OF.getParent();
        String out_parameter_file = out_dir + File.separator + "spectra_query_parameters.json";
        try {
            generate_timsquery_spectra_parameter_file(out_parameter_file);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        String[] cmd_list_short = new String[] {
                tims_query_bin,
                "query-index",
                "-a", "spectrum-aggregator",
                "-r", ms_file,
                "-t", out_parameter_file,
                "-e", psm_query_file,
                "-f", "ndjson",
                "-o", out_file
        };
        ArrayList<String> cmd_list = new ArrayList<>(Arrays.asList(cmd_list_short));

        System.out.println("cmd: " + StringUtils.join(cmd_list));
        // convert cmd_list to String []
        String[] cmd = new String[cmd_list.size()];
        cmd = cmd_list.toArray(cmd);
        run_cmd(cmd);
    }

    public void run_xic_query(String ms_file, String psm_query_file, String out_file) {
        String tims_query_bin = getTimsQueryBinaryPath();
        if (tims_query_bin == null) {
            return;
        }

        // get the folder of out_file
        File OF = new File(psm_query_file);
        String out_dir = OF.getParent();
        String out_parameter_file = out_dir + File.separator + "xic_query_parameters.json";
        try {
            generate_timsquery_xic_parameter_file(out_parameter_file);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        String[] cmd_list_short = new String[] {
                tims_query_bin,
                "query-index",
                "-a", "chromatogram-aggregator",
                "-r", ms_file,
                "-t", out_parameter_file,
                "-e", psm_query_file,
                "-f", "ndjson",
                "-o", out_file
        };
        ArrayList<String> cmd_list = new ArrayList<>(Arrays.asList(cmd_list_short));

        System.out.println("cmd: " + StringUtils.join(cmd_list));
        // convert cmd_list to String []
        String[] cmd = new String[cmd_list.size()];
        cmd = cmd_list.toArray(cmd);
        run_cmd(cmd);
    }

    private boolean run_cmd(String[] cmd) {
        boolean pass = true;
        Runtime rt = Runtime.getRuntime();
        Process p;
        try {
            p = rt.exec(cmd);
        } catch (IOException e) {
            pass = false;
            throw new RuntimeException(e);
        }

        List<String> filterPatterns;
        if (CParameter.verbose == CParameter.VerboseType.DEBUG){
            filterPatterns = new ArrayList<>();
        }else{
            // Define patterns to filter out from console output
            // Add patterns here to suppress specific messages
            filterPatterns = List.of(
                    // Add specific patterns to filter.
                    " INFO.*main_query_index",
                    " INFO.*read_parquet_peaks",
                    " INFO.*timscentroid::indexing",
                    " INFO.*check_bucket_sorted_heuristic",
                    "ERROR.*main_query_index.*metadata.json not found or unreadable"
            );
        }

        StreamLog errorLog = new StreamLog(p.getErrorStream(), "TimsQuery => Error:",
                true, filterPatterns);
        StreamLog stdLog = new StreamLog(p.getInputStream(), "TimsQuery => Message:",
                true, filterPatterns);

        errorLog.start();
        stdLog.start();

        try {
            int exitValue = p.waitFor();
            if (exitValue != 0) {
                pass = false;
                Cloger.getInstance().logger.error("TimsQuery => Error: " + exitValue);
            }
        } catch (InterruptedException e) {
            pass = false;
            throw new RuntimeException(e);
        }

        try {
            errorLog.join();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        try {
            stdLog.join();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

        return pass;
    }

    private void generate_timsquery_spectra_parameter_file(String out_file) throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(out_file));
        bw.write("{\n");
        bw.write("\"ms\": { \"" + this.itolu + "\": [" + (this.itol - this.itol_shift) + ","
                + (this.itol + this.itol_shift) + "]},\n");
        bw.write("\"rt\": { \"minutes\": [" + this.rt_win + "," + this.rt_win + "]},\n");
        bw.write("\"mobility\": { \"percent\": [" + this.mobility + "," + this.mobility + "]},\n");
        bw.write("\"quad\": { \"absolute\": [" + this.quad + "," + this.quad + "]}\n");
        bw.write("}\n");
        bw.close();
    }

    private void generate_timsquery_xic_parameter_file(String out_file) throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(out_file));
        bw.write("{\n");
        bw.write("\"ms\": { \"" + this.itolu + "\": [" + (this.itol - this.itol_shift) + ","
                + (this.itol + this.itol_shift) + "]},\n");
        bw.write("\"rt\": { \"minutes\": [" + this.rt_win + "," + this.rt_win + "]},\n");
        bw.write("\"mobility\": { \"percent\": [" + this.mobility + "," + this.mobility + "]},\n");
        bw.write("\"quad\": { \"absolute\": [" + this.quad + "," + this.quad + "]}\n");
        bw.write("}\n");
        bw.close();
    }
}
