package main.java.ai;

import ai.djl.Device;
import ai.djl.util.cuda.CudaUtils;
import com.alibaba.fastjson.JSON;
import com.compomics.util.experiment.biology.ions.Ion;
import com.compomics.util.experiment.biology.ions.NeutralLoss;
import com.compomics.util.experiment.biology.ions.impl.ElementaryIon;
import com.compomics.util.experiment.biology.ions.impl.PeptideFragmentIon;
import com.compomics.util.experiment.biology.proteins.Peptide;
import com.compomics.util.experiment.identification.matches.IonMatch;
import com.compomics.util.experiment.identification.matches.ModificationMatch;
import com.compomics.util.experiment.identification.protein_sequences.SingleProteinSequenceProvider;
import com.compomics.util.experiment.identification.spectrum_annotation.AnnotationParameters;
import com.compomics.util.experiment.identification.spectrum_annotation.NeutralLossesMap;
import com.compomics.util.experiment.identification.spectrum_annotation.SpecificAnnotationParameters;
import com.compomics.util.experiment.identification.spectrum_annotation.SpectrumAnnotator;
import com.compomics.util.experiment.identification.spectrum_annotation.spectrum_annotators.PeptideSpectrumAnnotator;
import com.compomics.util.experiment.identification.spectrum_assumptions.PeptideAssumption;
import com.compomics.util.experiment.io.biology.protein.SequenceProvider;
import com.compomics.util.experiment.mass_spectrometry.spectra.Spectrum;
import com.compomics.util.parameters.identification.advanced.SequenceMatchingParameters;
import com.compomics.util.parameters.identification.search.ModificationParameters;
import com.google.common.base.Splitter;
import com.google.common.math.Quantiles;
import main.java.input.ModificationUtils;
import main.java.db.DBGear;
import main.java.dia.*;
import main.java.input.*;
import main.java.util.StreamLog;
import main.java.util.Cloger;
import main.java.util.MgfUtils;
import main.java.xic.SGFilter;
import main.java.xic.SGFilter3points;
import main.java.xic.SGFilter5points;
import main.java.xic.SGFilter7points;
import org.apache.avro.Schema;
import org.apache.avro.generic.GenericRecord;
import org.apache.avro.reflect.ReflectData;
import org.apache.commons.cli.*;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.math3.exception.NumberIsTooSmallException;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.correlation.SpearmansCorrelation;
import org.apache.commons.math3.util.FastMath;
import org.apache.hadoop.conf.Configuration;
import org.apache.parquet.avro.AvroParquetReader;
import org.apache.parquet.avro.AvroParquetWriter;
import org.apache.avro.generic.GenericData;
import org.apache.parquet.hadoop.ParquetFileWriter;
import org.apache.parquet.hadoop.ParquetReader;
import org.apache.parquet.hadoop.ParquetWriter;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.apache.parquet.io.LocalInputFile;
import org.apache.parquet.io.LocalOutputFile;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.IntColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.io.csv.CsvReadOptions;
import tech.tablesaw.io.csv.CsvWriteOptions;
import java.io.*;
import java.lang.management.MemoryUsage;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.sql.SQLException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;
import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toMap;

public class AIGear {

    public double fdr_cutoff = 0.01;

    /**
     * The number of flank scans to consider when generating spectrum prediction training data
     * The default value is 0, which means that only the apex scan is considered.
     */
    public int n_flank_scans = 0;

    public HashSet<String> target_isolation_wins = new HashSet<>();

    public double nce = 27.0;

    public boolean use_user_provided_ms_instrument = false;
    public String user_provided_ms_instrument = "";
    public String ms_instrument = "Eclipse";

    public String device = "gpu";

    public int max_fragment_ion_charge = 2;

    public boolean I2L = false;

    /**
     * This is used to store all peptide matches data. Keys are ms file names, values are peptide matches.
     */
    private HashMap<String, ArrayList<String>> ms_file2psm = new HashMap<>();

    /**
     * Column name to index for PSM file.
     */
    private HashMap<String,Integer> hIndex = new HashMap<>();

    /**
     * Peptide sequence + modification -> Peptide object
     */
    private ConcurrentHashMap<String, Peptide> peptide_mod2Peptide = new ConcurrentHashMap<>();

    /**
     * The number of times that a fragment ion in a spectrum assigned to peptides.
     * scan -> fragment ion mz -> peptide count
     */
    private ConcurrentHashMap<Integer, ConcurrentHashMap<Double,Integer>> scan2mz2count = new ConcurrentHashMap<>();

    public String out_dir = "./";
    private final double fragment_ion_intensity_threshold = 0.00;
    private HashMap<String, Integer> ion_type2column_index = new HashMap<>();
    private boolean lossWaterNH3 = false;
    private final String fragmentation_method = "hcd";

    private String psm_head_line = "-";
    private String fragment_ion_intensity_head_line = "-";

    public int sg_smoothing_data_points = 3;

    private double rt_win_offset = 1.0;

    public boolean fragment_ion_intensity_normalization = false;

    public int min_n_fragment_ions = 4;

    public boolean export_valid_matches_only = false;

    public boolean fragment_ion_charge_less_than_precursor_charge = false;

    public boolean export_fragment_ion_mz_to_file = false;

    private HashMap<String, String> mod_map = new HashMap<>();

    public boolean export_skyline_transition_list_file = false;

    /**
     * Any fragment ion with mz <= 200 will not be considered as valid.
     */
    public double min_fragment_ion_mz = 200.0;
    public double max_fragment_ion_mz = 2000.0;

    public double lf_frag_mz_min = 200.0;
    public double lf_frag_mz_max = 1800.0;

    public int lf_precursor_charge_min = 2;
    public int lf_precursor_charge_max = 4;

    public int lf_top_n_fragment_ions = 20;
    public int lf_min_n_fragment_ions = 2;

    public int lf_frag_n_min = 2;

    public boolean refine_peak_boundary = false;

    public boolean remove_y1 = false;

    /**
     * The minimum number of high quality fragment ions to consider. Default is 4.
     */
    public int min_n_high_quality_fragment_ions = 4;

    /**
     * If it is true, don't do any filtering and use all peaks and all detected peptides.
     */
    private boolean use_all_peaks = false;

    /**
     * The minimum correlation cutoff to consider. Default is 0.75.
     */
    public double cor_cutoff = 0.75;

    private double rt_min = 0.0;
    double rt_max = 0.0;

    private String rt_merge_method = "min";

    // spectral library generation
    public String db = "";

    private String python_bin = "python";

    /**
     * The number of peptides to predict in each batch.
     */
    private int n_peptides_per_batch = 200000;

    /**
     * The precursor charge states to consider for spectral library generation
     */
    private int [] precursor_charges = new int[]{2,3,4};

    /**
     * The mz tol of fragment ions
     */
    private ArrayList<Double> fragment_ions_mz_tol = new ArrayList<>();

    /**
     * The search engine used to generate the identification result.
     */
    public String search_engine = "-";

    /**
     * Modification mapping: UniMod accession -> AI modification name
     */
    private HashMap<String,String> unimod2modification_code = new HashMap<>();
    private HashMap<String,String> modification_code2modification = new HashMap<>();

    /**
     * For modification peptide modeling or not:
     * "-" general peptide modeling
     * "phosphorylation" phosphorylation peptide modeling
     */
    public String mod_ai = "-";


    /**
     * The format of spectral library: DIA-NN, EncyclopeDIA, Skyline (blib) or mzSpecLib
     */
    public String export_spectral_library_format = "DIA-NN";

    /**
     * The maximum m/z of isolation window. Any MS2 scans with precursor m/z win larger than this value will be ignored.
     */
    public double isolation_win_mz_max = -1;
    private boolean export_xic = false;
    private double ptm_site_prob_cutoff = 0.75;
    private int n_ion_min = 0;
    private int c_ion_min = 0;
    private int global_random_seed = 2024;

    /**
     * Testing mode
     */
    private static boolean test_mode = false;

    public boolean use_parquet = false;

    /**
     * The exported file format of spectral library: tsv or parquet
     */
    public String export_spectral_library_file_format = "tsv";
    public boolean export_spectra_to_mgf = false;

    private final Splitter tab_splitter = Splitter.on('\t');

    private static final ModificationParameters modificationParameters = new ModificationParameters();
    private static final SequenceMatchingParameters sequenceMatchingParameters = new SequenceMatchingParameters();
    private static final SequenceProvider sequenceProvider = new SingleProteinSequenceProvider();

    public static void main(String[] args) throws ParseException, IOException {
        long startTime = System.currentTimeMillis();
        Options options = new Options();
        options.addOption("i", true, "PSM file");
        options.addOption("ms", true, "MS file in mzML format");
        options.addOption("fixMod", true, "Fixed modification, the format is like : 1,2,3. Use '-printPTM' to show all supported modifications. Default is 1 (Carbamidomethylation(C)[57.02]). " +
                "If there is no fixed modification, set it as '-fixMod no' or '-fixMod 0'.");
        options.addOption("varMod",true,"Variable modification, the format is the same with -fixMod. Default is 2 (Oxidation(M)[15.99]). "+
                "If there is no variable modification, set it as '-varMod no' or '-varMod 0'.");
        options.addOption("maxVar",true,"Max number of variable modifications, default is 1");
        options.addOption("db", true, "Protein database");
        options.addOption("o", true, "Output directory");
        // options.addOption("tol", true, "Fragment ion m/z tolerance in Da, default is 0.6");
        // options.addOption("tolu", true, "Fragment ion m/z tolerance in Da, default is 0.6");
        options.addOption("itol", true, "Fragment ion m/z tolerance in ppm, default is 20");
        options.addOption("itolu", true, "Fragment ion m/z tolerance unit, default is ppm");
        options.addOption("sg", true, "The number of data points for XIC smoothing, it's 3 in default");
        options.addOption("nm", false, "Perform fragment ion intensity normalization or not");
        options.addOption("nf", true, "The minimum number of matched fragment ions to consider, it's 4 in default");
        options.addOption("cs", false, "Fragment ion charge less than precursor charge or not");
        options.addOption("ez", false, "Export fragment ion mz to file or not");
        options.addOption("skyline", false, "Export skyline transition list file or not");
        options.addOption("valid", false, "Only export valid matches or not");
        options.addOption("na", true, "The number of adjacent scans to match: default is 0");
        options.addOption("fdr", true, "The minimum FDR cutoff to consider, default is 0.01");
        options.addOption("cor", true, "The minimum correlation cutoff to consider, default is 0.75");
        options.addOption("ptm_site_prob", true, "The minimum PTM site score to consider, default is 0.75");
        options.addOption("use_all_peaks", false, "Use all peaks for training");
        options.addOption("min_mz", true, "The minimum fragment ion m/z to consider, default is 200.0");
        options.addOption("min_n", true, "The minimum high quality fragment ion number to consider, default is 4");
        options.addOption("enzyme",true,"Enzyme used for protein digestion. 0:Non enzyme, 1:Trypsin (default), 2:Trypsin (no P rule), 3:Arg-C, 4:Arg-C (no P rule), 5:Arg-N, 6:Glu-C, 7:Lys-C");
        options.addOption("miss_c",true,"The max missed cleavages, default is 1");
        options.addOption("I2L",false,"Convert I to L");
        options.addOption("clip_n_m", false, "When digesting a protein starting with amino acid M, two copies of the leading peptides (with and without the N-terminal M) are considered or not. Default is false.");
        options.addOption("minLength", true, "The minimum length of peptide to consider, default is 7");
        options.addOption("maxLength", true, "The maximum length of peptide to consider, default is 35");
        options.addOption("min_pep_mz", true, "The minimum mz of peptide to consider, default is 400");
        options.addOption("max_pep_mz", true, "The maximum mz of peptide to consider, default is 1000");
        options.addOption("min_pep_charge", true, "The minimum precursor charge to consider, default is 2");
        options.addOption("max_pep_charge", true, "The maximum precursor charge to consider, default is 4");
        options.addOption("lf_type", true, "Spectral library format: DIA-NN (default), EncyclopeDIA, Skyline (blib) or mzSpecLib");
        options.addOption("lf_format", true, "Spectral library file format: tsv (default) or parquet");
        options.addOption("lf_frag_mz_min", true, "The minimum mz of fragment to consider for library generation, default is 200");
        options.addOption("lf_frag_mz_max", true, "The minimum mz of fragment to consider for library generation, default is 1800");
        options.addOption("lf_top_n_frag", true, "The maximum number of fragment ions to consider for library generation, default is 20");
        options.addOption("lf_min_n_frag", true, "The minimum number of fragment ions to consider for library generation, default is 2");
        options.addOption("lf_frag_n_min", true, "The minimum fragment ion number to consider for library generation, default is 2");
        options.addOption("rf", false, "Refine peak boundary or not");
        options.addOption("rf_rt_win", true, "RT window for refine peak boundary, default is 3 minutes");
        options.addOption("rt_win_offset", true, "RT window offset for XIC extraction, default is 1 minute");
        options.addOption("xic", false, "Export XIC to file or not");
        options.addOption("export_mgf", false, "Export spectra to a mgf file or not");

        options.addOption("y1", false, "Don't use y1 ion in training");
        options.addOption("n_ion_min", true, "For n-terminal fragment ions (such as b-ion) with number <= n_ion_min, they will be considered as invalid. Default is 0.");
        options.addOption("c_ion_min", true, "For c-terminal fragment ions (such as y-ion) with number <= n_ion_min, they will be considered as invalid. Default is 0.");

        // These settings will override the information extracted from input MS/MS data.
        options.addOption("nce", true, "NCE for in-silico spectral library");
        options.addOption("ms_instrument", true, "MS instrument for in-silico spectral library: default is Eclipse");
        options.addOption("device", true, "device for in-silico spectral library: default is gpu");
        options.addOption("se", true, "The search engine used to generate the identification result: DIA-NN");
        options.addOption("mode", true, "Data type: general or phosphorylation");
        options.addOption("tf", true, "Fine tune type: ms2, rt, all (default)");
        options.addOption("seed", true, "Random seed, 2024 in default");
        options.addOption("fast", false, "Save data to parquet format for speeding up reading and writing");

        options.addOption("h", false, "Help");

        CommandLineParser parser = new DefaultParser(false);
        CommandLine cmd = parser.parse(options, args);
        if (cmd.hasOption("h") || cmd.hasOption("help") || args.length == 0) {
            HelpFormatter f = new HelpFormatter();
            f.setWidth(100);
            f.setOptionComparator(null);
            System.out.println("java -Xmx4G -jar carafe.jar");
            f.printHelp("Options", options);
            return;
        }

        // Print modification list
        if (cmd.hasOption("printPTM")) {
            ModificationUtils.save_mod2file = false;
            CModification.getInstance().printPTM();
            return;
        }

        if(cmd.hasOption("fixMod")){
            CParameter.fixMods = cmd.getOptionValue("fixMod");
        }

        HashMap<Integer,Integer>mod_index2code = new HashMap<>();
        if(cmd.hasOption("varMod")){
            CParameter.varMods = cmd.getOptionValue("varMod");
            for(String mod: cmd.getOptionValue("varMod").split(",")){
                String[]d = mod.split(":");
                if(d.length>=2) {
                    mod_index2code.put(Integer.parseInt(d[0]), Integer.parseInt(d[1]));
                }
            }
        }

        if(cmd.hasOption("miss_c")){
            CParameter.maxMissedCleavages = Integer.parseInt(cmd.getOptionValue("miss_c"));
        }

        if(cmd.hasOption("enzyme")){
            CParameter.enzyme = Integer.parseInt(cmd.getOptionValue("enzyme"));
        }


        if (cmd.hasOption("maxVar")) {
            CParameter.maxVarMods = Integer.parseInt(cmd.getOptionValue("maxVar"));
        }

        if(cmd.hasOption("clip_n_m")){
            CParameter.clip_nTerm_M = true;
        }else{
            CParameter.clip_nTerm_M = false;
        }

        //if(cmd.hasOption("tol")){
        //     CParameter.tol = Double.parseDouble(cmd.getOptionValue("tol"));
        //}
        //if(cmd.hasOption("tolu")){
        //    CParameter.tolu = cmd.getOptionValue("tolu");
        //}
        if(cmd.hasOption("itol")){
            CParameter.itol = Double.parseDouble(cmd.getOptionValue("itol"));
        }

        if(cmd.hasOption("itolu")){
            CParameter.itolu = cmd.getOptionValue("itolu");
        }

        if (cmd.hasOption("minLength")) {
            CParameter.minPeptideLength = Integer.parseInt(cmd.getOptionValue("minLength"));
        }

        if (cmd.hasOption("maxLength")) {
            CParameter.maxPeptideLength = Integer.parseInt(cmd.getOptionValue("maxLength"));
        }

        if (cmd.hasOption("min_pep_mz")) {
            CParameter.minPeptideMz = Double.parseDouble(cmd.getOptionValue("min_pep_mz"));
        }
        if (cmd.hasOption("max_pep_mz")) {
            CParameter.maxPeptideMz = Double.parseDouble(cmd.getOptionValue("max_pep_mz"));
        }

        if(cmd.hasOption("rf_rt_win")){
            CParameter.rt_win = Double.parseDouble(cmd.getOptionValue("rf_rt_win"));
        }


        String psm_file = cmd.getOptionValue("i");
        CParameter.init();
        ModificationUtils.getInstance();

        AIGear aiGear = new AIGear();
        aiGear.load_mod_map();
        if (cmd.hasOption("sg")) {
            aiGear.sg_smoothing_data_points = Integer.parseInt(cmd.getOptionValue("sg"));
        }
        if (cmd.hasOption("nm")) {
            aiGear.fragment_ion_intensity_normalization = true;
        }
        if (cmd.hasOption("nf")) {
            aiGear.min_n_fragment_ions = Integer.parseInt(cmd.getOptionValue("nf"));
        }
        if (cmd.hasOption("valid")) {
            aiGear.export_valid_matches_only = true;
        }
        if (cmd.hasOption("cs")) {
            aiGear.fragment_ion_charge_less_than_precursor_charge = true;
        }

        if (cmd.hasOption("ez")) {
            aiGear.export_fragment_ion_mz_to_file = true;
        }

        if (cmd.hasOption("skyline")) {
            aiGear.export_skyline_transition_list_file = true;
        }

        if (cmd.hasOption("min_mz")) {
            aiGear.min_fragment_ion_mz = Double.parseDouble(cmd.getOptionValue("min_mz"));
        }

        if (cmd.hasOption("fdr")) {
            aiGear.fdr_cutoff = Double.parseDouble(cmd.getOptionValue("fdr"));
        }

        if (cmd.hasOption("min_n")) {
            aiGear.min_n_high_quality_fragment_ions = Integer.parseInt(cmd.getOptionValue("min_n"));
        }

        if (cmd.hasOption("cor")) {
            aiGear.cor_cutoff = Double.parseDouble(cmd.getOptionValue("cor"));
        }

        if(cmd.hasOption("use_all_peaks")){
            // Don't do any filtering. Use all peaks and all detected peptides.
            aiGear.use_all_peaks = true;
        }

        if (cmd.hasOption("db")) {
            aiGear.db = cmd.getOptionValue("db");
        }

        if(cmd.hasOption("se")){
            aiGear.search_engine = cmd.getOptionValue("se");
        }

        if(cmd.hasOption("mode")){
            aiGear.mod_ai = cmd.getOptionValue("mode");
        }

        if(cmd.hasOption("device")){
            aiGear.device = cmd.getOptionValue("device");
        }

        if(cmd.hasOption("lf_frag_mz_min")){
            aiGear.lf_frag_mz_min = Double.parseDouble(cmd.getOptionValue("lf_frag_mz_min"));
        }

        if(cmd.hasOption("lf_frag_mz_max")){
            aiGear.lf_frag_mz_max = Double.parseDouble(cmd.getOptionValue("lf_frag_mz_max"));
        }

        if(cmd.hasOption("lf_top_n_frag")){
            aiGear.lf_top_n_fragment_ions = Integer.parseInt(cmd.getOptionValue("lf_top_n_frag"));
        }

        if (cmd.hasOption("lf_min_n_frag")) {
            aiGear.lf_min_n_fragment_ions = Integer.parseInt(cmd.getOptionValue("lf_min_n_frag"));
        }

        if (cmd.hasOption("lf_frag_n_min")) {
            aiGear.lf_frag_n_min = Integer.parseInt(cmd.getOptionValue("lf_frag_n_min"));
        }

        if(cmd.hasOption("na")){
            aiGear.n_flank_scans = Integer.parseInt(cmd.getOptionValue("na"));
        }

        if(cmd.hasOption("ms_instrument")){
            aiGear.user_provided_ms_instrument = cmd.getOptionValue("ms_instrument");
            aiGear.use_user_provided_ms_instrument = true;
        }

        if(cmd.hasOption("xic")){
            aiGear.export_xic = true;
        }

        if(cmd.hasOption("rt_win_offset")){
            aiGear.rt_win_offset = Double.parseDouble(cmd.getOptionValue("rt_win_offset"));
        }

        if(cmd.hasOption("ptm_site_prob")){
            aiGear.ptm_site_prob_cutoff = Double.parseDouble(cmd.getOptionValue("ptm_site_prob"));
        }

        if(cmd.hasOption("seed")){
            aiGear.global_random_seed = Integer.parseInt(cmd.getOptionValue("seed"));
        }

        if(cmd.hasOption("fast")){
            aiGear.use_parquet = true;
        }

        if(cmd.hasOption("lf_format")){
            aiGear.export_spectral_library_file_format = cmd.getOptionValue("lf_format");
        }

        if(cmd.hasOption("min_pep_charge")){
            aiGear.lf_precursor_charge_min = Integer.parseInt(cmd.getOptionValue("min_pep_charge"));
        }

        if(cmd.hasOption("max_pep_charge")){
            aiGear.lf_precursor_charge_max = Integer.parseInt(cmd.getOptionValue("max_pep_charge"));
        }

        aiGear.precursor_charges = new int[aiGear.lf_precursor_charge_max - aiGear.lf_precursor_charge_min + 1];
        for(int charge = aiGear.lf_precursor_charge_min; charge <= aiGear.lf_precursor_charge_max; charge++){
            aiGear.precursor_charges[charge - aiGear.lf_precursor_charge_min] = charge;
        }

        if(cmd.hasOption("o")){
            aiGear.out_dir = cmd.getOptionValue("o");
            // create output directory
            File F = new File(aiGear.out_dir);
            if(!F.isDirectory()){
                F.mkdirs();
            }
            CParameter.outdir = aiGear.out_dir;
        }

        if(cmd.hasOption("lf_type")){
            aiGear.export_spectral_library_format = cmd.getOptionValue("lf_type");
        }

        if(cmd.hasOption("y1")){
            aiGear.remove_y1 = true;
        }

        if(cmd.hasOption("n_ion_min")){
            aiGear.n_ion_min = Integer.parseInt(cmd.getOptionValue("n_ion_min"));
        }

        if(cmd.hasOption("c_ion_min")){
            aiGear.c_ion_min = Integer.parseInt(cmd.getOptionValue("c_ion_min"));
        }

        if(cmd.hasOption("I2L")){
            aiGear.I2L = true;
        }

        if(cmd.hasOption("export_mgf")){
            aiGear.export_spectra_to_mgf = true;
        }

        if(cmd.hasOption("ms")) {
            String ms_file = cmd.getOptionValue("ms");

            if(cmd.hasOption("rf")){
                aiGear.refine_peak_boundary = true;
                System.out.println("Refine peak boundary");
            }

            if(cmd.hasOption("tf")){
                CParameter.tf_type  = cmd.getOptionValue("tf");
                test_mode = true;
            }
            Cloger.getInstance().set_job_start_time();
            aiGear.load_data(psm_file,ms_file,aiGear.fdr_cutoff);
            if(aiGear.search_engine.equalsIgnoreCase("DIA-NN") || aiGear.search_engine.equalsIgnoreCase("DIANN")){
                aiGear.get_ms2_matches_diann();
            }else{
                aiGear.get_ms2_matches();
            }
            Cloger.getInstance().logger.info("Time used for training data generation: "+Cloger.getInstance().get_job_run_time());

            Cloger.getInstance().set_job_start_time();
            HashMap<String,String> paraMap = new HashMap<>();
            aiGear.train_ms2_and_rt(paraMap,aiGear.out_dir,aiGear.out_dir,"test");
            Cloger.getInstance().logger.info("Time used for model training: "+Cloger.getInstance().get_job_run_time());
            if (cmd.hasOption("db")) {
                aiGear.db = cmd.getOptionValue("db");
                CParameter.db = cmd.getOptionValue("db");
                String model_dir = aiGear.out_dir;
                Cloger.getInstance().set_job_start_time();
                Map<String,HashMap<String,String>> res_files = aiGear.generate_spectral_library(model_dir);
                if(cmd.hasOption("tf") && cmd.getOptionValue("tf").equalsIgnoreCase("test")) {
                    aiGear.generate_multiple_library(res_files);
                }else{
                    aiGear.generate_spectral_library(res_files);
                }
                Cloger.getInstance().logger.info("Time used for spectral library generation: "+Cloger.getInstance().get_job_run_time());
            }
        }else {
            if (cmd.hasOption("db")) {
                aiGear.db = cmd.getOptionValue("db");
                CParameter.db = cmd.getOptionValue("db");
                String model_dir = aiGear.out_dir;
                Cloger.getInstance().set_job_start_time();
                Map<String,HashMap<String,String>> res_files = aiGear.generate_spectral_library(model_dir);
                aiGear.generate_spectral_library(res_files);
                Cloger.getInstance().logger.info("Time used for spectral library generation: "+Cloger.getInstance().get_job_run_time());
            }
        }

        long bTime = System.currentTimeMillis();
        Cloger.getInstance().logger.info("Time used for spectral library generation:" + (bTime - startTime) / 1000 + " s.");
        aiGear.print_parameters(StringUtils.join(args," "));
    }

    public void generate_multiple_library(Map<String,HashMap<String,String>> res_files){
        try {
            generate_spectral_library(res_files);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // pretrained model
        Map<String,HashMap<String,String>> p_res_files = new LinkedHashMap<>();
        for(String i : res_files.keySet()){
            System.out.println(i);
            p_res_files.put(i, new HashMap<>());
            String ms2_file = res_files.get(i).get("ms2");
            String ms2_intensity_file = res_files.get(i).get("ms2_intensity");
            String rt_file = res_files.get(i).get("rt");
            String ms2_mz_file = res_files.get(i).get("ms2_mz");

            File F = new File(ms2_file);
            // get the folder of file ms2_file
            String folder = F.getParent() + File.separator + "pretrained_models";
            p_res_files.get(i).put("ms2",get_file_path(ms2_file,folder));
            p_res_files.get(i).put("ms2_intensity",get_file_path(ms2_intensity_file,folder));
            p_res_files.get(i).put("rt",get_file_path(rt_file,folder));
            p_res_files.get(i).put("ms2_mz",get_file_path(ms2_mz_file,folder));
        }

        try {
            if(this.use_parquet) {
                if(this.export_spectral_library_format.equalsIgnoreCase("Skyline")) {
                    generate_spectral_library_parquet_skyline(p_res_files, out_dir, "SkylineAI_spectral_library_pretrained.tsv");
                }else if(this.export_spectral_library_format.equalsIgnoreCase("mzSpecLib")) {
                    generate_spectral_library_parquet_mzSpecLib(p_res_files, out_dir, "SkylineAI_spectral_library_pretrained.tsv");
                }else{
                    generate_spectral_library_parquet(p_res_files, out_dir, "SkylineAI_spectral_library_pretrained.tsv");
                }
            }else{
                generate_spectral_library(p_res_files, out_dir, "SkylineAI_spectral_library_pretrained.tsv");
            }
        } catch (IOException | SQLException e) {
            throw new RuntimeException(e);
        }

        // rt only model
        p_res_files.clear();
        p_res_files = new HashMap<>();
        for(String i : res_files.keySet()){
            System.out.println(i);
            p_res_files.put(i, new HashMap<>());
            String ms2_file = res_files.get(i).get("ms2");
            String ms2_intensity_file = res_files.get(i).get("ms2_intensity");
            String rt_file = res_files.get(i).get("rt");
            String ms2_mz_file = res_files.get(i).get("ms2_mz");

            File F = new File(ms2_file);
            // get the folder of file ms2_file
            String folder = F.getParent() + File.separator + "pretrained_models";
            p_res_files.get(i).put("ms2",get_file_path(ms2_file,folder));
            p_res_files.get(i).put("ms2_intensity",get_file_path(ms2_intensity_file,folder));
            p_res_files.get(i).put("rt",rt_file);
            p_res_files.get(i).put("ms2_mz",get_file_path(ms2_mz_file,folder));
        }

        try {
            if(this.use_parquet){
                if(this.export_spectral_library_format.equalsIgnoreCase("Skyline")) {
                    generate_spectral_library_parquet_skyline(p_res_files, out_dir, "SkylineAI_spectral_library_rt_only.tsv");
                }else if(this.export_spectral_library_format.equalsIgnoreCase("mzSpecLib")) {
                    generate_spectral_library_parquet_mzSpecLib(p_res_files, out_dir, "SkylineAI_spectral_library_rt_only.tsv");
                }else {
                    generate_spectral_library_parquet(p_res_files, out_dir, "SkylineAI_spectral_library_rt_only.tsv");
                }
            }else {
                generate_spectral_library(p_res_files, out_dir, "SkylineAI_spectral_library_rt_only.tsv");
            }
        } catch (IOException | SQLException e) {
            throw new RuntimeException(e);
        }

        // ms2 only model
        p_res_files.clear();
        p_res_files = new HashMap<>();
        for(String i : res_files.keySet()){
            System.out.println(i);
            p_res_files.put(i, new HashMap<>());
            String ms2_file = res_files.get(i).get("ms2");
            String ms2_intensity_file = res_files.get(i).get("ms2_intensity");
            String rt_file = res_files.get(i).get("rt");
            String ms2_mz_file = res_files.get(i).get("ms2_mz");

            File F = new File(ms2_file);
            // get the folder of file ms2_file
            String folder = F.getParent() + File.separator + "pretrained_models";
            p_res_files.get(i).put("ms2",ms2_file);
            p_res_files.get(i).put("ms2_intensity",ms2_intensity_file);
            p_res_files.get(i).put("rt",get_file_path(rt_file,folder));
            p_res_files.get(i).put("ms2_mz",ms2_mz_file);
        }

        try {
            if(this.use_parquet) {
                if(this.export_spectral_library_format.equalsIgnoreCase("Skyline")) {
                    generate_spectral_library_parquet_skyline(p_res_files, out_dir, "SkylineAI_spectral_library_ms2_only.tsv");
                }else if(this.export_spectral_library_format.equalsIgnoreCase("mzSpecLib")) {
                    generate_spectral_library_parquet_mzSpecLib(p_res_files, out_dir, "SkylineAI_spectral_library_ms2_only.tsv");
                }else {
                    generate_spectral_library_parquet(p_res_files, out_dir, "SkylineAI_spectral_library_ms2_only.tsv");
                }
            }else {
                generate_spectral_library(p_res_files, out_dir, "SkylineAI_spectral_library_ms2_only.tsv");
            }
        } catch (IOException | SQLException e) {
            throw new RuntimeException(e);
        }


    }

    private String get_file_path(String original_file, String new_folder){
        File F = new File(original_file);
        return new_folder + File.separator + F.getName();
    }

    public final Comparator<Peptide> comparator_peptide_mass_for_peptide_from_min2max = Comparator.comparingDouble(Peptide::getMass);

    public Map<String,HashMap<String,String>> generate_spectral_library(String model_dir) throws IOException {
        // digest proteins and generate peptide forms
        // need to consider for both small and large databases
        long startTime = System.currentTimeMillis();
        DBGear dbGear = new DBGear();
        dbGear.I2L = this.I2L;
        Set<String> searchedPeptides = new HashSet<>();
        List<Peptide> all_peptide_forms = new ArrayList<>();
        List<Integer> precursor_charge_list = new ArrayList<>();
        if(this.db.toLowerCase().endsWith(".fa") || this.db.toLowerCase().endsWith(".fasta")) {
            searchedPeptides = dbGear.protein_digest(this.db);

            all_peptide_forms = searchedPeptides.parallelStream()
                    .map(PeptideUtils::calcPeptideIsoforms)
                    .flatMap(List::stream).sorted(comparator_peptide_mass_for_peptide_from_min2max).collect(toList());
        }else if(this.db.toLowerCase().endsWith(".tsv") || this.db.toLowerCase().endsWith(".txt") || this.db.toLowerCase().endsWith(".csv")){
            Cloger.getInstance().logger.info("The input for spectral library generation is a peptide forms table: " + this.db);
            char sep = '\t';
            if (this.db.toLowerCase().endsWith(".csv")){
                sep = ',';
            }
            SkylineIO.load_skyline_precursor_table(this.db,sep,all_peptide_forms,precursor_charge_list);
        }
        Cloger.getInstance().logger.info("Generating peptide forms: " + all_peptide_forms.size());

        // generate input files for prediction
        ArrayList<String> input_files = new ArrayList<>();
        if(this.use_parquet) {
            System.out.println("Use parquet format ...");
            ParquetWriter<GenericRecord> pWriter = null;
            Schema schema = FileIO.getSchema4PredictionInput();
            int i_peptide = 0;
            int k = 0;
            boolean finished = false;
            ArrayList<GenericRecord> pep_out = new ArrayList<>();
            // valid peptide index: this is the index (0-based) in the library
            int pepID = 0;
            boolean file_is_closed = false;
            while (i_peptide <= all_peptide_forms.size()) {
                for (int i = 0; i < this.n_peptides_per_batch; i++) {
                    if (i_peptide >= all_peptide_forms.size()) {
                        finished = true;
                        break;
                    }
                    if(precursor_charge_list.isEmpty()) {
                        pep_out = get_InputRecord_for_prediction(all_peptide_forms.get(i_peptide), pepID, schema);
                    }else{
                        pep_out = get_InputRecord_for_prediction(all_peptide_forms.get(i_peptide), pepID, schema,precursor_charge_list.get(i_peptide));
                    }
                    if (i == 0) {
                        // first row in the batch
                        k++;
                        String o_file = this.out_dir + File.separator + "peptide_forms_" + k + ".parquet";
                        // org.apache.hadoop.fs.Path path = new org.apache.hadoop.fs.Path(o_file);
                        // OutputFile out_file = HadoopOutputFile.fromPath(path, new org.apache.hadoop.conf.Configuration());
                        LocalOutputFile localOutputFile = new LocalOutputFile(Paths.get(o_file));
                        pWriter = AvroParquetWriter.<GenericRecord>builder(localOutputFile)
                                .withSchema(schema)
                                //.withCompressionCodec(CompressionCodecName.SNAPPY)
                                .withCompressionCodec(CompressionCodecName.ZSTD)
                                .withPageSize(ParquetWriter.DEFAULT_PAGE_SIZE)
                                //.withConf(new org.apache.hadoop.conf.Configuration())
                                .withValidation(false)
                                // override when existing
                                .withWriteMode(ParquetFileWriter.Mode.OVERWRITE)
                                .withDictionaryEncoding(false)
                                .build();
                        input_files.add(o_file);
                        file_is_closed = false;
                        if (!pep_out.isEmpty()) {
                            for(GenericRecord record:pep_out){
                                pWriter.write(record);
                            }
                            pepID++;
                        }
                        i_peptide++;
                    } else if (i == (this.n_peptides_per_batch - 1)) {
                        // last row in this batch
                        if (!pep_out.isEmpty()) {
                            for(GenericRecord record:pep_out){
                                pWriter.write(record);
                            }
                            pepID++;
                        }
                        i_peptide++;
                        pWriter.close();
                        file_is_closed = true;
                    } else {
                        if (!pep_out.isEmpty()) {
                            for(GenericRecord record:pep_out){
                                pWriter.write(record);
                            }
                            pepID++;
                        }
                        i_peptide++;
                    }

                }
                if (finished) {
                    break;
                }
            }
            if (!file_is_closed) {
                pWriter.close();
            }
        }else{
            BufferedWriter pWriter = null;
            int i_peptide = 0;
            int k = 0;
            boolean finished = false;
            String pep_out = "";
            // valid peptide index: this is the index (0-based) in the library
            int pepID = 0;
            boolean file_is_closed = false;
            while (i_peptide <= all_peptide_forms.size()) {
                for (int i = 0; i < this.n_peptides_per_batch; i++) {
                    if (i_peptide >= all_peptide_forms.size()) {
                        finished = true;
                        break;
                    }
                    if(precursor_charge_list.isEmpty()) {
                        pep_out = get_input_for_prediction(all_peptide_forms.get(i_peptide), pepID);
                    }else{
                        pep_out = get_input_for_prediction(all_peptide_forms.get(i_peptide), pepID,precursor_charge_list.get(i_peptide));
                    }
                    if (i == 0) {
                        // first row in the batch
                        k++;
                        String o_file = this.out_dir + File.separator + "peptide_forms_" + k + ".tsv";
                        input_files.add(o_file);
                        pWriter = new BufferedWriter(new FileWriter(o_file));
                        file_is_closed = false;
                        pWriter.write("pepID\tsequence\tmz\tcharge\tmods\tmod_sites\n");
                        if (!pep_out.isEmpty()) {
                            pWriter.write(pep_out);
                            pepID++;
                        }
                        i_peptide++;
                    } else if (i == (this.n_peptides_per_batch - 1)) {
                        // last row in this batch
                        if (!pep_out.isEmpty()) {
                            pWriter.write(pep_out);
                            pepID++;
                        }
                        i_peptide++;
                        pWriter.close();
                        file_is_closed = true;
                    } else {
                        if (!pep_out.isEmpty()) {
                            pWriter.write(pep_out);
                            pepID++;
                        }
                        i_peptide++;
                    }

                }
                if (finished) {
                    break;
                }
            }
            if (!file_is_closed) {
                pWriter.close();
            }
        }

        Map<String,HashMap<String,String>> res_files = new LinkedHashMap<>();

        if(this.device.toLowerCase().contains("gpu")){
            MemoryUsage mem = CudaUtils.getGpuMemory(Device.gpu());
            long gpu_mem = mem.getMax(); // it should return 11GB
            Cloger.getInstance().logger.info("GPU memory "+gpu_mem);
            System.out.println(mem.toString());
        }

        AIWorker.fast_mode = this.use_parquet;

        if(this.device.toLowerCase().contains("cpu")){
            // only use 1 cpu for now
            ExecutorService fixedThreadPool = Executors.newFixedThreadPool(1);
            Cloger.getInstance().logger.info("Number of CPU jobs "+1);
            AIWorker.python_bin = this.python_bin;
            // perform spectrum and rt prediction.
            String mode = this.mod_ai.equalsIgnoreCase("-")?"general":this.mod_ai;
            System.out.println("NCE: "+this.nce);
            for(int i=0;i<input_files.size();i++){
                // prediction
                if(this.use_user_provided_ms_instrument) {
                    fixedThreadPool.execute(new AIWorker(model_dir, input_files.get(i), this.out_dir, i + "", this.device, this.user_provided_ms_instrument, this.nce, this.mod_ai));
                }else{
                    fixedThreadPool.execute(new AIWorker(model_dir, input_files.get(i), this.out_dir, i + "", this.device, this.ms_instrument, this.nce, this.mod_ai));
                }
                res_files.put(input_files.get(i),new HashMap<>());
                if(this.use_parquet) {
                    res_files.get(input_files.get(i)).put("ms2", this.out_dir + File.separator + i + "_ms2_df.parquet");
                    res_files.get(input_files.get(i)).put("ms2_mz", this.out_dir + File.separator + i + "_ms2_mz_df.parquet");
                    res_files.get(input_files.get(i)).put("ms2_intensity", this.out_dir + File.separator + i + "_ms2_pred.parquet");
                    res_files.get(input_files.get(i)).put("rt", this.out_dir + File.separator + i + "_rt_pred.parquet");
                }else{
                    res_files.get(input_files.get(i)).put("ms2", this.out_dir + File.separator + i + "_ms2_df.tsv");
                    res_files.get(input_files.get(i)).put("ms2_mz", this.out_dir + File.separator + i + "_ms2_mz_df.tsv");
                    res_files.get(input_files.get(i)).put("ms2_intensity", this.out_dir + File.separator + i + "_ms2_pred.tsv");
                    res_files.get(input_files.get(i)).put("rt", this.out_dir + File.separator + i + "_rt_pred.tsv");
                }
            }

            fixedThreadPool.shutdown();

            try {
                fixedThreadPool.awaitTermination(Long.MAX_VALUE, TimeUnit.HOURS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }else{
            double gpu_mem = get_gpu_mem();
            int n_gpu_jobs = (int)Math.floor(gpu_mem/2);
            n_gpu_jobs = Math.min(n_gpu_jobs,input_files.size());
            ExecutorService fixedThreadPool = Executors.newFixedThreadPool(n_gpu_jobs);
            Cloger.getInstance().logger.info("Number of GPU jobs "+n_gpu_jobs);
            AIWorker.python_bin = this.python_bin;
            // perform spectrum and rt prediction.
            String mode = this.mod_ai.equalsIgnoreCase("-")?"general":this.mod_ai;
            System.out.println("NCE: "+this.nce);
            for(int i=0;i<input_files.size();i++){
                // prediction
                if(this.use_user_provided_ms_instrument) {
                    fixedThreadPool.execute(new AIWorker(model_dir, input_files.get(i), this.out_dir, i + "", this.device, this.user_provided_ms_instrument, this.nce, this.mod_ai));
                }else{
                    fixedThreadPool.execute(new AIWorker(model_dir, input_files.get(i), this.out_dir, i + "", this.device, this.ms_instrument, this.nce, this.mod_ai));
                }
                res_files.put(input_files.get(i),new HashMap<>());
                if(this.use_parquet){
                    res_files.get(input_files.get(i)).put("ms2", this.out_dir + File.separator + i + "_ms2_df.parquet");
                    res_files.get(input_files.get(i)).put("ms2_mz", this.out_dir + File.separator + i + "_ms2_mz_df.parquet");
                    res_files.get(input_files.get(i)).put("ms2_intensity", this.out_dir + File.separator + i + "_ms2_pred.parquet");
                    res_files.get(input_files.get(i)).put("rt", this.out_dir + File.separator + i + "_rt_pred.parquet");
                }else {
                    res_files.get(input_files.get(i)).put("ms2", this.out_dir + File.separator + i + "_ms2_df.tsv");
                    res_files.get(input_files.get(i)).put("ms2_mz", this.out_dir + File.separator + i + "_ms2_mz_df.tsv");
                    res_files.get(input_files.get(i)).put("ms2_intensity", this.out_dir + File.separator + i + "_ms2_pred.tsv");
                    res_files.get(input_files.get(i)).put("rt", this.out_dir + File.separator + i + "_rt_pred.tsv");
                }
            }

            fixedThreadPool.shutdown();

            try {
                fixedThreadPool.awaitTermination(Long.MAX_VALUE, TimeUnit.HOURS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        long bTime = System.currentTimeMillis();
        searchedPeptides.clear();
        Cloger.getInstance().logger.info("Time used for spectral library generation:" + (bTime - startTime) / 1000 + " s.");
        return res_files;
    }

    public double get_gpu_mem(){
        MemoryUsage mem = CudaUtils.getGpuMemory(Device.gpu());
        return 1.0*mem.getMax()/1024/1024/1024;
    }

    public void generate_spectral_library(Map<String,HashMap<String,String>> res_files, String out_dir) throws IOException {
        String pep_index_file = out_dir+File.separator+"pep_index.tsv";
        String frag_ions_file = out_dir+File.separator+"frag_ions.tsv";

        BufferedWriter pepWriter = new BufferedWriter(new FileWriter(pep_index_file));
        BufferedWriter fragWriter = new BufferedWriter(new FileWriter(frag_ions_file));
        pepWriter.write("sequence\tmod_sites\tmods\tpepID\trt\n");
        fragWriter.write("pepID\tcharge\tmz\tintensity\n");

        String pepID;
        String sequence;
        String mods;
        String mod_sites;
        double rt;
        HashSet<String> cur_pepIDs = new HashSet<>();
        for(String i : res_files.keySet()){
            System.out.println(i);
            String ms2_file = res_files.get(i).get("ms2");
            String ms2_intensity_file = res_files.get(i).get("ms2_intensity");
            String rt_file = res_files.get(i).get("rt");
            String ms2_mz_file = res_files.get(i).get("ms2_mz");

            BufferedReader ms2Reader = new BufferedReader(new FileReader(ms2_file));
            BufferedReader ms2IntensityReader = new BufferedReader(new FileReader(ms2_intensity_file));
            BufferedReader rtReader = new BufferedReader(new FileReader(rt_file));
            BufferedReader ms2mzReader = new BufferedReader(new FileReader(ms2_mz_file));

            HashMap<String,Integer> ms2_col2index = this.get_column_name2index_from_head_line(ms2Reader.readLine().trim());
            HashMap<String,Integer> ms2_intensity_col2index = this.get_column_name2index_from_head_line(ms2IntensityReader.readLine().trim());
            HashMap<String,Integer> rt_col2index = this.get_column_name2index_from_head_line(rtReader.readLine().trim());
            String [] fragment_ion_column_names = ms2mzReader.readLine().trim().split("\t");

            String line;
            // RT information
            while((line=rtReader.readLine())!=null){
                String []d = line.split("\t");
                pepID = d[rt_col2index.get("pepID")];
                sequence = d[rt_col2index.get("sequence")];
                mods = d[rt_col2index.get("mods")];
                mod_sites = d[rt_col2index.get("mod_sites")];
                if(mods.isEmpty()){
                    mods = "-";
                    mod_sites = "-";
                }

                if(this.rt_max > 0){
                    rt = Double.parseDouble(d[rt_col2index.get("rt_pred")]);
                    rt = rt * this.rt_max;
                }else{
                    rt = Double.parseDouble(d[rt_col2index.get("irt_pred")]);
                }
                if(!cur_pepIDs.contains(pepID)) {
                    pepWriter.write(sequence + "\t" +
                            mod_sites + "\t" +
                            mods + "\t" +
                            pepID + "\t" +
                            rt + "\n");
                    cur_pepIDs.add(pepID);
                }
            }
            rtReader.close();

            // MS intensity
            ArrayList<String> ms2_intensity_lines = new ArrayList<>();
            while ((line=ms2IntensityReader.readLine())!=null){
                ms2_intensity_lines.add(line.trim());
            }
            ms2IntensityReader.close();

            // mz intensity
            ArrayList<String> ms2_mz_lines = new ArrayList<>();
            while ((line=ms2mzReader.readLine())!=null){
                ms2_mz_lines.add(line.trim());
            }
            ms2mzReader.close();

            // MS2 information
            String []ion_mz_intensity;
            while((line=ms2Reader.readLine())!=null){
                String []d = line.split("\t");
                pepID = d[ms2_col2index.get("pepID")];
                String charge = d[ms2_col2index.get("charge")];
                int frag_start_idx = Integer.parseInt(d[ms2_col2index.get("frag_start_idx")]);
                int frag_stop_idx = Integer.parseInt(d[ms2_col2index.get("frag_stop_idx")]);
                ion_mz_intensity = get_fragment_ion_intensity(ms2_mz_lines,
                        ms2_intensity_lines,
                        fragment_ion_column_names,
                        frag_start_idx,
                        frag_stop_idx,CParameter.top_n_fragment_ions);
                fragWriter.write(pepID+"\t"+charge+"\t"+ion_mz_intensity[0]+"\t"+ion_mz_intensity[1]+"\n");
            }

            ms2Reader.close();
            ms2IntensityReader.close();
        }

        pepWriter.close();
        fragWriter.close();

        // sort pep_index.tsv file by pepID
        String old_pep_index_file = pep_index_file.replaceAll("tsv$","") + "tmp";
        FileUtils.copyFile(new File(pep_index_file), new File(old_pep_index_file));
        CsvReadOptions.Builder builder = CsvReadOptions.builder(old_pep_index_file)
                .separator('\t')
                .header(true);
        CsvReadOptions options = builder.build();
        Table pepTable = Table.read().usingOptions(options);
        pepTable = pepTable.sortOn("pepID");
        // write to file in tsv format
        // Specify the file path for the output TSV file
        // Create CsvWriteOptions
        CsvWriteOptions writeOptions = CsvWriteOptions.builder(pep_index_file)
                .separator('\t')
                .header(true)
                .build();

        // Write the table to a TSV file
        pepTable.write().usingOptions(writeOptions);
    }

    private String[] get_fragment_ion_intensity(ArrayList<String> ms2_mz_lines,
                                            ArrayList<String> ms2_intensity_lines,
                                            String []column_names,
                                            int frag_start_idx,
                                            int frag_stop_idx,
                                            int top_n){
        String []res = new String[2];
        double []intensity;
        double []mz;
        HashMap<Double,Double> mz2intensity = new HashMap<>();
        for(int i=frag_start_idx;i<frag_stop_idx;i++) {
            intensity = Arrays.stream(ms2_intensity_lines.get(i).split("\t")).mapToDouble(Double::parseDouble).toArray();
            mz = Arrays.stream(ms2_mz_lines.get(i).split("\t")).mapToDouble(Double::parseDouble).toArray();
            for (int k = 0; k < column_names.length; k++) {
                //if (intensity[k] >= CParameter.min_fragment_ion_intensity && mz[k] >= this.min_fragment_ion_mz && mz[k] <= this.max_fragment_ion_mz) {
                if (intensity[k] > 0.0 && mz[k] >= this.min_fragment_ion_mz && mz[k] <= this.max_fragment_ion_mz) {
                    mz2intensity.put(mz[k], intensity[k]);
                }
            }
        }
        // sort mz2intensity by values from max to min and only keep the top n values
        mz2intensity = mz2intensity.entrySet().stream().sorted(Map.Entry.<Double,Double>comparingByValue().reversed()).limit(top_n).collect(toMap(Map.Entry::getKey,Map.Entry::getValue,(e1,e2)->e1,LinkedHashMap::new));
        res[0] = StringUtils.join(mz2intensity.keySet(),",");
        res[1] = StringUtils.join(mz2intensity.values(),",");
        return res;
    }

    ArrayList<LibFragment> get_fragment_ion_intensity4parquet_all(ArrayList<double[]> ms2_mz_lines,
                                                                          ArrayList<double[]> ms2_intensity_lines,
                                                                          String []column_names,
                                                                          int frag_start_idx,
                                                                          int frag_stop_idx,
                                                                          int top_n,
                                                                          String []ion_types,
                                                                          String []mod_losses,
                                                                          int []ion_charges,
                                                                          int frag_n_min){
        double []intensity;
        double []mz;
        int b_ion_num=1;
        int y_ion_num=frag_stop_idx-frag_start_idx;
        // ion string ID -> intensity
        HashMap<Integer, Double> ion2intensity = new HashMap<>();
        double max_intensity = 0;
        int ion_id = 0;
        for(int i=frag_start_idx;i<frag_stop_idx;i++) {
            intensity = ms2_intensity_lines.get(i);
            mz = ms2_mz_lines.get(i);
            for (int k = 0; k < column_names.length; k++) {
                ion_id++;
                //if (intensity[k] >= CParameter.min_fragment_ion_intensity && mz[k] >= this.min_fragment_ion_mz && mz[k] <= this.max_fragment_ion_mz) {
                if (intensity[k] > 0.0 && mz[k] >= this.lf_frag_mz_min && mz[k] <= this.lf_frag_mz_max) {
                    // StringBuilder stringBuilder = new StringBuilder();
                    if(ion_types[k].startsWith("b")) {
                        if(b_ion_num >= frag_n_min) {
                            ion2intensity.put(ion_id,intensity[k]);
                            if(max_intensity < intensity[k]){
                                max_intensity = intensity[k];
                            }
                        }
                    }else{
                        if(y_ion_num >= frag_n_min){
                            ion2intensity.put(ion_id,intensity[k]);
                            if(max_intensity < intensity[k]){
                                max_intensity = intensity[k];
                            }
                        }
                    }
                }
            }
            b_ion_num++;
            y_ion_num--;
        }
        // sort mz2intensity by values from max to min and only keep the top n values
        Map<Integer, Double> valid_mz_map = ion2intensity.entrySet()
                .stream()
                .sorted(Map.Entry.<Integer,Double>comparingByValue().reversed())
                .limit(top_n)
                .collect(toMap(Map.Entry::getKey,Map.Entry::getValue,(e1,e2)->e1,LinkedHashMap::new));

        // re-normalize fragment ion intensity
        b_ion_num=1;
        y_ion_num=frag_stop_idx-frag_start_idx;
        ArrayList<LibFragment> fragments = new ArrayList<>(valid_mz_map.size());
        ion_id = 0;
        for(int i=frag_start_idx;i<frag_stop_idx;i++) {
            intensity = ms2_intensity_lines.get(i);
            mz = ms2_mz_lines.get(i);
            for (int k = 0; k < column_names.length; k++) {
                ion_id++;
                //if (intensity[k] >= CParameter.min_fragment_ion_intensity && mz[k] >= this.min_fragment_ion_mz && mz[k] <= this.max_fragment_ion_mz) {
                if (intensity[k] > 0.0 && mz[k] >= this.lf_frag_mz_min && mz[k] <= this.lf_frag_mz_max) {
                    if(ion_types[k].startsWith("b")) {
                        if(b_ion_num >= frag_n_min) {
                            if(valid_mz_map.containsKey(ion_id)) {
                                LibFragment fragment = new LibFragment();
                                fragment.FragmentMz = (float)mz[k];
                                fragment.RelativeIntensity = (float)(intensity[k]/max_intensity);
                                fragment.FragmentNumber = b_ion_num;
                                fragment.FragmentType = ion_types[k];
                                fragment.FragmentCharge = ion_charges[k];
                                fragment.FragmentLossType = mod_losses[k];
                                fragments.add(fragment);
                            }
                        }
                    }else{
                        if(y_ion_num >= frag_n_min){
                            if(valid_mz_map.containsKey(ion_id)) {
                                LibFragment fragment = new LibFragment();
                                fragment.FragmentMz = (float)mz[k];
                                fragment.RelativeIntensity = (float)(intensity[k]/max_intensity);
                                fragment.FragmentNumber = y_ion_num;
                                fragment.FragmentType = ion_types[k];
                                fragment.FragmentCharge = ion_charges[k];
                                fragment.FragmentLossType = mod_losses[k];
                                fragments.add(fragment);
                            }
                        }
                    }
                }
            }
            b_ion_num++;
            y_ion_num--;
        }

        // sort ArrayList<LibFragment> fragments based on RelativeIntensity from max to min
        fragments.sort(Comparator.comparingDouble((LibFragment f) -> f.RelativeIntensity).reversed());
        return fragments;
    }

    private ArrayList<String> get_fragment_ion_intensity(ArrayList<String> ms2_mz_lines,
                                                         ArrayList<String> ms2_intensity_lines,
                                                         String []column_names,
                                                         int frag_start_idx,
                                                         int frag_stop_idx,
                                                         int top_n,
                                                         String []ion_types,
                                                         String []mod_losses,
                                                         int []ion_charges,
                                                         int frag_n_min){
        double []intensity;
        double []mz;
        int b_ion_num=1;
        int y_ion_num=frag_stop_idx-frag_start_idx;
        // ion string ID -> intensity
        HashMap<Integer, Double> ion2intensity = new HashMap<>();
        double max_intensity = 0;
        int ion_id = 0;
        for(int i=frag_start_idx;i<frag_stop_idx;i++) {
            intensity = tab_splitter.splitToStream(ms2_intensity_lines.get(i)).mapToDouble(Double::parseDouble).toArray();
            mz = tab_splitter.splitToStream(ms2_mz_lines.get(i)).mapToDouble(Double::parseDouble).toArray();
            for (int k = 0; k < column_names.length; k++) {
                ion_id++;
                //if (intensity[k] >= CParameter.min_fragment_ion_intensity && mz[k] >= this.min_fragment_ion_mz && mz[k] <= this.max_fragment_ion_mz) {
                if (intensity[k] > 0.0 && mz[k] >= this.lf_frag_mz_min && mz[k] <= this.lf_frag_mz_max) {
                    StringBuilder stringBuilder = new StringBuilder();
                    if(ion_types[k].startsWith("b")) {
                        if(b_ion_num >= frag_n_min) {
                            ion2intensity.put(ion_id,intensity[k]);
                            if(max_intensity < intensity[k]){
                                max_intensity = intensity[k];
                            }
                        }
                    }else{
                        if(y_ion_num >= frag_n_min){
                            ion2intensity.put(ion_id,intensity[k]);
                            if(max_intensity < intensity[k]){
                                max_intensity = intensity[k];
                            }
                        }
                    }
                }
            }
            b_ion_num++;
            y_ion_num--;
        }
        // sort mz2intensity by values from max to min and only keep the top n values
        Map<Integer, Double> valid_mz_map = ion2intensity.entrySet()
                .stream()
                .sorted(Map.Entry.<Integer,Double>comparingByValue().reversed())
                .limit(top_n)
                .collect(toMap(Map.Entry::getKey,Map.Entry::getValue,(e1,e2)->e1,LinkedHashMap::new));

        // re-normalize fragment ion intensity
        b_ion_num=1;
        y_ion_num=frag_stop_idx-frag_start_idx;
        HashMap<String, Double> ion_line2intensity = new HashMap<>();
        ion_id = 0;
        for(int i=frag_start_idx;i<frag_stop_idx;i++) {
            intensity = tab_splitter.splitToStream(ms2_intensity_lines.get(i)).mapToDouble(Double::parseDouble).toArray();
            mz = tab_splitter.splitToStream(ms2_mz_lines.get(i)).mapToDouble(Double::parseDouble).toArray();
            for (int k = 0; k < column_names.length; k++) {
                ion_id++;
                //if (intensity[k] >= CParameter.min_fragment_ion_intensity && mz[k] >= this.min_fragment_ion_mz && mz[k] <= this.max_fragment_ion_mz) {
                if (intensity[k] > 0.0 && mz[k] >= this.lf_frag_mz_min && mz[k] <= this.lf_frag_mz_max) {
                    StringBuilder stringBuilder = new StringBuilder();
                    if(ion_types[k].startsWith("b")) {
                        if(b_ion_num >= frag_n_min) {
                            if(valid_mz_map.containsKey(ion_id)) {
                                stringBuilder.append(mz[k]).append("\t")
                                        .append(String.format("%.4f", intensity[k]/max_intensity)).append("\t")
                                        .append(ion_types[k]).append("\t")
                                        .append(b_ion_num).append("\t")
                                        .append(ion_charges[k]).append("\t")
                                        .append(mod_losses[k]);
                                ion_line2intensity.put(stringBuilder.toString(), intensity[k]);
                            }
                        }
                    }else{
                        if(y_ion_num >= frag_n_min){
                            if(valid_mz_map.containsKey(ion_id)) {
                                stringBuilder.append(mz[k]).append("\t")
                                        .append(String.format("%.4f", intensity[k]/max_intensity)).append("\t")
                                        .append(ion_types[k]).append("\t")
                                        .append(y_ion_num).append("\t")
                                        .append(ion_charges[k]).append("\t")
                                        .append(mod_losses[k]);
                                ion_line2intensity.put(stringBuilder.toString(), intensity[k]);
                            }
                        }
                    }
                }
            }
            b_ion_num++;
            y_ion_num--;
        }

        Map<String, Double> valid_mz_map_final = ion_line2intensity.entrySet()
                .stream()
                .sorted(Map.Entry.<String,Double>comparingByValue().reversed())
                .collect(toMap(Map.Entry::getKey,Map.Entry::getValue,(e1,e2)->e1,LinkedHashMap::new));

        return new ArrayList<>(valid_mz_map_final.keySet());
    }


    private String get_input_for_prediction(Peptide peptide, int pepID){
        StringBuilder stringBuilder = new StringBuilder();
        double mz;
        String [] mods = convert_modification(peptide);
        for(int charge: this.precursor_charges){
            mz = this.get_mz(peptide.getMass(),charge);
            if(mz >= CParameter.minPeptideMz && mz <= CParameter.maxPeptideMz){
                // sequence, charge, mods, mod_sites
                stringBuilder.append(pepID).append("\t")
                        .append(peptide.getSequence()).append("\t")
                        .append(mz).append("\t")
                        .append(charge).append("\t")
                        .append(mods[0]).append("\t")
                        .append(mods[1]).append("\n");
            }
        }
        return stringBuilder.toString();
    }

    private String get_input_for_prediction(Peptide peptide, int pepID, int precursor_charge){
        StringBuilder stringBuilder = new StringBuilder();
        double mz;
        String [] mods = convert_modification(peptide);
        mz = this.get_mz(peptide.getMass(),precursor_charge);
        if(mz >= CParameter.minPeptideMz && mz <= CParameter.maxPeptideMz){
            // sequence, charge, mods, mod_sites
            stringBuilder.append(pepID).append("\t")
                    .append(peptide.getSequence()).append("\t")
                    .append(mz).append("\t")
                    .append(precursor_charge).append("\t")
                    .append(mods[0]).append("\t")
                    .append(mods[1]).append("\n");
        }
        return stringBuilder.toString();
    }


    private ArrayList<GenericRecord> get_InputRecord_for_prediction(Peptide peptide, int pepID, Schema schema){
        // StringBuilder stringBuilder = new StringBuilder();
        double mz;
        String [] mods = convert_modification(peptide);
        ArrayList<GenericRecord> records = new ArrayList<>();
        for(int charge: this.precursor_charges){
            mz = this.get_mz(peptide.getMass(),charge);
            if(mz >= CParameter.minPeptideMz && mz <= CParameter.maxPeptideMz){
                // sequence, charge, mods, mod_sites
                GenericRecord record = new GenericData.Record(schema);
                record.put("pepID", pepID);
                record.put("sequence", peptide.getSequence());
                record.put("mz", mz);
                record.put("charge", charge);
                record.put("mods", mods[0]);
                record.put("mod_sites", mods[1]);
                records.add(record);
            }
        }
        return records;
    }

    private ArrayList<GenericRecord> get_InputRecord_for_prediction(Peptide peptide, int pepID, Schema schema, int precursor_charge){
        double mz;
        String [] mods = convert_modification(peptide);
        ArrayList<GenericRecord> records = new ArrayList<>();
        mz = this.get_mz(peptide.getMass(),precursor_charge);
        if(mz >= CParameter.minPeptideMz && mz <= CParameter.maxPeptideMz){
            // sequence, charge, mods, mod_sites
            GenericRecord record = new GenericData.Record(schema);
            record.put("pepID", pepID);
            record.put("sequence", peptide.getSequence());
            record.put("mz", mz);
            record.put("charge", precursor_charge);
            record.put("mods", mods[0]);
            record.put("mod_sites", mods[1]);
            records.add(record);
        }
        return records;
    }

    public double get_mz(double mass, int charge) {
        return (mass + charge * ElementaryIon.proton.getTheoreticMass()) / charge;
    }

    private boolean run_cmd(String cmd){
        System.out.println(cmd);
        boolean pass = true;
        Runtime rt = Runtime.getRuntime();
        Process p;
        try {
            p = rt.exec(cmd);
        } catch (IOException e) {
            pass = false;
            throw new RuntimeException(e);
        }

        StreamLog errorLog = new StreamLog(p.getErrorStream(), "AI => Error:", true);
        StreamLog stdLog = new StreamLog(p.getInputStream(), "AI => Message:", true);

        errorLog.start();
        stdLog.start();

        try {
            int exitValue = p.waitFor();
            if (exitValue != 0) {
                pass = false;
                Cloger.getInstance().logger.error("AI error:" + exitValue);
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

    public void train_ms2_and_rt(HashMap<String,String> paraMap, String in_dir, String out_dir, String out_prefix){
        String psm_df = in_dir+"/psm_pdv.txt";
        String intensity_df = in_dir+"/fragment_intensity_df.tsv";
        String rt_data_file = in_dir+"/rt_data.tsv";
        String mode = this.mod_ai.equalsIgnoreCase("-")?"general":this.mod_ai;
        System.out.println("Model training ...");
        String ms_instrument_for_training;
        if(this.use_user_provided_ms_instrument){
            System.out.println("MS instrument extracted from MS/MS file:"+this.ms_instrument);
            System.out.println("Use user provided MS instrument:"+this.user_provided_ms_instrument);
            ms_instrument_for_training = this.user_provided_ms_instrument;
        }else{
            System.out.println("MS instrument:"+this.ms_instrument);
            ms_instrument_for_training = this.ms_instrument;
        }
        System.out.println("NCE:"+this.nce);
        String ai_py = get_jar_path() + File.separator + "ai.py";
        File F = new File(ai_py);
        if(!F.exists()){
            ai_py = get_py_path("/main/java/ai/ai.py","carafe_ai");
        }
        String cmd = this.python_bin + " " + ai_py +
                " --in_dir " + in_dir +
                " --out_dir " + out_dir +
                " --out_prefix "+out_prefix +
                " --device " + this.device +
                " --instrument " + ms_instrument_for_training +
                " --tf_type " + CParameter.tf_type +
                " --nce " + this.nce+
                " --seed " + this.global_random_seed +
                " --mode " + mode;
        run_cmd(cmd);
    }

    public static String get_jar_path(){
        try {
            String jar_file = AIGear.class.getProtectionDomain().getCodeSource().getLocation().toURI().getPath();
            File F = new File(jar_file);
            return F.getParent();
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }

    static String get_py_path(String py_file_path,String prefix){
        String py = "";
        // Extract the Python script from the JAR
        // e.g., "/main/java/ai/ai.py"
        InputStream input = AIWorker.class.getResourceAsStream(py_file_path);
        if (input != null) {
            try {
                File F = File.createTempFile(prefix, ".py");
                Files.copy(input, F.toPath(), StandardCopyOption.REPLACE_EXISTING);
                py = F.getAbsolutePath();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }else{
            Cloger.getInstance().logger.error(Thread.currentThread().getName()+": AI error: "+py_file_path+" not found");
        }
        return py;
    }

    public void load_data(String psm_file, String ms_file, double fdr_cutoff) {
        System.out.println("FDR cutoff:"+fdr_cutoff);
        try {
            hIndex = get_column_name2index(psm_file);
            if(this.search_engine.equalsIgnoreCase("DIANN") || this.search_engine.equalsIgnoreCase("DIA-NN")){
                System.out.println("DIANN search engine");
                this.load_UniMods();
                String new_psm_file = this.out_dir + File.separator + "psm_rank_" + fdr_cutoff + ".tsv";
                remove_interference_peptides_diann(psm_file,new_psm_file);
                ms_file2psm = get_ms_file2psm_diann(new_psm_file, ms_file, fdr_cutoff);
            }else {
                String new_psm_file = this.out_dir + File.separator + "psm_rank_" + fdr_cutoff + ".tsv";
                remove_interference_peptides(psm_file,new_psm_file,fdr_cutoff);
                ms_file2psm = get_ms_file2psm(new_psm_file, ms_file, fdr_cutoff);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }


    public void get_ms2_matches() throws IOException {
        this.ion_type2column_index.clear();
        double original_fragment_ion_intensity_cutoff = CParameter.fragment_ion_intensity_cutoff;
        CParameter.fragment_ion_intensity_cutoff = 0.0001;
        PeptideFrag.lossWaterNH3 = this.lossWaterNH3;
        PeptideFrag.max_fragment_ion_charge = this.max_fragment_ion_charge;
        PeptideFrag.fragment_ion_charge_less_than_precursor_charge = this.fragment_ion_charge_less_than_precursor_charge;

        boolean is_fragment_ion_tolu_ppm = CParameter.itolu.equalsIgnoreCase("ppm");

        this.load_mod_map();
        set_ion_type_column_index(this.fragmentation_method,this.max_fragment_ion_charge, this.lossWaterNH3);
        int n_ion_types = !(this.mod_ai.equals("-") || this.mod_ai.equalsIgnoreCase("general"))?this.max_fragment_ion_charge*2*2:this.max_fragment_ion_charge*2;
        System.out.println("The number of ion types:"+n_ion_types);
        DBGear dbGear = new DBGear();

        // for RT
        HashMap<String,PeptideRT> peptide2rt = new HashMap<>();

        // output
        int frag_start_idx = 0;
        int frag_stop_idx = 0;
        BufferedWriter psmWriter = new BufferedWriter(new FileWriter(this.out_dir+"/psm_pdv.txt"));
        psmWriter.write(this.psm_head_line+"\tmods\tmod_sites\tmax_fragment_ion_valid\tmax_cor_mz\tfrag_start_idx\tfrag_stop_idx\n");
        BufferedWriter msWriter = new BufferedWriter(new FileWriter(this.out_dir+"/ms_pdv.mgf"));
        BufferedWriter fragWriter = new BufferedWriter(new FileWriter(this.out_dir+"/fragment_intensity_df.tsv"));
        fragWriter.write(this.fragment_ion_intensity_head_line+"\n");

        BufferedWriter fragMzWriter = null;
        if(this.export_fragment_ion_mz_to_file){
            fragMzWriter = new BufferedWriter(new FileWriter(this.out_dir+"/fragment_mz.tsv"));
            fragMzWriter.write(this.fragment_ion_intensity_head_line+"\n");
        }

        BufferedWriter fragValidWriter = new BufferedWriter(new FileWriter(this.out_dir+"/fragment_intensity_valid.tsv"));
        fragValidWriter.write(this.fragment_ion_intensity_head_line+"\n");

        int n_total_matches = 0;
        int n_total_matches_valid = 0;
        int n_total_matches_max_fragment_ion_invalid = 0;

        //
        int fragment_ion_row_index = -1;

        // for exporting skyline input file
        BufferedWriter tfWriter = null;
        BufferedWriter tbWriter = null;

        if(this.export_skyline_transition_list_file){

            tfWriter = new BufferedWriter(new FileWriter(this.out_dir+"/skyline_input.tsv"));
            tfWriter.write("Peptide\tPrecursor m/z\tProduct m/z\tLibraryIntensity\tExplicit Retention Time\tExplicit Retention Time Window\tNote\n");

            // peak boundary file
            tbWriter = new BufferedWriter(new FileWriter(this.out_dir+"/skyline_boundary.tsv"));
            tbWriter.write("MinStartTime\tMaxEndTime\tFileName\tPeptideModifiedSequence\tPrecursorCharge\n");

        }

        for(String ms_file: this.ms_file2psm.keySet()){
            System.out.println("Process MS file:"+ms_file);
            // For store raw data
            DIAMeta meta = new DIAMeta();
            if(CParameter.itol>0.2 && CParameter.itolu.startsWith("da")){
                meta.fragment_ion_mz_bin_size = 0.5;
                System.out.println("Fragment ion bin size:"+meta.fragment_ion_mz_bin_size);
            }
            meta.isolation_win_mz_max = this.isolation_win_mz_max;
            meta.load_ms_data(ms_file);
            meta.get_ms_run_meta_data();

            this.min_fragment_ion_mz = meta.fragment_ion_mz_min;
            this.max_fragment_ion_mz = meta.fragment_ion_mz_max;
            System.out.println("Fragment ion m/z range:"+this.min_fragment_ion_mz+","+this.max_fragment_ion_mz);

            DIAMap diaMap_tmp = new DIAMap();
            diaMap_tmp.meta = meta;
            if(this.target_isolation_wins.isEmpty()){
                diaMap_tmp.target_isolation_wins.addAll(meta.isolationWindowMap.keySet());
            }else{
                diaMap_tmp.target_isolation_wins.addAll(this.target_isolation_wins);
            }


            if(meta.rt_max > this.rt_max){
                this.rt_max = meta.rt_max;
                System.out.println("RT max:"+this.rt_max);
            }else{
                System.out.println("RT max:"+this.rt_max);
            }

            // for output
            HashSet<String> save_spectra = new HashSet<>();

            HashMap<String,ArrayList<String>> isoWinID2PSMs = new HashMap<>();
            for(String line: this.ms_file2psm.get(ms_file)) {
                String []d = line.split("\t");
                String peptide = d[hIndex.get("peptide")];
                String modification = d[hIndex.get("modification")];
                int precursor_charge = Integer.parseInt(d[hIndex.get("charge")]);
                this.add_peptide(peptide,modification);
                String isoWinID = diaMap_tmp.get_isolation_window(dbGear.get_mz(this.get_peptide(peptide,modification).getMass(),precursor_charge));
                if(!isoWinID2PSMs.containsKey(isoWinID)){
                    isoWinID2PSMs.put(isoWinID,new ArrayList<>());
                }
                isoWinID2PSMs.get(isoWinID).add(line);

                String peptide_mod = peptide+"_"+modification;
                if(!peptide2rt.containsKey(peptide_mod)){
                    peptide2rt.put(peptide_mod,new PeptideRT());
                }
                peptide2rt.get(peptide_mod).peptide = peptide;
                peptide2rt.get(peptide_mod).modification = modification;
                peptide2rt.get(peptide_mod).rts.add(Double.parseDouble(d[hIndex.get("apex_rt")]));
                peptide2rt.get(peptide_mod).scores.add(Double.parseDouble(d[hIndex.get("q_value")]));
            }

            for(String isoWinID: isoWinID2PSMs.keySet()) {
                DIAIndex diaIndex = new DIAIndex();
                diaIndex.fragment_ion_intensity_threshold = this.fragment_ion_intensity_threshold;
                diaIndex.meta = meta;
                diaIndex.target_isolation_wins.add(isoWinID);
                diaIndex.index();
                diaIndex.sg_smoothing_data_points = this.sg_smoothing_data_points;

                HashMap<Integer, PeptideMatch> index2peptideMatch = new HashMap<>();
                int row_i = -1;
                for (String line : isoWinID2PSMs.get(isoWinID)) {
                    row_i = row_i + 1;
                    index2peptideMatch.put(row_i, new PeptideMatch());
                    String[] d = line.split("\t");
                    String peptide = d[hIndex.get("peptide")];
                    String modification = d[hIndex.get("modification")];
                    int precursor_charge = Integer.parseInt(d[hIndex.get("charge")]);
                    // double apex_rt = Double.parseDouble(d[hIndex.get("apex_rt")]);
                    double rt_start = Double.parseDouble(d[hIndex.get("rt_start")]);
                    double rt_end = Double.parseDouble(d[hIndex.get("rt_end")]);

                    int apex_scan = Integer.parseInt(d[hIndex.get("apex_scan")]);
                    Spectrum spectrum = diaIndex.get_spectrum_by_scan(apex_scan);
                    this.add_peptide(peptide, modification);
                    Peptide peptideObj = this.get_peptide(peptide, modification);
                    // System.out.println("Peptide:"+peptide+", "+modification);

                    // intensity
                    index2peptideMatch.get(row_i).ion_intensity_matrix = new double[peptide.length() - 1][n_ion_types];
                    // this may not need
                    index2peptideMatch.get(row_i).ion_mz_matrix = new double[peptide.length() - 1][n_ion_types];
                    // 0: valid, >=1 invalid
                    index2peptideMatch.get(row_i).ion_matrix = new int[peptide.length() - 1][n_ion_types];
                    index2peptideMatch.get(row_i).scan = apex_scan;
                    index2peptideMatch.get(row_i).rt_start = rt_start;
                    index2peptideMatch.get(row_i).rt_end = rt_end;
                    index2peptideMatch.get(row_i).rt_apex = Double.parseDouble(d[hIndex.get("apex_rt")]);
                    index2peptideMatch.get(row_i).peptide_length = peptide.length();
                    index2peptideMatch.get(row_i).precursor_charge = precursor_charge;

                    ArrayList<IonMatch> matched_ions = get_matched_ions(peptideObj, spectrum, precursor_charge, this.max_fragment_ion_charge, lossWaterNH3);
                    List<Double> matched_ion_mzs = new ArrayList<>();

                    // max fragment ion intensity
                    double max_fragment_ion_intensity = -1.0;
                    int max_fragment_ion_row_index = -1;
                    int max_fragment_ion_column_index = -1;

                    if (!matched_ions.isEmpty()) {
                        if (!this.scan2mz2count.containsKey(apex_scan)) {
                            this.scan2mz2count.put(apex_scan, new ConcurrentHashMap<>());
                        }
                        for (IonMatch ionMatch : matched_ions) {

                            index2peptideMatch.get(row_i).matched_ions = matched_ions;
                            if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION || ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION) {
                                // add fragment ion number
                                PeptideFragmentIon fragmentIon = ((PeptideFragmentIon) ionMatch.ion);
                                int ion_number = fragmentIon.getNumber();

                                int ion_type_column_index = this.get_ion_type_column_index(ionMatch);

                                // for y ion
                                if(ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION){
                                    fragment_ion_row_index = peptide.length() - ion_number - 1;
                                }else if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION){
                                    fragment_ion_row_index = ion_number - 1;
                                }else{
                                    System.err.println("Unrecognized fragment ion type:"+ionMatch.ion.getSubType()+","+ionMatch.ion.getSubTypeAsString());
                                    System.exit(1);
                                }

                                index2peptideMatch.get(row_i).mz2index.put(ionMatch.peakMz, new int[]{fragment_ion_row_index, ion_type_column_index});
                                index2peptideMatch.get(row_i).ion_intensity_matrix[fragment_ion_row_index][ion_type_column_index] = ionMatch.peakIntensity;
                                index2peptideMatch.get(row_i).ion_mz_matrix[fragment_ion_row_index][ion_type_column_index] = ionMatch.peakMz;
                                if (this.scan2mz2count.get(apex_scan).containsKey(ionMatch.peakMz)) {
                                    this.scan2mz2count.get(apex_scan).put(ionMatch.peakMz, this.scan2mz2count.get(apex_scan).get(ionMatch.peakMz) + 1);
                                } else {
                                    this.scan2mz2count.get(apex_scan).put(ionMatch.peakMz, 1);
                                }
                                matched_ion_mzs.add(ionMatch.peakMz);

                                if(max_fragment_ion_intensity<=ionMatch.peakIntensity){
                                    max_fragment_ion_intensity = ionMatch.peakIntensity;
                                    max_fragment_ion_row_index = fragment_ion_row_index;
                                    max_fragment_ion_column_index = ion_type_column_index;
                                }

                            }
                        }
                    }
                    if(!matched_ion_mzs.isEmpty()) {
                        index2peptideMatch.get(row_i).libSpectrum.spectrum.mz = new double[matched_ion_mzs.size()];
                        index2peptideMatch.get(row_i).libSpectrum.spectrum.intensity = new double[matched_ion_mzs.size()];
                        for (int i = 0; i < matched_ion_mzs.size(); i++) {
                            index2peptideMatch.get(row_i).libSpectrum.spectrum.mz[i] = matched_ion_mzs.get(i);
                        }
                        index2peptideMatch.get(row_i).max_fragment_ion_intensity = max_fragment_ion_intensity;
                        index2peptideMatch.get(row_i).max_fragment_ion_row_index = max_fragment_ion_row_index;
                        index2peptideMatch.get(row_i).max_fragment_ion_column_index = max_fragment_ion_column_index;
                    }

                }

                // Infer shared fragment ions based on the apex scan match
                row_i = -1;
                for (String line : isoWinID2PSMs.get(isoWinID)) {
                    row_i = row_i + 1;
                    ArrayList<IonMatch> matched_ions = index2peptideMatch.get(row_i).matched_ions;
                    int apex_scan = index2peptideMatch.get(row_i).scan;
                    if (!matched_ions.isEmpty()) {
                        for (IonMatch ionMatch : matched_ions) {
                            if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION || ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION) {
                                // add fragment ion number
                                PeptideFragmentIon fragmentIon = ((PeptideFragmentIon) ionMatch.ion);
                                int ion_number = fragmentIon.getNumber();
                                // for y ion
                                if(ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION){
                                    fragment_ion_row_index = index2peptideMatch.get(row_i).peptide_length - ion_number - 1;
                                }else if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION){
                                    fragment_ion_row_index = ion_number - 1;
                                }else{
                                    System.err.println("Unrecognized fragment ion type:"+ionMatch.ion.getSubType()+","+ionMatch.ion.getSubTypeAsString());
                                    System.exit(1);
                                }

                                int ion_type_column_index = this.get_ion_type_column_index(ionMatch);
                                index2peptideMatch.get(row_i).ion_matrix[fragment_ion_row_index][ion_type_column_index] = this.scan2mz2count.get(apex_scan).get(ionMatch.peakMz) - 1;
                            }
                        }
                    }

                }

                // Infer shared fragment ions based on the fragment ion correlation
                index2peptideMatch.values().parallelStream().forEach(peptideMatch -> xic_query(diaIndex,peptideMatch,isoWinID));
                row_i = -1;
                int [] ind = new int[]{0,0};
                for (String line : isoWinID2PSMs.get(isoWinID)) {
                    row_i = row_i + 1;
                    PeptideMatch peptideMatch = index2peptideMatch.get(row_i);
                    HashSet<Double> high_cor_mzs = new HashSet<>();
                    double max_cor_mz = 0;
                    double max_frag_cor = -100;
                    for(double mz: peptideMatch.mz2cor.keySet()){
                        if(peptideMatch.mz2cor.get(mz) >= this.cor_cutoff){
                            high_cor_mzs.add(mz);
                        }
                        if(peptideMatch.mz2cor.get(mz) > max_frag_cor){
                            max_frag_cor = peptideMatch.mz2cor.get(mz);
                            max_cor_mz = mz;
                        }
                    }
                    peptideMatch.max_cor_mz = max_cor_mz;
                    for(double mz: peptideMatch.mz2index.keySet()){
                        if(!high_cor_mzs.contains(mz)){
                            ind = peptideMatch.mz2index.get(mz);
                            peptideMatch.ion_matrix[ind[0]][ind[1]] = peptideMatch.ion_matrix[ind[0]][ind[1]]+1;
                        }
                    }
                }

                // low mass fragment ions
                // based on fragment ion m/z or ion number (such as b-1, b-2, y-1, y-2)
                row_i = -1;
                for (String line : isoWinID2PSMs.get(isoWinID)) {
                    String []d = line.split("\t");
                    row_i = row_i + 1;
                    HashMap<Integer, ArrayList<Ion>> theoretical_ions = this.generate_theoretical_fragment_ions(this.get_peptide(d[hIndex.get("peptide")],d[hIndex.get("modification")]),
                            index2peptideMatch.get(row_i).precursor_charge);
                    HashSet<Integer> possible_fragment_ion_charges = this.getPossibleFragmentIonCharges(index2peptideMatch.get(row_i).precursor_charge);
                    for(int k: theoretical_ions.keySet()){
                        for(Ion ion: theoretical_ions.get(k)){
                            if(ion.getSubType() == PeptideFragmentIon.B_ION || ion.getSubType() == PeptideFragmentIon.Y_ION){
                                PeptideFragmentIon fragmentIon = ((PeptideFragmentIon) ion);
                                int ion_number = fragmentIon.getNumber();
                                // for y ion
                                if(ion.getSubType() == PeptideFragmentIon.Y_ION){
                                    fragment_ion_row_index = index2peptideMatch.get(row_i).peptide_length - ion_number - 1;
                                }else if (ion.getSubType() == PeptideFragmentIon.B_ION){
                                    fragment_ion_row_index = ion_number - 1;
                                }else{
                                    System.err.println("Unrecognized fragment ion type:"+ion.getSubType()+","+ion.getSubTypeAsString());
                                    System.exit(1);
                                }
                                for(int frag_ion_charge: possible_fragment_ion_charges) {
                                    if (ion.getTheoreticMz(frag_ion_charge) < this.min_fragment_ion_mz || ion.getTheoreticMz(frag_ion_charge) > this.max_fragment_ion_mz) {
                                        // System.out.println("Low mass fragment ion:"+ion.getTheoreticMz(frag_ion_charge));
                                        int ion_type_column_index = this.get_ion_type_column_index(ion, frag_ion_charge);
                                        index2peptideMatch.get(row_i).ion_matrix[fragment_ion_row_index][ion_type_column_index] = index2peptideMatch.get(row_i).ion_matrix[fragment_ion_row_index][ion_type_column_index] + 1;
                                    }
                                }
                            }
                        }
                    }
                }


                // output
                row_i = -1;
                for (String line : isoWinID2PSMs.get(isoWinID)) {
                    n_total_matches++;
                    row_i = row_i + 1;
                    String []d = line.split("\t");
                    if(index2peptideMatch.get(row_i).max_fragment_ion_intensity>0 && index2peptideMatch.get(row_i).matched_ions.size()>=this.min_n_fragment_ions) {
                        boolean fragment_export = false;
                        String [] out_mod = convert_modification(d[hIndex.get("modification")]);
                        int n_valid_fragment_ions = get_n_valid_fragment_ions(index2peptideMatch.get(row_i).ion_intensity_matrix,index2peptideMatch.get(row_i).ion_matrix);
                        if(n_valid_fragment_ions >= this.min_n_high_quality_fragment_ions) {
                            if (index2peptideMatch.get(row_i).is_max_fragment_ion_intensity_valid()) {
                                n_total_matches_valid++;
                                frag_start_idx = frag_stop_idx;
                                frag_stop_idx = frag_start_idx + index2peptideMatch.get(row_i).ion_intensity_matrix.length;
                                psmWriter.write(line + "\t" + out_mod[0] + "\t" + out_mod[1] + "\t1\t" + index2peptideMatch.get(row_i).max_cor_mz + "\t" + frag_start_idx + "\t" + frag_stop_idx + "\n");
                                fragment_export = true;
                            } else {
                                if (!this.export_valid_matches_only) {
                                    frag_start_idx = frag_stop_idx;
                                    frag_stop_idx = frag_start_idx + index2peptideMatch.get(row_i).ion_intensity_matrix.length;
                                    psmWriter.write(line + "\t" + out_mod[0] + "\t" + out_mod[1] + "\t0\t" + index2peptideMatch.get(row_i).max_cor_mz + "\t" + frag_start_idx + "\t" + frag_stop_idx + "\n");
                                    fragment_export = true;
                                }
                            }

                            int apex_scan = Integer.parseInt(d[hIndex.get("apex_scan")]);
                            String spectrum_title = d[hIndex.get("spectrum_title")];
                            if (!save_spectra.contains(spectrum_title)) {
                                Spectrum spectrum = diaIndex.get_spectrum_by_scan(apex_scan);
                                int charge = Integer.parseInt(d[hIndex.get("charge")]);
                                if(this.export_spectra_to_mgf) {
                                    msWriter.write(MgfUtils.asMgf(spectrum, spectrum_title, charge, String.valueOf(apex_scan)) + "\n");
                                }
                                save_spectra.add(spectrum_title);
                            }

                            if (fragment_export) {
                                // fragment ion intensity
                                for (int i = 0; i < index2peptideMatch.get(row_i).ion_intensity_matrix.length; i++) {
                                    ArrayList<String> row = new ArrayList<>();
                                    for (int j = 0; j < index2peptideMatch.get(row_i).ion_intensity_matrix[i].length; j++) {
                                        if (this.fragment_ion_intensity_normalization) {
                                            row.add(String.valueOf(index2peptideMatch.get(row_i).ion_intensity_matrix[i][j] / index2peptideMatch.get(row_i).max_fragment_ion_intensity));
                                        } else {
                                            row.add(String.valueOf(index2peptideMatch.get(row_i).ion_intensity_matrix[i][j]));
                                        }
                                    }
                                    fragWriter.write(StringUtils.join(row, "\t") + "\n");
                                    if (this.export_fragment_ion_mz_to_file) {
                                        // could be optimized
                                        ArrayList<String> mz_row = new ArrayList<>();
                                        for (int j = 0; j < index2peptideMatch.get(row_i).ion_intensity_matrix[i].length; j++) {
                                            mz_row.add(String.valueOf(index2peptideMatch.get(row_i).ion_mz_matrix[i][j]));
                                        }
                                        fragMzWriter.write(StringUtils.join(mz_row, "\t") + "\n");
                                    }
                                }

                                // fragment ion intensity: valid or not
                                for (int i = 0; i < index2peptideMatch.get(row_i).ion_matrix.length; i++) {
                                    ArrayList<String> row = new ArrayList<>();
                                    for (int j = 0; j < index2peptideMatch.get(row_i).ion_matrix[i].length; j++) {
                                        row.add(String.valueOf(index2peptideMatch.get(row_i).ion_matrix[i][j]));
                                    }
                                    fragValidWriter.write(StringUtils.join(row, "\t") + "\n");
                                }

                                // for skyline
                                if (this.export_skyline_transition_list_file && tbWriter != null && tfWriter != null) {
                                    tbWriter.write(index2peptideMatch.get(row_i).rt_start + "\t" + index2peptideMatch.get(row_i).rt_end + "\t" + ms_file + "\t" +
                                            ModificationUtils.getInstance().getSkylineFormatPeptide(this.get_peptide(d[hIndex.get("peptide")], d[hIndex.get("modification")])) + "\t" + d[hIndex.get("charge")] + "\n");

                                    PeptideMatch peptideMatch = index2peptideMatch.get(row_i);
                                    for (double mz : peptideMatch.mz2cor.keySet()) {
                                        int[] ind_mz = peptideMatch.mz2index.get(mz);
                                        tfWriter.write(ModificationUtils.getInstance().getSkylineFormatPeptide(this.get_peptide(d[hIndex.get("peptide")], d[hIndex.get("modification")])) +
                                                "\t" +
                                                d[hIndex.get("mz")] + // may change the column name ot precursor_mz
                                                "\t" +
                                                mz +
                                                "\t" +
                                                peptideMatch.ion_intensity_matrix[ind_mz[0]][ind_mz[1]] +
                                                "\t" +
                                                index2peptideMatch.get(row_i).rt_apex +
                                                "\t" +
                                                "5" +
                                                "\t" +
                                                peptideMatch.mz2cor.get(mz) + "\n"

                                        );
                                    }
                                }

                                // for ms2 mz tol
                                PeptideMatch peptideMatch = index2peptideMatch.get(row_i);
                                for(IonMatch ionMatch: peptideMatch.matched_ions){
                                    this.fragment_ions_mz_tol.add(ionMatch.getError(is_fragment_ion_tolu_ppm));
                                }
                            }
                        }
                    }else{
                        n_total_matches_max_fragment_ion_invalid++;
                    }

                }
            }
        }

        psmWriter.close();
        msWriter.close();
        fragWriter.close();
        fragValidWriter.close();
        if(this.export_fragment_ion_mz_to_file && fragMzWriter != null){
            fragMzWriter.close();
        }
        if (this.export_skyline_transition_list_file && tfWriter != null && tbWriter != null) {
            tfWriter.close();
            tbWriter.close();
        }

        System.out.println("Total matches:"+n_total_matches);
        System.out.println("Total valid matches:"+n_total_matches_valid);
        System.out.println("Total matches with invalid max fragment ion intensity:"+n_total_matches_max_fragment_ion_invalid);

        generate_rt_train_data(peptide2rt,rt_merge_method,this.out_dir+"/rt_train_data.tsv");
        CParameter.fragment_ion_intensity_cutoff = original_fragment_ion_intensity_cutoff;

    }

    private void load_UniMods(){
        // AAAAC(UniMod:4)LDK2
        // AGEVLNQPM(UniMod:35)MMAAR2
        // AAAAAAAATMALAAPS(UniMod:21)SPTPESPTMLTK
        // AAAGPLDMSLPST(UniMod:21)PDLK
        unimod2modification_code.put("C(UniMod:4)", "0");
        unimod2modification_code.put("M(UniMod:35)", "1");
        unimod2modification_code.put("S(UniMod:21)", "2");
        unimod2modification_code.put("T(UniMod:21)", "3");
        unimod2modification_code.put("Y(UniMod:21)", "4");

        modification_code2modification.put("0", "Carbamidomethylation of C");
        modification_code2modification.put("1", "Oxidation of M");
        modification_code2modification.put("2", "Phosphorylation of S");
        modification_code2modification.put("3", "Phosphorylation of T");
        modification_code2modification.put("4", "Phosphorylation of Y");
    }


    public void get_ms2_matches_diann() throws IOException {
        this.load_UniMods();
        this.ion_type2column_index.clear();
        double original_fragment_ion_intensity_cutoff = CParameter.fragment_ion_intensity_cutoff;
        CParameter.fragment_ion_intensity_cutoff = 0.0001;
        PeptideFrag.lossWaterNH3 = this.lossWaterNH3;
        System.out.println(PeptideFrag.lossWaterNH3);
        PeptideFrag.max_fragment_ion_charge = this.max_fragment_ion_charge;
        PeptideFrag.fragment_ion_charge_less_than_precursor_charge = this.fragment_ion_charge_less_than_precursor_charge;

        boolean is_fragment_ion_tolu_ppm = CParameter.itolu.equalsIgnoreCase("ppm");

        this.load_mod_map();
        set_ion_type_column_index(this.fragmentation_method,this.max_fragment_ion_charge, this.lossWaterNH3);
        int n_ion_types = !(this.mod_ai.equals("-") || this.mod_ai.equalsIgnoreCase("general"))?this.max_fragment_ion_charge*2*2:this.max_fragment_ion_charge*2;
        System.out.println("The number of ion types:"+n_ion_types);
        DBGear dbGear = new DBGear();

        // for RT
        HashMap<String,PeptideRT> peptide2rt = new HashMap<>();

        // output
        int frag_start_idx = 0;
        int frag_stop_idx = 0;
        BufferedWriter psmWriter = new BufferedWriter(new FileWriter(this.out_dir+"/psm_pdv.txt"));
        //psmWriter.write(this.psm_head_line+"\tspectrum_title\tmz\tcharge\tpeptide\tmodification\tmods\tmod_sites\tmax_fragment_ion_valid\tmax_cor_mz\tfrag_start_idx\tfrag_stop_idx\n");
        psmWriter.write("psm_id\tspectrum_title\tms2_scan\tmz\tcharge\tpeptide\tmodification\tmods\tmod_sites\tmax_fragment_ion_valid\tmax_cor_mz\tfrag_start_idx\tfrag_stop_idx\tn_valid_fragment_ions\tn_total_matched_ions\tvalid\n");
        BufferedWriter msWriter = new BufferedWriter(new FileWriter(this.out_dir+"/ms_pdv.mgf"));
        BufferedWriter fragWriter = new BufferedWriter(new FileWriter(this.out_dir+"/fragment_intensity_df.tsv"));
        fragWriter.write(this.fragment_ion_intensity_head_line+"\n");

        BufferedWriter fragMzWriter = null;
        if(this.export_fragment_ion_mz_to_file){
            fragMzWriter = new BufferedWriter(new FileWriter(this.out_dir+"/fragment_mz.tsv"));
            fragMzWriter.write(this.fragment_ion_intensity_head_line+"\n");
        }

        BufferedWriter fragValidWriter = new BufferedWriter(new FileWriter(this.out_dir+"/fragment_intensity_valid.tsv"));
        fragValidWriter.write(this.fragment_ion_intensity_head_line+"\n");

        BufferedWriter sp_fragValidWriter = null;
        BufferedWriter pep_cor_fragValidWriter = null;
        BufferedWriter pep_shape_fragValidWriter = null;
        BufferedWriter pep_fragValidWriter = null;
        if(test_mode){
            sp_fragValidWriter = new BufferedWriter(new FileWriter(this.out_dir+"/fragment_intensity_valid_spectrum_centric.tsv"));
            sp_fragValidWriter.write(this.fragment_ion_intensity_head_line+"\n");
            pep_cor_fragValidWriter = new BufferedWriter(new FileWriter(this.out_dir+"/fragment_intensity_valid_peptide_centric_cor.tsv"));
            pep_cor_fragValidWriter.write(this.fragment_ion_intensity_head_line+"\n");
            pep_shape_fragValidWriter = new BufferedWriter(new FileWriter(this.out_dir+"/fragment_intensity_valid_peptide_centric_shape.tsv"));
            pep_shape_fragValidWriter.write(this.fragment_ion_intensity_head_line+"\n");

            pep_fragValidWriter = new BufferedWriter(new FileWriter(this.out_dir+"/fragment_intensity_valid_peptide_centric.tsv"));
            pep_fragValidWriter.write(this.fragment_ion_intensity_head_line+"\n");
        }

        int n_total_matches = 0;
        int n_total_matches_valid = 0;
        int n_total_psm_matches_valid = 0;
        int n_total_matches_max_fragment_ion_invalid = 0;
        int n_peak_overlap = 0;
        int n_ptm_site_low_confidence = 0;
        int n_less_than_min_n_high_quality_fragment_ions = 0;
        int n_less_than_min_n_fragment_ions = 0;

        //
        int fragment_ion_row_index = -1;


        // for exporting skyline input file
        BufferedWriter tfWriter = null;
        BufferedWriter tbWriter = null;

        if(this.export_skyline_transition_list_file){

            tfWriter = new BufferedWriter(new FileWriter(this.out_dir+"/skyline_input.tsv"));
            tfWriter.write("Peptide\tPrecursor m/z\tProduct m/z\tLibraryIntensity\tExplicit Retention Time\tExplicit Retention Time Window\tNote\n");

            // peak boundary file
            tbWriter = new BufferedWriter(new FileWriter(this.out_dir+"/skyline_boundary.tsv"));
            tbWriter.write("MinStartTime\tMaxEndTime\tFileName\tPeptideModifiedSequence\tPrecursorCharge\n");

        }

        BufferedWriter xicWriter = null;
        boolean first_xic = true;
        if(export_xic){
            xicWriter = new BufferedWriter(new FileWriter(this.out_dir+"/xic.json"));
            xicWriter.write("{\n");
        }

        // meta information about the MS data and model training
        BufferedWriter metaWriter = new BufferedWriter(new FileWriter(this.out_dir + "/meta.json"));
        metaWriter.write("{\n");

        int psm_id = 0;

        HashMap<String, JMeta> ms_file2meta = new HashMap<>();
        boolean first_meta = true;

        for(String ms_file: this.ms_file2psm.keySet()){
            System.out.println("Process MS file:"+ms_file);
            ms_file2meta.put(ms_file, new JMeta());
            ms_file2meta.get(ms_file).ms_file = ms_file;
            // For store raw data
            DIAMeta meta = new DIAMeta();
            if(CParameter.itol>0.2 && CParameter.itolu.startsWith("da")){
                meta.fragment_ion_mz_bin_size = 0.5;
                System.out.println("Fragment ion bin size:"+meta.fragment_ion_mz_bin_size);
            }
            meta.load_ms_data(ms_file);
            meta.get_ms_run_meta_data();
            CParameter.minPeptideMz = meta.precursor_ion_mz_min - 0.5;
            CParameter.maxPeptideMz = meta.precursor_ion_mz_max + 0.5;
            CParameter.min_fragment_ion_mz = meta.fragment_ion_mz_min - 0.5;
            if(CParameter.max_fragment_ion_mz > meta.fragment_ion_mz_max){
                CParameter.max_fragment_ion_mz = meta.fragment_ion_mz_max + 0.5;
            }
            CParameter.NCE = meta.nce;
            this.nce = meta.nce;
            System.out.println("NCE:"+CParameter.NCE);
            String ms_instrument_name = meta.get_ms_instrument(ms_file);
            if(!ms_instrument_name.isEmpty()){
                CParameter.ms_instrument = ms_instrument_name;
                this.ms_instrument = ms_instrument_name;
                System.out.println("MS instrument:"+ms_instrument_name);
            }else{
                System.out.println("No MS instrument detected from MS/MS data. Use default:"+this.ms_instrument+", "+CParameter.ms_instrument);
            }

            ms_file2meta.get(ms_file).ms_instrument = ms_instrument_name;
            ms_file2meta.get(ms_file).nce = meta.nce;
            ms_file2meta.get(ms_file).min_fragment_ion_mz = meta.fragment_ion_mz_min;
            ms_file2meta.get(ms_file).max_fragment_ion_mz = meta.fragment_ion_mz_max;
            ms_file2meta.get(ms_file).rt_max = meta.rt_max;
            if(first_meta) {
                metaWriter.write("\"" + ms_file + "\":" +JSON.toJSONString(ms_file2meta.get(ms_file)));
                first_meta = false;
            }else{
                metaWriter.write(",\n\"" + ms_file + "\":" + JSON.toJSONString(ms_file2meta.get(ms_file)));
            }

            // "DIA-NN scan numbers start with 0. And MS2 scans are numbered one after another, the numbering for MS1 ones is separate. That is the first MS2 scan has number 0, and the first MS1 scan also has number 0."
            // https://github.com/vdemichev/DiaNN/discussions/211
            HashMap<Integer,Integer> global_index2scan_num = new HashMap<>(meta.num2scanMap.size());
            int global_index = 0;
            for(int scan_num: meta.num2scanMap.keySet()){
                if(meta.num2scanMap.get(scan_num).getMsLevel()==2) {
                    global_index2scan_num.put(global_index, meta.num2scanMap.get(scan_num).getNum());
                    global_index++;
                }
            }
            System.out.println("Max index:"+global_index);

            this.min_fragment_ion_mz = meta.fragment_ion_mz_min;
            this.max_fragment_ion_mz = meta.fragment_ion_mz_max;
            System.out.println("Fragment ion m/z range:"+this.min_fragment_ion_mz+","+this.max_fragment_ion_mz);

            DIAMap diaMap_tmp = new DIAMap();
            diaMap_tmp.meta = meta;
            if(this.target_isolation_wins.isEmpty()){
                diaMap_tmp.target_isolation_wins.addAll(meta.isolationWindowMap.keySet());
            }else{
                diaMap_tmp.target_isolation_wins.addAll(this.target_isolation_wins);
            }


            if(meta.rt_max > this.rt_max){
                this.rt_max = meta.rt_max;
                System.out.println("RT max:"+this.rt_max);
            }else{
                System.out.println("RT max:"+this.rt_max);
            }

            // for output
            HashSet<String> save_spectra = new HashSet<>();

            HashMap<String,ArrayList<String>> isoWinID2PSMs = new HashMap<>();

            // for un-recognized PSMs: for example, no MS2 mapped.
            HashMap<String, Integer> un_recognized_PSMs = new HashMap<>();
            for(String line: this.ms_file2psm.get(ms_file)) {
                String []d = line.split("\t");
                // 
                String peptide = d[hIndex.get("Stripped.Sequence")];
                String modification = this.get_modification_diann(d[hIndex.get("Modified.Sequence")],peptide);
                int precursor_charge = Integer.parseInt(d[hIndex.get("Precursor.Charge")]);
                this.add_peptide(peptide,modification);
                ArrayList<String> isoWinIDs = diaMap_tmp.get_isolation_windows(dbGear.get_mz(this.get_peptide(peptide,modification).getMass(),precursor_charge));
                if (isoWinIDs.isEmpty()){
                    System.out.println("Isolation window ID is empty:"+line);
                    continue;
                }
                for(String isoWinID: isoWinIDs){
                    if (!isoWinID2PSMs.containsKey(isoWinID)) {
                        isoWinID2PSMs.put(isoWinID, new ArrayList<>());
                    }
                    isoWinID2PSMs.get(isoWinID).add(line);

                    String peptide_mod = peptide + "_" + modification;

                    if (hIndex.containsKey("PTM.Site.Confidence")) {
                        if (Double.parseDouble(d[hIndex.get("PTM.Site.Confidence")]) < this.ptm_site_prob_cutoff) {
                            continue;
                        }
                    }

                    if (!peptide2rt.containsKey(peptide_mod)) {
                        peptide2rt.put(peptide_mod, new PeptideRT());
                    }
                    peptide2rt.get(peptide_mod).peptide = peptide;
                    peptide2rt.get(peptide_mod).modification = modification;
                    peptide2rt.get(peptide_mod).rts.add(Double.parseDouble(d[hIndex.get("RT")])); // Apex RT
                    peptide2rt.get(peptide_mod).scores.add(Double.parseDouble(d[hIndex.get("Q.Value")]));
                }
            }

            for(String isoWinID: isoWinID2PSMs.keySet()) {
                DIAIndex diaIndex = new DIAIndex();
                diaIndex.fragment_ion_intensity_threshold = this.fragment_ion_intensity_threshold;
                diaIndex.meta = meta;
                diaIndex.target_isolation_wins.add(isoWinID);
                diaIndex.index();
                diaIndex.sg_smoothing_data_points = this.sg_smoothing_data_points;

                HashMap<Integer, PeptideMatch> index2peptideMatch = new HashMap<>();
                int row_i = -1;
                for (String line : isoWinID2PSMs.get(isoWinID)) {
                    row_i = row_i + 1;
                    psm_id++;
                    index2peptideMatch.put(row_i, new PeptideMatch());
                    index2peptideMatch.get(row_i).id = String.valueOf(psm_id);
                    String[] d = line.split("\t");
                    String peptide = d[hIndex.get("Stripped.Sequence")];
                    String modification = this.get_modification_diann(d[hIndex.get("Modified.Sequence")],peptide);
                    int precursor_charge = Integer.parseInt(d[hIndex.get("Precursor.Charge")]);
                    // double apex_rt = Double.parseDouble(d[hIndex.get("apex_rt")]);
                    double rt_start = Double.parseDouble(d[hIndex.get("RT.Start")]);
                    double rt_end = Double.parseDouble(d[hIndex.get("RT.Stop")]);

                    int apex_scan = global_index2scan_num.get(Integer.parseInt(d[hIndex.get("MS2.Scan")])); // index
                    Spectrum spectrum = diaIndex.get_spectrum_by_scan(apex_scan);
                    this.add_peptide(peptide, modification);
                    Peptide peptideObj = this.get_peptide(peptide, modification);

                    // intensity
                    index2peptideMatch.get(row_i).ion_intensity_matrix = new double[peptide.length() - 1][n_ion_types];
                    // this may not need
                    index2peptideMatch.get(row_i).ion_mz_matrix = new double[peptide.length() - 1][n_ion_types];
                    // 0: valid, >=1 invalid
                    index2peptideMatch.get(row_i).ion_matrix = new int[peptide.length() - 1][n_ion_types];
                    index2peptideMatch.get(row_i).scan = apex_scan;
                    index2peptideMatch.get(row_i).rt_start = rt_start;
                    index2peptideMatch.get(row_i).rt_end = rt_end;
                    index2peptideMatch.get(row_i).rt_apex = Double.parseDouble(d[hIndex.get("RT")]);
                    index2peptideMatch.get(row_i).peptide_length = peptide.length();
                    index2peptideMatch.get(row_i).precursor_charge = precursor_charge;
                    index2peptideMatch.get(row_i).index = Integer.parseInt(d[hIndex.get("MS2.Scan")]);
                    index2peptideMatch.get(row_i).peptide = peptideObj;

                    // for testing
                    if(test_mode){
                        index2peptideMatch.get(row_i).ion_matrix_map.put("spectrum_centric", new int[peptide.length() - 1][n_ion_types]);
                        index2peptideMatch.get(row_i).ion_matrix_map.put("peptide_centric_cor", new int[peptide.length() - 1][n_ion_types]);
                        index2peptideMatch.get(row_i).ion_matrix_map.put("peptide_centric_shape", new int[peptide.length() - 1][n_ion_types]);
                        index2peptideMatch.get(row_i).ion_matrix_map.put("low_mass", new int[peptide.length() - 1][n_ion_types]);
                        index2peptideMatch.get(row_i).ion_matrix_map.put("peptide_centric", new int[peptide.length() - 1][n_ion_types]);
                    }

                    if(spectrum==null){
                        if(!un_recognized_PSMs.containsKey(line)){
                            un_recognized_PSMs.put(line,0);
                        }
                        continue;
                    }else{
                        un_recognized_PSMs.put(line,1);
                    }
                    ArrayList<IonMatch> matched_ions = get_matched_ions(peptideObj, spectrum, precursor_charge, this.max_fragment_ion_charge, lossWaterNH3);
                    List<Double> matched_ion_mzs = new ArrayList<>();
                    // b or y
                    String ion_type = "";
                    List<String> matched_ion_types = new ArrayList<>();
                    // 1, 2, 3, ...
                    List<Integer> matched_ion_numbers = new ArrayList<>();

                    // max fragment ion intensity
                    double max_fragment_ion_intensity = -1.0;
                    int max_fragment_ion_row_index = -1;
                    int max_fragment_ion_column_index = -1;

                    if (!matched_ions.isEmpty()) {
                        if (!this.scan2mz2count.containsKey(apex_scan)) {
                            this.scan2mz2count.put(apex_scan, new ConcurrentHashMap<>());
                        }
                        for (IonMatch ionMatch : matched_ions) {
                            index2peptideMatch.get(row_i).matched_ions = matched_ions;
                            if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION || ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION) {
                                // add fragment ion number
                                PeptideFragmentIon fragmentIon = ((PeptideFragmentIon) ionMatch.ion);
                                int ion_number = fragmentIon.getNumber();
                                int ion_type_column_index = this.get_ion_type_column_index(ionMatch);
                                // for y ion
                                if(ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION){
                                    fragment_ion_row_index = peptide.length() - ion_number - 1;
                                    ion_type = "y";
                                }else if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION){
                                    fragment_ion_row_index = ion_number - 1;
                                    ion_type = "b";
                                }else{
                                    System.err.println("Unrecognized fragment ion type:"+ionMatch.ion.getSubType()+","+ionMatch.ion.getSubTypeAsString());
                                    System.exit(1);
                                }

                                index2peptideMatch.get(row_i).mz2index.put(ionMatch.peakMz, new int[]{fragment_ion_row_index, ion_type_column_index});
                                index2peptideMatch.get(row_i).ion_intensity_matrix[fragment_ion_row_index][ion_type_column_index] = ionMatch.peakIntensity;
                                index2peptideMatch.get(row_i).ion_mz_matrix[fragment_ion_row_index][ion_type_column_index] = ionMatch.peakMz;
                                if (this.scan2mz2count.get(apex_scan).containsKey(ionMatch.peakMz)) {
                                    this.scan2mz2count.get(apex_scan).put(ionMatch.peakMz, this.scan2mz2count.get(apex_scan).get(ionMatch.peakMz) + 1);
                                } else {
                                    this.scan2mz2count.get(apex_scan).put(ionMatch.peakMz, 1);
                                }
                                matched_ion_mzs.add(ionMatch.peakMz);
                                matched_ion_types.add(ion_type);
                                matched_ion_numbers.add(ion_number);

                                // If the fragment ion number is <= the minimum number of fragment ion used for spectral library generation,
                                // we don't consider it in getting the max intensity of fragment ions.
                                if(use_all_peaks || (max_fragment_ion_intensity<=ionMatch.peakIntensity && ion_number >= this.lf_frag_n_min)){
                                    max_fragment_ion_intensity = ionMatch.peakIntensity;
                                    max_fragment_ion_row_index = fragment_ion_row_index;
                                    max_fragment_ion_column_index = ion_type_column_index;
                                }
                            }
                        }
                    }
                    if(!matched_ion_mzs.isEmpty()) {
                        index2peptideMatch.get(row_i).libSpectrum.spectrum.mz = new double[matched_ion_mzs.size()];
                        index2peptideMatch.get(row_i).libSpectrum.ion_types = new String[matched_ion_mzs.size()];
                        index2peptideMatch.get(row_i).libSpectrum.ion_numbers = new int[matched_ion_mzs.size()];
                        index2peptideMatch.get(row_i).libSpectrum.spectrum.intensity = new double[matched_ion_mzs.size()];
                        for (int i = 0; i < matched_ion_mzs.size(); i++) {
                            index2peptideMatch.get(row_i).libSpectrum.spectrum.mz[i] = matched_ion_mzs.get(i);
                            index2peptideMatch.get(row_i).libSpectrum.ion_types[i] = matched_ion_types.get(i);
                            index2peptideMatch.get(row_i).libSpectrum.ion_numbers[i] = matched_ion_numbers.get(i);
                        }
                        index2peptideMatch.get(row_i).max_fragment_ion_intensity = max_fragment_ion_intensity;
                        index2peptideMatch.get(row_i).max_fragment_ion_row_index = max_fragment_ion_row_index;
                        index2peptideMatch.get(row_i).max_fragment_ion_column_index = max_fragment_ion_column_index;
                    }

                }

                // Infer shared fragment ions based on the apex scan match
                row_i = -1;
                for (String line : isoWinID2PSMs.get(isoWinID)) {
                    row_i = row_i + 1;
                    ArrayList<IonMatch> matched_ions = index2peptideMatch.get(row_i).matched_ions;
                    int apex_scan = index2peptideMatch.get(row_i).scan;
                    if (!matched_ions.isEmpty()) {
                        for (IonMatch ionMatch : matched_ions) {
                            if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION || ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION) {
                                // add fragment ion number
                                PeptideFragmentIon fragmentIon = ((PeptideFragmentIon) ionMatch.ion);
                                int ion_number = fragmentIon.getNumber();
                                // for y ion
                                if(ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION){
                                    fragment_ion_row_index = index2peptideMatch.get(row_i).peptide_length - ion_number - 1;
                                }else if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION){
                                    fragment_ion_row_index = ion_number - 1;
                                }else{
                                    System.err.println("Unrecognized fragment ion type:"+ionMatch.ion.getSubType()+","+ionMatch.ion.getSubTypeAsString());
                                    System.exit(1);
                                }
                                int ion_type_column_index = this.get_ion_type_column_index(ionMatch);
                                index2peptideMatch.get(row_i).ion_matrix[fragment_ion_row_index][ion_type_column_index] = this.scan2mz2count.get(apex_scan).get(ionMatch.peakMz) - 1;
                                if(test_mode){
                                    index2peptideMatch.get(row_i).ion_matrix_map.get("spectrum_centric")[fragment_ion_row_index][ion_type_column_index] = this.scan2mz2count.get(apex_scan).get(ionMatch.peakMz) - 1;
                                }
                            }
                        }
                    }

                }

                // Infer shared fragment ions based on the fragment ion correlation
                index2peptideMatch.values().parallelStream().forEach(peptideMatch -> xic_query(diaIndex,peptideMatch,isoWinID));
                row_i = -1;
                int [] ind = new int[]{0,0};
                for (String line : isoWinID2PSMs.get(isoWinID)) {
                    row_i = row_i + 1;
                    PeptideMatch peptideMatch = index2peptideMatch.get(row_i);
                    HashSet<Double> high_cor_mzs = new HashSet<>();
                    double max_cor_mz = 0;
                    double max_frag_cor = -100;
                    for(double mz: peptideMatch.mz2cor.keySet()){
                        if(peptideMatch.mz2cor.get(mz) >= this.cor_cutoff){
                            high_cor_mzs.add(mz);
                        }
                        if(peptideMatch.mz2cor.get(mz) > max_frag_cor){
                            max_frag_cor = peptideMatch.mz2cor.get(mz);
                            max_cor_mz = mz;
                        }
                    }
                    peptideMatch.max_cor_mz = max_cor_mz;
                    for(double mz: peptideMatch.mz2index.keySet()){
                        if(!high_cor_mzs.contains(mz)){
                            ind = peptideMatch.mz2index.get(mz);
                            peptideMatch.ion_matrix[ind[0]][ind[1]] = peptideMatch.ion_matrix[ind[0]][ind[1]]+1;
                            if(test_mode){
                                peptideMatch.ion_matrix_map.get("peptide_centric_cor")[ind[0]][ind[1]] = 1;
                                peptideMatch.ion_matrix_map.get("peptide_centric")[ind[0]][ind[1]] = 1;
                            }
                        }
                    }
                }

                // Infer shared fragment ions based on the fragment ion shape
                row_i = -1;
                for (String line : isoWinID2PSMs.get(isoWinID)) {
                    row_i = row_i + 1;
                    PeptideMatch peptideMatch = index2peptideMatch.get(row_i);
                    for(double mz: peptideMatch.mz2index.keySet()){
                        if(peptideMatch.mz2skewed_peaks.containsKey(mz) && peptideMatch.mz2skewed_peaks.get(mz) >= 2){
                            ind = peptideMatch.mz2index.get(mz);
                            peptideMatch.ion_matrix[ind[0]][ind[1]] = peptideMatch.ion_matrix[ind[0]][ind[1]]+1;
                            if(test_mode){
                                peptideMatch.ion_matrix_map.get("peptide_centric_shape")[ind[0]][ind[1]] = 1;
                                peptideMatch.ion_matrix_map.get("peptide_centric")[ind[0]][ind[1]] = 1;
                            }
                        }
                    }
                }

                // low mass fragment ions
                // based on fragment ion m/z or ion number (such as b-1, b-2, y-1, y-2)
                row_i = -1;
                for (String line : isoWinID2PSMs.get(isoWinID)) {
                    String []d = line.split("\t");
                    row_i = row_i + 1;
                    // only need to return +1 fragment ion here
                    HashMap<Integer, ArrayList<Ion>> theoretical_ions = this.generate_theoretical_fragment_ions(this.get_peptide(d[hIndex.get("Stripped.Sequence")],
                                    this.get_modification_diann(d[hIndex.get("Modified.Sequence")],d[hIndex.get("Stripped.Sequence")])),
                                    //index2peptideMatch.get(row_i).precursor_charge);
                                    1);
                    HashSet<Integer> possible_fragment_ion_charges = this.getPossibleFragmentIonCharges(index2peptideMatch.get(row_i).precursor_charge);
                    for(int k: theoretical_ions.keySet()){
                        for(Ion ion: theoretical_ions.get(k)){
                            if(ion.getSubType() == PeptideFragmentIon.B_ION || ion.getSubType() == PeptideFragmentIon.Y_ION){
                                boolean is_y1 = false;
                                PeptideFragmentIon fragmentIon = ((PeptideFragmentIon) ion);
                                int ion_number = fragmentIon.getNumber();
                                // for y ion
                                if(ion.getSubType() == PeptideFragmentIon.Y_ION){
                                    fragment_ion_row_index = index2peptideMatch.get(row_i).peptide_length - ion_number - 1;
                                    if(ion_number == 1){
                                        is_y1 = true;
                                    }
                                }else if (ion.getSubType() == PeptideFragmentIon.B_ION){
                                    fragment_ion_row_index = ion_number - 1;
                                }else{
                                    System.err.println("Unrecognized fragment ion type:"+ion.getSubType()+","+ion.getSubTypeAsString());
                                    System.exit(1);
                                }

                                for(int frag_ion_charge: possible_fragment_ion_charges) {
                                    if(this.remove_y1 && is_y1) {
                                        if(ion.getTheoreticMz(frag_ion_charge) < this.min_fragment_ion_mz || ion.getTheoreticMz(frag_ion_charge) > this.max_fragment_ion_mz) {
                                            // System.out.println("Low mass fragment ion:"+ion.getTheoreticMz(frag_ion_charge));
                                            int ion_type_column_index = this.get_ion_type_column_index(ion, frag_ion_charge);
                                            // index2peptideMatch.get(row_i).ion_matrix[fragment_ion_row_index][ion_type_column_index] = index2peptideMatch.get(row_i).ion_matrix[fragment_ion_row_index][ion_type_column_index] + 1;
                                            index2peptideMatch.get(row_i).ion_matrix[fragment_ion_row_index][ion_type_column_index] = 0;
                                            index2peptideMatch.get(row_i).ion_intensity_matrix[fragment_ion_row_index][ion_type_column_index] = 0.0;
                                        }else {
                                            int ion_type_column_index = this.get_ion_type_column_index(ion, frag_ion_charge);
                                            double y1_intensity = index2peptideMatch.get(row_i).ion_intensity_matrix[fragment_ion_row_index][ion_type_column_index] / index2peptideMatch.get(row_i).max_fragment_ion_intensity;
                                            if (y1_intensity >= 0.5) {
                                                index2peptideMatch.get(row_i).ion_matrix[fragment_ion_row_index][ion_type_column_index] = index2peptideMatch.get(row_i).ion_matrix[fragment_ion_row_index][ion_type_column_index] + 1;
                                            }
                                        }
                                    }else {
                                        if(ion.getTheoreticMz(frag_ion_charge) < this.min_fragment_ion_mz || ion.getTheoreticMz(frag_ion_charge) > this.max_fragment_ion_mz) {
                                            int ion_type_column_index = this.get_ion_type_column_index(ion, frag_ion_charge);
                                            index2peptideMatch.get(row_i).ion_matrix[fragment_ion_row_index][ion_type_column_index] = 0;
                                            index2peptideMatch.get(row_i).ion_intensity_matrix[fragment_ion_row_index][ion_type_column_index] = 0.0;
                                        }
                                    }

                                    if(this.n_ion_min>=1 && ion.getSubType() == PeptideFragmentIon.B_ION && ion_number<=this.n_ion_min){
                                        int ion_type_column_index = this.get_ion_type_column_index(ion, frag_ion_charge);
                                        double frag_ion_mz = index2peptideMatch.get(row_i).ion_mz_matrix[fragment_ion_row_index][ion_type_column_index];
                                        double frag_ion_cor = 0.0;
                                        double mz_skewness = 1;
                                        if(index2peptideMatch.get(row_i).mz2cor.containsKey(frag_ion_mz)){
                                            frag_ion_cor = index2peptideMatch.get(row_i).mz2cor.get(frag_ion_mz);
                                            mz_skewness = index2peptideMatch.get(row_i).mz2skewed_peaks.get(frag_ion_mz);
                                        }
                                        if(!(frag_ion_cor > 0.9 && mz_skewness <= 1)){
                                            if(index2peptideMatch.get(row_i).ion_intensity_matrix[fragment_ion_row_index][ion_type_column_index]/index2peptideMatch.get(row_i).max_fragment_ion_intensity >=0.5) {
                                                index2peptideMatch.get(row_i).ion_matrix[fragment_ion_row_index][ion_type_column_index] = index2peptideMatch.get(row_i).ion_matrix[fragment_ion_row_index][ion_type_column_index] + 1;
                                            }
                                        }
                                    }else if(this.c_ion_min>=1 && ion.getSubType() == PeptideFragmentIon.Y_ION && ion_number<=this.c_ion_min){
                                        int ion_type_column_index = this.get_ion_type_column_index(ion, frag_ion_charge);
                                        double frag_ion_mz = index2peptideMatch.get(row_i).ion_mz_matrix[fragment_ion_row_index][ion_type_column_index];
                                        double frag_ion_cor = 0.0;
                                        double mz_skewness = 1;
                                        if(index2peptideMatch.get(row_i).mz2cor.containsKey(frag_ion_mz)){
                                            frag_ion_cor = index2peptideMatch.get(row_i).mz2cor.get(frag_ion_mz);
                                            mz_skewness = index2peptideMatch.get(row_i).mz2skewed_peaks.get(frag_ion_mz);
                                        }
                                        if(!(frag_ion_cor > 0.90 && mz_skewness <= 1)){
                                            if(index2peptideMatch.get(row_i).ion_intensity_matrix[fragment_ion_row_index][ion_type_column_index]/index2peptideMatch.get(row_i).max_fragment_ion_intensity >=0.5) {
                                                index2peptideMatch.get(row_i).ion_matrix[fragment_ion_row_index][ion_type_column_index] = index2peptideMatch.get(row_i).ion_matrix[fragment_ion_row_index][ion_type_column_index] + 1;
                                            }
                                        }
                                    }

                                    // Since we don't use this fragment ions in spectral library generation, we don't use them during model training.
                                    if(ion_number < this.lf_frag_n_min){
                                        int ion_type_column_index = this.get_ion_type_column_index(ion, frag_ion_charge);
                                        index2peptideMatch.get(row_i).ion_matrix[fragment_ion_row_index][ion_type_column_index] = index2peptideMatch.get(row_i).ion_matrix[fragment_ion_row_index][ion_type_column_index] + 1;
                                    }
                                }
                            }
                        }
                    }
                }
                // output
                row_i = -1;
                for (String line : isoWinID2PSMs.get(isoWinID)) {
                    n_total_matches++;
                    row_i = row_i + 1;
                    String []d = line.split("\t");
                    if(hIndex.containsKey("peak_overlap")){
                        if(Integer.parseInt(d[hIndex.get("peak_overlap")]) >=1){
                            n_peak_overlap++;
                            continue;
                        }
                    }

                    if(hIndex.containsKey("PTM.Site.Confidence")){
                        if(Double.parseDouble(d[hIndex.get("PTM.Site.Confidence")]) < this.ptm_site_prob_cutoff){
                            n_ptm_site_low_confidence++;
                            continue;
                        }
                    }
                    if(index2peptideMatch.get(row_i).max_fragment_ion_intensity>0 && index2peptideMatch.get(row_i).matched_ions.size()>=this.min_n_fragment_ions) {
                        boolean fragment_export = false;

                        String [] out_mod = convert_modification(this.get_modification_diann(d[hIndex.get("Modified.Sequence")],d[hIndex.get("Stripped.Sequence")]));
                        int n_valid_fragment_ions = get_n_valid_fragment_ions(index2peptideMatch.get(row_i).ion_intensity_matrix,index2peptideMatch.get(row_i).ion_matrix);
                        int n_total_fragment_ions = get_n_matched_fragment_ions(index2peptideMatch.get(row_i).ion_intensity_matrix);
                        if(n_valid_fragment_ions >= this.min_n_high_quality_fragment_ions) {

                            // get adjacent scans
                            ArrayList<PeptideMatch> pMatches = get_adjacent_ms2_matches(index2peptideMatch.get(row_i),this.n_flank_scans,diaIndex,isoWinID);
                            if(this.n_flank_scans>=1 && pMatches.isEmpty()){
                                // TODO: don't remove this line
                                // System.out.println("Ignore row:"+row_i+" => "+line);
                                // continue;
                            }

                            String spectrum_title = d[hIndex.get("MS2.Scan")];
                            double pdv_precursor_mz = dbGear.get_mz(this.get_peptide(d[hIndex.get("Stripped.Sequence")], this.get_modification_diann(d[hIndex.get("Modified.Sequence")],d[hIndex.get("Stripped.Sequence")])).getMass(),
                                    Integer.parseInt(d[hIndex.get("Precursor.Charge")]));
                            String pdv_precursor_charge = d[hIndex.get("Precursor.Charge")];
                            String pdv_peptide = d[hIndex.get("Stripped.Sequence")];
                            String pdv_modification = this.get_modification_diann(d[hIndex.get("Modified.Sequence")],d[hIndex.get("Stripped.Sequence")]);
                            // true || true
                            if (index2peptideMatch.get(row_i).is_max_fragment_ion_intensity_valid() || use_all_peaks) {
                                n_total_matches_valid++;
                                n_total_psm_matches_valid++;
                                frag_start_idx = frag_stop_idx;
                                frag_stop_idx = frag_start_idx + index2peptideMatch.get(row_i).ion_intensity_matrix.length;
                                psmWriter.write(index2peptideMatch.get(row_i).id+"\t"+spectrum_title+"\t"+index2peptideMatch.get(row_i).scan+ "\t" +pdv_precursor_mz +"\t" +pdv_precursor_charge +"\t" +pdv_peptide + "\t" +pdv_modification+  "\t" + out_mod[0] + "\t" + out_mod[1] + "\t1\t" + index2peptideMatch.get(row_i).max_cor_mz + "\t" + frag_start_idx + "\t" + frag_stop_idx +
                                        "\t" + n_valid_fragment_ions + "\t" + n_total_fragment_ions + "\t1\n");
                                if(!pMatches.isEmpty()){
                                    for(PeptideMatch pMatch: pMatches){
                                        n_total_psm_matches_valid++;
                                        frag_start_idx = frag_stop_idx;
                                        frag_stop_idx = frag_start_idx + pMatch.ion_intensity_matrix.length;
                                        n_total_fragment_ions = get_n_matched_fragment_ions(pMatch.ion_intensity_matrix);
                                        // TODO: update spectrum_title
                                        psmWriter.write(index2peptideMatch.get(row_i).id+"-"+pMatch.scan+"\t"+spectrum_title +"\t"+pMatch.scan+ "\t" + pdv_precursor_mz + "\t" + pdv_precursor_charge + "\t" + pdv_peptide + "\t" + pdv_modification + "\t" + out_mod[0] + "\t" + out_mod[1] + "\t1\t" + index2peptideMatch.get(row_i).max_cor_mz + "\t" + frag_start_idx + "\t" + frag_stop_idx +
                                                "\t" + n_valid_fragment_ions + "\t" + n_total_fragment_ions + "\t1\n");
                                    }
                                }
                                fragment_export = true;
                            } else {
                                n_total_matches_max_fragment_ion_invalid++;
                                if (!this.export_valid_matches_only) {
                                    frag_start_idx = frag_stop_idx;
                                    frag_stop_idx = frag_start_idx + index2peptideMatch.get(row_i).ion_intensity_matrix.length;
                                    psmWriter.write(index2peptideMatch.get(row_i).id+"\t"+spectrum_title+ "\t"+index2peptideMatch.get(row_i).scan+ "\t" +pdv_precursor_mz +"\t" +pdv_precursor_charge +"\t" +pdv_peptide + "\t" +pdv_modification+ "\t" + out_mod[0] + "\t" + out_mod[1] + "\t0\t" + index2peptideMatch.get(row_i).max_cor_mz + "\t" + frag_start_idx + "\t" + frag_stop_idx +
                                            "\t" + n_valid_fragment_ions + "\t" + n_total_fragment_ions + "\t0\n");
                                    if(!pMatches.isEmpty()){
                                        for(PeptideMatch pMatch: pMatches){
                                            frag_start_idx = frag_stop_idx;
                                            frag_stop_idx = frag_start_idx + pMatch.ion_intensity_matrix.length;
                                            n_total_fragment_ions = get_n_matched_fragment_ions(pMatch.ion_intensity_matrix);
                                            psmWriter.write(index2peptideMatch.get(row_i).id+"-"+pMatch.scan+"\t"+spectrum_title + "\t"+pMatch.scan+"\t" + pdv_precursor_mz + "\t" + pdv_precursor_charge + "\t" + pdv_peptide + "\t" + pdv_modification + "\t" + out_mod[0] + "\t" + out_mod[1] + "\t1\t" + index2peptideMatch.get(row_i).max_cor_mz + "\t" + frag_start_idx + "\t" + frag_stop_idx +
                                                    "\t" + n_valid_fragment_ions + "\t" + n_total_fragment_ions + "\t0\n");
                                        }
                                    }
                                    fragment_export = true;
                                }
                            }

                            int apex_scan = global_index2scan_num.get(Integer.parseInt(d[hIndex.get("MS2.Scan")]));
                            // String spectrum_title = d[hIndex.get("MS2.Scan")];
                            if (!save_spectra.contains(spectrum_title)) {
                                Spectrum spectrum = diaIndex.get_spectrum_by_scan(apex_scan);
                                int charge = Integer.parseInt(d[hIndex.get("Precursor.Charge")]);
                                if(this.export_spectra_to_mgf) {
                                    msWriter.write(MgfUtils.asMgf(spectrum, spectrum_title, charge, String.valueOf(apex_scan)) + "\n");
                                }
                                save_spectra.add(spectrum_title);
                                // TODO: add spectra for adjacent scans if they are used
                            }

                            if (fragment_export) {
                                // fragment ion intensity
                                for (int i = 0; i < index2peptideMatch.get(row_i).ion_intensity_matrix.length; i++) {
                                    ArrayList<String> row = new ArrayList<>();
                                    for (int j = 0; j < index2peptideMatch.get(row_i).ion_intensity_matrix[i].length; j++) {
                                        if (this.fragment_ion_intensity_normalization) {
                                            row.add(String.valueOf(index2peptideMatch.get(row_i).ion_intensity_matrix[i][j] / index2peptideMatch.get(row_i).max_fragment_ion_intensity));
                                        } else {
                                            row.add(String.valueOf(index2peptideMatch.get(row_i).ion_intensity_matrix[i][j]));
                                        }
                                    }
                                    fragWriter.write(StringUtils.join(row, "\t") + "\n");
                                    if (this.export_fragment_ion_mz_to_file) {
                                        // could be optimized
                                        ArrayList<String> mz_row = new ArrayList<>();
                                        for (int j = 0; j < index2peptideMatch.get(row_i).ion_intensity_matrix[i].length; j++) {
                                            mz_row.add(String.valueOf(index2peptideMatch.get(row_i).ion_mz_matrix[i][j]));
                                        }
                                        fragMzWriter.write(StringUtils.join(mz_row, "\t") + "\n");
                                    }
                                }

                                // fragment ion intensity for adjacent scans if they are used
                                if(!pMatches.isEmpty()){
                                    for(PeptideMatch pMatch: pMatches) {
                                        for (int i = 0; i < pMatch.ion_intensity_matrix.length; i++) {
                                            ArrayList<String> row = new ArrayList<>();
                                            for (int j = 0; j < pMatch.ion_intensity_matrix[i].length; j++) {
                                                if (this.fragment_ion_intensity_normalization) {
                                                    row.add(String.valueOf(pMatch.ion_intensity_matrix[i][j] / pMatch.max_fragment_ion_intensity));
                                                } else {
                                                    row.add(String.valueOf(pMatch.ion_intensity_matrix[i][j]));
                                                }
                                            }
                                            fragWriter.write(StringUtils.join(row, "\t") + "\n");
                                            if (this.export_fragment_ion_mz_to_file) {
                                                // could be optimized
                                                ArrayList<String> mz_row = new ArrayList<>();
                                                for (int j = 0; j < pMatch.ion_intensity_matrix[i].length; j++) {
                                                    mz_row.add(String.valueOf(pMatch.ion_mz_matrix[i][j]));
                                                }
                                                fragMzWriter.write(StringUtils.join(mz_row, "\t") + "\n");
                                            }
                                        }
                                    }
                                }

                                // fragment ion intensity: valid or not
                                for (int i = 0; i < index2peptideMatch.get(row_i).ion_matrix.length; i++) {
                                    ArrayList<String> row = new ArrayList<>();
                                    for (int j = 0; j < index2peptideMatch.get(row_i).ion_matrix[i].length; j++) {
                                        row.add(String.valueOf(index2peptideMatch.get(row_i).ion_matrix[i][j]));
                                    }
                                    fragValidWriter.write(StringUtils.join(row, "\t") + "\n");
                                    if(test_mode){
                                        row = new ArrayList<>();
                                        for (int j = 0; j < index2peptideMatch.get(row_i).ion_matrix_map.get("spectrum_centric")[i].length; j++) {
                                            row.add(String.valueOf(index2peptideMatch.get(row_i).ion_matrix_map.get("spectrum_centric")[i][j]));
                                        }
                                        sp_fragValidWriter.write(StringUtils.join(row, "\t") + "\n");

                                        row = new ArrayList<>();
                                        for (int j = 0; j < index2peptideMatch.get(row_i).ion_matrix_map.get("peptide_centric_cor")[i].length; j++) {
                                            row.add(String.valueOf(index2peptideMatch.get(row_i).ion_matrix_map.get("peptide_centric_cor")[i][j]));
                                        }
                                        pep_cor_fragValidWriter.write(StringUtils.join(row, "\t") + "\n");

                                        row = new ArrayList<>();
                                        for (int j = 0; j < index2peptideMatch.get(row_i).ion_matrix_map.get("peptide_centric_shape")[i].length; j++) {
                                            row.add(String.valueOf(index2peptideMatch.get(row_i).ion_matrix_map.get("peptide_centric_shape")[i][j]));
                                        }
                                        pep_shape_fragValidWriter.write(StringUtils.join(row, "\t") + "\n");

                                        row = new ArrayList<>();
                                        for (int j = 0; j < index2peptideMatch.get(row_i).ion_matrix_map.get("peptide_centric")[i].length; j++) {
                                            row.add(String.valueOf(index2peptideMatch.get(row_i).ion_matrix_map.get("peptide_centric")[i][j]));
                                        }
                                        pep_fragValidWriter.write(StringUtils.join(row, "\t") + "\n");
                                    }
                                }

                                // fragment ion intensity for adjacent scans if they are used
                                if(!pMatches.isEmpty()){
                                    for(PeptideMatch pMatch: pMatches) {
                                        // use the information from the apex scan for this.
                                        for (int i = 0; i < index2peptideMatch.get(row_i).ion_matrix.length; i++) {
                                            ArrayList<String> row = new ArrayList<>();
                                            for (int j = 0; j < index2peptideMatch.get(row_i).ion_matrix[i].length; j++) {
                                                row.add(String.valueOf(index2peptideMatch.get(row_i).ion_matrix[i][j]));
                                            }
                                            fragValidWriter.write(StringUtils.join(row, "\t") + "\n");
                                        }
                                    }
                                }

                                // for skyline
                                if (this.export_skyline_transition_list_file && tbWriter != null && tfWriter != null) {

                                    double precursor_mz = dbGear.get_mz(this.get_peptide(d[hIndex.get("Stripped.Sequence")], this.get_modification_diann(d[hIndex.get("Modified.Sequence")],d[hIndex.get("Stripped.Sequence")])).getMass(),
                                            Integer.parseInt(d[hIndex.get("Precursor.Charge")]));

                                    tbWriter.write(index2peptideMatch.get(row_i).rt_start + "\t" + index2peptideMatch.get(row_i).rt_end + "\t" + ms_file + "\t" +
                                            ModificationUtils.getInstance().getSkylineFormatPeptide(this.get_peptide(d[hIndex.get("Stripped.Sequence")], this.get_modification_diann(d[hIndex.get("Modified.Sequence")],d[hIndex.get("Stripped.Sequence")]))) + "\t" + d[hIndex.get("Precursor.Charge")] + "\n");

                                    PeptideMatch peptideMatch = index2peptideMatch.get(row_i);
                                    for (double mz : peptideMatch.mz2cor.keySet()) {
                                        int[] ind_mz = peptideMatch.mz2index.get(mz);
                                        tfWriter.write(ModificationUtils.getInstance().getSkylineFormatPeptide(this.get_peptide(d[hIndex.get("Stripped.Sequence")], this.get_modification_diann(d[hIndex.get("Modified.Sequence")],d[hIndex.get("Stripped.Sequence")]))) +
                                                "\t" +
                                                precursor_mz + // may change the column name ot precursor_mz
                                                "\t" +
                                                mz +
                                                "\t" +
                                                peptideMatch.ion_intensity_matrix[ind_mz[0]][ind_mz[1]] +
                                                "\t" +
                                                index2peptideMatch.get(row_i).rt_apex +
                                                "\t" +
                                                "5" +
                                                "\t" +
                                                peptideMatch.mz2cor.get(mz) + "\n"

                                        );
                                    }
                                }

                                if(this.export_xic){
                                    if(first_xic) {
                                        xicWriter.write("\"" + index2peptideMatch.get(row_i).id + "\":" + get_xic_json(index2peptideMatch.get(row_i).id, index2peptideMatch.get(row_i)));
                                        first_xic = false;
                                    }else{
                                        xicWriter.write(",\n\"" + index2peptideMatch.get(row_i).id + "\":" + get_xic_json(index2peptideMatch.get(row_i).id, index2peptideMatch.get(row_i)));
                                    }
                                }

                                // for ms2 mz tol
                                PeptideMatch peptideMatch = index2peptideMatch.get(row_i);
                                for(IonMatch ionMatch: peptideMatch.matched_ions){
                                    this.fragment_ions_mz_tol.add(ionMatch.getError(is_fragment_ion_tolu_ppm));
                                }
                            }
                        }else{
                            n_less_than_min_n_high_quality_fragment_ions++;
                        }
                    }else{
                        n_less_than_min_n_fragment_ions++;
                    }

                }
            }
            if(!un_recognized_PSMs.isEmpty()){
                int n_un_recognized_PSMs = 0;
                for(String line: un_recognized_PSMs.keySet()){
                    if(un_recognized_PSMs.get(line)==0){
                        n_un_recognized_PSMs++;
                        System.out.println("Spectrum not found:"+ line);
                    }
                }
                if(n_un_recognized_PSMs >= 1){
                    System.out.println("Spectrum not found:"+ n_un_recognized_PSMs);
                }
            }
        }

        psmWriter.close();
        msWriter.close();
        fragWriter.close();
        fragValidWriter.close();
        if(this.export_fragment_ion_mz_to_file && fragMzWriter != null){
            fragMzWriter.close();
        }
        if (this.export_skyline_transition_list_file && tfWriter != null && tbWriter != null) {
            tfWriter.close();
            tbWriter.close();
        }

        if(export_xic){
            xicWriter.write("\n}");
            xicWriter.close();
        }

        if(test_mode){
            sp_fragValidWriter.close();
            pep_cor_fragValidWriter.close();
            pep_shape_fragValidWriter.close();
            pep_fragValidWriter.close();
        }

        metaWriter.write("\n}");
        metaWriter.close();

        System.out.println("Total matches:"+n_total_matches);
        System.out.println("Total valid matches:"+n_total_matches_valid);
        System.out.println("Total valid PSM matches:"+n_total_psm_matches_valid);
        System.out.println("Total matches with invalid max fragment ion intensity:"+n_total_matches_max_fragment_ion_invalid);
        System.out.println("Total matches with peak overlap:"+n_peak_overlap);
        System.out.println("Total matches with less than min_n_high_quality_fragment_ions="+min_n_high_quality_fragment_ions+":"+n_less_than_min_n_high_quality_fragment_ions);
        System.out.println("Total matches with less than min_n_fragment_ions="+min_n_fragment_ions+":"+n_less_than_min_n_fragment_ions);
        if(n_ptm_site_low_confidence >0){
            System.out.println("Total matches with PTM site low confidence:"+n_ptm_site_low_confidence);
        }
        generate_rt_train_data(peptide2rt,rt_merge_method,this.out_dir+"/rt_train_data.tsv");
        CParameter.fragment_ion_intensity_cutoff = original_fragment_ion_intensity_cutoff;

    }


    private String get_xic_json(String id, PeptideMatch pMatch){
        JXIC xic = new JXIC();
        xic.fragment_ion_mzs = pMatch.peak.fragment_ions_mz
                .stream()
                .mapToDouble(Double::doubleValue)
                .toArray();
        xic.smoothed_fragment_intensities = pMatch.smoothed_fragment_intensities.getData();
        xic.raw_fragment_intensities = pMatch.raw_fragment_intensities;
        xic.xic_rt_values = pMatch.xic_rt_values;
        xic.fragment_ion_skewness = pMatch.skewed_peaks;

        xic.fragment_ion_cors = new double[xic.fragment_ion_mzs.length];
        for(int i=0;i<xic.fragment_ion_mzs.length;i++){
            if(pMatch.mz2cor.containsKey(xic.fragment_ion_mzs[i])) {
                xic.fragment_ion_cors[i] = pMatch.mz2cor.get(xic.fragment_ion_mzs[i]);
            }else{
                System.err.println("Error: missing fragment ion correlation:"+xic.fragment_ion_mzs[i]);
                System.exit(1);
            }
        }

        xic.peptide = pMatch.peptide.getSequence();
        xic.charge = pMatch.precursor_charge;
        xic.modification = ModificationUtils.getInstance().getModificationString(pMatch.peptide);
        xic.rt_apex = pMatch.rt_apex;
        xic.rt_start = pMatch.rt_start;
        xic.rt_end = pMatch.rt_end;
        xic.id = id;
        return(JSON.toJSONString(xic));
    }


    ArrayList<PeptideMatch> get_adjacent_ms2_matches(PeptideMatch peptideMatch, int n_flank_scans, DIAIndex diaIndex, String iso_win){
        int scan_num = peptideMatch.scan;
        int scan_index = diaIndex.get_index_by_scan(iso_win,scan_num);
        // 10 -> 10-2=8, 10+2 = 12 -> 8, 9, 11, 12
        int start_scan_index = scan_index - n_flank_scans;
        int end_scan_index = scan_index + n_flank_scans;
        ArrayList<PeptideMatch> pMatches = new ArrayList<>(2*n_flank_scans);
        for(int index=start_scan_index; index<=end_scan_index; index++){
            if(index==scan_index){
                continue;
            }
            if(diaIndex.isolation_win2index2scan.get(iso_win).containsKey(index)){
                int scan = diaIndex.get_scan_by_index(iso_win, index);
                Spectrum spectrum = diaIndex.get_spectrum_by_scan(scan);
                if(spectrum==null){
                    System.out.println("Spectrum is null:"+index+"\t"+scan);
                    System.out.println(iso_win);
                    continue;
                }
                ArrayList<IonMatch> matched_ions = get_matched_ions(peptideMatch.peptide, spectrum, peptideMatch.precursor_charge, this.max_fragment_ion_charge, lossWaterNH3);
                List<Double> matched_ion_mzs = new ArrayList<>();
                // max fragment ion intensity
                double max_fragment_ion_intensity = -1.0;
                int max_fragment_ion_row_index = -1;
                int max_fragment_ion_column_index = -1;
                int fragment_ion_row_index = -1;
                PeptideMatch pMatch = new PeptideMatch();
                pMatch.scan = scan;

                // intensity
                pMatch.ion_intensity_matrix = new double[peptideMatch.ion_intensity_matrix.length][peptideMatch.ion_intensity_matrix[0].length];
                // this may not need
                pMatch.ion_mz_matrix = new double[peptideMatch.ion_intensity_matrix.length][peptideMatch.ion_intensity_matrix[0].length];
                // 0: valid, >=1 invalid
                pMatch.ion_matrix = new int[peptideMatch.ion_intensity_matrix.length][peptideMatch.ion_intensity_matrix[0].length];



                if (!matched_ions.isEmpty()) {
                    if (!this.scan2mz2count.containsKey(scan)) {
                        this.scan2mz2count.put(scan, new ConcurrentHashMap<>());
                    }
                    for (IonMatch ionMatch : matched_ions) {

                        pMatch.matched_ions = matched_ions;
                        if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION || ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION) {
                            // add fragment ion number
                            PeptideFragmentIon fragmentIon = ((PeptideFragmentIon) ionMatch.ion);
                            int ion_number = fragmentIon.getNumber();

                            int ion_type_column_index = this.get_ion_type_column_index(ionMatch);
                            // for y ion
                            if(ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION){
                                fragment_ion_row_index = peptideMatch.peptide.getSequence().length() - ion_number - 1;
                            }else if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION){
                                fragment_ion_row_index = ion_number - 1;
                            }else{
                                System.err.println("Unrecognized fragment ion type:"+ionMatch.ion.getSubType()+","+ionMatch.ion.getSubTypeAsString());
                                System.exit(1);
                            }

                            pMatch.mz2index.put(ionMatch.peakMz, new int[]{fragment_ion_row_index, ion_type_column_index});
                            pMatch.ion_intensity_matrix[fragment_ion_row_index][ion_type_column_index] = ionMatch.peakIntensity;
                            pMatch.ion_mz_matrix[fragment_ion_row_index][ion_type_column_index] = ionMatch.peakMz;
                            if (this.scan2mz2count.get(scan).containsKey(ionMatch.peakMz)) {
                                this.scan2mz2count.get(scan).put(ionMatch.peakMz, this.scan2mz2count.get(scan).get(ionMatch.peakMz) + 1);
                            } else {
                                this.scan2mz2count.get(scan).put(ionMatch.peakMz, 1);
                            }
                            matched_ion_mzs.add(ionMatch.peakMz);

                            if(max_fragment_ion_intensity<=ionMatch.peakIntensity){
                                max_fragment_ion_intensity = ionMatch.peakIntensity;
                                max_fragment_ion_row_index = fragment_ion_row_index;
                                max_fragment_ion_column_index = ion_type_column_index;
                            }


                        }
                    }
                }
                if(!matched_ion_mzs.isEmpty()) {
                    pMatch.libSpectrum.spectrum.mz = new double[matched_ion_mzs.size()];
                    pMatch.libSpectrum.spectrum.intensity = new double[matched_ion_mzs.size()];
                    for (int i = 0; i < matched_ion_mzs.size(); i++) {
                        pMatch.libSpectrum.spectrum.mz[i] = matched_ion_mzs.get(i);
                    }
                    pMatch.max_fragment_ion_intensity = max_fragment_ion_intensity;
                    pMatch.max_fragment_ion_row_index = max_fragment_ion_row_index;
                    pMatch.max_fragment_ion_column_index = max_fragment_ion_column_index;
                }

                double cor = calc_spectrum_correlation(peptideMatch, pMatch);
                if(cor>=0.9){
                    pMatches.add(pMatch);
                }

            }

        }
        return pMatches;
    }
    
    private double calc_spectrum_correlation(PeptideMatch x, PeptideMatch y){
        int n_valid_peaks = 0;
        int n_total_peaks = 0;
        // x.ion_intensity_matrix is a 2D matrix
        // n_col is its column number
        // n_row is its row number
        int n_row = x.ion_intensity_matrix.length;
        int n_col = x.ion_intensity_matrix[0].length;
        ArrayList<Double> x_int = new ArrayList<>();
        ArrayList<Double> y_int = new ArrayList<>();
        for(int i=0;i<n_row;i++){
            for(int j=0;j<n_col;j++){
                if(x.ion_intensity_matrix[i][j]>0){
                    n_total_peaks++;
                }
                if(x.ion_intensity_matrix[i][j]>0 && x.ion_matrix[i][j]<=0){
                    x_int.add(x.ion_intensity_matrix[i][j]);
                    y_int.add(y.ion_intensity_matrix[i][j]);
                    n_valid_peaks++;
                }
            }
        }
        double cor = -100.0;
        if(x_int.size()>=3) {
            double[] x_double = new double[x_int.size()];
            double[] y_double = new double[y_int.size()];
            for (int i = 0; i < x_int.size(); i++) {
                x_double[i] = x_int.get(i);
                y_double[i] = y_int.get(i);
            }
            // cor = new PearsonsCorrelation().correlation(x_double, y_double);
            cor = new SpearmansCorrelation().correlation(x_double, y_double);
            if (Double.isNaN(cor) || cor < 0) {
                cor = 0;
            }
        }
        return cor;
    }

    private double log_transform(double x){
        return FastMath.log10(x+1)/3;
    }

    private String get_modification_diann(String mod_seq, String peptide){
        // AAAAC(UniMod:4)LDK2
        // AGEVLNQPM(UniMod:35)MMAAR2
        // AAAAAAAATMALAAPS(UniMod:21)SPTPESPTMLTK
        // AAAGPLDMSLPST(UniMod:21)PDLK
        if(mod_seq.equalsIgnoreCase(peptide)){
            return "-";
        }else{
            for(String unimod: this.unimod2modification_code.keySet()){
                if(mod_seq.contains(unimod)){
                    mod_seq = mod_seq.replace(unimod,this.unimod2modification_code.get(unimod));
                }
            }
            if(mod_seq.length()!=peptide.length()){
                System.err.println("Unrecognized modification:"+mod_seq+","+peptide);
                System.exit(1);
            }
            String [] aas = mod_seq.split("");
            ArrayList<String> mods = new ArrayList<>();
            int pos;
            String mod_name;
            for (int i = 0; i < aas.length; i++) {
                if(this.modification_code2modification.containsKey(aas[i])){
                    // Oxidation of M@12[15.9949]
                    pos = i+1;
                    mod_name = this.modification_code2modification.get(aas[i]);
                    mods.add(mod_name+"@"+pos+"["+ CModification.getInstance().getPTMbyName(mod_name).getMass()+"]");
                }
            }
            return StringUtils.join(mods,";");
        }

    }

    private void generate_rt_train_data(HashMap<String,PeptideRT> peptide2rt, String method, String out_file) throws IOException {

        peptide2rt.values().parallelStream().forEach(peptideRT -> {
            if(peptideRT.rts.size()==1){
                peptideRT.rt = peptideRT.rts.get(0);
                peptideRT.rt_max = peptideRT.rts.get(0);
                peptideRT.rt_min = peptideRT.rts.get(0);
            }else{
                peptideRT.rt_max = Collections.max(peptideRT.rts);
                peptideRT.rt_min = Collections.min(peptideRT.rts);
                if(method.equalsIgnoreCase("max")){
                    int maxIndex = peptideRT.scores.indexOf(Collections.max(peptideRT.scores));
                    peptideRT.rt = peptideRT.rts.get(maxIndex);
                }else if (method.equalsIgnoreCase("min")){
                    int minIndex = peptideRT.scores.indexOf(Collections.min(peptideRT.scores));
                    peptideRT.rt = peptideRT.rts.get(minIndex);
                }else if (method.equalsIgnoreCase("mean")){
                    double sum = 0;
                    for (double num : peptideRT.rts) {
                        sum += num;
                    }
                    peptideRT.rt = sum / peptideRT.rts.size();
                }else{
                    System.err.println("Unrecognized method:"+method);
                    System.exit(1);
                }
            }
            //peptideRT.rt_norm = (peptideRT.rt - this.rt_min)/(this.rt_max - this.rt_min);
            peptideRT.rt_norm = peptideRT.rt/this.rt_max;
            peptideRT.mods = convert_modification(peptideRT.modification);
        });

        // output
        BufferedWriter bWriter = new BufferedWriter(new FileWriter(out_file));
        bWriter.write("peptide\tsequence\tnAA\tmodification\tmods\tmod_sites\tx\ty\trt\trt_norm\n");
        for(String pep_mod: peptide2rt.keySet()){
            bWriter.write(peptide2rt.get(pep_mod).peptide+"\t"+
                    peptide2rt.get(pep_mod).peptide+"\t"+
                    peptide2rt.get(pep_mod).peptide.length()+"\t"+
                    peptide2rt.get(pep_mod).modification+"\t"+
                    peptide2rt.get(pep_mod).mods[0]+"\t"+
                    peptide2rt.get(pep_mod).mods[1]+"\t"+
                    convert_modification(peptide2rt.get(pep_mod).peptide,peptide2rt.get(pep_mod).modification)+"\t"+
                    peptide2rt.get(pep_mod).rt+"\t"+
                    peptide2rt.get(pep_mod).rt+"\t"+
                    peptide2rt.get(pep_mod).rt_norm+"\n");
        }
        bWriter.close();
        System.out.println("RT train data:"+out_file);
    }


    private int get_n_valid_fragment_ions(double [][] ion_intensity_matrix, int [][] ion_valid_matrix){
        int n = 0;
        for (int i = 0; i < ion_intensity_matrix.length; i++) {
            for (int j = 0; j < ion_intensity_matrix[i].length; j++) {
                if(ion_intensity_matrix[i][j]>0 && ion_valid_matrix[i][j]==0){
                    n++;
                }
            }
        }
        return n;
    }

    private int get_n_matched_fragment_ions(double [][] ion_intensity_matrix){
        int n = 0;
        for (int i = 0; i < ion_intensity_matrix.length; i++) {
            for (int j = 0; j < ion_intensity_matrix[i].length; j++) {
                if(ion_intensity_matrix[i][j]>0){
                    n++;
                }
            }
        }
        return n;
    }

    private String[] convert_modification(String modification){
        if(modification.equalsIgnoreCase("-")){
            return new String[]{"",""};
        }else {
            // "MLSECYR"
            // Carbamidomethyl@C;Oxidation@M   5;1
            // Oxidation of M@17[15.9949];Carbamidomethylation of C@6[57.0215]
            String[] m = modification.split(";");
            ArrayList<String> mod_name_list = new ArrayList<>(m.length);
            ArrayList<String> mod_pos_list = new ArrayList<>(m.length);
            for (String ptm : m) {
                String mod_name = ptm.split("@")[0];
                String pos = ptm.split("@")[1].split("\\[")[0];
                if(this.mod_map.containsKey(mod_name)){
                    mod_name_list.add(this.mod_map.get(mod_name));
                    mod_pos_list.add(pos);
                }else{
                    System.err.println("Unrecognized modification:"+mod_name);
                    System.exit(1);
                }

            }
            return new String[]{StringUtils.join(mod_name_list,";"),StringUtils.join(mod_pos_list,";")};
        }
    }

    private String[] convert_modification(Peptide peptide){
        String modification = ModificationUtils.getInstance().getModificationString(peptide);
        return convert_modification(modification);
    }

    private String convert_modification(String peptide, String modification){
        String x = peptide;
        boolean unrecognized_mod_found = false;
        if(!modification.equals("-")){
            String []m = modification.split(";");
            String []aa= peptide.split("");
            for(String ptm:m){
                String mod_name = ptm.split("@")[0];
                String pos = ptm.split("@")[1].split("\\[")[0];
                if(mod_name.equalsIgnoreCase("Oxidation of M")) {
                    aa[Integer.parseInt(pos) - 1] = String.valueOf(1);
                }else if(mod_name.equalsIgnoreCase("Phosphorylation of S")){
                    aa[Integer.parseInt(pos)-1] = String.valueOf(2);
                }else if(mod_name.equalsIgnoreCase("Phosphorylation of T")){
                    aa[Integer.parseInt(pos)-1] = String.valueOf(3);
                }else if(mod_name.equalsIgnoreCase("Phosphorylation of Y")){
                    aa[Integer.parseInt(pos)-1] = String.valueOf(4);
                }else if(mod_name.equalsIgnoreCase("Carbamidomethylation of C")){
                    // no need to change
                    // fixed modification.
                }else{
                    System.out.println("Unrecognized modification found:"+peptide+" -> "+modification);
                    unrecognized_mod_found = true;
                }
            }
            if (unrecognized_mod_found){
                x = "-"; //
            }else {
                x = StringUtils.join(aa, "");
            }
        }
        return x;
    }

    public void load_mod_map(){
        this.mod_map.put("Carbamidomethylation of C","Carbamidomethyl@C");
        this.mod_map.put("Oxidation of M","Oxidation@M");
        this.mod_map.put("Phosphorylation of S","Phospho@S");
        this.mod_map.put("Phosphorylation of T","Phospho@T");
        this.mod_map.put("Phosphorylation of Y","Phospho@Y");
    }


    private void xic_query(DIAIndex ms2index, PeptideMatch peptideMatch, String isoWinID) {
        LibSpectrum libSpectrum = peptideMatch.libSpectrum;
        boolean is_ppm = CParameter.itolu.equalsIgnoreCase("ppm");
        ArrayList<LPeak> peaks = new ArrayList<>(libSpectrum.spectrum.mz.length);
        IntStream.range(0,libSpectrum.spectrum.mz.length).forEach(i -> {
            LPeak p = new LPeak(libSpectrum.spectrum.mz[i],0.0);
            peaks.add(p);});
        peaks.sort(new LPeakComparatorMax2Min());

        double rt_start;
        double rt_end;
        if(this.refine_peak_boundary){
            rt_start = Math.max(0,peptideMatch.rt_apex - CParameter.rt_win);
            rt_end = peptideMatch.rt_apex + CParameter.rt_win;
        }else{
            rt_start = peptideMatch.rt_start - this.rt_win_offset;
            rt_end = peptideMatch.rt_end + this.rt_win_offset;
        }

        Map<Double, ArrayList<JFragmentIon>> res = peaks.subList(0, peaks.size())
                .stream()
                .map(p -> p.mz)
                .distinct()
                .collect(toMap(
                        mz -> mz,
                        mz -> this.single_fragment_ion_query_for_dia(ms2index, mz, rt_start, rt_end, is_ppm,isoWinID)));

        List<Double> all_mzs = new ArrayList<>(res.keySet());
        for(double mz: all_mzs){
            if(res.get(mz).isEmpty()){
                res.remove(mz);
            }
        }
        // mz of each fragment ion
        List<Double> fragment_ions = res.keySet().stream().sorted().collect(toList());
        if(res.size()>=4){
            List<Integer> unique_scans = res.values().stream()
                    .flatMap(Collection::stream)
                    .map(ion -> ion.scan)
                    .distinct()
                    .sorted()
                    .collect(toList());

            if(unique_scans.size()>=ms2index.min_scan_for_peak) {

                int scan_min = Collections.min(unique_scans);
                int scan_max = Collections.max(unique_scans);
                int index_min = ms2index.isolation_win2scan2index.get(isoWinID).get(scan_min);
                int index_max = ms2index.isolation_win2scan2index.get(isoWinID).get(scan_max);

                // extend to the extraction window
                // left side
                if(ms2index.get_rt_by_scan(isoWinID,scan_min) > rt_start && Math.abs(ms2index.get_rt_by_scan(isoWinID,scan_min) - rt_start) > 0.01){
                    int scan_i = scan_min;
                    int index_i = index_min;
                    while(ms2index.isolation_win2scan2rt.get(isoWinID).containsKey(scan_i) && ms2index.get_rt_by_scan(isoWinID,scan_i) > rt_start){
                        index_i = index_i - 1;
                        if(ms2index.isolation_win2index2scan.get(isoWinID).containsKey(index_i)){
                            scan_i = ms2index.get_scan_by_index(isoWinID, index_i);
                        }else{
                            index_i = index_i + 1;
                            break;
                        }
                    }
                    index_min = index_i;
                }
                // right side
                if(ms2index.get_rt_by_scan(isoWinID,scan_max) < rt_end && Math.abs(ms2index.get_rt_by_scan(isoWinID,scan_max) - rt_end) > 0.01) {
                    int scan_i = scan_max;
                    int index_i = index_max;
                    while (ms2index.isolation_win2scan2rt.get(isoWinID).containsKey(scan_i) && ms2index.get_rt_by_scan(isoWinID, scan_i) < rt_end) {
                        index_i = index_i + 1;
                        if(ms2index.isolation_win2index2scan.get(isoWinID).containsKey(index_i)){
                            scan_i = ms2index.get_scan_by_index(isoWinID, index_i);
                        }else{
                            index_i = index_i - 1;
                            break;
                        }
                    }
                    index_max = index_i;
                }

                int scan_num = index_max - index_min + 1;

                HashMap<Integer, Integer> scan2index = new HashMap<>(unique_scans.size());
                HashMap<Integer, Integer> index2scan = new HashMap<>(unique_scans.size());
                //HashMap<Integer, Float> scan2rt = new HashMap<>(unique_scans.size());
                double [] index2rt = new double[scan_num];

                PeptidePeak peak = new PeptidePeak();
                peak.fragment_ions_mz = fragment_ions;
                double rt;
                double max_rt = 0;
                boolean apex_found = false;
                for (int i = 0; i < scan_num; i++) {
                    int cur_index = index_min + i;
                    int cur_scan = ms2index.get_scan_by_index(isoWinID,cur_index);
                    scan2index.put(cur_scan, i);
                    index2scan.put(i, cur_scan);

                    rt = ms2index.get_rt_by_scan(isoWinID,cur_scan);
                    index2rt[i] = rt;

                    if(Math.abs(rt-peptideMatch.rt_apex)<=0.01){
                        peak.apex_index = i;
                        apex_found = true;
                    }
                    if(Math.abs(rt-peptideMatch.rt_start)<=0.01){
                        peak.boundary_left_index = i;
                    }
                    if(Math.abs(rt-peptideMatch.rt_end)<=0.01){
                        peak.boundary_right_index = i;
                    }
                    if(max_rt < rt){
                        max_rt = rt;
                    }

                }
                if(!apex_found){
                    System.out.println("Apex not found:"+peptideMatch.rt_apex+","+peptideMatch.rt_start+","+peptideMatch.rt_end);
                    System.exit(1);
                }
                int mz_length = libSpectrum.spectrum.mz.length;
                HashMap<Double,Double> mz2int = new HashMap<>(mz_length);
                for(int i=0;i<mz_length;i++){
                    mz2int.put(libSpectrum.spectrum.mz[i],libSpectrum.spectrum.intensity[i]);
                }

                // fragment ion intensity
                double[][] frag_int = new double[res.size()][index_max-index_min+1];

                for (int j = 0; j < fragment_ions.size(); j++) {
                    for (JFragmentIon ion : res.get(fragment_ions.get(j))) {
                        //System.out.println(j+"\t"+ scan2index.get(ion.scan));
                        if(frag_int[j][scan2index.get(ion.scan)] < ion.intensity){
                            frag_int[j][scan2index.get(ion.scan)] = ion.intensity;
                        }
                    }
                }

                RealMatrix pepXIC_smoothed;
                if(ms2index.sg_smoothing_data_points==5){
                    pepXIC_smoothed = SGFilter5points.paddedSavitzkyGolaySmooth3(frag_int);
                }else if(ms2index.sg_smoothing_data_points==7){
                    pepXIC_smoothed = SGFilter7points.paddedSavitzkyGolaySmooth3(frag_int);
                }else if(ms2index.sg_smoothing_data_points==9){
                    pepXIC_smoothed = SGFilter.paddedSavitzkyGolaySmooth3(frag_int);
                }else if(ms2index.sg_smoothing_data_points==3){
                    pepXIC_smoothed = SGFilter3points.paddedSavitzkyGolaySmooth3(frag_int);
                }else{
                    // in default use 5 data points
                    pepXIC_smoothed = SGFilter5points.paddedSavitzkyGolaySmooth3(frag_int);
                }

                if(peak.boundary_right_index >= peak.apex_index) {

                    try {
                        if(refine_peak_boundary){
                            long original_peak_index = peak.apex_index;
                            long boundary_left_index = peak.boundary_left_index;
                            long boundary_right_index = peak.boundary_right_index;
                            boolean is_refined = refine_peak_boundary_detection(pepXIC_smoothed, peak, ms2index, isoWinID, index2scan,libSpectrum);
                            if(is_refined){
                                if(peak.boundary_left_index <= original_peak_index && original_peak_index <= peak.boundary_right_index ){
                                    //peak.apex_index = original_peak_index;
                                    peptideMatch.rt_start = peak.boundary_left_rt;
                                    peptideMatch.rt_end = peak.boundary_right_rt;
                                    peptideMatch.rt_apex = peak.apex_rt;
                                }else{
                                    System.err.println("The original apex index is not in the refined peak boundary:"+original_peak_index+","+
                                            boundary_left_index+","+
                                            boundary_right_index+","+
                                            peak.apex_index+","+
                                            peak.boundary_left_index+","+peak.boundary_right_index+","+
                                            peak.boundary_left_rt+","+
                                            peak.apex_rt+","+
                                            peak.boundary_right_rt+","+
                                            pepXIC_smoothed.getRowDimension()+","+
                                            pepXIC_smoothed.getColumnDimension());
                                    System.err.println(peptideMatch.peptide.getSequence()+"\t"+peptideMatch.index+"\t"+peptideMatch.precursor_charge+"\t"+peptideMatch.scan+"\t"+peptideMatch.rt_start+"\t"+peptideMatch.rt_apex+"\t"+peptideMatch.rt_end);
                                    // If this is the case, use the original peak boundary
                                    peak.boundary_left_index = boundary_left_index;
                                    peak.boundary_right_index = boundary_right_index;
                                    peak.apex_index = original_peak_index;
                                    peak.boundary_left_rt = ms2index.get_rt_by_scan(isoWinID,index2scan.get((int) peak.boundary_left_index));
                                    peak.boundary_right_rt = ms2index.get_rt_by_scan(isoWinID,index2scan.get((int) peak.boundary_right_index));
                                    peak.apex_rt = ms2index.get_rt_by_scan(isoWinID,index2scan.get((int) peak.apex_index));
                                    peptideMatch.rt_start = peak.boundary_left_rt;
                                    peptideMatch.rt_end = peak.boundary_right_rt;
                                    peptideMatch.rt_apex = peak.apex_rt;
                                }
                            }else{
                                System.out.println("No refining: "+peptideMatch.index+"\t"+peptideMatch.scan+"\t"+peptideMatch.rt_start+"\t"+peptideMatch.rt_apex+"\t"+peptideMatch.rt_end);
                            }
                        }
                        peak.cor_to_best_ion = ms2index.detect_best_ion(pepXIC_smoothed, (int) peak.boundary_left_index, (int) peak.boundary_right_index, (int) peak.apex_index, peptideMatch);
                        peptideMatch.peak = peak;
                        for (int j = 0; j < fragment_ions.size(); j++) {
                            peptideMatch.mz2cor.put(fragment_ions.get(j), peak.cor_to_best_ion[j]);
                            peptideMatch.mz2skewed_peaks.put(fragment_ions.get(j), peptideMatch.skewed_peaks[j]);
                        }

                        // For XIC
                        peptideMatch.smoothed_fragment_intensities = pepXIC_smoothed;
                        peptideMatch.raw_fragment_intensities = frag_int;
                        peptideMatch.xic_rt_values = index2rt;
                    } catch (NumberIsTooSmallException e) {
                        System.out.println("index_start: " + peak.boundary_left_index);
                        System.out.println("index_end: " + peak.boundary_right_index);
                        System.out.println("index_apex: " + peak.apex_index);
                        System.out.println("x: " + pepXIC_smoothed.getRowDimension());
                        System.out.println("x: " + pepXIC_smoothed.getColumnDimension());
                        System.out.println(peptideMatch.rt_start);
                        System.out.println(peptideMatch.rt_apex);
                        System.out.println(peptideMatch.rt_end);
                        System.out.println(max_rt);

                        for (int i = 0; i < scan_num; i++) {
                            int cur_index = index_min + i;
                            int cur_scan = ms2index.get_scan_by_index(isoWinID, cur_index);
                            rt = ms2index.get_rt_by_scan(isoWinID, cur_scan);
                            System.out.println(rt);

                        }
                        e.printStackTrace();
                        System.exit(1);
                    }
                }else{
                    System.out.println("index_start: " + peak.boundary_left_index);
                    System.out.println("index_end: " + peak.boundary_right_index);
                    System.out.println("index_apex: " + peak.apex_index);
                    System.out.println("x: " + pepXIC_smoothed.getRowDimension());
                    System.out.println("x: " + pepXIC_smoothed.getColumnDimension());
                    System.out.println(peptideMatch.rt_start);
                    System.out.println(peptideMatch.rt_apex);
                    System.out.println(peptideMatch.rt_end);
                    System.out.println(max_rt);

                    for (int i = 0; i < scan_num; i++) {
                        int cur_index = index_min + i;
                        int cur_scan = ms2index.get_scan_by_index(isoWinID, cur_index);
                        rt = ms2index.get_rt_by_scan(isoWinID, cur_scan);
                        System.out.println(rt);

                    }
                }

            }
        }
    }

    public boolean refine_peak_boundary_detection(RealMatrix x, PeptidePeak peak, DIAIndex ms2index, String isoWinID, HashMap<Integer, Integer> index2scan, LibSpectrum libSpectrum){
        boolean is_refined = false;
        // select the top 12 high abundant fragments
        int flank_scans = 2; // a total of 5 scans are considered to determine
        int n_ions = x.getRowDimension();
        HashMap<Integer,Double> index2intensity = new HashMap<>(n_ions);
        int left_index = Math.max((int) peak.apex_index - flank_scans,0);
        int right_index = Math.min((int) peak.apex_index + flank_scans,x.getColumnDimension()-1);
        int n_data_points = right_index - left_index + 1;
        HashMap<Double,Integer> mz2i = new HashMap<>();
        if(this.lf_frag_n_min>1){
            for(int i=0;i<libSpectrum.ion_numbers.length;i++){
                mz2i.put(libSpectrum.spectrum.mz[i],i);
            }
        }
        for(int i=0;i<n_ions;i++){
            // only consider the fragment ions with n>=this.lf_frag_n_min
            if(this.lf_frag_n_min>1){
                if(libSpectrum.ion_numbers[mz2i.get(peak.fragment_ions_mz.get(i))] < this.lf_frag_n_min){
                    continue;
                }
            }
            index2intensity.put(i,StatUtils.percentile(x.getRow(i), left_index, n_data_points, 50));
        }
        if(index2intensity.size()<=3){
            // use all fragment ions
            for(int i=0;i<n_ions;i++){
                index2intensity.put(i,StatUtils.percentile(x.getRow(i), left_index, n_data_points, 50));
            }
        }
        Map<Integer, Double> sorted_index2intensity = index2intensity.entrySet().stream()
                .sorted(Map.Entry.<Integer,Double>comparingByValue().reversed())
                .limit(12)
                .filter(entry -> entry.getValue()>0)
                .collect(toMap(Map.Entry::getKey, Map.Entry::getValue));
        if(sorted_index2intensity.size()>=3){
            int n_scans = x.getColumnDimension();
            int[] scan_index = new int[n_scans];
            int k = 0;
            for (int i = 0; i < n_scans; i++) {
                scan_index[k] = i;
                k = k + 1;
            }
            RealMatrix new_x = x.getSubMatrix(sorted_index2intensity.keySet().stream().mapToInt(i -> i).toArray(), scan_index);
            double[] median_peaks = new double[n_scans];
            if (sorted_index2intensity.size() == 1) {
                median_peaks = new_x.getRow(0);
            } else {
                for (int i = 0; i < n_scans; i++) {
                    // median_peaks[i] = StatUtils.percentile(new_x.getColumn(i), 50);
                    median_peaks[i] = Quantiles.median().compute(new_x.getColumn(i));
                }
            }
            XICtool xiCtool = new XICtool();
            PeptidePeak new_peak = xiCtool.find_max_peak(median_peaks,(int) peak.apex_index);
            if ((new_peak.boundary_right_index - new_peak.boundary_left_index + 1) >= 2) {
                // left_index = (int) peak.boundary_left_index;
                peak.boundary_left_index = new_peak.boundary_left_index;
                peak.boundary_right_index = new_peak.boundary_right_index;
                peak.apex_index = new_peak.apex_index;
                peak.min_smoothed_intensity = new_peak.min_smoothed_intensity;
                // refine peak
                peak.cor_to_best_ion = xiCtool.detect_best_ion(new_x,(int) peak.boundary_left_index, (int) peak.boundary_right_index,(int) peak.apex_index);
                xiCtool.refine_peak(new_x,peak,peak.cor_to_best_ion,0.75,false);
                peak.boundary_left_rt = ms2index.get_rt_by_scan(isoWinID,index2scan.get((int) peak.boundary_left_index));
                peak.boundary_right_rt = ms2index.get_rt_by_scan(isoWinID,index2scan.get((int) peak.boundary_right_index));
                peak.apex_rt = ms2index.get_rt_by_scan(isoWinID,index2scan.get((int) peak.apex_index));
                is_refined = true;
            }else{
                System.out.println("Peak too narrow!");
            }

        }else{
            System.out.println("few fragment ions detected");

        }
        return is_refined;
    }

    private ArrayList<JFragmentIon> single_fragment_ion_query_for_dia(DIAIndex ms2index, double mz, double rt_start, double rt_end, boolean is_ppm, String isoWinID){
        ArrayList<JFragmentIon> scans = new ArrayList<>();
        double[] frag_mz_range = CParameter.getRangeOfMass(mz,CParameter.itol,is_ppm);
        long mass_bin_left = ms2index.meta.get_fragment_ion_mz_bin_index(frag_mz_range[0]);
        long mass_bin_right = ms2index.meta.get_fragment_ion_mz_bin_index(frag_mz_range[1]);
        for(long frag_ion_bin = mass_bin_left; frag_ion_bin<= mass_bin_right; frag_ion_bin++){
            //System.out.println("Here:"+isoWinID);
            if(ms2index.frag_ion_index.get(isoWinID).containsKey(frag_ion_bin)){
                List<JFragmentIon> res = ms2index.frag_ion_index.get(isoWinID).get(frag_ion_bin)
                        .stream()
                        .filter(jFragmentIon -> Math.abs(CParameter.get_mass_error(jFragmentIon.mz,mz,is_ppm)) <= CParameter.itol &&
                                        jFragmentIon.rt >= rt_start && jFragmentIon.rt <= rt_end)
                        .collect(toList());
                scans.addAll(res);
            }
        }
        return scans;
    }

    private int get_ion_type_column_index(IonMatch ionMatch){
        String ion_type = "";
        if(ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION){
            ion_type = "b";
        }else if(ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION){
            ion_type = "y";
        }
        if(ionMatch.ion.hasNeutralLosses()){
            if(ionMatch.ion.getNeutralLosses().length==1){
                if(ionMatch.ion.getNeutralLosses()[0].isSameAs(NeutralLoss.H3PO4) && this.mod_ai.equals("phosphorylation")){
                    ion_type = ion_type + "_modloss_z" + ionMatch.charge;
                }else{
                    System.out.println("Neutral loss is not supported yet");
                    System.out.println(ionMatch.ion.getNeutralLosses()[0].name);
                    System.out.println(ionMatch.ion.getNeutralLosses()[0].getMass());
                    System.exit(1);
                }

            }else{
                // >=2 neutral losses
                System.out.println(">=2 neutral losses");
                for(NeutralLoss nl: ionMatch.ion.getNeutralLosses()){
                    System.out.println(nl.name);
                }
                System.exit(1);
            }

        }else{
            // no neutral loss
            ion_type = ion_type + "_z" + ionMatch.charge;
        }
        //ion_type = ion_type + "_z" + ionMatch.charge;
        return(this.ion_type2column_index.get(ion_type));
    }

    private int get_ion_type_column_index(Ion ion, int charge){
        String ion_type = "";
        if(ion.getSubType() == PeptideFragmentIon.B_ION){
            ion_type = "b";
        }else if(ion.getSubType() == PeptideFragmentIon.Y_ION){
            ion_type = "y";
        }
        if(ion.hasNeutralLosses()){
            if(ion.getNeutralLosses().length==1){
                if(ion.getNeutralLosses()[0].isSameAs(NeutralLoss.H3PO4) && this.mod_ai.equals("phosphorylation")){
                    ion_type = ion_type + "_modloss_z" + charge;
                }else{
                    System.out.println("Neutral loss is not supported yet");
                    System.out.println(ion.getNeutralLosses()[0].name);
                    System.out.println(ion.getNeutralLosses()[0].getMass());
                    System.exit(1);
                }

            }else{
                // >=2 neutral losses
                System.out.println(">=2 neutral losses");
                for(NeutralLoss nl: ion.getNeutralLosses()){
                    System.out.println(nl.toString());
                }
                System.exit(1);
            }

        }else{
            // no neutral loss
            ion_type = ion_type + "_z" + charge;
        }

        return(this.ion_type2column_index.get(ion_type));
    }

    private void set_ion_type_column_index(String fragmentation_type, int max_fragment_ion_charge, boolean lossWaterNH3){
        if(fragmentation_type.equalsIgnoreCase("hcd") || fragmentation_type.equalsIgnoreCase("cid")){
            ArrayList<String> col_names = new ArrayList<>();

            int column_index = 0;
            // b ion
            for(int i=1;i<=max_fragment_ion_charge;i++){
                this.ion_type2column_index.put("b_z"+i,column_index);
                column_index = column_index + 1;
                col_names.add("b_z"+i);
            }
            // y ion
            for(int i=1;i<=max_fragment_ion_charge;i++){
                this.ion_type2column_index.put("y_z"+i,column_index);
                column_index = column_index + 1;
                col_names.add("y_z"+i);
            }

            if(!(this.mod_ai.equals("-") || this.mod_ai.equals("general"))){
                // neutral loss
                // b ion
                for(int i=1;i<=max_fragment_ion_charge;i++){
                    this.ion_type2column_index.put("b_modloss_z"+i,column_index);
                    column_index = column_index + 1;
                    col_names.add("b_modloss_z"+i);
                }
                // y ion
                for(int i=1;i<=max_fragment_ion_charge;i++){
                    this.ion_type2column_index.put("y_modloss_z"+i,column_index);
                    column_index = column_index + 1;
                    col_names.add("y_modloss_z"+i);
                }
            }

            this.fragment_ion_intensity_head_line = StringUtils.join(col_names,"\t");
        }
    }

    private int get_valid_max_fragment_ions(int precursor_charge){
        if(precursor_charge<=2){
            return precursor_charge;

        }
        return Math.min(precursor_charge,this.max_fragment_ion_charge);
    }

    private ArrayList<IonMatch> get_matched_ions(Peptide objPeptide, Spectrum spectrum, int precursor_charge, int max_fragment_ion_charge, boolean lossWaterNH3) {

        PeptideSpectrumAnnotator peptideSpectrumAnnotator = new PeptideSpectrumAnnotator();

        // int charge = spectrum.getPrecursor().possibleCharges[0];
        PeptideAssumption peptideAssumption = new PeptideAssumption(objPeptide, precursor_charge);
        SpecificAnnotationParameters specificAnnotationPreferences = new SpecificAnnotationParameters();

        HashSet<Integer> charges = new HashSet<>(4);
        int precursorCharge = peptideAssumption.getIdentificationCharge();
        if (precursorCharge <= 1) {
            charges.add(precursorCharge);
        } else {
            int cur_max_fragment_ion_charge = Math.min(precursorCharge, max_fragment_ion_charge);
            if(this.fragment_ion_charge_less_than_precursor_charge) {
                if (precursor_charge >= 2 && precursorCharge == cur_max_fragment_ion_charge) {
                    cur_max_fragment_ion_charge = cur_max_fragment_ion_charge - 1;
                }
            }
            for (int c = 1; c <= cur_max_fragment_ion_charge; c++) {
                charges.add(c);
            }
        }
        specificAnnotationPreferences.setSelectedCharges(charges);

        specificAnnotationPreferences.addIonType(Ion.IonType.PEPTIDE_FRAGMENT_ION, PeptideFragmentIon.B_ION);
        specificAnnotationPreferences.addIonType(Ion.IonType.PEPTIDE_FRAGMENT_ION, PeptideFragmentIon.Y_ION);
        specificAnnotationPreferences.setFragmentIonAccuracy(CParameter.itol);
        specificAnnotationPreferences.setFragmentIonPpm(CParameter.itolu.startsWith("ppm"));
        specificAnnotationPreferences.setNeutralLossesAuto(false);
        specificAnnotationPreferences.clearNeutralLosses();
        // this is important
        specificAnnotationPreferences.setPrecursorCharge(precursorCharge);

        if (lossWaterNH3) {
            specificAnnotationPreferences.addNeutralLoss(NeutralLoss.H2O);
            specificAnnotationPreferences.addNeutralLoss(NeutralLoss.NH3);
        }

        AnnotationParameters annotationSettings = new AnnotationParameters();
        // annotationSettings.setTiesResolution(SpectrumAnnotator.TiesResolution.mostIntense);
        annotationSettings.setTiesResolution(SpectrumAnnotator.TiesResolution.mostAccurateMz);
        annotationSettings.setFragmentIonAccuracy(CParameter.itol);
        annotationSettings.setFragmentIonPpm(CParameter.itolu.startsWith("p"));
        annotationSettings.setIntensityLimit(CParameter.fragment_ion_intensity_cutoff);
        annotationSettings.setNeutralLossesSequenceAuto(false);
        annotationSettings.setIntensityThresholdType(AnnotationParameters.IntensityThresholdType.percentile);


        if(this.mod_ai.equals("general") || this.mod_ai.equals("-")) {
            // no any neutral loss
        }else if(this.mod_ai.equals("phosphorylation")) {
            if (ModificationUtils.getInstance().getModificationString(objPeptide).toLowerCase().contains("phosphorylation")) {
                //annotationSettings.addNeutralLoss(NeutralLoss.H3PO4);
                specificAnnotationPreferences.setNeutralLossesMap(getNeutralLossesMap(objPeptide));
                //annotationSettings.addNeutralLoss(NeutralLoss.HPO3);
                //specificAnnotationPreferences.addNeutralLoss(NeutralLoss.H3PO4);
                //specificAnnotationPreferences.addNeutralLoss(NeutralLoss.HPO3);
            }
        }else{
            // TODO
        }

        annotationSettings.setIntensityThresholdType(AnnotationParameters.IntensityThresholdType.percentile);

        ModificationParameters modificationParameters = new ModificationParameters();
        SequenceMatchingParameters sequenceMatchingParameters = new SequenceMatchingParameters();
        JSequenceProvider jSequenceProvider = new JSequenceProvider();
        IonMatch[] matches = peptideSpectrumAnnotator.getSpectrumAnnotation(annotationSettings,
                specificAnnotationPreferences,
                "",
                "",
                spectrum,
                objPeptide,
                modificationParameters,
                jSequenceProvider,
                sequenceMatchingParameters);


        if (matches == null || matches.length == 0) {
            System.err.println("No ions matched!");
            return (new ArrayList<>());
        }else{
            return new ArrayList<>(Arrays.asList(matches));
        }

    }

    public static NeutralLossesMap getNeutralLossesMap(Peptide peptide) {
        // ModificationFactory modificationFactory = ModificationFactory.getInstance();
        NeutralLossesMap neutralLossesMap = new NeutralLossesMap();
        String sequence = peptide.getSequence();
        int aaMin = sequence.length();
        int aaMax = 0;

        ModificationMatch[] modificationMatches = peptide.getVariableModifications();

        for (ModificationMatch modMatch : modificationMatches) {

            if(modMatch.getModification().equals("Phosphorylation of S") || modMatch.getModification().equals("Phosphorylation of T")) {
                int site = com.compomics.util.experiment.identification.utils.ModificationUtils.getSite(
                        modMatch.getSite(),
                        sequence.length()
                );
                aaMin = site;
                aaMax = sequence.length() - site + 1;

                neutralLossesMap.addNeutralLoss(
                        NeutralLoss.H3PO4,
                        aaMin,
                        aaMax
                );
            }
        }
        return neutralLossesMap;
    }


    private Peptide get_peptide(String peptide_sequence, String modification){
        String peptide_mod = peptide_sequence + modification;
        return peptide_mod2Peptide.get(peptide_mod);
    }

    private void add_peptide(String peptide_sequence, String modification){
        String peptide_mod = peptide_sequence + modification;
        if(!this.peptide_mod2Peptide.containsKey(peptide_mod)){
            Peptide peptide = generatePeptide(peptide_sequence,modification);
            this.peptide_mod2Peptide.put(peptide_mod,peptide);
        }
    }

    public Peptide generatePeptide(String peptideSequence, String modifications){
        Peptide peptide = new Peptide(peptideSequence);
        if(!modifications.equals("-")){
            // TMT 10-plex of K@8[229.1629];TMT 10-plex of K@9[229.1629];TMT 10-plex of peptide N-term@0[229.1629]
            String [] names = modifications.split(";");
            for (String s : names) {
                String name = s.replaceAll("@.*$", "");
                String pos = s.replaceAll(".*@(\\d+).*$", "$1");
                peptide.addVariableModification(new ModificationMatch(name, Integer.parseInt(pos)));
            }
        }
        peptide.getMass(modificationParameters,sequenceProvider,sequenceMatchingParameters);
        return peptide;
    }

    // only use this for preprocessing DIA-NN result for model training.
    public void remove_interference_peptides(String psm_file, String new_psm_file, double fdr_cutoff){
        CsvReadOptions.Builder builder = CsvReadOptions.builder(psm_file)
                .separator('\t')
                .header(true);
        CsvReadOptions options = builder.build();
        Table psmTable = Table.read().usingOptions(options);
        if(search_engine.equalsIgnoreCase("DIA-NN") || search_engine.equalsIgnoreCase("DIANN")){
            // psmTable = psmTable.sortOn("File.Name","Q.Value","");
            // TODO
        }else{

            if(fdr_cutoff >0 && fdr_cutoff < 1){
                psmTable = psmTable.where(psmTable.doubleColumn("q_value").isLessThanOrEqualTo(fdr_cutoff)).copy();
            }
            psmTable = remove_interference_peptides(psmTable);
        }

        CsvWriteOptions writeOptions = CsvWriteOptions.builder(new_psm_file)
                .separator('\t')
                .header(true)
                .build();
        // Write the table to a TSV file
        psmTable.write().usingOptions(writeOptions);

    }

    public static Table remove_interference_peptides(Table psmTable){
        String peptide;
        String mz;
        String charge;
        double rt_start;
        double apex_rt;
        double rt_end;
        HashMap<String,ArrayList<JPeakGroup>> peptide_form2peptides = new HashMap<>();
        psmTable = psmTable.sortOn("peptide","charge","mz","-rescore");
        String peptide_form;
        IntColumn valid_column = IntColumn.create("peak_share",psmTable.rowCount());
        for(int i=0;i<psmTable.rowCount();i++){
            peptide = psmTable.getString(i, "peptide");
            mz = psmTable.getString(i, "mz");
            charge = psmTable.getString(i, "charge");
            rt_start = psmTable.row(i).getDouble("rt_start");
            apex_rt = psmTable.row(i).getDouble("apex_rt");
            rt_end = psmTable.row(i).getDouble("rt_end");
            peptide_form = peptide + "|" + charge + "|" + mz;
            if(!peptide_form2peptides.containsKey(peptide_form)){
                peptide_form2peptides.put(peptide_form,new ArrayList<>());
                JPeakGroup peak = new JPeakGroup();
                peak.rt_start = rt_start;
                peak.apex_rt = apex_rt;
                peak.rt_end = rt_end;
                peak.id = i;
                peptide_form2peptides.get(peptide_form).add(peak);
                valid_column.set(i, 1);
            }else{
                // need to compare with each previous peptide form
                boolean keep = true;
                for(JPeakGroup p: peptide_form2peptides.get(peptide_form)){
                    if(p.rt_start <= rt_start && rt_end <= p.rt_end){
                        // one contained by another
                        keep = false;
                        break;
                    }else if(rt_start <= p.rt_start && p.rt_end <= rt_end){
                        // one contained by another
                        keep = false;
                        break;
                    }else if(p.rt_start <= rt_start && rt_start <= p.rt_end){
                        // overlap
                        double overlap = p.rt_end - rt_start;
                        double overlap_ratio = Math.max(overlap / (rt_end - rt_start), overlap/ (p.rt_end - p.rt_start));
                        if(overlap_ratio >= 0.5){
                            keep = false;
                            break;
                        }
                    }else if(p.rt_start <= rt_end && rt_end <= p.rt_end){
                        // overlap
                        double overlap =rt_end - p.rt_start;
                        double overlap_ratio = Math.max(overlap / (rt_end - rt_start), overlap/ (p.rt_end - p.rt_start));
                        if(overlap_ratio >= 0.5){
                            keep = false;
                            break;
                        }
                    }
                }
                if(keep){
                    JPeakGroup peak = new JPeakGroup();
                    peak.rt_start = rt_start;
                    peak.apex_rt = apex_rt;
                    peak.rt_end = rt_end;
                    peak.id = i;
                    peptide_form2peptides.get(peptide_form).add(peak);
                    valid_column.set(i, 1);
                }else{
                    valid_column.set(i, 0);
                }
            }

        }
        psmTable.addColumns(valid_column);
        return psmTable;
    }

    public void remove_interference_peptides_diann(String psm_file, String new_psm_file){
        CsvReadOptions.Builder builder = CsvReadOptions.builder(psm_file)
                .maxCharsPerColumn(10000000)
                .separator('\t')
                .header(true);
        CsvReadOptions options = builder.build();
        Table psmTable = Table.read().usingOptions(options);

        String peptide;
        double mz;
        String charge;
        String mod_seq;
        double rt_start;
        double apex_rt;
        double rt_end;
        String modification;

        DoubleColumn mz_column = DoubleColumn.create("mz",psmTable.rowCount());
        for(int i=0;i<psmTable.rowCount();i++) {
            peptide = psmTable.getString(i, "Stripped.Sequence");
            charge = psmTable.getString(i, "Precursor.Charge");
            mod_seq = psmTable.getString(i, "Modified.Sequence");
            modification = this.get_modification_diann(mod_seq, peptide);
            this.add_peptide(peptide, modification);
            mz_column.set(i, get_mz(this.get_peptide(peptide, modification).getMass(), Integer.parseInt(charge)));
        }
        psmTable.addColumns(mz_column);
        HashMap<String,ArrayList<JPeakGroup>> peptide_form2peptides = new HashMap<>();
        psmTable = psmTable.sortOn("File.Name","Stripped.Sequence","Precursor.Charge","mz","Q.Value","PEP");
        String peptide_form;
        // if a peptide form is overlapped with a peptide form with higher score
        IntColumn valid_column = IntColumn.create("peak_share",psmTable.rowCount());
        // HashMap<String,Integer> peptide_mz2count = new HashMap<>();
        HashMap<Integer,JPeakGroup> id2peak = new HashMap<>();
        for(int i=0;i<psmTable.rowCount();i++){
            peptide = psmTable.getString(i, "Stripped.Sequence");
            charge = psmTable.getString(i, "Precursor.Charge");
            mz = mz_column.get(i);
            rt_start = psmTable.row(i).getDouble("RT.Start");
            apex_rt = psmTable.row(i).getDouble("RT");
            rt_end = psmTable.row(i).getDouble("RT.Stop");
            peptide_form = peptide + "|" + charge + "|" + mz;
            if(!peptide_form2peptides.containsKey(peptide_form)){
                peptide_form2peptides.put(peptide_form,new ArrayList<>());
                JPeakGroup peak = new JPeakGroup();
                peak.rt_start = rt_start;
                peak.apex_rt = apex_rt;
                peak.rt_end = rt_end;
                peak.id = i;
                peptide_form2peptides.get(peptide_form).add(peak);
                valid_column.set(i, 1);
                id2peak.put(i,peak);
                //peptide_mz2count.put(peptide_form,1);
            }else{
                // need to compare with each previous peptide form
                boolean keep = true;
                for(JPeakGroup p: peptide_form2peptides.get(peptide_form)){
                    if(p.rt_start <= rt_start && rt_end <= p.rt_end){
                        // one contained by another
                        //peptide_mz2count.put(peptide_form,peptide_mz2count.get(peptide_form)+1);
                        p.n_shared_peaks++;
                        keep = false;
                        break;
                    }else if(rt_start <= p.rt_start && p.rt_end <= rt_end){
                        // one contained by another
                        //peptide_mz2count.put(peptide_form,peptide_mz2count.get(peptide_form)+1);
                        p.n_shared_peaks++;
                        keep = false;
                        break;
                    }else if(p.rt_start <= rt_start && rt_start <= p.rt_end){
                        // overlap
                        double overlap = p.rt_end - rt_start;
                        double overlap_ratio = Math.max(overlap / (rt_end - rt_start), overlap/ (p.rt_end - p.rt_start));
                        if(overlap_ratio >= 0.5){
                            p.n_shared_peaks++;
                            keep = false;
                            // peptide_mz2count.put(peptide_form,peptide_mz2count.get(peptide_form)+1);
                            break;
                        }
                    }else if(p.rt_start <= rt_end && rt_end <= p.rt_end){
                        // overlap
                        double overlap =rt_end - p.rt_start;
                        double overlap_ratio = Math.max(overlap / (rt_end - rt_start), overlap/ (p.rt_end - p.rt_start));
                        if(overlap_ratio >= 0.5){
                            //peptide_mz2count.put(peptide_form,peptide_mz2count.get(peptide_form)+1);
                            p.n_shared_peaks++;
                            keep = false;
                            break;
                        }
                    }
                }
                JPeakGroup peak = new JPeakGroup();
                peak.rt_start = rt_start;
                peak.apex_rt = apex_rt;
                peak.rt_end = rt_end;
                peak.id = i;
                id2peak.put(i,peak);
                if(keep){
                    valid_column.set(i, 1);
                }else{
                    valid_column.set(i, 0);
                    peak.n_shared_peaks++;
                }
                peptide_form2peptides.get(peptide_form).add(peak);

            }

        }

        // if a peptide form is overlapped with any peptide form.
        IntColumn overlap_column = IntColumn.create("peak_overlap",psmTable.rowCount());
        for(int i=0;i<psmTable.rowCount();i++){
            overlap_column.set(i, id2peak.get(i).n_shared_peaks);
        }
        psmTable.addColumns(overlap_column);
        psmTable.addColumns(valid_column);
        CsvWriteOptions writeOptions = CsvWriteOptions.builder(new_psm_file)
                .separator('\t')
                .header(true)
                .build();
        // Write the table to a TSV file
        psmTable.write().usingOptions(writeOptions);
    }

    public HashMap<String, ArrayList<String>> get_ms_file2psm(String psm_file, String ms_file, double fdr_cutoff) throws IOException {
        HashMap<String,Integer> hIndex = get_column_name2index(psm_file);
        BufferedReader psmReader = new BufferedReader(new FileReader(psm_file));
        psmReader.readLine();
        String line;
        String cur_ms_file = "-";

        HashMap<String, ArrayList<String>> ms_file2psm = new HashMap<>();
        int n_valid_row = 0;
        int n_total_row = 0;

        int n_peak_share = 0;

        while((line=psmReader.readLine())!=null){
            line = line.trim();
            n_total_row = n_total_row + 1;
            String []d = line.split("\t");

            if(hIndex.containsKey("q_value")){
                double q_value = Double.parseDouble(d[hIndex.get("q_value")]);
                if(q_value>fdr_cutoff){
                    continue;
                }
            }
            if(hIndex.containsKey("decoy")){
                String decoy = d[hIndex.get("decoy")];
                if(decoy.equalsIgnoreCase("Yes")){
                    continue;
                }
            }

            if(hIndex.containsKey("peak_share")){
                int peak_share = Integer.parseInt(d[hIndex.get("peak_share")]);
                if(peak_share==0){
                    n_peak_share++;
                    continue;
                }
            }

            if(hIndex.containsKey("ms_file")){
                cur_ms_file = d[hIndex.get("ms_file")];
            }else{
                cur_ms_file = ms_file;
            }

            if(!ms_file2psm.containsKey(ms_file)){
                ms_file2psm.put(cur_ms_file,new ArrayList<>());
            }
            ms_file2psm.get(cur_ms_file).add(line);
            n_valid_row = n_valid_row + 1;
        }

        psmReader.close();
        System.out.println("The number of MS files:"+ms_file2psm.size());
        System.out.println("The number of valid rows:"+n_valid_row);
        System.out.println("The number of total rows:"+n_total_row);
        if(n_peak_share>=1){
            System.out.println("The number of shared peaks:"+n_peak_share);
        }
        return ms_file2psm;
    }

    public HashMap<String, ArrayList<String>> get_ms_file2psm_diann(String psm_file, String ms_file, double fdr_cutoff) throws IOException {
        HashMap<String,Integer> hIndex = get_column_name2index(psm_file);
        BufferedReader psmReader = new BufferedReader(new FileReader(psm_file));
        psmReader.readLine();
        String line;
        String cur_ms_file = "-";

        HashMap<String, ArrayList<String>> ms_file2psm = new HashMap<>();
        int n_valid_row = 0;
        int n_total_row = 0;

        while((line=psmReader.readLine())!=null){
            line = line.trim();
            n_total_row = n_total_row + 1;
            String []d = line.split("\t");

            if(hIndex.containsKey("Q.Value")){
                double q_value = Double.parseDouble(d[hIndex.get("Q.Value")]);
                if(q_value>fdr_cutoff){
                    continue;
                }
            }


            if(hIndex.containsKey("File.Name")){
                cur_ms_file = d[hIndex.get("File.Name")];
                File F = new File(cur_ms_file);
                if(!F.exists()){
                    // ms_file should be a folder
                    Path path = Paths.get(cur_ms_file);
                    File MS = new File(ms_file);
                    if(MS.isFile()){
                        cur_ms_file = ms_file;
                    }else {
                        // ms_file is a folder
                        cur_ms_file = ms_file + File.separator + path.getFileName().toString();
                        // check this file exists or not
                        F = new File(cur_ms_file);
                        if (!F.exists()) {
                            System.out.println("File not found:" + cur_ms_file);
                            System.exit(1);
                        }
                    }
                }
            }else{
                cur_ms_file = ms_file;
            }

            if(!ms_file2psm.containsKey(cur_ms_file)){
                ms_file2psm.put(cur_ms_file,new ArrayList<>());
            }
            ms_file2psm.get(cur_ms_file).add(line);
            n_valid_row = n_valid_row + 1;
        }

        psmReader.close();
        System.out.println("The number of MS files:"+ms_file2psm.size());
        System.out.println("The number of valid rows:"+n_valid_row);
        System.out.println("The number of total rows:"+n_total_row);
        return ms_file2psm;
    }

    public HashMap<String,Integer> get_column_name2index(String file) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String head_line= reader.readLine().trim();
        this.psm_head_line = head_line;
        HashMap<String,Integer> hIndex = get_column_name2index_from_head_line(head_line);
        reader.close();
        return hIndex;
    }

    public HashMap<String,Integer> get_column_name2index_from_head_line(String head_line){
        this.psm_head_line = head_line;
        String []h = head_line.split("\t");
        HashMap<String,Integer> hIndex = new HashMap<>();
        for(int i=0;i<h.length;i++){
            hIndex.put(h[i],i);
        }
        return hIndex;
    }

    public int get_n_rows(String file, boolean header) throws IOException {
        BufferedReader pReader = new BufferedReader(new FileReader(file));
        String line;
        int n = 0;
        while ((line = pReader.readLine()) != null) {
            n++;
        }
        pReader.close();
        if(header){
            n = n - 1;
        }
        return n;
    }

    private HashMap<Integer, ArrayList<Ion>> generate_theoretical_fragment_ions(Peptide peptide, int precursor_charge){
        // generate theoretical fragment ions.
        PeptideFrag peptideFrag = new PeptideFrag();
        peptideFrag.init(precursor_charge, peptide, this.mod_ai);
        return peptideFrag.getExpectedFragIons(peptide);

    }

    private HashSet<Integer> getPossibleFragmentIonCharges(int precursorCharge) {
        HashSet<Integer> charges = new HashSet<>(4);
        if (precursorCharge <= 1) {
            charges.add(precursorCharge);
        } else {
            int cur_max_fragment_ion_charge = Math.min(precursorCharge, max_fragment_ion_charge);
            if(fragment_ion_charge_less_than_precursor_charge) {
                if (precursorCharge == cur_max_fragment_ion_charge) {
                    cur_max_fragment_ion_charge = cur_max_fragment_ion_charge - 1;
                }
            }
            for (int c = 1; c <= cur_max_fragment_ion_charge; c++) {
                charges.add(c);
            }
        }
        return charges;
    }

    public void generate_spectral_library(Map<String,HashMap<String,String>> res_files) throws IOException {
        if(this.use_parquet) {
            if(this.export_spectral_library_format.equalsIgnoreCase("Skyline")){
                try {
                    generate_spectral_library_parquet_skyline(res_files, this.out_dir, "SkylineAI_spectral_library.tsv");
                } catch (SQLException e) {
                    throw new RuntimeException(e);
                }
            }else if(this.export_spectral_library_format.equalsIgnoreCase("mzSpecLib")) {
                try {
                    generate_spectral_library_parquet_mzSpecLib(res_files, this.out_dir, "SkylineAI_spectral_library.tsv");
                } catch (SQLException e) {
                    throw new RuntimeException(e);
                }
            }else {
                generate_spectral_library_parquet(res_files, this.out_dir, "SkylineAI_spectral_library.tsv");
            }
        }else{
            generate_spectral_library(res_files, this.out_dir, "SkylineAI_spectral_library.tsv");
        }
    }

    public void generate_spectral_library(Map<String,HashMap<String,String>> res_files, String out_folder, String file_name) throws IOException {
        //String out_library_file = this.out_dir + File.separator + "SkylineAI_spectral_library.tsv";
        String out_library_file = out_folder + File.separator + file_name;
        BufferedWriter libWriter = new BufferedWriter(new FileWriter(out_library_file));
        libWriter.write("ModifiedPeptide\tStrippedPeptide\tPrecursorMz\tPrecursorCharge\tTr_recalibrated\tProteinID\tDecoy\tFragmentMz\tRelativeIntensity\tFragmentType\tFragmentNumber\tFragmentCharge\tFragmentLossType\n");

        DBGear dbGear = new DBGear();

        int pepID;
        String sequence;
        String mods;
        String mod_sites;
        double rt;
        String rt_str;
        for(String i : res_files.keySet()){
            Cloger.getInstance().logger.info(i);
            String ms2_file = res_files.get(i).get("ms2");
            if(ms2_file.endsWith("parquet")){
                // TODO
                System.err.println("Parquet is not supported in this function!");
                System.exit(1);
            }else{
                if(!get_column_name2index(ms2_file).containsKey("protein")){
                    if(CParameter.db.toLowerCase().endsWith(".fa") || CParameter.db.toLowerCase().endsWith(".fasta")) {
                        dbGear.add_protein_to_psm_table(ms2_file, CParameter.db);
                    }
                }
            }

            String ms2_intensity_file = res_files.get(i).get("ms2_intensity");
            String rt_file = res_files.get(i).get("rt");
            String ms2_mz_file = res_files.get(i).get("ms2_mz");

            // "_ms2_df.tsv"
            // pepID   sequence        charge  mods    mod_sites       nce     instrument      nAA     frag_start_idx  frag_stop_idx
            // "_ms2_mz_df.tsv"
            // b_z1    b_z2    y_z1    y_z2    b_modloss_z1    b_modloss_z2    y_modloss_z1    y_modloss_z2
            // "_ms2_pred.tsv"
            // b_z1    b_z2    y_z1    y_z2    b_modloss_z1    b_modloss_z2    y_modloss_z1    y_modloss_z2
            // "_rt_pred.tsv"
            // pepID   sequence        mods    mod_sites       nAA     rt_pred rt_norm_pred    irt_pred

            BufferedReader ms2Reader = new BufferedReader(new FileReader(ms2_file));
            BufferedReader ms2IntensityReader = new BufferedReader(new FileReader(ms2_intensity_file));
            BufferedReader rtReader = new BufferedReader(new FileReader(rt_file));
            BufferedReader ms2mzReader = new BufferedReader(new FileReader(ms2_mz_file));

            HashMap<String,Integer> ms2_col2index = this.get_column_name2index_from_head_line(ms2Reader.readLine().trim());
            HashMap<String,Integer> ms2_intensity_col2index = this.get_column_name2index_from_head_line(ms2IntensityReader.readLine().trim());
            HashMap<String,Integer> rt_col2index = this.get_column_name2index_from_head_line(rtReader.readLine().trim());
            String [] fragment_ion_column_names = ms2mzReader.readLine().trim().split("\t");
            //HashMap<String,Integer> ms2_mz_col2index = this.get_column_name2index_from_head_line(ms2mzReader.readLine().trim());

            String []ion_types = new String[fragment_ion_column_names.length];
            String []mod_losses = new String[fragment_ion_column_names.length];
            int []ion_charges = new int[fragment_ion_column_names.length];
            for(int j=0;j<fragment_ion_column_names.length;j++){
                if(fragment_ion_column_names[j].startsWith("b")){
                    ion_types[j] = "b";
                } else if (fragment_ion_column_names[j].startsWith("y")) {
                    ion_types[j] = "y";
                }else{
                    System.err.println("Unknown fragment ion type:"+fragment_ion_column_names[j]);
                    System.exit(1);
                }

                if(fragment_ion_column_names[j].endsWith("_z1")){
                    ion_charges[j] = 1;
                }else if(fragment_ion_column_names[j].endsWith("_z2")){
                    ion_charges[j] = 2;
                }else{
                    System.err.println("Unknown fragment ion charge:"+fragment_ion_column_names[j]);
                    System.exit(1);
                }

                if(fragment_ion_column_names[j].contains("modloss")) {
                    if(this.mod_ai.equalsIgnoreCase("phosphorylation")){
                        mod_losses[j] = "H3PO4";
                    }
                }else{
                    mod_losses[j] = "noloss";
                }
            }


            String line;
            // RT information
            HashMap<Integer,Double> pepID2rt = new HashMap<>();
            while((line=rtReader.readLine())!=null){
                String []d = line.split("\t");
                pepID = Integer.parseInt(d[rt_col2index.get("pepID")]);
                if(this.rt_max > 0){
                    rt = Double.parseDouble(d[rt_col2index.get("rt_pred")]);
                    // convert rt to normal rt value since the rt is min-max normalized rt
                    // peptideRT.rt_norm = (peptideRT.rt - this.rt_min)/(this.rt_max - this.rt_min);
                    //rt = rt * (this.rt_max - this.rt_min) + this.rt_min;
                    rt = rt * this.rt_max;
                }else{
                    rt = Double.parseDouble(d[rt_col2index.get("irt_pred")]);
                }
                pepID2rt.put(pepID, rt);
            }
            rtReader.close();

            // MS intensity
            ArrayList<String> ms2_intensity_lines = new ArrayList<>();
            while ((line=ms2IntensityReader.readLine())!=null){
                ms2_intensity_lines.add(line.trim());
            }
            ms2IntensityReader.close();

            // mz intensity
            ArrayList<String> ms2_mz_lines = new ArrayList<>();
            while ((line=ms2mzReader.readLine())!=null){
                ms2_mz_lines.add(line.trim());
            }
            ms2mzReader.close();

            // MS2 information
            while((line=ms2Reader.readLine())!=null){
                String []d = line.split("\t");
                pepID = Integer.parseInt(d[ms2_col2index.get("pepID")]);
                int frag_start_idx = Integer.parseInt(d[ms2_col2index.get("frag_start_idx")]);
                int frag_stop_idx = Integer.parseInt(d[ms2_col2index.get("frag_stop_idx")]);
                ArrayList<String> lines = get_fragment_ion_intensity(ms2_mz_lines,
                        ms2_intensity_lines,
                        fragment_ion_column_names,
                        frag_start_idx,
                        frag_stop_idx,this.lf_top_n_fragment_ions,
                        ion_types,
                        mod_losses,
                        ion_charges,
                        this.lf_frag_n_min);

                if(lines.size()<this.lf_min_n_fragment_ions){
                    continue;
                }

                rt_str = String.format("%.2f",pepID2rt.get(pepID));
                String mod_pep;
                if(this.export_spectral_library_format.equalsIgnoreCase("DIANN") || this.export_spectral_library_format.equalsIgnoreCase("DIA-NN")){
                    mod_pep = get_modified_peptide_diann(d[ms2_col2index.get("sequence")],d[ms2_col2index.get("mods")],d[ms2_col2index.get("mod_sites")]);
                }else if(this.export_spectral_library_format.equalsIgnoreCase("EncyclopeDIA")){
                    mod_pep = get_modified_peptide_encyclopedia(d[ms2_col2index.get("sequence")],d[ms2_col2index.get("mods")],d[ms2_col2index.get("mod_sites")]);
                }else{
                    mod_pep = get_modified_peptide(d[ms2_col2index.get("sequence")],d[ms2_col2index.get("mods")],d[ms2_col2index.get("mod_sites")]);
                }
                for(String l: lines) {
                    StringBuilder ob = new StringBuilder();
                    ob.append(mod_pep).append("\t")
                            .append(d[ms2_col2index.get("sequence")]).append("\t")
                            .append(d[ms2_col2index.get("mz")]).append("\t")
                            .append(d[ms2_col2index.get("charge")]).append("\t")
                            .append(rt_str).append("\t")
                            .append(d[ms2_col2index.get("protein")]).append("\t")
                            .append(d[ms2_col2index.get("decoy")].startsWith("Yes")?1:0).append("\t")
                            .append(l).append("\n");
                    libWriter.write(ob.toString());
                }
            }

            ms2Reader.close();
            ms2IntensityReader.close();
        }
        libWriter.close();
    }

    public void generate_spectral_library_parquet(Map<String,HashMap<String,String>> res_files, String out_folder, String file_name) throws IOException {
        //String out_library_file = this.out_dir + File.separator + "SkylineAI_spectral_library.tsv";
        String out_library_file = out_folder + File.separator + file_name;
        BufferedWriter libWriter = null;
        boolean export_tsv = true;
        ParquetWriter<LibFragment> pWriter = null;
        if(this.export_spectral_library_file_format.equalsIgnoreCase("parquet")){
            export_tsv = false;
            Schema schema = FileIO.getSchema4SpectralLib();
            String o_file = "";
            if(out_library_file.endsWith("tsv")){
                o_file = out_library_file.replaceAll("tsv$", "parquet");
            }else if(out_library_file.endsWith("txt")){
                o_file = out_library_file.replaceAll("txt$", "parquet");
            }else if(out_library_file.endsWith("csv")){
                o_file = out_library_file.replaceAll("csv$", "parquet");
            }else{
                if(!out_library_file.endsWith("parquet")){
                    System.err.println("The spectral library file suffix is not supported:"+out_library_file);
                    System.exit(1);
                }
            }
            // org.apache.hadoop.fs.Path path = new org.apache.hadoop.fs.Path(o_file);
            // OutputFile out_file = HadoopOutputFile.fromPath(path, new org.apache.hadoop.conf.Configuration());
            LocalOutputFile localOutputFile = new LocalOutputFile(Paths.get(o_file));
            pWriter = AvroParquetWriter.<LibFragment>builder(localOutputFile)
                    .withSchema(ReflectData.AllowNull.get().getSchema(LibFragment.class))
                    .withDataModel(ReflectData.get())
                    //.withCompressionCodec(CompressionCodecName.SNAPPY)
                    .withCompressionCodec(CompressionCodecName.ZSTD)
                    .withPageSize(ParquetWriter.DEFAULT_PAGE_SIZE)
                    //.withConf(new org.apache.hadoop.conf.Configuration())
                    .withValidation(false)
                    // override when existing
                    .withWriteMode(ParquetFileWriter.Mode.OVERWRITE)
                    .withDictionaryEncoding(true)
                    .build();
        }else {
            libWriter = new BufferedWriter(new FileWriter(out_library_file));
            libWriter.write("ModifiedPeptide\tStrippedPeptide\tPrecursorMz\tPrecursorCharge\tTr_recalibrated\tProteinID\tDecoy\tFragmentMz\tRelativeIntensity\tFragmentType\tFragmentNumber\tFragmentCharge\tFragmentLossType\n");
        }

        boolean export_diann_format = false;
        boolean export_EncyclopeDIA_format = false;
        boolean export_generic_format = false;
        if(this.export_spectral_library_format.equalsIgnoreCase("DIANN") || this.export_spectral_library_format.equalsIgnoreCase("DIA-NN")){
            export_diann_format = true;
        }else if(this.export_spectral_library_format.equalsIgnoreCase("EncyclopeDIA")){
            export_EncyclopeDIA_format = true;
        }else{
            export_generic_format = true;
        }


        DBGear dbGear = new DBGear();
        Map<String, String> pep2pro = new HashMap<>();
        if(CParameter.db.toLowerCase().endsWith(".fa") || CParameter.db.toLowerCase().endsWith(".fasta")) {
            pep2pro = dbGear.digest_protein(CParameter.db);
        }
        int pepID;
        String sequence;
        String mods;
        String mod_sites;
        double rt;
        String rt_str;
        double mz;
        int charge;
        String decoy;
        int decoy_label;
        String protein;
        int frag_start_idx;
        int frag_stop_idx;
        LibFragment libFragment = new LibFragment();
        for(String i : res_files.keySet()){
            Cloger.getInstance().logger.info(i);
            String ms2_file = res_files.get(i).get("ms2");
            if(!ms2_file.endsWith("parquet")){
                if(!get_column_name2index(ms2_file).containsKey("protein")){
                    if(CParameter.db.toLowerCase().endsWith(".fa") || CParameter.db.toLowerCase().endsWith(".fasta")) {
                        dbGear.add_protein_to_psm_table(ms2_file, CParameter.db);
                    }
                }
            }
            String ms2_intensity_file = res_files.get(i).get("ms2_intensity");
            String rt_file = res_files.get(i).get("rt");
            String ms2_mz_file = res_files.get(i).get("ms2_mz");

            // "_ms2_df.tsv"
            // pepID   sequence        charge  mods    mod_sites       nce     instrument      nAA     frag_start_idx  frag_stop_idx
            // "_ms2_mz_df.tsv"
            // b_z1    b_z2    y_z1    y_z2    b_modloss_z1    b_modloss_z2    y_modloss_z1    y_modloss_z2
            // "_ms2_pred.tsv"
            // b_z1    b_z2    y_z1    y_z2    b_modloss_z1    b_modloss_z2    y_modloss_z1    y_modloss_z2
            // "_rt_pred.tsv"
            // pepID   sequence        mods    mod_sites       nAA     rt_pred rt_norm_pred    irt_pred

            // HashMap<String,Integer> ms2_col2index = FileIO.get_column_name2index_from_head_line(ms2_file);
            // HashMap<String,Integer> ms2_intensity_col2index = FileIO.get_column_name2index_from_head_line(ms2_intensity_file);
            // HashMap<String,Integer> rt_col2index = FileIO.get_column_name2index_from_head_line(rt_file);
            String [] fragment_ion_column_names = FileIO.get_column_names_from_parquet(ms2_mz_file);
            //HashMap<String,Integer> ms2_mz_col2index = this.get_column_name2index_from_head_line(ms2mzReader.readLine().trim());

            String []ion_types = new String[fragment_ion_column_names.length];
            String []mod_losses = new String[fragment_ion_column_names.length];
            int []ion_charges = new int[fragment_ion_column_names.length];
            for(int j=0;j<fragment_ion_column_names.length;j++){
                if(fragment_ion_column_names[j].startsWith("b")){
                    ion_types[j] = "b";
                } else if (fragment_ion_column_names[j].startsWith("y")) {
                    ion_types[j] = "y";
                }else{
                    System.err.println("Unknown fragment ion type:"+fragment_ion_column_names[j]);
                    System.exit(1);
                }

                if(fragment_ion_column_names[j].endsWith("_z1")){
                    ion_charges[j] = 1;
                }else if(fragment_ion_column_names[j].endsWith("_z2")){
                    ion_charges[j] = 2;
                }else{
                    System.err.println("Unknown fragment ion charge:"+fragment_ion_column_names[j]);
                    System.exit(1);
                }

                if(fragment_ion_column_names[j].contains("modloss")) {
                    if(this.mod_ai.equalsIgnoreCase("phosphorylation")){
                        mod_losses[j] = "H3PO4";
                    }
                }else{
                    mod_losses[j] = "noloss";
                }
            }
            // RT information
            HashMap<Integer,Double> pepID2rt = FileIO.load_rt_data(rt_file,this.rt_max);
            // MS intensity
            ArrayList<double[]> ms2_intensity_lines = FileIO.load_matrix(ms2_intensity_file);
            // mz intensity
            ArrayList<double[]> ms2_mz_lines = FileIO.load_matrix(ms2_mz_file);
            // MS2 information
            String line;
            // ms2_file is a parquet file. read the data one row at a time
            // Create a configuration
            Configuration conf = new Configuration();
            LocalInputFile inputFile = new LocalInputFile(Paths.get(ms2_file));
            ParquetReader<GenericRecord> reader = AvroParquetReader.<GenericRecord>builder(inputFile).withConf(conf).build();
            GenericRecord record;
            while ((record = reader.read()) != null) {
                // get column "pepID"
                pepID = (int) record.get("pepID");
                frag_start_idx = ((Long)record.get("frag_start_idx")).intValue();
                frag_stop_idx = ((Long)record.get("frag_stop_idx")).intValue();
                sequence = record.get("sequence").toString();
                mods = record.get("mods").toString();
                mod_sites = record.get("mod_sites").toString();
                mz = (double) record.get("mz");
                charge = (int) record.get("charge");
                // decoy = group.getString("decoy",0);
                if(pep2pro.containsKey(sequence)){
                    protein = pep2pro.get(sequence);
                }else{
                    protein = "-";
                }
                libFragment.StrippedPeptide = sequence;
                libFragment.PrecursorMz = (float) mz;
                libFragment.PrecursorCharge = charge;
                libFragment.ProteinID = protein;
                libFragment.Decoy = 0;
                libFragment.Tr_recalibrated = pepID2rt.get(pepID).floatValue();
                ArrayList<LibFragment> lines = get_fragment_ion_intensity4parquet_all(ms2_mz_lines,
                        ms2_intensity_lines,
                        fragment_ion_column_names,
                        frag_start_idx,
                        frag_stop_idx,
                        this.lf_top_n_fragment_ions,
                        ion_types,
                        mod_losses,
                        ion_charges,
                        this.lf_frag_n_min);

                if(lines.size()<this.lf_min_n_fragment_ions){
                    continue;
                }
                // decoy_label = decoy.startsWith("Yes")?1:0;
                decoy_label = 0;
                rt_str = String.format("%.2f",pepID2rt.get(pepID));
                String mod_pep;
                if(export_diann_format){
                    mod_pep = get_modified_peptide_diann(sequence,mods,mod_sites);
                }else if(export_EncyclopeDIA_format){
                    mod_pep = get_modified_peptide_encyclopedia(sequence,mods,mod_sites);
                }else{
                    mod_pep = get_modified_peptide(sequence,mods,mod_sites);
                }
                libFragment.ModifiedPeptide = mod_pep;
                for(LibFragment l: lines) {
                    if(export_tsv) {
                        StringBuilder ob = new StringBuilder();
                        ob.append(mod_pep).append("\t")
                                .append(sequence).append("\t")
                                .append(mz).append("\t")
                                .append(charge).append("\t")
                                .append(rt_str).append("\t")
                                .append(protein).append("\t")
                                .append(decoy_label).append("\t")
                                // FragmentMz	RelativeIntensity	FragmentType	FragmentNumber	FragmentCharge	FragmentLossType
                                .append(l.FragmentMz).append("\t")
                                .append(String.format("%.4f",l.RelativeIntensity)).append("\t")
                                .append(l.FragmentType).append("\t")
                                .append(l.FragmentNumber).append("\t")
                                .append(l.FragmentCharge).append("\t")
                                .append(l.FragmentLossType).append("\n");
                        libWriter.write(ob.toString());
                    }else{
                        // write to parquet
                        // FragmentMz	RelativeIntensity	FragmentType	FragmentNumber	FragmentCharge	FragmentLossType
                        libFragment.FragmentMz = l.FragmentMz;
                        libFragment.RelativeIntensity = l.RelativeIntensity;
                        libFragment.FragmentType = l.FragmentType;
                        libFragment.FragmentNumber = l.FragmentNumber;
                        libFragment.FragmentCharge = l.FragmentCharge;
                        libFragment.FragmentLossType = l.FragmentLossType;
                        pWriter.write(libFragment);
                    }
                }
            }
            reader.close();
        }
        if(export_tsv) {
            libWriter.close();
        }else{
            pWriter.close();
        }
    }

    public void generate_spectral_library_parquet_skyline(Map<String,HashMap<String,String>> res_files, String out_folder, String file_name) throws IOException, SQLException {
        //String out_library_file = this.out_dir + File.separator + "SkylineAI_spectral_library.tsv";
        String out_library_file = out_folder + File.separator + file_name;
        if(out_library_file.endsWith("tsv")){
            out_library_file = out_library_file.replaceAll("tsv$", "blib");
        } else if (out_library_file.endsWith("txt")){
            out_library_file = out_library_file.replaceAll("txt$", "blib");
        } else if (out_library_file.endsWith("csv")){
            out_library_file = out_library_file.replaceAll("csv$", "blib");
        } else if (out_library_file.endsWith("parquet")){
            out_library_file = out_library_file.replaceAll("parquet$", "blib");
        }
        SkylineIO skylineIO = new SkylineIO(out_library_file);
        skylineIO.add_SpectrumSourceFiles();
        skylineIO.add_ScoreTypes();
        skylineIO.add_RefSpectraPeakAnnotations();
        skylineIO.create_RefSpectra();
        skylineIO.create_Modifications();
        skylineIO.create_RetentionTimes();
        skylineIO.pStatementRefSpectra.setNull(5, java.sql.Types.CHAR);
        skylineIO.pStatementRefSpectra.setNull(6, java.sql.Types.CHAR);
        skylineIO.pStatementRefSpectra.setNull(8, java.sql.Types.DOUBLE);
        skylineIO.pStatementRefSpectra.setNull(9, java.sql.Types.DOUBLE);
        skylineIO.pStatementRefSpectra.setNull(10, java.sql.Types.DOUBLE);
        skylineIO.pStatementRefSpectra.setNull(13, java.sql.Types.VARCHAR);
        skylineIO.create_RefSpectraPeaks();
        skylineIO.connection.setAutoCommit(false);

        DBGear dbGear = new DBGear();
        Map<String, String> pep2pro = new HashMap<>();
        if(CParameter.db.toLowerCase().endsWith(".fa") || CParameter.db.toLowerCase().endsWith(".fasta")) {
            pep2pro = dbGear.digest_protein(CParameter.db);
        }
        int pepID;
        String sequence;
        String mods;
        String mod_sites;
        double rt;
        String rt_str;
        double mz;
        int charge;
        String decoy;
        int decoy_label;
        String protein = "-";
        int frag_start_idx;
        int frag_stop_idx;
        LibFragment libFragment = new LibFragment();
        int RefSpectraID = 0;
        int i_batch = 0;
        for(String i : res_files.keySet()){
            Cloger.getInstance().logger.info(i);
            String ms2_file = res_files.get(i).get("ms2");
            if(!ms2_file.endsWith("parquet")){
                if(!get_column_name2index(ms2_file).containsKey("protein")){
                    if(CParameter.db.toLowerCase().endsWith(".fa") || CParameter.db.toLowerCase().endsWith(".fasta")) {
                        dbGear.add_protein_to_psm_table(ms2_file, CParameter.db);
                    }
                }
            }
            String ms2_intensity_file = res_files.get(i).get("ms2_intensity");
            String rt_file = res_files.get(i).get("rt");
            String ms2_mz_file = res_files.get(i).get("ms2_mz");

            // "_ms2_df.tsv"
            // pepID   sequence        charge  mods    mod_sites       nce     instrument      nAA     frag_start_idx  frag_stop_idx
            // "_ms2_mz_df.tsv"
            // b_z1    b_z2    y_z1    y_z2    b_modloss_z1    b_modloss_z2    y_modloss_z1    y_modloss_z2
            // "_ms2_pred.tsv"
            // b_z1    b_z2    y_z1    y_z2    b_modloss_z1    b_modloss_z2    y_modloss_z1    y_modloss_z2
            // "_rt_pred.tsv"
            // pepID   sequence        mods    mod_sites       nAA     rt_pred rt_norm_pred    irt_pred

            // HashMap<String,Integer> ms2_col2index = FileIO.get_column_name2index_from_head_line(ms2_file);
            // HashMap<String,Integer> ms2_intensity_col2index = FileIO.get_column_name2index_from_head_line(ms2_intensity_file);
            // HashMap<String,Integer> rt_col2index = FileIO.get_column_name2index_from_head_line(rt_file);
            String [] fragment_ion_column_names = FileIO.get_column_names_from_parquet(ms2_mz_file);
            //HashMap<String,Integer> ms2_mz_col2index = this.get_column_name2index_from_head_line(ms2mzReader.readLine().trim());

            String []ion_types = new String[fragment_ion_column_names.length];
            String []mod_losses = new String[fragment_ion_column_names.length];
            int []ion_charges = new int[fragment_ion_column_names.length];
            for(int j=0;j<fragment_ion_column_names.length;j++){
                if(fragment_ion_column_names[j].startsWith("b")){
                    ion_types[j] = "b";
                } else if (fragment_ion_column_names[j].startsWith("y")) {
                    ion_types[j] = "y";
                }else{
                    System.err.println("Unknown fragment ion type:"+fragment_ion_column_names[j]);
                    System.exit(1);
                }

                if(fragment_ion_column_names[j].endsWith("_z1")){
                    ion_charges[j] = 1;
                }else if(fragment_ion_column_names[j].endsWith("_z2")){
                    ion_charges[j] = 2;
                }else{
                    System.err.println("Unknown fragment ion charge:"+fragment_ion_column_names[j]);
                    System.exit(1);
                }

                if(fragment_ion_column_names[j].contains("modloss")) {
                    if(this.mod_ai.equalsIgnoreCase("phosphorylation")){
                        mod_losses[j] = "H3PO4";
                    }
                }else{
                    mod_losses[j] = "noloss";
                }
            }
            // RT information
            HashMap<Integer,Double> pepID2rt = FileIO.load_rt_data(rt_file,this.rt_max);
            // MS intensity
            ArrayList<double[]> ms2_intensity_lines = FileIO.load_matrix(ms2_intensity_file);
            // mz intensity
            ArrayList<double[]> ms2_mz_lines = FileIO.load_matrix(ms2_mz_file);
            // MS2 information
            String line;
            // ms2_file is a parquet file. read the data one row at a time
            // Create a configuration
            Configuration conf = new Configuration();
            LocalInputFile inputFile = new LocalInputFile(Paths.get(ms2_file));
            ParquetReader<GenericRecord> reader = AvroParquetReader.<GenericRecord>builder(inputFile).withConf(conf).build();
            GenericRecord record;
            while ((record = reader.read()) != null) {
                // get column "pepID"
                pepID = (int) record.get("pepID");
                frag_start_idx = ((Long)record.get("frag_start_idx")).intValue();
                frag_stop_idx = ((Long)record.get("frag_stop_idx")).intValue();
                sequence = record.get("sequence").toString();
                mods = record.get("mods").toString();
                mod_sites = record.get("mod_sites").toString();
                mz = (double) record.get("mz");
                charge = (int) record.get("charge");
                // decoy = group.getString("decoy",0);
                if(pep2pro.containsKey(sequence)){
                    pep2pro.get(sequence);
                }else {
                    protein = "-";
                }
                libFragment.StrippedPeptide = sequence;
                libFragment.PrecursorMz = (float) mz;
                libFragment.PrecursorCharge = charge;
                libFragment.ProteinID = protein;
                libFragment.Decoy = 0;
                libFragment.Tr_recalibrated = pepID2rt.get(pepID).floatValue();
                ArrayList<LibFragment> lines = get_fragment_ion_intensity4parquet_all(ms2_mz_lines,
                        ms2_intensity_lines,
                        fragment_ion_column_names,
                        frag_start_idx,
                        frag_stop_idx,
                        this.lf_top_n_fragment_ions,
                        ion_types,
                        mod_losses,
                        ion_charges,
                        this.lf_frag_n_min);

                if(lines.size()<this.lf_min_n_fragment_ions){
                    continue;
                }

                // decoy_label = decoy.startsWith("Yes")?1:0;
                decoy_label = 0;
                String mod_pep = get_modified_peptide_skyline(sequence,mods,mod_sites);
                libFragment.ModifiedPeptide = mod_pep;
                RefSpectraID++;
                i_batch++;
                try {
                    skylineIO.pStatementRefSpectra.setString(1, sequence); // peptideSeq VARCHAR(150)
                    skylineIO.pStatementRefSpectra.setDouble(2, mz); // precursorMZ REAL
                    skylineIO.pStatementRefSpectra.setInt(3, charge); // precursorCharge INTEGER
                    skylineIO.pStatementRefSpectra.setString(4, mod_pep); // peptideModSeq VARCHAR(200)
                    //skylineIO.pStatementRefSpectra.setString(5, ""); // prevAA CHAR(1)
                    //skylineIO.pStatementRefSpectra.setString(6, ""); // nextAA CHAR(1)
                    skylineIO.pStatementRefSpectra.setInt(7, lines.size()); // numPeaks INTEGER
                    // skylineIO.pStatementRefSpectra.setDouble(9, null); // ionMobility REAL
                    // skylineIO.pStatementRefSpectra.setDouble(10, null); // collisionalCrossSectionSqA REAL
                    // skylineIO.pStatementRefSpectra.setDouble(11, null); // ionMobilityHighEnergyOffset REAL
                    skylineIO.pStatementRefSpectra.setInt(11, 0); // ionMobilityType TINYINT
                    skylineIO.pStatementRefSpectra.setDouble(12, pepID2rt.get(pepID)); // retentionTime REAL
                    // skylineIO.pStatementRefSpectra.setInt(14, 1); // fileID INTEGER DEFAULT 1
                    // skylineIO.pStatementRefSpectra.setString(15, null); // SpecIDinFile VARCHAR(256)
                    // skylineIO.pStatementRefSpectra.setDouble(16, 0); // score REAL DEFAULT 0
                    // skylineIO.pStatementRefSpectra.setInt(17, 0); // scoreType TINYINT DEFAULT 0
                    skylineIO.pStatementRefSpectra.addBatch();
                }catch (SQLException e) {
                    System.out.println("Error inserting into RefSpectra: " + e.getMessage());
                }

                // save the mz and intensity values to two arrays
                double [] mz_values = new double[lines.size()];
                float [] intensity_values = new float[lines.size()];
                for(int k=0;k<mz_values.length;k++){
                    mz_values[k] = lines.get(k).FragmentMz;
                    intensity_values[k] = lines.get(k).RelativeIntensity;
                }
                skylineIO.pStatementRefSpectraPeaks.setInt(1, RefSpectraID); // RefSpectraID INTEGER
                skylineIO.pStatementRefSpectraPeaks.setBytes(2, SkylineIO.doublesToLittleEndianBytes(mz_values)); // mz BLOB
                skylineIO.pStatementRefSpectraPeaks.setBytes(3, SkylineIO.floatsToLittleEndianBytes(intensity_values)); // intensity BLOB
                skylineIO.pStatementRefSpectraPeaks.addBatch();

                // RT
                skylineIO.pStatementRetentionTimes.setInt(1, RefSpectraID); // RefSpectraID INTEGER
                skylineIO.pStatementRetentionTimes.setDouble(2, pepID2rt.get(pepID)); // mz BLOB
                skylineIO.pStatementRetentionTimes.addBatch();

                // modification
                if(!mods.isEmpty()){
                    String [] names = mods.split(";");
                    String [] pos = mod_sites.split(";");
                    for(int j=0;j<pos.length;j++) {
                        skylineIO.pStatementModifications.setInt(1, RefSpectraID); // RefSpectraID INTEGER
                        skylineIO.pStatementModifications.setInt(2, Integer.parseInt(pos[j])); // position INTEGER
                        skylineIO.pStatementModifications.setDouble(3, SkylineIO.mod2mass.get(names[j])); // mass REAL
                        skylineIO.pStatementModifications.addBatch();
                    }
                }
                if(i_batch == 10000){
                    skylineIO.pStatementRefSpectra.executeBatch();
                    skylineIO.pStatementRefSpectraPeaks.executeBatch();
                    skylineIO.pStatementModifications.executeBatch();
                    skylineIO.pStatementRetentionTimes.executeBatch();
                    i_batch = 0;
                }
            }
            reader.close();
        }
        if(i_batch >= 1){
            skylineIO.pStatementRefSpectra.executeBatch();
            skylineIO.pStatementRefSpectraPeaks.executeBatch();
            skylineIO.pStatementModifications.executeBatch();
            skylineIO.pStatementRetentionTimes.executeBatch();
            i_batch = 0;
        }
        skylineIO.numSpectra = RefSpectraID;
        skylineIO.add_LibInfo();
        skylineIO.add_index();
        skylineIO.connection.commit(); // Commit the transaction, making all changes permanent
        skylineIO.connection.setAutoCommit(true); // Re-enable auto-commit if needed
        skylineIO.close();
    }

    public void generate_spectral_library_parquet_mzSpecLib(Map<String,HashMap<String,String>> res_files, String out_folder, String file_name) throws IOException, SQLException {
        //String out_library_file = this.out_dir + File.separator + "SkylineAI_spectral_library.tsv";
        String out_library_file = out_folder + File.separator + file_name;
        if(out_library_file.endsWith("tsv")){
            out_library_file = out_library_file.replaceAll("tsv$", "mzlib.txt");
        } else if (out_library_file.endsWith("txt")){
            out_library_file = out_library_file.replaceAll("txt$", "mzlib.txt");
        } else if (out_library_file.endsWith("csv")){
            out_library_file = out_library_file.replaceAll("csv$", "mzlib.txt");
        } else if (out_library_file.endsWith("parquet")){
            out_library_file = out_library_file.replaceAll("parquet$", "mzlib.txt");
        }

        BufferedWriter mzLibWriter = new BufferedWriter(new FileWriter(out_library_file));
        mzLibWriter.write("<mzSpecLib 1.0>\n");
        mzLibWriter.write("MS:1003186|library format version=1.0\n");
        File F = new File(out_library_file);
        mzLibWriter.write("MS:1003188|library name="+F.getName()+"\n");
        mzLibWriter.write("MS:1003207|library creation software=Carafe\n");
        mzLibWriter.write("<AttributeSet Spectrum=all>\n");
        mzLibWriter.write("<AttributeSet Analyte=all>\n");
        mzLibWriter.write("<AttributeSet Interpretation=all>\n");

        DBGear dbGear = new DBGear();
        Map<String, String> pep2pro = new HashMap<>();
        if(CParameter.db.toLowerCase().endsWith(".fa") || CParameter.db.toLowerCase().endsWith(".fasta")) {
            pep2pro = dbGear.digest_protein(CParameter.db);
        }
        int pepID;
        String sequence;
        String mods;
        String mod_sites;
        double rt;
        String rt_str;
        double mz;
        int charge;
        String decoy;
        int decoy_label;
        String protein = "-";
        int frag_start_idx;
        int frag_stop_idx;
        LibFragment libFragment = new LibFragment();
        int RefSpectraID = 0;
        int i_batch = 0;
        StringBuilder stringBuilder = new StringBuilder();
        for(String i : res_files.keySet()){
            Cloger.getInstance().logger.info(i);
            String ms2_file = res_files.get(i).get("ms2");
            if(!ms2_file.endsWith("parquet")){
                if(!get_column_name2index(ms2_file).containsKey("protein")){
                    if(CParameter.db.toLowerCase().endsWith(".fa") || CParameter.db.toLowerCase().endsWith(".fasta")) {
                        dbGear.add_protein_to_psm_table(ms2_file, CParameter.db);
                    }
                }
            }
            String ms2_intensity_file = res_files.get(i).get("ms2_intensity");
            String rt_file = res_files.get(i).get("rt");
            String ms2_mz_file = res_files.get(i).get("ms2_mz");

            // "_ms2_df.tsv"
            // pepID   sequence        charge  mods    mod_sites       nce     instrument      nAA     frag_start_idx  frag_stop_idx
            // "_ms2_mz_df.tsv"
            // b_z1    b_z2    y_z1    y_z2    b_modloss_z1    b_modloss_z2    y_modloss_z1    y_modloss_z2
            // "_ms2_pred.tsv"
            // b_z1    b_z2    y_z1    y_z2    b_modloss_z1    b_modloss_z2    y_modloss_z1    y_modloss_z2
            // "_rt_pred.tsv"
            // pepID   sequence        mods    mod_sites       nAA     rt_pred rt_norm_pred    irt_pred

            // HashMap<String,Integer> ms2_col2index = FileIO.get_column_name2index_from_head_line(ms2_file);
            // HashMap<String,Integer> ms2_intensity_col2index = FileIO.get_column_name2index_from_head_line(ms2_intensity_file);
            // HashMap<String,Integer> rt_col2index = FileIO.get_column_name2index_from_head_line(rt_file);
            String [] fragment_ion_column_names = FileIO.get_column_names_from_parquet(ms2_mz_file);
            //HashMap<String,Integer> ms2_mz_col2index = this.get_column_name2index_from_head_line(ms2mzReader.readLine().trim());

            String []ion_types = new String[fragment_ion_column_names.length];
            String []mod_losses = new String[fragment_ion_column_names.length];
            int []ion_charges = new int[fragment_ion_column_names.length];
            for(int j=0;j<fragment_ion_column_names.length;j++){
                if(fragment_ion_column_names[j].startsWith("b")){
                    ion_types[j] = "b";
                } else if (fragment_ion_column_names[j].startsWith("y")) {
                    ion_types[j] = "y";
                }else{
                    System.err.println("Unknown fragment ion type:"+fragment_ion_column_names[j]);
                    System.exit(1);
                }

                if(fragment_ion_column_names[j].endsWith("_z1")){
                    ion_charges[j] = 1;
                }else if(fragment_ion_column_names[j].endsWith("_z2")){
                    ion_charges[j] = 2;
                }else{
                    System.err.println("Unknown fragment ion charge:"+fragment_ion_column_names[j]);
                    System.exit(1);
                }

                if(fragment_ion_column_names[j].contains("modloss")) {
                    if(this.mod_ai.equalsIgnoreCase("phosphorylation")){
                        mod_losses[j] = "H3PO4";
                    }
                }else{
                    mod_losses[j] = "noloss";
                }
            }
            // RT information
            HashMap<Integer,Double> pepID2rt = FileIO.load_rt_data(rt_file,this.rt_max);
            // MS intensity
            ArrayList<double[]> ms2_intensity_lines = FileIO.load_matrix(ms2_intensity_file);
            // mz intensity
            ArrayList<double[]> ms2_mz_lines = FileIO.load_matrix(ms2_mz_file);
            // MS2 information
            String line;
            // ms2_file is a parquet file. read the data one row at a time
            // Create a configuration
            Configuration conf = new Configuration();
            LocalInputFile inputFile = new LocalInputFile(Paths.get(ms2_file));
            ParquetReader<GenericRecord> reader = AvroParquetReader.<GenericRecord>builder(inputFile).withConf(conf).build();
            GenericRecord record;
            while ((record = reader.read()) != null) {
                // get column "pepID"
                pepID = (int) record.get("pepID");
                frag_start_idx = ((Long)record.get("frag_start_idx")).intValue();
                frag_stop_idx = ((Long)record.get("frag_stop_idx")).intValue();
                sequence = record.get("sequence").toString();
                mods = record.get("mods").toString();
                mod_sites = record.get("mod_sites").toString();
                mz = (double) record.get("mz");
                charge = (int) record.get("charge");
                // decoy = group.getString("decoy",0);
                if(pep2pro.containsKey(sequence)){
                    pep2pro.get(sequence);
                }else {
                    protein = "-";
                }
                libFragment.StrippedPeptide = sequence;
                libFragment.PrecursorMz = (float) mz;
                libFragment.PrecursorCharge = charge;
                libFragment.ProteinID = protein;
                libFragment.Decoy = 0;
                libFragment.Tr_recalibrated = pepID2rt.get(pepID).floatValue();
                ArrayList<LibFragment> lines = get_fragment_ion_intensity4parquet_all(ms2_mz_lines,
                        ms2_intensity_lines,
                        fragment_ion_column_names,
                        frag_start_idx,
                        frag_stop_idx,
                        this.lf_top_n_fragment_ions,
                        ion_types,
                        mod_losses,
                        ion_charges,
                        this.lf_frag_n_min);

                if(lines.size()<this.lf_min_n_fragment_ions){
                    continue;
                }

                // decoy_label = decoy.startsWith("Yes")?1:0;
                decoy_label = 0;
                String mod_pep = get_modified_peptide_skyline(sequence,mods,mod_sites);
                libFragment.ModifiedPeptide = mod_pep;
                RefSpectraID++;
                stringBuilder.setLength(0);
                stringBuilder.append("<Spectrum=").append(RefSpectraID).append(">\n");
                stringBuilder.append("MS:1003061|library spectrum name=").append(mod_pep).append(charge).append("\n");
                stringBuilder.append("MS:1003053|theoretical monoisotopic m/z=").append(mz).append("\n");
                stringBuilder.append("MS:1003072|spectrum origin type=MS:1003074|predicted spectrum\n");
                stringBuilder.append("[1]MS:1000894|retention time=").append(pepID2rt.get(pepID)).append("\n");
                stringBuilder.append("[1]UO:0000000|unit=UO:0000031|minute\n");
                stringBuilder.append("MS:1003059|number of peaks=").append(lines.size()).append("\n");
                stringBuilder.append("<Analyte=1>\n");
                stringBuilder.append("MS:1000888|stripped peptide sequence=").append(sequence).append("\n");
                stringBuilder.append("MS:1000041|charge state=").append(charge).append("\n");
                stringBuilder.append("<Peaks>\n");
                for(int k=0;k<lines.size();k++){
                    stringBuilder.append(lines.get(k).FragmentMz).append("\t").append(lines.get(k).RelativeIntensity).append("\t");
                    stringBuilder.append(lines.get(k).FragmentType).append(lines.get(k).FragmentNumber);
                    if(!lines.get(k).FragmentLossType.equals("noloss")){
                        stringBuilder.append("-").append(lines.get(k).FragmentLossType);
                    }
                    if(lines.get(k).FragmentCharge>1){
                        stringBuilder.append("^").append(lines.get(k).FragmentCharge);
                    }
                    stringBuilder.append("/0.0\n");
                }
                mzLibWriter.write(stringBuilder.toString()+"\n");
            }
            reader.close();
        }
        mzLibWriter.close();
    }

    String get_modified_peptide(String peptide, String mods, String mod_sites){
        if(mods.isEmpty()){
            return "_"+peptide+"_";
        }else{
            String [] names = mods.split(";");
            String [] pos = mod_sites.split(";");
            String [] aa = peptide.split("");
            for(int i=0;i<names.length;i++) {
                String ptm_name = "";
                int ptm_pos = Integer.parseInt(pos[i]) -1;
                switch (names[i]) {
                    case "Oxidation@M":
                        ptm_name = "M[Oxidation]";
                        aa[ptm_pos] = ptm_name;
                        break;
                    case "Carbamidomethyl@C":
                        ptm_name = "C[Carbamidomethyl]";
                        aa[ptm_pos] = ptm_name;
                        break;
                    case "Phospho@S":
                        ptm_name = "S[Phospho]";
                        aa[ptm_pos] = ptm_name;
                        break;
                    case "Phospho@T":
                        ptm_name = "T[Phospho]";
                        aa[ptm_pos] = ptm_name;
                        break;
                    case "Phospho@Y":
                        ptm_name = "Y[Phospho]";
                        aa[ptm_pos] = ptm_name;
                        break;
                    default:
                        System.out.println("Modification " + names[i] + " is not supported yet!");
                        System.exit(1);
                }
            }
            return "_"+StringUtils.join(aa,"")+"_";
        }

    }


    String get_modified_peptide_diann(String peptide, String mods, String mod_sites){
        if(mods.isEmpty()){
            return "_"+peptide+"_";
        }else{
            String [] names = mods.split(";");
            String [] pos = mod_sites.split(";");
            String [] aa = peptide.split("");
            for(int i=0;i<names.length;i++) {
                String ptm_name = "";
                int ptm_pos = Integer.parseInt(pos[i]) -1;
                switch (names[i]) {
                    case "Oxidation@M":
                        ptm_name = "M[UniMod:35]";
                        aa[ptm_pos] = ptm_name;
                        break;
                    case "Carbamidomethyl@C":
                        ptm_name = "C[UniMod:4]";
                        aa[ptm_pos] = ptm_name;
                        break;
                    case "Phospho@S":
                        ptm_name = "S[UniMod:21]";
                        aa[ptm_pos] = ptm_name;
                        break;
                    case "Phospho@T":
                        ptm_name = "T[UniMod:21]";
                        aa[ptm_pos] = ptm_name;
                        break;
                    case "Phospho@Y":
                        ptm_name = "Y[UniMod:21]";
                        aa[ptm_pos] = ptm_name;
                        break;
                    default:
                        System.out.println("Modification " + names[i] + " is not supported yet!");
                        System.exit(1);
                }
            }
            return "_"+StringUtils.join(aa,"")+"_";
        }

    }


    String get_modified_peptide_encyclopedia(String peptide, String mods, String mod_sites){
        if(mods.isEmpty()){
            return "_"+peptide+"_";
        }else{
            String [] names = mods.split(";");
            String [] pos = mod_sites.split(";");
            String [] aa = peptide.split("");
            for(int i=0;i<names.length;i++) {
                String ptm_name = "";
                int ptm_pos = Integer.parseInt(pos[i]) -1;
                switch (names[i]) {
                    case "Oxidation@M":
                        ptm_name = "M[Oxidation (M)]";
                        aa[ptm_pos] = ptm_name;
                        break;
                    case "Carbamidomethyl@C":
                        ptm_name = "C[Carbamidomethyl (C)]";
                        aa[ptm_pos] = ptm_name;
                        break;
                    case "Phospho@S":
                        ptm_name = "S[Phosphorylation (ST)]";
                        aa[ptm_pos] = ptm_name;
                        break;
                    case "Phospho@T":
                        ptm_name = "T[Phosphorylation (ST)]";
                        aa[ptm_pos] = ptm_name;
                        break;
                    case "Phospho@Y":
                        ptm_name = "Y[Phosphorylation (Y)]";
                        aa[ptm_pos] = ptm_name;
                        break;
                    default:
                        System.out.println("Modification " + names[i] + " is not supported yet!");
                        System.exit(1);
                }
            }
            return "_"+StringUtils.join(aa,"")+"_";
        }

    }

    String get_modified_peptide_skyline(String peptide, String mods, String mod_sites){
        if(mods.isEmpty()){
            return peptide;
        }else{
            String [] names = mods.split(";");
            String [] pos = mod_sites.split(";");
            String [] aa = peptide.split("");
            for(int i=0;i<names.length;i++) {
                String ptm_name = "";
                int ptm_pos = Integer.parseInt(pos[i]) -1;
                switch (names[i]) {
                    case "Oxidation@M":
                        ptm_name = "M[+15.994915]";
                        aa[ptm_pos] = ptm_name;
                        break;
                    case "Carbamidomethyl@C":
                        ptm_name = "C[57.021464]";
                        aa[ptm_pos] = ptm_name;
                        break;
                    case "Phospho@S":
                        ptm_name = "S[+79.966331]";
                        aa[ptm_pos] = ptm_name;
                        break;
                    case "Phospho@T":
                        ptm_name = "T[+79.966331]";
                        aa[ptm_pos] = ptm_name;
                        break;
                    case "Phospho@Y":
                        ptm_name = "Y[+79.966331]";
                        aa[ptm_pos] = ptm_name;
                        break;
                    default:
                        System.out.println("Modification " + names[i] + " is not supported yet!");
                        System.exit(1);
                }
            }
            return StringUtils.join(aa,"");
        }

    }

    private void print_parameters(String cmd) throws IOException {
        String itol_unit;
        if(CParameter.itolu.equalsIgnoreCase("ppm")){
            itol_unit = "ppm";
        }else{
            itol_unit = "Da";
        }

        System.out.println("#############################################");
        System.out.println("Parameter:");

        StringBuilder sBuilder = new StringBuilder();
        sBuilder.append("Version: ").append(CParameter.getVersion()).append("\n");
        if(!cmd.equalsIgnoreCase("-")){
            sBuilder.append("Command line: ").append(cmd).append("\n");
        }
        sBuilder.append("CPU: ").append(CParameter.cpu).append("\n");
        if(cmd.contains(" -i ")) {
            // related to training data generation
            sBuilder.append("## Parameters related to training data generation:\n");
            sBuilder.append("FDR threshold: ").append(fdr_cutoff).append("\n");
            if (!mod_ai.equalsIgnoreCase("general")) {
                sBuilder.append("PTM site probability cutoff: ").append(ptm_site_prob_cutoff).append("\n");
            }
            // sBuilder.append("Precursor mass tolerance: ").append(CParameter.tol).append("\n");
            // sBuilder.append("Precursor ion mass tolerance unit: ").append(tol_unit).append("\n");
            sBuilder.append("Fragment ion mass tolerance: ").append(CParameter.itol).append("\n");
            sBuilder.append("Fragment ion mass tolerance unit: ").append(itol_unit).append("\n");
            sBuilder.append("Refine peak detection: ").append(refine_peak_boundary).append("\n");
            sBuilder.append("The number of flanking spectra to consider: ").append(n_flank_scans).append("\n");
            if (this.refine_peak_boundary) {
                sBuilder.append("RT window: ").append(CParameter.rt_win).append("\n");
                sBuilder.append("RT window unit: ").append("minute").append("\n");
            }
            sBuilder.append("RT window offset: ").append(rt_win_offset).append("\n");
            sBuilder.append("RT window offset unit: ").append("minute").append("\n");
            sBuilder.append("Data points used for XIC smoothing: ").append(sg_smoothing_data_points).append("\n");
            sBuilder.append("Fragment ion intensity normalization: ").append(fragment_ion_intensity_normalization).append("\n");
            sBuilder.append("Export valid PSM only: ").append(export_valid_matches_only).append("\n");
            sBuilder.append("Export Skyline transition list file for visualization: ").append(export_skyline_transition_list_file).append("\n");
            sBuilder.append("Valid min fragment ion m/z: ").append(min_fragment_ion_mz).append("\n");
            sBuilder.append("Valid max fragment ion m/z: ").append(max_fragment_ion_mz).append("\n");
            sBuilder.append("Minimum C-terminal ion number to consider: ").append(c_ion_min).append("\n");
            sBuilder.append("Minimum N-terminal ion number to consider: ").append(n_ion_min).append("\n");
            sBuilder.append("Remove y1 ion: ").append(remove_y1).append("\n");
            sBuilder.append("Peak cor cutoff: ").append(cor_cutoff).append("\n");
            sBuilder.append("The minimum number of matched fragment ions with high correlation to consider: ").append(min_n_high_quality_fragment_ions).append("\n");
            sBuilder.append("The minimum number of matched fragment ions to consider: ").append(min_n_fragment_ions).append("\n");
            sBuilder.append("Search engine used to generate the input training data: ").append(search_engine).append("\n");

            sBuilder.append("## Parameters related to model training:\n");
            sBuilder.append("Model type: ").append(mod_ai).append("\n");
            sBuilder.append("Device: ").append(device).append("\n");
            sBuilder.append("Use user provided MS instrument type: ").append(use_user_provided_ms_instrument).append("\n");
            if (use_user_provided_ms_instrument) {
                sBuilder.append("User provided MS instrument type: ").append(user_provided_ms_instrument).append("\n");
            }
            sBuilder.append("MS instrument type: ").append(this.ms_instrument).append("\n");
            sBuilder.append("NCE: ").append(this.nce).append("\n");
            sBuilder.append("Export XIC data: ").append(export_xic).append("\n");
            sBuilder.append("Export spectra data in MGF format: ").append(export_spectra_to_mgf).append("\n");
            sBuilder.append("Random seed: ").append(global_random_seed).append("\n");
        }
        // related to library generation
        sBuilder.append("## Parameters related to spectral library generation:\n");
        sBuilder.append("Protein database: ").append(db).append("\n");
        sBuilder.append("Convert I to L: ").append(I2L).append("\n");
        sBuilder.append("Fixed modification: ").append(CParameter.fixMods).append(" = ").append(ModificationUtils.getInstance().getModificationString(CParameter.fixMods)).append("\n");
        sBuilder.append("Variable modification: ").append(CParameter.varMods).append(" = ").append(ModificationUtils.getInstance().getModificationString(CParameter.varMods)).append("\n");
        sBuilder.append("Max allowed variable modification: ").append(CParameter.maxVarMods).append("\n");
        sBuilder.append("Enzyme: ").append(CParameter.enzyme).append(" = ").append(DBGear.getEnzymeByIndex(CParameter.enzyme).getName()).append("\n");
        sBuilder.append("Max missed cleavages: ").append(CParameter.maxMissedCleavages).append("\n");
        sBuilder.append("Clip protein N-terminal methionine: ").append(CParameter.clip_nTerm_M).append("\n");
        sBuilder.append("Library min precursor charge: ").append(lf_precursor_charge_min).append("\n");
        sBuilder.append("Library max precursor charge: ").append(lf_precursor_charge_max).append("\n");
        sBuilder.append("Library precursor charges: ").append(StringUtils.join(this.precursor_charges,',')).append("\n");
        sBuilder.append("Library min peptide length: ").append(CParameter.minPeptideLength).append("\n");
        sBuilder.append("Library max peptide length: ").append(CParameter.maxPeptideLength).append("\n");
        // this could be changed based on the information from the training DIA file
        sBuilder.append("Library min peptide m/z: ").append(CParameter.minPeptideMz).append("\n");
        sBuilder.append("Library max peptide m/z: ").append(CParameter.maxPeptideMz).append("\n");
        sBuilder.append("Library min fragment ion m/z: ").append(lf_frag_mz_min).append("\n");
        sBuilder.append("Library max fragment ion m/z: ").append(lf_frag_mz_max).append("\n");
        sBuilder.append("Library top N fragments: ").append(lf_top_n_fragment_ions).append("\n");
        sBuilder.append("Library minimum fragment number: ").append(lf_frag_n_min).append("\n");
        sBuilder.append("Library minimum fragments: ").append(lf_min_n_fragment_ions).append("\n");
        sBuilder.append("Library file format: ").append(export_spectral_library_file_format).append("\n");
        sBuilder.append("Library data format: ").append(export_spectral_library_format).append("\n");
        sBuilder.append("Training type: ").append(CParameter.tf_type).append("\n");
        sBuilder.append("Output folder: ").append(out_dir).append("\n");

        System.out.print(sBuilder.toString());
        System.out.println("#############################################");
        // Save parameters and command line information to a file
        String para_file = this.out_dir + "/parameter.txt";
        BufferedWriter bWriter = new BufferedWriter(new FileWriter(para_file));
        bWriter.write(sBuilder.toString());
        bWriter.close();
    }

}
