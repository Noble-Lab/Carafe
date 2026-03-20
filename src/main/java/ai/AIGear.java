package main.java.ai;

import ai.djl.Device;
import ai.djl.util.cuda.CudaUtils;
import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONWriter;
import com.alibaba.fastjson2.TypeReference;
import com.compomics.util.experiment.biology.ions.Ion;
import com.compomics.util.experiment.biology.ions.NeutralLoss;
import com.compomics.util.experiment.biology.ions.impl.ElementaryIon;
import com.compomics.util.experiment.biology.ions.impl.PeptideFragmentIon;
import com.compomics.util.experiment.biology.modifications.Modification;
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
import com.compomics.util.experiment.io.mass_spectrometry.mgf.MgfFileIterator;
import com.compomics.util.experiment.mass_spectrometry.spectra.Precursor;
import com.compomics.util.experiment.mass_spectrometry.spectra.Spectrum;
import com.compomics.util.gui.waiting.waitinghandlers.WaitingHandlerCLIImpl;
import com.compomics.util.parameters.identification.advanced.SequenceMatchingParameters;
import com.compomics.util.parameters.identification.search.ModificationParameters;
import com.compomics.util.waiting.WaitingHandler;
import com.google.common.base.Splitter;
import com.google.common.math.Quantiles;
import com.jerolba.carpet.CarpetReader;

import main.java.db.DBGear;
import main.java.dia.*;
import main.java.input.*;
import main.java.util.*;
import main.java.xic.SGFilter;
import main.java.xic.SGFilter3points;
import main.java.xic.SGFilter5points;
import main.java.xic.SGFilter7points;
import net.tlabs.tablesaw.parquet.TablesawParquetReader;
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
import tech.tablesaw.aggregate.AggregateFunctions;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.IntColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.io.Source;
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
import java.sql.Types;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.IntStream;
import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toMap;

/**
 * AIGear is the main class of Carafe for spectral library generation using AI
 * models.
 *
 * @author Bo Wen
 */
public class AIGear {

    /**
     * FDR threshold used to filter peptide detection result for model training.
     * Default is 1%.
     */
    public double fdr_cutoff = 0.01;

    /**
     * The number of flank scans to consider when generating spectrum prediction
     * training data
     * The default value is 0, which means that only the apex scan is considered.
     */
    public int n_flank_scans = 0;

    /**
     * A HashSet object to store m/z isolation windows data.
     */
    public HashSet<String> target_isolation_wins = new HashSet<>();

    /**
     * The NCE setting used for fragment ion intensity model training/prediction.
     * This value could be automatically extracted from the MS file when it is
     * available in the MS file.
     */
    public double nce = 27.0;

    /**
     * This is used to indicate whether the user has provided a specific MS
     * instrument for spectral library generation.
     * If true, the user_provided_ms_instrument will be used. Otherwise, the
     * instrument information extracted from
     * the MS file used for model training or the default ms_instrument setting will
     * be used.
     */
    public boolean use_user_provided_ms_instrument = false;

    /**
     * User provided MS instrument setting for fragment ion intensity model
     * training/prediction.
     */
    public String user_provided_ms_instrument = "";

    /**
     * MS instrument setting for fragment ion intensity model training/prediction.
     */
    public String ms_instrument = "Eclipse";

    /**
     * Device setting for model training/prediction. It's on GPU by default. If GPU
     * is not found, cpu will be used.
     */
    public String device = "gpu";

    /**
     * The maximum fragment ion charge state to consider. It is 2+ in default.
     */
    public int max_fragment_ion_charge = 2;

    /**
     * This is used to indicate whether to convert I to L when generating spectral
     * library. Default is false.
     */
    public boolean I2L = false;

    /**
     * This is used to store all peptide matches data. Keys are ms file names,
     * values are peptide matches.
     */
    HashMap<String, ArrayList<String>> ms_file2psm = new HashMap<>();

    /**
     * Column name to index for PSM file.
     */
    HashMap<String, Integer> hIndex = new HashMap<>();

    /**
     * Peptide sequence + modification -> Peptide object
     */
    private ConcurrentHashMap<String, Peptide> peptide_mod2Peptide = new ConcurrentHashMap<>();

    /**
     * The number of times that a fragment ion in a spectrum assigned to peptides.
     * scan -> fragment ion mz -> peptide count
     */
    private ConcurrentHashMap<Integer, ConcurrentHashMap<Double, Integer>> scan2mz2count = new ConcurrentHashMap<>();

    /**
     * Output directory. It's the current directory (i.e., "./") by default.
     */
    public String out_dir = "./";

    /**
     * Fragment ion intensity threshold. Any fragment ion with intensity below this
     * value will be ignored when indexing MS data.
     */
    private final double fragment_ion_intensity_threshold = 0.00;

    /**
     * A HashMap object to store the mapping between ion type and its column index
     * in the fragment ion data file.
     */
    private HashMap<String, Integer> ion_type2column_index = new HashMap<>();

    /**
     * This is used to indicate whether neutral loss of water (H2O) and ammonia
     * (NH3) should be considered when annotating fragment ions.
     * Default is false, meaning that neutral loss of water and ammonia will not be
     * considered.
     */
    private boolean lossWaterNH3 = false;

    /**
     * This is used to indicate whether precursor ion should be added when
     * generating fragment ion intensity model training data.
     * Default is false, meaning that precursor ion will not be considered.
     */
    private boolean add_precursor_ion = true;

    /**
     * Fragmentation method, it is HCD by default (i.e., b and y ions will be
     * considered).
     */
    private final String fragmentation_method = "hcd";

    /**
     * The first row or head line of the PSM file used for model training.
     */
    private String psm_head_line = "-";

    /**
     * The first row or head line of the fragment ion intensity file.
     */
    private String fragment_ion_intensity_head_line = "-";

    /**
     * The number of data points used for smoothing XIC. Default is 3.
     */
    public int sg_smoothing_data_points = 3;

    /**
     * The RT window offset used for XIC extraction. Default is 1 minute.
     */
    private double rt_win_offset = 1.0;

    /**
     * This is used to indicate whether fragment ion intensity normalization should
     * be performed.
     */
    public boolean fragment_ion_intensity_normalization = false;

    /**
     * The minimum number of fragment ions to consider when generating fragment ion
     * intensity model training data.
     * It is 4 in default.
     */
    public int min_n_fragment_ions = 4;

    /**
     * If it is true, only export valid matches.
     */
    public boolean export_valid_matches_only = false;

    /**
     * If it is true, fragment ion charge will be less than precursor charge. It is
     * false by default.
     */
    public boolean fragment_ion_charge_less_than_precursor_charge = false;

    /**
     * If it is true, fragment ion m/z will be exported to a file. It is false by
     * default.
     */
    public boolean export_fragment_ion_mz_to_file = false;

    /**
     * A HashMap object stores modification mapping information.
     */
    HashMap<String, String> mod_map = new HashMap<>();

    /**
     * If it is true, export skyline transition list file. It is false by default.
     */
    public boolean export_skyline_transition_list_file = false;

    /**
     * Any fragment ion with mz <= 200 (in default) will not be considered as valid
     * when generating training data
     * for fragment ion intensity model fine-tuning. This could be updated by the
     * fragment ion m/z information
     * extracted from the MS file used for model training.
     */
    public double min_fragment_ion_mz = 200.0;

    /**
     * The maximum m/z of fragment ions to consider as valid. Default is 2000.0.
     */
    public double max_fragment_ion_mz = 2000.0;

    /**
     * The minimum m/z of fragment ions to consider when generating spectral
     * library.
     */
    public double lf_frag_mz_min = 200.0;

    /**
     * The maximum m/z of fragment ions to consider when generating spectral
     * library.
     */
    public double lf_frag_mz_max = 1800.0;

    /**
     * The minimum precursor charge state to consider when generating spectral
     * library.
     */
    public int lf_precursor_charge_min = 2;

    /**
     * The maximum precursor charge state to consider when generating spectral
     * library.
     */
    public int lf_precursor_charge_max = 4;

    /**
     * The maximum number of fragment ions to consider when generating spectral
     * library.
     */
    public int lf_top_n_fragment_ions = 20;

    /**
     * The minimum number of fragment ions to consider when generating spectral
     * library.
     */
    public int lf_min_n_fragment_ions = 2;

    /**
     * The minimum fragment ion number (b2,b3,b4, ...) to consider when generating
     * spectral library.
     * This is different from "lf_min_n_fragment_ions".
     */
    public int lf_frag_n_min = 2;

    /**
     * If it is true, refine peak boundary when generating training data for
     * fragment ion intensity model training.
     */
    public boolean refine_peak_boundary = false;

    /**
     * If it is true, y1 ion will not be considered as valid in fragment ion
     * intensity model training.
     */
    public boolean remove_y1 = false;

    /**
     * The minimum number of high quality fragment ions to consider. Default is 4.
     */
    public int min_n_high_quality_fragment_ions = 4;

    /**
     * If it is true, don't do any filtering and use all peaks and all detected
     * peptides.
     */
    private boolean use_all_peaks = false;

    /**
     * The minimum correlation cutoff to consider. Default is 0.75.
     */
    public double cor_cutoff = 0.75;

    /**
     * The minimum RT.
     */
    private double rt_min = 0.0;

    /**
     * The maximum RT.
     */
    double rt_max = 0.0;

    /**
     * The RT aggregation method there are multiple RT values for the same peptide
     * precursor.
     */
    String rt_merge_method = "min";

    /**
     * For the same peptide precursor, keep the best MS2 spectrum match (in default)
     * when there are multiple MS runs.
     */
    private String ms2_merge_method = "best";

    /**
     * The protein database used for spectral library generation.
     */
    public String db = "";

    /**
     * The path of python executable to use for model training and prediction.
     */
    private String python_bin = "python";
    /**
     * The version of ai scripts to use: v1 or v2 (default)
     */
    public String ai_version = "v2";

    /**
     * Use torch.compile to speed up training and inference
     */
    public boolean torch_compile = false;

    /**
     * The number of peptides to predict in each batch.
     */
    private int n_peptides_per_batch = 200000;

    /**
     * The precursor charge states to consider for spectral library generation
     */
    private int[] precursor_charges = new int[] { 2, 3, 4 };

    /**
     * The mz tol of fragment ions
     */
    private ArrayList<Double> fragment_ions_mz_tol = new ArrayList<>();

    /**
     * The search engine used to generate the identification result.
     */
    public String search_engine = "-";

    /**
     * The data type for model training: DDA (dda) or DIA (dia). Default is DIA.
     */
    public String data_type = "DIA";

    /**
     * enable peak masking or not: default is true
     */
    public boolean no_masking = false;

    /**
     * For modification peptide modeling or not:
     * "-" general peptide modeling
     * "phosphorylation" phosphorylation peptide modeling
     */
    public String mod_ai = "-";

    /**
     * Train a CCS model or not
     */
    public boolean ccs_enabled = false;
    /**
     * CCS merge method: mean (default), min or max
     */
    public String ccs_merge_method = "mean";

    /**
     * Use a fixed CE or NCE for all peptides during model training and prediction.
     * Default is true.
     */
    public boolean use_fixed_ce = true;

    /**
     * CCS DIA meta information
     */
    private CCSDIAMeta ccs_dia_meta = new CCSDIAMeta();

    /**
     * The format of spectral library: DIA-NN, EncyclopeDIA, Skyline (blib) or
     * mzSpecLib
     */
    public String export_spectral_library_format = "DIA-NN";

    /**
     * The maximum m/z of isolation window. Any MS2 scans with precursor m/z win
     * larger than this value will be ignored.
     */
    public double isolation_win_mz_max = -1;

    /**
     * If it is true, export XIC data to a file.
     */
    private boolean export_xic = false;

    /**
     * PTM site probability cutoff. Default is 0.75. This is only used during PTM
     * model training data generation
     */
    public double ptm_site_prob_cutoff = 0.75;

    /**
     * PTM q-value cutoff. This is only used during PTM model training data
     * generation.
     */
    public double ptm_site_qvalue_cutoff = 1.0;

    /**
     * For n-terminal fragment ions (such as b-ion) with number <= n_ion_min, they
     * will be considered as invalid.
     */
    public int n_ion_min = 0;

    /**
     * For c-terminal fragment ions (such as y-ion) with number <= n_ion_min, they
     * will be considered as invalid.
     */
    public int c_ion_min = 0;

    /**
     * A global random seed used for reproducibility.
     */
    private int global_random_seed = 2024;

    /**
     * Testing mode or not. If it is true, extra information will be exported to
     * files for evaluation.
     */
    private static boolean test_mode = false;

    /**
     * Use parquet format for saving data or not.
     */
    public boolean use_parquet = false;

    /**
     * The folder of the model to use for spectral library generation
     */
    public String model_dir = "";

    /**
     * User-provided pretrained MS2 model path
     */
    public String ms2_model = "";

    /**
     * The exported file format of spectral library: tsv or parquet
     */
    public String export_spectral_library_file_format = "tsv";

    /**
     * Export spectra to a mgf file or not. If it is true, all MS2 spectra matched
     * to precursors used for model training
     * data generation will be exported to an MGF file.
     */
    public boolean export_spectra_to_mgf = false;

    /**
     * Column separator
     */
    private final Splitter tab_splitter = Splitter.on('\t');

    /**
     * An ModificationParameters object used for spectra annotation during model
     * training data generation.
     */
    private static final ModificationParameters modificationParameters = new ModificationParameters();

    /**
     * An SequenceMatchingParameters object used for spectra annotation during model
     * training data generation.
     */
    private static final SequenceMatchingParameters sequenceMatchingParameters = new SequenceMatchingParameters();

    /**
     * An SequenceProvider object used for spectra annotation during model training
     * data generation.
     */
    private static final SequenceProvider sequenceProvider = new SingleProteinSequenceProvider();

    /**
     * The main function of Carafe
     * 
     * @param args Command line arguments
     * @throws ParseException If a parse exception occurs
     * @throws IOException    If an I/O error occurs
     */
    public static void main(String[] args) throws ParseException, IOException {
        long startTime = System.currentTimeMillis();
        Options options = new Options();
        options.addOption("i", true, "Peptide detection file from DIA-NN (e.g., report.tsv or report.parquet) or Skyline");
        options.addOption("ms", true, "MS file in mzML format: a single mzML or a folder containing mzML files.");
        options.addOption("fixMod", true, "Fixed modification, the format is like : 1,2,3. Use '-printPTM' to show all supported modifications. Default is 1 (Carbamidomethylation(C)[57.02]). " +
                "If there is no fixed modification, set it as '-fixMod no' or '-fixMod 0'.");
        options.addOption("varMod",true,"Variable modification, the format is the same with -fixMod. Default is 2 (Oxidation(M)[15.99]). "+
                "If there is no variable modification, set it as '-varMod no' or '-varMod 0'.");
        options.addOption("maxVar",true,"Max number of variable modifications, default is 1");
        options.addOption("printPTM",false,"Print all supported PTMs");
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
        options.addOption("ptm_site_qvalue", true, "The threshold of PTM site qvalue, default is 1 (no filtering)");
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
        options.addOption("lf_frag_mz_max", true, "The maximum mz of fragment to consider for library generation, default is 1800");
        options.addOption("lf_top_n_frag", true, "The maximum number of fragment ions to consider for library generation, default is 20");
        options.addOption("lf_min_n_frag", true, "The minimum number of fragment ions to consider for library generation, default is 2");
        options.addOption("lf_frag_n_min", true, "The minimum fragment ion number to consider for library generation, default is 2");
        options.addOption("rf", false, "Refine peak boundary or not");
        options.addOption("rf_rt_win", true, "RT window for refine peak boundary, default is 3 minutes");
        options.addOption("rt_win_offset", true, "RT window offset for XIC extraction, default is 1 minute");
        options.addOption("rt_max", true, "The max RT, default is 0.0, meaning using the max RT from the input MS file");
        options.addOption("xic", false, "Export XIC to file or not");
        options.addOption("export_mgf", false, "Export spectra to a mgf file or not");
        options.addOption("data_type", true, "DDA or DIA (default)");
        options.addOption("no_masking", false, "No peak masking");

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
        options.addOption("mod2mass",true,"Change the mass of a modification. The format is like: 2@0");
        // Format: mod_name@aa[mass][composition];mod_name@aa[mass][composition]
        // These modifications will be treated as variable modifications and add to the analysis
        options.addOption("user_var_mods",true,"User defined variable modifications");

        options.addOption("ccs", false, "CCS training");

        // This is the fine-tuned model directory
        options.addOption("model_dir", true, "The directory of the model to use for spectral library generation");
        options.addOption("ms2_model", true, "User-provided pretrained MS2 model path");

        options.addOption("verbose", true, "The level of detail of the log: 1 (info, default), 2 (debug)");
        options.addOption("ai_version", true, "The version of AI scripts to use: v1, v2 (default)");
        options.addOption("torch_compile", false, "Use torch.compile to speed up training and inference");

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

        if (cmd.hasOption("verbose")) {
            int log_level = Integer.parseInt(cmd.getOptionValue("verbose"));
            if (log_level == 1) {
                CParameter.verbose = CParameter.VerboseType.INFO;
            } else if (log_level == 2) {
                CParameter.verbose = CParameter.VerboseType.DEBUG;
            }
        }

        // Configure third-party loggers based on verbose level
        CParameter.configureThirdPartyLoggers();

        if (cmd.hasOption("user_var_mods")) {
            CParameter.user_var_mods = cmd.getOptionValue("user_var_mods");
        }

        // Print modification list
        if (cmd.hasOption("printPTM")) {
            ModificationUtils.save_mod2file = false;
            CModification.getInstance().printPTM();
            return;
        }

        if (cmd.hasOption("fixMod")) {
            CParameter.fixMods = cmd.getOptionValue("fixMod");
        }

        // HashMap<Integer,Integer>mod_index2code = new HashMap<>();
        if (cmd.hasOption("varMod")) {
            CParameter.varMods = cmd.getOptionValue("varMod");
            for (String mod : cmd.getOptionValue("varMod").split(",")) {
                String[] d = mod.split(":");
                // if(d.length>=2) {
                // mod_index2code.put(Integer.parseInt(d[0]), Integer.parseInt(d[1]));
                // }
            }
        }

        if (cmd.hasOption("user_var_mods")) {
            AIWorker.user_mod = cmd.getOptionValue("user_var_mods");
            String[] user_mods = cmd.getOptionValue("user_var_mods").split(";");
            int i = 0;
            ArrayList<Integer> mod_i = new ArrayList<>();
            for (String m : user_mods) {
                i++;
                // mod_name,aa,mass,composition
                String[] d = m.split(",");
                String mod_name = d[0];
                String aa = d[1];
                double mass = Double.parseDouble(d[2]);
                Modification ptm = null;
                String ptmName;
                if (CParameter.varMods.matches("^[1-9].*$")) {
                    int mi = CModification.getInstance().ptm_name2id.get(mod_name + " of " + aa);
                    mod_i.add(mi);
                }
            }
            String use_var_mod_str = StringUtils.join(mod_i, ',');
            if (!CParameter.varMods.isEmpty() && !CParameter.varMods.equals("0") && !CParameter.varMods.equals("no")) {
                CParameter.varMods = CParameter.varMods + "," + use_var_mod_str;
            } else {
                CParameter.varMods = use_var_mod_str;
            }
            CModification.getInstance().addVarMods(use_var_mod_str);
            Cloger.getInstance().logger.info(" -varMods " + CParameter.varMods);
        }

        if (cmd.hasOption("miss_c")) {
            CParameter.maxMissedCleavages = Integer.parseInt(cmd.getOptionValue("miss_c"));
        }

        DBGear.init_enzymes();
        if (cmd.hasOption("enzyme")) {
            if (cmd.getOptionValue("enzyme").equalsIgnoreCase("NoCut")) {
                CParameter.enzyme = DBGear.getEnzymeIndexByName(cmd.getOptionValue("enzyme"));
            } else {
                CParameter.enzyme = Integer.parseInt(cmd.getOptionValue("enzyme"));
            }
        }

        // for non-specific digestion, set the missed cleavages to be equal to 100
        if(DBGear.isNonSpecificEnzyme()){
            CParameter.maxMissedCleavages = 100;
        }

        if (cmd.hasOption("maxVar")) {
            CParameter.maxVarMods = Integer.parseInt(cmd.getOptionValue("maxVar"));
        }

        if (cmd.hasOption("clip_n_m")) {
            CParameter.clip_nTerm_M = true;
        } else {
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

        if (cmd.hasOption("itolu")) {
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

        if (cmd.hasOption("rf_rt_win")) {
            if (cmd.getOptionValue("rf_rt_win").equalsIgnoreCase("auto")) {
                // will determine rt window based on LC gradient length
                CParameter.rt_win = 0;
            } else {
                CParameter.rt_win = Double.parseDouble(cmd.getOptionValue("rf_rt_win"));
            }
        }

        String psm_file = cmd.getOptionValue("i");
        CParameter.init();
        ModificationUtils.getInstance();

        AIGear aiGear = new AIGear();
        aiGear.load_mod_map();

        if (cmd.hasOption("ai_version")) {
            aiGear.ai_version = cmd.getOptionValue("ai_version");
        }

        if (cmd.hasOption("torch_compile")) {
            aiGear.torch_compile = true;
            AIWorker.torch_compile = true;
        }

        if (cmd.hasOption("mod2mass")) {
            // change the mass of a modification
            ArrayList<String> mod2mass_list = new ArrayList<>();
            for (String mod : cmd.getOptionValue("mod2mass").split(",")) {
                String[] d = mod.split("@");
                CModification.getInstance().change_mod_mass(Integer.parseInt(d[0]), Double.parseDouble(d[1]));
                String mod_name = CModification.getInstance().id2ptmname.get(Integer.parseInt(d[0]));
                String psi_name_site = aiGear.mod_map.get(mod_name);
                mod2mass_list.add(psi_name_site + "=" + d[1]);
            }
            AIWorker.mod2mass = StringUtils.join(mod2mass_list, ',');
        }

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

        if (cmd.hasOption("use_all_peaks")) {
            // Don't do any filtering. Use all peaks and all detected peptides.
            aiGear.use_all_peaks = true;
        }

        if (cmd.hasOption("db")) {
            aiGear.db = cmd.getOptionValue("db");
        }

        if (cmd.hasOption("se")) {
            aiGear.search_engine = cmd.getOptionValue("se");
        }

        if (cmd.hasOption("data_type")) {
            aiGear.data_type = cmd.getOptionValue("data_type");
        }

        if (cmd.hasOption("no_masking")) {
            aiGear.no_masking = true;
        }

        if (cmd.hasOption("mode")) {
            aiGear.mod_ai = cmd.getOptionValue("mode");
        }

        if (cmd.hasOption("device")) {
            aiGear.device = cmd.getOptionValue("device");
        }

        if (aiGear.device.toLowerCase().contains("gpu")) {
            if (!CudaUtils.hasCuda()) {
                GPUTools tools = new GPUTools();
                GPUTools.TorchGpuStatus st = tools.checkTorchGpu();
                if (st.gpuAvailable) {
                    System.out.println("GPU is enabled!");
                } else {
                    aiGear.device = "cpu";
                    System.out.println("GPU is not available! Use CPU instead.");
                }
            } else {
                System.out.println("GPU is enabled!");
            }
        }

        if (cmd.hasOption("lf_frag_mz_min")) {
            aiGear.lf_frag_mz_min = Double.parseDouble(cmd.getOptionValue("lf_frag_mz_min"));
        }

        if (cmd.hasOption("lf_frag_mz_max")) {
            aiGear.lf_frag_mz_max = Double.parseDouble(cmd.getOptionValue("lf_frag_mz_max"));
        }

        if (cmd.hasOption("lf_top_n_frag")) {
            aiGear.lf_top_n_fragment_ions = Integer.parseInt(cmd.getOptionValue("lf_top_n_frag"));
        }

        if (cmd.hasOption("lf_min_n_frag")) {
            aiGear.lf_min_n_fragment_ions = Integer.parseInt(cmd.getOptionValue("lf_min_n_frag"));
        }

        if (cmd.hasOption("lf_frag_n_min")) {
            aiGear.lf_frag_n_min = Integer.parseInt(cmd.getOptionValue("lf_frag_n_min"));
        }

        if (cmd.hasOption("na")) {
            aiGear.n_flank_scans = Integer.parseInt(cmd.getOptionValue("na"));
        }

        if (cmd.hasOption("ms_instrument")) {
            aiGear.user_provided_ms_instrument = cmd.getOptionValue("ms_instrument");
            aiGear.use_user_provided_ms_instrument = true;
        }

        if (cmd.hasOption("nce")) {
            aiGear.nce = Double.parseDouble(cmd.getOptionValue("nce"));
            CParameter.NCE = aiGear.nce;
        }

        if (cmd.hasOption("xic")) {
            aiGear.export_xic = true;
        }

        if (cmd.hasOption("rt_win_offset")) {
            aiGear.rt_win_offset = Double.parseDouble(cmd.getOptionValue("rt_win_offset"));
        }

        if (cmd.hasOption("rt_max")) {
            aiGear.rt_max = Double.parseDouble(cmd.getOptionValue("rt_max"));
        }

        if (cmd.hasOption("ptm_site_prob")) {
            aiGear.ptm_site_prob_cutoff = Double.parseDouble(cmd.getOptionValue("ptm_site_prob"));
        }

        if (cmd.hasOption("ptm_site_qvalue")) {
            aiGear.ptm_site_qvalue_cutoff = Double.parseDouble(cmd.getOptionValue("ptm_site_qvalue"));
        }

        if (cmd.hasOption("seed")) {
            aiGear.global_random_seed = Integer.parseInt(cmd.getOptionValue("seed"));
        }

        if (cmd.hasOption("fast")) {
            aiGear.use_parquet = true;
        }

        if (cmd.hasOption("lf_format")) {
            aiGear.export_spectral_library_file_format = cmd.getOptionValue("lf_format");
        }

        if (cmd.hasOption("min_pep_charge")) {
            aiGear.lf_precursor_charge_min = Integer.parseInt(cmd.getOptionValue("min_pep_charge"));
        }

        if (cmd.hasOption("max_pep_charge")) {
            aiGear.lf_precursor_charge_max = Integer.parseInt(cmd.getOptionValue("max_pep_charge"));
        }

        aiGear.precursor_charges = new int[aiGear.lf_precursor_charge_max - aiGear.lf_precursor_charge_min + 1];
        for (int charge = aiGear.lf_precursor_charge_min; charge <= aiGear.lf_precursor_charge_max; charge++) {
            aiGear.precursor_charges[charge - aiGear.lf_precursor_charge_min] = charge;
        }

        if (cmd.hasOption("o")) {
            aiGear.out_dir = cmd.getOptionValue("o");
            // create output directory
            File F = new File(aiGear.out_dir);
            if (!F.isDirectory()) {
                F.mkdirs();
            }
            CParameter.outdir = aiGear.out_dir;
        }

        if (cmd.hasOption("lf_type")) {
            aiGear.export_spectral_library_format = cmd.getOptionValue("lf_type");
        }

        if (cmd.hasOption("y1")) {
            aiGear.remove_y1 = true;
        }

        if (cmd.hasOption("n_ion_min")) {
            aiGear.n_ion_min = Integer.parseInt(cmd.getOptionValue("n_ion_min"));
        }

        if (cmd.hasOption("c_ion_min")) {
            aiGear.c_ion_min = Integer.parseInt(cmd.getOptionValue("c_ion_min"));
        }

        if (cmd.hasOption("I2L")) {
            aiGear.I2L = true;
        }

        if (cmd.hasOption("export_mgf")) {
            aiGear.export_spectra_to_mgf = true;
        }

        if (cmd.hasOption("ccs")) {
            aiGear.ccs_enabled = true;
            AIWorker.ccs_enabled = true;
        }else if(cmd.hasOption("i") && cmd.hasOption("ms")){
            if(aiGear.is_timsTOF(cmd.getOptionValue("i"),cmd.getOptionValue("ms"))){
                aiGear.ccs_enabled = true;
                AIWorker.ccs_enabled = true;
            }
        }

        GenericUtils.get_system_memory_available();
        Runtime rt = Runtime.getRuntime();
        System.out.printf("Xms (current heap) = %.2f MB%n", rt.totalMemory() / 1024.0 / 1024.0);
        System.out.printf("Xmx (max heap) = %.2f GB%n", rt.maxMemory() / 1024.0 / 1024.0 / 1024.0);

        if (cmd.hasOption("ms2_model")) {
            aiGear.ms2_model = cmd.getOptionValue("ms2_model");
            Cloger.getInstance().logger.info("Use user-provided ms2 model: " + aiGear.ms2_model);
        }

        if (cmd.hasOption("ms") && !cmd.hasOption("model_dir")) {
            String ms_file = cmd.getOptionValue("ms");

            if (cmd.hasOption("rf")) {
                aiGear.refine_peak_boundary = true;
                System.out.println("Refine peak boundary");
            }

            if (cmd.hasOption("tf")) {
                CParameter.tf_type = cmd.getOptionValue("tf");
                test_mode = true;
            }
            Cloger.getInstance().set_job_start_time();
            // aiGear.load_data(psm_file, ms_file, aiGear.fdr_cutoff);
            if (aiGear.search_engine.equalsIgnoreCase("DIA-NN") || aiGear.search_engine.equalsIgnoreCase("DIANN")) {
                aiGear.load_data(psm_file, ms_file, aiGear.fdr_cutoff);
                if (aiGear.data_type.equalsIgnoreCase("dia")) {
                    if (aiGear.ccs_enabled) {
                        if (ms_file.endsWith("parquet")) {
                            aiGear.get_ms2_matches_diann_ccs();
                        } else {
                            aiGear.get_ms2_matches_diann_ccs_rust();
                        }
                    } else {
                        aiGear.get_ms2_matches_diann();
                    }
                } else {
                    aiGear.get_ms2_matches_diann_dda();
                }

            } else if (aiGear.search_engine.equalsIgnoreCase("generic") && aiGear.data_type.equalsIgnoreCase("dda")) {
                aiGear.rt_merge_method = "mean";
                File F = new File(ms_file);
                if (ms_file.endsWith(".mzML") || ms_file.endsWith(".mzml") || F.isDirectory()) {
                    String mgf_file = aiGear.out_dir + File.separator + "temp.mgf";
                    FileIO.generate_mgf_for_PSMs(psm_file, ms_file, mgf_file);
                    ms_file = mgf_file;
                }
                aiGear.load_data(psm_file, ms_file, aiGear.fdr_cutoff);
                aiGear.get_ms2_matches_generic_dda();
            } else if (aiGear.search_engine.equalsIgnoreCase("skyline") && aiGear.data_type.equalsIgnoreCase("dia")) {
                // DIA skyline
                CModification.getInstance();
                PSMConfig.use_skyline_report_column_names();
                String new_psm_file = aiGear.add_ms2spectrum_index(psm_file, ms_file);
                aiGear.load_data(new_psm_file, ms_file, aiGear.fdr_cutoff);
                aiGear.get_ms2_matches_diann();
            } else {
                aiGear.load_data(psm_file, ms_file, aiGear.fdr_cutoff);
                // TODO: need to update for CCS
                if (aiGear.ccs_enabled) {
                    System.out.println("CCS is not supported using user-defined input!");
                    System.exit(0);
                }
                aiGear.get_ms2_matches();
            }
            Cloger.getInstance().logger.info("Time used for training data generation: " + Cloger.getInstance().get_job_run_time());

            Cloger.getInstance().set_job_start_time();
            HashMap<String, String> paraMap = new HashMap<>();
            if (CParameter.tf_type.equalsIgnoreCase("nce")) {
                aiGear.nce = aiGear.select_best_nce(aiGear.out_dir, 20, 40);
            } else {
                aiGear.train_ms2_and_rt(paraMap, aiGear.out_dir, aiGear.out_dir, "test");
            }
            Cloger.getInstance().logger.info("Time used for model training: " + Cloger.getInstance().get_job_run_time());
            if (cmd.hasOption("db")) {
                aiGear.db = cmd.getOptionValue("db");
                CParameter.db = cmd.getOptionValue("db");
                String model_dir = aiGear.out_dir;
                Cloger.getInstance().set_job_start_time();
                Map<String, HashMap<String, String>> res_files = aiGear.generate_spectral_library(model_dir);
                if (cmd.hasOption("tf") && cmd.getOptionValue("tf").equalsIgnoreCase("test")) {
                    // aiGear.generate_multiple_library(res_files);
                    aiGear.generate_multiple_library_parallel(res_files);
                } else {
                    aiGear.generate_spectral_library(res_files);
                }
                Cloger.getInstance().logger.info("Time used for spectral library generation: " + Cloger.getInstance().get_job_run_time());
            }

        } else if (cmd.hasOption("model_dir") && cmd.hasOption("db")) {
            aiGear.model_dir = cmd.getOptionValue("model_dir");
            Cloger.getInstance().logger.info("Use the model in the folder: "+aiGear.model_dir+" for spectral library generation");
            // use the model in the model_dir for spectral library generation
            if (cmd.hasOption("tf")) {
                CParameter.tf_type = cmd.getOptionValue("tf");
                test_mode = true;
            }
            String meta_file = aiGear.model_dir + File.separator + "meta.json";
            String meta_json_string = Files.readString(Paths.get(meta_file));
            HashMap<String, JMeta> file2jMeta = JSON.parseObject(meta_json_string, new TypeReference<HashMap<String, JMeta>>() {});
            for(String m_file: file2jMeta.keySet()){
                aiGear.lf_frag_mz_max = file2jMeta.get(m_file).lf_frag_mz_max;
                aiGear.lf_frag_mz_min = file2jMeta.get(m_file).lf_frag_mz_min;
                // aiGear.lf_top_n_fragment_ions = file2jMeta.get(m_file).lf_top_n_fragment_ions;
                aiGear.max_fragment_ion_mz = file2jMeta.get(m_file).max_fragment_ion_mz;
                aiGear.min_fragment_ion_mz = file2jMeta.get(m_file).min_fragment_ion_mz;
                aiGear.ms_instrument = file2jMeta.get(m_file).ms_instrument;
                aiGear.nce = file2jMeta.get(m_file).nce;
                aiGear.rt_max = file2jMeta.get(m_file).rt_max;
                aiGear.rt_min = file2jMeta.get(m_file).rt_min;
                CParameter.minPeptideMz = file2jMeta.get(m_file).precursor_ion_mz_min - 0.5;
                CParameter.maxPeptideMz = file2jMeta.get(m_file).precursor_ion_mz_max - 0.5;
            }
            //
            Cloger.getInstance().set_job_start_time();
            if (cmd.hasOption("db")) {
                aiGear.db = cmd.getOptionValue("db");
                CParameter.db = cmd.getOptionValue("db");
                String model_dir = aiGear.model_dir;
                Cloger.getInstance().set_job_start_time();
                Map<String, HashMap<String, String>> res_files = aiGear.generate_spectral_library(model_dir);
                if (cmd.hasOption("tf") && cmd.getOptionValue("tf").equalsIgnoreCase("test")) {
                    // aiGear.generate_multiple_library(res_files);
                    aiGear.generate_multiple_library_parallel(res_files);
                } else {
                    aiGear.generate_spectral_library(res_files);
                }
                Cloger.getInstance().logger.info("Time used for spectral library generation: " + Cloger.getInstance().get_job_run_time());
            }

        } else {
            if (cmd.hasOption("db")) {
                aiGear.db = cmd.getOptionValue("db");
                CParameter.db = cmd.getOptionValue("db");
                String model_dir = aiGear.out_dir;
                Cloger.getInstance().set_job_start_time();
                Map<String, HashMap<String, String>> res_files = aiGear.generate_spectral_library(model_dir);
                aiGear.generate_spectral_library(res_files);
                Cloger.getInstance().logger.info("Time used for spectral library generation: "+Cloger.getInstance().get_job_run_time());
            }
        }

        long bTime = System.currentTimeMillis();
        Cloger.getInstance().logger.info("Time used for spectral library generation:" + (bTime - startTime) / 1000 + " s.");
        aiGear.print_parameters(StringUtils.join(args," "));
    }

    /**
     * Generate multiple spectral libraries for different models based on the
     * provided resource files.
     * 
     * @param res_files A HashMap containing data files used for spectral library
     *                  generation.
     */
    public void generate_multiple_library(Map<String, HashMap<String, String>> res_files) {
        try {
            generate_spectral_library(res_files);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // pretrained model
        Map<String, HashMap<String, String>> p_res_files = new LinkedHashMap<>();
        for (String i : res_files.keySet()) {
            System.out.println(i);
            p_res_files.put(i, new HashMap<>());
            String ms2_file = res_files.get(i).get("ms2");
            String ms2_intensity_file = res_files.get(i).get("ms2_intensity");
            String rt_file = res_files.get(i).get("rt");
            String ms2_mz_file = res_files.get(i).get("ms2_mz");

            File F = new File(ms2_file);
            // get the folder of file ms2_file
            String folder = F.getParent() + File.separator + "pretrained_models";
            p_res_files.get(i).put("ms2", get_file_path(ms2_file, folder));
            p_res_files.get(i).put("ms2_intensity", get_file_path(ms2_intensity_file, folder));
            p_res_files.get(i).put("rt", get_file_path(rt_file, folder));
            p_res_files.get(i).put("ms2_mz", get_file_path(ms2_mz_file, folder));
            if (ccs_enabled) {
                String ccs_file = res_files.get(i).get("ccs");
                p_res_files.get(i).put("ccs", get_file_path(ccs_file, folder));
            }
        }

        try {
            if(this.use_parquet) {
                if(this.export_spectral_library_format.equalsIgnoreCase("Skyline")) {
                    generate_spectral_library_parquet_skyline(p_res_files, out_dir, "carafe_spectral_library_pretrained.tsv");
                }else if(this.export_spectral_library_format.equalsIgnoreCase("mzSpecLib")) {
                    generate_spectral_library_parquet_mzSpecLib(p_res_files, out_dir, "carafe_spectral_library_pretrained.tsv");
                }else{
                    generate_spectral_library_parquet(p_res_files, out_dir, "carafe_spectral_library_pretrained.tsv");
                }
            } else {
                generate_spectral_library(p_res_files, out_dir, "carafe_spectral_library_pretrained.tsv");
            }
        } catch (IOException | SQLException e) {
            throw new RuntimeException(e);
        }

        // rt only model
        p_res_files.clear();
        p_res_files = new LinkedHashMap<>();
        for (String i : res_files.keySet()) {
            System.out.println(i);
            p_res_files.put(i, new HashMap<>());
            String ms2_file = res_files.get(i).get("ms2");
            String ms2_intensity_file = res_files.get(i).get("ms2_intensity");
            String rt_file = res_files.get(i).get("rt");
            String ms2_mz_file = res_files.get(i).get("ms2_mz");

            File F = new File(ms2_file);
            // get the folder of file ms2_file
            String folder = F.getParent() + File.separator + "pretrained_models";
            p_res_files.get(i).put("ms2", get_file_path(ms2_file, folder));
            p_res_files.get(i).put("ms2_intensity", get_file_path(ms2_intensity_file, folder));
            // Using fine-tuned RT model
            p_res_files.get(i).put("rt", rt_file);
            p_res_files.get(i).put("ms2_mz", get_file_path(ms2_mz_file, folder));
            if (ccs_enabled) {
                String ccs_file = res_files.get(i).get("ccs");
                p_res_files.get(i).put("ccs", get_file_path(ccs_file, folder));
            }
        }

        try {
            if(this.use_parquet){
                if(this.export_spectral_library_format.equalsIgnoreCase("Skyline")) {
                    generate_spectral_library_parquet_skyline(p_res_files, out_dir, "carafe_spectral_library_rt_only.tsv");
                }else if(this.export_spectral_library_format.equalsIgnoreCase("mzSpecLib")) {
                    generate_spectral_library_parquet_mzSpecLib(p_res_files, out_dir, "carafe_spectral_library_rt_only.tsv");
                }else {
                    generate_spectral_library_parquet(p_res_files, out_dir, "carafe_spectral_library_rt_only.tsv");
                }
            } else {
                generate_spectral_library(p_res_files, out_dir, "carafe_spectral_library_rt_only.tsv");
            }
        } catch (IOException | SQLException e) {
            throw new RuntimeException(e);
        }

        // ms2 only model
        p_res_files.clear();
        p_res_files = new LinkedHashMap<>();
        for (String i : res_files.keySet()) {
            System.out.println(i);
            p_res_files.put(i, new HashMap<>());
            String ms2_file = res_files.get(i).get("ms2");
            String ms2_intensity_file = res_files.get(i).get("ms2_intensity");
            String rt_file = res_files.get(i).get("rt");
            String ms2_mz_file = res_files.get(i).get("ms2_mz");

            File F = new File(ms2_file);
            // get the folder of file ms2_file
            String folder = F.getParent() + File.separator + "pretrained_models";
            // Using fine-tuned MS2 model
            p_res_files.get(i).put("ms2", ms2_file);
            p_res_files.get(i).put("ms2_intensity", ms2_intensity_file);
            p_res_files.get(i).put("rt", get_file_path(rt_file, folder));
            p_res_files.get(i).put("ms2_mz", ms2_mz_file);
            if (ccs_enabled) {
                String ccs_file = res_files.get(i).get("ccs");
                p_res_files.get(i).put("ccs", get_file_path(ccs_file, folder));
            }
        }

        try {
            if(this.use_parquet) {
                if(this.export_spectral_library_format.equalsIgnoreCase("Skyline")) {
                    generate_spectral_library_parquet_skyline(p_res_files, out_dir, "carafe_spectral_library_ms2_only.tsv");
                }else if(this.export_spectral_library_format.equalsIgnoreCase("mzSpecLib")) {
                    generate_spectral_library_parquet_mzSpecLib(p_res_files, out_dir, "carafe_spectral_library_ms2_only.tsv");
                }else {
                    generate_spectral_library_parquet(p_res_files, out_dir, "carafe_spectral_library_ms2_only.tsv");
                }
            } else {
                generate_spectral_library(p_res_files, out_dir, "carafe_spectral_library_ms2_only.tsv");
            }
        } catch (IOException | SQLException e) {
            throw new RuntimeException(e);
        }

        // CCS only model
        if (ccs_enabled) {
            p_res_files.clear();
            p_res_files = new LinkedHashMap<>();
            for (String i : res_files.keySet()) {
                System.out.println(i);
                p_res_files.put(i, new HashMap<>());
                String ms2_file = res_files.get(i).get("ms2");
                String ms2_intensity_file = res_files.get(i).get("ms2_intensity");
                String rt_file = res_files.get(i).get("rt");
                String ms2_mz_file = res_files.get(i).get("ms2_mz");
                String ccs_file = res_files.get(i).get("ccs");

                File F = new File(ms2_file);
                // get the folder of file ms2_file
                String folder = F.getParent() + File.separator + "pretrained_models";
                p_res_files.get(i).put("ms2", get_file_path(ms2_file, folder));
                p_res_files.get(i).put("ms2_intensity", get_file_path(ms2_intensity_file, folder));
                p_res_files.get(i).put("rt", get_file_path(rt_file, folder));
                p_res_files.get(i).put("ms2_mz", get_file_path(ms2_mz_file, folder));
                // Using fine-tuned CCS model
                p_res_files.get(i).put("ccs", ccs_file);
            }

            try {
                if(this.use_parquet) {
                    if(this.export_spectral_library_format.equalsIgnoreCase("Skyline")) {
                        generate_spectral_library_parquet_skyline(p_res_files, out_dir, "carafe_spectral_library_ccs_only.tsv");
                    }else if(this.export_spectral_library_format.equalsIgnoreCase("mzSpecLib")) {
                        generate_spectral_library_parquet_mzSpecLib(p_res_files, out_dir, "carafe_spectral_library_ccs_only.tsv");
                    }else {
                        generate_spectral_library_parquet(p_res_files, out_dir, "carafe_spectral_library_ccs_only.tsv");
                    }
                } else {
                    generate_spectral_library(p_res_files, out_dir, "carafe_spectral_library_ccs_only.tsv");
                }
            } catch (IOException | SQLException e) {
                throw new RuntimeException(e);
            }
        }

    }

    /**
     * Generate multiple spectral libraries for different models based on the
     * provided resource files in parallel.
     * This is an optimized version that uses multithreading to speed up the
     * process. It is expected to generate
     * the same results as the generate_multiple_library method.
     * 
     * @param res_files A HashMap containing data files used for spectral library
     *                  generation.
     */
    public void generate_multiple_library_parallel(Map<String, HashMap<String, String>> res_files) {

        // 0. Add protein information to files
        try {
            this.add_protein_to_psm_table(res_files);
            Map<String, HashMap<String, String>> p_res_files = new LinkedHashMap<>();
            for (String i : res_files.keySet()) {
                p_res_files.put(i, new HashMap<>());
                String ms2_file = res_files.get(i).get("ms2");
                File F = new File(ms2_file);
                // get the folder of file ms2_file
                String folder = F.getParent() + File.separator + "pretrained_models";
                p_res_files.get(i).put("ms2", get_file_path(ms2_file, folder));
            }
            this.add_protein_to_psm_table(p_res_files);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // 1. Create the thread pool manually
        // Fixed pool of 5 because we have a known max number of library types
        int n_threads = ccs_enabled ? 5 : 4;
        n_threads = Math.min(n_threads, Runtime.getRuntime().availableProcessors());
        ExecutorService executor = Executors.newFixedThreadPool(n_threads);

        // 2. Define the tasks list
        List<Callable<Void>> tasks = new ArrayList<>();

        // --- Task 1: Standard Spectral Library ---
        tasks.add(() -> {
            try {
                generate_spectral_library(res_files);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            return null;
        });

        // --- Task 2: Pretrained Model ---
        tasks.add(() -> {
            Map<String, HashMap<String, String>> p_res_files = new LinkedHashMap<>();
            for (String i : res_files.keySet()) {
                System.out.println("Pretrained: " + i);
                p_res_files.put(i, new HashMap<>());
                String ms2_file = res_files.get(i).get("ms2");
                String ms2_intensity_file = res_files.get(i).get("ms2_intensity");
                String rt_file = res_files.get(i).get("rt");
                String ms2_mz_file = res_files.get(i).get("ms2_mz");

                File F = new File(ms2_file);
                // get the folder of file ms2_file
                String folder = F.getParent() + File.separator + "pretrained_models";

                p_res_files.get(i).put("ms2", get_file_path(ms2_file, folder));
                p_res_files.get(i).put("ms2_intensity", get_file_path(ms2_intensity_file, folder));
                p_res_files.get(i).put("rt", get_file_path(rt_file, folder));
                p_res_files.get(i).put("ms2_mz", get_file_path(ms2_mz_file, folder));

                if (ccs_enabled) {
                    String ccs_file = res_files.get(i).get("ccs");
                    p_res_files.get(i).put("ccs", get_file_path(ccs_file, folder));
                }
            }
            run_spectral_library_generation(p_res_files, "carafe_spectral_library_pretrained.tsv");
            return null;
        });

        // --- Task 3: RT Only Model ---
        tasks.add(() -> {
            Map<String, HashMap<String, String>> p_res_files = new LinkedHashMap<>();
            for (String i : res_files.keySet()) {
                System.out.println("RT Only: " + i);
                p_res_files.put(i, new HashMap<>());
                String ms2_file = res_files.get(i).get("ms2");
                String ms2_intensity_file = res_files.get(i).get("ms2_intensity");
                String rt_file = res_files.get(i).get("rt");
                String ms2_mz_file = res_files.get(i).get("ms2_mz");

                File F = new File(ms2_file);
                // get the folder of file ms2_file
                String folder = F.getParent() + File.separator + "pretrained_models";
                p_res_files.get(i).put("ms2", get_file_path(ms2_file, folder));
                p_res_files.get(i).put("ms2_intensity", get_file_path(ms2_intensity_file, folder));
                // Using fine-tuned RT model
                p_res_files.get(i).put("rt", rt_file);
                p_res_files.get(i).put("ms2_mz", get_file_path(ms2_mz_file, folder));

                if (ccs_enabled) {
                    String ccs_file = res_files.get(i).get("ccs");
                    p_res_files.get(i).put("ccs", get_file_path(ccs_file, folder));
                }
            }
            run_spectral_library_generation(p_res_files, "carafe_spectral_library_rt_only.tsv");
            return null;
        });

        // --- Task 4: MS2 Only Model ---
        tasks.add(() -> {
            Map<String, HashMap<String, String>> p_res_files = new LinkedHashMap<>();
            for (String i : res_files.keySet()) {
                System.out.println("MS2 Only: " + i);
                p_res_files.put(i, new HashMap<>());
                String ms2_file = res_files.get(i).get("ms2");
                String ms2_intensity_file = res_files.get(i).get("ms2_intensity");
                String rt_file = res_files.get(i).get("rt");
                String ms2_mz_file = res_files.get(i).get("ms2_mz");

                File F = new File(ms2_file);
                // get the folder of file ms2_file
                String folder = F.getParent() + File.separator + "pretrained_models";
                // Using fine-tuned MS2 model
                p_res_files.get(i).put("ms2", ms2_file);
                p_res_files.get(i).put("ms2_intensity", ms2_intensity_file);
                p_res_files.get(i).put("rt", get_file_path(rt_file, folder));
                p_res_files.get(i).put("ms2_mz", ms2_mz_file);

                if (ccs_enabled) {
                    String ccs_file = res_files.get(i).get("ccs");
                    p_res_files.get(i).put("ccs", get_file_path(ccs_file, folder));
                }
            }
            run_spectral_library_generation(p_res_files, "carafe_spectral_library_ms2_only.tsv");
            return null;
        });

        // --- Task 5: CCS Only Model ---
        if (ccs_enabled) {
            tasks.add(() -> {
                Map<String, HashMap<String, String>> p_res_files = new LinkedHashMap<>();
                for (String i : res_files.keySet()) {
                    System.out.println("CCS Only: " + i);
                    p_res_files.put(i, new HashMap<>());
                    String ms2_file = res_files.get(i).get("ms2");
                    String ms2_intensity_file = res_files.get(i).get("ms2_intensity");
                    String rt_file = res_files.get(i).get("rt");
                    String ms2_mz_file = res_files.get(i).get("ms2_mz");
                    String ccs_file = res_files.get(i).get("ccs");

                    File F = new File(ms2_file);
                    // get the folder of file ms2_file
                    String folder = F.getParent() + File.separator + "pretrained_models";
                    p_res_files.get(i).put("ms2", get_file_path(ms2_file, folder));
                    p_res_files.get(i).put("ms2_intensity", get_file_path(ms2_intensity_file, folder));
                    p_res_files.get(i).put("rt", get_file_path(rt_file, folder));
                    p_res_files.get(i).put("ms2_mz", get_file_path(ms2_mz_file, folder));
                    // Using fine-tuned CCS model
                    p_res_files.get(i).put("ccs", ccs_file);
                }
                run_spectral_library_generation(p_res_files, "carafe_spectral_library_ccs_only.tsv");
                return null;
            });
        }

        // --- 3. Execute with classic Try/Catch/Finally ---
        try {
            List<Future<Void>> futures = executor.invokeAll(tasks);

            // Wait for all tasks to verify no exceptions occurred
            for (Future<Void> future : futures) {
                future.get();
            }
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException("Error during parallel library generation", e);
        } finally {
            // Essential: This ensures the threads are killed even if the tasks fail
            executor.shutdown();
        }
    }

    /**
     * Helper method for generate_multiple_library_parallel above.
     */
    private void run_spectral_library_generation(Map<String, HashMap<String, String>> files,
            String spectral_library_file_name) throws IOException, SQLException {
        if (this.use_parquet) {
            if (this.export_spectral_library_format.equalsIgnoreCase("Skyline")) {
                generate_spectral_library_parquet_skyline(files, out_dir, spectral_library_file_name);
            } else if (this.export_spectral_library_format.equalsIgnoreCase("mzSpecLib")) {
                generate_spectral_library_parquet_mzSpecLib(files, out_dir, spectral_library_file_name);
            } else {
                generate_spectral_library_parquet(files, out_dir, spectral_library_file_name);
            }
        } else {
            generate_spectral_library(files, out_dir, spectral_library_file_name);
        }
    }

    /**
     * A file path which is the same as the original_file but in the folder of
     * "new_folder".
     * 
     * @param original_file A file path of the original file
     * @param new_folder    A folder path
     * @return
     */
    private String get_file_path(String original_file, String new_folder) {
        File F = new File(original_file);
        return new_folder + File.separator + F.getName();
    }

    /**
     * A comparator for sorting peptides by their mass in ascending order.
     */
    public final Comparator<Peptide> comparator_peptide_mass_for_peptide_from_min2max = Comparator
            .comparingDouble(Peptide::getMass);

    /**
     * Generate a spectral library based on the models stored in the provided model
     * directory.
     * 
     * @param model_dir A folder path containing the trained models for spectral
     *                  library generation.
     * @return A map containing the prediction data files for spectral library
     *         generation.
     * @throws IOException
     */
    public Map<String, HashMap<String, String>> generate_spectral_library(String model_dir) throws IOException {
        // digest proteins and generate peptide forms
        // need to consider for both small and large databases
        long startTime = System.currentTimeMillis();
        DBGear dbGear = new DBGear();
        dbGear.I2L = this.I2L;
        Set<String> searchedPeptides = new HashSet<>();
        List<Peptide> all_peptide_forms = new ArrayList<>();
        List<Integer> precursor_charge_list = new ArrayList<>();
        if (this.db.toLowerCase().endsWith(".fa") || this.db.toLowerCase().endsWith(".fasta")) {
            searchedPeptides = dbGear.protein_digest(this.db);

            all_peptide_forms = searchedPeptides.parallelStream()
                    .map(PeptideUtils::calcPeptideIsoforms)
                    .flatMap(List::stream).sorted(comparator_peptide_mass_for_peptide_from_min2max).collect(toList());
        }else if(this.db.toLowerCase().endsWith(".tsv") || this.db.toLowerCase().endsWith(".txt") || this.db.toLowerCase().endsWith(".csv")){
            Cloger.getInstance().logger.info("The input for spectral library generation is a peptide forms table: " + this.db);
            char sep = '\t';
            if (this.db.toLowerCase().endsWith(".csv")) {
                sep = ',';
            }
            SkylineIO.load_skyline_precursor_table(this.db, sep, all_peptide_forms, precursor_charge_list);
        }
        Cloger.getInstance().logger.info("Generating peptide forms: " + all_peptide_forms.size());

        // decide whether to add nce column
        boolean add_nce_column = false;
        if (this.ccs_enabled && !this.use_fixed_ce) {
            add_nce_column = true;
        }
        // generate input files for prediction
        ArrayList<String> input_files = new ArrayList<>();
        if (this.use_parquet) {
            System.out.println("Use parquet format ...");
            ParquetWriter<GenericRecord> pWriter = null;
            Schema schema = FileIO.getSchema4PredictionInput(add_nce_column);
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
                        if(add_nce_column){
                            pep_out = get_InputRecord_for_prediction_ce(all_peptide_forms.get(i_peptide), pepID, schema);
                        }else {
                            pep_out = get_InputRecord_for_prediction(all_peptide_forms.get(i_peptide), pepID, schema);
                        }
                    }else{
                        if(add_nce_column){
                            pep_out = get_InputRecord_for_prediction_ce(all_peptide_forms.get(i_peptide), pepID, schema,precursor_charge_list.get(i_peptide));
                        }else {
                            pep_out = get_InputRecord_for_prediction(all_peptide_forms.get(i_peptide), pepID, schema, precursor_charge_list.get(i_peptide));
                        }
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
                            for (GenericRecord record : pep_out) {
                                pWriter.write(record);
                            }
                            pepID++;
                        }
                        i_peptide++;
                    } else if (i == (this.n_peptides_per_batch - 1)) {
                        // last row in this batch
                        if (!pep_out.isEmpty()) {
                            for (GenericRecord record : pep_out) {
                                pWriter.write(record);
                            }
                            pepID++;
                        }
                        i_peptide++;
                        pWriter.close();
                        file_is_closed = true;
                    } else {
                        if (!pep_out.isEmpty()) {
                            for (GenericRecord record : pep_out) {
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
        } else {
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
                    if (precursor_charge_list.isEmpty()) {
                        if (add_nce_column) {
                            pep_out = get_input_for_prediction_ce(all_peptide_forms.get(i_peptide), pepID);
                        } else {
                            pep_out = get_input_for_prediction(all_peptide_forms.get(i_peptide), pepID);
                        }
                    }else{
                        if(add_nce_column){
                            pep_out = get_input_for_prediction_ce(all_peptide_forms.get(i_peptide), pepID, precursor_charge_list.get(i_peptide));
                        }else {
                            pep_out = get_input_for_prediction(all_peptide_forms.get(i_peptide), pepID, precursor_charge_list.get(i_peptide));
                        }
                    }
                    if (i == 0) {
                        // first row in the batch
                        k++;
                        String o_file = this.out_dir + File.separator + "peptide_forms_" + k + ".tsv";
                        input_files.add(o_file);
                        pWriter = new BufferedWriter(new FileWriter(o_file));
                        file_is_closed = false;
                        if (add_nce_column) {
                            pWriter.write("pepID\tsequence\tmz\tcharge\tnce\tmods\tmod_sites\n");
                        } else {
                            pWriter.write("pepID\tsequence\tmz\tcharge\tmods\tmod_sites\n");
                        }
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
            if (!file_is_closed && pWriter != null) {
                pWriter.close();
            }
        }

        Map<String, HashMap<String, String>> res_files = new LinkedHashMap<>();

        if (this.device.toLowerCase().contains("gpu")) {
            if (CudaUtils.hasCuda()) {
                MemoryUsage mem = CudaUtils.getGpuMemory(Device.gpu());
                long gpu_mem = mem.getMax(); // it should return 11GB
                Cloger.getInstance().logger.info("GPU memory: " + gpu_mem);
                Cloger.getInstance().logger.info("CUDA version: " + CudaUtils.getCudaVersionString());
                Cloger.getInstance().logger.info("GPU number: " + CudaUtils.getGpuCount());
                Cloger.getInstance().logger.info(mem.toString());
            } else {
                GPUTools tools = new GPUTools();
                GPUTools.TorchGpuStatus st = tools.checkTorchGpu();
                if (st.gpuAvailable) {
                    Cloger.getInstance().logger.info("GPU memory: " + st.gpu_memory);
                    Cloger.getInstance().logger.info("CUDA version: " + st.cudaVersion);
                    Cloger.getInstance().logger.info("GPU: " + st.deviceName);
                } else {
                    Cloger.getInstance().logger.warn("GPU not available; falling back to CPU.");
                    this.device = "cpu";
                }
            }
        }

        AIWorker.fast_mode = this.use_parquet;

        if (this.device.toLowerCase().contains("cpu")) {
            // only use 1 cpu for now
            ExecutorService fixedThreadPool = Executors.newFixedThreadPool(1);
            Cloger.getInstance().logger.info("Number of CPU jobs " + 1);
            AIWorker.python_bin = this.python_bin;
            // perform spectrum and rt prediction.
            String mode = this.mod_ai.equalsIgnoreCase("-") ? "general" : this.mod_ai;
            System.out.println("NCE: " + this.nce);
            for (int i = 0; i < input_files.size(); i++) {
                // prediction
                if(this.use_user_provided_ms_instrument) {
                    fixedThreadPool.execute(new AIWorker(model_dir, input_files.get(i), this.out_dir, i + "", this.device, this.user_provided_ms_instrument, this.nce, this.mod_ai, this.ai_version));
                }else{
                    fixedThreadPool.execute(new AIWorker(model_dir, input_files.get(i), this.out_dir, i + "", this.device, this.ms_instrument, this.nce, this.mod_ai, this.ai_version));
                }
                res_files.put(input_files.get(i), new HashMap<>());
                if (this.use_parquet) {
                    res_files.get(input_files.get(i)).put("ms2", this.out_dir + File.separator + i + "_ms2_df.parquet");
                    res_files.get(input_files.get(i)).put("ms2_mz", this.out_dir + File.separator + i + "_ms2_mz_df.parquet");
                    res_files.get(input_files.get(i)).put("ms2_intensity", this.out_dir + File.separator + i + "_ms2_pred.parquet");
                    res_files.get(input_files.get(i)).put("rt", this.out_dir + File.separator + i + "_rt_pred.parquet");
                    if(ccs_enabled){
                        res_files.get(input_files.get(i)).put("ccs", this.out_dir + File.separator + i + "_ccs_pred.parquet");
                    }
                } else {
                    res_files.get(input_files.get(i)).put("ms2", this.out_dir + File.separator + i + "_ms2_df.tsv");
                    res_files.get(input_files.get(i)).put("ms2_mz", this.out_dir + File.separator + i + "_ms2_mz_df.tsv");
                    res_files.get(input_files.get(i)).put("ms2_intensity", this.out_dir + File.separator + i + "_ms2_pred.tsv");
                    res_files.get(input_files.get(i)).put("rt", this.out_dir + File.separator + i + "_rt_pred.tsv");
                    if(ccs_enabled){
                        res_files.get(input_files.get(i)).put("ccs", this.out_dir + File.separator + i + "_ccs_pred.tsv");
                    }
                }
            }

            fixedThreadPool.shutdown();

            try {
                fixedThreadPool.awaitTermination(Long.MAX_VALUE, TimeUnit.HOURS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            double gpu_mem = get_gpu_mem();
            // Each prediction process loads MS2, RT, and CCS models which use ~3-4GB total
            int n_gpu_jobs = (int) Math.floor(gpu_mem / 3);
            try {
                if(DBGear.isNonSpecificEnzyme() || all_peptide_forms.size() >= 10_000_000 || this.mod_ai.equalsIgnoreCase("phosphorylation")){
                    // If it is non-specific enzyme digestion, use 5GB per job
                    n_gpu_jobs = (int) Math.floor(gpu_mem / 5);
                }
            } catch (Exception e) {
                // If there is any error, use default
                n_gpu_jobs = (int) Math.floor(gpu_mem / 3);
            }
            n_gpu_jobs = Math.max(1, n_gpu_jobs); // Ensure at least 1 job
            n_gpu_jobs = Math.min(n_gpu_jobs, input_files.size());
            // check if this is on a windows system
            if (System.getProperty("os.name").toLowerCase().contains("win")) {
                Cloger.getInstance().logger.info("Running on Windows");
                n_gpu_jobs = 1;
            }
            ExecutorService fixedThreadPool = Executors.newFixedThreadPool(n_gpu_jobs);
            Cloger.getInstance().logger.info("Number of GPU jobs " + n_gpu_jobs);
            AIWorker.python_bin = this.python_bin;
            // perform spectrum and rt prediction.
            String mode = this.mod_ai.equalsIgnoreCase("-") ? "general" : this.mod_ai;
            Cloger.getInstance().logger.info("NCE: " + this.nce);
            List<Future<?>> futures = new ArrayList<>();
            for (int i = 0; i < input_files.size(); i++) {
                // prediction
                AIWorker worker;
                if(this.use_user_provided_ms_instrument) {
                    worker = new AIWorker(model_dir, input_files.get(i), this.out_dir, i + "", this.device, this.user_provided_ms_instrument, this.nce, this.mod_ai, this.ai_version);
                }else{
                    worker = new AIWorker(model_dir, input_files.get(i), this.out_dir, i + "", this.device, this.ms_instrument, this.nce, this.mod_ai, this.ai_version);
                }
                futures.add(fixedThreadPool.submit(worker));
                res_files.put(input_files.get(i), new HashMap<>());
                if (this.use_parquet) {
                    res_files.get(input_files.get(i)).put("ms2", this.out_dir + File.separator + i + "_ms2_df.parquet");
                    res_files.get(input_files.get(i)).put("ms2_mz", this.out_dir + File.separator + i + "_ms2_mz_df.parquet");
                    res_files.get(input_files.get(i)).put("ms2_intensity", this.out_dir + File.separator + i + "_ms2_pred.parquet");
                    res_files.get(input_files.get(i)).put("rt", this.out_dir + File.separator + i + "_rt_pred.parquet");
                    if(ccs_enabled){
                        res_files.get(input_files.get(i)).put("ccs", this.out_dir + File.separator + i + "_ccs_pred.parquet");
                    }
                } else {
                    res_files.get(input_files.get(i)).put("ms2", this.out_dir + File.separator + i + "_ms2_df.tsv");
                    res_files.get(input_files.get(i)).put("ms2_mz", this.out_dir + File.separator + i + "_ms2_mz_df.tsv");
                    res_files.get(input_files.get(i)).put("ms2_intensity", this.out_dir + File.separator + i + "_ms2_pred.tsv");
                    res_files.get(input_files.get(i)).put("rt", this.out_dir + File.separator + i + "_rt_pred.tsv");
                    if(ccs_enabled){
                        res_files.get(input_files.get(i)).put("ccs", this.out_dir + File.separator + i + "_ccs_pred.tsv");
                    }
                }
            }

            fixedThreadPool.shutdown();

            try {
                for (Future<?> future : futures) {
                    future.get();
                }
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
                Cloger.getInstance().logger.error("AI prediction task failed: " + e.getMessage());
                if (e.getCause() != null) {
                    Cloger.getInstance().logger.error("Cause: " + e.getCause().getMessage());
                }
                System.exit(1);
            }
        }

        long bTime = System.currentTimeMillis();
        searchedPeptides.clear();
        Cloger.getInstance().logger.info("Time used for spectral library generation:" + (bTime - startTime) / 1000 + " s.");
        return res_files;
    }

    /**
     * Get the GPU memory available for the current device.
     * 
     * @return The total GPU memory in GB.
     */
    public double get_gpu_mem() {
        if (CudaUtils.hasCuda()) {
            MemoryUsage mem = CudaUtils.getGpuMemory(Device.gpu());
            return 1.0 * mem.getMax() / 1024 / 1024 / 1024;
        } else {
            GPUTools tools = new GPUTools();
            GPUTools.TorchGpuStatus st = tools.checkTorchGpu();
            return 1.0 * st.gpu_memory / 1024 / 1024 / 1024;
        }
    }

    /**
     * Generate a spectral library based on the provided prediction data files.
     * 
     * @param res_files A map containing the prediction data files for spectral
     *                  library generation.
     * @param out_dir   Output directory where the spectral library files will be
     *                  saved.
     * @throws IOException
     */
    public void generate_spectral_library(Map<String, HashMap<String, String>> res_files, String out_dir)
            throws IOException {
        String pep_index_file = out_dir + File.separator + "pep_index.tsv";
        String frag_ions_file = out_dir + File.separator + "frag_ions.tsv";

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
        for (String i : res_files.keySet()) {
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
            while ((line = rtReader.readLine()) != null) {
                String[] d = line.split("\t");
                pepID = d[rt_col2index.get("pepID")];
                sequence = d[rt_col2index.get("sequence")];
                mods = d[rt_col2index.get("mods")];
                mod_sites = d[rt_col2index.get("mod_sites")];
                if (mods.isEmpty()) {
                    mods = "-";
                    mod_sites = "-";
                }

                if (this.rt_max > 0) {
                    rt = Double.parseDouble(d[rt_col2index.get("rt_pred")]);
                    rt = rt * this.rt_max;
                } else {
                    rt = Double.parseDouble(d[rt_col2index.get("irt_pred")]);
                }
                if (!cur_pepIDs.contains(pepID)) {
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
            while ((line = ms2IntensityReader.readLine()) != null) {
                ms2_intensity_lines.add(line.trim());
            }
            ms2IntensityReader.close();

            // mz intensity
            ArrayList<String> ms2_mz_lines = new ArrayList<>();
            while ((line = ms2mzReader.readLine()) != null) {
                ms2_mz_lines.add(line.trim());
            }
            ms2mzReader.close();

            // MS2 information
            String[] ion_mz_intensity;
            while ((line = ms2Reader.readLine()) != null) {
                String[] d = line.split("\t");
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
        String old_pep_index_file = pep_index_file.replaceAll("tsv$", "") + "tmp";
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

    /**
     * Get the fragment ion intensity for a precursor.
     * 
     * @param ms2_mz_lines        ArrayList containing the MS2 m/z lines.
     * @param ms2_intensity_lines ArrayList containing the MS2 intensity lines.
     * @param column_names        Array of column names for the fragment ions.
     * @param frag_start_idx      Starting index for the fragment ions.
     * @param frag_stop_idx       Stopping index for the fragment ions.
     * @param top_n               Number of top fragment ions to consider.
     * @return An array containing two strings: m/z values and their corresponding
     *         intensities.
     */
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
        mz2intensity = mz2intensity.entrySet().stream().sorted(Map.Entry.<Double, Double>comparingByValue().reversed())
                .limit(top_n)
                .collect(toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e1, LinkedHashMap::new));
        res[0] = StringUtils.join(mz2intensity.keySet(), ",");
        res[1] = StringUtils.join(mz2intensity.values(), ",");
        return res;
    }

    /**
     * Get the fragment ion intensity for a precursor.
     * 
     * @param ms2_mz_lines        ArrayList containing the MS2 m/z lines.
     * @param ms2_intensity_lines ArrayList containing the MS2 intensity lines.
     * @param column_names        Array of column names for the fragment ions.
     * @param frag_start_idx      Starting index for the fragment ions.
     * @param frag_stop_idx       Stopping index for the fragment ions.
     * @param top_n               Number of top fragment ions to consider.
     * @param ion_types           Array of ion types.
     * @param mod_losses          Array of modification losses.
     * @param ion_charges         Array of ion charges.
     * @param frag_n_min          Minimum fragment ion number to consider.
     * @return An ArrayList of LibFragment objects containing the fragment ion
     *         information.
     */
    ArrayList<LibFragment> get_fragment_ion_intensity4parquet_all(ArrayList<double[]> ms2_mz_lines,
            ArrayList<double[]> ms2_intensity_lines,
            String[] column_names,
            int frag_start_idx,
            int frag_stop_idx,
            int top_n,
            String[] ion_types,
            String[] mod_losses,
            int[] ion_charges,
            int frag_n_min) {
        double[] intensity;
        double[] mz;
        int b_ion_num = 1;
        int y_ion_num = frag_stop_idx - frag_start_idx;
        // ion string ID -> intensity
        HashMap<Integer, Double> ion2intensity = new HashMap<>();
        double max_intensity = 0;
        int ion_id = 0;
        for (int i = frag_start_idx; i < frag_stop_idx; i++) {
            intensity = ms2_intensity_lines.get(i);
            mz = ms2_mz_lines.get(i);
            for (int k = 0; k < column_names.length; k++) {
                ion_id++;
                //if (intensity[k] >= CParameter.min_fragment_ion_intensity && mz[k] >= this.min_fragment_ion_mz && mz[k] <= this.max_fragment_ion_mz) {
                if (intensity[k] > 0.0 && mz[k] >= this.lf_frag_mz_min && mz[k] <= this.lf_frag_mz_max) {
                    // StringBuilder stringBuilder = new StringBuilder();
                    if (ion_types[k].startsWith("b")) {
                        if (b_ion_num >= frag_n_min) {
                            ion2intensity.put(ion_id, intensity[k]);
                            if (max_intensity < intensity[k]) {
                                max_intensity = intensity[k];
                            }
                        }
                    } else {
                        if (y_ion_num >= frag_n_min) {
                            ion2intensity.put(ion_id, intensity[k]);
                            if (max_intensity < intensity[k]) {
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
                .sorted(Map.Entry.<Integer, Double>comparingByValue().reversed())
                .limit(top_n)
                .collect(toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e1, LinkedHashMap::new));

        // re-normalize fragment ion intensity
        b_ion_num = 1;
        y_ion_num = frag_stop_idx - frag_start_idx;
        ArrayList<LibFragment> fragments = new ArrayList<>(valid_mz_map.size());
        ion_id = 0;
        for (int i = frag_start_idx; i < frag_stop_idx; i++) {
            intensity = ms2_intensity_lines.get(i);
            mz = ms2_mz_lines.get(i);
            for (int k = 0; k < column_names.length; k++) {
                ion_id++;
                //if (intensity[k] >= CParameter.min_fragment_ion_intensity && mz[k] >= this.min_fragment_ion_mz && mz[k] <= this.max_fragment_ion_mz) {
                if (intensity[k] > 0.0 && mz[k] >= this.lf_frag_mz_min && mz[k] <= this.lf_frag_mz_max) {
                    if (ion_types[k].startsWith("b")) {
                        if (b_ion_num >= frag_n_min) {
                            if (valid_mz_map.containsKey(ion_id)) {
                                LibFragment fragment = new LibFragment();
                                fragment.FragmentMz = (float) mz[k];
                                fragment.RelativeIntensity = (float) (intensity[k] / max_intensity);
                                fragment.FragmentNumber = b_ion_num;
                                fragment.FragmentType = ion_types[k];
                                fragment.FragmentCharge = ion_charges[k];
                                fragment.FragmentLossType = mod_losses[k];
                                fragments.add(fragment);
                            }
                        }
                    } else {
                        if (y_ion_num >= frag_n_min) {
                            if (valid_mz_map.containsKey(ion_id)) {
                                LibFragment fragment = new LibFragment();
                                fragment.FragmentMz = (float) mz[k];
                                fragment.RelativeIntensity = (float) (intensity[k] / max_intensity);
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

        // sort ArrayList<LibFragment> fragments based on RelativeIntensity from max to
        // min
        fragments.sort(Comparator.comparingDouble((LibFragment f) -> f.RelativeIntensity).reversed());
        return fragments;
    }

    /**
     * Get the fragment ion intensity for a precursor.
     * 
     * @param ms2_mz_lines        ArrayList containing the MS2 m/z lines.
     * @param ms2_intensity_lines ArrayList containing the MS2 intensity lines.
     * @param column_names        Array of column names for the fragment ions.
     * @param frag_start_idx      Starting index for the fragment ions.
     * @param frag_stop_idx       Stopping index for the fragment ions.
     * @param top_n               Number of top fragment ions to consider.
     * @param ion_types           Array of ion types.
     * @param mod_losses          Array of modification losses.
     * @param ion_charges         Array of ion charges.
     * @param frag_n_min          Minimum fragment ion number to consider.
     * @return An ArrayList of strings containing the fragment ion information.
     */
    private ArrayList<String> get_fragment_ion_intensity(ArrayList<String> ms2_mz_lines,
            ArrayList<String> ms2_intensity_lines,
            String[] column_names,
            int frag_start_idx,
            int frag_stop_idx,
            int top_n,
            String[] ion_types,
            String[] mod_losses,
            int[] ion_charges,
            int frag_n_min) {
        double[] intensity;
        double[] mz;
        int b_ion_num = 1;
        int y_ion_num = frag_stop_idx - frag_start_idx;
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
                    } else {
                        if (y_ion_num >= frag_n_min) {
                            ion2intensity.put(ion_id, intensity[k]);
                            if (max_intensity < intensity[k]) {
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
                .sorted(Map.Entry.<Integer, Double>comparingByValue().reversed())
                .limit(top_n)
                .collect(toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e1, LinkedHashMap::new));

        // re-normalize fragment ion intensity
        b_ion_num = 1;
        y_ion_num = frag_stop_idx - frag_start_idx;
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
                    if (ion_types[k].startsWith("b")) {
                        if (b_ion_num >= frag_n_min) {
                            if (valid_mz_map.containsKey(ion_id)) {
                                stringBuilder.append(mz[k]).append("\t")
                                        .append(String.format("%.4f", intensity[k] / max_intensity)).append("\t")
                                        .append(ion_types[k]).append("\t")
                                        .append(b_ion_num).append("\t")
                                        .append(ion_charges[k]).append("\t")
                                        .append(mod_losses[k]);
                                ion_line2intensity.put(stringBuilder.toString(), intensity[k]);
                            }
                        }
                    } else {
                        if (y_ion_num >= frag_n_min) {
                            if (valid_mz_map.containsKey(ion_id)) {
                                stringBuilder.append(mz[k]).append("\t")
                                        .append(String.format("%.4f", intensity[k] / max_intensity)).append("\t")
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
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .collect(toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e1, LinkedHashMap::new));

        return new ArrayList<>(valid_mz_map_final.keySet());
    }

    /**
     * Generate input for a peptide for prediction. Different precursor charges are
     * considered.
     * 
     * @param peptide A Peptide object containing the peptide sequence and
     *                modifications when available.
     * @param pepID   Peptide ID, which is the index of the peptide in a peptide
     *                List<Peptide>
     * @return A string formatted for prediction input.
     */
    private String get_input_for_prediction(Peptide peptide, int pepID) {
        StringBuilder stringBuilder = new StringBuilder();
        double mz;
        String[] mods = convert_modification(peptide);
        for (int charge : this.precursor_charges) {
            mz = this.get_mz(peptide.getMass(), charge);
            if (mz >= CParameter.minPeptideMz && mz <= CParameter.maxPeptideMz) {
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

    /**
     * Generate input for a peptide for prediction. Different precursor charges are
     * considered.
     * 
     * @param peptide A Peptide object containing the peptide sequence and
     *                modifications when available.
     * @param pepID   Peptide ID, which is the index of the peptide in a peptide
     *                List<Peptide>
     * @return A string formatted for prediction input.
     */
    private String get_input_for_prediction_ce(Peptide peptide, int pepID) {
        StringBuilder stringBuilder = new StringBuilder();
        double mz;
        String[] mods = convert_modification(peptide);
        for (int charge : this.precursor_charges) {
            mz = this.get_mz(peptide.getMass(), charge);
            if (mz >= CParameter.minPeptideMz && mz <= CParameter.maxPeptideMz) {
                // sequence, charge, mods, mod_sites
                stringBuilder.append(pepID).append("\t")
                        .append(peptide.getSequence()).append("\t")
                        .append(mz).append("\t")
                        .append(charge).append("\t")
                        .append(get_ce_for_mz(mz)).append("\t")
                        .append(mods[0]).append("\t")
                        .append(mods[1]).append("\n");
            }
        }
        return stringBuilder.toString();
    }

    public double get_ce_for_mz(double mz) {
        return this.ccs_dia_meta.get_ce_for_mz(mz);
    }

    /**
     * Generate a single line of input for a peptide for prediction with a specific
     * precursor charge.
     * 
     * @param peptide          A Peptide object containing the peptide sequence and
     *                         modifications when available.
     * @param pepID            Peptide ID, which is the index of the peptide in a
     *                         peptide List<Peptide>
     * @param precursor_charge The precursor charge of the peptide.
     * @return A single line string formatted for prediction input.
     */
    private String get_input_for_prediction(Peptide peptide, int pepID, int precursor_charge) {
        StringBuilder stringBuilder = new StringBuilder();
        double mz;
        String[] mods = convert_modification(peptide);
        mz = this.get_mz(peptide.getMass(), precursor_charge);
        if (mz >= CParameter.minPeptideMz && mz <= CParameter.maxPeptideMz) {
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

    /**
     * Generate a single line of input for a peptide for prediction with a specific
     * precursor charge.
     * 
     * @param peptide          A Peptide object containing the peptide sequence and
     *                         modifications when available.
     * @param pepID            Peptide ID, which is the index of the peptide in a
     *                         peptide List<Peptide>
     * @param precursor_charge The precursor charge of the peptide.
     * @return A single line string formatted for prediction input.
     */
    private String get_input_for_prediction_ce(Peptide peptide, int pepID, int precursor_charge) {
        StringBuilder stringBuilder = new StringBuilder();
        double mz;
        String[] mods = convert_modification(peptide);
        mz = this.get_mz(peptide.getMass(), precursor_charge);
        if (mz >= CParameter.minPeptideMz && mz <= CParameter.maxPeptideMz) {
            // sequence, charge, mods, mod_sites
            stringBuilder.append(pepID).append("\t")
                    .append(peptide.getSequence()).append("\t")
                    .append(mz).append("\t")
                    .append(precursor_charge).append("\t")
                    .append(get_ce_for_mz(mz)).append("\t")
                    .append(mods[0]).append("\t")
                    .append(mods[1]).append("\n");
        }
        return stringBuilder.toString();
    }

    /**
     * Generate input records for a peptide for prediction. Different precursor
     * charges are considered.
     * 
     * @param peptide A Peptide object containing the peptide sequence and
     *                modifications when available.
     * @param pepID   Peptide ID, which is the index of the peptide in a peptide
     *                List<Peptide>
     * @param schema  The Avro schema for the input records.
     * @return An ArrayList of GenericRecord objects formatted for prediction input.
     */
    private ArrayList<GenericRecord> get_InputRecord_for_prediction(Peptide peptide, int pepID, Schema schema) {
        // StringBuilder stringBuilder = new StringBuilder();
        double mz;
        String[] mods = convert_modification(peptide);
        ArrayList<GenericRecord> records = new ArrayList<>();
        for (int charge : this.precursor_charges) {
            mz = this.get_mz(peptide.getMass(), charge);
            if (mz >= CParameter.minPeptideMz && mz <= CParameter.maxPeptideMz) {
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

    /**
     * Generate input records for a peptide for prediction. Different precursor
     * charges are considered.
     * 
     * @param peptide A Peptide object containing the peptide sequence and
     *                modifications when available.
     * @param pepID   Peptide ID, which is the index of the peptide in a peptide
     *                List<Peptide>
     * @param schema  The Avro schema for the input records.
     * @return An ArrayList of GenericRecord objects formatted for prediction input.
     */
    private ArrayList<GenericRecord> get_InputRecord_for_prediction_ce(Peptide peptide, int pepID, Schema schema) {
        // StringBuilder stringBuilder = new StringBuilder();
        double mz;
        String[] mods = convert_modification(peptide);
        ArrayList<GenericRecord> records = new ArrayList<>();
        for (int charge : this.precursor_charges) {
            mz = this.get_mz(peptide.getMass(), charge);
            if (mz >= CParameter.minPeptideMz && mz <= CParameter.maxPeptideMz) {
                // sequence, charge, mods, mod_sites
                GenericRecord record = new GenericData.Record(schema);
                record.put("pepID", pepID);
                record.put("sequence", peptide.getSequence());
                record.put("mz", mz);
                record.put("charge", charge);
                record.put("nce", get_ce_for_mz(mz));
                record.put("mods", mods[0]);
                record.put("mod_sites", mods[1]);
                records.add(record);
            }
        }
        return records;
    }

    /**
     * Generate an input record for a peptide for prediction with a specific
     * precursor charge.
     * 
     * @param peptide          A Peptide object containing the peptide sequence and
     *                         modifications when available.
     * @param pepID            Peptide ID, which is the index of the peptide in a
     *                         peptide List<Peptide>
     * @param schema           The Avro schema for the input records.
     * @param precursor_charge The precursor charge of the peptide.
     * @return An ArrayList of GenericRecord objects formatted for prediction input.
     */
    private ArrayList<GenericRecord> get_InputRecord_for_prediction(Peptide peptide, int pepID, Schema schema,
            int precursor_charge) {
        double mz;
        String[] mods = convert_modification(peptide);
        ArrayList<GenericRecord> records = new ArrayList<>();
        mz = this.get_mz(peptide.getMass(), precursor_charge);
        if (mz >= CParameter.minPeptideMz && mz <= CParameter.maxPeptideMz) {
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

    /**
     * Generate an input record for a peptide for prediction with a specific
     * precursor charge.
     * 
     * @param peptide          A Peptide object containing the peptide sequence and
     *                         modifications when available.
     * @param pepID            Peptide ID, which is the index of the peptide in a
     *                         peptide List<Peptide>
     * @param schema           The Avro schema for the input records.
     * @param precursor_charge The precursor charge of the peptide.
     * @return An ArrayList of GenericRecord objects formatted for prediction input.
     */
    private ArrayList<GenericRecord> get_InputRecord_for_prediction_ce(Peptide peptide, int pepID, Schema schema,
            int precursor_charge) {
        double mz;
        String[] mods = convert_modification(peptide);
        ArrayList<GenericRecord> records = new ArrayList<>();
        mz = this.get_mz(peptide.getMass(), precursor_charge);
        if (mz >= CParameter.minPeptideMz && mz <= CParameter.maxPeptideMz) {
            // sequence, charge, mods, mod_sites
            GenericRecord record = new GenericData.Record(schema);
            record.put("pepID", pepID);
            record.put("sequence", peptide.getSequence());
            record.put("mz", mz);
            record.put("charge", precursor_charge);
            record.put("nce", get_ce_for_mz(mz));
            record.put("mods", mods[0]);
            record.put("mod_sites", mods[1]);
            records.add(record);
        }
        return records;
    }

    /**
     * Calculate the m/z value for a given mass and charge.
     * 
     * @param mass   The mass of the ion.
     * @param charge The charge state of the ion.
     * @return The m/z value calculated using the formula (mass + charge * proton
     *         mass) / charge.
     */
    public double get_mz(double mass, int charge) {
        return (mass + charge * ElementaryIon.proton.getTheoreticMass()) / charge;
    }

    /**
     * Run an external command.
     * 
     * @param cmd The command to run as an array of strings.
     * @return True if the command executed successfully, false otherwise.
     */
    private boolean run_cmd(String[] cmd, String message_prefix) {
        System.out.println(String.join(" ", cmd));
        boolean pass = true;
        Runtime rt = Runtime.getRuntime();
        Process p;
        try {
            p = rt.exec(cmd);
        } catch (IOException e) {
            pass = false;
            throw new RuntimeException(e);
        }

        StreamLog errorLog = new StreamLog(p.getErrorStream(), message_prefix+" => Error:", true);
        StreamLog stdLog = new StreamLog(p.getInputStream(), message_prefix+" => Message:", true);

        errorLog.start();
        stdLog.start();

        try {
            int exitValue = p.waitFor();
            if (exitValue != 0) {
                pass = false;
                Cloger.getInstance().logger.error(message_prefix+" => Error:" + exitValue);
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

    /**
     * Train MS2 and retention time prediction models.
     * 
     * @param paraMap    A map containing parameters for training.
     * @param in_dir     Input directory containing training data files.
     * @param out_dir    Output directory where the trained model will be saved.
     * @param out_prefix Prefix for the output files.
     */
    public void train_ms2_and_rt(HashMap<String, String> paraMap, String in_dir, String out_dir, String out_prefix) {
        String psm_df = in_dir + "/psm_pdv.txt";
        String intensity_df = in_dir + "/fragment_intensity_df.tsv";
        String rt_data_file = in_dir + "/rt_data.tsv";
        String mode = this.mod_ai.equalsIgnoreCase("-") ? "general" : this.mod_ai;
        System.out.println("Model training ...");
        String ms_instrument_for_training;
        if (this.use_user_provided_ms_instrument) {
            System.out.println("MS instrument extracted from MS/MS file:" + this.ms_instrument);
            System.out.println("Use user provided MS instrument:" + this.user_provided_ms_instrument);
            ms_instrument_for_training = this.user_provided_ms_instrument;
        } else {
            System.out.println("MS instrument:" + this.ms_instrument);
            ms_instrument_for_training = this.ms_instrument;
        }
        System.out.println("NCE:" + this.nce);
        String ai_py_name = this.ai_version.equalsIgnoreCase("v2") ? "v2/ai.py" : "ai.py";
        String ai_py = get_jar_path() + File.separator + ai_py_name.replace("/", File.separator);
        File F = new File(ai_py);
        String py_resource_root = this.ai_version.equalsIgnoreCase("v2") ? "/py/" : "/main/java/ai/";
        if (!F.exists()) {
            ai_py = get_py_path(py_resource_root + ai_py_name, "carafe_ai", this.ai_version.equalsIgnoreCase("v2"));
        }
        // String cmd = this.python_bin + " " + ai_py +
        // " --in_dir " + in_dir +
        // " --out_dir " + out_dir +
        // " --out_prefix "+out_prefix +
        // " --device " + this.device +
        // " --instrument " + ms_instrument_for_training +
        // " --tf_type " + CParameter.tf_type +
        // " --nce " + this.nce+
        // " --seed " + this.global_random_seed +
        // " --mode " + mode;

        String[] cmd_list_short = new String[] {
                this.python_bin,
                "-u", // Forces immediate flushing to pipes
                ai_py,
                "--in_dir", in_dir,
                "--out_dir", out_dir,
                "--out_prefix", out_prefix,
                "--device", this.device,
                "--instrument", ms_instrument_for_training,
                "--tf_type", CParameter.tf_type,
                "--nce", String.valueOf(this.nce),
                "--seed", String.valueOf(this.global_random_seed),
                "--mode", mode,
                "--verbose", String.valueOf(CParameter.verbose.getValue())
        };
        ArrayList<String> cmd_list = new ArrayList<>(Arrays.asList(cmd_list_short));

        if(this.torch_compile && this.ai_version.equalsIgnoreCase("v2")){
            // cmd += " --torch_compile";
            cmd_list.add("--torch_compile");
        }

        if (this.no_masking) {
            // cmd = cmd + " --no_masking";
            cmd_list.add("--no_masking");
        }
        if (!CParameter.user_var_mods.isEmpty() && !CParameter.user_var_mods.equalsIgnoreCase("-")) {
            cmd_list.add("--user_mod");
            cmd_list.add("\"" + CParameter.user_var_mods + "\"");
            // cmd = cmd + " --user_mod \""+CParameter.user_var_mods + "\"";
        }

        if(!this.ms2_model.isEmpty()){
            cmd_list.add("--ms2_model");
            cmd_list.add(this.ms2_model);
        }
        // convert cmd_list to String []
        String[] cmd = new String[cmd_list.size()];
        cmd = cmd_list.toArray(cmd);
        run_cmd(cmd,"Model training");
    }

    /**
     * Get the parent directory of the JAR file
     * 
     * @return The parent directory of the JAR file.
     */
    public static String get_jar_path() {
        try {
            String jar_file = AIGear.class.getProtectionDomain().getCodeSource().getLocation().toURI().getPath();
            File F = new File(jar_file);
            return F.getParent();
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Extract the Python script from the JAR file to a temporary file and then
     * return its absolute path.
     * 
     * @param py_file_path          The path to the Python script within the JAR
     *                              file, e.g., "/py/v2/ai.py".
     * @param prefix                A prefix for the temporary file name.
     * @param extract_dependencies  If true, also extract models.py to the same
     *                              directory.
     * @return The absolute path of the extracted Python script file.
     */
    static String get_py_path(String py_file_path, String prefix, boolean extract_dependencies) {
        try {
            // Create a unique temporary directory
            Path tempDir = Files.createTempDirectory(prefix + "_");
            File F = new File(tempDir.toFile(), new File(py_file_path).getName());

            InputStream input = AIWorker.class.getResourceAsStream(py_file_path);
            if (input != null) {
                Files.copy(input, F.toPath(), StandardCopyOption.REPLACE_EXISTING);

                if (extract_dependencies) {
                    File models_file = new File(tempDir.toFile(), "models.py");
                    InputStream models_input = AIWorker.class.getResourceAsStream("/py/v2/models.py");
                    if (models_input != null) {
                        try {
                            Files.copy(models_input, models_file.toPath(), StandardCopyOption.REPLACE_EXISTING);
                        } catch (IOException e) {
                            Cloger.getInstance().logger.warn("Could not copy models.py: " + e.getMessage());
                        }
                    } else {
                        Cloger.getInstance().logger.error(Thread.currentThread().getName() + ": AI error: models.py not found in JAR");
                    }
                }
                // Register for deletion on exit
                F.deleteOnExit();
                if (extract_dependencies) {
                    new File(tempDir.toFile(), "models.py").deleteOnExit();
                }
                tempDir.toFile().deleteOnExit();

                return F.getAbsolutePath();
            } else {
                Cloger.getInstance().logger.error(Thread.currentThread().getName() + ": AI error: " + py_file_path + " not found");
                return "";
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Load peptide detection data.
     * 
     * @param psm_file   A file containing peptide detection data.
     * @param ms_file    A file containing MS data a folder containing MS files.
     * @param fdr_cutoff The FDR cutoff value for filtering peptide detection data.
     */
    public void load_data(String psm_file, String ms_file, double fdr_cutoff) {
        System.out.println("FDR cutoff:" + fdr_cutoff);
        try {
            if (!psm_file.endsWith(".parquet")) {
                // If the psm_file is not in parquet format, it is a text file
                hIndex = get_column_name2index(psm_file);
            }
            if (this.search_engine.equalsIgnoreCase("DIANN") || this.search_engine.equalsIgnoreCase("DIA-NN")) {
                System.out.println("DIANN search engine");
                String new_psm_file = this.out_dir + File.separator + "psm_rank_" + fdr_cutoff + ".tsv";
                remove_interference_peptides_diann(psm_file, new_psm_file);
                if (psm_file.endsWith(".parquet")) {
                    // need to get the column index from the new text file
                    hIndex = get_column_name2index(new_psm_file);
                    // check if ms2 scan is present in the file
                    if (!hIndex.containsKey(PSMConfig.ms2_index_column_name)) {
                        if (!is_timsTOF_ms_file(ms_file)) {
                            // if not timsTOF ms file, then report error
                            Cloger.getInstance().logger.error("MS2 scan column (" + PSMConfig.ms2_index_column_name + ") is missing in the input file: " + psm_file + ". If the DIA-NN search is done using DIA-NN v2.2.0, please add --export-quant to the command line!");
                            System.exit(1);
                        }
                    }
                }
                // ms_file2psm = get_ms_file2psm_diann(new_psm_file, ms_file, fdr_cutoff);
                ms_file2psm = get_ms_file2psm_diann_multiple_ms_runs(new_psm_file, ms_file, fdr_cutoff);
            } else if (this.search_engine.equalsIgnoreCase("generic") && this.data_type.equalsIgnoreCase("DDA")) {
                System.out.println("Generic search engine format for DDA data");
                ms_file2psm = get_ms_file2psm(psm_file, ms_file, fdr_cutoff);
            } else {
                String new_psm_file = this.out_dir + File.separator + "psm_rank_" + fdr_cutoff + ".tsv";
                remove_interference_peptides(psm_file, new_psm_file, fdr_cutoff);
                ms_file2psm = get_ms_file2psm(new_psm_file, ms_file, fdr_cutoff);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Load peptide detection data.
     * 
     * @param psm_file   A file containing peptide detection data.
     * @param fdr_cutoff The FDR cutoff value for filtering peptide detection data.
     */
    public void load_data(String psm_file, double fdr_cutoff) {
        System.out.println("FDR cutoff:" + fdr_cutoff);
        try {
            if (!psm_file.endsWith(".parquet")) {
                // If the psm_file is not in parquet format, it is a text file
                hIndex = get_column_name2index(psm_file);
            }
            if (this.search_engine.equalsIgnoreCase("DIANN") || this.search_engine.equalsIgnoreCase("DIA-NN")) {
                System.out.println("DIANN search engine");
                String new_psm_file = this.out_dir + File.separator + "psm_rank_" + fdr_cutoff + ".tsv";
                remove_interference_peptides_diann(psm_file, new_psm_file);
                if (psm_file.endsWith(".parquet")) {
                    // need to get the column index from the new text file
                    hIndex = get_column_name2index(new_psm_file);
                    // check if ms2 scan is present in the file
                    if(!hIndex.containsKey(PSMConfig.ms2_index_column_name)){
                        Cloger.getInstance().logger.error("MS2 scan column ("+PSMConfig.ms2_index_column_name+") is missing in the input file: "+psm_file+". If the DIA-NN search is done using DIA-NN v2.2.0, please add --export-quant to the command line!");
                        // System.exit(1);
                    }
                }
                // ms_file2psm = get_ms_file2psm_diann(new_psm_file, ms_file, fdr_cutoff);
                ms_file2psm = get_ms_file2psm_diann_multiple_ms_runs(new_psm_file, fdr_cutoff);
            } else if (this.search_engine.equalsIgnoreCase("generic") && this.data_type.equalsIgnoreCase("DDA")) {
                // TODO: to be implemented
            } else {
                // TODO: to be implemented
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Checks if the provided file path corresponds to a timsTOF mass spectrometry file.
     * The method determines this by looking for the presence of an "analysis.tdf" file
     * in the specified directory or its nested subdirectories.
     *
     * @param ms_file The path to the mass spectrometry file or directory to be checked.
     * @return {@code true} if the provided path corresponds to a timsTOF mass spectrometry file;
     *         {@code false} otherwise.
     */
    private boolean is_timsTOF_ms_file(String ms_file) {
        boolean is_timsTOF = false;
        File F = new File(ms_file);
        if (F.isDirectory()) {
            String tdf_file = ms_file + File.separator + "analysis.tdf";
            File F_tdf = new File(tdf_file);
            if (F_tdf.exists()) {
                is_timsTOF = true;
            } else {
                // check if there is any .d file in the folder
                File[] files = F.listFiles();
                if (files != null) {
                    for (File f : files) {
                        if (f.isDirectory()) {
                            tdf_file = f.getAbsolutePath() + File.separator + "analysis.tdf";
                            F_tdf = new File(tdf_file);
                            if (F_tdf.exists()) {
                                is_timsTOF = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
        return is_timsTOF;
    }

    /**
     * Determines if the data is from a timsTOF instrument based on the provided
     * DIA-NN report file and mass spectrometry (MS) file.
     *
     * @param diann_report_file The path to the DIA-NN report file.
     * @param ms_file The path to the mass spectrometry raw data file.
     * @return {@code true} if the data is determined to be from a timsTOF instrument; {@code false} otherwise.
     */
    private boolean is_timsTOF(String diann_report_file, String ms_file){
        boolean is_timsTOF = false;
        if(diann_report_file.endsWith(".tsv")){
            try (BufferedReader reader = new BufferedReader(new FileReader(diann_report_file))) {
                String headLine = reader.readLine();
                if (headLine != null) {
                    HashMap<String, Integer> hIndex = this.get_column_name2index_from_head_line(headLine.trim());
                    if (hIndex.containsKey("IM")) {
                        String line;
                        while ((line = reader.readLine()) != null) {
                            if (line.isBlank()) {
                                continue;
                            }
                            String[] d = line.trim().split("\t");
                            if (d.length <= hIndex.get("IM")) {
                                continue;
                            }
                            String imValue = d[hIndex.get("IM")].trim();
                            if (imValue.isEmpty()) {
                                continue;
                            }
                            try {
                                if (Double.parseDouble(imValue) > 0) {
                                    is_timsTOF = true;
                                    break;
                                }
                            } catch (NumberFormatException ignored) {
                            }
                        }
                    }
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }else if(diann_report_file.endsWith(".parquet")){
            try {
                Table reportTable = new TablesawParquetReader().read(new Source(new File(diann_report_file)));
                if (reportTable.columnNames().contains("IM")) {
                    for (int i = 0; i < reportTable.rowCount(); i++) {
                        Object imObject = reportTable.column("IM").get(i);
                        if (imObject instanceof Number && ((Number) imObject).doubleValue() > 0) {
                            is_timsTOF = true;
                            break;
                        }
                    }
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        if (!is_timsTOF && ms_file != null && !ms_file.isEmpty()) {
            is_timsTOF = is_timsTOF_ms_file(ms_file);
        }

        return is_timsTOF;
    }

    /**
     * Generate training data for MS2 and retention time prediction.
     * It writes the results to output files for model training.
     * 
     * @throws IOException If there is an error reading or writing files.
     */
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
        HashMap<String, PeptideRT> peptide2rt = new HashMap<>();

        // output
        int frag_start_idx = 0;
        int frag_stop_idx = 0;
        BufferedWriter psmWriter = new BufferedWriter(new FileWriter(this.out_dir+"/psm_pdv.txt"));
        psmWriter.write(this.psm_head_line+"\tmods\tmod_sites\tmax_fragment_ion_valid\tmax_cor_mz\tfrag_start_idx\tfrag_stop_idx\n");
        BufferedWriter msWriter = new BufferedWriter(new FileWriter(this.out_dir+"/ms_pdv.mgf"));
        BufferedWriter fragWriter = new BufferedWriter(new FileWriter(this.out_dir+"/fragment_intensity_df.tsv"));
        fragWriter.write(this.fragment_ion_intensity_head_line+"\n");

        BufferedWriter fragMzWriter = null;
        if (this.export_fragment_ion_mz_to_file) {
            fragMzWriter = new BufferedWriter(new FileWriter(this.out_dir + "/fragment_mz.tsv"));
            fragMzWriter.write(this.fragment_ion_intensity_head_line + "\n");
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

        if (this.export_skyline_transition_list_file) {

            tfWriter = new BufferedWriter(new FileWriter(this.out_dir+"/skyline_input.tsv"));
            tfWriter.write("Peptide\tPrecursor m/z\tProduct m/z\tLibraryIntensity\tExplicit Retention Time\tExplicit Retention Time Window\tNote\n");

            // peak boundary file
            tbWriter = new BufferedWriter(new FileWriter(this.out_dir + "/skyline_boundary.tsv"));
            tbWriter.write("MinStartTime\tMaxEndTime\tFileName\tPeptideModifiedSequence\tPrecursorCharge\n");

        }

        for (String ms_file : this.ms_file2psm.keySet()) {
            System.out.println("Process MS file:" + ms_file);
            // For store raw data
            DIAMeta meta = new DIAMeta();
            if (CParameter.itol > 0.2 && CParameter.itolu.startsWith("da")) {
                meta.fragment_ion_mz_bin_size = 0.5;
                System.out.println("Fragment ion bin size:" + meta.fragment_ion_mz_bin_size);
            }
            meta.isolation_win_mz_max = this.isolation_win_mz_max;
            meta.load_ms_data(ms_file);
            meta.get_ms_run_meta_data();

            this.min_fragment_ion_mz = meta.fragment_ion_mz_min;
            this.max_fragment_ion_mz = meta.fragment_ion_mz_max;
            System.out.println("Fragment ion m/z range:" + this.min_fragment_ion_mz + "," + this.max_fragment_ion_mz);

            DIAMap diaMap_tmp = new DIAMap();
            diaMap_tmp.meta = meta;
            if (this.target_isolation_wins.isEmpty()) {
                diaMap_tmp.target_isolation_wins.addAll(meta.isolationWindowMap.keySet());
            } else {
                diaMap_tmp.target_isolation_wins.addAll(this.target_isolation_wins);
            }

            if (meta.rt_max > this.rt_max) {
                this.rt_max = meta.rt_max;
                System.out.println("RT max:" + this.rt_max);
            } else {
                System.out.println("RT max:" + this.rt_max);
            }

            // for output
            HashSet<String> save_spectra = new HashSet<>();

            HashMap<String, ArrayList<String>> isoWinID2PSMs = new HashMap<>();
            for (String line : this.ms_file2psm.get(ms_file)) {
                String[] d = line.split("\t");
                String peptide = d[hIndex.get("peptide")];
                String modification = d[hIndex.get("modification")];
                int precursor_charge = Integer.parseInt(d[hIndex.get("charge")]);
                this.add_peptide(peptide,modification);
                String isoWinID = diaMap_tmp.get_isolation_window(dbGear.get_mz(this.get_peptide(peptide,modification).getMass(),precursor_charge));
                if(!isoWinID2PSMs.containsKey(isoWinID)){
                    isoWinID2PSMs.put(isoWinID,new ArrayList<>());
                }
                isoWinID2PSMs.get(isoWinID).add(line);

                String peptide_mod = peptide + "_" + modification;
                if (!peptide2rt.containsKey(peptide_mod)) {
                    peptide2rt.put(peptide_mod, new PeptideRT());
                }
                peptide2rt.get(peptide_mod).peptide = peptide;
                peptide2rt.get(peptide_mod).modification = modification;
                peptide2rt.get(peptide_mod).rts.add(Double.parseDouble(d[hIndex.get("apex_rt")]));
                peptide2rt.get(peptide_mod).scores.add(Double.parseDouble(d[hIndex.get("q_value")]));
            }

            for (String isoWinID : isoWinID2PSMs.keySet()) {
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
                            if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION
                                    || ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION) {
                                // add fragment ion number
                                PeptideFragmentIon fragmentIon = ((PeptideFragmentIon) ionMatch.ion);
                                int ion_number = fragmentIon.getNumber();

                                int ion_type_column_index = this.get_ion_type_column_index(ionMatch);

                                // for y ion
                                if (ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION) {
                                    fragment_ion_row_index = peptide.length() - ion_number - 1;
                                } else if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION) {
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

                                if (max_fragment_ion_intensity <= ionMatch.peakIntensity) {
                                    max_fragment_ion_intensity = ionMatch.peakIntensity;
                                    max_fragment_ion_row_index = fragment_ion_row_index;
                                    max_fragment_ion_column_index = ion_type_column_index;
                                }

                            }
                        }
                    }
                    if (!matched_ion_mzs.isEmpty()) {
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
                            if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION
                                    || ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION) {
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
                index2peptideMatch.values().parallelStream()
                        .forEach(peptideMatch -> xic_query(diaIndex, peptideMatch, isoWinID));
                row_i = -1;
                int[] ind = new int[] { 0, 0 };
                for (String line : isoWinID2PSMs.get(isoWinID)) {
                    row_i = row_i + 1;
                    PeptideMatch peptideMatch = index2peptideMatch.get(row_i);
                    HashSet<Double> high_cor_mzs = new HashSet<>();
                    double max_cor_mz = 0;
                    double max_frag_cor = -100;
                    for (double mz : peptideMatch.mz2cor.keySet()) {
                        if (peptideMatch.mz2cor.get(mz) >= this.cor_cutoff) {
                            high_cor_mzs.add(mz);
                        }
                        if (peptideMatch.mz2cor.get(mz) > max_frag_cor) {
                            max_frag_cor = peptideMatch.mz2cor.get(mz);
                            max_cor_mz = mz;
                        }
                    }
                    peptideMatch.max_cor_mz = max_cor_mz;
                    for (double mz : peptideMatch.mz2index.keySet()) {
                        if (!high_cor_mzs.contains(mz)) {
                            ind = peptideMatch.mz2index.get(mz);
                            peptideMatch.ion_matrix[ind[0]][ind[1]] = peptideMatch.ion_matrix[ind[0]][ind[1]] + 1;
                        }
                    }
                }

                // low mass fragment ions
                // based on fragment ion m/z or ion number (such as b-1, b-2, y-1, y-2)
                row_i = -1;
                for (String line : isoWinID2PSMs.get(isoWinID)) {
                    String[] d = line.split("\t");
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
                    String[] d = line.split("\t");
                    if (index2peptideMatch.get(row_i).max_fragment_ion_intensity > 0
                            && index2peptideMatch.get(row_i).matched_ions.size() >= this.min_n_fragment_ions) {
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
                                if (this.export_spectra_to_mgf) {
                                    msWriter.write(
                                            MgfUtils.asMgf(spectrum, spectrum_title, charge, String.valueOf(apex_scan))
                                                    + "\n");
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
                                for (IonMatch ionMatch : peptideMatch.matched_ions) {
                                    this.fragment_ions_mz_tol.add(ionMatch.getError(is_fragment_ion_tolu_ppm));
                                }
                            }
                        }
                    } else {
                        n_total_matches_max_fragment_ion_invalid++;
                    }

                }
            }
        }

        psmWriter.close();
        msWriter.close();
        fragWriter.close();
        fragValidWriter.close();
        if (this.export_fragment_ion_mz_to_file && fragMzWriter != null) {
            fragMzWriter.close();
        }
        if (this.export_skyline_transition_list_file && tfWriter != null && tbWriter != null) {
            tfWriter.close();
            tbWriter.close();
        }

        System.out.println("Total matches:"+n_total_matches);
        System.out.println("Total valid matches:"+n_total_matches_valid);
        System.out.println("Total matches with invalid max fragment ion intensity:"+n_total_matches_max_fragment_ion_invalid);

        generate_rt_train_data(peptide2rt, rt_merge_method, this.out_dir + "/rt_train_data.tsv");
        CParameter.fragment_ion_intensity_cutoff = original_fragment_ion_intensity_cutoff;

    }

    /**
     * Generate training data for MS2 and retention time prediction based on DIA-NN
     * search result.
     * It writes the results to output files for model training.
     * 
     * @throws IOException If there is an error reading or writing files.
     */
    public void get_ms2_matches_diann() throws IOException {
        CModification.getInstance();
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
        HashMap<String, PeptideRT> peptide2rt = new HashMap<>();

        // for CCS
        HashMap<String, PeptideCCS> peptide2ccs = new HashMap<>();

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
        if (this.export_fragment_ion_mz_to_file) {
            fragMzWriter = new BufferedWriter(new FileWriter(this.out_dir + "/fragment_mz.tsv"));
            fragMzWriter.write(this.fragment_ion_intensity_head_line + "\n");
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

        if (this.export_skyline_transition_list_file) {

            tfWriter = new BufferedWriter(new FileWriter(this.out_dir+"/skyline_input.tsv"));
            tfWriter.write("Peptide\tPrecursor m/z\tProduct m/z\tLibraryIntensity\tExplicit Retention Time\tExplicit Retention Time Window\tNote\n");

            // peak boundary file
            tbWriter = new BufferedWriter(new FileWriter(this.out_dir + "/skyline_boundary.tsv"));
            tbWriter.write("MinStartTime\tMaxEndTime\tFileName\tPeptideModifiedSequence\tPrecursorCharge\n");

        }

        BufferedWriter xicWriter = null;
        boolean first_xic = true;
        if (export_xic) {
            xicWriter = new BufferedWriter(new FileWriter(this.out_dir + "/xic.json"));
            xicWriter.write("{\n");
        }

        // meta information about the MS data and model training
        BufferedWriter metaWriter = new BufferedWriter(new FileWriter(this.out_dir + "/meta.json"));
        metaWriter.write("{\n");

        int psm_id = 0;

        HashMap<String, JMeta> ms_file2meta = new HashMap<>();
        boolean first_meta = true;

        // useful when there are multiple MS files and they may have different precursor
        // m/z ranges.
        double global_minPeptideMz = Double.POSITIVE_INFINITY;
        double global_maxPeptideMz = 0.0;

        for (String ms_file : this.ms_file2psm.keySet()) {
            System.out.println("Process MS file:" + ms_file);
            ms_file2meta.put(ms_file, new JMeta());
            ms_file2meta.get(ms_file).ms_file = ms_file;
            // For store raw data
            DIAMeta meta = new DIAMeta();
            if (CParameter.itol > 0.2 && CParameter.itolu.startsWith("da")) {
                meta.fragment_ion_mz_bin_size = 0.5;
                System.out.println("Fragment ion bin size:" + meta.fragment_ion_mz_bin_size);
            }
            meta.load_ms_data(ms_file);
            meta.get_ms_run_meta_data();
            if (this.rt_max > meta.rt_max) {
                meta.rt_max = this.rt_max;
                System.out.println("Use user-provided RT max:" + this.rt_max);
            }
            CParameter.minPeptideMz = meta.precursor_ion_mz_min - 0.5;
            CParameter.maxPeptideMz = meta.precursor_ion_mz_max + 0.5;

            if (global_minPeptideMz >= CParameter.minPeptideMz) {
                global_minPeptideMz = CParameter.minPeptideMz;
            }

            if (global_maxPeptideMz <= CParameter.maxPeptideMz) {
                global_maxPeptideMz = CParameter.maxPeptideMz;
            }

            CParameter.min_fragment_ion_mz = meta.fragment_ion_mz_min - 0.5;
            if (CParameter.max_fragment_ion_mz > meta.fragment_ion_mz_max) {
                CParameter.max_fragment_ion_mz = meta.fragment_ion_mz_max + 0.5;
            }
            CParameter.NCE = meta.nce;
            this.nce = meta.nce;
            System.out.println("NCE:" + CParameter.NCE);
            String ms_instrument_name = meta.get_ms_instrument(ms_file);
            if (!ms_instrument_name.isEmpty()) {
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
            ms_file2meta.get(ms_file).precursor_ion_mz_min = meta.precursor_ion_mz_min;
            ms_file2meta.get(ms_file).precursor_ion_mz_max = meta.precursor_ion_mz_max;
            if (first_meta) {
                metaWriter.write("\"" + ms_file + "\":" + JSON.toJSONString(ms_file2meta.get(ms_file)));
                first_meta = false;
            } else {
                metaWriter.write(",\n\"" + ms_file + "\":" + JSON.toJSONString(ms_file2meta.get(ms_file)));
            }

            // "DIA-NN scan numbers start with 0. And MS2 scans are numbered one after another, the numbering for MS1 ones is separate. That is the first MS2 scan has number 0, and the first MS1 scan also has number 0."
            // https://github.com/vdemichev/DiaNN/discussions/211
            HashMap<Integer, Integer> global_index2scan_num = new HashMap<>(meta.num2scanMap.size());
            int global_index = 0;
            for (int scan_num : meta.num2scanMap.keySet()) {
                if (meta.num2scanMap.get(scan_num).getMsLevel() == 2) {
                    global_index2scan_num.put(global_index, meta.num2scanMap.get(scan_num).getNum());
                    global_index++;
                }
            }
            System.out.println("Max index:" + global_index);

            this.min_fragment_ion_mz = meta.fragment_ion_mz_min;
            this.max_fragment_ion_mz = meta.fragment_ion_mz_max;
            System.out.println("Fragment ion m/z range:" + this.min_fragment_ion_mz + "," + this.max_fragment_ion_mz);

            DIAMap diaMap_tmp = new DIAMap();
            diaMap_tmp.meta = meta;
            if (this.target_isolation_wins.isEmpty()) {
                diaMap_tmp.target_isolation_wins.addAll(meta.isolationWindowMap.keySet());
            } else {
                diaMap_tmp.target_isolation_wins.addAll(this.target_isolation_wins);
            }

            if (meta.rt_max > this.rt_max) {
                this.rt_max = meta.rt_max;
                System.out.println("RT max:" + this.rt_max);
            } else {
                System.out.println("RT max:" + this.rt_max);
            }

            // for output
            HashSet<String> save_spectra = new HashSet<>();

            HashMap<String, ArrayList<String>> isoWinID2PSMs = new HashMap<>();

            boolean show_mod_ai_only_one_time = true;

            // for un-recognized PSMs: for example, no MS2 mapped.
            HashMap<String, Integer> un_recognized_PSMs = new HashMap<>();
            for (String line : this.ms_file2psm.get(ms_file)) {
                String[] d = line.split("\t");
                //
                String peptide = d[hIndex.get(PSMConfig.stripped_peptide_sequence_column_name)];
                String modification = this.get_modification_diann(d[hIndex.get(PSMConfig.peptide_modification_column_name)],peptide);
                int precursor_charge = Integer.parseInt(d[hIndex.get(PSMConfig.precursor_charge_column_name)]);
                this.add_peptide(peptide,modification);
                ArrayList<String> isoWinIDs = diaMap_tmp.get_isolation_windows(dbGear.get_mz(this.get_peptide(peptide,modification).getMass(),precursor_charge));
                if (isoWinIDs.isEmpty()){
                    System.out.println("Isolation window ID is empty:"+line);
                    continue;
                }
                for (String isoWinID : isoWinIDs) {
                    if (!isoWinID2PSMs.containsKey(isoWinID)) {
                        isoWinID2PSMs.put(isoWinID, new ArrayList<>());
                    }
                    isoWinID2PSMs.get(isoWinID).add(line);

                    String peptide_mod = peptide + "_" + modification;

                    if (this.mod_ai.equalsIgnoreCase("-") || this.mod_ai.equalsIgnoreCase("general")) {
                        if (show_mod_ai_only_one_time) {
                            Cloger.getInstance().logger.info("Training data generation for general modeling!");
                            show_mod_ai_only_one_time = false;
                        }
                    } else if (this.mod_ai.equalsIgnoreCase("phosphorylation")) {
                        String mod_seq = d[hIndex.get(PSMConfig.peptide_modification_column_name)];
                        if (show_mod_ai_only_one_time) {
                            Cloger.getInstance().logger.info("Training data generation for phosphorylation modeling!");
                            show_mod_ai_only_one_time = false;
                        }
                        if (hIndex.containsKey(PSMConfig.ptm_site_confidence_column_name)
                                && mod_seq.contains("UniMod:21")) {
                            // only filtering out low confidence phosphorylation peptides
                            if (Double.parseDouble(d[hIndex.get(PSMConfig.ptm_site_confidence_column_name)]) < this.ptm_site_prob_cutoff) {
                                continue;
                            }
                            if (Double.parseDouble(d[hIndex.get(PSMConfig.ptm_site_qvalue_column_name)]) > this.ptm_site_qvalue_cutoff) {
                                continue;
                            }
                        }
                    } else {
                        System.err.println("Modification type is not supported:" + this.mod_ai);
                        System.exit(1);
                    }

                    if (!peptide2rt.containsKey(peptide_mod)) {
                        peptide2rt.put(peptide_mod, new PeptideRT());
                    }
                    peptide2rt.get(peptide_mod).peptide = peptide;
                    peptide2rt.get(peptide_mod).modification = modification;
                    peptide2rt.get(peptide_mod).rts.add(Double.parseDouble(d[hIndex.get(PSMConfig.rt_column_name)])); // Apex RT
                    peptide2rt.get(peptide_mod).scores.add(Double.parseDouble(d[hIndex.get(PSMConfig.qvalue_column_name)]));

                    // for CCS
                    if (ccs_enabled) {
                        String peptide_mode_charge = peptide + "_" + modification + "_" + precursor_charge;
                        if (!peptide2ccs.containsKey(peptide_mode_charge)) {
                            peptide2ccs.put(peptide_mode_charge, new PeptideCCS());
                        }
                        peptide2ccs.get(peptide_mode_charge).peptide = peptide;
                        peptide2ccs.get(peptide_mode_charge).modification = modification;
                        // In DIA-NN, iIM refers to the reference ion mobility in the spectral library, IM refers to the empirically measured.
                        peptide2ccs.get(peptide_mode_charge).ccs_values.add(Double.parseDouble(d[hIndex.get(PSMConfig.im_column_name)]));
                        peptide2ccs.get(peptide_mode_charge).scores.add(Double.parseDouble(d[hIndex.get(PSMConfig.qvalue_column_name)]));
                    }
                }
            }

            for (String isoWinID : isoWinID2PSMs.keySet()) {
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
                    String peptide = d[hIndex.get(PSMConfig.stripped_peptide_sequence_column_name)];
                    String modification = this.get_modification_diann(d[hIndex.get(PSMConfig.peptide_modification_column_name)],peptide);
                    int precursor_charge = Integer.parseInt(d[hIndex.get(PSMConfig.precursor_charge_column_name)]);
                    // double apex_rt = Double.parseDouble(d[hIndex.get("apex_rt")]);
                    double rt_start = Double.parseDouble(d[hIndex.get(PSMConfig.rt_start_column_name)]);
                    double rt_end = Double.parseDouble(d[hIndex.get(PSMConfig.rt_end_column_name)]);

                    int apex_scan = global_index2scan_num.get(Integer.parseInt(d[hIndex.get(PSMConfig.ms2_index_column_name)])); // index
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
                    index2peptideMatch.get(row_i).rt_apex = Double.parseDouble(d[hIndex.get(PSMConfig.rt_column_name)]);
                    index2peptideMatch.get(row_i).peptide_length = peptide.length();
                    index2peptideMatch.get(row_i).precursor_charge = precursor_charge;
                    index2peptideMatch.get(row_i).index = Integer.parseInt(d[hIndex.get(PSMConfig.ms2_index_column_name)]);
                    index2peptideMatch.get(row_i).peptide = peptideObj;

                    // for testing
                    if(test_mode){
                        index2peptideMatch.get(row_i).ion_matrix_map.put("spectrum_centric", new int[peptide.length() - 1][n_ion_types]);
                        index2peptideMatch.get(row_i).ion_matrix_map.put("peptide_centric_cor", new int[peptide.length() - 1][n_ion_types]);
                        index2peptideMatch.get(row_i).ion_matrix_map.put("peptide_centric_shape", new int[peptide.length() - 1][n_ion_types]);
                        index2peptideMatch.get(row_i).ion_matrix_map.put("low_mass", new int[peptide.length() - 1][n_ion_types]);
                        index2peptideMatch.get(row_i).ion_matrix_map.put("peptide_centric", new int[peptide.length() - 1][n_ion_types]);
                    }

                    if (spectrum == null || spectrum.getNPeaks() == 0) {
                        if (!un_recognized_PSMs.containsKey(line)) {
                            un_recognized_PSMs.put(line, 0);
                        }
                        continue;
                    } else {
                        un_recognized_PSMs.put(line, 1);
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
                            if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION
                                    || ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION) {
                                // add fragment ion number
                                PeptideFragmentIon fragmentIon = ((PeptideFragmentIon) ionMatch.ion);
                                int ion_number = fragmentIon.getNumber();
                                int ion_type_column_index = this.get_ion_type_column_index(ionMatch);
                                // for y ion
                                if (ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION) {
                                    fragment_ion_row_index = peptide.length() - ion_number - 1;
                                    ion_type = "y";
                                } else if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION) {
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

                                // If the fragment ion number is <= the minimum number of fragment ion used for
                                // spectral library generation,
                                // we don't consider it in getting the max intensity of fragment ions.
                                if (use_all_peaks || (max_fragment_ion_intensity <= ionMatch.peakIntensity
                                        && ion_number >= this.lf_frag_n_min)) {
                                    max_fragment_ion_intensity = ionMatch.peakIntensity;
                                    max_fragment_ion_row_index = fragment_ion_row_index;
                                    max_fragment_ion_column_index = ion_type_column_index;
                                }
                            }
                        }
                    }
                    if (!matched_ion_mzs.isEmpty()) {
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
                int[] ind = new int[] { 0, 0 };
                for (String line : isoWinID2PSMs.get(isoWinID)) {
                    row_i = row_i + 1;
                    PeptideMatch peptideMatch = index2peptideMatch.get(row_i);
                    HashSet<Double> high_cor_mzs = new HashSet<>();
                    double max_cor_mz = 0;
                    double max_frag_cor = -100;
                    for (double mz : peptideMatch.mz2cor.keySet()) {
                        if (peptideMatch.mz2cor.get(mz) >= this.cor_cutoff) {
                            high_cor_mzs.add(mz);
                        }
                        if (peptideMatch.mz2cor.get(mz) > max_frag_cor) {
                            max_frag_cor = peptideMatch.mz2cor.get(mz);
                            max_cor_mz = mz;
                        }
                    }
                    peptideMatch.max_cor_mz = max_cor_mz;
                    for (double mz : peptideMatch.mz2index.keySet()) {
                        if (!high_cor_mzs.contains(mz)) {
                            ind = peptideMatch.mz2index.get(mz);
                            peptideMatch.ion_matrix[ind[0]][ind[1]] = peptideMatch.ion_matrix[ind[0]][ind[1]] + 1;
                            if (test_mode) {
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
                    for (double mz : peptideMatch.mz2index.keySet()) {
                        if (peptideMatch.mz2skewed_peaks.containsKey(mz) && peptideMatch.mz2skewed_peaks.get(mz) >= 2) {
                            ind = peptideMatch.mz2index.get(mz);
                            peptideMatch.ion_matrix[ind[0]][ind[1]] = peptideMatch.ion_matrix[ind[0]][ind[1]] + 1;
                            if (test_mode) {
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
                    String[] d = line.split("\t");
                    row_i = row_i + 1;
                    // only need to return +1 fragment ion here
                    HashMap<Integer, ArrayList<Ion>> theoretical_ions = this.generate_theoretical_fragment_ions(this.get_peptide(d[hIndex.get(PSMConfig.stripped_peptide_sequence_column_name)],
                                    this.get_modification_diann(d[hIndex.get(PSMConfig.peptide_modification_column_name)],d[hIndex.get(PSMConfig.stripped_peptide_sequence_column_name)])),
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
                                } else if (ion.getSubType() == PeptideFragmentIon.B_ION) {
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
                    String[] d = line.split("\t");
                    if (hIndex.containsKey("peak_overlap")) {
                        if (Integer.parseInt(d[hIndex.get("peak_overlap")]) >= 1) {
                            n_peak_overlap++;
                            continue;
                        }
                    }

                    if (this.mod_ai.equalsIgnoreCase("-") || this.mod_ai.equalsIgnoreCase("general")) {
                        // nothing to do
                    } else if (this.mod_ai.equalsIgnoreCase("phosphorylation")) {
                        String mod_seq = d[hIndex.get(PSMConfig.peptide_modification_column_name)];
                        if (hIndex.containsKey(PSMConfig.ptm_site_confidence_column_name) && mod_seq.contains("UniMod:21")) {
                            // only filtering out low confidence phosphorylation peptides
                            if(Double.parseDouble(d[hIndex.get(PSMConfig.ptm_site_confidence_column_name)]) < this.ptm_site_prob_cutoff){
                                n_ptm_site_low_confidence++;
                                continue;
                            }
                            if (Double.parseDouble(d[hIndex.get(PSMConfig.ptm_site_qvalue_column_name)]) > this.ptm_site_qvalue_cutoff) {
                                n_ptm_site_low_confidence++;
                                continue;
                            }
                        }
                    } else {
                        System.err.println("Modification type is not supported:" + this.mod_ai);
                        System.exit(1);
                    }


                    if(index2peptideMatch.get(row_i).max_fragment_ion_intensity>0 && index2peptideMatch.get(row_i).matched_ions.size()>=this.min_n_fragment_ions) {
                        boolean fragment_export = false;

                        String [] out_mod = convert_modification(this.get_modification_diann(d[hIndex.get(PSMConfig.peptide_modification_column_name)],d[hIndex.get(PSMConfig.stripped_peptide_sequence_column_name)]));
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

                            String spectrum_title = d[hIndex.get(PSMConfig.ms2_index_column_name)];
                            double pdv_precursor_mz = dbGear.get_mz(this.get_peptide(d[hIndex.get(PSMConfig.stripped_peptide_sequence_column_name)], this.get_modification_diann(d[hIndex.get(PSMConfig.peptide_modification_column_name)],d[hIndex.get(PSMConfig.stripped_peptide_sequence_column_name)])).getMass(),
                                    Integer.parseInt(d[hIndex.get(PSMConfig.precursor_charge_column_name)]));
                            String pdv_precursor_charge = d[hIndex.get(PSMConfig.precursor_charge_column_name)];
                            String pdv_peptide = d[hIndex.get(PSMConfig.stripped_peptide_sequence_column_name)];
                            String pdv_modification = this.get_modification_diann(d[hIndex.get(PSMConfig.peptide_modification_column_name)],d[hIndex.get(PSMConfig.stripped_peptide_sequence_column_name)]);
                            // true || true
                            if (index2peptideMatch.get(row_i).is_max_fragment_ion_intensity_valid() || use_all_peaks) {
                                n_total_matches_valid++;
                                n_total_psm_matches_valid++;
                                frag_start_idx = frag_stop_idx;
                                frag_stop_idx = frag_start_idx + index2peptideMatch.get(row_i).ion_intensity_matrix.length;
                                psmWriter.write(index2peptideMatch.get(row_i).id+"\t"+spectrum_title+"\t"+index2peptideMatch.get(row_i).scan+ "\t" +pdv_precursor_mz +"\t" +pdv_precursor_charge +"\t" +pdv_peptide + "\t" +pdv_modification+  "\t" + out_mod[0] + "\t" + out_mod[1] + "\t1\t" + index2peptideMatch.get(row_i).max_cor_mz + "\t" + frag_start_idx + "\t" + frag_stop_idx +
                                        "\t" + n_valid_fragment_ions + "\t" + n_total_fragment_ions + "\t1\n");
                                if (!pMatches.isEmpty()) {
                                    for (PeptideMatch pMatch : pMatches) {
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

                            int apex_scan = global_index2scan_num.get(Integer.parseInt(d[hIndex.get(PSMConfig.ms2_index_column_name)]));
                            // String spectrum_title = d[hIndex.get("MS2.Scan")];
                            if (!save_spectra.contains(spectrum_title)) {
                                Spectrum spectrum = diaIndex.get_spectrum_by_scan(apex_scan);
                                int charge = Integer.parseInt(d[hIndex.get(PSMConfig.precursor_charge_column_name)]);
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
                                if (!pMatches.isEmpty()) {
                                    for (PeptideMatch pMatch : pMatches) {
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
                                    if (test_mode) {
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
                                if (!pMatches.isEmpty()) {
                                    for (PeptideMatch pMatch : pMatches) {
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

                                    double precursor_mz = dbGear.get_mz(this.get_peptide(d[hIndex.get(PSMConfig.stripped_peptide_sequence_column_name)], this.get_modification_diann(d[hIndex.get(PSMConfig.peptide_modification_column_name)],d[hIndex.get(PSMConfig.stripped_peptide_sequence_column_name)])).getMass(),
                                            Integer.parseInt(d[hIndex.get(PSMConfig.precursor_charge_column_name)]));

                                    tbWriter.write(index2peptideMatch.get(row_i).rt_start + "\t" + index2peptideMatch.get(row_i).rt_end + "\t" + ms_file + "\t" +
                                            ModificationUtils.getInstance().getSkylineFormatPeptide(this.get_peptide(d[hIndex.get(PSMConfig.stripped_peptide_sequence_column_name)], this.get_modification_diann(d[hIndex.get(PSMConfig.peptide_modification_column_name)],d[hIndex.get(PSMConfig.stripped_peptide_sequence_column_name)]))) + "\t" + d[hIndex.get(PSMConfig.precursor_charge_column_name)] + "\n");

                                    PeptideMatch peptideMatch = index2peptideMatch.get(row_i);
                                    for (double mz : peptideMatch.mz2cor.keySet()) {
                                        int[] ind_mz = peptideMatch.mz2index.get(mz);
                                        tfWriter.write(ModificationUtils.getInstance().getSkylineFormatPeptide(this.get_peptide(d[hIndex.get(PSMConfig.stripped_peptide_sequence_column_name)], this.get_modification_diann(d[hIndex.get(PSMConfig.peptide_modification_column_name)],d[hIndex.get(PSMConfig.stripped_peptide_sequence_column_name)]))) +
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
                                for (IonMatch ionMatch : peptideMatch.matched_ions) {
                                    this.fragment_ions_mz_tol.add(ionMatch.getError(is_fragment_ion_tolu_ppm));
                                }
                            }
                        } else {
                            n_less_than_min_n_high_quality_fragment_ions++;
                        }
                    } else {
                        n_less_than_min_n_fragment_ions++;
                    }

                }
            }
            if (!un_recognized_PSMs.isEmpty()) {
                int n_un_recognized_PSMs = 0;
                for (String line : un_recognized_PSMs.keySet()) {
                    if (un_recognized_PSMs.get(line) == 0) {
                        n_un_recognized_PSMs++;
                        System.out.println("Spectrum not found:" + line);
                    }
                }
                if (n_un_recognized_PSMs >= 1) {
                    System.out.println("Spectrum not found:" + n_un_recognized_PSMs);
                }
            }
        }

        psmWriter.close();
        msWriter.close();
        fragWriter.close();
        fragValidWriter.close();
        if (this.export_fragment_ion_mz_to_file && fragMzWriter != null) {
            fragMzWriter.close();
        }
        if (this.export_skyline_transition_list_file && tfWriter != null && tbWriter != null) {
            tfWriter.close();
            tbWriter.close();
        }

        if (export_xic) {
            xicWriter.write("\n}");
            xicWriter.close();
        }

        if (test_mode) {
            sp_fragValidWriter.close();
            pep_cor_fragValidWriter.close();
            pep_shape_fragValidWriter.close();
            pep_fragValidWriter.close();
        }

        metaWriter.write("\n}");
        metaWriter.close();

        if (this.ms_file2psm.size() >= 2) {
            CParameter.minPeptideMz = global_minPeptideMz;
            CParameter.maxPeptideMz = global_maxPeptideMz;
        }

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
        generate_rt_train_data(peptide2rt, rt_merge_method, this.out_dir + "/rt_train_data.tsv");
        CParameter.fragment_ion_intensity_cutoff = original_fragment_ion_intensity_cutoff;

    }

    /**
     * Generate training data for MS2 and retention time prediction without
     * interference detection based on DIA-NN search result.
     * This should be only used for testing purpose.
     * It writes the results to output files for model training.
     * 
     * @throws IOException If there is an error reading or writing files.
     */
    public void get_ms2_matches_diann_dda() throws IOException {
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
        HashMap<String, PeptideRT> peptide2rt = new HashMap<>();

        // for CCS
        HashMap<String, PeptideCCS> peptide2ccs = new HashMap<>();

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
        if (this.export_fragment_ion_mz_to_file) {
            fragMzWriter = new BufferedWriter(new FileWriter(this.out_dir + "/fragment_mz.tsv"));
            fragMzWriter.write(this.fragment_ion_intensity_head_line + "\n");
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

        if (this.export_skyline_transition_list_file) {

            tfWriter = new BufferedWriter(new FileWriter(this.out_dir+"/skyline_input.tsv"));
            tfWriter.write("Peptide\tPrecursor m/z\tProduct m/z\tLibraryIntensity\tExplicit Retention Time\tExplicit Retention Time Window\tNote\n");

            // peak boundary file
            tbWriter = new BufferedWriter(new FileWriter(this.out_dir + "/skyline_boundary.tsv"));
            tbWriter.write("MinStartTime\tMaxEndTime\tFileName\tPeptideModifiedSequence\tPrecursorCharge\n");

        }

        BufferedWriter xicWriter = null;
        boolean first_xic = true;
        if (export_xic) {
            xicWriter = new BufferedWriter(new FileWriter(this.out_dir + "/xic.json"));
            xicWriter.write("{\n");
        }

        // meta information about the MS data and model training
        BufferedWriter metaWriter = new BufferedWriter(new FileWriter(this.out_dir + "/meta.json"));
        metaWriter.write("{\n");

        int psm_id = 0;

        HashMap<String, JMeta> ms_file2meta = new HashMap<>();
        boolean first_meta = true;

        for (String ms_file : this.ms_file2psm.keySet()) {
            System.out.println("Process MS file:" + ms_file);
            ms_file2meta.put(ms_file, new JMeta());
            ms_file2meta.get(ms_file).ms_file = ms_file;
            // For store raw data
            DIAMeta meta = new DIAMeta();
            if (CParameter.itol > 0.2 && CParameter.itolu.startsWith("da")) {
                meta.fragment_ion_mz_bin_size = 0.5;
                System.out.println("Fragment ion bin size:" + meta.fragment_ion_mz_bin_size);
            }
            meta.load_ms_data(ms_file);
            meta.get_ms_run_meta_data();
            if (this.rt_max > meta.rt_max) {
                meta.rt_max = this.rt_max;
                System.out.println("Use user-provided RT max:" + this.rt_max);
            }
            CParameter.minPeptideMz = meta.precursor_ion_mz_min - 0.5;
            CParameter.maxPeptideMz = meta.precursor_ion_mz_max + 0.5;
            CParameter.min_fragment_ion_mz = meta.fragment_ion_mz_min - 0.5;
            if (CParameter.max_fragment_ion_mz > meta.fragment_ion_mz_max) {
                CParameter.max_fragment_ion_mz = meta.fragment_ion_mz_max + 0.5;
            }
            CParameter.NCE = meta.nce;
            this.nce = meta.nce;
            System.out.println("NCE:" + CParameter.NCE);
            String ms_instrument_name = meta.get_ms_instrument(ms_file);
            if (!ms_instrument_name.isEmpty()) {
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
            ms_file2meta.get(ms_file).precursor_ion_mz_min = meta.precursor_ion_mz_min;
            ms_file2meta.get(ms_file).precursor_ion_mz_max = meta.precursor_ion_mz_max;
            if (first_meta) {
                metaWriter.write("\"" + ms_file + "\":" + JSON.toJSONString(ms_file2meta.get(ms_file)));
                first_meta = false;
            } else {
                metaWriter.write(",\n\"" + ms_file + "\":" + JSON.toJSONString(ms_file2meta.get(ms_file)));
            }

            // "DIA-NN scan numbers start with 0. And MS2 scans are numbered one after another, the numbering for MS1 ones is separate. That is the first MS2 scan has number 0, and the first MS1 scan also has number 0."
            // https://github.com/vdemichev/DiaNN/discussions/211
            HashMap<Integer, Integer> global_index2scan_num = new HashMap<>(meta.num2scanMap.size());
            int global_index = 0;
            for (int scan_num : meta.num2scanMap.keySet()) {
                if (meta.num2scanMap.get(scan_num).getMsLevel() == 2) {
                    global_index2scan_num.put(global_index, meta.num2scanMap.get(scan_num).getNum());
                    global_index++;
                }
            }
            System.out.println("Max index:" + global_index);

            this.min_fragment_ion_mz = meta.fragment_ion_mz_min;
            this.max_fragment_ion_mz = meta.fragment_ion_mz_max;
            System.out.println("Fragment ion m/z range:" + this.min_fragment_ion_mz + "," + this.max_fragment_ion_mz);

            DIAMap diaMap_tmp = new DIAMap();
            diaMap_tmp.meta = meta;
            if (this.target_isolation_wins.isEmpty()) {
                diaMap_tmp.target_isolation_wins.addAll(meta.isolationWindowMap.keySet());
            } else {
                diaMap_tmp.target_isolation_wins.addAll(this.target_isolation_wins);
            }

            if (meta.rt_max > this.rt_max) {
                this.rt_max = meta.rt_max;
                System.out.println("RT max:" + this.rt_max);
            } else {
                System.out.println("RT max:" + this.rt_max);
            }

            // for output
            HashSet<String> save_spectra = new HashSet<>();

            HashMap<String, ArrayList<String>> isoWinID2PSMs = new HashMap<>();

            boolean show_mod_ai_only_one_time = true;

            // for un-recognized PSMs: for example, no MS2 mapped.
            HashMap<String, Integer> un_recognized_PSMs = new HashMap<>();
            for (String line : this.ms_file2psm.get(ms_file)) {
                String[] d = line.split("\t");
                //
                String peptide = d[hIndex.get("Stripped.Sequence")];
                String modification = this.get_modification_diann(d[hIndex.get("Modified.Sequence")], peptide);
                int precursor_charge = Integer.parseInt(d[hIndex.get("Precursor.Charge")]);
                this.add_peptide(peptide,modification);
                ArrayList<String> isoWinIDs = diaMap_tmp.get_isolation_windows(dbGear.get_mz(this.get_peptide(peptide,modification).getMass(),precursor_charge));
                if (isoWinIDs.isEmpty()){
                    System.out.println("Isolation window ID is empty:"+line);
                    continue;
                }
                for (String isoWinID : isoWinIDs) {
                    if (!isoWinID2PSMs.containsKey(isoWinID)) {
                        isoWinID2PSMs.put(isoWinID, new ArrayList<>());
                    }
                    isoWinID2PSMs.get(isoWinID).add(line);

                    String peptide_mod = peptide + "_" + modification;

                    if (this.mod_ai.equalsIgnoreCase("-") || this.mod_ai.equalsIgnoreCase("general")) {
                        if (show_mod_ai_only_one_time) {
                            Cloger.getInstance().logger.info("Training data generation for general modeling!");
                            show_mod_ai_only_one_time = false;
                        }
                    } else if (this.mod_ai.equalsIgnoreCase("phosphorylation")) {
                        String mod_seq = d[hIndex.get("Modified.Sequence")];
                        if (show_mod_ai_only_one_time) {
                            Cloger.getInstance().logger.info("Training data generation for phosphorylation modeling!");
                            show_mod_ai_only_one_time = false;
                        }
                        if (hIndex.containsKey("PTM.Site.Confidence") && mod_seq.contains("UniMod:21")) {
                            // only filtering out low confidence phosphorylation peptides
                            if (Double.parseDouble(d[hIndex.get("PTM.Site.Confidence")]) < this.ptm_site_prob_cutoff) {
                                continue;
                            }
                            if (Double.parseDouble(d[hIndex.get("PTM.Q.Value")]) > this.ptm_site_qvalue_cutoff) {
                                continue;
                            }
                        }
                    } else {
                        System.err.println("Modification type is not supported:" + this.mod_ai);
                        System.exit(1);
                    }

                    if (!peptide2rt.containsKey(peptide_mod)) {
                        peptide2rt.put(peptide_mod, new PeptideRT());
                    }
                    peptide2rt.get(peptide_mod).peptide = peptide;
                    peptide2rt.get(peptide_mod).modification = modification;
                    peptide2rt.get(peptide_mod).rts.add(Double.parseDouble(d[hIndex.get("RT")])); // Apex RT
                    peptide2rt.get(peptide_mod).scores.add(Double.parseDouble(d[hIndex.get("Q.Value")]));

                    // for CCS
                    if (ccs_enabled) {
                        String peptide_mode_charge = peptide + "_" + modification + "_" + precursor_charge;
                        if (!peptide2ccs.containsKey(peptide_mode_charge)) {
                            peptide2ccs.put(peptide_mode_charge, new PeptideCCS());
                        }
                        peptide2ccs.get(peptide_mode_charge).peptide = peptide;
                        peptide2ccs.get(peptide_mode_charge).modification = modification;
                        // In DIA-NN, iIM refers to the reference ion mobility in the spectral library, IM refers to the empirically measured.
                        peptide2ccs.get(peptide_mode_charge).ccs_values.add(Double.parseDouble(d[hIndex.get("IM")]));
                        peptide2ccs.get(peptide_mode_charge).scores.add(Double.parseDouble(d[hIndex.get("Q.Value")]));
                    }
                }
            }

            for (String isoWinID : isoWinID2PSMs.keySet()) {
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
                    String modification = this.get_modification_diann(d[hIndex.get("Modified.Sequence")], peptide);
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

                    if (spectrum == null) {
                        if (!un_recognized_PSMs.containsKey(line)) {
                            un_recognized_PSMs.put(line, 0);
                        }
                        continue;
                    } else {
                        un_recognized_PSMs.put(line, 1);
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
                            if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION
                                    || ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION) {
                                // add fragment ion number
                                PeptideFragmentIon fragmentIon = ((PeptideFragmentIon) ionMatch.ion);
                                int ion_number = fragmentIon.getNumber();
                                int ion_type_column_index = this.get_ion_type_column_index(ionMatch);
                                // for y ion
                                if (ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION) {
                                    fragment_ion_row_index = peptide.length() - ion_number - 1;
                                    ion_type = "y";
                                } else if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION) {
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
                                if (use_all_peaks || (max_fragment_ion_intensity <= ionMatch.peakIntensity
                                        && ion_number >= this.lf_frag_n_min)) {
                                    max_fragment_ion_intensity = ionMatch.peakIntensity;
                                    max_fragment_ion_row_index = fragment_ion_row_index;
                                    max_fragment_ion_column_index = ion_type_column_index;
                                }
                            }
                        }
                    }
                    if (!matched_ion_mzs.isEmpty()) {
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

                // only extract XIC
                refine_peak_boundary = false;
                index2peptideMatch.values().parallelStream().forEach(peptideMatch -> xic_query(diaIndex,peptideMatch,isoWinID));

                // output
                row_i = -1;
                for (String line : isoWinID2PSMs.get(isoWinID)) {
                    n_total_matches++;
                    row_i = row_i + 1;
                    String[] d = line.split("\t");

                    if (this.mod_ai.equalsIgnoreCase("-") || this.mod_ai.equalsIgnoreCase("general")) {
                        // nothing to do
                    } else if (this.mod_ai.equalsIgnoreCase("phosphorylation")) {
                        String mod_seq = d[hIndex.get("Modified.Sequence")];
                        if (hIndex.containsKey("PTM.Site.Confidence") && mod_seq.contains("UniMod:21")) {
                            // only filtering out low confidence phosphorylation peptides
                            if (Double.parseDouble(d[hIndex.get("PTM.Site.Confidence")]) < this.ptm_site_prob_cutoff) {
                                n_ptm_site_low_confidence++;
                                continue;
                            }
                            if (Double.parseDouble(d[hIndex.get("PTM.Q.Value")]) > this.ptm_site_qvalue_cutoff) {
                                n_ptm_site_low_confidence++;
                                continue;
                            }
                        }
                    } else {
                        System.err.println("Modification type is not supported:" + this.mod_ai);
                        System.exit(1);
                    }

                    // TODO: need to determine if this filter is necessary: index2peptideMatch.get(row_i).matched_ions.size()>=this.min_n_fragment_ions
                    if(index2peptideMatch.get(row_i).max_fragment_ion_intensity>0 && index2peptideMatch.get(row_i).matched_ions.size()>=this.min_n_fragment_ions) {
                        boolean fragment_export = false;

                        String [] out_mod = convert_modification(this.get_modification_diann(d[hIndex.get("Modified.Sequence")],d[hIndex.get("Stripped.Sequence")]));
                        int n_valid_fragment_ions = get_n_valid_fragment_ions(index2peptideMatch.get(row_i).ion_intensity_matrix,index2peptideMatch.get(row_i).ion_matrix);
                        int n_total_fragment_ions = get_n_matched_fragment_ions(index2peptideMatch.get(row_i).ion_intensity_matrix);

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
                            for (IonMatch ionMatch : peptideMatch.matched_ions) {
                                this.fragment_ions_mz_tol.add(ionMatch.getError(is_fragment_ion_tolu_ppm));
                            }
                        }
                    } else {
                        n_less_than_min_n_fragment_ions++;
                    }

                }
            }
            if (!un_recognized_PSMs.isEmpty()) {
                int n_un_recognized_PSMs = 0;
                for (String line : un_recognized_PSMs.keySet()) {
                    if (un_recognized_PSMs.get(line) == 0) {
                        n_un_recognized_PSMs++;
                        System.out.println("Spectrum not found:" + line);
                    }
                }
                if (n_un_recognized_PSMs >= 1) {
                    System.out.println("Spectrum not found:" + n_un_recognized_PSMs);
                }
            }
        }

        psmWriter.close();
        msWriter.close();
        fragWriter.close();
        if (this.export_fragment_ion_mz_to_file && fragMzWriter != null) {
            fragMzWriter.close();
        }
        if (this.export_skyline_transition_list_file && tfWriter != null && tbWriter != null) {
            tfWriter.close();
            tbWriter.close();
        }

        if (export_xic) {
            xicWriter.write("\n}");
            xicWriter.close();
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
        generate_rt_train_data(peptide2rt, rt_merge_method, this.out_dir + "/rt_train_data.tsv");
        CParameter.fragment_ion_intensity_cutoff = original_fragment_ion_intensity_cutoff;

    }

    /**
     * Generate training data for MS2 and retention time prediction using DDA data
     * with a generic format of peptide detection result.
     * 
     * @throws IOException
     */
    public void get_ms2_matches_generic_dda() throws IOException {
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
        HashMap<String, PeptideRT> peptide2rt = new HashMap<>();

        // for CCS
        HashMap<String, PeptideCCS> peptide2ccs = new HashMap<>();

        // output
        int frag_start_idx = 0;
        int frag_stop_idx = 0;
        BufferedWriter psmWriter = new BufferedWriter(new FileWriter(this.out_dir+"/psm_pdv.txt"));
        //psmWriter.write(this.psm_head_line+"\tspectrum_title\tmz\tcharge\tpeptide\tmodification\tmods\tmod_sites\tmax_fragment_ion_valid\tmax_cor_mz\tfrag_start_idx\tfrag_stop_idx\n");
        boolean ce_present = false;
        int ce_column_index = -1;
        if(hIndex.containsKey("ce")){
            ce_present = true;
            ce_column_index = hIndex.get("ce");
        }else if(hIndex.containsKey("nce")){
            ce_present = true;
            ce_column_index = hIndex.get("nce");
        }else if(hIndex.containsKey("CE")) {
            ce_present = true;
            ce_column_index = hIndex.get("CE");
        }else if(hIndex.containsKey("NCE")) {
            ce_present = true;
            ce_column_index = hIndex.get("NCE");
        }

        if(ce_present) {
            psmWriter.write("psm_id\tspectrum_title\tms2_scan\tmz\tcharge\tpeptide\tmodification\tmods\tmod_sites\tnce\tmax_fragment_ion_valid\tmax_cor_mz\tfrag_start_idx\tfrag_stop_idx\tn_valid_fragment_ions\tn_total_matched_ions\tvalid\n");
        }else{
            psmWriter.write("psm_id\tspectrum_title\tms2_scan\tmz\tcharge\tpeptide\tmodification\tmods\tmod_sites\tmax_fragment_ion_valid\tmax_cor_mz\tfrag_start_idx\tfrag_stop_idx\tn_valid_fragment_ions\tn_total_matched_ions\tvalid\n");
        }
        BufferedWriter msWriter = new BufferedWriter(new FileWriter(this.out_dir+"/ms_pdv.mgf"));
        BufferedWriter fragWriter = new BufferedWriter(new FileWriter(this.out_dir+"/fragment_intensity_df.tsv"));
        fragWriter.write(this.fragment_ion_intensity_head_line+"\n");

        BufferedWriter fragMzWriter = null;
        if (this.export_fragment_ion_mz_to_file) {
            fragMzWriter = new BufferedWriter(new FileWriter(this.out_dir + "/fragment_mz.tsv"));
            fragMzWriter.write(this.fragment_ion_intensity_head_line + "\n");
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

        // meta information about the MS data and model training
        BufferedWriter metaWriter = new BufferedWriter(new FileWriter(this.out_dir + "/meta.json"));
        metaWriter.write("{\n");

        int psm_id = 0;

        HashMap<String, JMeta> ms_file2meta = new HashMap<>();
        boolean first_meta = true;

        for (String ms_file : this.ms_file2psm.keySet()) {
            System.out.println("Process MS file:" + ms_file);
            ms_file2meta.put(ms_file, new JMeta());
            ms_file2meta.get(ms_file).ms_file = ms_file;
            // For store raw data
            DIAMeta meta = new DIAMeta();
            if (CParameter.itol > 0.2 && CParameter.itolu.startsWith("da")) {
                meta.fragment_ion_mz_bin_size = 0.5;
                System.out.println("Fragment ion bin size:" + meta.fragment_ion_mz_bin_size);
            }
            if (ms_file.endsWith(".mzML") || ms_file.endsWith(".mzml")) {
                meta.load_ms_data(ms_file);
                meta.get_ms_run_meta_data();
                CParameter.minPeptideMz = meta.precursor_ion_mz_min - 0.5;
                CParameter.maxPeptideMz = meta.precursor_ion_mz_max + 0.5;
                CParameter.min_fragment_ion_mz = meta.fragment_ion_mz_min - 0.5;
                if (CParameter.max_fragment_ion_mz > meta.fragment_ion_mz_max) {
                    CParameter.max_fragment_ion_mz = meta.fragment_ion_mz_max + 0.5;
                }
                CParameter.NCE = meta.nce;
                this.nce = meta.nce;
            } else if (ms_file.endsWith(".mgf") || ms_file.endsWith(".MGF")) {
                meta.fragment_ion_mz_min = CParameter.min_fragment_ion_mz;
                meta.fragment_ion_mz_max = CParameter.max_fragment_ion_mz;
                meta.precursor_ion_mz_min = CParameter.minPeptideMz;
                meta.precursor_ion_mz_max = CParameter.maxPeptideMz;
            } else {
                System.err.println("Error: unrecognized MS file format:" + ms_file);
                System.exit(1);
            }

            if (this.rt_max > meta.rt_max) {
                meta.rt_max = this.rt_max;
                System.out.println("Use user-provided RT max:" + this.rt_max);
            }

            System.out.println("NCE:" + CParameter.NCE);
            String ms_instrument_name = "";
            if (ms_file.endsWith(".mzML") || ms_file.endsWith(".mzml")) {
                ms_instrument_name = meta.get_ms_instrument(ms_file);
            }
            if (!ms_instrument_name.isEmpty()) {
                CParameter.ms_instrument = ms_instrument_name;
                this.ms_instrument = ms_instrument_name;
                System.out.println("MS instrument:"+ms_instrument_name);
            }else{
                System.out.println("No MS instrument detected from MS/MS data. Use default:"+this.ms_instrument+", "+CParameter.ms_instrument);
            }

            ms_file2meta.get(ms_file).ms_instrument = ms_instrument_name;
            // ms_file2meta.get(ms_file).nce = meta.nce;
            ms_file2meta.get(ms_file).nce = this.nce;
            ms_file2meta.get(ms_file).min_fragment_ion_mz = meta.fragment_ion_mz_min;
            ms_file2meta.get(ms_file).max_fragment_ion_mz = meta.fragment_ion_mz_max;
            ms_file2meta.get(ms_file).rt_max = meta.rt_max;
            ms_file2meta.get(ms_file).precursor_ion_mz_min = meta.precursor_ion_mz_min;
            ms_file2meta.get(ms_file).precursor_ion_mz_max = meta.precursor_ion_mz_max;
            if (first_meta) {
                metaWriter.write("\"" + ms_file + "\":" + JSON.toJSONString(ms_file2meta.get(ms_file)));
                first_meta = false;
            } else {
                metaWriter.write(",\n\"" + ms_file + "\":" + JSON.toJSONString(ms_file2meta.get(ms_file)));
            }

            this.min_fragment_ion_mz = meta.fragment_ion_mz_min;
            this.max_fragment_ion_mz = meta.fragment_ion_mz_max;
            System.out.println("Fragment ion m/z range:" + this.min_fragment_ion_mz + "," + this.max_fragment_ion_mz);

            if (meta.rt_max > this.rt_max) {
                this.rt_max = meta.rt_max;
                System.out.println("RT max:" + this.rt_max);
            } else {
                System.out.println("RT max:" + this.rt_max);
            }

            // for output
            HashSet<String> save_spectra = new HashSet<>();

            HashMap<String, ArrayList<String>> isoWinID2PSMs = new HashMap<>();

            boolean show_mod_ai_only_one_time = true;

            // for un-recognized PSMs: for example, no MS2 mapped.
            HashMap<String, Integer> un_recognized_PSMs = new HashMap<>();
            HashMap<String, HashMap<String, ArrayList<Integer>>> ms_file2spectrumID2row_index = new HashMap<>();
            int row_i = -1;
            for (String line : this.ms_file2psm.get(ms_file)) {
                row_i = row_i + 1;
                String[] d = line.split("\t");
                String peptide = d[hIndex.get("peptide")];
                String modification = d[hIndex.get("modification")];
                int precursor_charge = Integer.parseInt(d[hIndex.get("charge")]);
                try {
                    this.add_peptide(peptide,modification);
                } catch (NullPointerException e){
                    System.err.println("Error: "+peptide+":"+modification+":"+d[hIndex.get("spectrum_title")]);
                    System.exit(1);
                }

                String peptide_mod = peptide + "_" + modification;
                String spectrum_id = d[hIndex.get("spectrum_title")];
                if (!ms_file2spectrumID2row_index.containsKey(ms_file)) {
                    ms_file2spectrumID2row_index.put(ms_file, new HashMap<>());
                }
                if (!ms_file2spectrumID2row_index.get(ms_file).containsKey(spectrum_id)) {
                    ms_file2spectrumID2row_index.get(ms_file).put(spectrum_id, new ArrayList<>());
                }
                ms_file2spectrumID2row_index.get(ms_file).get(spectrum_id).add(row_i);

                if (hIndex.containsKey("rt")) {
                    if (!peptide2rt.containsKey(peptide_mod)) {
                        peptide2rt.put(peptide_mod, new PeptideRT());
                    }
                    peptide2rt.get(peptide_mod).peptide = peptide;
                    peptide2rt.get(peptide_mod).modification = modification;
                    peptide2rt.get(peptide_mod).rts.add(Double.parseDouble(d[hIndex.get("rt")])); // Apex RT
                    if (hIndex.containsKey("qvalue")) {
                        peptide2rt.get(peptide_mod).scores.add(Double.parseDouble(d[hIndex.get("qvalue")]));
                    } else if (hIndex.containsKey("q_value")) {
                        peptide2rt.get(peptide_mod).scores.add(Double.parseDouble(d[hIndex.get("q_value")]));
                    } else if (hIndex.containsKey("score")) {
                        peptide2rt.get(peptide_mod).scores.add(Double.parseDouble(d[hIndex.get("score")]));
                    } else {
                        peptide2rt.get(peptide_mod).scores.add(1.0);
                    }
                }

                // for CCS
                if (ccs_enabled && hIndex.containsKey("im")) {
                    String peptide_mode_charge = peptide + "_" + modification + "_" + precursor_charge;
                    if (!peptide2ccs.containsKey(peptide_mode_charge)) {
                        peptide2ccs.put(peptide_mode_charge, new PeptideCCS());
                    }
                    peptide2ccs.get(peptide_mode_charge).peptide = peptide;
                    peptide2ccs.get(peptide_mode_charge).modification = modification;
                    // In DIA-NN, iIM refers to the reference ion mobility in the spectral library, IM refers to the empirically measured.
                    peptide2ccs.get(peptide_mode_charge).ccs_values.add(Double.parseDouble(d[hIndex.get("im")]));
                    if (hIndex.containsKey("qvalue")) {
                        peptide2ccs.get(peptide_mode_charge).scores.add(Double.parseDouble(d[hIndex.get("qvalue")]));
                    }
                }
            }

            if (ms_file.endsWith(".mgf") || ms_file.endsWith(".MGF")) {
                Cloger.getInstance().logger.info("Process MS file:" + ms_file);
                // spectrumFactory.addSpectra(mgfFile, null);
                WaitingHandler waitingHandler = new WaitingHandlerCLIImpl();
                waitingHandler.setDisplayProgress(false);
                MgfFileIterator mgfFileIterator = new MgfFileIterator(new File(ms_file), waitingHandler);
                String spectrum_title;
                Spectrum spectrum;
                int spectrum_index = -1;
                int global_match_id = 0;
                HashMap<Integer, PeptideMatch> index2peptideMatch = new HashMap<>();
                int apex_scan = -1;
                while ((spectrum_title = mgfFileIterator.next()) != null) {
                    spectrum_index++;
                    apex_scan = spectrum_index;
                    spectrum = mgfFileIterator.getSpectrum();
                    if (ms_file2spectrumID2row_index.get(ms_file).containsKey(spectrum_title)) {
                        // peptide to spectrum matching
                        for (int row_index : ms_file2spectrumID2row_index.get(ms_file).get(spectrum_title)) {
                            row_i = row_index;
                            index2peptideMatch.put(row_i, new PeptideMatch());
                            index2peptideMatch.get(row_i).id = String.valueOf(row_i);
                            String line = this.ms_file2psm.get(ms_file).get(row_i);
                            String[] d = line.split("\t");
                            String peptide = d[hIndex.get("peptide")];
                            String modification = d[hIndex.get("modification")];
                            int precursor_charge = Integer.parseInt(d[hIndex.get("charge")]);
                            Peptide peptideObj = this.get_peptide(peptide, modification);
                            // intensity
                            index2peptideMatch.get(row_i).ion_intensity_matrix = new double[peptide.length() - 1][n_ion_types];
                            // this may not need
                            index2peptideMatch.get(row_i).ion_mz_matrix = new double[peptide.length() - 1][n_ion_types];
                            // 0: valid, >=1 invalid
                            index2peptideMatch.get(row_i).ion_matrix = new int[peptide.length() - 1][n_ion_types];
                            // index2peptideMatch.get(row_i).scan = apex_scan;
                            // index2peptideMatch.get(row_i).rt_start = rt_start;
                            // index2peptideMatch.get(row_i).rt_end = rt_end;
                            // index2peptideMatch.get(row_i).rt_apex = Double.parseDouble(d[hIndex.get("RT")]);
                            index2peptideMatch.get(row_i).peptide_length = peptide.length();
                            index2peptideMatch.get(row_i).precursor_charge = precursor_charge;
                            // index2peptideMatch.get(row_i).index = Integer.parseInt(d[hIndex.get("MS2.Scan")]);
                            index2peptideMatch.get(row_i).peptide = peptideObj;

                            if (spectrum == null) {
                                if (!un_recognized_PSMs.containsKey(line)) {
                                    un_recognized_PSMs.put(line, 0);
                                }
                                continue;
                            } else {
                                un_recognized_PSMs.put(line, 1);
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
                                    if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION
                                            || ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION) {
                                        // add fragment ion number
                                        PeptideFragmentIon fragmentIon = ((PeptideFragmentIon) ionMatch.ion);
                                        int ion_number = fragmentIon.getNumber();
                                        int ion_type_column_index = this.get_ion_type_column_index(ionMatch);
                                        // for y ion
                                        if (ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION) {
                                            fragment_ion_row_index = peptide.length() - ion_number - 1;
                                            ion_type = "y";
                                        } else if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION) {
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
                    }
                }
                // output
                row_i = -1;
                for (String line : this.ms_file2psm.get(ms_file)) {
                    n_total_matches++;
                    row_i = row_i + 1;
                    String[] d = line.split("\t");

                    if (this.mod_ai.equalsIgnoreCase("-") || this.mod_ai.equalsIgnoreCase("general")) {
                        // nothing to do
                    } else if (this.mod_ai.equalsIgnoreCase("phosphorylation")) {
                        String mod_seq = d[hIndex.get("Modified.Sequence")];
                        if (hIndex.containsKey("PTM.Site.Confidence") && mod_seq.contains("UniMod:21")) {
                            // only filtering out low confidence phosphorylation peptides
                            if (Double.parseDouble(d[hIndex.get("PTM.Site.Confidence")]) < this.ptm_site_prob_cutoff) {
                                n_ptm_site_low_confidence++;
                                continue;
                            }
                            if (Double.parseDouble(d[hIndex.get("PTM.Q.Value")]) > this.ptm_site_qvalue_cutoff) {
                                n_ptm_site_low_confidence++;
                                continue;
                            }
                        }
                    } else {
                        System.err.println("Modification type is not supported:" + this.mod_ai);
                        System.exit(1);
                    }

                    // TODO: need to determine if this filter is necessary: index2peptideMatch.get(row_i).matched_ions.size()>=this.min_n_fragment_ions
                    if(index2peptideMatch.containsKey(row_i) && index2peptideMatch.get(row_i).max_fragment_ion_intensity>0 && index2peptideMatch.get(row_i).matched_ions.size()>=this.min_n_fragment_ions) {
                        boolean fragment_export = false;

                        String [] out_mod = convert_modification(d[hIndex.get("modification")]);
                        int n_valid_fragment_ions = get_n_valid_fragment_ions(index2peptideMatch.get(row_i).ion_intensity_matrix,index2peptideMatch.get(row_i).ion_matrix);
                        int n_total_fragment_ions = get_n_matched_fragment_ions(index2peptideMatch.get(row_i).ion_intensity_matrix);
                        // TODO: fix this
                        spectrum_title = d[hIndex.get("spectrum_title")];
                        double pdv_precursor_mz = dbGear.get_mz(this.get_peptide(d[hIndex.get("peptide")], d[hIndex.get("modification")]).getMass(),
                                Integer.parseInt(d[hIndex.get("charge")]));
                        String pdv_precursor_charge = d[hIndex.get("charge")];
                        String pdv_peptide = d[hIndex.get("peptide")];
                        String pdv_modification = d[hIndex.get("modification")];
                        // true || true
                        if (index2peptideMatch.get(row_i).is_max_fragment_ion_intensity_valid() || use_all_peaks) {
                            n_total_matches_valid++;
                            n_total_psm_matches_valid++;
                            frag_start_idx = frag_stop_idx;
                            frag_stop_idx = frag_start_idx + index2peptideMatch.get(row_i).ion_intensity_matrix.length;
                            if(ce_present){
                                psmWriter.write(index2peptideMatch.get(row_i).id + "\t" + spectrum_title + "\t" + index2peptideMatch.get(row_i).scan + "\t" + pdv_precursor_mz + "\t" + pdv_precursor_charge + "\t" + pdv_peptide + "\t" + pdv_modification + "\t" + out_mod[0] + "\t" + out_mod[1] + "\t"+d[ce_column_index]+ "\t1\t" + index2peptideMatch.get(row_i).max_cor_mz + "\t" + frag_start_idx + "\t" + frag_stop_idx +
                                        "\t" + n_valid_fragment_ions + "\t" + n_total_fragment_ions + "\t1\n");
                            }else {
                                psmWriter.write(index2peptideMatch.get(row_i).id + "\t" + spectrum_title + "\t" + index2peptideMatch.get(row_i).scan + "\t" + pdv_precursor_mz + "\t" + pdv_precursor_charge + "\t" + pdv_peptide + "\t" + pdv_modification + "\t" + out_mod[0] + "\t" + out_mod[1] + "\t1\t" + index2peptideMatch.get(row_i).max_cor_mz + "\t" + frag_start_idx + "\t" + frag_stop_idx +
                                        "\t" + n_valid_fragment_ions + "\t" + n_total_fragment_ions + "\t1\n");
                            }
                            fragment_export = true;
                        } else {
                            n_total_matches_max_fragment_ion_invalid++;
                            if (!this.export_valid_matches_only) {
                                frag_start_idx = frag_stop_idx;
                                frag_stop_idx = frag_start_idx + index2peptideMatch.get(row_i).ion_intensity_matrix.length;
                                if(ce_present){
                                    psmWriter.write(index2peptideMatch.get(row_i).id + "\t" + spectrum_title + "\t" + index2peptideMatch.get(row_i).scan + "\t" + pdv_precursor_mz + "\t" + pdv_precursor_charge + "\t" + pdv_peptide + "\t" + pdv_modification + "\t" + out_mod[0] + "\t" + out_mod[1] + "\t"+d[ce_column_index]+ "\t0\t" + index2peptideMatch.get(row_i).max_cor_mz + "\t" + frag_start_idx + "\t" + frag_stop_idx +
                                            "\t" + n_valid_fragment_ions + "\t" + n_total_fragment_ions + "\t0\n");
                                }else {
                                    psmWriter.write(index2peptideMatch.get(row_i).id + "\t" + spectrum_title + "\t" + index2peptideMatch.get(row_i).scan + "\t" + pdv_precursor_mz + "\t" + pdv_precursor_charge + "\t" + pdv_peptide + "\t" + pdv_modification + "\t" + out_mod[0] + "\t" + out_mod[1] + "\t0\t" + index2peptideMatch.get(row_i).max_cor_mz + "\t" + frag_start_idx + "\t" + frag_stop_idx +
                                            "\t" + n_valid_fragment_ions + "\t" + n_total_fragment_ions + "\t0\n");
                                }
                                fragment_export = true;
                            }
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

                            // for ms2 mz tol
                            PeptideMatch peptideMatch = index2peptideMatch.get(row_i);
                            for (IonMatch ionMatch : peptideMatch.matched_ions) {
                                this.fragment_ions_mz_tol.add(ionMatch.getError(is_fragment_ion_tolu_ppm));
                            }
                        }
                    } else {
                        if(!index2peptideMatch.containsKey(row_i)){
                            Cloger.getInstance().logger.error("PSM not matched to MS2 spectrum:"+row_i+" => "+line);
                        }
                        n_less_than_min_n_fragment_ions++;
                    }

                }

            } else {
                // TODO: implement this for .mzML
                System.err.println("Invalid MS file format:" + ms_file);
                System.exit(1);
            }
        }

        psmWriter.close();
        msWriter.close();
        fragWriter.close();
        if (this.export_fragment_ion_mz_to_file && fragMzWriter != null) {
            fragMzWriter.close();
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
        generate_rt_train_data(peptide2rt, rt_merge_method, this.out_dir + "/rt_train_data.tsv");
        CParameter.fragment_ion_intensity_cutoff = original_fragment_ion_intensity_cutoff;

    }

    /**
     * Generate training data for MS2 and retention time prediction based on DIA-NN
     * search result for TIMS-TOF data.
     * This function is still in development and may not work as expected.
     * It writes the results to output files for model training.
     * 
     * @throws IOException If there is an error reading or writing files.
     */
    public void get_ms2_matches_diann_ccs() throws IOException {
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
        HashMap<String, PeptideRT> peptide2rt = new HashMap<>();

        // for CCS
        HashMap<String, PeptideCCS> peptide2ccs = new HashMap<>();

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
        if (this.export_fragment_ion_mz_to_file) {
            fragMzWriter = new BufferedWriter(new FileWriter(this.out_dir + "/fragment_mz.tsv"));
            fragMzWriter.write(this.fragment_ion_intensity_head_line + "\n");
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

        if (this.export_skyline_transition_list_file) {

            tfWriter = new BufferedWriter(new FileWriter(this.out_dir+"/skyline_input.tsv"));
            tfWriter.write("Peptide\tPrecursor m/z\tProduct m/z\tLibraryIntensity\tExplicit Retention Time\tExplicit Retention Time Window\tNote\n");

            // peak boundary file
            tbWriter = new BufferedWriter(new FileWriter(this.out_dir + "/skyline_boundary.tsv"));
            tbWriter.write("MinStartTime\tMaxEndTime\tFileName\tPeptideModifiedSequence\tPrecursorCharge\n");

        }

        BufferedWriter xicWriter = null;
        boolean first_xic = true;
        if (export_xic) {
            xicWriter = new BufferedWriter(new FileWriter(this.out_dir + "/xic.json"));
            xicWriter.write("{\n");
        }

        // meta information about the MS data and model training
        BufferedWriter metaWriter = new BufferedWriter(new FileWriter(this.out_dir + "/meta.json"));
        metaWriter.write("{\n");

        int psm_id = 0;

        HashMap<String, JMeta> ms_file2meta = new HashMap<>();
        boolean first_meta = true;

        for (String ms_file : this.ms_file2psm.keySet()) {
            System.out.println("Process MS file:" + ms_file);
            ms_file2meta.put(ms_file, new JMeta());
            ms_file2meta.get(ms_file).ms_file = ms_file;
            // For store raw data
            CCSDIAMeta meta = new CCSDIAMeta();
            if (CParameter.itol > 0.2 && CParameter.itolu.startsWith("da")) {
                meta.fragment_ion_mz_bin_size = 0.5;
                System.out.println("Fragment ion bin size:" + meta.fragment_ion_mz_bin_size);
            }
            meta.load_ms_data(ms_file);
            meta.get_ms_run_meta_data(ms_file);
            CParameter.minPeptideMz = meta.precursor_ion_mz_min - 0.5;
            CParameter.maxPeptideMz = meta.precursor_ion_mz_max + 0.5;
            CParameter.min_fragment_ion_mz = meta.fragment_ion_mz_min - 0.5;
            if (CParameter.max_fragment_ion_mz > meta.fragment_ion_mz_max) {
                CParameter.max_fragment_ion_mz = meta.fragment_ion_mz_max + 0.5;
            }
            CParameter.NCE = meta.nce;
            this.nce = meta.nce;
            System.out.println("NCE:" + CParameter.NCE);
            String ms_instrument_name = meta.get_ms_instrument();
            if (!ms_instrument_name.isEmpty()) {
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
            if (first_meta) {
                metaWriter.write("\"" + ms_file + "\":" + JSON.toJSONString(ms_file2meta.get(ms_file)));
                first_meta = false;
            } else {
                metaWriter.write(",\n\"" + ms_file + "\":" + JSON.toJSONString(ms_file2meta.get(ms_file)));
            }

            this.min_fragment_ion_mz = meta.fragment_ion_mz_min;
            this.max_fragment_ion_mz = meta.fragment_ion_mz_max;
            System.out.println("Fragment ion m/z range:" + this.min_fragment_ion_mz + "," + this.max_fragment_ion_mz);

            CCSDIAMap diaMap_tmp = new CCSDIAMap();
            diaMap_tmp.meta = meta;
            if (this.target_isolation_wins.isEmpty()) {
                diaMap_tmp.target_isolation_wins.addAll(meta.isolationWindowMap.keySet());
            } else {
                diaMap_tmp.target_isolation_wins.addAll(this.target_isolation_wins);
            }

            if (meta.rt_max > this.rt_max) {
                this.rt_max = meta.rt_max;
                System.out.println("RT max:" + this.rt_max);
            } else {
                System.out.println("RT max:" + this.rt_max);
            }

            // for output
            HashSet<String> save_spectra = new HashSet<>();

            HashMap<String, ArrayList<String>> isoWinID2PSMs = new HashMap<>();

            // key: PSM line, value: spectrum index in the original MS data (0-based)
            ConcurrentHashMap<String,ApexMatch> i2ms2index = this.get_ms2spectrum_index(this.ms_file2psm.get(ms_file), ms_file, dbGear,out_dir);

            // for un-recognized PSMs: for example, no MS2 mapped.
            HashMap<String, Integer> un_recognized_PSMs = new HashMap<>();
            for (String line : this.ms_file2psm.get(ms_file)) {
                String[] d = line.split("\t");
                String peptide = d[hIndex.get("Stripped.Sequence")];
                String modification = this.get_modification_diann(d[hIndex.get("Modified.Sequence")], peptide);
                int precursor_charge = Integer.parseInt(d[hIndex.get("Precursor.Charge")]);
                this.add_peptide(peptide,modification);
                // ArrayList<String> isoWinIDs = diaMap_tmp.get_isolation_windows(dbGear.get_mz(this.get_peptide(peptide,modification).getMass(),precursor_charge));
                ArrayList<String> isoWinIDs = new ArrayList<>();
                isoWinIDs.add(i2ms2index.get(line).isolation_window);
                if (isoWinIDs.isEmpty()) {
                    System.out.println("Isolation window ID is empty:" + line);
                    continue;
                }
                for (String isoWinID : isoWinIDs) {
                    if (!isoWinID2PSMs.containsKey(isoWinID)) {
                        isoWinID2PSMs.put(isoWinID, new ArrayList<>());
                    }
                    isoWinID2PSMs.get(isoWinID).add(line);

                    String peptide_mod = peptide + "_" + modification;

                    if (this.mod_ai.equalsIgnoreCase("-") || this.mod_ai.equalsIgnoreCase("general")) {
                        // nothing to do
                    } else if (this.mod_ai.equalsIgnoreCase("phosphorylation")) {
                        String mod_seq = d[hIndex.get("Modified.Sequence")];
                        if (hIndex.containsKey("PTM.Site.Confidence") && mod_seq.contains("UniMod:21")) {
                            // only filtering out low confidence phosphorylation peptides
                            if (Double.parseDouble(d[hIndex.get("PTM.Site.Confidence")]) < this.ptm_site_prob_cutoff) {
                                continue;
                            }
                            if (Double.parseDouble(d[hIndex.get("PTM.Q.Value")]) > this.ptm_site_qvalue_cutoff) {
                                continue;
                            }
                        }
                    } else {
                        System.err.println("Modification type is not supported:" + this.mod_ai);
                        System.exit(1);
                    }

                    if (!peptide2rt.containsKey(peptide_mod)) {
                        peptide2rt.put(peptide_mod, new PeptideRT());
                    }
                    peptide2rt.get(peptide_mod).peptide = peptide;
                    peptide2rt.get(peptide_mod).modification = modification;
                    peptide2rt.get(peptide_mod).rts.add(Double.parseDouble(d[hIndex.get("RT")])); // Apex RT
                    peptide2rt.get(peptide_mod).scores.add(Double.parseDouble(d[hIndex.get("Q.Value")]));

                    // for CCS
                    if (ccs_enabled) {
                        String peptide_mode_charge = peptide + "_" + modification + "_" + precursor_charge;
                        if (!peptide2ccs.containsKey(peptide_mode_charge)) {
                            peptide2ccs.put(peptide_mode_charge, new PeptideCCS());
                        }
                        peptide2ccs.get(peptide_mode_charge).peptide = peptide;
                        peptide2ccs.get(peptide_mode_charge).modification = modification;
                        // In DIA-NN, iIM refers to the reference ion mobility in the spectral library, IM refers to the empirically measured.
                        peptide2ccs.get(peptide_mode_charge).ccs_values.add(Double.parseDouble(d[hIndex.get("IM")]));
                        peptide2ccs.get(peptide_mode_charge).scores.add(Double.parseDouble(d[hIndex.get("Q.Value")]));
                        peptide2ccs.get(peptide_mode_charge).charge = precursor_charge;
                    }
                }
            }

            for (String isoWinID : isoWinID2PSMs.keySet()) {
                CCSDIAIndex diaIndex = new CCSDIAIndex();
                diaIndex.fragment_ion_intensity_threshold = this.fragment_ion_intensity_threshold;
                diaIndex.meta = meta;
                diaIndex.target_isolation_wins.add(isoWinID);
                System.out.println("Isolation window:" + isoWinID);
                diaIndex.index(ms_file);
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
                    String modification = this.get_modification_diann(d[hIndex.get("Modified.Sequence")], peptide);
                    int precursor_charge = Integer.parseInt(d[hIndex.get("Precursor.Charge")]);
                    // double apex_rt = Double.parseDouble(d[hIndex.get("apex_rt")]);
                    double rt_start = Double.parseDouble(d[hIndex.get("RT.Start")]);
                    double rt_end = Double.parseDouble(d[hIndex.get("RT.Stop")]);

                    // The index (0-based) of apex scan in the original MS data file
                    int apex_scan=i2ms2index.get(line).ms2index; //= global_index2scan_num.get(Integer.parseInt(d[hIndex.get("MS2.Scan")])); // index
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
                    index2peptideMatch.get(row_i).im = Double.parseDouble(d[hIndex.get("IM")]);

                    // for testing
                    if(test_mode){
                        index2peptideMatch.get(row_i).ion_matrix_map.put("spectrum_centric", new int[peptide.length() - 1][n_ion_types]);
                        index2peptideMatch.get(row_i).ion_matrix_map.put("peptide_centric_cor", new int[peptide.length() - 1][n_ion_types]);
                        index2peptideMatch.get(row_i).ion_matrix_map.put("peptide_centric_shape", new int[peptide.length() - 1][n_ion_types]);
                        index2peptideMatch.get(row_i).ion_matrix_map.put("low_mass", new int[peptide.length() - 1][n_ion_types]);
                        index2peptideMatch.get(row_i).ion_matrix_map.put("peptide_centric", new int[peptide.length() - 1][n_ion_types]);
                    }

                    if (spectrum == null) {
                        if (!un_recognized_PSMs.containsKey(line)) {
                            un_recognized_PSMs.put(line, 0);
                        }
                        System.out.println("Spectrum is invalid!");
                        System.out.println(line);
                        System.out.println("ms2 index:"+apex_scan);
                        System.out.println("# Spectra:"+diaIndex.scan2spectrum.size());
                        //for(int jj: diaIndex.scan2spectrum.keySet()){
                        //    System.out.println(jj+"\t"+diaIndex.get_spectrum_by_scan(jj).getPrecursor().mz);
                        //}
                        //System.exit(1);
                        continue;
                    } else {
                        un_recognized_PSMs.put(line, 1);
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
                            if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION
                                    || ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION) {
                                // add fragment ion number
                                PeptideFragmentIon fragmentIon = ((PeptideFragmentIon) ionMatch.ion);
                                int ion_number = fragmentIon.getNumber();
                                int ion_type_column_index = this.get_ion_type_column_index(ionMatch);
                                // for y ion
                                if (ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION) {
                                    fragment_ion_row_index = peptide.length() - ion_number - 1;
                                    ion_type = "y";
                                } else if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION) {
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

                                // If the fragment ion number is <= the minimum number of fragment ion used for
                                // spectral library generation,
                                // we don't consider it in getting the max intensity of fragment ions.
                                if (use_all_peaks || (max_fragment_ion_intensity <= ionMatch.peakIntensity
                                        && ion_number >= this.lf_frag_n_min)) {
                                    max_fragment_ion_intensity = ionMatch.peakIntensity;
                                    max_fragment_ion_row_index = fragment_ion_row_index;
                                    max_fragment_ion_column_index = ion_type_column_index;
                                }
                            }
                        }
                    } else {
                        System.out.println("No match!");
                    }
                    if (!matched_ion_mzs.isEmpty()) {
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
                            if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION
                                    || ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION) {
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
                System.out.println("Peptide to query:"+index2peptideMatch.size());
                index2peptideMatch.values().parallelStream().forEach(peptideMatch -> xic_query_ccs(diaIndex,peptideMatch,isoWinID));
                row_i = -1;
                int[] ind = new int[] { 0, 0 };
                for (String line : isoWinID2PSMs.get(isoWinID)) {
                    row_i = row_i + 1;
                    PeptideMatch peptideMatch = index2peptideMatch.get(row_i);
                    HashSet<Double> high_cor_mzs = new HashSet<>();
                    double max_cor_mz = 0;
                    double max_frag_cor = -100;
                    for (double mz : peptideMatch.mz2cor.keySet()) {
                        if (peptideMatch.mz2cor.get(mz) >= this.cor_cutoff) {
                            high_cor_mzs.add(mz);
                        }
                        if (peptideMatch.mz2cor.get(mz) > max_frag_cor) {
                            max_frag_cor = peptideMatch.mz2cor.get(mz);
                            max_cor_mz = mz;
                        }
                    }
                    peptideMatch.max_cor_mz = max_cor_mz;
                    for (double mz : peptideMatch.mz2index.keySet()) {
                        if (!high_cor_mzs.contains(mz)) {
                            ind = peptideMatch.mz2index.get(mz);
                            peptideMatch.ion_matrix[ind[0]][ind[1]] = peptideMatch.ion_matrix[ind[0]][ind[1]] + 1;
                            if (test_mode) {
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
                    for (double mz : peptideMatch.mz2index.keySet()) {
                        if (peptideMatch.mz2skewed_peaks.containsKey(mz) && peptideMatch.mz2skewed_peaks.get(mz) >= 2) {
                            ind = peptideMatch.mz2index.get(mz);
                            peptideMatch.ion_matrix[ind[0]][ind[1]] = peptideMatch.ion_matrix[ind[0]][ind[1]] + 1;
                            if (test_mode) {
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
                    String[] d = line.split("\t");
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
                                } else if (ion.getSubType() == PeptideFragmentIon.B_ION) {
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
                    String[] d = line.split("\t");
                    if (hIndex.containsKey("peak_overlap")) {
                        if (Integer.parseInt(d[hIndex.get("peak_overlap")]) >= 1) {
                            n_peak_overlap++;
                            continue;
                        }
                    }

                    if (this.mod_ai.equalsIgnoreCase("-") || this.mod_ai.equalsIgnoreCase("general")) {
                        // nothing to do
                    } else if (this.mod_ai.equalsIgnoreCase("phosphorylation")) {
                        String mod_seq = d[hIndex.get("Modified.Sequence")];
                        if (hIndex.containsKey("PTM.Site.Confidence") && mod_seq.contains("UniMod:21")) {
                            // only filtering out low confidence phosphorylation peptides
                            if (Double.parseDouble(d[hIndex.get("PTM.Site.Confidence")]) < this.ptm_site_prob_cutoff) {
                                n_ptm_site_low_confidence++;
                                continue;
                            }
                            if (Double.parseDouble(d[hIndex.get("PTM.Q.Value")]) > this.ptm_site_qvalue_cutoff) {
                                n_ptm_site_low_confidence++;
                                continue;
                            }
                        }
                    } else {
                        System.err.println("Modification type is not supported:" + this.mod_ai);
                        System.exit(1);
                    }

                    if(index2peptideMatch.get(row_i).max_fragment_ion_intensity>0 && index2peptideMatch.get(row_i).matched_ions.size()>=this.min_n_fragment_ions) {
                        boolean fragment_export = false;

                        String [] out_mod = convert_modification(this.get_modification_diann(d[hIndex.get("Modified.Sequence")],d[hIndex.get("Stripped.Sequence")]));
                        int n_valid_fragment_ions = get_n_valid_fragment_ions(index2peptideMatch.get(row_i).ion_intensity_matrix,index2peptideMatch.get(row_i).ion_matrix);
                        int n_total_fragment_ions = get_n_matched_fragment_ions(index2peptideMatch.get(row_i).ion_intensity_matrix);
                        if(n_valid_fragment_ions >= this.min_n_high_quality_fragment_ions) {

                            // get adjacent scans
                            ArrayList<PeptideMatch> pMatches = get_adjacent_ms2_matches_ccs(index2peptideMatch.get(row_i),this.n_flank_scans,diaIndex,isoWinID);
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

                            int apex_scan = i2ms2index.get(line).ms2index;;//= global_index2scan_num.get(Integer.parseInt(d[hIndex.get("MS2.Scan")]));
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
                                    if (test_mode) {
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
                                if (!pMatches.isEmpty()) {
                                    for (PeptideMatch pMatch : pMatches) {
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
                                for (IonMatch ionMatch : peptideMatch.matched_ions) {
                                    this.fragment_ions_mz_tol.add(ionMatch.getError(is_fragment_ion_tolu_ppm));
                                }
                            }
                        } else {
                            n_less_than_min_n_high_quality_fragment_ions++;
                        }
                    } else {
                        n_less_than_min_n_fragment_ions++;
                    }

                }
            }
            if (!un_recognized_PSMs.isEmpty()) {
                int n_un_recognized_PSMs = 0;
                for (String line : un_recognized_PSMs.keySet()) {
                    if (un_recognized_PSMs.get(line) == 0) {
                        n_un_recognized_PSMs++;
                        System.out.println("Spectrum not found:" + line);
                    }
                }
                if (n_un_recognized_PSMs >= 1) {
                    System.out.println("Spectrum not found:" + n_un_recognized_PSMs);
                }
            }
        }

        psmWriter.close();
        msWriter.close();
        fragWriter.close();
        fragValidWriter.close();
        if (this.export_fragment_ion_mz_to_file && fragMzWriter != null) {
            fragMzWriter.close();
        }
        if (this.export_skyline_transition_list_file && tfWriter != null && tbWriter != null) {
            tfWriter.close();
            tbWriter.close();
        }

        if (export_xic) {
            xicWriter.write("\n}");
            xicWriter.close();
        }

        if (test_mode) {
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
        generate_rt_train_data(peptide2rt, rt_merge_method, this.out_dir + "/rt_train_data.tsv");
        generate_ccs_train_data(peptide2ccs, ccs_merge_method, this.out_dir + "/ccs_train_data.tsv");
        CParameter.fragment_ion_intensity_cutoff = original_fragment_ion_intensity_cutoff;

    }

    /**
     * Generate training data for MS2 and retention time prediction based on DIA-NN
     * search result for TIMS-TOF data.
     * This function uses a
     * <a href="https://github.com/TalusBio/timsbuktoolkit">Rust-based library</a>
     * to parse TIMS-TOF data.
     * This function is still in development and may not work as expected.
     * It writes the results to output files for model training.
     * 
     * @throws IOException If there is an error reading or writing files.
     */
    public void get_ms2_matches_diann_ccs_rust() throws IOException {
        this.ion_type2column_index.clear();
        double original_fragment_ion_intensity_cutoff = CParameter.fragment_ion_intensity_cutoff;
        CParameter.fragment_ion_intensity_cutoff = 0.0001;
        PeptideFrag.lossWaterNH3 = this.lossWaterNH3;
        Cloger.getInstance().logger.info("lossWaterNH3:"+PeptideFrag.lossWaterNH3);
        PeptideFrag.max_fragment_ion_charge = this.max_fragment_ion_charge;
        PeptideFrag.fragment_ion_charge_less_than_precursor_charge = this.fragment_ion_charge_less_than_precursor_charge;

        boolean is_fragment_ion_tolu_ppm = CParameter.itolu.equalsIgnoreCase("ppm");

        this.load_mod_map();
        set_ion_type_column_index(this.fragmentation_method,this.max_fragment_ion_charge, this.lossWaterNH3);
        int n_ion_types = !(this.mod_ai.equals("-") || this.mod_ai.equalsIgnoreCase("general"))?this.max_fragment_ion_charge*2*2:this.max_fragment_ion_charge*2;
        Cloger.getInstance().logger.info("The number of ion types:"+n_ion_types);
        DBGear dbGear = new DBGear();

        // for RT
        HashMap<String, PeptideRT> peptide2rt = new HashMap<>();

        // for CCS
        HashMap<String, PeptideCCS> peptide2ccs = new HashMap<>();

        // output
        int frag_start_idx = 0;
        int frag_stop_idx = 0;
        BufferedWriter psmWriter = new BufferedWriter(new FileWriter(this.out_dir+"/psm_pdv.txt"));
        //psmWriter.write(this.psm_head_line+"\tspectrum_title\tmz\tcharge\tpeptide\tmodification\tmods\tmod_sites\tmax_fragment_ion_valid\tmax_cor_mz\tfrag_start_idx\tfrag_stop_idx\n");
        psmWriter.write("psm_id\tspectrum_title\tms2_scan\tmz\tcharge\tnce\tpeptide\tmodification\tmods\tmod_sites\tmax_fragment_ion_valid\tmax_cor_mz\tfrag_start_idx\tfrag_stop_idx\tn_valid_fragment_ions\tn_total_matched_ions\tvalid\n");
        BufferedWriter msWriter = new BufferedWriter(new FileWriter(this.out_dir+"/ms_pdv.mgf"));
        BufferedWriter fragWriter = new BufferedWriter(new FileWriter(this.out_dir+"/fragment_intensity_df.tsv"));
        fragWriter.write(this.fragment_ion_intensity_head_line+"\n");

        BufferedWriter fragMzWriter = null;
        if (this.export_fragment_ion_mz_to_file) {
            fragMzWriter = new BufferedWriter(new FileWriter(this.out_dir + "/fragment_mz.tsv"));
            fragMzWriter.write(this.fragment_ion_intensity_head_line + "\n");
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

        if (this.export_skyline_transition_list_file) {

            tfWriter = new BufferedWriter(new FileWriter(this.out_dir+"/skyline_input.tsv"));
            tfWriter.write("Peptide\tPrecursor m/z\tProduct m/z\tLibraryIntensity\tExplicit Retention Time\tExplicit Retention Time Window\tNote\n");

            // peak boundary file
            tbWriter = new BufferedWriter(new FileWriter(this.out_dir + "/skyline_boundary.tsv"));
            tbWriter.write("MinStartTime\tMaxEndTime\tFileName\tPeptideModifiedSequence\tPrecursorCharge\n");

        }

        BufferedWriter xicWriter = null;
        boolean first_xic = true;
        if (export_xic) {
            xicWriter = new BufferedWriter(new FileWriter(this.out_dir + "/xic.json"));
            xicWriter.write("{\n");
        }

        // meta information about the MS data and model training
        BufferedWriter metaWriter = new BufferedWriter(new FileWriter(this.out_dir + "/meta.json"));
        metaWriter.write("{\n");

        // global psm_id
        int psm_id = 0;

        HashMap<String, JMeta> ms_file2meta = new HashMap<>();
        boolean first_meta = true;

        for (String ms_file : this.ms_file2psm.keySet()) {
            psm_id++;
            Cloger.getInstance().logger.info("Process MS file:" + ms_file);
            ms_file2meta.put(ms_file, new JMeta());
            ms_file2meta.get(ms_file).ms_file = ms_file;
            // For store raw data
            CCSDIAMeta meta = new CCSDIAMeta();
            if (CParameter.itol > 0.2 && CParameter.itolu.startsWith("da")) {
                meta.fragment_ion_mz_bin_size = 0.5;
                Cloger.getInstance().logger.info("Fragment ion bin size:" + meta.fragment_ion_mz_bin_size);
            }
            // meta.load_ms_data(ms_file);
            meta.get_ms_run_meta_data_using_sql(ms_file);
            meta.generate_mz2ce_map();
            this.ccs_dia_meta = meta;
            this.use_fixed_ce = false;
            // CParameter.minPeptideMz = meta.precursor_ion_mz_min - 0.5;
            CParameter.minPeptideMz = meta.precursor_ion_mz_min;
            // CParameter.maxPeptideMz = meta.precursor_ion_mz_max + 0.5;
            CParameter.maxPeptideMz = meta.precursor_ion_mz_max;
            CParameter.min_fragment_ion_mz = meta.fragment_ion_mz_min - 0.5;
            if (CParameter.max_fragment_ion_mz > meta.fragment_ion_mz_max) {
                CParameter.max_fragment_ion_mz = meta.fragment_ion_mz_max + 0.5;
            }
            CParameter.NCE = meta.nce;
            this.nce = meta.nce;
            Cloger.getInstance().logger.info("NCE:" + CParameter.NCE);
            String ms_instrument_name = meta.get_ms_instrument();
            if (!ms_instrument_name.isEmpty()) {
                CParameter.ms_instrument = ms_instrument_name;
                this.ms_instrument = ms_instrument_name;
                Cloger.getInstance().logger.info("MS instrument:"+ms_instrument_name);
            }else{
                Cloger.getInstance().logger.info("No MS instrument detected from MS/MS data. Use default:"+this.ms_instrument+", "+CParameter.ms_instrument);
            }

            ms_file2meta.get(ms_file).ms_instrument = ms_instrument_name;
            ms_file2meta.get(ms_file).nce = meta.nce;
            ms_file2meta.get(ms_file).min_fragment_ion_mz = meta.fragment_ion_mz_min;
            ms_file2meta.get(ms_file).max_fragment_ion_mz = meta.fragment_ion_mz_max;
            ms_file2meta.get(ms_file).rt_max = meta.rt_max;
            if (first_meta) {
                metaWriter.write("\"" + ms_file + "\":" + JSON.toJSONString(ms_file2meta.get(ms_file)));
                first_meta = false;
            } else {
                metaWriter.write(",\n\"" + ms_file + "\":" + JSON.toJSONString(ms_file2meta.get(ms_file)));
            }

            this.min_fragment_ion_mz = meta.fragment_ion_mz_min;
            this.max_fragment_ion_mz = meta.fragment_ion_mz_max;
            Cloger.getInstance().logger.info("Fragment ion m/z range:" + this.min_fragment_ion_mz + "," + this.max_fragment_ion_mz);

            CCSDIAMap diaMap_tmp = new CCSDIAMap();
            diaMap_tmp.meta = meta;
            if (this.target_isolation_wins.isEmpty()) {
                diaMap_tmp.target_isolation_wins.addAll(meta.isolationWindowMap.keySet());
            } else {
                diaMap_tmp.target_isolation_wins.addAll(this.target_isolation_wins);
            }

            if (meta.rt_max > this.rt_max) {
                this.rt_max = meta.rt_max;
                Cloger.getInstance().logger.info("RT max:" + this.rt_max);
            } else {
                Cloger.getInstance().logger.info("RT max:" + this.rt_max);
            }

            // for output
            HashSet<String> save_spectra = new HashSet<>();

            // key: PSM line, value: spectrum index in the original MS data (0-based)
            ConcurrentHashMap<String, ApexMatch> i2ms2index = new ConcurrentHashMap<>(); // =
                                                                                         // this.get_ms2spectrum_index(this.ms_file2psm.get(ms_file),
                                                                                         // ms_file, dbGear,out_dir);

            List<PSMQuery> psm_query_list = new ArrayList<>();

            // for debugging
            // invalid max fragment ion intensity
            HashSet<Integer> invalid_max_fragment_ion_intensity_indices = new HashSet<>();
            // less than min_n_high_quality_fragment_ions
            HashSet<Integer> invalid_min_n_high_quality_fragment_ions_indices = new HashSet<>();

            // Get the MS1 and MS2 m/z error from DIA-NN search result
            ArrayList<Double> ms1_mz_errors = new ArrayList<>(this.ms_file2psm.get(ms_file).size());
            ArrayList<Double> ms2_mz_errors = new ArrayList<>(this.ms_file2psm.get(ms_file).size());

            // for un-recognized PSMs: for example, no MS2 mapped.
            HashMap<String, Integer> un_recognized_PSMs = new HashMap<>();
            HashSet<Integer> invalid_psm_indices = new HashSet<>();
            int row_i = -1;
            for (String line : this.ms_file2psm.get(ms_file)) {
                row_i++;
                String[] d = line.split("\t");
                String peptide = d[hIndex.get("Stripped.Sequence")];
                String modification = this.get_modification_diann(d[hIndex.get("Modified.Sequence")], peptide);
                int precursor_charge = Integer.parseInt(d[hIndex.get("Precursor.Charge")]);
                this.add_peptide(peptide, modification);
                String peptide_mod = peptide + "_" + modification;

                if (this.mod_ai.equalsIgnoreCase("-") || this.mod_ai.equalsIgnoreCase("general")) {
                    // nothing to do
                } else if (this.mod_ai.equalsIgnoreCase("phosphorylation")) {
                    String mod_seq = d[hIndex.get("Modified.Sequence")];
                    if (hIndex.containsKey("PTM.Site.Confidence") && mod_seq.contains("UniMod:21")) {
                        // only filtering out low confidence phosphorylation peptides
                        if (Double.parseDouble(d[hIndex.get("PTM.Site.Confidence")]) < this.ptm_site_prob_cutoff) {
                            invalid_psm_indices.add(row_i);
                            n_ptm_site_low_confidence++;
                            continue;
                        }
                        if (hIndex.containsKey("PTM.Q.Value")
                                && Double.parseDouble(d[hIndex.get("PTM.Q.Value")]) > this.ptm_site_qvalue_cutoff) {
                            invalid_psm_indices.add(row_i);
                            n_ptm_site_low_confidence++;
                            continue;
                        }
                        // new version of DIA-NN: Peptidoform.Q.Value
                        if (hIndex.containsKey("Peptidoform.Q.Value") && Double.parseDouble(d[hIndex.get("Peptidoform.Q.Value")]) > this.ptm_site_qvalue_cutoff) {
                            invalid_psm_indices.add(row_i);
                            n_ptm_site_low_confidence++;
                            continue;
                        }
                    }
                } else {
                    System.err.println("Modification type is not supported:" + this.mod_ai);
                    invalid_psm_indices.add(row_i);
                    System.exit(1);
                }

                if (!peptide2rt.containsKey(peptide_mod)) {
                    peptide2rt.put(peptide_mod, new PeptideRT());
                }
                peptide2rt.get(peptide_mod).peptide = peptide;
                peptide2rt.get(peptide_mod).modification = modification;
                peptide2rt.get(peptide_mod).rts.add(Double.parseDouble(d[hIndex.get("RT")])); // Apex RT
                peptide2rt.get(peptide_mod).scores.add(Double.parseDouble(d[hIndex.get("Q.Value")]));

                // for CCS
                if (ccs_enabled) {
                    String peptide_mode_charge = peptide + "_" + modification + "_" + precursor_charge;
                    if (!peptide2ccs.containsKey(peptide_mode_charge)) {
                        peptide2ccs.put(peptide_mode_charge, new PeptideCCS());
                    }
                    peptide2ccs.get(peptide_mode_charge).peptide = peptide;
                    peptide2ccs.get(peptide_mode_charge).modification = modification;
                    // In DIA-NN, iIM refers to the reference ion mobility in the spectral library, IM refers to the empirically measured.
                    peptide2ccs.get(peptide_mode_charge).ccs_values.add(Double.parseDouble(d[hIndex.get("IM")]));
                    peptide2ccs.get(peptide_mode_charge).scores.add(Double.parseDouble(d[hIndex.get("Q.Value")]));
                    peptide2ccs.get(peptide_mode_charge).charge = precursor_charge;
                }

                // Get the MS1 and MS2 m/z error from DIA-NN search result
                if (hIndex.containsKey("Precursor.Mz") && hIndex.containsKey("Ms1.Apex.Mz.Delta")) {
                    double precursor_mz = Double.parseDouble(d[hIndex.get("Precursor.Mz")]);
                    double ms1_mz_error = Double.parseDouble(d[hIndex.get("Ms1.Apex.Mz.Delta")]);
                    if (CParameter.tolu.startsWith("ppm")) {
                        // convert to ppm
                        ms1_mz_error = ms1_mz_error / precursor_mz * 1e6;
                        ms1_mz_errors.add(ms1_mz_error);
                    } else {
                        ms1_mz_errors.add(ms1_mz_error);
                    }
                }

                if (hIndex.containsKey("Best.Fr.Mz") && hIndex.containsKey("Best.Fr.Mz.Delta")) {
                    double best_fr_mz = Double.parseDouble(d[hIndex.get("Best.Fr.Mz")]);
                    double ms2_mz_error = Double.parseDouble(d[hIndex.get("Best.Fr.Mz.Delta")]);
                    if (CParameter.itolu.startsWith("ppm")) {
                        // convert to ppm
                        ms2_mz_error = ms2_mz_error / best_fr_mz * 1e6;
                        ms2_mz_errors.add(ms2_mz_error);
                    } else {
                        ms2_mz_errors.add(ms2_mz_error);
                    }
                }

                // save data for extracting XICs and MS2 spectra later using the rust library
                PSMQuery psm_query = new PSMQuery();
                psm_query.id = row_i;
                psm_query.mobility = Double.parseDouble(d[hIndex.get("IM")]);
                // second
                psm_query.rt_seconds = Double.parseDouble(d[hIndex.get("RT")]) * 60.0;
                Peptide peptide_obj = this.get_peptide(peptide, modification);
                double precursor_mz = dbGear.get_mz(peptide_obj.getMass(), precursor_charge);
                psm_query.precursor = precursor_mz;
                psm_query.precursor_charge = precursor_charge;
                // add 0,1,2 isotope peaks for precursor
                psm_query.precursor_isotopes.add(0);
                psm_query.precursor_isotopes.add(1);
                psm_query.precursor_isotopes.add(2);
                HashMap<Integer, ArrayList<Ion>> theoretical_ions = this.generate_theoretical_fragment_ions(peptide_obj,precursor_charge);
                HashSet<Integer> possible_fragment_ion_charges = this.getPossibleFragmentIonCharges(precursor_charge);
                // This is used to remove redundant ions
                HashSet<String> ion_names = new HashSet<>();
                String ion_id = "";
                for (int k : theoretical_ions.keySet()) {
                    for (Ion ion : theoretical_ions.get(k)) {
                        if (ion.getSubType() == PeptideFragmentIon.B_ION
                                || ion.getSubType() == PeptideFragmentIon.Y_ION) {
                            PeptideFragmentIon fragmentIon = ((PeptideFragmentIon) ion);
                            for (int frag_charge : possible_fragment_ion_charges) {
                                double frag_mz = fragmentIon.getTheoreticMz(frag_charge);
                                ion_id = fragmentIon.getNameWithNumber() + "_" + frag_mz;
                                if (ion_names.contains(ion_id)) {
                                    continue;
                                } else {
                                    ion_names.add(ion_id);
                                }
                                psm_query.fragments.add(frag_mz);
                                if (frag_charge == 1) {
                                    psm_query.fragment_labels.add(fragmentIon.getNameWithNumber());
                                } else {
                                    psm_query.fragment_labels.add(fragmentIon.getNameWithNumber() + "^" + frag_charge);
                                }
                            }
                        }
                    }
                }
                if (this.add_precursor_ion) {
                    // add precursor ion m/z for XIC extraction from MS2 spectra
                    psm_query.fragments.add(precursor_mz);
                    if (precursor_charge == 1) {
                        psm_query.fragment_labels.add("p");
                    } else {
                        psm_query.fragment_labels.add("p^" + precursor_charge);
                    }
                }
                psm_query_list.add(psm_query);
            }

            String psm_query_file = this.out_dir + File.separator + "psm_query.json";
            try (FileOutputStream fos = new FileOutputStream(psm_query_file);
                 BufferedOutputStream bos = new BufferedOutputStream(fos)) {
                JSON.writeTo(
                        bos,
                        psm_query_list,
                        JSONWriter.Feature.PrettyFormat,
                        JSONWriter.Feature.LargeObject);
            } catch (IOException e) {
                Cloger.getInstance().logger.error("Failed to write psm query file: " + psm_query_file);
                e.printStackTrace();
            }

            // get the median MS1 and MS2 m/z error
            double ms1_error_shift = 0.0;
            if (!ms1_mz_errors.isEmpty()) {
                ms1_error_shift = Quantiles.median().compute(ms1_mz_errors);
                Cloger.getInstance().logger.info("MS1 m/z error => error range: "+ Collections.min(ms1_mz_errors) + ", " + Collections.max(ms1_mz_errors) + ", median MS1 m/z error: "+ms1_error_shift+(is_fragment_ion_tolu_ppm?" ppm":" Da"));
            }
            double ms2_error_shift = 0.0;
            if (!ms2_mz_errors.isEmpty()) {
                ms2_error_shift = Quantiles.median().compute(ms2_mz_errors);
                Cloger.getInstance().logger.info("MS2 m/z error => error range: "+ Collections.min(ms2_mz_errors) + ", " + Collections.max(ms2_mz_errors) + ", Median MS2 m/z error: "+ms2_error_shift+(is_fragment_ion_tolu_ppm?" ppm":" Da"));
            }

            CallTimsQuery callTimsQuery = new CallTimsQuery();
            String spectra_result_dir = this.out_dir + File.separator + "psm_query_result/";
            callTimsQuery.rt_win = 0.1; // in minutes
            callTimsQuery.itolu = CParameter.itolu;
            callTimsQuery.itol = CParameter.itol;
            if (CParameter.tolu.equalsIgnoreCase(CParameter.itolu)) {
                callTimsQuery.itol_shift = Math.max(ms1_error_shift, ms2_error_shift);
            } else {
                callTimsQuery.itol_shift = ms2_error_shift;
            }
            callTimsQuery.run_ms2_spectra_query(ms_file, psm_query_file, spectra_result_dir);
            callTimsQuery.rt_win = CParameter.rt_win;
            String xic_result_dir = this.out_dir + File.separator + "xic_query_result/";
            callTimsQuery.run_xic_query(ms_file, psm_query_file, xic_result_dir);


            String spectra_result_file = spectra_result_dir + File.separator + "results.json";
            HashMap<Integer,PSMQueryResult> psm_query_results = new HashMap<>();
            // try (JSONReader reader = JSONReader.of(new FileReader(spectra_result_file))) {
            //     reader.startArray();
            //     while (!reader.isEnd()) {
            //         PSMQueryResult pqr = reader.read(PSMQueryResult.class);
            //         if (pqr != null) {
            //             psm_query_results.put(pqr.id, pqr);
            //         }
            //     }
            //     reader.endArray();
            // } catch (Exception e) {
            //     e.printStackTrace();
            //     System.exit(1);
            // }
            BufferedReader psm_br = new BufferedReader(new FileReader(spectra_result_file));
            String psm_line;
            while ((psm_line = psm_br.readLine()) != null) {
                PSMQueryResult obj = JSON.parseObject(psm_line, PSMQueryResult.class);
                psm_query_results.put(obj.id, obj);
            }
            psm_br.close();

            if (this.add_precursor_ion) {
                // get matched precursor ion intensities from psm_query_results
                int i_precursor_ion_matched = 0;
                for (int psm_idx : psm_query_results.keySet()) {
                    PSMQueryResult pqr = psm_query_results.get(psm_idx);
                    for (int i = 0; i < pqr.fragment_mzs.length; i++) {
                        double frag_mz = pqr.fragment_mzs[i];
                        if (frag_mz == pqr.precursor_mz) {
                            if (pqr.fragment_intensities[i] > 0) {
                                i_precursor_ion_matched++;
                                break;
                            }
                        }
                    }
                }
                Cloger.getInstance().logger.info("Precursor ion matched index in fragment ions:" + i_precursor_ion_matched);
                Cloger.getInstance().logger.info("Total PSM queries:" + psm_query_results.size());
            }

            String xic_result_file = xic_result_dir + File.separator + "results.json";
            HashMap<Integer, XICQueryResult> xic_query_results = new HashMap<>();
            BufferedReader xic_br = new BufferedReader(new FileReader(xic_result_file));
            String xic_line;
            while ((xic_line = xic_br.readLine()) != null) {
                XICQueryResult obj = JSON.parseObject(xic_line, XICQueryResult.class);
                xic_query_results.put(obj.id, obj);
            }
            xic_br.close();

            Cloger.getInstance().logger.info("Loading spectra query results done!");

            CCSDIAIndex diaIndex = new CCSDIAIndex();
            diaIndex.fragment_ion_intensity_threshold = this.fragment_ion_intensity_threshold;
            // diaIndex.meta = meta;
            // diaIndex.target_isolation_wins.add(isoWinID);
            // System.out.println("Isolation window:"+isoWinID);
            // diaIndex.index(ms_file);
            diaIndex.sg_smoothing_data_points = this.sg_smoothing_data_points;

            // First round of peak refinement
            HashMap<Integer, PeptideMatch> index2peptideMatch = new HashMap<>();
            row_i = -1;
            // for (String line : isoWinID2PSMs.get(isoWinID)) {
            for (String line : this.ms_file2psm.get(ms_file)) {
                row_i++;
                if (invalid_psm_indices.contains(row_i)) {
                    continue;
                }
                index2peptideMatch.put(row_i, new PeptideMatch());
                index2peptideMatch.get(row_i).id = String.valueOf(row_i);
                String[] d = line.split("\t");
                String peptide = d[hIndex.get("Stripped.Sequence")];
                String modification = this.get_modification_diann(d[hIndex.get("Modified.Sequence")], peptide);
                int precursor_charge = Integer.parseInt(d[hIndex.get("Precursor.Charge")]);
                // double apex_rt = Double.parseDouble(d[hIndex.get("apex_rt")]);
                double rt_start = Double.parseDouble(d[hIndex.get("RT.Start")]);
                double rt_end = Double.parseDouble(d[hIndex.get("RT.Stop")]);

                // The index (0-based) of apex scan in the original MS data file
                // this is not useful in this case for timsTOF data
                // int apex_scan=i2ms2index.get(line).ms2index; //= global_index2scan_num.get(Integer.parseInt(d[hIndex.get("MS2.Scan")])); // index
                // Spectrum spectrum = diaIndex.get_spectrum_by_scan(apex_scan);
                this.add_peptide(peptide, modification);
                Peptide peptideObj = this.get_peptide(peptide, modification);

                // intensity
                index2peptideMatch.get(row_i).ion_intensity_matrix = new double[peptide.length() - 1][n_ion_types];
                // this may not need
                index2peptideMatch.get(row_i).ion_mz_matrix = new double[peptide.length() - 1][n_ion_types];
                // 0: valid, >=1 invalid
                index2peptideMatch.get(row_i).ion_matrix = new int[peptide.length() - 1][n_ion_types];
                // index2peptideMatch.get(row_i).scan = apex_scan;
                index2peptideMatch.get(row_i).rt_start = rt_start;
                index2peptideMatch.get(row_i).rt_end = rt_end;
                index2peptideMatch.get(row_i).rt_apex = Double.parseDouble(d[hIndex.get("RT")]);
                index2peptideMatch.get(row_i).peptide_length = peptide.length();
                index2peptideMatch.get(row_i).precursor_charge = precursor_charge;
                index2peptideMatch.get(row_i).index = get_rt_index(xic_query_results.get(row_i).retention_time_results_seconds,true, index2peptideMatch.get(row_i).rt_apex);
                index2peptideMatch.get(row_i).peptide = peptideObj;
                index2peptideMatch.get(row_i).im = Double.parseDouble(d[hIndex.get("IM")]);

                // ArrayList<IonMatch> matched_ions = get_matched_ions(peptideObj, psm_query_results.get(row_i), precursor_charge, this.max_fragment_ion_charge, lossWaterNH3);
                // ArrayList<IonMatch> matched_ions = get_matched_ions(peptideObj, precursor_charge, 60.0*index2peptideMatch.get(row_i).rt_apex, xic_query_results.get(row_i), this.max_fragment_ion_charge, lossWaterNH3);
                ArrayList<IonMatch> matched_ions = new ArrayList<>();
                // double matched_apex_rt = get_matched_ions(peptideObj, precursor_charge, 60.0*index2peptideMatch.get(row_i).rt_apex, xic_query_results.get(row_i), this.max_fragment_ion_charge, lossWaterNH3, matched_ions);
                double matched_apex_rt = get_matched_ions(peptideObj, precursor_charge,
                        60.0*index2peptideMatch.get(row_i).rt_apex,
                        60.0*index2peptideMatch.get(row_i).rt_start,
                        60.0*index2peptideMatch.get(row_i).rt_end,
                        xic_query_results.get(row_i),
                        this.max_fragment_ion_charge,
                        lossWaterNH3,
                        matched_ions);
                if (matched_apex_rt > 0) {
                    index2peptideMatch.get(row_i).rt_apex = matched_apex_rt;
                }
                List<Double> matched_ion_mzs = new ArrayList<>();
                // b or y
                String ion_type = "";
                List<String> matched_ion_types = new ArrayList<>();
                // 1, 2, 3, ...
                List<Integer> matched_ion_numbers = new ArrayList<>();
                HashMap<Double, Double> mz2intensity = new HashMap<>();

                // max fragment ion intensity
                double max_fragment_ion_intensity = -1.0;
                int max_fragment_ion_row_index = -1;
                int max_fragment_ion_column_index = -1;

                if (!matched_ions.isEmpty()) {
                    // use psm_id not psm_i for now
                    //if (!this.scan2mz2count.containsKey(apex_scan)) {
                    //if (!this.scan2mz2count.containsKey(psm_id)) {
                        //this.scan2mz2count.put(apex_scan, new ConcurrentHashMap<>());
                    //    this.scan2mz2count.put(psm_id, new ConcurrentHashMap<>());
                    //}
                    index2peptideMatch.get(row_i).matched_ions = matched_ions;
                    for (IonMatch ionMatch : matched_ions) {
                        if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION
                                || ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION) {
                            // add fragment ion number
                            PeptideFragmentIon fragmentIon = ((PeptideFragmentIon) ionMatch.ion);
                            int ion_number = fragmentIon.getNumber();
                            int ion_type_column_index = this.get_ion_type_column_index(ionMatch);
                            // for y ion
                            if (ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION) {
                                fragment_ion_row_index = peptide.length() - ion_number - 1;
                                ion_type = "y";
                            } else if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION) {
                                fragment_ion_row_index = ion_number - 1;
                                ion_type = "b";
                            }else{
                                System.err.println("Unrecognized fragment ion type:"+ionMatch.ion.getSubType()+","+ionMatch.ion.getSubTypeAsString());
                                System.exit(1);
                            }

                            index2peptideMatch.get(row_i).mz2index.put(ionMatch.peakMz, new int[]{fragment_ion_row_index, ion_type_column_index});
                            index2peptideMatch.get(row_i).ion_intensity_matrix[fragment_ion_row_index][ion_type_column_index] = ionMatch.peakIntensity;
                            index2peptideMatch.get(row_i).ion_mz_matrix[fragment_ion_row_index][ion_type_column_index] = ionMatch.peakMz;
                            //if (this.scan2mz2count.get(apex_scan).containsKey(ionMatch.peakMz)) {
                            //    this.scan2mz2count.get(apex_scan).put(ionMatch.peakMz, this.scan2mz2count.get(apex_scan).get(ionMatch.peakMz) + 1);
                            //} else {
                            //    this.scan2mz2count.get(apex_scan).put(ionMatch.peakMz, 1);
                            //}
                            // use psm_id not apex_scan for now: psm_id is unique to each PSM.
                            //if (this.scan2mz2count.get(psm_id).containsKey(ionMatch.peakMz)) {
                            //    this.scan2mz2count.get(psm_id).put(ionMatch.peakMz, this.scan2mz2count.get(psm_id).get(ionMatch.peakMz) + 1);
                            //} else {
                            //    this.scan2mz2count.get(psm_id).put(ionMatch.peakMz, 1);
                            //}
                            matched_ion_mzs.add(ionMatch.peakMz);
                            matched_ion_types.add(ion_type);
                            matched_ion_numbers.add(ion_number);
                            mz2intensity.put(ionMatch.peakMz, ionMatch.peakIntensity);

                            // If the fragment ion number is <= the minimum number of fragment ion used for spectral library generation,
                            // we don't consider it in getting the max intensity of fragment ions.
                            if(use_all_peaks || (max_fragment_ion_intensity<=ionMatch.peakIntensity && ion_number >= this.lf_frag_n_min)){
                                max_fragment_ion_intensity = ionMatch.peakIntensity;
                                max_fragment_ion_row_index = fragment_ion_row_index;
                                max_fragment_ion_column_index = ion_type_column_index;
                            }
                        }
                    }
                }//else{
                    // invalid_psm_indices.add(row_i);
                    // System.out.println("No match: "+row_i + " => "+line);
                    // System.out.println("Match information: query apex RT="+index2peptideMatch.get(row_i).rt_apex+", return apex RT="+matched_apex_rt);
                //}
                if(!matched_ion_mzs.isEmpty()) {
                    index2peptideMatch.get(row_i).libSpectrum.spectrum.mz = new double[matched_ion_mzs.size()];
                    index2peptideMatch.get(row_i).libSpectrum.ion_types = new String[matched_ion_mzs.size()];
                    index2peptideMatch.get(row_i).libSpectrum.ion_numbers = new int[matched_ion_mzs.size()];
                    index2peptideMatch.get(row_i).libSpectrum.spectrum.intensity = new double[matched_ion_mzs.size()];
                    for (int i = 0; i < matched_ion_mzs.size(); i++) {
                        index2peptideMatch.get(row_i).libSpectrum.spectrum.mz[i] = matched_ion_mzs.get(i);
                        index2peptideMatch.get(row_i).libSpectrum.ion_types[i] = matched_ion_types.get(i);
                        index2peptideMatch.get(row_i).libSpectrum.ion_numbers[i] = matched_ion_numbers.get(i);
                        index2peptideMatch.get(row_i).libSpectrum.spectrum.intensity[i] = mz2intensity.get(matched_ion_mzs.get(i));
                    }
                    index2peptideMatch.get(row_i).max_fragment_ion_intensity = max_fragment_ion_intensity;
                    index2peptideMatch.get(row_i).max_fragment_ion_row_index = max_fragment_ion_row_index;
                    index2peptideMatch.get(row_i).max_fragment_ion_column_index = max_fragment_ion_column_index;
                }//else{
                    // invalid_psm_indices.add(row_i);
                //}

            }

            // refine peak apex using refined peak detection.
            // use the apex fragment ion matches for downstream analysis
            row_i = -1;
            // for (String line : isoWinID2PSMs.get(isoWinID)) {
            for (String line : this.ms_file2psm.get(ms_file)) {
                row_i++;
                if (invalid_psm_indices.contains(row_i)) {
                    continue;
                }
                PeptideMatch peptideMatch = index2peptideMatch.get(row_i);

                String[] d = line.split("\t");
                String peptide = d[hIndex.get("Stripped.Sequence")];

                // intensity
                peptideMatch.ion_intensity_matrix = new double[peptide.length() - 1][n_ion_types];
                peptideMatch.ion_mz_matrix = new double[peptide.length() - 1][n_ion_types];
                // 0: valid, >=1 invalid
                peptideMatch.ion_matrix = new int[peptide.length() - 1][n_ion_types];
                peptideMatch.index = get_rt_index(xic_query_results.get(row_i).retention_time_results_seconds,true, index2peptideMatch.get(row_i).rt_apex);


                // for testing
                if(test_mode){
                    index2peptideMatch.get(row_i).ion_matrix_map.put("spectrum_centric", new int[peptide.length() - 1][n_ion_types]);
                    index2peptideMatch.get(row_i).ion_matrix_map.put("peptide_centric_cor", new int[peptide.length() - 1][n_ion_types]);
                    index2peptideMatch.get(row_i).ion_matrix_map.put("peptide_centric_shape", new int[peptide.length() - 1][n_ion_types]);
                    index2peptideMatch.get(row_i).ion_matrix_map.put("low_mass", new int[peptide.length() - 1][n_ion_types]);
                    index2peptideMatch.get(row_i).ion_matrix_map.put("peptide_centric", new int[peptide.length() - 1][n_ion_types]);
                }

                // xic_query_ccs(xic_query_results.get(row_i),psm_query_results.get(row_i),peptideMatch,diaIndex);
                // refine peak apex and peak boundaries
                xic_query_ccs(xic_query_results.get(row_i),index2peptideMatch.get(row_i).libSpectrum.spectrum.mz,peptideMatch,diaIndex);
                ArrayList<IonMatch> matched_ions = new ArrayList<>();
                // get the matched ions again using the refined apex RT
                double matched_apex_rt = get_matched_ions(index2peptideMatch.get(row_i).peptide, index2peptideMatch.get(row_i).precursor_charge, 60.0*index2peptideMatch.get(row_i).rt_apex, xic_query_results.get(row_i), this.max_fragment_ion_charge, lossWaterNH3, matched_ions);
                if(matched_apex_rt > 0){
                    index2peptideMatch.get(row_i).rt_apex = matched_apex_rt;
                }
                List<Double> matched_ion_mzs = new ArrayList<>();
                // b or y
                String ion_type = "";
                List<String> matched_ion_types = new ArrayList<>();
                // 1, 2, 3, ...
                List<Integer> matched_ion_numbers = new ArrayList<>();
                HashMap<Double, Double> mz2intensity = new HashMap<>();

                // max fragment ion intensity
                double max_fragment_ion_intensity = -1.0;
                int max_fragment_ion_row_index = -1;
                int max_fragment_ion_column_index = -1;

                if (!matched_ions.isEmpty()) {
                    // use psm_id not psm_i for now
                    // if (!this.scan2mz2count.containsKey(apex_scan)) {
                    if (!this.scan2mz2count.containsKey(psm_id)) {
                        // this.scan2mz2count.put(apex_scan, new ConcurrentHashMap<>());
                        this.scan2mz2count.put(psm_id, new ConcurrentHashMap<>());
                    }
                    index2peptideMatch.get(row_i).matched_ions = matched_ions;
                    for (IonMatch ionMatch : matched_ions) {
                        if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION
                                || ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION) {
                            // add fragment ion number
                            PeptideFragmentIon fragmentIon = ((PeptideFragmentIon) ionMatch.ion);
                            int ion_number = fragmentIon.getNumber();
                            int ion_type_column_index = this.get_ion_type_column_index(ionMatch);
                            // for y ion
                            if(ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION){
                                fragment_ion_row_index = index2peptideMatch.get(row_i).peptide.getSequence().length() - ion_number - 1;
                                ion_type = "y";
                            } else if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION) {
                                fragment_ion_row_index = ion_number - 1;
                                ion_type = "b";
                            }else{
                                System.err.println("Unrecognized fragment ion type:"+ionMatch.ion.getSubType()+","+ionMatch.ion.getSubTypeAsString());
                                System.exit(1);
                            }

                            index2peptideMatch.get(row_i).mz2index.put(ionMatch.peakMz, new int[]{fragment_ion_row_index, ion_type_column_index});
                            index2peptideMatch.get(row_i).ion_intensity_matrix[fragment_ion_row_index][ion_type_column_index] = ionMatch.peakIntensity;
                            index2peptideMatch.get(row_i).ion_mz_matrix[fragment_ion_row_index][ion_type_column_index] = ionMatch.peakMz;
                            //if (this.scan2mz2count.get(apex_scan).containsKey(ionMatch.peakMz)) {
                            //    this.scan2mz2count.get(apex_scan).put(ionMatch.peakMz, this.scan2mz2count.get(apex_scan).get(ionMatch.peakMz) + 1);
                            //} else {
                            //    this.scan2mz2count.get(apex_scan).put(ionMatch.peakMz, 1);
                            //}
                            // use psm_id not apex_scan for now: psm_id is unique to each PSM.
                            if (this.scan2mz2count.get(psm_id).containsKey(ionMatch.peakMz)) {
                                this.scan2mz2count.get(psm_id).put(ionMatch.peakMz, this.scan2mz2count.get(psm_id).get(ionMatch.peakMz) + 1);
                            } else {
                                this.scan2mz2count.get(psm_id).put(ionMatch.peakMz, 1);
                            }
                            matched_ion_mzs.add(ionMatch.peakMz);
                            matched_ion_types.add(ion_type);
                            matched_ion_numbers.add(ion_number);
                            mz2intensity.put(ionMatch.peakMz, ionMatch.peakIntensity);

                            // If the fragment ion number is <= the minimum number of fragment ion used for spectral library generation,
                            // we don't consider it in getting the max intensity of fragment ions.
                            if (use_all_peaks || (max_fragment_ion_intensity <= ionMatch.peakIntensity
                                    && ion_number >= this.lf_frag_n_min)) {
                                max_fragment_ion_intensity = ionMatch.peakIntensity;
                                max_fragment_ion_row_index = fragment_ion_row_index;
                                max_fragment_ion_column_index = ion_type_column_index;
                            }
                        }
                    }
                } else {
                    index2peptideMatch.get(row_i).matched_ions = new ArrayList<>();
                    invalid_psm_indices.add(row_i);
                    if (CParameter.verbose == CParameter.VerboseType.DEBUG) {
                        Cloger.getInstance().logger.warn("No match: "+row_i + " => "+line);
                        Cloger.getInstance().logger.warn("Match information: query apex RT="+index2peptideMatch.get(row_i).rt_apex+", return apex RT="+matched_apex_rt);
                    }
                }
                if (!matched_ion_mzs.isEmpty()) {
                    index2peptideMatch.get(row_i).libSpectrum.spectrum.mz = new double[matched_ion_mzs.size()];
                    index2peptideMatch.get(row_i).libSpectrum.ion_types = new String[matched_ion_mzs.size()];
                    index2peptideMatch.get(row_i).libSpectrum.ion_numbers = new int[matched_ion_mzs.size()];
                    index2peptideMatch.get(row_i).libSpectrum.spectrum.intensity = new double[matched_ion_mzs.size()];
                    for (int i = 0; i < matched_ion_mzs.size(); i++) {
                        index2peptideMatch.get(row_i).libSpectrum.spectrum.mz[i] = matched_ion_mzs.get(i);
                        index2peptideMatch.get(row_i).libSpectrum.ion_types[i] = matched_ion_types.get(i);
                        index2peptideMatch.get(row_i).libSpectrum.ion_numbers[i] = matched_ion_numbers.get(i);
                        index2peptideMatch.get(row_i).libSpectrum.spectrum.intensity[i] = mz2intensity.get(matched_ion_mzs.get(i));
                    }
                    index2peptideMatch.get(row_i).max_fragment_ion_intensity = max_fragment_ion_intensity;
                    index2peptideMatch.get(row_i).max_fragment_ion_row_index = max_fragment_ion_row_index;
                    index2peptideMatch.get(row_i).max_fragment_ion_column_index = max_fragment_ion_column_index;
                } else {
                    invalid_psm_indices.add(row_i);
                }

            }

            // Infer shared fragment ions based on the apex scan match
            row_i = -1;
            for (String line : this.ms_file2psm.get(ms_file)) {
                row_i = row_i + 1;
                if (invalid_psm_indices.contains(row_i)) {
                    continue;
                }
                ArrayList<IonMatch> matched_ions = index2peptideMatch.get(row_i).matched_ions;
                int apex_scan = index2peptideMatch.get(row_i).scan;
                if (!matched_ions.isEmpty()) {
                    for (IonMatch ionMatch : matched_ions) {
                        if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION
                                || ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION) {
                            // add fragment ion number
                            PeptideFragmentIon fragmentIon = ((PeptideFragmentIon) ionMatch.ion);
                            int ion_number = fragmentIon.getNumber();
                            // for y ion
                            if (ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION) {
                                fragment_ion_row_index = index2peptideMatch.get(row_i).peptide_length - ion_number - 1;
                            } else if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION) {
                                fragment_ion_row_index = ion_number - 1;
                            }else{
                                System.err.println("Unrecognized fragment ion type:"+ionMatch.ion.getSubType()+","+ionMatch.ion.getSubTypeAsString());
                                System.exit(1);
                            }
                            int ion_type_column_index = this.get_ion_type_column_index(ionMatch);
                            if(this.scan2mz2count.containsKey(apex_scan) && this.scan2mz2count.get(apex_scan).containsKey(ionMatch.peakMz)){
                                index2peptideMatch.get(row_i).ion_matrix[fragment_ion_row_index][ion_type_column_index] = this.scan2mz2count.get(apex_scan).get(ionMatch.peakMz) - 1;
                            }else{
                                index2peptideMatch.get(row_i).ion_matrix[fragment_ion_row_index][ion_type_column_index] = 0;
                            }
                            if(test_mode){
                                if(this.scan2mz2count.containsKey(apex_scan) && this.scan2mz2count.get(apex_scan).containsKey(ionMatch.peakMz)){
                                    index2peptideMatch.get(row_i).ion_matrix_map.get("spectrum_centric")[fragment_ion_row_index][ion_type_column_index] = this.scan2mz2count.get(apex_scan).get(ionMatch.peakMz) - 1;
                                }else{
                                    index2peptideMatch.get(row_i).ion_matrix_map.get("spectrum_centric")[fragment_ion_row_index][ion_type_column_index] = 0;
                                }
                            }
                        }
                    }
                }

            }

            // Infer shared fragment ions based on the fragment ion correlation
            // System.out.println("Peptide to query:"+index2peptideMatch.size());
            // index2peptideMatch.values().parallelStream().forEach(peptideMatch ->
            // xic_query_ccs(diaIndex,peptideMatch,isoWinID));
            row_i = -1;
            int[] ind = new int[] { 0, 0 };
            // for (String line : isoWinID2PSMs.get(isoWinID)) {
            for (String line : this.ms_file2psm.get(ms_file)) {
                row_i = row_i + 1;
                if (invalid_psm_indices.contains(row_i)) {
                    continue;
                }
                PeptideMatch peptideMatch = index2peptideMatch.get(row_i);
                // xic_query_ccs(xic_query_results.get(row_i),psm_query_results.get(row_i),peptideMatch,diaIndex);
                xic_query_ccs(xic_query_results.get(row_i),index2peptideMatch.get(row_i).libSpectrum.spectrum.mz,peptideMatch,diaIndex);
                HashSet<Double> high_cor_mzs = new HashSet<>();
                double max_cor_mz = 0;
                double max_frag_cor = -100;
                for (double mz : peptideMatch.mz2cor.keySet()) {
                    if (peptideMatch.mz2cor.get(mz) >= this.cor_cutoff) {
                        high_cor_mzs.add(mz);
                    }
                    if (peptideMatch.mz2cor.get(mz) > max_frag_cor) {
                        max_frag_cor = peptideMatch.mz2cor.get(mz);
                        max_cor_mz = mz;
                    }
                }
                peptideMatch.max_cor_mz = max_cor_mz;
                for (double mz : peptideMatch.mz2index.keySet()) {
                    if (!high_cor_mzs.contains(mz)) {
                        ind = peptideMatch.mz2index.get(mz);
                        peptideMatch.ion_matrix[ind[0]][ind[1]] = peptideMatch.ion_matrix[ind[0]][ind[1]] + 1;
                        if (test_mode) {
                            peptideMatch.ion_matrix_map.get("peptide_centric_cor")[ind[0]][ind[1]] = 1;
                            peptideMatch.ion_matrix_map.get("peptide_centric")[ind[0]][ind[1]] = 1;
                        }
                    }
                }
            }

            // Infer shared fragment ions based on the fragment ion shape
            row_i = -1;
            for (String line : this.ms_file2psm.get(ms_file)) {
                row_i = row_i + 1;
                if (invalid_psm_indices.contains(row_i)) {
                    continue;
                }
                PeptideMatch peptideMatch = index2peptideMatch.get(row_i);
                for (double mz : peptideMatch.mz2index.keySet()) {
                    if (peptideMatch.mz2skewed_peaks.containsKey(mz) && peptideMatch.mz2skewed_peaks.get(mz) >= 2) {
                        ind = peptideMatch.mz2index.get(mz);
                        peptideMatch.ion_matrix[ind[0]][ind[1]] = peptideMatch.ion_matrix[ind[0]][ind[1]] + 1;
                        if (test_mode) {
                            peptideMatch.ion_matrix_map.get("peptide_centric_shape")[ind[0]][ind[1]] = 1;
                            peptideMatch.ion_matrix_map.get("peptide_centric")[ind[0]][ind[1]] = 1;
                        }
                    }
                }
            }

            // low mass fragment ions
            // based on fragment ion m/z or ion number (such as b-1, b-2, y-1, y-2)
            row_i = -1;
            for (String line : this.ms_file2psm.get(ms_file)) {
                String[] d = line.split("\t");
                row_i = row_i + 1;
                if (invalid_psm_indices.contains(row_i)) {
                    continue;
                }
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
                            if (ion.getSubType() == PeptideFragmentIon.Y_ION) {
                                fragment_ion_row_index = index2peptideMatch.get(row_i).peptide_length - ion_number - 1;
                                if (ion_number == 1) {
                                    is_y1 = true;
                                }
                            } else if (ion.getSubType() == PeptideFragmentIon.B_ION) {
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

                                if (this.n_ion_min >= 1 && ion.getSubType() == PeptideFragmentIon.B_ION
                                        && ion_number <= this.n_ion_min) {
                                    int ion_type_column_index = this.get_ion_type_column_index(ion, frag_ion_charge);
                                    double frag_ion_mz = index2peptideMatch.get(row_i).ion_mz_matrix[fragment_ion_row_index][ion_type_column_index];
                                    double frag_ion_cor = 0.0;
                                    double mz_skewness = 1;
                                    if (index2peptideMatch.get(row_i).mz2cor.containsKey(frag_ion_mz)) {
                                        frag_ion_cor = index2peptideMatch.get(row_i).mz2cor.get(frag_ion_mz);
                                        mz_skewness = index2peptideMatch.get(row_i).mz2skewed_peaks.get(frag_ion_mz);
                                    }
                                    if(!(frag_ion_cor > 0.9 && mz_skewness <= 1)){
                                        if(index2peptideMatch.get(row_i).ion_intensity_matrix[fragment_ion_row_index][ion_type_column_index]/index2peptideMatch.get(row_i).max_fragment_ion_intensity >=0.5) {
                                            index2peptideMatch.get(row_i).ion_matrix[fragment_ion_row_index][ion_type_column_index] = index2peptideMatch.get(row_i).ion_matrix[fragment_ion_row_index][ion_type_column_index] + 1;
                                        }
                                    }
                                } else if (this.c_ion_min >= 1 && ion.getSubType() == PeptideFragmentIon.Y_ION
                                        && ion_number <= this.c_ion_min) {
                                    int ion_type_column_index = this.get_ion_type_column_index(ion, frag_ion_charge);
                                    double frag_ion_mz = index2peptideMatch.get(row_i).ion_mz_matrix[fragment_ion_row_index][ion_type_column_index];
                                    double frag_ion_cor = 0.0;
                                    double mz_skewness = 1;
                                    if (index2peptideMatch.get(row_i).mz2cor.containsKey(frag_ion_mz)) {
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
            for (String line : this.ms_file2psm.get(ms_file)) {
                n_total_matches++;
                row_i = row_i + 1;
                if (invalid_psm_indices.contains(row_i)) {
                    continue;
                }
                String[] d = line.split("\t");
                if (hIndex.containsKey("peak_overlap")) {
                    if (Integer.parseInt(d[hIndex.get("peak_overlap")]) >= 1) {
                        n_peak_overlap++;
                        continue;
                    }
                }

                if (this.mod_ai.equalsIgnoreCase("-") || this.mod_ai.equalsIgnoreCase("general")) {
                    // nothing to do
                } else if (this.mod_ai.equalsIgnoreCase("phosphorylation")) {
                    String mod_seq = d[hIndex.get("Modified.Sequence")];
                    if (hIndex.containsKey("PTM.Site.Confidence") && mod_seq.contains("UniMod:21")) {
                        // only filtering out low confidence phosphorylation peptides
                        if (Double.parseDouble(d[hIndex.get("PTM.Site.Confidence")]) < this.ptm_site_prob_cutoff) {
                            n_ptm_site_low_confidence++;
                            continue;
                        }
                        if (hIndex.containsKey("PTM.Q.Value")
                                && Double.parseDouble(d[hIndex.get("PTM.Q.Value")]) > this.ptm_site_qvalue_cutoff) {
                            n_ptm_site_low_confidence++;
                            continue;
                        }
                        // new version of DIA-NN: Peptidoform.Q.Value
                        if (hIndex.containsKey("Peptidoform.Q.Value") && Double.parseDouble(d[hIndex.get("Peptidoform.Q.Value")]) > this.ptm_site_qvalue_cutoff) {
                            n_ptm_site_low_confidence++;
                            continue;
                        }
                    }
                } else {
                    System.err.println("Modification type is not supported:" + this.mod_ai);
                    System.exit(1);
                }

                if (index2peptideMatch.get(row_i).max_fragment_ion_intensity > 0
                        && index2peptideMatch.get(row_i).matched_ions.size() >= this.min_n_fragment_ions) {
                    boolean fragment_export = false;

                    String [] out_mod = convert_modification(this.get_modification_diann(d[hIndex.get("Modified.Sequence")],d[hIndex.get("Stripped.Sequence")]));
                    int n_valid_fragment_ions = get_n_valid_fragment_ions(index2peptideMatch.get(row_i).ion_intensity_matrix,index2peptideMatch.get(row_i).ion_matrix);
                    int n_total_fragment_ions = get_n_matched_fragment_ions(index2peptideMatch.get(row_i).ion_intensity_matrix);
                    if(n_valid_fragment_ions >= this.min_n_high_quality_fragment_ions) {

                        // get adjacent scans
                        // ArrayList<PeptideMatch> pMatches = get_adjacent_ms2_matches_ccs(index2peptideMatch.get(row_i),this.n_flank_scans,diaIndex,isoWinID);
                        ArrayList<PeptideMatch> pMatches = new ArrayList<>();
                        if (this.n_flank_scans >= 1 && pMatches.isEmpty()) {
                            // TODO: don't remove this line
                            // System.out.println("Ignore row:"+row_i+" => "+line);
                            // continue;
                        }

                        String spectrum_title = hIndex.containsKey("MS2.Scan")?d[hIndex.get("MS2.Scan")]:""+row_i;
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
                            psmWriter.write(index2peptideMatch.get(row_i).id+"\t"+
                                    spectrum_title+"\t"+
                                    index2peptideMatch.get(row_i).scan+ "\t" +
                                    pdv_precursor_mz +"\t" +
                                    pdv_precursor_charge +"\t" +
                                    meta.get_ce_for_mz(pdv_precursor_mz) + "\t" +
                                    pdv_peptide + "\t" +
                                    pdv_modification + "\t" +
                                    out_mod[0] + "\t" +
                                    out_mod[1] + "\t1\t" +
                                    index2peptideMatch.get(row_i).max_cor_mz + "\t" +
                                    frag_start_idx + "\t" +
                                    frag_stop_idx + "\t" +
                                    n_valid_fragment_ions + "\t" +
                                    n_total_fragment_ions + "\t1\n");
                            if (!pMatches.isEmpty()) {
                                for (PeptideMatch pMatch : pMatches) {
                                    n_total_psm_matches_valid++;
                                    frag_start_idx = frag_stop_idx;
                                    frag_stop_idx = frag_start_idx + pMatch.ion_intensity_matrix.length;
                                    n_total_fragment_ions = get_n_matched_fragment_ions(pMatch.ion_intensity_matrix);
                                    // TODO: update spectrum_title
                                    psmWriter.write(index2peptideMatch.get(row_i).id+"-"+pMatch.scan+"\t"+spectrum_title +"\t"+pMatch.scan+ "\t" +
                                            pdv_precursor_mz + "\t" +
                                            pdv_precursor_charge + "\t" +
                                            meta.get_ce_for_mz(pdv_precursor_mz) + "\t" +
                                            pdv_peptide + "\t" + pdv_modification + "\t" + out_mod[0] + "\t" + out_mod[1] + "\t1\t" + index2peptideMatch.get(row_i).max_cor_mz + "\t" + frag_start_idx + "\t" + frag_stop_idx +
                                            "\t" + n_valid_fragment_ions + "\t" + n_total_fragment_ions + "\t1\n");
                                }
                            }
                            fragment_export = true;
                        } else {
                            n_total_matches_max_fragment_ion_invalid++;
                            invalid_max_fragment_ion_intensity_indices.add(row_i);
                            if (!this.export_valid_matches_only) {
                                frag_start_idx = frag_stop_idx;
                                frag_stop_idx = frag_start_idx + index2peptideMatch.get(row_i).ion_intensity_matrix.length;
                                psmWriter.write(index2peptideMatch.get(row_i).id+"\t"+spectrum_title+ "\t"+index2peptideMatch.get(row_i).scan+ "\t" +
                                        pdv_precursor_mz +"\t" +
                                        pdv_precursor_charge +"\t" +
                                        meta.get_ce_for_mz(pdv_precursor_mz) + "\t" +
                                        pdv_peptide + "\t" +pdv_modification+ "\t" + out_mod[0] + "\t" + out_mod[1] + "\t0\t" + index2peptideMatch.get(row_i).max_cor_mz + "\t" + frag_start_idx + "\t" + frag_stop_idx +
                                        "\t" + n_valid_fragment_ions + "\t" + n_total_fragment_ions + "\t0\n");
                                if(!pMatches.isEmpty()){
                                    for(PeptideMatch pMatch: pMatches){
                                        frag_start_idx = frag_stop_idx;
                                        frag_stop_idx = frag_start_idx + pMatch.ion_intensity_matrix.length;
                                        n_total_fragment_ions = get_n_matched_fragment_ions(pMatch.ion_intensity_matrix);
                                        psmWriter.write(index2peptideMatch.get(row_i).id+"-"+pMatch.scan+"\t"+spectrum_title + "\t"+pMatch.scan+"\t" +
                                                pdv_precursor_mz + "\t" +
                                                pdv_precursor_charge + "\t" +
                                                meta.get_ce_for_mz(pdv_precursor_mz) + "\t" +
                                                pdv_peptide + "\t" + pdv_modification + "\t" + out_mod[0] + "\t" + out_mod[1] + "\t1\t" + index2peptideMatch.get(row_i).max_cor_mz + "\t" + frag_start_idx + "\t" + frag_stop_idx +
                                                "\t" + n_valid_fragment_ions + "\t" + n_total_fragment_ions + "\t0\n");
                                    }
                                }
                                fragment_export = true;
                            }
                        }

                        // int apex_scan = i2ms2index.get(line).ms2index;//= global_index2scan_num.get(Integer.parseInt(d[hIndex.get("MS2.Scan")]));
                        // String spectrum_title = d[hIndex.get("MS2.Scan")];
                        if (!save_spectra.contains(spectrum_title)) {
                            // Spectrum spectrum = diaIndex.get_spectrum_by_scan(apex_scan);
                            //int charge = Integer.parseInt(d[hIndex.get("Precursor.Charge")]);
                            if(this.export_spectra_to_mgf) {
                               // msWriter.write(MgfUtils.asMgf(spectrum, spectrum_title, charge, String.valueOf(apex_scan)) + "\n");
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
                            if (!pMatches.isEmpty()) {
                                for (PeptideMatch pMatch : pMatches) {
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
                                if (test_mode) {
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
                            if (!pMatches.isEmpty()) {
                                for (PeptideMatch pMatch : pMatches) {
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
                            for (IonMatch ionMatch : peptideMatch.matched_ions) {
                                this.fragment_ions_mz_tol.add(ionMatch.getError(is_fragment_ion_tolu_ppm));
                            }
                        }
                    } else {
                        n_less_than_min_n_high_quality_fragment_ions++;
                        invalid_min_n_high_quality_fragment_ions_indices.add(row_i);
                    }
                } else {
                    n_less_than_min_n_fragment_ions++;
                }

            }
            if (!un_recognized_PSMs.isEmpty()) {
                int n_un_recognized_PSMs = 0;
                for (String line : un_recognized_PSMs.keySet()) {
                    if (un_recognized_PSMs.get(line) == 0) {
                        n_un_recognized_PSMs++;
                        if (CParameter.verbose == CParameter.VerboseType.DEBUG) {
                            Cloger.getInstance().logger.warn("Spectrum not found:" + line);
                        }
                    }
                }
                if (n_un_recognized_PSMs >= 1) {
                    if (CParameter.verbose == CParameter.VerboseType.DEBUG) {
                        Cloger.getInstance().logger.warn("Spectrum not found:" + n_un_recognized_PSMs);
                    }
                }
            }

            // for debug
            if (export_xic) {
                // export the xic query input
                List<PSMQuery> tmp_psm_query_list = new ArrayList<>();
                for (PSMQuery psm_q : psm_query_list) {
                    if (invalid_max_fragment_ion_intensity_indices.contains(psm_q.id)) {
                        tmp_psm_query_list.add(psm_q);
                    }
                }
                if (!tmp_psm_query_list.isEmpty()) {
                    // Pretty-printed JSON string
                    String tmp_json = JSON.toJSONString(tmp_psm_query_list, JSONWriter.Feature.PrettyFormat, JSONWriter.Feature.LargeObject);
                    // Write to file
                    String tmp_psm_query_file = this.out_dir + File.separator + "invalid_max_fragment_ion_intensity_psm_query.json";
                    try (FileWriter writer = new FileWriter(tmp_psm_query_file)) {
                        writer.write(tmp_json);
                    }
                    tmp_psm_query_list.clear();
                }

                for (PSMQuery psm_q : psm_query_list) {
                    if (invalid_min_n_high_quality_fragment_ions_indices.contains(psm_q.id)) {
                        tmp_psm_query_list.add(psm_q);
                    }
                }
                if (!tmp_psm_query_list.isEmpty()) {
                    // Pretty-printed JSON string
                    String tmp_json = JSON.toJSONString(tmp_psm_query_list, JSONWriter.Feature.PrettyFormat, JSONWriter.Feature.LargeObject);
                    // Write to file
                    String tmp_psm_query_file = this.out_dir + File.separator + "invalid_min_n_high_quality_fragment_ions_psm_query.json";
                    try (FileWriter writer = new FileWriter(tmp_psm_query_file)) {
                        writer.write(tmp_json);
                    }
                    tmp_psm_query_list.clear();
                }
            }
        }

        psmWriter.close();
        msWriter.close();
        fragWriter.close();
        fragValidWriter.close();
        if (this.export_fragment_ion_mz_to_file && fragMzWriter != null) {
            fragMzWriter.close();
        }
        if (this.export_skyline_transition_list_file && tfWriter != null && tbWriter != null) {
            tfWriter.close();
            tbWriter.close();
        }

        if (export_xic) {
            xicWriter.write("\n}");
            xicWriter.close();
        }

        if (test_mode) {
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
        generate_rt_train_data(peptide2rt, rt_merge_method, this.out_dir + "/rt_train_data.tsv");
        generate_ccs_train_data(peptide2ccs, ccs_merge_method, this.out_dir + "/ccs_train_data.tsv");
        CParameter.fragment_ion_intensity_cutoff = original_fragment_ion_intensity_cutoff;

    }

    /**
     * Get the index of the retention time closest to the apex retention time.
     * 
     * @param rts     List of retention times
     * @param apex_rt Apex retention time
     * @return Index of the closest retention time
     */
    public int get_rt_index(double[] rts, boolean unit_second, double apex_rt) {
        double min_diff = Double.POSITIVE_INFINITY;
        int index = -1;
        double rt;
        for (int i = 0; i < rts.length; i++) {
            if (unit_second) {
                rt = rts[i] / 60.0;
            } else {
                rt = rts[i];
            }
            if (Math.abs(rt - apex_rt) < min_diff) {
                min_diff = Math.abs(rt - apex_rt);
                index = i;
            }
        }
        return index;
    }

    /**
     * Export MS2 spectra in MGF format
     */
    public boolean export_ccs_mgf = false;

    /**
     * A record to store MS2 spectrum metadata.
     * 
     * @param index           Spectrum index
     * @param precursor_rt    Precursor retention time
     * @param precursor_im    Precursor ion mobility
     * @param isolation_mz    Isolation m/z
     * @param isolation_width Isolation width
     */
    public record MS2SpectraMeta(int index,
            double precursor_rt,
            double precursor_im,
            double isolation_mz,
            double isolation_width) {
    }

    /**
     * A record to store MS2 spectra data.
     * 
     * @param index           Spectrum index
     * @param mz_values       m/z values of the spectrum
     * @param intensities     Intensities of the spectrum
     * @param precursor_rt    Precursor retention time
     * @param precursor_im    Precursor ion mobility
     * @param isolation_mz    Isolation m/z
     * @param isolation_width Isolation width
     */
    public record MS2SpectraAll(int index,
            ArrayList<Double> mz_values,
            ArrayList<Double> intensities,
            double precursor_rt,
            double precursor_im,
            double isolation_mz,
            double isolation_width) {
    }

    /**
     * Get the apex MS2 scan information for each peptide match.
     * 
     * @param matches List of matches
     * @param ms_file Path to the MS file
     * @param dbGear  A DBGear instance for database access
     * @param out_dir Output directory for results
     * @return A map of matches to their corresponding MS2 spectra metadata
     */
    public ConcurrentHashMap<String, ApexMatch> get_ms2spectrum_index(ArrayList<String> matches, String ms_file, DBGear dbGear, String out_dir) throws IOException {
        // ConcurrentHashMap<String, Integer> index2index = new ConcurrentHashMap<>();
        ConcurrentHashMap<String, ApexMatch> index2index = new ConcurrentHashMap<>();
        HashMap<Integer, HashMap<String, Double>> index = new HashMap<>();
        CarpetReader<MS2SpectraMeta> reader = new CarpetReader<>(new File(ms_file), MS2SpectraMeta.class);
        for (MS2SpectraMeta spectrum : reader) {
            index.put(spectrum.index, new HashMap<>());
            index.get(spectrum.index).put("ccs", spectrum.precursor_im);
            index.get(spectrum.index).put("rt", spectrum.precursor_rt / 60);
            index.get(spectrum.index).put("isolation_mz", spectrum.isolation_mz);
            index.get(spectrum.index).put("isolation_width", spectrum.isolation_width);
        }
        double ccs_cutoff = 0.05;
        File F = new File(ms_file);
        String out_prefix = F.getName();
        if (out_prefix.endsWith(".mzML")) {
            out_prefix = out_prefix.replaceAll(".mzML", "");
        } else if (out_prefix.endsWith(".mzml")) {
            out_prefix = out_prefix.replaceAll(".mzml", "");
        } else if (out_prefix.endsWith(".parquet")) {
            out_prefix = out_prefix.replaceAll(".parquet", "");
        }
        String out_file = out_dir + File.separator + out_prefix + "_apex_ms2spectra.tsv";
        BufferedWriter writer = new BufferedWriter(new FileWriter(out_file));
        Cloger.getInstance().logger.info("Export Apex MS2 mapping information to file: " + out_file);
        writer.write("row_index\tRT\tdelta_rt\tdelta_ccs\tprecursor_mz\tms2index\tisolation_mz\tMS2.Scan\n");
        IntStream.range(0, matches.size()).parallel().forEach(k -> {
            String[] d = matches.get(k).split("\t");
            String peptide = d[hIndex.get("Stripped.Sequence")];
            String modification = this.get_modification_diann(d[hIndex.get("Modified.Sequence")], peptide);
            int precursor_charge = Integer.parseInt(d[hIndex.get("Precursor.Charge")]);
            double rt = Double.parseDouble(d[hIndex.get("RT")]);
            double ccs = Double.parseDouble(d[hIndex.get("IM")]);
            synchronized (this) {
                this.add_peptide(peptide, modification);
            }

            double precursor_mz = dbGear.get_mz(this.get_peptide(d[hIndex.get("Stripped.Sequence")],
                            this.get_modification_diann(d[hIndex.get("Modified.Sequence")], d[hIndex.get("Stripped.Sequence")])).getMass(),
                    Integer.parseInt(d[hIndex.get("Precursor.Charge")]));

            double delta_rt = Double.POSITIVE_INFINITY;
            double delta_ccs = Double.POSITIVE_INFINITY;
            int matched_index = -1;

            for (Integer i : index.keySet()) {
                if (is_within_isolation_win(precursor_mz, index.get(i).get("isolation_mz"), index.get(i).get("isolation_width"))){
                    if (Math.abs(index.get(i).get("rt") - rt) < delta_rt && Math.abs(index.get(i).get("ccs") - ccs) < ccs_cutoff) {
                        delta_rt = Math.abs(index.get(i).get("rt") - rt);
                        delta_ccs = Math.abs(index.get(i).get("ccs") - ccs);
                        matched_index = i;
                    } else if (index.get(i).get("rt") > rt + 2) {
                        break;
                    }
                }
            }
            if (matched_index == -1) {
                for (int i : index.keySet()) {
                    if (is_within_isolation_win(precursor_mz, index.get(i).get("isolation_mz"), index.get(i).get("isolation_width"))){
                        if (Math.abs(index.get(i).get("rt") - rt) < delta_rt) {
                            delta_rt = Math.abs(index.get(i).get("rt") - rt);
                            delta_ccs = Math.abs(index.get(i).get("ccs") - ccs);
                            matched_index = i;
                        } else if (index.get(i).get("rt") > rt + 2) {
                            break;
                        }
                    }
                }
            }

            synchronized (this) {
                if (index.containsKey(matched_index)) {
                    ApexMatch apexMatch = new ApexMatch();
                    apexMatch.ms2index = matched_index;
                    double[] iso_win_range = CCSDIAMeta.get_isolation_window(index.get(matched_index).get("isolation_mz"), index.get(matched_index).get("isolation_width"));
                    apexMatch.isolation_window = IsolationWindow.generate_id(iso_win_range[0], iso_win_range[1]);
                    index2index.put(matches.get(k), apexMatch);
                } else {
                    System.out.println("No apex scan found: " + matches.get(k));
                }
            }

            // Using synchronized block to safely write to the writer from multiple threads
            synchronized (writer) {
                try {
                    writer.write(k+"\t"+rt + "\t" + delta_rt + "\t" + delta_ccs + "\t" + precursor_mz +"\t" + matched_index+"\t"+ index.get(matched_index).get("isolation_mz") + "\t" + d[hIndex.get("MS2.Scan")] + "\n");
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        });

        writer.close();
        return index2index;
    }

    /**
     * Add apex MS2 index and scan numbers to the peptide detection file.
     * 
     * @param psm_file A PSM table file from DIA search
     * @param ms_file  A MS spectrum file in mzML format
     * @return A map of PSM to ApexMatch
     * @throws IOException
     */
    public String add_ms2spectrum_index(String psm_file, String ms_file) throws IOException {
        ConcurrentHashMap<Integer, ApexMatch> row2index = new ConcurrentHashMap<>();
        // only need to save information from MS2 spectra
        // index -> rt and isolation_mz
        // Extract MS2 spectra index and MS2 spectra RT and isolation win from mzML
        HashMap<Integer, HashMap<String, Double>> index = new HashMap<>();
        DIAMeta meta = new DIAMeta();
        meta.load_ms_data(ms_file);
        int global_index = 0;
        for (int scan_num : meta.num2scanMap.keySet()) {
            if (meta.num2scanMap.get(scan_num).getMsLevel() == 2) {
                index.put(global_index, new HashMap<>());
                index.get(global_index).put("rt", meta.num2scanMap.get(scan_num).getRt());
                index.get(global_index).put("scan_number", (double) scan_num);
                index.get(global_index).put("isolation_mz_start",meta.num2scanMap.get(scan_num).getPrecursor().getMzRangeStart());
                index.get(global_index).put("isolation_mz_end",meta.num2scanMap.get(scan_num).getPrecursor().getMzRangeEnd());
                global_index++;
            }
        }
        File F = new File(psm_file);
        String out_prefix = F.getName();
        if (out_prefix.endsWith(".csv")) {
            out_prefix = out_prefix.replaceAll(".csv", "");
        } else if (out_prefix.endsWith(".tsv")) {
            out_prefix = out_prefix.replaceAll(".tsv", "");
        } else if (out_prefix.endsWith(".txt")) {
            out_prefix = out_prefix.replaceAll(".txt", "");
        }
        String out_file = out_dir + File.separator + out_prefix + "_added_ms2index.tsv";
        BufferedWriter writer = new BufferedWriter(new FileWriter(out_file));
        Cloger.getInstance().logger.info("Add Apex MS2 information to file: " + out_file);
        BufferedReader psmReader = new BufferedReader(new FileReader(psm_file));
        String head_line = psmReader.readLine();
        head_line = head_line.trim();
        HashMap<String, Integer> cIndex = this.get_column_name2index_from_head_line(head_line);
        String line;
        ArrayList<String> matches = new ArrayList<>();
        while ((line = psmReader.readLine()) != null) {
            line = line.trim();
            String []d= line.split("\t");
            if(d[cIndex.get(PSMConfig.peptide_modification_column_name)].contains("unimod:")){
                d[cIndex.get(PSMConfig.peptide_modification_column_name)] = d[cIndex.get(PSMConfig.peptide_modification_column_name)].replaceAll("unimod:","UniMod:");
                matches.add(StringUtils.join(d,"\t"));
            }else{
                matches.add(line);
            }

        }
        psmReader.close();
        DBGear dbGear = new DBGear();
        IntStream.range(0, matches.size()).parallel().forEach(k -> {
            String[] d = matches.get(k).split("\t");
            double rt = Double.parseDouble(d[cIndex.get(PSMConfig.rt_column_name)]);
            double precursor_mz;
            if (cIndex.containsKey(PSMConfig.precursor_mz_column_name)) {
                precursor_mz = Double.parseDouble(d[cIndex.get(PSMConfig.precursor_mz_column_name)]);
            } else {
                String peptide = d[cIndex.get(PSMConfig.stripped_peptide_sequence_column_name)];
                String modification = this.get_modification_diann(d[cIndex.get(PSMConfig.peptide_modification_column_name)], peptide);
                synchronized (this) {
                    this.add_peptide(peptide, modification);
                }

                precursor_mz = dbGear.get_mz(this.get_peptide(d[cIndex.get(PSMConfig.stripped_peptide_sequence_column_name)],
                                this.get_modification_diann(d[cIndex.get(PSMConfig.peptide_modification_column_name)], d[cIndex.get(PSMConfig.stripped_peptide_sequence_column_name)])).getMass(),
                        Integer.parseInt(d[cIndex.get(PSMConfig.precursor_charge_column_name)]));
            }

            double delta_rt = Double.POSITIVE_INFINITY;
            int matched_index = -1;

            for (Integer i : index.keySet()) {
                if (index.get(i).get("isolation_mz_start") <= precursor_mz
                        && precursor_mz <= index.get(i).get("isolation_mz_end")) {
                    if (Math.abs(index.get(i).get("rt") - rt) < delta_rt) {
                    // if (Math.abs(index.get(i).get("rt") - rt) < delta_rt && Math.abs(index.get(i).get("ccs") - ccs) < ccs_cutoff) {
                        delta_rt = Math.abs(index.get(i).get("rt") - rt);
                        // delta_ccs = Math.abs(index.get(i).get("ccs") - ccs);
                        matched_index = i;
                    } else if (index.get(i).get("rt") > rt + 2) {
                        break;
                    }
                }
            }
            if (matched_index == -1) {
                for (int i : index.keySet()) {
                    if (index.get(i).get("isolation_mz_start") <= precursor_mz
                            && precursor_mz <= index.get(i).get("isolation_mz_end")) {
                        if (Math.abs(index.get(i).get("rt") - rt) < delta_rt) {
                            delta_rt = Math.abs(index.get(i).get("rt") - rt);
                            matched_index = i;
                        } else if (index.get(i).get("rt") > rt + 2) {
                            break;
                        }
                    }
                }
            }

            synchronized (this) {
                if (index.containsKey(matched_index)) {
                    ApexMatch apexMatch = new ApexMatch();
                    apexMatch.ms2index = matched_index;
                    apexMatch.apex_rt = index.get(matched_index).get("rt");
                    apexMatch.delta_rt = index.get(matched_index).get("rt") - rt;
                    apexMatch.scan_number = index.get(matched_index).get("scan_number").intValue();
                    row2index.put(k, apexMatch);
                } else {
                    System.out.println("No apex scan found: " + matches.get(k));
                }
            }
        });
        if(!cIndex.containsKey(PSMConfig.qvalue_column_name)){
            writer.write(head_line+"\tDetection Q Value\tapex_rt\tms2index_delta_rt\tms2index\tscan\tisolation_mz_start\tisolation_mz_end\n");
        }else{
            writer.write(head_line+"\tapex_rt\tms2index_delta_rt\tms2index\tscan\tisolation_mz_start\tisolation_mz_end\n");
        }

        IntStream.range(0, matches.size()).forEach(k -> {
            try {
                if(!cIndex.containsKey(PSMConfig.qvalue_column_name)){
                    writer.write(matches.get(k)+"\t0\t"+row2index.get(k).apex_rt+"\t"+row2index.get(k).delta_rt+"\t"+row2index.get(k).ms2index+"\t"+row2index.get(k).scan_number+"\t"+index.get(row2index.get(k).ms2index).get("isolation_mz_start")+"\t"+index.get(row2index.get(k).ms2index).get("isolation_mz_end")+"\n");
                }else{
                    writer.write(matches.get(k)+"\t"+row2index.get(k).apex_rt+"\t"+row2index.get(k).delta_rt+"\t"+row2index.get(k).ms2index+"\t"+row2index.get(k).scan_number+"\t"+index.get(row2index.get(k).ms2index).get("isolation_mz_start")+"\t"+index.get(row2index.get(k).ms2index).get("isolation_mz_end")+"\n");
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
        writer.close();
        PSMConfig.rt_column_name = "apex_rt";
        return (out_file);
    }

    /**
     * Check if the m/z value is within the isolation window.
     * 
     * @param mz     m/z value to check
     * @param iso_mz Isolation m/z value in the center
     * @param width  Isolation width
     * @return true if within isolation window, false otherwise
     */
    private boolean is_within_isolation_win(double mz, double iso_mz, double width) {
        return mz >= (iso_mz - width / 2.0) && mz <= (iso_mz + width / 2.0);
    }

    /**
     * Get MS2 spectrum match information for each peptide match and export matched
     * spectra to an MGF file.
     * 
     * @param matches List of matches
     * @param ms_file Path to the MS file
     * @param dbGear  A DBGear instance for database access
     * @param out_dir Output directory for results
     * @return A map of matches to their corresponding MS2 spectra metadata
     * @throws IOException If an I/O error occurs
     */
    public ConcurrentHashMap<String, Integer> get_ms2spectrum_index_and_export_mgf(ArrayList<String> matches, String ms_file, DBGear dbGear, String out_dir) throws IOException {
        ConcurrentHashMap<String, Integer> index2index = new ConcurrentHashMap<>();
        HashMap<Integer, HashMap<String, Double>> index = new HashMap<>();
        CarpetReader<MS2SpectraAll> reader = new CarpetReader<>(new File(ms_file), MS2SpectraAll.class);

        HashMap<Integer, Spectrum> index2spectra = new HashMap<>();
        for (MS2SpectraAll spectrum : reader) {
            index.put(spectrum.index, new HashMap<>());
            index.get(spectrum.index).put("ccs", spectrum.precursor_im);
            index.get(spectrum.index).put("rt", spectrum.precursor_rt / 60);
            index.get(spectrum.index).put("isolation_mz", spectrum.isolation_mz);
            index.get(spectrum.index).put("isolation_width", spectrum.isolation_width);

            Precursor precursor = new Precursor(spectrum.precursor_rt, spectrum.isolation_mz, new int[] { 2 });
            Spectrum spec = new Spectrum(precursor,
                    spectrum.mz_values.stream().mapToDouble(Double::doubleValue).toArray(),
                    spectrum.intensities.stream().mapToDouble(Double::doubleValue).toArray());
            index2spectra.put(spectrum.index, spec);
        }
        double ccs_cutoff = 0.05;
        File F = new File(ms_file);
        String out_prefix = F.getName();
        if (out_prefix.endsWith(".mzML")) {
            out_prefix = out_prefix.replaceAll(".mzML", "");
        } else if (out_prefix.endsWith(".mzml")) {
            out_prefix = out_prefix.replaceAll(".mzml", "");
        } else if (out_prefix.endsWith(".parquet")) {
            out_prefix = out_prefix.replaceAll(".parquet", "");
        }
        String out_file = out_dir + File.separator + out_prefix + "_apex_ms2spectra.tsv";
        BufferedWriter writer = new BufferedWriter(new FileWriter(out_file));
        Cloger.getInstance().logger.info("Export Apex MS2 mapping information to file: " + out_file);
        writer.write("row_index\tRT\tdelta_rt\tdelta_ccs\tms2index\tMS2.Scan\n");

        String mgf_out_file = out_dir + File.separator + out_prefix + ".mgf";
        BufferedWriter mgf_writer = new BufferedWriter(new FileWriter(mgf_out_file));

        BufferedWriter psmWriter = new BufferedWriter(new FileWriter(out_dir + "/psm_pdv.txt"));
        // psmWriter.write(this.psm_head_line+"\tspectrum_title\tmz\tcharge\tpeptide\tmodification\tmods\tmod_sites\tmax_fragment_ion_valid\tmax_cor_mz\tfrag_start_idx\tfrag_stop_idx\n");
        psmWriter.write("psm_id\tspectrum_title\tmz\tcharge\tpeptide\tmodification\n");

        IntStream.range(0, matches.size()).parallel().forEach(k -> {
            String[] d = matches.get(k).split("\t");
            String peptide = d[hIndex.get("Stripped.Sequence")];
            String modification = this.get_modification_diann(d[hIndex.get("Modified.Sequence")], peptide);
            int precursor_charge = Integer.parseInt(d[hIndex.get("Precursor.Charge")]);
            double rt = Double.parseDouble(d[hIndex.get("RT")]);
            double ccs = Double.parseDouble(d[hIndex.get("IM")]);
            synchronized (this) {
                this.add_peptide(peptide, modification);
            }

            double precursor_mz = dbGear.get_mz(this.get_peptide(d[hIndex.get("Stripped.Sequence")],
                            this.get_modification_diann(d[hIndex.get("Modified.Sequence")], d[hIndex.get("Stripped.Sequence")])).getMass(),
                    Integer.parseInt(d[hIndex.get("Precursor.Charge")]));

            double delta_rt = Double.POSITIVE_INFINITY;
            double delta_ccs = Double.POSITIVE_INFINITY;
            int matched_index = -1;

            for (int i : index.keySet()) {
                if (is_within_isolation_win(precursor_mz, index.get(i).get("isolation_mz"), index.get(i).get("isolation_width"))){
                    if (Math.abs(index.get(i).get("rt") - rt) < delta_rt && Math.abs(index.get(i).get("ccs") - ccs) < ccs_cutoff) {
                        delta_rt = Math.abs(index.get(i).get("rt") - rt);
                        delta_ccs = Math.abs(index.get(i).get("ccs") - ccs);
                        matched_index = i;
                    } else if (index.get(i).get("rt") > rt + 2) {
                        break;
                    }
                }
            }
            if (matched_index == -1) {
                for (int i : index.keySet()) {
                    if (is_within_isolation_win(precursor_mz, index.get(i).get("isolation_mz"), index.get(i).get("isolation_width"))){
                        if (Math.abs(index.get(i).get("rt") - rt) < delta_rt) {
                            delta_rt = Math.abs(index.get(i).get("rt") - rt);
                            delta_ccs = Math.abs(index.get(i).get("ccs") - ccs);
                            matched_index = i;
                        } else if (index.get(i).get("rt") > rt + 2) {
                            break;
                        }
                    }
                }
            }

            if (matched_index != -1) {

                synchronized (this) {
                    index2index.put(matches.get(k), matched_index);
                }

                // Using synchronized block to safely write to the writer from multiple threads
                synchronized (writer) {
                    try {
                        writer.write(k + "\t" + rt + "\t" + delta_rt + "\t" + delta_ccs + "\t" + matched_index + "\t" + d[hIndex.get("MS2.Scan")] + "\n");
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }

                synchronized (mgf_writer) {
                    try {
                        mgf_writer.write(MgfUtils.asMgf(index2spectra.get(matched_index), String.valueOf(matched_index), 2, String.valueOf(matched_index)) + "\n");
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }

                String pdv_mod = this.get_modification_diann(d[hIndex.get("Modified.Sequence")], d[hIndex.get("Stripped.Sequence")]);
                synchronized (psmWriter) {
                    try {
                        psmWriter.write(d[hIndex.get("MS2.Scan")] + "\t" + matched_index + "\t" + precursor_mz + "\t" + precursor_charge + "\t" + peptide + "\t" + pdv_mod + "\n");
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }

        });

        writer.close();
        mgf_writer.close();
        psmWriter.close();
        return index2index;
    }

    /**
     * Get MS2 spectrum match information for each peptide match.
     * 
     * @param psm_file A PSM table file from DIA search
     * @param ms_file  A MS spectrum file in mzML format
     * @param dbGear   A DBGear instance for database access
     * @param out_dir  Output directory for results
     */
    public void get_ms2spectrum_index(String psm_file, String ms_file, DBGear dbGear, String out_dir) {
        ArrayList<String> matches = new ArrayList<>();
        try {
            BufferedReader pReader = new BufferedReader(new FileReader(psm_file));
            String header = pReader.readLine().trim();
            String h[] = header.split("\t");
            for (int i = 0; i < h.length; i++) {
                hIndex.put(h[i], i);
            }
            String line;

            while ((line = pReader.readLine()) != null) {
                matches.add(line.trim());
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        try {
            get_ms2spectrum_index(matches, ms_file, dbGear, out_dir);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Get MS2 spectrum match information for each peptide match and export matched
     * spectra to an MGF file.
     * 
     * @param psm_file A PSM table file from DIA search
     * @param ms_file  A MS spectrum file in mzML format
     * @param dbGear   A DBGear instance for database access
     * @param out_dir  Output directory for results
     */
    public void get_ms2spectrum_index_and_export_mgf(String psm_file, String ms_file, DBGear dbGear, String out_dir) {
        ArrayList<String> matches = new ArrayList<>();
        try {
            BufferedReader pReader = new BufferedReader(new FileReader(psm_file));
            String header = pReader.readLine().trim();
            String h[] = header.split("\t");
            for (int i = 0; i < h.length; i++) {
                hIndex.put(h[i], i);
            }
            String line;

            while ((line = pReader.readLine()) != null) {
                matches.add(line.trim());
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        try {
            get_ms2spectrum_index_and_export_mgf(matches, ms_file, dbGear, out_dir);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Save XIC data to a json format string
     * 
     * @param id     XIC ID
     * @param pMatch A PeptideMatch object which stores peptide XIC data
     * @return A JSON string representation of the XIC data
     */
    private String get_xic_json(String id, PeptideMatch pMatch) {
        JXIC xic = new JXIC();
        if (pMatch.peak.fragment_ions_mz != null) {
            xic.fragment_ion_mzs = pMatch.peak.fragment_ions_mz
                    .stream()
                    .mapToDouble(Double::doubleValue)
                    .toArray();
            xic.smoothed_fragment_intensities = pMatch.smoothed_fragment_intensities.getData();
            xic.raw_fragment_intensities = pMatch.raw_fragment_intensities;
            xic.xic_rt_values = pMatch.xic_rt_values;
            xic.fragment_ion_skewness = pMatch.skewed_peaks;

            xic.fragment_ion_cors = new double[xic.fragment_ion_mzs.length];
            for (int i = 0; i < xic.fragment_ion_mzs.length; i++) {
                if (pMatch.mz2cor.containsKey(xic.fragment_ion_mzs[i])) {
                    xic.fragment_ion_cors[i] = pMatch.mz2cor.get(xic.fragment_ion_mzs[i]);
                } else {
                    System.err.println("Error: missing fragment ion correlation:" + xic.fragment_ion_mzs[i]);
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
        }
        return (JSON.toJSONString(xic));
    }

    /**
     * Get adjacent MS2 matches for a given peptide match.
     * 
     * @param peptideMatch  A PeptideMatch object
     * @param n_flank_scans Number of flank scans to consider
     * @param diaIndex      A DIAIndex object containing indexed DIA data
     * @param iso_win       Isolation window ID
     * @return A list of adjacent PeptideMatch objects
     */
    ArrayList<PeptideMatch> get_adjacent_ms2_matches(PeptideMatch peptideMatch, int n_flank_scans, DIAIndex diaIndex, String iso_win){
        int scan_num = peptideMatch.scan;
        int scan_index = diaIndex.get_index_by_scan(iso_win, scan_num);
        // 10 -> 10-2=8, 10+2 = 12 -> 8, 9, 11, 12
        int start_scan_index = scan_index - n_flank_scans;
        int end_scan_index = scan_index + n_flank_scans;
        ArrayList<PeptideMatch> pMatches = new ArrayList<>(2 * n_flank_scans);
        for (int index = start_scan_index; index <= end_scan_index; index++) {
            if (index == scan_index) {
                continue;
            }
            if (diaIndex.isolation_win2index2scan.get(iso_win).containsKey(index)) {
                int scan = diaIndex.get_scan_by_index(iso_win, index);
                Spectrum spectrum = diaIndex.get_spectrum_by_scan(scan);
                if (spectrum == null) {
                    System.out.println("Spectrum is null:" + index + "\t" + scan);
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
                        if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION
                                || ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION) {
                            // add fragment ion number
                            PeptideFragmentIon fragmentIon = ((PeptideFragmentIon) ionMatch.ion);
                            int ion_number = fragmentIon.getNumber();

                            int ion_type_column_index = this.get_ion_type_column_index(ionMatch);
                            // for y ion
                            if (ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION) {
                                fragment_ion_row_index = peptideMatch.peptide.getSequence().length() - ion_number - 1;
                            } else if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION) {
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

                            if (max_fragment_ion_intensity <= ionMatch.peakIntensity) {
                                max_fragment_ion_intensity = ionMatch.peakIntensity;
                                max_fragment_ion_row_index = fragment_ion_row_index;
                                max_fragment_ion_column_index = ion_type_column_index;
                            }

                        }
                    }
                }
                if (!matched_ion_mzs.isEmpty()) {
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
                if (cor >= 0.9) {
                    pMatches.add(pMatch);
                }

            }

        }
        return pMatches;
    }

    /**
     * Get adjacent MS2 matches for a given peptide match for TIMS-TOF data
     * 
     * @param peptideMatch  A PeptideMatch object
     * @param n_flank_scans Number of flank scans to consider
     * @param diaIndex      A CCSDIAIndex object containing indexed DIA data
     * @param iso_win       Isolation window ID
     * @return A list of adjacent PeptideMatch objects
     */
    ArrayList<PeptideMatch> get_adjacent_ms2_matches_ccs(PeptideMatch peptideMatch, int n_flank_scans, CCSDIAIndex diaIndex, String iso_win){
        int scan_num = peptideMatch.scan;
        int scan_index = diaIndex.get_index_by_scan(iso_win, scan_num);
        // 10 -> 10-2=8, 10+2 = 12 -> 8, 9, 11, 12
        int start_scan_index = scan_index - n_flank_scans;
        int end_scan_index = scan_index + n_flank_scans;
        ArrayList<PeptideMatch> pMatches = new ArrayList<>(2 * n_flank_scans);
        for (int index = start_scan_index; index <= end_scan_index; index++) {
            if (index == scan_index) {
                continue;
            }
            if (diaIndex.isolation_win2index2scan.get(iso_win).containsKey(index)) {
                int scan = diaIndex.get_scan_by_index(iso_win, index);
                Spectrum spectrum = diaIndex.get_spectrum_by_scan(scan);
                if (spectrum == null) {
                    System.out.println("Spectrum is null:" + index + "\t" + scan);
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
                        if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION
                                || ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION) {
                            // add fragment ion number
                            PeptideFragmentIon fragmentIon = ((PeptideFragmentIon) ionMatch.ion);
                            int ion_number = fragmentIon.getNumber();

                            int ion_type_column_index = this.get_ion_type_column_index(ionMatch);
                            // for y ion
                            if (ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION) {
                                fragment_ion_row_index = peptideMatch.peptide.getSequence().length() - ion_number - 1;
                            } else if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION) {
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

                            if (max_fragment_ion_intensity <= ionMatch.peakIntensity) {
                                max_fragment_ion_intensity = ionMatch.peakIntensity;
                                max_fragment_ion_row_index = fragment_ion_row_index;
                                max_fragment_ion_column_index = ion_type_column_index;
                            }

                        }
                    }
                }
                if (!matched_ion_mzs.isEmpty()) {
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
                if (cor >= 0.9) {
                    pMatches.add(pMatch);
                }

            }

        }
        return pMatches;
    }

    /**
     * Spectra correlation calculation
     * 
     * @param x A PeptideMatch object
     * @param y A PeptideMatch object
     * @return The correlation value between the two spectra
     */
    private double calc_spectrum_correlation(PeptideMatch x, PeptideMatch y) {
        int n_valid_peaks = 0;
        int n_total_peaks = 0;
        // x.ion_intensity_matrix is a 2D matrix
        // n_col is its column number
        // n_row is its row number
        int n_row = x.ion_intensity_matrix.length;
        int n_col = x.ion_intensity_matrix[0].length;
        ArrayList<Double> x_int = new ArrayList<>();
        ArrayList<Double> y_int = new ArrayList<>();
        for (int i = 0; i < n_row; i++) {
            for (int j = 0; j < n_col; j++) {
                if (x.ion_intensity_matrix[i][j] > 0) {
                    n_total_peaks++;
                }
                if (x.ion_intensity_matrix[i][j] > 0 && x.ion_matrix[i][j] <= 0) {
                    x_int.add(x.ion_intensity_matrix[i][j]);
                    y_int.add(y.ion_intensity_matrix[i][j]);
                    n_valid_peaks++;
                }
            }
        }
        double cor = -100.0;
        if (x_int.size() >= 3) {
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

    /**
     * Log10 transformation
     * 
     * @param x The input value to be transformed
     * @return Log10 transformed value
     */
    private double log_transform(double x) {
        return FastMath.log10(x + 1) / 3;
    }

    /**
     * Extraction modification from DIA-NN format inputs
     * 
     * @param mod_seq The modified sequence string from DIA-NN
     * @param peptide Peptide sequence
     * @return A string containing the modification information
     */
    public String get_modification_diann(String mod_seq, String peptide) {
        // AAAAC(UniMod:4)LDK2
        // AGEVLNQPM(UniMod:35)MMAAR2
        // AAAAAAAATMALAAPS(UniMod:21)SPTPESPTMLTK
        // AAAGPLDMSLPST(UniMod:21)PDLK
        // (UniMod:1)AAVTLHLR
        if (mod_seq.equalsIgnoreCase(peptide)) {
            return "-";
        } else {
            for (String unimod : CModification.getInstance().unimod2modification_code.keySet()) {
                if (mod_seq.contains(unimod)) {
                    mod_seq = mod_seq.replace(unimod, CModification.getInstance().unimod2modification_code.get(unimod));
                }
            }
            ArrayList<String> mods = new ArrayList<>();
            int n_term_mods = 0;
            if (mod_seq.length() != peptide.length()) {
                // TODO: improve this part
                if (mod_seq.startsWith("(UniMod:1)") && CParameter.varMods.contains("5")) {
                    // Protein N-term Acetylation
                    String unimod = "Protein N-term(UniMod:1)";
                    mod_seq = mod_seq.replace("(UniMod:1)",CModification.getInstance().unimod2modification_code.get(unimod));
                    n_term_mods = n_term_mods + 1;
                } else {
                    System.err.println("Unrecognized modification:" + mod_seq + "," + peptide);
                    System.exit(1);
                }
            }
            String[] aas = mod_seq.split("");
            int pos;
            String mod_name;
            for (int i = 0; i < aas.length; i++) {
                if (CModification.getInstance().modification_code2modification.containsKey(aas[i])) {
                    // Oxidation of M@12[15.9949]
                    pos = i + 1 - n_term_mods;
                    mod_name = CModification.getInstance().modification_code2modification.get(aas[i]);
                    mods.add(mod_name+"@"+pos+"["+ CModification.getInstance().getPTMbyName(mod_name).getMass()+"]");
                }
            }
            return StringUtils.join(mods, ";");
        }

    }

    /**
     * Generate RT training data.
     * 
     * @param peptide2rt A HashMap containing peptide retention time information
     * @param method     The method for aggregating retention times (max, min, mean)
     *                   when there are multiple retention time values
     *                   available for a given peptide precursor.
     * @param out_file   Output file
     * @throws IOException If an I/O error occurs
     */
    void generate_rt_train_data(HashMap<String, PeptideRT> peptide2rt, String method, String out_file)
            throws IOException {

        peptide2rt.values().parallelStream().forEach(peptideRT -> {
            if (peptideRT.rts.size() == 1) {
                peptideRT.rt = peptideRT.rts.get(0);
                peptideRT.rt_max = peptideRT.rts.get(0);
                peptideRT.rt_min = peptideRT.rts.get(0);
            } else {
                peptideRT.rt_max = Collections.max(peptideRT.rts);
                peptideRT.rt_min = Collections.min(peptideRT.rts);
                if (method.equalsIgnoreCase("max")) {
                    int maxIndex = peptideRT.scores.indexOf(Collections.max(peptideRT.scores));
                    peptideRT.rt = peptideRT.rts.get(maxIndex);
                } else if (method.equalsIgnoreCase("min")) {
                    int minIndex = peptideRT.scores.indexOf(Collections.min(peptideRT.scores));
                    peptideRT.rt = peptideRT.rts.get(minIndex);
                } else if (method.equalsIgnoreCase("mean")) {
                    double sum = 0;
                    for (double num : peptideRT.rts) {
                        sum += num;
                    }
                    peptideRT.rt = sum / peptideRT.rts.size();
                } else {
                    System.err.println("Unrecognized method:" + method);
                    System.exit(1);
                }
            }
            // peptideRT.rt_norm = (peptideRT.rt - this.rt_min)/(this.rt_max - this.rt_min);
            peptideRT.rt_norm = peptideRT.rt / this.rt_max;
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
        System.out.println("RT train data:" + out_file);
    }

    /**
     * Generate CCS training data.
     * 
     * @param peptide2ccs A HashMap containing peptide CCS information
     * @param method      The method for aggregating CCS values (max, min, mean)
     *                    when there are multiple CCS values
     *                    available for a given peptide precursor.
     * @param out_file    Output file
     * @throws IOException If an I/O error occurs
     */
    private void generate_ccs_train_data(HashMap<String, PeptideCCS> peptide2ccs, String method, String out_file)
            throws IOException {

        peptide2ccs.values().parallelStream().forEach(peptideCCS -> {
            if (peptideCCS.ccs_values.size() == 1) {
                peptideCCS.ccs = peptideCCS.ccs_values.get(0);
            } else {
                if (method.equalsIgnoreCase("max")) {
                    int maxIndex = peptideCCS.scores.indexOf(Collections.max(peptideCCS.scores));
                    peptideCCS.ccs = peptideCCS.ccs_values.get(maxIndex);
                } else if (method.equalsIgnoreCase("min")) {
                    int minIndex = peptideCCS.scores.indexOf(Collections.min(peptideCCS.scores));
                    peptideCCS.ccs = peptideCCS.ccs_values.get(minIndex);
                } else if (method.equalsIgnoreCase("mean")) {
                    double sum = 0;
                    for (double num : peptideCCS.ccs_values) {
                        sum += num;
                    }
                    peptideCCS.ccs = sum / peptideCCS.ccs_values.size();
                } else {
                    System.err.println("Unrecognized method:" + method);
                    System.exit(1);
                }
            }
            peptideCCS.mods = convert_modification(peptideCCS.modification);
        });

        // output
        BufferedWriter bWriter = new BufferedWriter(new FileWriter(out_file));
        bWriter.write("peptide\tsequence\tnAA\tcharge\tmodification\tmods\tmod_sites\tx\ty\tmobility\n");
        for (String pep_mod : peptide2ccs.keySet()) {
            bWriter.write(peptide2ccs.get(pep_mod).peptide + "\t" +
                    peptide2ccs.get(pep_mod).peptide + "\t" +
                    peptide2ccs.get(pep_mod).peptide.length() + "\t" +
                    peptide2ccs.get(pep_mod).charge + "\t" +
                    peptide2ccs.get(pep_mod).modification + "\t" +
                    peptide2ccs.get(pep_mod).mods[0] + "\t" +
                    peptide2ccs.get(pep_mod).mods[1] + "\t" +
                    convert_modification(peptide2ccs.get(pep_mod).peptide, peptide2ccs.get(pep_mod).modification) + "\t" +
                    peptide2ccs.get(pep_mod).ccs + "\t" +
                    peptide2ccs.get(pep_mod).ccs + "\n");
        }
        bWriter.close();
        System.out.println("CCS train data:" + out_file);
    }

    /**
     * Get the number of valid fragment ions in the ion intensity matrix: intensity
     * > 0 and valid
     * 
     * @param ion_intensity_matrix Ion intensity matrix
     * @param ion_valid_matrix     Ion intensity valid matrix. The same shape with
     *                             ion_intensity_matrix
     * @return The number of valid fragment ions
     */
    public int get_n_valid_fragment_ions(double[][] ion_intensity_matrix, int[][] ion_valid_matrix) {
        int n = 0;
        for (int i = 0; i < ion_intensity_matrix.length; i++) {
            for (int j = 0; j < ion_intensity_matrix[i].length; j++) {
                if (ion_intensity_matrix[i][j] > 0 && ion_valid_matrix[i][j] == 0) {
                    n++;
                }
            }
        }
        return n;
    }

    /**
     * Get the number of matched fragment ions: intensity > 0
     * 
     * @param ion_intensity_matrix Ion intensity matrix
     * @return The number of matched fragment ions
     */
    public int get_n_matched_fragment_ions(double[][] ion_intensity_matrix) {
        int n = 0;
        for (int i = 0; i < ion_intensity_matrix.length; i++) {
            for (int j = 0; j < ion_intensity_matrix[i].length; j++) {
                if (ion_intensity_matrix[i][j] > 0) {
                    n++;
                }
            }
        }
        return n;
    }

    /**
     * Extract modification name and position from a modification string.
     * 
     * @param modification A modification string. Multiple modifications are
     *                     separated by ";".
     * @return An array of two strings: the first string contains modification
     *         names, and the second string contains modification positions.
     */
    public String[] convert_modification(String modification) {
        if (modification.equalsIgnoreCase("-")) {
            return new String[] { "", "" };
        } else {
            // "MLSECYR"
            // Carbamidomethyl@C;Oxidation@M 5;1
            // Oxidation of M@17[15.9949];Carbamidomethylation of C@6[57.0215]
            String[] m = modification.split(";");
            ArrayList<String> mod_name_list = new ArrayList<>(m.length);
            ArrayList<String> mod_pos_list = new ArrayList<>(m.length);
            for (String ptm : m) {
                String mod_name = ptm.split("@")[0];
                String pos = ptm.split("@")[1].split("\\[")[0];
                if (this.mod_map.containsKey(mod_name)) {
                    mod_name_list.add(this.mod_map.get(mod_name));
                    mod_pos_list.add(pos);
                } else {
                    System.err.println("Unrecognized modification:" + mod_name);
                    System.exit(1);
                }

            }
            return new String[] { StringUtils.join(mod_name_list, ";"), StringUtils.join(mod_pos_list, ";") };
        }
    }

    /**
     * Extract modification name and position from a modification string.
     * 
     * @param peptide A Peptide object.
     * @return An array of two strings: the first string contains modification
     *         names, and the second string contains modification positions.
     */
    private String[] convert_modification(Peptide peptide) {
        String modification = ModificationUtils.getInstance().getModificationString(peptide);
        return convert_modification(modification);
    }

    /**
     * Convert a peptide sequence with modifications to a string representation:
     * each modified amino acid is represented
     * using a specific integer number.
     * 
     * @param peptide      Peptide sequence
     * @param modification Peptide modification string, such as Oxidation of
     *                     M@17[15.9949];Carbamidomethylation of C@6[57.0215].
     * @return A peptide sequence with modified amino acids converted to integers.
     */
    private String convert_modification(String peptide, String modification) {
        String x = peptide;
        boolean unrecognized_mod_found = false;
        if (!modification.equals("-")) {
            String[] m = modification.split(";");
            String n_term_char = "";
            String[] aa = peptide.split("");
            for (String ptm : m) {
                String mod_name = ptm.split("@")[0];
                String pos = ptm.split("@")[1].split("\\[")[0];
                if (mod_name.equalsIgnoreCase("Oxidation of M")) {
                    aa[Integer.parseInt(pos) - 1] = String.valueOf(1);
                } else if (mod_name.equalsIgnoreCase("Phosphorylation of S")
                        || mod_name.equalsIgnoreCase("Phospho of S")) {
                    aa[Integer.parseInt(pos) - 1] = String.valueOf(2);
                } else if (mod_name.equalsIgnoreCase("Phosphorylation of T")
                        || mod_name.equalsIgnoreCase("Phospho of T")) {
                    aa[Integer.parseInt(pos) - 1] = String.valueOf(3);
                } else if (mod_name.equalsIgnoreCase("Phosphorylation of Y")
                        || mod_name.equalsIgnoreCase("Phospho of Y")) {
                    aa[Integer.parseInt(pos) - 1] = String.valueOf(4);
                } else if (mod_name.equalsIgnoreCase("Carbamidomethylation of C")
                        || mod_name.equalsIgnoreCase("Carbamidomethyl of C")) {
                    // no need to change
                    // fixed modification.
                } else if (mod_name.equalsIgnoreCase("Acetyl of protein N-term")) {
                    // no need to change
                    n_term_char = "5";
                }else{
                    if(CModification.getInstance().ptm_name2id.containsKey(mod_name)) {
                        aa[Integer.parseInt(pos) - 1] = String.valueOf(CModification.getInstance().ptm_name2id.get(mod_name));
                    }else {
                        System.out.println("Unrecognized modification found:" + peptide + " -> " + modification);
                        unrecognized_mod_found = true;
                    }
                }
            }
            if (unrecognized_mod_found) {
                x = "-"; //
            } else {
                x = StringUtils.join(aa, "");
            }
            if (!n_term_char.equalsIgnoreCase("")) {
                x = n_term_char + x + CParameter.terminal_char;
            } else {
                x = CParameter.terminal_char + x + CParameter.terminal_char;
            }
        } else {
            x = CParameter.terminal_char + peptide + CParameter.terminal_char;
        }
        return x;
    }

    /**
     * Load modification map from the ModificationUtils class.
     */
    public void load_mod_map() {
        this.mod_map.put("Carbamidomethylation of C", "Carbamidomethyl@C");
        this.mod_map.put("Oxidation of M", "Oxidation@M");
        this.mod_map.put("Phosphorylation of S", "Phospho@S");
        this.mod_map.put("Phosphorylation of T", "Phospho@T");
        this.mod_map.put("Phosphorylation of Y", "Phospho@Y");
        for (String mod_name : ModificationUtils.getInstance().mod_name2JMod.keySet()) {
            String psi_name = ModificationUtils.getInstance().mod_name2JMod.get(mod_name).psi_ms_name;
            if (mod_name.contains("protein N-term")) {
                psi_name = psi_name + "@Protein_N-term";
            } else {
                psi_name = psi_name + "@" + ModificationUtils.getInstance().mod_name2JMod.get(mod_name).site;
            }
            this.mod_map.put(mod_name, psi_name);
        }
    }

    /**
     * Extract the XIC (Extracted Ion Chromatogram) for a given peptide match.
     * 
     * @param ms2index     DIA index containing MS2 data
     * @param peptideMatch A PeptideMatch object containing information about the
     *                     peptide
     * @param isoWinID     Isolation window ID for the XIC extraction
     */
    private void xic_query(DIAIndex ms2index, PeptideMatch peptideMatch, String isoWinID) {
        LibSpectrum libSpectrum = peptideMatch.libSpectrum;
        boolean is_ppm = CParameter.itolu.equalsIgnoreCase("ppm");
        ArrayList<LPeak> peaks = new ArrayList<>(libSpectrum.spectrum.mz.length);
        IntStream.range(0, libSpectrum.spectrum.mz.length).forEach(i -> {
            LPeak p = new LPeak(libSpectrum.spectrum.mz[i], 0.0);
            peaks.add(p);
        });
        peaks.sort(new LPeakComparatorMax2Min());

        double rt_start;
        double rt_end;
        if (this.refine_peak_boundary) {
            rt_start = Math.max(0, peptideMatch.rt_apex - CParameter.rt_win);
            rt_end = peptideMatch.rt_apex + CParameter.rt_win;
        } else {
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
        for (double mz : all_mzs) {
            if (res.get(mz).isEmpty()) {
                res.remove(mz);
            }
        }
        // mz of each fragment ion
        List<Double> fragment_ions = res.keySet().stream().sorted().collect(toList());
        if (res.size() >= 4) {
            List<Integer> unique_scans = res.values().stream()
                    .flatMap(Collection::stream)
                    .map(ion -> ion.scan)
                    .distinct()
                    .sorted()
                    .collect(toList());

            if (unique_scans.size() >= ms2index.min_scan_for_peak) {

                int scan_min = Collections.min(unique_scans);
                int scan_max = Collections.max(unique_scans);
                int index_min = ms2index.isolation_win2scan2index.get(isoWinID).get(scan_min);
                int index_max = ms2index.isolation_win2scan2index.get(isoWinID).get(scan_max);

                // extend to the extraction window
                // left side
                if (ms2index.get_rt_by_scan(isoWinID, scan_min) > rt_start
                        && Math.abs(ms2index.get_rt_by_scan(isoWinID, scan_min) - rt_start) > 0.01) {
                    int scan_i = scan_min;
                    int index_i = index_min;
                    while (ms2index.isolation_win2scan2rt.get(isoWinID).containsKey(scan_i)
                            && ms2index.get_rt_by_scan(isoWinID, scan_i) > rt_start) {
                        index_i = index_i - 1;
                        if (ms2index.isolation_win2index2scan.get(isoWinID).containsKey(index_i)) {
                            scan_i = ms2index.get_scan_by_index(isoWinID, index_i);
                        } else {
                            index_i = index_i + 1;
                            break;
                        }
                    }
                    index_min = index_i;
                }
                // right side
                if (ms2index.get_rt_by_scan(isoWinID, scan_max) < rt_end
                        && Math.abs(ms2index.get_rt_by_scan(isoWinID, scan_max) - rt_end) > 0.01) {
                    int scan_i = scan_max;
                    int index_i = index_max;
                    while (ms2index.isolation_win2scan2rt.get(isoWinID).containsKey(scan_i)
                            && ms2index.get_rt_by_scan(isoWinID, scan_i) < rt_end) {
                        index_i = index_i + 1;
                        if (ms2index.isolation_win2index2scan.get(isoWinID).containsKey(index_i)) {
                            scan_i = ms2index.get_scan_by_index(isoWinID, index_i);
                        } else {
                            index_i = index_i - 1;
                            break;
                        }
                    }
                    index_max = index_i;
                }

                int scan_num = index_max - index_min + 1;

                HashMap<Integer, Integer> scan2index = new HashMap<>(unique_scans.size());
                HashMap<Integer, Integer> index2scan = new HashMap<>(unique_scans.size());
                // HashMap<Integer, Float> scan2rt = new HashMap<>(unique_scans.size());
                double[] index2rt = new double[scan_num];

                PeptidePeak peak = new PeptidePeak();
                peak.fragment_ions_mz = fragment_ions;
                double rt;
                double max_rt = 0;
                boolean apex_found = false;
                boolean boundary_left_found = false;
                boolean boundary_right_found = false;
                for (int i = 0; i < scan_num; i++) {
                    int cur_index = index_min + i;
                    int cur_scan = ms2index.get_scan_by_index(isoWinID, cur_index);
                    scan2index.put(cur_scan, i);
                    index2scan.put(i, cur_scan);

                    rt = ms2index.get_rt_by_scan(isoWinID, cur_scan);
                    index2rt[i] = rt;

                    if (Math.abs(rt - peptideMatch.rt_apex) <= 0.01) {
                        peak.apex_index = i;
                        apex_found = true;
                    }
                    if (Math.abs(rt - peptideMatch.rt_start) <= 0.01) {
                        peak.boundary_left_index = i;
                        boundary_left_found = true;
                    }
                    if (Math.abs(rt - peptideMatch.rt_end) <= 0.01) {
                        peak.boundary_right_index = i;
                        boundary_right_found = true;
                    }
                    if (max_rt < rt) {
                        max_rt = rt;
                    }

                }
                if(!apex_found){
                    Cloger.getInstance().logger.error("Apex not found:"+peptideMatch.rt_apex+","+peptideMatch.rt_start+","+peptideMatch.rt_end);
                    System.exit(1);
                }
                if (!boundary_left_found || !boundary_right_found || !apex_found) {
                    // redo the peak index detection
                    rt = 0;
                    max_rt = 0;
                    apex_found = false;
                    boundary_left_found = false;
                    boundary_right_found = false;
                    double delta_rt_apex = Double.POSITIVE_INFINITY;
                    double delta_rt_start = Double.POSITIVE_INFINITY;
                    double delta_rt_end = Double.POSITIVE_INFINITY;
                    for (int i = 0; i < scan_num; i++) {
                        int cur_index = index_min + i;
                        int cur_scan = ms2index.get_scan_by_index(isoWinID, cur_index);
                        scan2index.put(cur_scan, i);
                        index2scan.put(i, cur_scan);

                        rt = ms2index.get_rt_by_scan(isoWinID, cur_scan);
                        index2rt[i] = rt;

                        if (Math.abs(rt - peptideMatch.rt_apex) <= delta_rt_apex) {
                            peak.apex_index = i;
                            delta_rt_apex = Math.abs(rt - peptideMatch.rt_apex);
                        }
                        if (Math.abs(rt - peptideMatch.rt_start) <= delta_rt_start) {
                            peak.boundary_left_index = i;
                            delta_rt_start = Math.abs(rt - peptideMatch.rt_start);
                        }
                        if (Math.abs(rt - peptideMatch.rt_end) <= delta_rt_end) {
                            peak.boundary_right_index = i;
                            delta_rt_end = Math.abs(rt - peptideMatch.rt_end);
                        }
                        if (max_rt < rt) {
                            max_rt = rt;
                        }

                    }
                }
                int mz_length = libSpectrum.spectrum.mz.length;
                HashMap<Double, Double> mz2int = new HashMap<>(mz_length);
                for (int i = 0; i < mz_length; i++) {
                    mz2int.put(libSpectrum.spectrum.mz[i], libSpectrum.spectrum.intensity[i]);
                }

                // fragment ion intensity
                double[][] frag_int = new double[res.size()][index_max - index_min + 1];

                for (int j = 0; j < fragment_ions.size(); j++) {
                    for (JFragmentIon ion : res.get(fragment_ions.get(j))) {
                        // System.out.println(j+"\t"+ scan2index.get(ion.scan));
                        if (frag_int[j][scan2index.get(ion.scan)] < ion.intensity) {
                            frag_int[j][scan2index.get(ion.scan)] = ion.intensity;
                        }
                    }
                }

                RealMatrix pepXIC_smoothed;
                if (ms2index.sg_smoothing_data_points == 5) {
                    pepXIC_smoothed = SGFilter5points.paddedSavitzkyGolaySmooth3(frag_int);
                } else if (ms2index.sg_smoothing_data_points == 7) {
                    pepXIC_smoothed = SGFilter7points.paddedSavitzkyGolaySmooth3(frag_int);
                } else if (ms2index.sg_smoothing_data_points == 9) {
                    pepXIC_smoothed = SGFilter.paddedSavitzkyGolaySmooth3(frag_int);
                } else if (ms2index.sg_smoothing_data_points == 3) {
                    pepXIC_smoothed = SGFilter3points.paddedSavitzkyGolaySmooth3(frag_int);
                } else {
                    // in default use 5 data points
                    pepXIC_smoothed = SGFilter5points.paddedSavitzkyGolaySmooth3(frag_int);
                }

                if (peak.boundary_right_index >= peak.apex_index) {

                    try {
                        if (refine_peak_boundary) {
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
                                    if(CParameter.verbose == CParameter.VerboseType.DEBUG) {
                                        System.err.println("The original apex index is not in the refined peak boundary:" + original_peak_index + "," +
                                                boundary_left_index + "," +
                                                boundary_right_index + "," +
                                                peak.apex_index + "," +
                                                peak.boundary_left_index + "," + peak.boundary_right_index + "," +
                                                peak.boundary_left_rt + "," +
                                                peak.apex_rt + "," +
                                                peak.boundary_right_rt + "," +
                                                pepXIC_smoothed.getRowDimension() + "," +
                                                pepXIC_smoothed.getColumnDimension());
                                        System.err.println(peptideMatch.peptide.getSequence() + "\t" + peptideMatch.index + "\t" + peptideMatch.precursor_charge + "\t" + peptideMatch.scan + "\t" + peptideMatch.rt_start + "\t" + peptideMatch.rt_apex + "\t" + peptideMatch.rt_end);
                                    }
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
                } else {
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

    /**
     * Extract the XIC (Extracted Ion Chromatogram) for a given peptide match. This
     * is used for TIMS-TOF DIA data.
     * 
     * @param ms2index     DIA index containing MS2 data
     * @param peptideMatch A PeptideMatch object containing information about the
     *                     peptide
     * @param isoWinID     Isolation window ID for the XIC extraction
     */
    private void xic_query_ccs(CCSDIAIndex ms2index, PeptideMatch peptideMatch, String isoWinID) {
        LibSpectrum libSpectrum = peptideMatch.libSpectrum;
        boolean is_ppm = CParameter.itolu.equalsIgnoreCase("ppm");
        ArrayList<LPeak> peaks = new ArrayList<>(libSpectrum.spectrum.mz.length);
        IntStream.range(0, libSpectrum.spectrum.mz.length).forEach(i -> {
            LPeak p = new LPeak(libSpectrum.spectrum.mz[i], 0.0);
            peaks.add(p);
        });
        peaks.sort(new LPeakComparatorMax2Min());

        double rt_start;
        double rt_end;
        if (this.refine_peak_boundary) {
            rt_start = Math.max(0, peptideMatch.rt_apex - CParameter.rt_win);
            rt_end = peptideMatch.rt_apex + CParameter.rt_win;
        } else {
            rt_start = peptideMatch.rt_start - this.rt_win_offset;
            rt_end = peptideMatch.rt_end + this.rt_win_offset;
        }

        Map<Double, ArrayList<JFragmentIonIM>> res = peaks.subList(0, peaks.size())
                .stream()
                .map(p -> p.mz)
                .distinct()
                .collect(toMap(
                        mz -> mz,
                        mz -> this.single_fragment_ion_query_for_dia_ccs(ms2index, mz, peptideMatch.im, rt_start, rt_end, is_ppm,isoWinID)));

        List<Double> all_mzs = new ArrayList<>(res.keySet());
        for (double mz : all_mzs) {
            if (res.get(mz).isEmpty()) {
                res.remove(mz);
            }
        }
        // mz of each fragment ion
        List<Double> fragment_ions = res.keySet().stream().sorted().collect(toList());
        if (res.size() >= 4) {
            List<Integer> unique_scans = res.values().stream()
                    .flatMap(Collection::stream)
                    .map(ion -> ion.scan)
                    .distinct()
                    .sorted()
                    .toList();

            if (unique_scans.size() >= ms2index.min_scan_for_peak) {

                int scan_min = Collections.min(unique_scans);
                int scan_max = Collections.max(unique_scans);
                int index_min = ms2index.isolation_win2scan2index.get(isoWinID).get(scan_min);
                int index_max = ms2index.isolation_win2scan2index.get(isoWinID).get(scan_max);

                // extend to the extraction window
                // left side
                if (ms2index.get_rt_by_scan(isoWinID, scan_min) > rt_start
                        && Math.abs(ms2index.get_rt_by_scan(isoWinID, scan_min) - rt_start) > 0.01) {
                    int scan_i = scan_min;
                    int index_i = index_min;
                    while (ms2index.isolation_win2scan2rt.get(isoWinID).containsKey(scan_i)
                            && ms2index.get_rt_by_scan(isoWinID, scan_i) > rt_start) {
                        index_i = index_i - 1;
                        if (ms2index.isolation_win2index2scan.get(isoWinID).containsKey(index_i)) {
                            scan_i = ms2index.get_scan_by_index(isoWinID, index_i);
                        } else {
                            index_i = index_i + 1;
                            break;
                        }
                    }
                    index_min = index_i;
                }
                // right side
                if (ms2index.get_rt_by_scan(isoWinID, scan_max) < rt_end
                        && Math.abs(ms2index.get_rt_by_scan(isoWinID, scan_max) - rt_end) > 0.01) {
                    int scan_i = scan_max;
                    int index_i = index_max;
                    while (ms2index.isolation_win2scan2rt.get(isoWinID).containsKey(scan_i)
                            && ms2index.get_rt_by_scan(isoWinID, scan_i) < rt_end) {
                        index_i = index_i + 1;
                        if (ms2index.isolation_win2index2scan.get(isoWinID).containsKey(index_i)) {
                            scan_i = ms2index.get_scan_by_index(isoWinID, index_i);
                        } else {
                            index_i = index_i - 1;
                            break;
                        }
                    }
                    index_max = index_i;
                }

                int scan_num = index_max - index_min + 1;

                HashMap<Integer, Integer> scan2index = new HashMap<>(unique_scans.size());
                HashMap<Integer, Integer> index2scan = new HashMap<>(unique_scans.size());
                // HashMap<Integer, Float> scan2rt = new HashMap<>(unique_scans.size());
                double[] index2rt_tmp = new double[scan_num];

                PeptidePeak peak = new PeptidePeak();
                peak.fragment_ions_mz = fragment_ions;
                double rt;
                double max_rt = 0;
                boolean apex_found = false;
                double left_rt_diff = Double.POSITIVE_INFINITY;
                double right_rt_diff = Double.POSITIVE_INFINITY;
                double apex_rt_diff = Double.POSITIVE_INFINITY;
                for (int k = 0, i = -1; k < scan_num; k++) {

                    int cur_index = index_min + k;
                    int cur_scan = ms2index.get_scan_by_index(isoWinID, cur_index);
                    // filter by ccs
                    if (Math.abs(peptideMatch.im - ms2index.get_ccs_by_scan(isoWinID, cur_scan)) > CParameter.ccs_tol) {
                        /**
                        System.out.println(peptideMatch.peptide.getSequence()+"\t"+k+"\t"+i+"\t"+peptideMatch.im+"\t"+
                                ms2index.get_ccs_by_scan(isoWinID, cur_scan)+"\t"+
                                ms2index.get_rt_by_scan(isoWinID,cur_scan)+"\t"+
                                peptideMatch.rt_apex+"\t"+
                                Math.abs(ms2index.get_rt_by_scan(isoWinID,cur_scan)-peptideMatch.rt_apex));
                         **/
                        continue;
                    }
                    i++;
                    scan2index.put(cur_scan, i);
                    index2scan.put(i, cur_scan);
                    /**
                    System.out.println("="+peptideMatch.peptide.getSequence()+"\t"+k+"\t"+i+"\t"+cur_scan+"\t"+peptideMatch.im+"\t"+
                            ms2index.get_ccs_by_scan(isoWinID, cur_scan)+"\t"+
                            ms2index.get_rt_by_scan(isoWinID,cur_scan)+"\t"+
                            peptideMatch.rt_apex+"\t"+
                            Math.abs(ms2index.get_rt_by_scan(isoWinID,cur_scan)-peptideMatch.rt_apex));
                     **/
                    rt = ms2index.get_rt_by_scan(isoWinID, cur_scan);
                    index2rt_tmp[i] = rt;

                    if (Math.abs(rt - peptideMatch.rt_apex) < apex_rt_diff) {
                        peak.apex_index = i;
                        apex_rt_diff = Math.abs(rt - peptideMatch.rt_apex);
                    }
                    if (Math.abs(rt - peptideMatch.rt_start) <= left_rt_diff) {
                        peak.boundary_left_index = i;
                        left_rt_diff = Math.abs(rt - peptideMatch.rt_start);
                    }
                    if (Math.abs(rt - peptideMatch.rt_end) <= right_rt_diff) {
                        peak.boundary_right_index = i;
                        right_rt_diff = Math.abs(rt - peptideMatch.rt_end);
                    }
                    if (max_rt < rt) {
                        max_rt = rt;
                    }

                }

                double[] index2rt = new double[index2scan.size()]; // Create a new array with 5 elements.
                System.arraycopy(index2rt_tmp, 0, index2rt, 0, index2scan.size()); // Copy elements from index 0 to 4

                scan_num = index2scan.size();
                if(apex_rt_diff > 0.1){
                    System.out.println("Apex detection warning:"+apex_rt_diff+"\t"+peptideMatch.peptide.getSequence()+"\t"+peptideMatch.rt_apex+","+peptideMatch.rt_start+","+peptideMatch.rt_end);
                    // System.exit(1);
                }
                int mz_length = libSpectrum.spectrum.mz.length;
                HashMap<Double, Double> mz2int = new HashMap<>(mz_length);
                for (int i = 0; i < mz_length; i++) {
                    mz2int.put(libSpectrum.spectrum.mz[i], libSpectrum.spectrum.intensity[i]);
                }

                // fragment ion intensity
                double[][] frag_int = new double[res.size()][scan_num];

                for (int j = 0; j < fragment_ions.size(); j++) {
                    for (JFragmentIonIM ion : res.get(fragment_ions.get(j))) {
                        // System.out.println(j+"\t"+ scan2index.get(ion.scan));
                        if (frag_int[j][scan2index.get(ion.scan)] < ion.intensity) {
                            frag_int[j][scan2index.get(ion.scan)] = ion.intensity;
                        }
                    }
                }

                RealMatrix pepXIC_smoothed;
                if (ms2index.sg_smoothing_data_points == 5) {
                    pepXIC_smoothed = SGFilter5points.paddedSavitzkyGolaySmooth3(frag_int);
                } else if (ms2index.sg_smoothing_data_points == 7) {
                    pepXIC_smoothed = SGFilter7points.paddedSavitzkyGolaySmooth3(frag_int);
                } else if (ms2index.sg_smoothing_data_points == 9) {
                    pepXIC_smoothed = SGFilter.paddedSavitzkyGolaySmooth3(frag_int);
                } else if (ms2index.sg_smoothing_data_points == 3) {
                    pepXIC_smoothed = SGFilter3points.paddedSavitzkyGolaySmooth3(frag_int);
                } else {
                    // in default use 5 data points
                    pepXIC_smoothed = SGFilter5points.paddedSavitzkyGolaySmooth3(frag_int);
                }

                if (peak.boundary_right_index >= peak.apex_index) {

                    try {
                        if (refine_peak_boundary) {
                            long original_peak_index = peak.apex_index;
                            long boundary_left_index = peak.boundary_left_index;
                            long boundary_right_index = peak.boundary_right_index;
                            boolean is_refined = refine_peak_boundary_detection_ccs(pepXIC_smoothed, peak, ms2index, isoWinID, index2scan,libSpectrum);
                            if(is_refined){
                                if(peak.boundary_left_index <= original_peak_index && original_peak_index <= peak.boundary_right_index ){
                                    //peak.apex_index = original_peak_index;
                                    peptideMatch.rt_start = peak.boundary_left_rt;
                                    peptideMatch.rt_end = peak.boundary_right_rt;
                                    peptideMatch.rt_apex = peak.apex_rt;
                                }else{
                                    if(CParameter.verbose == CParameter.VerboseType.DEBUG) {
                                        System.err.println("The original apex index is not in the refined peak boundary:" + original_peak_index + "," +
                                                boundary_left_index + "," +
                                                boundary_right_index + "," +
                                                peak.apex_index + "," +
                                                peak.boundary_left_index + "," + peak.boundary_right_index + "," +
                                                peak.boundary_left_rt + "," +
                                                peak.apex_rt + "," +
                                                peak.boundary_right_rt + "," +
                                                pepXIC_smoothed.getRowDimension() + "," +
                                                pepXIC_smoothed.getColumnDimension());
                                        System.err.println(peptideMatch.peptide.getSequence() + "\t" + peptideMatch.index + "\t" + peptideMatch.precursor_charge + "\t" + peptideMatch.scan + "\t" + peptideMatch.rt_start + "\t" + peptideMatch.rt_apex + "\t" + peptideMatch.rt_end);
                                    }
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
                } else {
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

    /**
     * XIC (Extracted Ion Chromatogram) data processing for a given peptide match.
     * This is used for TIMS-TOF DIA data.
     * 
     * @param xicQueryResult An XICQueryResult object which contains the extracted
     *                       XIC data
     * @param peptideMatch   A PeptideMatch object containing information about the
     *                       peptide
     */
    private void xic_query_ccs(XICQueryResult xicQueryResult, PSMQueryResult PSMQueryResult, PeptideMatch peptideMatch,
            CCSDIAIndex ms2index) {
        LibSpectrum libSpectrum = peptideMatch.libSpectrum;
        boolean is_ppm = CParameter.itolu.equalsIgnoreCase("ppm");
        // only contain fragment ions with intensity > 0 from spectra query
        ArrayList<LPeak> peaks = new ArrayList<>();
        HashSet<Double> valid_mzs = new HashSet<>();
        for (int i = 0; i < PSMQueryResult.fragment_mzs.length; i++) {
            if (PSMQueryResult.fragment_intensities[i] > 0) {
                LPeak p = new LPeak(PSMQueryResult.fragment_mzs[i], 0.0);
                peaks.add(p);
                valid_mzs.add(PSMQueryResult.fragment_mzs[i]);
            }
        }
        peaks.sort(new LPeakComparatorMax2Min());

        double rt_start;
        double rt_end;
        if (this.refine_peak_boundary) {
            rt_start = Math.max(0, peptideMatch.rt_apex - CParameter.rt_win);
            rt_end = peptideMatch.rt_apex + CParameter.rt_win;
        } else {
            rt_start = peptideMatch.rt_start - this.rt_win_offset;
            rt_end = peptideMatch.rt_end + this.rt_win_offset;
        }
        Map<Double, ArrayList<JFragmentIonIM>> res = new HashMap<>();
        for (int i = 0; i < xicQueryResult.fragment_mzs.length; i++) {
            double mz = xicQueryResult.fragment_mzs[i];
            if (valid_mzs.contains(mz)) {
                res.put(mz, new ArrayList<>());
                for (int j = 0; j < xicQueryResult.retention_time_results_seconds.length; j++) {
                    // scan number is from 0 to xicQueryResult.retention_time_results_seconds.size() - 1
                    JFragmentIonIM ion = new JFragmentIonIM((float) mz,
                            (float) xicQueryResult.fragment_intensities[i][j],
                            (float) xicQueryResult.mobility_ook0,
                            j);
                    ion.rt = (float) (xicQueryResult.retention_time_results_seconds[i] / 60.0F);
                    res.get(mz).add(ion);
                }
            }
        }

        List<Double> all_mzs = new ArrayList<>(res.keySet());
        for (double mz : all_mzs) {
            if (res.get(mz).isEmpty()) {
                res.remove(mz);
            }
        }
        // mz of each fragment ion
        List<Double> fragment_ions = res.keySet().stream().sorted().collect(toList());
        if (res.size() >= 4) {
            List<Integer> unique_scans = res.values().stream()
                    .flatMap(Collection::stream)
                    .map(ion -> ion.scan)
                    .distinct()
                    .sorted()
                    .toList();

            if (!unique_scans.isEmpty()) {

                int scan_min = Collections.min(unique_scans);
                int scan_max = Collections.max(unique_scans);
                int index_min = scan_min;
                int index_max = scan_max;

                int scan_num = index_max - index_min + 1;

                HashMap<Integer, Integer> scan2index = new HashMap<>(unique_scans.size());
                HashMap<Integer, Integer> index2scan = new HashMap<>(unique_scans.size());
                // HashMap<Integer, Float> scan2rt = new HashMap<>(unique_scans.size());
                double[] index2rt_tmp = new double[scan_num];

                PeptidePeak peak = new PeptidePeak();
                peak.fragment_ions_mz = fragment_ions;
                double rt;
                double max_rt = 0;
                boolean apex_found = false;
                double left_rt_diff = Double.POSITIVE_INFINITY;
                double right_rt_diff = Double.POSITIVE_INFINITY;
                double apex_rt_diff = Double.POSITIVE_INFINITY;
                for (int k = 0; k < scan_num; k++) {
                    scan2index.put(k, k);
                    index2scan.put(k, k);
                    rt = xicQueryResult.retention_time_results_seconds[k] / 60.0F;
                    index2rt_tmp[k] = rt;

                    if (Math.abs(rt - peptideMatch.rt_apex) < apex_rt_diff) {
                        peak.apex_index = k;
                        apex_rt_diff = Math.abs(rt - peptideMatch.rt_apex);
                    }
                    if (Math.abs(rt - peptideMatch.rt_start) <= left_rt_diff) {
                        peak.boundary_left_index = k;
                        left_rt_diff = Math.abs(rt - peptideMatch.rt_start);
                    }
                    if (Math.abs(rt - peptideMatch.rt_end) <= right_rt_diff) {
                        peak.boundary_right_index = k;
                        right_rt_diff = Math.abs(rt - peptideMatch.rt_end);
                    }
                    if (max_rt < rt) {
                        max_rt = rt;
                    }

                }

                double[] index2rt = new double[index2scan.size()]; // Create a new array with 5 elements.
                System.arraycopy(index2rt_tmp, 0, index2rt, 0, index2scan.size()); // Copy elements from index 0 to 4

                scan_num = index2scan.size();
                if(apex_rt_diff > 0.1){
                    System.out.println("Apex detection warning:"+apex_rt_diff+"\t"+peptideMatch.peptide.getSequence()+"\t"+peptideMatch.rt_apex+","+peptideMatch.rt_start+","+peptideMatch.rt_end);
                    // System.exit(1);
                }
                int mz_length = libSpectrum.spectrum.mz.length;
                HashMap<Double, Double> mz2int = new HashMap<>(mz_length);
                for (int i = 0; i < mz_length; i++) {
                    mz2int.put(libSpectrum.spectrum.mz[i], libSpectrum.spectrum.intensity[i]);
                }

                // fragment ion intensity
                double[][] frag_int = new double[res.size()][scan_num];

                for (int j = 0; j < fragment_ions.size(); j++) {
                    for (JFragmentIonIM ion : res.get(fragment_ions.get(j))) {
                        // System.out.println(j+"\t"+ scan2index.get(ion.scan));
                        if (frag_int[j][scan2index.get(ion.scan)] < ion.intensity) {
                            frag_int[j][scan2index.get(ion.scan)] = ion.intensity;
                        }
                    }
                }

                RealMatrix pepXIC_smoothed;
                if (ms2index.sg_smoothing_data_points == 5) {
                    pepXIC_smoothed = SGFilter5points.paddedSavitzkyGolaySmooth3(frag_int);
                } else if (ms2index.sg_smoothing_data_points == 7) {
                    pepXIC_smoothed = SGFilter7points.paddedSavitzkyGolaySmooth3(frag_int);
                } else if (ms2index.sg_smoothing_data_points == 9) {
                    pepXIC_smoothed = SGFilter.paddedSavitzkyGolaySmooth3(frag_int);
                } else if (ms2index.sg_smoothing_data_points == 3) {
                    pepXIC_smoothed = SGFilter3points.paddedSavitzkyGolaySmooth3(frag_int);
                } else {
                    // in default use 5 data points
                    pepXIC_smoothed = SGFilter5points.paddedSavitzkyGolaySmooth3(frag_int);
                }

                if (peak.boundary_right_index >= peak.apex_index) {

                    try {
                        if (refine_peak_boundary) {
                            long original_peak_index = peak.apex_index;
                            long boundary_left_index = peak.boundary_left_index;
                            long boundary_right_index = peak.boundary_right_index;
                            boolean is_refined = refine_peak_boundary_detection_ccs(pepXIC_smoothed, peak, xicQueryResult, libSpectrum);
                            if(is_refined){
                                if(peak.boundary_left_index <= original_peak_index && original_peak_index <= peak.boundary_right_index ){
                                    //peak.apex_index = original_peak_index;
                                    peptideMatch.rt_start = peak.boundary_left_rt;
                                    peptideMatch.rt_end = peak.boundary_right_rt;
                                    peptideMatch.rt_apex = peak.apex_rt;
                                }else{
                                    if(CParameter.verbose == CParameter.VerboseType.DEBUG) {
                                        System.err.println("The original apex index is not in the refined peak boundary:" + original_peak_index + "," +
                                                boundary_left_index + "," +
                                                boundary_right_index + "," +
                                                peak.apex_index + "," +
                                                peak.boundary_left_index + "," + peak.boundary_right_index + "," +
                                                peak.boundary_left_rt + "," +
                                                peak.apex_rt + "," +
                                                peak.boundary_right_rt + "," +
                                                pepXIC_smoothed.getRowDimension() + "," +
                                                pepXIC_smoothed.getColumnDimension());
                                        System.err.println(peptideMatch.peptide.getSequence() + "\t" + peptideMatch.index + "\t" + peptideMatch.precursor_charge + "\t" + peptideMatch.scan + "\t" + peptideMatch.rt_start + "\t" + peptideMatch.rt_apex + "\t" + peptideMatch.rt_end);
                                    }
                                    // If this is the case, use the original peak boundary
                                    peak.boundary_left_index = boundary_left_index;
                                    peak.boundary_right_index = boundary_right_index;
                                    peak.apex_index = original_peak_index;
                                    peak.boundary_left_rt = xicQueryResult.retention_time_results_seconds[(int) peak.boundary_left_index]/60.0;
                                    peak.boundary_right_rt = xicQueryResult.retention_time_results_seconds[(int) peak.boundary_right_index]/60.0;
                                    peak.apex_rt = xicQueryResult.retention_time_results_seconds[(int) peak.apex_index]/60.0;
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
                            int cur_scan = cur_index;
                            rt = xicQueryResult.retention_time_results_seconds[cur_scan] / 60.0;
                            System.out.println(rt);

                        }
                        e.printStackTrace();
                        System.exit(1);
                    }
                } else {
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
                        int cur_scan = cur_index;
                        rt = xicQueryResult.retention_time_results_seconds[cur_scan] / 60.0;
                        System.out.println(rt);

                    }
                }

            }
        }
    }

    public void xic_query_ccs(XICQueryResult xicQueryResult, double[] matched_mzs, PeptideMatch peptideMatch,
            CCSDIAIndex ms2index) {
        LibSpectrum libSpectrum = peptideMatch.libSpectrum;
        boolean is_ppm = CParameter.itolu.equalsIgnoreCase("ppm");
        // only contain fragment ions with intensity > 0 from spectra query
        ArrayList<LPeak> peaks = new ArrayList<>();
        HashSet<Double> valid_mzs = new HashSet<>();
        for (int i = 0; i < matched_mzs.length; i++) {
            LPeak p = new LPeak(matched_mzs[i], 0.0);
            peaks.add(p);
            valid_mzs.add(matched_mzs[i]);
        }
        peaks.sort(new LPeakComparatorMax2Min());

        double rt_start;
        double rt_end;
        if (this.refine_peak_boundary) {
            rt_start = Math.max(0, peptideMatch.rt_apex - CParameter.rt_win);
            rt_end = peptideMatch.rt_apex + CParameter.rt_win;
        } else {
            rt_start = peptideMatch.rt_start - this.rt_win_offset;
            rt_end = peptideMatch.rt_end + this.rt_win_offset;
        }
        Map<Double, ArrayList<JFragmentIonIM>> res = new HashMap<>();
        for (int i = 0; i < xicQueryResult.fragment_mzs.length; i++) {
            double mz = xicQueryResult.fragment_mzs[i];
            if (valid_mzs.contains(mz)) {
                res.put(mz, new ArrayList<>());
                for (int j = 0; j < xicQueryResult.retention_time_results_seconds.length; j++) {
                    // scan number is from 0 to xicQueryResult.retention_time_results_seconds.size() - 1
                    JFragmentIonIM ion = new JFragmentIonIM((float) mz,
                            (float) xicQueryResult.fragment_intensities[i][j],
                            (float) xicQueryResult.mobility_ook0,
                            j);
                    ion.rt = (float) (xicQueryResult.retention_time_results_seconds[j] / 60.0F);
                    res.get(mz).add(ion);
                }
            }
        }

        List<Double> all_mzs = new ArrayList<>(res.keySet());
        for (double mz : all_mzs) {
            if (res.get(mz).isEmpty()) {
                res.remove(mz);
            }
        }
        // mz of each fragment ion
        List<Double> fragment_ions = res.keySet().stream().sorted().collect(toList());
        if (res.size() >= 4) {
            List<Integer> unique_scans = res.values().stream()
                    .flatMap(Collection::stream)
                    .map(ion -> ion.scan)
                    .distinct()
                    .sorted()
                    .toList();

            if (!unique_scans.isEmpty()) {

                int scan_min = Collections.min(unique_scans);
                int scan_max = Collections.max(unique_scans);
                int index_min = scan_min;
                int index_max = scan_max;

                int scan_num = index_max - index_min + 1;

                HashMap<Integer, Integer> scan2index = new HashMap<>(unique_scans.size());
                HashMap<Integer, Integer> index2scan = new HashMap<>(unique_scans.size());
                // HashMap<Integer, Float> scan2rt = new HashMap<>(unique_scans.size());
                double[] index2rt_tmp = new double[scan_num];

                PeptidePeak peak = new PeptidePeak();
                peak.fragment_ions_mz = fragment_ions;
                double rt;
                double max_rt = 0;
                boolean apex_found = false;
                double left_rt_diff = Double.POSITIVE_INFINITY;
                double right_rt_diff = Double.POSITIVE_INFINITY;
                double apex_rt_diff = Double.POSITIVE_INFINITY;
                for (int k = 0; k < scan_num; k++) {
                    scan2index.put(k, k);
                    index2scan.put(k, k);
                    rt = xicQueryResult.retention_time_results_seconds[k] / 60.0F;
                    index2rt_tmp[k] = rt;

                    if (Math.abs(rt - peptideMatch.rt_apex) < apex_rt_diff) {
                        peak.apex_index = k;
                        apex_rt_diff = Math.abs(rt - peptideMatch.rt_apex);
                    }
                    if (Math.abs(rt - peptideMatch.rt_start) <= left_rt_diff) {
                        peak.boundary_left_index = k;
                        left_rt_diff = Math.abs(rt - peptideMatch.rt_start);
                    }
                    if (Math.abs(rt - peptideMatch.rt_end) <= right_rt_diff) {
                        peak.boundary_right_index = k;
                        right_rt_diff = Math.abs(rt - peptideMatch.rt_end);
                    }
                    if (max_rt < rt) {
                        max_rt = rt;
                    }

                }

                double[] index2rt = new double[index2scan.size()]; // Create a new array with 5 elements.
                System.arraycopy(index2rt_tmp, 0, index2rt, 0, index2scan.size()); // Copy elements from index 0 to 4

                scan_num = index2scan.size();
                if(apex_rt_diff > 0.1){
                    System.out.println("Apex detection warning:"+apex_rt_diff+"\t"+peptideMatch.peptide.getSequence()+"\t"+peptideMatch.rt_apex+","+peptideMatch.rt_start+","+peptideMatch.rt_end);
                    // System.exit(1);
                }
                int mz_length = libSpectrum.spectrum.mz.length;
                HashMap<Double, Double> mz2int = new HashMap<>(mz_length);
                for (int i = 0; i < mz_length; i++) {
                    mz2int.put(libSpectrum.spectrum.mz[i], libSpectrum.spectrum.intensity[i]);
                }

                // fragment ion intensity
                double[][] frag_int = new double[res.size()][scan_num];

                for (int j = 0; j < fragment_ions.size(); j++) {
                    for (JFragmentIonIM ion : res.get(fragment_ions.get(j))) {
                        // System.out.println(j+"\t"+ scan2index.get(ion.scan));
                        if (frag_int[j][scan2index.get(ion.scan)] < ion.intensity) {
                            frag_int[j][scan2index.get(ion.scan)] = ion.intensity;
                        }
                    }
                }

                RealMatrix pepXIC_smoothed;
                if (ms2index.sg_smoothing_data_points == 5) {
                    pepXIC_smoothed = SGFilter5points.paddedSavitzkyGolaySmooth3(frag_int);
                } else if (ms2index.sg_smoothing_data_points == 7) {
                    pepXIC_smoothed = SGFilter7points.paddedSavitzkyGolaySmooth3(frag_int);
                } else if (ms2index.sg_smoothing_data_points == 9) {
                    pepXIC_smoothed = SGFilter.paddedSavitzkyGolaySmooth3(frag_int);
                } else if (ms2index.sg_smoothing_data_points == 3) {
                    pepXIC_smoothed = SGFilter3points.paddedSavitzkyGolaySmooth3(frag_int);
                } else {
                    // in default use 5 data points
                    pepXIC_smoothed = SGFilter5points.paddedSavitzkyGolaySmooth3(frag_int);
                }

                if (peak.boundary_right_index >= peak.apex_index) {
                    boolean refined_peak_boundary = false;
                    try {
                        if (refine_peak_boundary) {
                            long original_peak_index = peak.apex_index;
                            long boundary_left_index = peak.boundary_left_index;
                            long boundary_right_index = peak.boundary_right_index;
                            boolean is_refined = refine_peak_boundary_detection_ccs(pepXIC_smoothed, peak, xicQueryResult, libSpectrum);

                            if(is_refined){
                                if(peak.boundary_left_index <= original_peak_index && original_peak_index <= peak.boundary_right_index ){
                                    //peak.apex_index = original_peak_index;
                                    peptideMatch.rt_start = peak.boundary_left_rt;
                                    peptideMatch.rt_end = peak.boundary_right_rt;
                                    peptideMatch.rt_apex = peak.apex_rt;
                                    refined_peak_boundary = true;
                                }else{
                                    if(CParameter.verbose == CParameter.VerboseType.DEBUG) {
                                        System.err.println("The original apex index is not in the refined peak boundary:" +
                                                xicQueryResult.id + "," +
                                                original_peak_index + "," +
                                                boundary_left_index + "," +
                                                boundary_right_index + "," +
                                                peak.apex_index + "," +
                                                peak.boundary_left_index + "," + peak.boundary_right_index + "," +
                                                peak.boundary_left_rt + "," +
                                                peak.apex_rt + "," +
                                                peak.boundary_right_rt + "," +
                                                pepXIC_smoothed.getRowDimension() + "," +
                                                pepXIC_smoothed.getColumnDimension());
                                        System.err.println(peptideMatch.peptide.getSequence() + "\t" + peptideMatch.index + "\t" + peptideMatch.precursor_charge + "\t" + peptideMatch.scan + "\t" + peptideMatch.rt_start + "\t" + peptideMatch.rt_apex + "\t" + peptideMatch.rt_end);
                                    }
                                    // If this is the case, use the original peak boundary
                                    peak.boundary_left_index = boundary_left_index;
                                    peak.boundary_right_index = boundary_right_index;
                                    peak.apex_index = original_peak_index;
                                    peak.boundary_left_rt = xicQueryResult.retention_time_results_seconds[(int) peak.boundary_left_index]/60.0;
                                    peak.boundary_right_rt = xicQueryResult.retention_time_results_seconds[(int) peak.boundary_right_index]/60.0;
                                    peak.apex_rt = xicQueryResult.retention_time_results_seconds[(int) peak.apex_index]/60.0;
                                    peptideMatch.rt_start = peak.boundary_left_rt;
                                    peptideMatch.rt_end = peak.boundary_right_rt;
                                    peptideMatch.rt_apex = peak.apex_rt;
                                }
                            }else{
                                if(CParameter.verbose == CParameter.VerboseType.DEBUG){
                                    Cloger.getInstance().logger.warn("No refining: "+xicQueryResult.id+"\t"+peptideMatch.index+"\t"+peptideMatch.scan+"\t"+peptideMatch.rt_start+"\t"+peptideMatch.rt_apex+"\t"+peptideMatch.rt_end);
                                }
                            }
                        }
                        if(refined_peak_boundary){
                            // when peak is refined successfully, use the best ion already calculated in the refining step
                            if(peak.best_ion_index >=0) {
                                peak.cor_to_best_ion = ms2index.detect_best_ion(pepXIC_smoothed, (int) peak.boundary_left_index, (int) peak.boundary_right_index, (int) peak.apex_index, peptideMatch, peak.best_ion_index);
                            }else{
                                peak.cor_to_best_ion = ms2index.detect_best_ion(pepXIC_smoothed, (int) peak.boundary_left_index, (int) peak.boundary_right_index, (int) peak.apex_index, peptideMatch);
                            }
                        } else {
                            // when peak is not refined, calculate the best ion using all fragment ions
                            peak.cor_to_best_ion = ms2index.detect_best_ion(pepXIC_smoothed, (int) peak.boundary_left_index, (int) peak.boundary_right_index, (int) peak.apex_index, peptideMatch);
                        }
                        peptideMatch.peak = peak;
                        for (int j = 0; j < fragment_ions.size(); j++) {
                            peptideMatch.mz2cor.put(fragment_ions.get(j), peak.cor_to_best_ion[j]);
                            peptideMatch.mz2skewed_peaks.put(fragment_ions.get(j), peptideMatch.skewed_peaks[j]);
                        }

                        // For XIC
                        peptideMatch.smoothed_fragment_intensities = pepXIC_smoothed;
                        peptideMatch.raw_fragment_intensities = frag_int;
                        peptideMatch.xic_rt_values = index2rt;
                        peptideMatch.rt_apex = xicQueryResult.retention_time_results_seconds[(int) peak.apex_index]/60.0;
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
                            int cur_scan = cur_index;
                            rt = xicQueryResult.retention_time_results_seconds[cur_scan] / 60.0;
                            System.out.println(rt);

                        }
                        e.printStackTrace();
                        System.exit(1);
                    }
                } else {
                    if(CParameter.verbose == CParameter.VerboseType.DEBUG){
                        System.out.println("index_start: " + peak.boundary_left_index);
                        System.out.println("index_end: " + peak.boundary_right_index);
                        System.out.println("index_apex: " + peak.apex_index);
                        System.out.println("x: " + pepXIC_smoothed.getRowDimension());
                        System.out.println("x: " + pepXIC_smoothed.getColumnDimension());
                        System.out.println(peptideMatch.rt_start);
                        System.out.println(peptideMatch.rt_apex);
                        System.out.println(peptideMatch.rt_end);
                        System.out.println(max_rt);
                    }

                    for (int i = 0; i < scan_num; i++) {
                        int cur_index = index_min + i;
                        int cur_scan = cur_index;
                        rt = xicQueryResult.retention_time_results_seconds[cur_scan] / 60.0;
                        if(CParameter.verbose == CParameter.VerboseType.DEBUG){
                            System.out.println(rt);
                        }

                    }
                }

            }
        }
    }

    /**
     * Refine the peak boundary detection for a given peptide peak in DIA data.
     * 
     * @param x           A RealMatrix format matrix containing the XIC data
     * @param peak        A PeptidePeak object containing the peak information
     * @param ms2index    DIA index containing MS2 data
     * @param isoWinID    Isolation window ID for the XIC extraction
     * @param index2scan  A map between spectrum index and scan number
     * @param libSpectrum A LibSpectrum object containing the query m/z information
     * @return A boolean value indicating whether the peak boundary is refined
     *         successfully.
     */
    public boolean refine_peak_boundary_detection(RealMatrix x, PeptidePeak peak, DIAIndex ms2index, String isoWinID,
            HashMap<Integer, Integer> index2scan, LibSpectrum libSpectrum) {
        boolean is_refined = false;
        // select the top 12 high abundant fragments
        int flank_scans = 2; // a total of 5 scans are considered to determine
        int n_ions = x.getRowDimension();
        HashMap<Integer, Double> index2intensity = new HashMap<>(n_ions);
        int left_index = Math.max((int) peak.apex_index - flank_scans, 0);
        int right_index = Math.min((int) peak.apex_index + flank_scans, x.getColumnDimension() - 1);
        int n_data_points = right_index - left_index + 1;
        HashMap<Double, Integer> mz2i = new HashMap<>();
        if (this.lf_frag_n_min > 1) {
            for (int i = 0; i < libSpectrum.ion_numbers.length; i++) {
                mz2i.put(libSpectrum.spectrum.mz[i], i);
            }
        }
        for (int i = 0; i < n_ions; i++) {
            // only consider the fragment ions with n>=this.lf_frag_n_min
            if (this.lf_frag_n_min > 1) {
                if (libSpectrum.ion_numbers[mz2i.get(peak.fragment_ions_mz.get(i))] < this.lf_frag_n_min) {
                    continue;
                }
            }
            index2intensity.put(i, StatUtils.percentile(x.getRow(i), left_index, n_data_points, 50));
        }
        if (index2intensity.size() <= 3) {
            // use all fragment ions
            for (int i = 0; i < n_ions; i++) {
                index2intensity.put(i, StatUtils.percentile(x.getRow(i), left_index, n_data_points, 50));
            }
        }
        Map<Integer, Double> sorted_index2intensity = index2intensity.entrySet().stream()
                .sorted(Map.Entry.<Integer, Double>comparingByValue().reversed())
                .limit(12)
                .filter(entry -> entry.getValue() > 0)
                .collect(toMap(Map.Entry::getKey, Map.Entry::getValue));
        if (sorted_index2intensity.size() >= 3) {
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
            PeptidePeak new_peak = xiCtool.find_max_peak(median_peaks, (int) peak.apex_index);
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
            } else {
                if(CParameter.verbose == CParameter.VerboseType.DEBUG){
                    Cloger.getInstance().logger.warn("Peak too narrow!");
                }
            }

        } else {
            if(CParameter.verbose == CParameter.VerboseType.DEBUG){
                Cloger.getInstance().logger.warn("few fragment ions detected");
            }

        }
        return is_refined;
    }

    /**
     * Refine the peak boundary detection for a given peptide peak in DIA data with
     * CCS information.
     * 
     * @param x           A RealMatrix format matrix containing the XIC data
     * @param peak        A PeptidePeak object containing the peak information
     * @param ms2index    DIA index containing MS2 data
     * @param isoWinID    Isolation window ID for the XIC extraction
     * @param index2scan  A map between spectrum index and scan number
     * @param libSpectrum A LibSpectrum object containing the query m/z information
     * @return A boolean value indicating whether the peak boundary is refined
     *         successfully.
     */
    public boolean refine_peak_boundary_detection_ccs(RealMatrix x, PeptidePeak peak, CCSDIAIndex ms2index,
            String isoWinID, HashMap<Integer, Integer> index2scan, LibSpectrum libSpectrum) {
        boolean is_refined = false;
        // select the top 12 high abundant fragments
        int flank_scans = 2; // a total of 5 scans are considered to determine
        int n_ions = x.getRowDimension();
        HashMap<Integer, Double> index2intensity = new HashMap<>(n_ions);
        int left_index = Math.max((int) peak.apex_index - flank_scans, 0);
        int right_index = Math.min((int) peak.apex_index + flank_scans, x.getColumnDimension() - 1);
        int n_data_points = right_index - left_index + 1;
        HashMap<Double, Integer> mz2i = new HashMap<>();
        if (this.lf_frag_n_min > 1) {
            for (int i = 0; i < libSpectrum.ion_numbers.length; i++) {
                mz2i.put(libSpectrum.spectrum.mz[i], i);
            }
        }
        for (int i = 0; i < n_ions; i++) {
            // only consider the fragment ions with n>=this.lf_frag_n_min
            if (this.lf_frag_n_min > 1) {
                if (libSpectrum.ion_numbers[mz2i.get(peak.fragment_ions_mz.get(i))] < this.lf_frag_n_min) {
                    continue;
                }
            }
            index2intensity.put(i, StatUtils.percentile(x.getRow(i), left_index, n_data_points, 50));
        }
        if (index2intensity.size() <= 3) {
            // use all fragment ions
            for (int i = 0; i < n_ions; i++) {
                index2intensity.put(i, StatUtils.percentile(x.getRow(i), left_index, n_data_points, 50));
            }
        }
        Map<Integer, Double> sorted_index2intensity = index2intensity.entrySet().stream()
                .sorted(Map.Entry.<Integer, Double>comparingByValue().reversed())
                .limit(12)
                .filter(entry -> entry.getValue() > 0)
                .collect(toMap(Map.Entry::getKey, Map.Entry::getValue));
        if (sorted_index2intensity.size() >= 3) {
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
            PeptidePeak new_peak = xiCtool.find_max_peak_v2(median_peaks, (int) peak.apex_index);
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
            } else {
                if(CParameter.verbose == CParameter.VerboseType.DEBUG){
                    Cloger.getInstance().logger.warn("Peak too narrow!");
                }
            }

        } else {
            if(CParameter.verbose == CParameter.VerboseType.DEBUG){
                Cloger.getInstance().logger.warn("few fragment ions detected");
            }

        }
        return is_refined;
    }

    /**
     * Refine the peak boundary detection for a given peptide peak in DIA data with
     * CCS information.
     * 
     * @param x              A RealMatrix format matrix containing the XIC data
     * @param peak           A PeptidePeak object containing the peak information
     * @param xicQueryResult An XICQueryResult object which contains the extracted
     *                       XIC data
     * @param libSpectrum    A LibSpectrum object containing the query m/z
     *                       information
     * @return A boolean value indicating whether the peak boundary is refined
     *         successfully.
     */
    public boolean refine_peak_boundary_detection_ccs(RealMatrix x, PeptidePeak peak, XICQueryResult xicQueryResult,
            LibSpectrum libSpectrum) {
        boolean is_refined = false;
        // select the top 12 high abundant fragments
        int flank_scans = 2; // a total of 5 scans are considered to determine
        int n_ions = x.getRowDimension();
        HashMap<Integer, Double> index2intensity = new HashMap<>(n_ions);
        int left_index = Math.max((int) peak.apex_index - flank_scans, 0);
        int right_index = Math.min((int) peak.apex_index + flank_scans, x.getColumnDimension() - 1);
        int n_data_points = right_index - left_index + 1;
        HashMap<Double, Integer> mz2i = new HashMap<>();
        if (this.lf_frag_n_min > 1) {
            for (int i = 0; i < libSpectrum.ion_numbers.length; i++) {
                mz2i.put(libSpectrum.spectrum.mz[i], i);
            }
        }
        for (int i = 0; i < n_ions; i++) {
            // only consider the fragment ions with n>=this.lf_frag_n_min
            if (this.lf_frag_n_min > 1) {
                if (libSpectrum.ion_numbers[mz2i.get(peak.fragment_ions_mz.get(i))] < this.lf_frag_n_min) {
                    continue;
                }
            }
            index2intensity.put(i, StatUtils.percentile(x.getRow(i), left_index, n_data_points, 50));
        }
        if (index2intensity.size() <= 3) {
            // use all fragment ions
            for (int i = 0; i < n_ions; i++) {
                index2intensity.put(i, StatUtils.percentile(x.getRow(i), left_index, n_data_points, 50));
            }
        }
        Map<Integer, Double> sorted_index2intensity = index2intensity.entrySet().stream()
                .sorted(Map.Entry.<Integer, Double>comparingByValue().reversed())
                .limit(12)
                .filter(entry -> entry.getValue() > 0)
                .collect(toMap(Map.Entry::getKey, Map.Entry::getValue));
        if (sorted_index2intensity.size() >= 3) {
            int n_scans = x.getColumnDimension();
            int[] scan_index = new int[n_scans];
            int k = 0;
            for (int i = 0; i < n_scans; i++) {
                scan_index[k] = i;
                k = k + 1;
            }
            // get the sorted keys of sorted_index2intensity to an int[]
            int[] top_ion_indices = sorted_index2intensity.keySet().stream().mapToInt(i -> i).toArray();
            // sort the int[] in ascending order: small to large
            Arrays.sort(top_ion_indices);
            // new_x only contain the top 12 fragment ions
            RealMatrix new_x = x.getSubMatrix(top_ion_indices, scan_index);
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
            PeptidePeak new_peak = xiCtool.find_max_peak_v2(median_peaks, (int) peak.apex_index);
            if ((new_peak.boundary_right_index - new_peak.boundary_left_index + 1) >= 2) {
                // left_index = (int) peak.boundary_left_index;
                peak.boundary_left_index = new_peak.boundary_left_index;
                peak.boundary_right_index = new_peak.boundary_right_index;
                peak.apex_index = new_peak.apex_index;
                // check if any of the boundaries overlaps with the apex index
                if (peak.boundary_left_index == peak.apex_index) {
                    // move left boundary
                    double [] tmp_cor2best_ion = xiCtool.detect_best_ion(new_x,(int) peak.boundary_left_index, (int) peak.boundary_right_index,(int) peak.apex_index);
                    double [] best_ion_xic = xiCtool.get_best_ion_xic(new_x, xiCtool.get_max_index(tmp_cor2best_ion));
                    PeptidePeak tmp_peak = xiCtool.find_max_peak_v2(best_ion_xic,(int) peak.apex_index);
                    peak.boundary_left_index = tmp_peak.boundary_left_index;
                    peak.apex_index = tmp_peak.apex_index;
                } else if (peak.boundary_right_index == peak.apex_index) {
                    // move right boundary
                    double [] tmp_cor2best_ion = xiCtool.detect_best_ion(new_x,(int) peak.boundary_left_index, (int) peak.boundary_right_index,(int) peak.apex_index);
                    double [] best_ion_xic = xiCtool.get_best_ion_xic(new_x, xiCtool.get_max_index(tmp_cor2best_ion));
                    PeptidePeak tmp_peak = xiCtool.find_max_peak_v2(best_ion_xic,(int) peak.apex_index);
                    peak.boundary_right_index = tmp_peak.boundary_right_index;
                    peak.apex_index = tmp_peak.apex_index;
                }
                peak.min_smoothed_intensity = new_peak.min_smoothed_intensity;
                // refine peak
                peak.cor_to_best_ion = xiCtool.detect_best_ion(new_x,(int) peak.boundary_left_index, (int) peak.boundary_right_index,(int) peak.apex_index);
                xiCtool.refine_peak(new_x,peak,peak.cor_to_best_ion,0.75,false);
                if(peak.boundary_left_index==peak.apex_index){
                    // move left boundary
                    double [] tmp_cor2best_ion = xiCtool.detect_best_ion(new_x,(int) peak.boundary_left_index, (int) peak.boundary_right_index,(int) peak.apex_index);
                    double [] best_ion_xic = xiCtool.get_best_ion_xic(new_x, xiCtool.get_max_index(tmp_cor2best_ion));
                    PeptidePeak tmp_peak = xiCtool.find_max_peak_v2(best_ion_xic,(int) peak.apex_index);
                    peak.boundary_left_index = tmp_peak.boundary_left_index;
                    peak.apex_index = tmp_peak.apex_index;
                    peak.cor_to_best_ion = xiCtool.detect_best_ion(new_x,(int) peak.boundary_left_index, (int) peak.boundary_right_index,(int) peak.apex_index);
                }else if(peak.boundary_right_index==peak.apex_index){
                    // move right boundary
                    double [] tmp_cor2best_ion = xiCtool.detect_best_ion(new_x,(int) peak.boundary_left_index, (int) peak.boundary_right_index,(int) peak.apex_index);
                    double [] best_ion_xic = xiCtool.get_best_ion_xic(new_x, xiCtool.get_max_index(tmp_cor2best_ion));
                    PeptidePeak tmp_peak = xiCtool.find_max_peak_v2(best_ion_xic,(int) peak.apex_index);
                    peak.boundary_right_index = tmp_peak.boundary_right_index;
                    peak.apex_index = tmp_peak.apex_index;
                    peak.cor_to_best_ion = xiCtool.detect_best_ion(new_x,(int) peak.boundary_left_index, (int) peak.boundary_right_index,(int) peak.apex_index);
                }
                // check if peak boundaries still valid
                double [] tmp_cor2best_ion = xiCtool.detect_best_ion(new_x,(int) peak.boundary_left_index, (int) peak.boundary_right_index,(int) peak.apex_index);
                double [] best_ion_xic = xiCtool.get_best_ion_xic(new_x, xiCtool.get_max_index(tmp_cor2best_ion));

                PeptidePeak tmp_peak = xiCtool.find_max_peak_v2(best_ion_xic, (int) peak.apex_index);
                peak.boundary_left_index = tmp_peak.boundary_left_index;
                peak.apex_index = tmp_peak.apex_index;
                peak.boundary_right_index = tmp_peak.boundary_right_index;
                peak.best_ion_index = top_ion_indices[xiCtool.get_max_index(tmp_cor2best_ion)];

                peak.boundary_left_rt = xicQueryResult.retention_time_results_seconds[(int) peak.boundary_left_index]/60.0;
                peak.boundary_right_rt = xicQueryResult.retention_time_results_seconds[(int) peak.boundary_right_index]/60.0;
                peak.apex_rt = xicQueryResult.retention_time_results_seconds[(int) peak.apex_index]/60.0;
                is_refined = true;
            } else {
                if(CParameter.verbose == CParameter.VerboseType.DEBUG){
                    Cloger.getInstance().logger.warn("Peak too narrow!");
                }
            }

        } else {
            if(CParameter.verbose == CParameter.VerboseType.DEBUG){
                Cloger.getInstance().logger.warn("few fragment ions detected");
            }

        }
        return is_refined;
    }

    /**
     * Query a single fragment ion against DIA data.
     * 
     * @param ms2index DIA index containing MS2 data
     * @param mz       The fragment ion m/z to query
     * @param rt_start The start of the retention time range for query
     * @param rt_end   The end of the retention time range for query
     * @param is_ppm   A boolean value indicating whether the m/z tolerance is in
     *                 ppm or not
     * @param isoWinID Isolation window ID for the query
     * @return An ArrayList of JFragmentIon objects containing extract fragment ion
     *         data.
     */
    private ArrayList<JFragmentIon> single_fragment_ion_query_for_dia(DIAIndex ms2index, double mz, double rt_start,
            double rt_end, boolean is_ppm, String isoWinID) {
        ArrayList<JFragmentIon> scans = new ArrayList<>();
        double[] frag_mz_range = CParameter.getRangeOfMass(mz, CParameter.itol, is_ppm);
        long mass_bin_left = ms2index.meta.get_fragment_ion_mz_bin_index(frag_mz_range[0]);
        long mass_bin_right = ms2index.meta.get_fragment_ion_mz_bin_index(frag_mz_range[1]);
        for (long frag_ion_bin = mass_bin_left; frag_ion_bin <= mass_bin_right; frag_ion_bin++) {
            // System.out.println("Here:"+isoWinID);
            if (ms2index.frag_ion_index.get(isoWinID).containsKey(frag_ion_bin)) {
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

    /**
     * Query a single fragment ion against DIA data from TIMS-TOF.
     * 
     * @param ms2index DIA index containing MS2 data
     * @param mz       The fragment ion m/z to query
     * @param ccs      CCS value
     * @param rt_start The start of the retention time range for query
     * @param rt_end   The end of the retention time range for query
     * @param is_ppm   A boolean value indicating whether the m/z tolerance is in
     *                 ppm or not
     * @param isoWinID Isolation window ID for the query
     * @return An ArrayList of JFragmentIon objects containing extract fragment ion
     *         data.
     */
    private ArrayList<JFragmentIonIM> single_fragment_ion_query_for_dia_ccs(CCSDIAIndex ms2index, double mz, double ccs,
            double rt_start, double rt_end, boolean is_ppm, String isoWinID) {
        ArrayList<JFragmentIonIM> scans = new ArrayList<>();
        double[] frag_mz_range = CParameter.getRangeOfMass(mz, CParameter.itol, is_ppm);
        long mass_bin_left = ms2index.meta.get_fragment_ion_mz_bin_index(frag_mz_range[0]);
        long mass_bin_right = ms2index.meta.get_fragment_ion_mz_bin_index(frag_mz_range[1]);
        for (long frag_ion_bin = mass_bin_left; frag_ion_bin <= mass_bin_right; frag_ion_bin++) {
            if (ms2index.frag_ion_index.get(isoWinID).containsKey(frag_ion_bin)) {
                List<JFragmentIonIM> res = ms2index.frag_ion_index.get(isoWinID).get(frag_ion_bin)
                        .stream()
                        .filter(jFragmentIon -> Math.abs(CParameter.get_mass_error(jFragmentIon.mz, mz, is_ppm)) <= CParameter.itol &&
                                jFragmentIon.rt >= rt_start && jFragmentIon.rt <= rt_end &&
                                Math.abs(jFragmentIon.im - ccs) <= CParameter.ccs_tol)
                        .collect(toList());
                scans.addAll(res);
            }
        }
        return scans;
    }

    /**
     * Get the column index of the ion type in the fragment ion intensity matrix.
     * 
     * @param ionMatch A IonMatch object which contains the information for a single
     *                 fragment ion match
     * @return The column index of the ion type in the fragment ion intensity
     *         matrix.
     */
    public int get_ion_type_column_index(IonMatch ionMatch) {
        String ion_type = "";
        if (ionMatch.ion.getSubType() == PeptideFragmentIon.B_ION) {
            ion_type = "b";
        } else if (ionMatch.ion.getSubType() == PeptideFragmentIon.Y_ION) {
            ion_type = "y";
        }
        if (ionMatch.ion.hasNeutralLosses()) {
            if (ionMatch.ion.getNeutralLosses().length == 1) {
                if (ionMatch.ion.getNeutralLosses()[0].isSameAs(NeutralLoss.H3PO4)
                        && this.mod_ai.equals("phosphorylation")) {
                    ion_type = ion_type + "_modloss_z" + ionMatch.charge;
                } else {
                    System.out.println("Neutral loss is not supported yet");
                    System.out.println(ionMatch.ion.getNeutralLosses()[0].name);
                    System.out.println(ionMatch.ion.getNeutralLosses()[0].getMass());
                    System.exit(1);
                }

            } else {
                // >=2 neutral losses
                System.out.println(">=2 neutral losses");
                for (NeutralLoss nl : ionMatch.ion.getNeutralLosses()) {
                    System.out.println(nl.name);
                }
                System.exit(1);
            }

        } else {
            // no neutral loss
            ion_type = ion_type + "_z" + ionMatch.charge;
        }
        // ion_type = ion_type + "_z" + ionMatch.charge;
        return (this.ion_type2column_index.get(ion_type));
    }

    /**
     * Get the column index of the ion type in the fragment ion intensity matrix.
     * 
     * @param ion    An Ion object which contains the information for a single
     *               fragment ion
     * @param charge The charge state of the fragment ion
     * @return The column index of the ion type in the fragment ion intensity
     *         matrix.
     */
    public int get_ion_type_column_index(Ion ion, int charge) {
        String ion_type = "";
        if (ion.getSubType() == PeptideFragmentIon.B_ION) {
            ion_type = "b";
        } else if (ion.getSubType() == PeptideFragmentIon.Y_ION) {
            ion_type = "y";
        }
        if (ion.hasNeutralLosses()) {
            if (ion.getNeutralLosses().length == 1) {
                if (ion.getNeutralLosses()[0].isSameAs(NeutralLoss.H3PO4) && this.mod_ai.equals("phosphorylation")) {
                    ion_type = ion_type + "_modloss_z" + charge;
                } else {
                    System.out.println("Neutral loss is not supported yet");
                    System.out.println(ion.getNeutralLosses()[0].name);
                    System.out.println(ion.getNeutralLosses()[0].getMass());
                    System.exit(1);
                }

            } else {
                // >=2 neutral losses
                System.out.println(">=2 neutral losses");
                for (NeutralLoss nl : ion.getNeutralLosses()) {
                    System.out.println(nl.toString());
                }
                System.exit(1);
            }

        } else {
            // no neutral loss
            ion_type = ion_type + "_z" + charge;
        }

        return (this.ion_type2column_index.get(ion_type));
    }

    /**
     * Set the column index for different types of fragment ions based on the
     * fragmentation type and maximum fragment ion charge.
     * 
     * @param fragmentation_type      Fragmentation type: HCD or CID
     * @param max_fragment_ion_charge The maximum fragment ion charge to consider
     * @param lossWaterNH3            A boolean value indicating whether to consider
     *                                neutral losses of water and ammonia
     */
    public void set_ion_type_column_index(String fragmentation_type, int max_fragment_ion_charge,
            boolean lossWaterNH3) {
        if (fragmentation_type.equalsIgnoreCase("hcd") || fragmentation_type.equalsIgnoreCase("cid")) {
            ArrayList<String> col_names = new ArrayList<>();

            int column_index = 0;
            // b ion
            for (int i = 1; i <= max_fragment_ion_charge; i++) {
                this.ion_type2column_index.put("b_z" + i, column_index);
                column_index = column_index + 1;
                col_names.add("b_z" + i);
            }
            // y ion
            for (int i = 1; i <= max_fragment_ion_charge; i++) {
                this.ion_type2column_index.put("y_z" + i, column_index);
                column_index = column_index + 1;
                col_names.add("y_z" + i);
            }

            if (!(this.mod_ai.equals("-") || this.mod_ai.equals("general"))) {
                // neutral loss
                // b ion
                for (int i = 1; i <= max_fragment_ion_charge; i++) {
                    this.ion_type2column_index.put("b_modloss_z" + i, column_index);
                    column_index = column_index + 1;
                    col_names.add("b_modloss_z" + i);
                }
                // y ion
                for (int i = 1; i <= max_fragment_ion_charge; i++) {
                    this.ion_type2column_index.put("y_modloss_z" + i, column_index);
                    column_index = column_index + 1;
                    col_names.add("y_modloss_z" + i);
                }
            }

            this.fragment_ion_intensity_head_line = StringUtils.join(col_names, "\t");
        }
    }

    /**
     * Get the valid maximum fragment ion charge based on the precursor charge and
     * the maximum fragment ion charge setting.
     * 
     * @param precursor_charge The charge state of precursor ion
     * @return The valid maximum fragment ion charge.
     */
    private int get_valid_max_fragment_ions(int precursor_charge) {
        if (precursor_charge <= 2) {
            return precursor_charge;

        }
        return Math.min(precursor_charge, this.max_fragment_ion_charge);
    }

    /**
     * Get the matched fragment ions for a given peptide and spectrum.
     * 
     * @param objPeptide              A Peptide object containing peptide sequence
     *                                and modification information
     * @param spectrum                A Spectrum object containing the MS2 spectrum
     *                                to annotate
     * @param precursor_charge        Precursor charge state
     * @param max_fragment_ion_charge Maximum fragment ion charge state to consider
     * @param lossWaterNH3            A boolean value indicating whether to consider
     *                                neutral losses of water and ammonia
     * @return An ArrayList containing the matched fragment ion information.
     */
    private ArrayList<IonMatch> get_matched_ions(Peptide objPeptide, Spectrum spectrum, int precursor_charge,
            int max_fragment_ion_charge, boolean lossWaterNH3) {

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
            if (this.fragment_ion_charge_less_than_precursor_charge) {
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

        if (this.mod_ai.equals("general") || this.mod_ai.equals("-")) {
            // no any neutral loss
        }else if(this.mod_ai.equals("phosphorylation")) {
            if (ModificationUtils.getInstance().getModificationString(objPeptide).toLowerCase().contains("phosphorylation") ||
                    ModificationUtils.getInstance().getModificationString(objPeptide).toLowerCase().contains("phospho")) {
                //annotationSettings.addNeutralLoss(NeutralLoss.H3PO4);
                specificAnnotationPreferences.setNeutralLossesMap(getNeutralLossesMap(objPeptide));
                // annotationSettings.addNeutralLoss(NeutralLoss.HPO3);
                // specificAnnotationPreferences.addNeutralLoss(NeutralLoss.H3PO4);
                // specificAnnotationPreferences.addNeutralLoss(NeutralLoss.HPO3);
            }
        } else {
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
            if (CParameter.verbose == CParameter.VerboseType.DEBUG) {
                System.err.println("No ions matched!");
            }
            return (new ArrayList<>());
        } else {
            return new ArrayList<>(Arrays.asList(matches));
        }

    }

    /**
     * Get the matched fragment ions for a given peptide and spectrum.
     * 
     * @param objPeptide              A Peptide object containing peptide sequence
     *                                and modification information
     * @param spectrum                A PSMQueryResult object containing the matched
     *                                fragment ion information
     * @param precursor_charge        Precursor charge state
     * @param max_fragment_ion_charge Maximum fragment ion charge state to consider
     * @param lossWaterNH3            A boolean value indicating whether to consider
     *                                neutral losses of water and ammonia
     * @return An ArrayList containing the matched fragment ion information.
     */
    private ArrayList<IonMatch> get_matched_ions(Peptide objPeptide, PSMQueryResult spectrum, int precursor_charge,
            int max_fragment_ion_charge, boolean lossWaterNH3) {

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
            if (this.fragment_ion_charge_less_than_precursor_charge) {
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

        if (this.mod_ai.equals("general") || this.mod_ai.equals("-")) {
            // no any neutral loss
        }else if(this.mod_ai.equals("phosphorylation")) {
            if (ModificationUtils.getInstance().getModificationString(objPeptide).toLowerCase().contains("phosphorylation") ||
                    ModificationUtils.getInstance().getModificationString(objPeptide).toLowerCase().contains("phospho")) {
                //annotationSettings.addNeutralLoss(NeutralLoss.H3PO4);
                specificAnnotationPreferences.setNeutralLossesMap(getNeutralLossesMap(objPeptide));
                // annotationSettings.addNeutralLoss(NeutralLoss.HPO3);
                // specificAnnotationPreferences.addNeutralLoss(NeutralLoss.H3PO4);
                // specificAnnotationPreferences.addNeutralLoss(NeutralLoss.HPO3);
            }
        } else {
            // TODO
        }

        annotationSettings.setIntensityThresholdType(AnnotationParameters.IntensityThresholdType.percentile);

        ModificationParameters modificationParameters = new ModificationParameters();
        SequenceMatchingParameters sequenceMatchingParameters = new SequenceMatchingParameters();
        JSequenceProvider jSequenceProvider = new JSequenceProvider();

        HashMap<Double, Double> fragment_mz2intensity = new HashMap<>();
        for (int i = 0; i < spectrum.fragment_mzs.length; i++) {
            if (spectrum.fragment_intensities[i] > 0) {
                fragment_mz2intensity.put(spectrum.fragment_mzs[i], spectrum.fragment_intensities[i]);
            }
        }

        ArrayList<IonMatch> ion_matches = new ArrayList<>();

        HashMap<Integer, ArrayList<Ion>> theoretical_ions = this.generate_theoretical_fragment_ions(objPeptide,precursor_charge);
        HashSet<Integer> possible_fragment_ion_charges = this.getPossibleFragmentIonCharges(precursor_charge);
        for (int k : theoretical_ions.keySet()) {
            for (Ion ion : theoretical_ions.get(k)) {
                if (ion.getSubType() == PeptideFragmentIon.B_ION || ion.getSubType() == PeptideFragmentIon.Y_ION) {
                    PeptideFragmentIon fragmentIon = ((PeptideFragmentIon) ion);
                    for (int frag_charge : possible_fragment_ion_charges) {
                        double frag_mz = fragmentIon.getTheoreticMz(frag_charge);
                        if (fragment_mz2intensity.containsKey(frag_mz)) {
                            IonMatch ionMatch = new IonMatch();
                            ionMatch.ion = ion;
                            ionMatch.charge = frag_charge;
                            ionMatch.peakIntensity = fragment_mz2intensity.get(frag_mz);
                            ionMatch.peakMz = frag_mz;
                            ion_matches.add(ionMatch);
                        }
                    }
                }
            }
        }

        if (ion_matches.isEmpty()) {
            if (CParameter.verbose == CParameter.VerboseType.DEBUG) {
                System.err.println("No ions matched!");
            }
            return (new ArrayList<>());
        } else {
            return ion_matches;
        }

    }

    /**
     * Get the matched fragment ions for a given peptide and spectrum.
     * 
     * @param objPeptide              A Peptide object containing peptide sequence
     *                                and modification information
     * @param precursor_charge        Precursor charge state
     * @param apex_rt_in_seconds      The apex retention time of the peptide
     *                                precursor
     * @param xic_data                A XICQueryResult object containing the matched
     *                                fragment ion information
     * @param max_fragment_ion_charge Maximum fragment ion charge state to consider
     * @param lossWaterNH3            A boolean value indicating whether to consider
     *                                neutral losses of water and ammonia
     * @param ion_matches             An ArrayList to store the matched fragment ion
     *                                information
     * @return The apex retention time of the peptide precursor in minute. A value
     *         of -1 indicates no matched ions found.
     */
    public double get_matched_ions(Peptide objPeptide, int precursor_charge, double apex_rt_in_seconds,
            XICQueryResult xic_data, int max_fragment_ion_charge, boolean lossWaterNH3,
            ArrayList<IonMatch> ion_matches) {

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
            if (this.fragment_ion_charge_less_than_precursor_charge) {
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

        if (this.mod_ai.equals("general") || this.mod_ai.equals("-")) {
            // no any neutral loss
        }else if(this.mod_ai.equals("phosphorylation")) {
            if (ModificationUtils.getInstance().getModificationString(objPeptide).toLowerCase().contains("phosphorylation") ||
                    ModificationUtils.getInstance().getModificationString(objPeptide).toLowerCase().contains("phospho")) {
                //annotationSettings.addNeutralLoss(NeutralLoss.H3PO4);
                specificAnnotationPreferences.setNeutralLossesMap(getNeutralLossesMap(objPeptide));
                // annotationSettings.addNeutralLoss(NeutralLoss.HPO3);
                // specificAnnotationPreferences.addNeutralLoss(NeutralLoss.H3PO4);
                // specificAnnotationPreferences.addNeutralLoss(NeutralLoss.HPO3);
            }
        } else {
            // TODO
        }

        annotationSettings.setIntensityThresholdType(AnnotationParameters.IntensityThresholdType.percentile);

        ModificationParameters modificationParameters = new ModificationParameters();
        SequenceMatchingParameters sequenceMatchingParameters = new SequenceMatchingParameters();
        JSequenceProvider jSequenceProvider = new JSequenceProvider();

        HashMap<Double, Double> fragment_mz2intensity = new HashMap<>();
        // binary search to find the apex RT index with the smallest difference
        int apex_rt_index = 0;
        double min_diff = Double.MAX_VALUE;
        for (int i = 0; i < xic_data.retention_time_results_seconds.length; i++) {
            double diff = Math.abs(xic_data.retention_time_results_seconds[i] - apex_rt_in_seconds);
            if (diff < min_diff) {
                min_diff = diff;
                apex_rt_index = i;
            }
        }
        // int apex_rt_index = Collections.binarySearch(xic_data.retention_time_results_seconds, apex_rt_in_seconds);
        for(int i=0;i<xic_data.fragment_mzs.length;i++){
            if(xic_data.fragment_intensities[i][apex_rt_index]>0) {
                fragment_mz2intensity.put(xic_data.fragment_mzs[i], xic_data.fragment_intensities[i][apex_rt_index]);
            }
        }

        // move to both sides to find the closest RT with matched ions if no matched ions found at the apex RT
        if(fragment_mz2intensity.isEmpty()){
            int left_index = apex_rt_index - 1;
            int right_index = apex_rt_index + 1;
            // get the sum of fragment ion intensities at each RT and pick the RT with the highest sum
            double left_sum = 0.0;
            double right_sum = 0.0;
            for (int i = 0; i < xic_data.fragment_mzs.length; i++) {
                if (xic_data.fragment_intensities[i][left_index] > 0) {
                    left_sum = left_sum + xic_data.fragment_intensities[i][left_index];
                }
            }
            for (int i = 0; i < xic_data.fragment_mzs.length; i++) {
                if (xic_data.fragment_intensities[i][right_index] > 0) {
                    right_sum = right_sum + xic_data.fragment_intensities[i][right_index];
                }
            }

            if (left_sum >= right_sum && left_sum > 0) {
                apex_rt_index = left_index;
                for(int i=0;i<xic_data.fragment_mzs.length;i++){
                    if(xic_data.fragment_intensities[i][apex_rt_index]>0) {
                        fragment_mz2intensity.put(xic_data.fragment_mzs[i], xic_data.fragment_intensities[i][apex_rt_index]);
                    }
                }
            } else if (right_sum > left_sum && right_sum > 0) {
                apex_rt_index = right_index;
                for (int i = 0; i < xic_data.fragment_mzs.length; i++) {
                    if (xic_data.fragment_intensities[i][apex_rt_index] > 0) {
                        fragment_mz2intensity.put(xic_data.fragment_mzs[i], xic_data.fragment_intensities[i][apex_rt_index]);
                    }
                }
            }
        }

        // ArrayList<IonMatch> ion_matches = new ArrayList<>();

        HashMap<Integer, ArrayList<Ion>> theoretical_ions = this.generate_theoretical_fragment_ions(objPeptide,precursor_charge);
        HashSet<Integer> possible_fragment_ion_charges = this.getPossibleFragmentIonCharges(precursor_charge);
        // This is used to avoid adding the same matched ion multiple times
        HashSet<Double> matched_mzs = new HashSet<>();
        for (int k : theoretical_ions.keySet()) {
            for (Ion ion : theoretical_ions.get(k)) {
                if (ion.getSubType() == PeptideFragmentIon.B_ION || ion.getSubType() == PeptideFragmentIon.Y_ION) {
                    PeptideFragmentIon fragmentIon = ((PeptideFragmentIon) ion);
                    for (int frag_charge : possible_fragment_ion_charges) {
                        double frag_mz = fragmentIon.getTheoreticMz(frag_charge);
                        if (fragment_mz2intensity.containsKey(frag_mz) && !matched_mzs.contains(frag_mz)) {
                            IonMatch ionMatch = new IonMatch();
                            ionMatch.ion = ion;
                            ionMatch.charge = frag_charge;
                            ionMatch.peakIntensity = fragment_mz2intensity.get(frag_mz);
                            ionMatch.peakMz = frag_mz;
                            ion_matches.add(ionMatch);
                            matched_mzs.add(frag_mz);
                        }
                    }
                }
            }
        }

        if (ion_matches.isEmpty()) {
            if (CParameter.verbose == CParameter.VerboseType.DEBUG) {
                System.err.println("No ions matched!");
            }
            // return (new ArrayList<>());
            return -1;
        } else {
            // return ion_matches;
            return xic_data.retention_time_results_seconds[apex_rt_index] / 60.0;
        }

    }

    /**
     * Get the matched fragment ions for a given peptide and spectrum: considering
     * RT range (all matches within the peak boundary).
     * 
     * @param objPeptide              A Peptide object containing peptide sequence
     *                                and modification information
     * @param precursor_charge        Precursor charge state
     * @param apex_rt_in_seconds      The apex retention time of the peptide
     *                                precursor
     * @param start_rt_in_seconds     The start retention time of the peptide
     *                                precursor
     * @param end_rt_in_seconds       The end retention time of the peptide
     *                                precursor
     * @param xic_data                A XICQueryResult object containing the matched
     *                                fragment ion information
     * @param max_fragment_ion_charge Maximum fragment ion charge state to consider
     * @param lossWaterNH3            A boolean value indicating whether to consider
     *                                neutral losses of water and ammonia
     * @param ion_matches             An ArrayList to store the matched fragment ion
     *                                information
     * @return The apex retention time of the peptide precursor in minute. A value
     *         of -1 indicates no matched ions found.
     */
    public double get_matched_ions(Peptide objPeptide, int precursor_charge, double apex_rt_in_seconds,
            double start_rt_in_seconds, double end_rt_in_seconds, XICQueryResult xic_data, int max_fragment_ion_charge,
            boolean lossWaterNH3, ArrayList<IonMatch> ion_matches) {

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
            if (this.fragment_ion_charge_less_than_precursor_charge) {
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

        if (this.mod_ai.equals("general") || this.mod_ai.equals("-")) {
            // no any neutral loss
        }else if(this.mod_ai.equals("phosphorylation")) {
            if (ModificationUtils.getInstance().getModificationString(objPeptide).toLowerCase().contains("phosphorylation") ||
                    ModificationUtils.getInstance().getModificationString(objPeptide).toLowerCase().contains("phospho")) {
                //annotationSettings.addNeutralLoss(NeutralLoss.H3PO4);
                specificAnnotationPreferences.setNeutralLossesMap(getNeutralLossesMap(objPeptide));
                // annotationSettings.addNeutralLoss(NeutralLoss.HPO3);
                // specificAnnotationPreferences.addNeutralLoss(NeutralLoss.H3PO4);
                // specificAnnotationPreferences.addNeutralLoss(NeutralLoss.HPO3);
            }
        } else {
            // TODO
        }

        annotationSettings.setIntensityThresholdType(AnnotationParameters.IntensityThresholdType.percentile);

        ModificationParameters modificationParameters = new ModificationParameters();
        SequenceMatchingParameters sequenceMatchingParameters = new SequenceMatchingParameters();
        JSequenceProvider jSequenceProvider = new JSequenceProvider();

        HashMap<Double, Double> fragment_mz2intensity = new HashMap<>();
        double[] intensity_list = new double[xic_data.fragment_mzs.length];
        for (int i = 0; i < xic_data.retention_time_results_seconds.length; i++) {
            if (xic_data.retention_time_results_seconds[i] >= start_rt_in_seconds
                    && xic_data.retention_time_results_seconds[i] <= end_rt_in_seconds) {
                for (int j = 0; j < xic_data.fragment_mzs.length; j++) {
                    intensity_list[j] = intensity_list[j] + xic_data.fragment_intensities[j][i];
                }
            }
        }
        // int apex_rt_index = Collections.binarySearch(xic_data.retention_time_results_seconds, apex_rt_in_seconds);
        for(int i=0;i<xic_data.fragment_mzs.length;i++){
            if(intensity_list[i]>0) {
                fragment_mz2intensity.put(xic_data.fragment_mzs[i], intensity_list[i]);
            }
        }

        // ArrayList<IonMatch> ion_matches = new ArrayList<>();

        HashMap<Integer, ArrayList<Ion>> theoretical_ions = this.generate_theoretical_fragment_ions(objPeptide,precursor_charge);
        HashSet<Integer> possible_fragment_ion_charges = this.getPossibleFragmentIonCharges(precursor_charge);
        // This is used to avoid adding the same matched ion multiple times
        HashSet<Double> matched_mzs = new HashSet<>();
        for (int k : theoretical_ions.keySet()) {
            for (Ion ion : theoretical_ions.get(k)) {
                if (ion.getSubType() == PeptideFragmentIon.B_ION || ion.getSubType() == PeptideFragmentIon.Y_ION) {
                    PeptideFragmentIon fragmentIon = ((PeptideFragmentIon) ion);
                    for (int frag_charge : possible_fragment_ion_charges) {
                        double frag_mz = fragmentIon.getTheoreticMz(frag_charge);
                        if (fragment_mz2intensity.containsKey(frag_mz) && !matched_mzs.contains(frag_mz)) {
                            IonMatch ionMatch = new IonMatch();
                            ionMatch.ion = ion;
                            ionMatch.charge = frag_charge;
                            ionMatch.peakIntensity = fragment_mz2intensity.get(frag_mz);
                            ionMatch.peakMz = frag_mz;
                            ion_matches.add(ionMatch);
                            matched_mzs.add(frag_mz);
                        }
                    }
                }
            }
        }

        if (ion_matches.isEmpty()) {
            if (CParameter.verbose == CParameter.VerboseType.DEBUG) {
                System.err.println("No ions matched!");
            }
            // return (new ArrayList<>());
            return -1;
        } else {
            // return ion_matches;
            return apex_rt_in_seconds / 60.0;
        }

    }

    /**
     * Get the NeutralLossesMap for a given peptide.
     * 
     * @param peptide A Peptide object
     * @return A NeutralLossesMap object.
     */
    public static NeutralLossesMap getNeutralLossesMap(Peptide peptide) {
        // ModificationFactory modificationFactory = ModificationFactory.getInstance();
        NeutralLossesMap neutralLossesMap = new NeutralLossesMap();
        String sequence = peptide.getSequence();
        int aaMin = sequence.length();
        int aaMax = 0;

        ModificationMatch[] modificationMatches = peptide.getVariableModifications();

        for (ModificationMatch modMatch : modificationMatches) {

            if(modMatch.getModification().equals("Phosphorylation of S") || modMatch.getModification().equals("Phosphorylation of T") ||
                    modMatch.getModification().equals("Phospho of S") || modMatch.getModification().equals("Phospho of T")) {
                int site = com.compomics.util.experiment.identification.utils.ModificationUtils.getSite(
                        modMatch.getSite(),
                        sequence.length());
                aaMin = site;
                aaMax = sequence.length() - site + 1;

                neutralLossesMap.addNeutralLoss(
                        NeutralLoss.H3PO4,
                        aaMin,
                        aaMax);
            }
        }
        return neutralLossesMap;
    }

    /**
     * Return a Peptide object based on the peptide sequence and modification from
     * the global peptide_mod2Peptide map.
     * 
     * @param peptide_sequence Peptide sequence
     * @param modification     Peptide modification
     * @return A Peptide object.
     */
    public Peptide get_peptide(String peptide_sequence, String modification) {
        String peptide_mod = peptide_sequence + modification;
        return peptide_mod2Peptide.get(peptide_mod);
    }

    /**
     * Generate a Peptide object based on the peptide sequence and modification,
     * then add the object to the global
     * peptide_mod2Peptide map.
     * 
     * @param peptide_sequence Peptide sequence
     * @param modification     Peptide modification
     */
    public void add_peptide(String peptide_sequence, String modification) {
        String peptide_mod = peptide_sequence + modification;
        if (!this.peptide_mod2Peptide.containsKey(peptide_mod)) {
            Peptide peptide = generatePeptide(peptide_sequence, modification);
            this.peptide_mod2Peptide.put(peptide_mod, peptide);
        }
    }

    /**
     * Generate a Peptide object based on the peptide sequence and modification
     * 
     * @param peptideSequence Peptide sequence
     * @param modifications   Peptide modification
     * @return A Peptide object.
     */
    public Peptide generatePeptide(String peptideSequence, String modifications) {
        Peptide peptide = new Peptide(peptideSequence);
        if (!modifications.equals("-")) {
            // TMT 10-plex of K@8[229.1629];TMT 10-plex of K@9[229.1629];TMT 10-plex of
            // peptide N-term@0[229.1629]
            String[] names = modifications.split(";");
            for (String s : names) {
                String name = s.replaceAll("@.*$", "");
                String pos = s.replaceAll(".*@(\\d+).*$", "$1");
                peptide.addVariableModification(new ModificationMatch(name, Integer.parseInt(pos)));
            }
        }
        peptide.getMass(modificationParameters, sequenceProvider, sequenceMatchingParameters);
        return peptide;
    }

    /**
     * Remove interference peptides from a PSM file and save the results to a new
     * PSM file. This is only used when the
     * input psm_file is in a generic format.
     * 
     * @param psm_file     A peptide detection result file
     * @param new_psm_file A new PSM file to save the data after removing interfered
     *                     peptides
     * @param fdr_cutoff   FDR cutoff used to filter the original peptide detection
     *                     result.
     */
    public void remove_interference_peptides(String psm_file, String new_psm_file, double fdr_cutoff) {
        CsvReadOptions.Builder builder = CsvReadOptions.builder(psm_file)
                .separator('\t')
                .header(true);
        CsvReadOptions options = builder.build();
        Table psmTable = Table.read().usingOptions(options);
        if (search_engine.equalsIgnoreCase("DIA-NN") || search_engine.equalsIgnoreCase("DIANN")
                || search_engine.equalsIgnoreCase("skyline")) {
            // psmTable = psmTable.sortOn("File.Name","Q.Value","");
            // TODO
        } else {

            if (fdr_cutoff > 0 && fdr_cutoff < 1) {
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

    /**
     * Remove interference peptides from a PSM file and save the results to a new
     * PSM file. This is only used when the
     * input psm_file is in a generic format.
     * 
     * @param psmTable A Table object containing peptide detection result
     * @return A Table object containing the data after removing interfered peptides
     */
    public static Table remove_interference_peptides(Table psmTable) {
        String peptide;
        String mz;
        String charge;
        double rt_start;
        double apex_rt;
        double rt_end;
        HashMap<String, ArrayList<JPeakGroup>> peptide_form2peptides = new HashMap<>();
        psmTable = psmTable.sortOn("peptide", "charge", "mz", "-rescore");
        String peptide_form;
        IntColumn valid_column = IntColumn.create("peak_share", psmTable.rowCount());
        for (int i = 0; i < psmTable.rowCount(); i++) {
            peptide = psmTable.getString(i, "peptide");
            mz = psmTable.getString(i, "mz");
            charge = psmTable.getString(i, "charge");
            rt_start = psmTable.row(i).getDouble("rt_start");
            apex_rt = psmTable.row(i).getDouble("apex_rt");
            rt_end = psmTable.row(i).getDouble("rt_end");
            peptide_form = peptide + "|" + charge + "|" + mz;
            if (!peptide_form2peptides.containsKey(peptide_form)) {
                peptide_form2peptides.put(peptide_form, new ArrayList<>());
                JPeakGroup peak = new JPeakGroup();
                peak.rt_start = rt_start;
                peak.apex_rt = apex_rt;
                peak.rt_end = rt_end;
                peak.id = i;
                peptide_form2peptides.get(peptide_form).add(peak);
                valid_column.set(i, 1);
            } else {
                // need to compare with each previous peptide form
                boolean keep = true;
                for (JPeakGroup p : peptide_form2peptides.get(peptide_form)) {
                    if (p.rt_start <= rt_start && rt_end <= p.rt_end) {
                        // one contained by another
                        keep = false;
                        break;
                    } else if (rt_start <= p.rt_start && p.rt_end <= rt_end) {
                        // one contained by another
                        keep = false;
                        break;
                    } else if (p.rt_start <= rt_start && rt_start <= p.rt_end) {
                        // overlap
                        double overlap = p.rt_end - rt_start;
                        double overlap_ratio = Math.max(overlap / (rt_end - rt_start), overlap/ (p.rt_end - p.rt_start));
                        if(overlap_ratio >= 0.5){
                            keep = false;
                            break;
                        }
                    } else if (p.rt_start <= rt_end && rt_end <= p.rt_end) {
                        // overlap
                        double overlap =rt_end - p.rt_start;
                        double overlap_ratio = Math.max(overlap / (rt_end - rt_start), overlap/ (p.rt_end - p.rt_start));
                        if(overlap_ratio >= 0.5){
                            keep = false;
                            break;
                        }
                    }
                }
                if (keep) {
                    JPeakGroup peak = new JPeakGroup();
                    peak.rt_start = rt_start;
                    peak.apex_rt = apex_rt;
                    peak.rt_end = rt_end;
                    peak.id = i;
                    peptide_form2peptides.get(peptide_form).add(peak);
                    valid_column.set(i, 1);
                } else {
                    valid_column.set(i, 0);
                }
            }

        }
        psmTable.addColumns(valid_column);
        return psmTable;
    }

    /**
     * Remove interference peptides from a PSM file and save the results to a new
     * PSM file.
     * This is used for DIA-NN result.
     * 
     * @param psm_file     A peptide detection result file
     * @param new_psm_file A new PSM file to save the data after removing interfered
     *                     peptides
     */
    public void remove_interference_peptides_diann(String psm_file, String new_psm_file) {

        Table psmTable;
        if (psm_file.endsWith(".parquet")) {
            System.out.println("The input file format is DIA-NN parquet format:" + psm_file);
            // There is no File.Name column in the parquet file
            psmTable = new TablesawParquetReader().read(new Source(new File(psm_file)));
            // if "File.Name" column is not in the parquet file, then we need to use "Run"
            if (!psmTable.columnNames().contains(PSMConfig.ms_file_column_name)) {
                if (psmTable.columnNames().contains("Run")) {
                    PSMConfig.ms_file_column_name = "Run";
                }
            }
            if (!psmTable.columnNames().contains(PSMConfig.ms2_index_column_name)) {
                if (psmTable.columnNames().contains("Ms2.Scan")) {
                    PSMConfig.ms2_index_column_name = "Ms2.Scan";
                }
            }
            // for PTM data
            if (!psmTable.columnNames().contains(PSMConfig.ptm_site_qvalue_column_name)) {
                if (psmTable.columnNames().contains("Global.Peptidoform.Q.Value")) {
                    PSMConfig.ptm_site_qvalue_column_name = "Global.Peptidoform.Q.Value";
                }
            }
            // for (String columnName : psmTable.columnNames()) {
            // System.out.println(columnName);
            // }
        } else {
            CsvReadOptions.Builder builder = CsvReadOptions.builder(psm_file)
                    .maxCharsPerColumn(10000000)
                    .separator('\t')
                    .header(true);
            CsvReadOptions options = builder.build();
            psmTable = Table.read().usingOptions(options);
        }

        String peptide;
        double mz;
        String charge;
        String mod_seq;
        double rt_start;
        double apex_rt;
        double rt_end;
        String modification;

        DoubleColumn mz_column = DoubleColumn.create("mz", psmTable.rowCount());
        for (int i = 0; i < psmTable.rowCount(); i++) {
            // peptide = psmTable.getString(i, "Stripped.Sequence");
            peptide = psmTable.getString(i, PSMConfig.stripped_peptide_sequence_column_name);
            // charge = psmTable.getString(i, "Precursor.Charge");
            charge = psmTable.getString(i, PSMConfig.precursor_charge_column_name);
            // mod_seq = psmTable.getString(i, "Modified.Sequence");
            mod_seq = psmTable.getString(i, PSMConfig.peptide_modification_column_name);
            modification = this.get_modification_diann(mod_seq, peptide);
            this.add_peptide(peptide, modification);
            mz_column.set(i, get_mz(this.get_peptide(peptide, modification).getMass(), Integer.parseInt(charge)));
        }
        psmTable.addColumns(mz_column);
        HashMap<String,ArrayList<JPeakGroup>> peptide_form2peptides = new HashMap<>();
        //psmTable = psmTable.sortOn("File.Name","Stripped.Sequence","Precursor.Charge","mz","Q.Value","PEP");
        psmTable = psmTable.sortOn(PSMConfig.ms_file_column_name,PSMConfig.stripped_peptide_sequence_column_name,PSMConfig.precursor_charge_column_name,"mz",PSMConfig.qvalue_column_name,PSMConfig.PEP_column_name);
        String peptide_form;
        // if a peptide form is overlapped with a peptide form with higher score
        IntColumn valid_column = IntColumn.create("peak_share", psmTable.rowCount());
        // HashMap<String,Integer> peptide_mz2count = new HashMap<>();
        HashMap<Integer, JPeakGroup> id2peak = new HashMap<>();
        for (int i = 0; i < psmTable.rowCount(); i++) {
            // peptide = psmTable.getString(i, "Stripped.Sequence");
            peptide = psmTable.getString(i, PSMConfig.stripped_peptide_sequence_column_name);
            // charge = psmTable.getString(i, "Precursor.Charge");
            charge = psmTable.getString(i, PSMConfig.precursor_charge_column_name);
            mz = mz_column.get(i);
            // rt_start = psmTable.row(i).getDouble("RT.Start");
            rt_start = psmTable.row(i).getDouble(PSMConfig.rt_start_column_name);
            // apex_rt = psmTable.row(i).getDouble("RT");
            apex_rt = psmTable.row(i).getDouble(PSMConfig.rt_column_name);
            // rt_end = psmTable.row(i).getDouble("RT.Stop");
            rt_end = psmTable.row(i).getDouble(PSMConfig.rt_end_column_name);
            peptide_form = peptide + "|" + charge + "|" + mz;
            if (!peptide_form2peptides.containsKey(peptide_form)) {
                peptide_form2peptides.put(peptide_form, new ArrayList<>());
                JPeakGroup peak = new JPeakGroup();
                peak.rt_start = rt_start;
                peak.apex_rt = apex_rt;
                peak.rt_end = rt_end;
                peak.id = i;
                peptide_form2peptides.get(peptide_form).add(peak);
                valid_column.set(i, 1);
                id2peak.put(i, peak);
                // peptide_mz2count.put(peptide_form,1);
            } else {
                // need to compare with each previous peptide form
                boolean keep = true;
                for (JPeakGroup p : peptide_form2peptides.get(peptide_form)) {
                    if (p.rt_start <= rt_start && rt_end <= p.rt_end) {
                        // one contained by another
                        // peptide_mz2count.put(peptide_form,peptide_mz2count.get(peptide_form)+1);
                        p.n_shared_peaks++;
                        keep = false;
                        break;
                    } else if (rt_start <= p.rt_start && p.rt_end <= rt_end) {
                        // one contained by another
                        // peptide_mz2count.put(peptide_form,peptide_mz2count.get(peptide_form)+1);
                        p.n_shared_peaks++;
                        keep = false;
                        break;
                    } else if (p.rt_start <= rt_start && rt_start <= p.rt_end) {
                        // overlap
                        double overlap = p.rt_end - rt_start;
                        double overlap_ratio = Math.max(overlap / (rt_end - rt_start), overlap/ (p.rt_end - p.rt_start));
                        if(overlap_ratio >= 0.5){
                            p.n_shared_peaks++;
                            keep = false;
                            // peptide_mz2count.put(peptide_form,peptide_mz2count.get(peptide_form)+1);
                            break;
                        }
                    } else if (p.rt_start <= rt_end && rt_end <= p.rt_end) {
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
                id2peak.put(i, peak);
                if (keep) {
                    valid_column.set(i, 1);
                } else {
                    valid_column.set(i, 0);
                    peak.n_shared_peaks++;
                }
                peptide_form2peptides.get(peptide_form).add(peak);

            }

        }

        // if a peptide form is overlapped with any peptide form.
        IntColumn overlap_column = IntColumn.create("peak_overlap", psmTable.rowCount());
        for (int i = 0; i < psmTable.rowCount(); i++) {
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

    /**
     * Read the peptide detection result and save the data to a HashMap object.
     * 
     * @param psm_file   A peptide detection result file
     * @param ms_file    An MS file
     * @param fdr_cutoff FDR cutoff used to filter peptide detection result
     * @return A HashMap<String, ArrayList<String>> which stores the peptide
     *         detection result.
     * @throws IOException If an I/O error occurs
     */
    public HashMap<String, ArrayList<String>> get_ms_file2psm(String psm_file, String ms_file, double fdr_cutoff)
            throws IOException {
        HashMap<String, Integer> hIndex = get_column_name2index(psm_file);
        BufferedReader psmReader = new BufferedReader(new FileReader(psm_file));
        psmReader.readLine();
        String line;
        String cur_ms_file = "-";

        HashMap<String, ArrayList<String>> ms_file2psm = new HashMap<>();
        int n_valid_row = 0;
        int n_total_row = 0;

        int n_peak_share = 0;

        while ((line = psmReader.readLine()) != null) {
            line = line.trim();
            n_total_row = n_total_row + 1;
            String[] d = line.split("\t");

            if (hIndex.containsKey("q_value")) {
                double q_value = Double.parseDouble(d[hIndex.get("q_value")]);
                if (q_value > fdr_cutoff) {
                    continue;
                }
            }
            if (hIndex.containsKey("decoy")) {
                String decoy = d[hIndex.get("decoy")];
                if (decoy.equalsIgnoreCase("Yes")) {
                    continue;
                }
            }

            if (hIndex.containsKey("peak_share")) {
                int peak_share = Integer.parseInt(d[hIndex.get("peak_share")]);
                if (peak_share == 0) {
                    n_peak_share++;
                    continue;
                }
            }

            if (hIndex.containsKey("ms_file")) {
                cur_ms_file = d[hIndex.get("ms_file")];
            } else {
                cur_ms_file = ms_file;
            }

            if (!ms_file2psm.containsKey(ms_file)) {
                ms_file2psm.put(cur_ms_file, new ArrayList<>());
            }
            ms_file2psm.get(cur_ms_file).add(line);
            n_valid_row = n_valid_row + 1;
        }

        psmReader.close();
        System.out.println("The number of MS files:" + ms_file2psm.size());
        System.out.println("The number of valid rows:" + n_valid_row);
        System.out.println("The number of total rows:" + n_total_row);
        if (n_peak_share >= 1) {
            System.out.println("The number of shared peaks:" + n_peak_share);
        }
        return ms_file2psm;
    }

    /**
     * Read the peptide detection result and save the data to a HashMap object.
     * This will be deleted in the future. Use
     * get_ms_file2psm_diann_multiple_ms_runs instead.
     * 
     * @param psm_file   A peptide detection result file
     * @param ms_file    An MS file
     * @param fdr_cutoff FDR cutoff used to filter peptide detection result
     * @return A HashMap<String, ArrayList<String>> which stores the peptide
     *         detection result.
     * @throws IOException If an I/O error occurs
     */
    public HashMap<String, ArrayList<String>> get_ms_file2psm_diann_will_be_deleted(String psm_file, String ms_file,
            double fdr_cutoff) throws IOException {
        HashMap<String, Integer> hIndex = get_column_name2index(psm_file);
        BufferedReader psmReader = new BufferedReader(new FileReader(psm_file));
        psmReader.readLine();
        String line;
        String cur_ms_file = "-";

        HashMap<String, ArrayList<String>> ms_file2psm = new HashMap<>();
        int n_valid_row = 0;
        int n_total_row = 0;

        while ((line = psmReader.readLine()) != null) {
            line = line.trim();
            n_total_row = n_total_row + 1;
            String[] d = line.split("\t");

            // if(hIndex.containsKey("Q.Value")){
            if (hIndex.containsKey(PSMConfig.qvalue_column_name)) {
                // double q_value = Double.parseDouble(d[hIndex.get("Q.Value")]);
                double q_value = Double.parseDouble(d[hIndex.get(PSMConfig.qvalue_column_name)]);
                if (q_value > fdr_cutoff) {
                    continue;
                }
            }

            // if(hIndex.containsKey("File.Name")){
            if (hIndex.containsKey(PSMConfig.ms_file_column_name)) {
                cur_ms_file = d[hIndex.get(PSMConfig.ms_file_column_name)];
                File F = new File(cur_ms_file);
                if (!F.exists()) {
                    // ms_file should be a folder
                    Path path = Paths.get(cur_ms_file);
                    File MS = new File(ms_file);
                    if (MS.isFile()) {
                        cur_ms_file = ms_file;
                    } else {
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
            } else {
                cur_ms_file = ms_file;
            }

            if (!ms_file2psm.containsKey(cur_ms_file)) {
                ms_file2psm.put(cur_ms_file, new ArrayList<>());
            }
            ms_file2psm.get(cur_ms_file).add(line);
            n_valid_row = n_valid_row + 1;
        }

        psmReader.close();
        System.out.println("The number of MS files:" + ms_file2psm.size());
        System.out.println("The number of valid rows:" + n_valid_row);
        System.out.println("The number of total rows:" + n_total_row);
        return ms_file2psm;
    }

    /**
     * Read the peptide detection result and save the data to a HashMap object.
     * 
     * @param psm_file   A peptide detection result file
     * @param ms_file    An MS file or a folder containing the MS files.
     * @param fdr_cutoff FDR cutoff used to filter peptide detection result
     * @return A HashMap<String, ArrayList<String>> which stores the peptide
     *         detection result.
     * @throws IOException If an I/O error occurs
     */
    public HashMap<String, ArrayList<String>> get_ms_file2psm_diann_multiple_ms_runs(String psm_file, String ms_file,
            double fdr_cutoff) throws IOException {
        HashMap<String, Integer> hIndex = get_column_name2index(psm_file);
        BufferedReader psmReader = new BufferedReader(new FileReader(psm_file));
        psmReader.readLine();
        String line;
        String cur_ms_file = "-";

        HashMap<String, ArrayList<String>> ms_file2psm = new HashMap<>();
        int n_valid_row = 0;
        int n_total_row = 0;

        while ((line = psmReader.readLine()) != null) {
            line = line.trim();
            n_total_row = n_total_row + 1;
            String[] d = line.split("\t");

            // if(hIndex.containsKey("Q.Value")){
            if (hIndex.containsKey(PSMConfig.qvalue_column_name)) {
                // double q_value = Double.parseDouble(d[hIndex.get("Q.Value")]);
                double q_value = Double.parseDouble(d[hIndex.get(PSMConfig.qvalue_column_name)]);
                if (q_value > fdr_cutoff) {
                    continue;
                }
            }

            // if(hIndex.containsKey("File.Name")){
            if (hIndex.containsKey(PSMConfig.ms_file_column_name)) {
                cur_ms_file = d[hIndex.get(PSMConfig.ms_file_column_name)];
                File F = new File(cur_ms_file);
                if (!F.exists()) {
                    // ms_file should be a folder
                    Path path = Paths.get(cur_ms_file);
                    File MS = new File(ms_file);
                    if (MS.isFile()) {
                        cur_ms_file = ms_file;
                    } else {
                        // check if it is TIMS-TOF data
                        File tdf_F = new File(ms_file + File.separator + "analysis.tdf");
                        if (tdf_F.isFile()) {
                            cur_ms_file = ms_file;
                        } else {
                            // ms_file is a folder
                            cur_ms_file = ms_file + File.separator + path.getFileName().toString();
                            // check this file exists or not
                            F = new File(cur_ms_file);
                            if (!F.exists()) {
                                // try to check if the file with .mzML extension exists
                                cur_ms_file = ms_file + File.separator + path.getFileName().toString() + ".mzML";
                                F = new File(cur_ms_file);
                                if (!F.exists()) {
                                    // check if it is TIMS-TOF data (folder without .d in filename)
                                    cur_ms_file = ms_file + File.separator + path.getFileName().toString();
                                    tdf_F = new File(cur_ms_file + File.separator + "analysis.tdf");
                                    if (tdf_F.exists()) {
                                        // it is TIMS-TOF data
                                    } else {
                                        // Try appending .d extension (DIA-NN may report filename without .d)
                                        cur_ms_file = ms_file + File.separator + path.getFileName().toString() + ".d";
                                        tdf_F = new File(cur_ms_file + File.separator + "analysis.tdf");
                                        if (tdf_F.exists()) {
                                            // it is TIMS-TOF data with .d extension
                                        } else {
                                            Cloger.getInstance().logger.error("File not found:" + ms_file
                                                    + File.separator + path.getFileName().toString());
                                            System.exit(1);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                cur_ms_file = ms_file;
            }

            if (!ms_file2psm.containsKey(cur_ms_file)) {
                ms_file2psm.put(cur_ms_file, new ArrayList<>());
            }
            ms_file2psm.get(cur_ms_file).add(line);
            n_valid_row = n_valid_row + 1;
        }

        psmReader.close();
        System.out.println("The number of MS files:" + ms_file2psm.size());
        System.out.println("The number of valid rows:" + n_valid_row);
        System.out.println("The number of total rows:" + n_total_row);

        if (ms_file2psm.size() >= 2) {
            System.out.println("There are multiple MS files in the PSM file: " + psm_file);
            if (this.ms2_merge_method.equalsIgnoreCase("best")) {
                // for each precursor, only keep the best detection
                HashMap<String, Double> precursor2best_score = new HashMap<>();
                HashMap<String, String> precursor2best_match = new HashMap<>();
                HashMap<String, String> precursor2best_ms_run = new HashMap<>();
                for (String ms_run : ms_file2psm.keySet()) {
                    for (String psm_line : ms_file2psm.get(ms_run)) {
                        String[] d = psm_line.split("\t");
                        // precursor id is a combination of peptide sequence, modification and precursor
                        // charge state
                        String precursor_id = d[hIndex.get(PSMConfig.precursor_id_column_name)];
                        // PEP score, lower is better
                        double score = Double.parseDouble(d[hIndex.get(PSMConfig.PEP_column_name)]);
                        if (!precursor2best_score.containsKey(precursor_id)) {
                            precursor2best_score.put(precursor_id, score);
                            precursor2best_match.put(precursor_id, psm_line);
                            precursor2best_ms_run.put(precursor_id, ms_run);
                        } else {
                            // PEP score: lower is better
                            if (score < precursor2best_score.get(precursor_id)) {
                                precursor2best_score.put(precursor_id, score);
                                precursor2best_match.put(precursor_id, psm_line);
                                precursor2best_ms_run.put(precursor_id, ms_run);
                            }
                        }
                    }
                }
                // update ms_file2psm to only keep the best detection for each precursor
                ms_file2psm.clear();
                for (String precursor_id : precursor2best_match.keySet()) {
                    String best_match = precursor2best_match.get(precursor_id);
                    String best_ms_run = precursor2best_ms_run.get(precursor_id);
                    if (!ms_file2psm.containsKey(best_ms_run)) {
                        ms_file2psm.put(best_ms_run, new ArrayList<>());
                    }
                    ms_file2psm.get(best_ms_run).add(best_match);
                }
            } else {
                // not supported
                System.err.println("The MS2 merge method: "+this.ms2_merge_method+" is not supported for multiple MS runs in the PSM file: "+psm_file);
                System.exit(1);
            }

        }
        return ms_file2psm;
    }

    /**
     * Read the peptide detection result and save the data to a HashMap object.
     * 
     * @param psm_file   A peptide detection result file
     * @param fdr_cutoff FDR cutoff used to filter peptide detection result
     * @return A HashMap<String, ArrayList<String>> which stores the peptide
     *         detection result.
     * @throws IOException If an I/O error occurs
     */
    public HashMap<String, ArrayList<String>> get_ms_file2psm_diann_multiple_ms_runs(String psm_file, double fdr_cutoff)
            throws IOException {
        HashMap<String, Integer> hIndex = get_column_name2index(psm_file);
        BufferedReader psmReader = new BufferedReader(new FileReader(psm_file));
        psmReader.readLine();
        String line;
        String cur_ms_file = "-";

        HashMap<String, ArrayList<String>> ms_file2psm = new HashMap<>();
        int n_valid_row = 0;
        int n_total_row = 0;

        while ((line = psmReader.readLine()) != null) {
            line = line.trim();
            n_total_row = n_total_row + 1;
            String[] d = line.split("\t");

            // if(hIndex.containsKey("Q.Value")){
            if (hIndex.containsKey(PSMConfig.qvalue_column_name)) {
                // double q_value = Double.parseDouble(d[hIndex.get("Q.Value")]);
                double q_value = Double.parseDouble(d[hIndex.get(PSMConfig.qvalue_column_name)]);
                if (q_value > fdr_cutoff) {
                    continue;
                }
            }

            // if(hIndex.containsKey("File.Name")){
            if (hIndex.containsKey(PSMConfig.ms_file_column_name)) {
                cur_ms_file = d[hIndex.get(PSMConfig.ms_file_column_name)];
            }

            if (!ms_file2psm.containsKey(cur_ms_file)) {
                ms_file2psm.put(cur_ms_file, new ArrayList<>());
            }
            ms_file2psm.get(cur_ms_file).add(line);
            n_valid_row = n_valid_row + 1;
        }

        psmReader.close();
        System.out.println("The number of MS files:" + ms_file2psm.size());
        System.out.println("The number of valid rows:" + n_valid_row);
        System.out.println("The number of total rows:" + n_total_row);

        if (ms_file2psm.size() >= 2) {
            System.out.println("There are multiple MS files in the PSM file: " + psm_file);
            if (this.ms2_merge_method.equalsIgnoreCase("best")) {
                // for each precursor, only keep the best detection
                HashMap<String, Double> precursor2best_score = new HashMap<>();
                HashMap<String, String> precursor2best_match = new HashMap<>();
                HashMap<String, String> precursor2best_ms_run = new HashMap<>();
                for (String ms_run : ms_file2psm.keySet()) {
                    for (String psm_line : ms_file2psm.get(ms_run)) {
                        String[] d = psm_line.split("\t");
                        // precursor id is a combination of peptide sequence, modification and precursor
                        // charge state
                        String precursor_id = d[hIndex.get(PSMConfig.precursor_id_column_name)];
                        // PEP score, lower is better
                        double score = Double.parseDouble(d[hIndex.get(PSMConfig.PEP_column_name)]);
                        if (!precursor2best_score.containsKey(precursor_id)) {
                            precursor2best_score.put(precursor_id, score);
                            precursor2best_match.put(precursor_id, psm_line);
                            precursor2best_ms_run.put(precursor_id, ms_run);
                        } else {
                            // PEP score: lower is better
                            if (score < precursor2best_score.get(precursor_id)) {
                                precursor2best_score.put(precursor_id, score);
                                precursor2best_match.put(precursor_id, psm_line);
                                precursor2best_ms_run.put(precursor_id, ms_run);
                            }
                        }
                    }
                }
                // update ms_file2psm to only keep the best detection for each precursor
                ms_file2psm.clear();
                for (String precursor_id : precursor2best_match.keySet()) {
                    String best_match = precursor2best_match.get(precursor_id);
                    String best_ms_run = precursor2best_ms_run.get(precursor_id);
                    if (!ms_file2psm.containsKey(best_ms_run)) {
                        ms_file2psm.put(best_ms_run, new ArrayList<>());
                    }
                    ms_file2psm.get(best_ms_run).add(best_match);
                }
            } else {
                // not supported
                System.err.println("The MS2 merge method: "+this.ms2_merge_method+" is not supported for multiple MS runs in the PSM file: "+psm_file);
                System.exit(1);
            }

        }
        return ms_file2psm;
    }

    /**
     * Get the column name to index mapping from a file.
     * 
     * @param file A file in which the first line is the header line containing
     *             column names.
     * @return A HashMap<String,Integer>: key -> column name, value -> column index
     *         (0-based).
     * @throws IOException If an I/O error occurs
     */
    public HashMap<String, Integer> get_column_name2index(String file) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String head_line = reader.readLine().trim();
        this.psm_head_line = head_line;
        HashMap<String, Integer> hIndex = get_column_name2index_from_head_line(head_line);
        reader.close();
        return hIndex;
    }

    /**
     * Get the column name to index mapping from a header line.
     * 
     * @param head_line The first line of a file
     * @return A HashMap<String,Integer>: key -> column name, value -> column index
     *         (0-based).
     */
    public HashMap<String, Integer> get_column_name2index_from_head_line(String head_line) {
        this.psm_head_line = head_line;
        String[] h = head_line.split("\t");
        HashMap<String, Integer> hIndex = new HashMap<>();
        for (int i = 0; i < h.length; i++) {
            hIndex.put(h[i], i);
        }
        return hIndex;
    }

    /**
     * Get the number of rows for a file
     * 
     * @param file   The file to count the number of rows in it.
     * @param header If the file has a header line, set it to true, otherwise set it
     *               to false.
     * @return The number of rows in the file
     * @throws IOException If an I/O error occurs
     */
    public int get_n_rows(String file, boolean header) throws IOException {
        BufferedReader pReader = new BufferedReader(new FileReader(file));
        String line;
        int n = 0;
        while ((line = pReader.readLine()) != null) {
            n++;
        }
        pReader.close();
        if (header) {
            n = n - 1;
        }
        return n;
    }

    /**
     * Generate theoretical fragment ions for a peptide
     * 
     * @param peptide          A Peptide object
     * @param precursor_charge Precursor charge state
     * @return A HashMap<Integer, ArrayList<Ion>> containing the theoretical
     *         fragment ions.
     */
    public HashMap<Integer, ArrayList<Ion>> generate_theoretical_fragment_ions(Peptide peptide, int precursor_charge) {
        // generate theoretical fragment ions.
        PeptideFrag peptideFrag = new PeptideFrag();
        peptideFrag.init(precursor_charge, peptide, this.mod_ai);
        return peptideFrag.getExpectedFragIons(peptide);

    }

    /**
     * Get possible fragment ion charges based on the precursor charge.
     * 
     * @param precursorCharge Precursor charge state
     * @return A HashSet<Integer> containing possible fragment ion charges.
     */
    public HashSet<Integer> getPossibleFragmentIonCharges(int precursorCharge) {
        HashSet<Integer> charges = new HashSet<>(4);
        if (precursorCharge <= 1) {
            charges.add(precursorCharge);
        } else {
            int cur_max_fragment_ion_charge = Math.min(precursorCharge, max_fragment_ion_charge);
            if (fragment_ion_charge_less_than_precursor_charge) {
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

    /**
     * Generate a spectral library.
     * 
     * @param res_files A Map<String,HashMap<String,String>> containing prediction
     *                  data used for spectral library generation
     * @throws IOException If an I/O error occurs
     */
    public void generate_spectral_library(Map<String, HashMap<String, String>> res_files) throws IOException {
        if (this.use_parquet) {
            // need to check if the format is a skyline format first
            // if(this.export_spectral_library_format.equalsIgnoreCase("Skyline")){
            if (this.export_spectral_library_format.toLowerCase().contains("skyline")) {
                try {
                    generate_spectral_library_parquet_skyline(res_files, this.out_dir, "carafe_spectral_library.tsv");
                } catch (SQLException e) {
                    throw new RuntimeException(e);
                }
                // check if need to generate a library with a different format
                if (this.export_spectral_library_format.contains(",")) {
                    String[] lf_formats = this.export_spectral_library_format.split(",");
                    for (String f : lf_formats) {
                        if (!f.equalsIgnoreCase("Skyline")) {
                            Cloger.getInstance().logger.info("Generate a spectral library with format: " + f);
                            this.export_spectral_library_format = f;
                            this.export_spectral_library_file_format = "tsv";
                            // only generate tsv format in this case
                            generate_spectral_library_parquet(res_files, this.out_dir, "carafe_spectral_library.tsv");
                        }
                    }
                }
            } else if (this.export_spectral_library_format.equalsIgnoreCase("mzSpecLib")) {
                try {
                    generate_spectral_library_parquet_mzSpecLib(res_files, this.out_dir, "carafe_spectral_library.tsv");
                } catch (SQLException e) {
                    throw new RuntimeException(e);
                }
            } else {
                generate_spectral_library_parquet(res_files, this.out_dir, "carafe_spectral_library.tsv");
            }
        } else {
            generate_spectral_library(res_files, this.out_dir, "carafe_spectral_library.tsv");
        }
    }

    /**
     * Generate a spectral library.
     * 
     * @param res_files  A Map<String,HashMap<String,String>> containing prediction
     *                   data used for spectral library generation
     * @param out_folder Output folder
     * @param file_name  Output file name
     * @throws IOException If an I/O error occurs
     */
    public void generate_spectral_library(Map<String, HashMap<String, String>> res_files, String out_folder,
            String file_name) throws IOException {
        // String out_library_file = this.out_dir + File.separator +
        // "carafe_spectral_library.tsv";
        String out_library_file = out_folder + File.separator + file_name;
        BufferedWriter libWriter = new BufferedWriter(new FileWriter(out_library_file));
        if(this.ccs_enabled) {
            libWriter.write("ModifiedPeptide\tStrippedPeptide\tPrecursorMz\tPrecursorCharge\tTr_recalibrated\tIonMobility\tProteinID\tDecoy\tFragmentMz\tRelativeIntensity\tFragmentType\tFragmentNumber\tFragmentCharge\tFragmentLossType\n");
        }else{
            libWriter.write("ModifiedPeptide\tStrippedPeptide\tPrecursorMz\tPrecursorCharge\tTr_recalibrated\tProteinID\tDecoy\tFragmentMz\tRelativeIntensity\tFragmentType\tFragmentNumber\tFragmentCharge\tFragmentLossType\n");
        }
        DBGear dbGear = new DBGear();

        int pepID;
        String sequence;
        String mods;
        String mod_sites;
        double rt;
        String rt_str;
        for (String i : res_files.keySet()) {
            Cloger.getInstance().logger.info(i);
            String ms2_file = res_files.get(i).get("ms2");
            if (ms2_file.endsWith("parquet")) {
                // This TSV-based method doesn't support Parquet - should use generate_spectral_library_parquet instead
                // Skip with a warning (protein info is handled via dbGear.digest_protein in Parquet workflows)
                Cloger.getInstance().logger.warn("Skipping Parquet file in TSV-based library generation (use generate_spectral_library_parquet instead): " + ms2_file);
                continue;
            } else {
                if (!get_column_name2index(ms2_file).containsKey("protein")) {
                    if (CParameter.db.toLowerCase().endsWith(".fa") || CParameter.db.toLowerCase().endsWith(".fasta")) {
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
            // b_z1 b_z2 y_z1 y_z2 b_modloss_z1 b_modloss_z2 y_modloss_z1 y_modloss_z2
            // "_ms2_pred.tsv"
            // b_z1 b_z2 y_z1 y_z2 b_modloss_z1 b_modloss_z2 y_modloss_z1 y_modloss_z2
            // "_rt_pred.tsv"
            // pepID sequence mods mod_sites nAA rt_pred rt_norm_pred irt_pred

            BufferedReader ms2Reader = new BufferedReader(new FileReader(ms2_file));
            BufferedReader ms2IntensityReader = new BufferedReader(new FileReader(ms2_intensity_file));
            BufferedReader rtReader = new BufferedReader(new FileReader(rt_file));
            BufferedReader ms2mzReader = new BufferedReader(new FileReader(ms2_mz_file));

            HashMap<String,Integer> ms2_col2index = this.get_column_name2index_from_head_line(ms2Reader.readLine().trim());
            HashMap<String,Integer> ms2_intensity_col2index = this.get_column_name2index_from_head_line(ms2IntensityReader.readLine().trim());
            HashMap<String,Integer> rt_col2index = this.get_column_name2index_from_head_line(rtReader.readLine().trim());
            String [] fragment_ion_column_names = ms2mzReader.readLine().trim().split("\t");
            //HashMap<String,Integer> ms2_mz_col2index = this.get_column_name2index_from_head_line(ms2mzReader.readLine().trim());

            String[] ion_types = new String[fragment_ion_column_names.length];
            String[] mod_losses = new String[fragment_ion_column_names.length];
            int[] ion_charges = new int[fragment_ion_column_names.length];
            for (int j = 0; j < fragment_ion_column_names.length; j++) {
                if (fragment_ion_column_names[j].startsWith("b")) {
                    ion_types[j] = "b";
                } else if (fragment_ion_column_names[j].startsWith("y")) {
                    ion_types[j] = "y";
                } else {
                    System.err.println("Unknown fragment ion type:" + fragment_ion_column_names[j]);
                    System.exit(1);
                }

                if (fragment_ion_column_names[j].endsWith("_z1")) {
                    ion_charges[j] = 1;
                } else if (fragment_ion_column_names[j].endsWith("_z2")) {
                    ion_charges[j] = 2;
                } else {
                    System.err.println("Unknown fragment ion charge:" + fragment_ion_column_names[j]);
                    System.exit(1);
                }

                if (fragment_ion_column_names[j].contains("modloss")) {
                    if (this.mod_ai.equalsIgnoreCase("phosphorylation")) {
                        mod_losses[j] = "H3PO4";
                    }
                } else {
                    mod_losses[j] = "noloss";
                }
            }

            String line;
            // RT information
            HashMap<Integer, Double> pepID2rt = new HashMap<>();
            while ((line = rtReader.readLine()) != null) {
                String[] d = line.split("\t");
                pepID = Integer.parseInt(d[rt_col2index.get("pepID")]);
                if (this.rt_max > 0) {
                    rt = Double.parseDouble(d[rt_col2index.get("rt_pred")]);
                    // convert rt to normal rt value since the rt is min-max normalized rt
                    // peptideRT.rt_norm = (peptideRT.rt - this.rt_min)/(this.rt_max - this.rt_min);
                    // rt = rt * (this.rt_max - this.rt_min) + this.rt_min;
                    rt = rt * this.rt_max;
                } else {
                    rt = Double.parseDouble(d[rt_col2index.get("irt_pred")]);
                }
                pepID2rt.put(pepID, rt);
            }
            rtReader.close();

            // CCS information
            HashMap<String, Double> pepID_charge2ccs = new HashMap<>();
            if (ccs_enabled) {
                String ccs_file = res_files.get(i).get("ccs");
                BufferedReader ccsReader = new BufferedReader(new FileReader(ccs_file));
                HashMap<String,Integer> ccs_col2index = this.get_column_name2index_from_head_line(ccsReader.readLine().trim());
                while((line=ccsReader.readLine())!=null){
                    String []d = line.trim().split("\t");
                    pepID_charge2ccs.put(d[ccs_col2index.get("pepID")]+d[ccs_col2index.get("charge")], Double.parseDouble(d[ccs_col2index.get("mobility_pred")]));
                }
                ccsReader.close();
            }

            // MS intensity
            ArrayList<String> ms2_intensity_lines = new ArrayList<>();
            while ((line = ms2IntensityReader.readLine()) != null) {
                ms2_intensity_lines.add(line.trim());
            }
            ms2IntensityReader.close();

            // mz intensity
            ArrayList<String> ms2_mz_lines = new ArrayList<>();
            while ((line = ms2mzReader.readLine()) != null) {
                ms2_mz_lines.add(line.trim());
            }
            ms2mzReader.close();

            // MS2 information
            while ((line = ms2Reader.readLine()) != null) {
                String[] d = line.split("\t");
                pepID = Integer.parseInt(d[ms2_col2index.get("pepID")]);
                int frag_start_idx = Integer.parseInt(d[ms2_col2index.get("frag_start_idx")]);
                int frag_stop_idx = Integer.parseInt(d[ms2_col2index.get("frag_stop_idx")]);
                ArrayList<String> lines = get_fragment_ion_intensity(ms2_mz_lines,
                        ms2_intensity_lines,
                        fragment_ion_column_names,
                        frag_start_idx,
                        frag_stop_idx, this.lf_top_n_fragment_ions,
                        ion_types,
                        mod_losses,
                        ion_charges,
                        this.lf_frag_n_min);

                if (lines.size() < this.lf_min_n_fragment_ions) {
                    continue;
                }

                rt_str = String.format("%.2f", pepID2rt.get(pepID));
                String mod_pep;
                if(this.export_spectral_library_format.equalsIgnoreCase("DIANN") || this.export_spectral_library_format.equalsIgnoreCase("DIA-NN")){
                    mod_pep = get_modified_peptide_diann(d[ms2_col2index.get("sequence")],d[ms2_col2index.get("mods")],d[ms2_col2index.get("mod_sites")]);
                }else if(this.export_spectral_library_format.equalsIgnoreCase("EncyclopeDIA")){
                    mod_pep = get_modified_peptide_encyclopedia(d[ms2_col2index.get("sequence")],d[ms2_col2index.get("mods")],d[ms2_col2index.get("mod_sites")]);
                }else{
                    mod_pep = get_modified_peptide(d[ms2_col2index.get("sequence")],d[ms2_col2index.get("mods")],d[ms2_col2index.get("mod_sites")]);
                }
                StringBuilder pep_level_info = new StringBuilder();
                if (ccs_enabled) {
                    pep_level_info.append(mod_pep).append("\t")
                            .append(d[ms2_col2index.get("sequence")]).append("\t")
                            .append(d[ms2_col2index.get("mz")]).append("\t")
                            .append(d[ms2_col2index.get("charge")]).append("\t")
                            .append(rt_str).append("\t")
                            .append(String.format("%.4f",pepID_charge2ccs.get(d[ms2_col2index.get("pepID")]+d[ms2_col2index.get("charge")]))).append("\t")
                            .append(d[ms2_col2index.get("protein")]).append("\t")
                            .append(d[ms2_col2index.get("decoy")].startsWith("Yes") ? 1 : 0).append("\t");
                } else {
                    pep_level_info.append(mod_pep).append("\t")
                            .append(d[ms2_col2index.get("sequence")]).append("\t")
                            .append(d[ms2_col2index.get("mz")]).append("\t")
                            .append(d[ms2_col2index.get("charge")]).append("\t")
                            .append(rt_str).append("\t")
                            .append(d[ms2_col2index.get("protein")]).append("\t")
                            .append(d[ms2_col2index.get("decoy")].startsWith("Yes") ? 1 : 0).append("\t");
                }
                for (String l : lines) {
                    libWriter.write(pep_level_info + l + "\n");
                }
            }

            ms2Reader.close();
            ms2IntensityReader.close();
        }
        libWriter.close();
    }

    /**
     * Add protein information to PSM tables if the protein column is missing.
     * 
     * @param res_files A Map<String,HashMap<String,String>> containing prediction
     *                  data used for spectral library generation
     * @throws IOException If an I/O error occurs
     */
    public void add_protein_to_psm_table(Map<String, HashMap<String, String>> res_files) throws IOException {
        DBGear dbGear = new DBGear();
        for (String i : res_files.keySet()) {
            Cloger.getInstance().logger.info(i);
            String ms2_file = res_files.get(i).get("ms2");
            if (ms2_file.endsWith("parquet")) {
                // Parquet files handle protein info differently via dbGear.digest_protein in the spectral library generation
                // Skip adding protein to Parquet PSM tables since it's handled during library generation
                Cloger.getInstance().logger.info("Skipping protein annotation for Parquet file (handled during library generation): " + ms2_file);
            } else {
                if (!get_column_name2index(ms2_file).containsKey("protein")) {
                    if (CParameter.db.toLowerCase().endsWith(".fa") || CParameter.db.toLowerCase().endsWith(".fasta")) {
                        dbGear.add_protein_to_psm_table(ms2_file, CParameter.db);
                    }
                }
            }
        }
    }

    /**
     * Generate a spectral library in Parquet format.
     * 
     * @param res_files  A Map<String,HashMap<String,String>> containing prediction
     *                   data used for spectral library generation
     * @param out_folder Output folder
     * @param file_name  Output file name
     * @throws IOException If an I/O error occurs
     */
    public void generate_spectral_library_parquet(Map<String, HashMap<String, String>> res_files, String out_folder,
            String file_name) throws IOException {
        // String out_library_file = this.out_dir + File.separator +
        // "carafe_spectral_library.tsv";
        String out_library_file = out_folder + File.separator + file_name;
        BufferedWriter libWriter = null;
        boolean export_tsv = true;
        ParquetWriter<LibFragment> pWriter = null;
        if (this.export_spectral_library_file_format.equalsIgnoreCase("parquet")) {
            export_tsv = false;
            Schema schema = FileIO.getSchema4SpectralLib();
            String o_file = "";
            if (out_library_file.endsWith("tsv")) {
                o_file = out_library_file.replaceAll("tsv$", "parquet");
            } else if (out_library_file.endsWith("txt")) {
                o_file = out_library_file.replaceAll("txt$", "parquet");
            } else if (out_library_file.endsWith("csv")) {
                o_file = out_library_file.replaceAll("csv$", "parquet");
            } else {
                if (!out_library_file.endsWith("parquet")) {
                    System.err.println("The spectral library file suffix is not supported:" + out_library_file);
                    System.exit(1);
                }
            }
            // org.apache.hadoop.fs.Path path = new org.apache.hadoop.fs.Path(o_file);
            // OutputFile out_file = HadoopOutputFile.fromPath(path, new
            // org.apache.hadoop.conf.Configuration());
            LocalOutputFile localOutputFile = new LocalOutputFile(Paths.get(o_file));
            pWriter = AvroParquetWriter.<LibFragment>builder(localOutputFile)
                    .withSchema(ReflectData.AllowNull.get().getSchema(LibFragment.class))
                    .withDataModel(ReflectData.get())
                    // .withCompressionCodec(CompressionCodecName.SNAPPY)
                    .withCompressionCodec(CompressionCodecName.ZSTD)
                    .withPageSize(ParquetWriter.DEFAULT_PAGE_SIZE)
                    // .withConf(new org.apache.hadoop.conf.Configuration())
                    .withValidation(false)
                    // override when existing
                    .withWriteMode(ParquetFileWriter.Mode.OVERWRITE)
                    .withDictionaryEncoding(true)
                    .build();
        } else {
            libWriter = new BufferedWriter(new FileWriter(out_library_file));
            if (this.ccs_enabled) {
                libWriter.write("ModifiedPeptide\tStrippedPeptide\tPrecursorMz\tPrecursorCharge\tTr_recalibrated\tIonMobility\tProteinID\tDecoy\tFragmentMz\tRelativeIntensity\tFragmentType\tFragmentNumber\tFragmentCharge\tFragmentLossType\n");
            } else {
                libWriter.write("ModifiedPeptide\tStrippedPeptide\tPrecursorMz\tPrecursorCharge\tTr_recalibrated\tProteinID\tDecoy\tFragmentMz\tRelativeIntensity\tFragmentType\tFragmentNumber\tFragmentCharge\tFragmentLossType\n");
            }
        }

        boolean export_diann_format = false;
        boolean export_EncyclopeDIA_format = false;
        boolean export_generic_format = false;
        if (this.export_spectral_library_format.equalsIgnoreCase("DIANN")
                || this.export_spectral_library_format.equalsIgnoreCase("DIA-NN")) {
            export_diann_format = true;
        } else if (this.export_spectral_library_format.equalsIgnoreCase("EncyclopeDIA")) {
            export_EncyclopeDIA_format = true;
        } else {
            export_generic_format = true;
        }

        DBGear dbGear = new DBGear();
        Map<String, String> pep2pro = new HashMap<>();
        if (CParameter.db.toLowerCase().endsWith(".fa") || CParameter.db.toLowerCase().endsWith(".fasta")) {
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
        for (String i : res_files.keySet()) {
            Cloger.getInstance().logger.info(i);
            String ms2_file = res_files.get(i).get("ms2");
            if (!ms2_file.endsWith("parquet")) {
                if (!get_column_name2index(ms2_file).containsKey("protein")) {
                    if (CParameter.db.toLowerCase().endsWith(".fa") || CParameter.db.toLowerCase().endsWith(".fasta")) {
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
            // b_z1 b_z2 y_z1 y_z2 b_modloss_z1 b_modloss_z2 y_modloss_z1 y_modloss_z2
            // "_ms2_pred.tsv"
            // b_z1 b_z2 y_z1 y_z2 b_modloss_z1 b_modloss_z2 y_modloss_z1 y_modloss_z2
            // "_rt_pred.tsv"
            // pepID sequence mods mod_sites nAA rt_pred rt_norm_pred irt_pred

            // HashMap<String,Integer> ms2_col2index = FileIO.get_column_name2index_from_head_line(ms2_file);
            // HashMap<String,Integer> ms2_intensity_col2index = FileIO.get_column_name2index_from_head_line(ms2_intensity_file);
            // HashMap<String,Integer> rt_col2index = FileIO.get_column_name2index_from_head_line(rt_file);
            String [] fragment_ion_column_names = FileIO.get_column_names_from_parquet(ms2_mz_file);
            //HashMap<String,Integer> ms2_mz_col2index = this.get_column_name2index_from_head_line(ms2mzReader.readLine().trim());

            String[] ion_types = new String[fragment_ion_column_names.length];
            String[] mod_losses = new String[fragment_ion_column_names.length];
            int[] ion_charges = new int[fragment_ion_column_names.length];
            for (int j = 0; j < fragment_ion_column_names.length; j++) {
                if (fragment_ion_column_names[j].startsWith("b")) {
                    ion_types[j] = "b";
                } else if (fragment_ion_column_names[j].startsWith("y")) {
                    ion_types[j] = "y";
                } else {
                    System.err.println("Unknown fragment ion type:" + fragment_ion_column_names[j]);
                    System.exit(1);
                }

                if (fragment_ion_column_names[j].endsWith("_z1")) {
                    ion_charges[j] = 1;
                } else if (fragment_ion_column_names[j].endsWith("_z2")) {
                    ion_charges[j] = 2;
                } else {
                    System.err.println("Unknown fragment ion charge:" + fragment_ion_column_names[j]);
                    System.exit(1);
                }

                if (fragment_ion_column_names[j].contains("modloss")) {
                    if (this.mod_ai.equalsIgnoreCase("phosphorylation")) {
                        mod_losses[j] = "H3PO4";
                    }
                } else {
                    mod_losses[j] = "noloss";
                }
            }
            // RT information
            HashMap<Integer, Double> pepID2rt = FileIO.load_rt_data(rt_file, this.rt_max);
            // CCS information
            HashMap<String, Double> pepID_charge2ccs = new HashMap<>();
            if (ccs_enabled) {
                String ccs_file = res_files.get(i).get("ccs");
                pepID_charge2ccs = FileIO.load_ccs_data(ccs_file);
            }
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
                frag_start_idx = ((Long) record.get("frag_start_idx")).intValue();
                frag_stop_idx = ((Long) record.get("frag_stop_idx")).intValue();
                sequence = record.get("sequence").toString();
                mods = record.get("mods").toString();
                mod_sites = record.get("mod_sites").toString();
                mz = (double) record.get("mz");
                charge = (int) record.get("charge");
                // decoy = group.getString("decoy",0);
                if (pep2pro.containsKey(sequence)) {
                    protein = pep2pro.get(sequence);
                } else {
                    protein = "-";
                }
                libFragment.StrippedPeptide = sequence;
                libFragment.PrecursorMz = (float) mz;
                libFragment.PrecursorCharge = charge;
                libFragment.ProteinID = protein;
                libFragment.Decoy = 0;
                libFragment.Tr_recalibrated = pepID2rt.get(pepID).floatValue();
                if (ccs_enabled) {
                    String ccsKey = pepID + String.valueOf(charge);
                    libFragment.IonMobility = pepID_charge2ccs.getOrDefault(ccsKey, 0.0).floatValue();
                } else {
                    libFragment.IonMobility = 0.0f;
                }
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

                if (lines.size() < this.lf_min_n_fragment_ions) {
                    continue;
                }
                // decoy_label = decoy.startsWith("Yes")?1:0;
                decoy_label = 0;
                rt_str = String.format("%.2f", pepID2rt.get(pepID));
                String mod_pep;
                if (export_diann_format) {
                    mod_pep = get_modified_peptide_diann(sequence, mods, mod_sites);
                } else if (export_EncyclopeDIA_format) {
                    mod_pep = get_modified_peptide_encyclopedia(sequence, mods, mod_sites);
                } else {
                    mod_pep = get_modified_peptide(sequence, mods, mod_sites);
                }
                libFragment.ModifiedPeptide = mod_pep;
                for (LibFragment l : lines) {
                    if (export_tsv) {
                        StringBuilder ob = new StringBuilder();
                        ob.append(mod_pep).append("\t")
                                .append(sequence).append("\t")
                                .append(mz).append("\t")
                                .append(charge).append("\t")
                                .append(rt_str).append("\t");
                        if (ccs_enabled) {
                            String ccsKey = pepID + String.valueOf(charge);
                            ob.append(String.format("%.4f", pepID_charge2ccs.getOrDefault(ccsKey, 0.0))).append("\t");
                        }
                        ob.append(protein).append("\t")
                                .append(decoy_label).append("\t")
                                // FragmentMz	RelativeIntensity	FragmentType	FragmentNumber	FragmentCharge	FragmentLossType
                                .append(l.FragmentMz).append("\t")
                                .append(String.format("%.4f", l.RelativeIntensity)).append("\t")
                                .append(l.FragmentType).append("\t")
                                .append(l.FragmentNumber).append("\t")
                                .append(l.FragmentCharge).append("\t")
                                .append(l.FragmentLossType).append("\n");
                        libWriter.write(ob.toString());
                    } else {
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
        if (export_tsv) {
            libWriter.close();
        } else {
            pWriter.close();
        }
    }

    /**
     * Generate a spectral library in Skyline format.
     * 
     * @param res_files  A Map<String,HashMap<String,String>> containing prediction
     *                   data used for spectral library generation
     * @param out_folder Output folder
     * @param file_name  Output file name
     * @throws IOException  If an I/O error occurs
     * @throws SQLException If an SQL error occurs
     */
    public void generate_spectral_library_parquet_skyline(Map<String, HashMap<String, String>> res_files,
            String out_folder, String file_name) throws IOException, SQLException {
        // String out_library_file = this.out_dir + File.separator +
        // "carafe_spectral_library.tsv";
        String out_library_file = out_folder + File.separator + file_name;
        if (out_library_file.endsWith("tsv")) {
            out_library_file = out_library_file.replaceAll("tsv$", "blib");
        } else if (out_library_file.endsWith("txt")) {
            out_library_file = out_library_file.replaceAll("txt$", "blib");
        } else if (out_library_file.endsWith("csv")) {
            out_library_file = out_library_file.replaceAll("csv$", "blib");
        } else if (out_library_file.endsWith("parquet")) {
            out_library_file = out_library_file.replaceAll("parquet$", "blib");
        }
        Cloger.getInstance().logger.info("The spectral library file is saved to " + out_library_file);
        SkylineIO skylineIO = new SkylineIO(out_library_file);
        skylineIO.add_SpectrumSourceFiles();
        skylineIO.add_ScoreTypes();
        skylineIO.add_IonMobilityTypes();
        skylineIO.add_RefSpectraPeakAnnotations();
        skylineIO.create_RefSpectra();
        skylineIO.create_Modifications();
        skylineIO.create_RetentionTimes();
        skylineIO.pStatementRefSpectra.setNull(5, Types.CHAR);
        skylineIO.pStatementRefSpectra.setNull(6, Types.CHAR);
        skylineIO.pStatementRefSpectra.setNull(13, Types.VARCHAR);
        skylineIO.create_RefSpectraPeaks();
        skylineIO.connection.setAutoCommit(false);

        DBGear dbGear = new DBGear();
        Map<String, String> pep2pro = new HashMap<>();
        if (CParameter.db.toLowerCase().endsWith(".fa") || CParameter.db.toLowerCase().endsWith(".fasta")) {
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
        for (String i : res_files.keySet()) {
            Cloger.getInstance().logger.info(i);
            String ms2_file = res_files.get(i).get("ms2");
            if (!ms2_file.endsWith("parquet")) {
                if (!get_column_name2index(ms2_file).containsKey("protein")) {
                    if (CParameter.db.toLowerCase().endsWith(".fa") || CParameter.db.toLowerCase().endsWith(".fasta")) {
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
            // b_z1 b_z2 y_z1 y_z2 b_modloss_z1 b_modloss_z2 y_modloss_z1 y_modloss_z2
            // "_ms2_pred.tsv"
            // b_z1 b_z2 y_z1 y_z2 b_modloss_z1 b_modloss_z2 y_modloss_z1 y_modloss_z2
            // "_rt_pred.tsv"
            // pepID sequence mods mod_sites nAA rt_pred rt_norm_pred irt_pred

            // HashMap<String,Integer> ms2_col2index = FileIO.get_column_name2index_from_head_line(ms2_file);
            // HashMap<String,Integer> ms2_intensity_col2index = FileIO.get_column_name2index_from_head_line(ms2_intensity_file);
            // HashMap<String,Integer> rt_col2index = FileIO.get_column_name2index_from_head_line(rt_file);
            String [] fragment_ion_column_names = FileIO.get_column_names_from_parquet(ms2_mz_file);
            //HashMap<String,Integer> ms2_mz_col2index = this.get_column_name2index_from_head_line(ms2mzReader.readLine().trim());

            String[] ion_types = new String[fragment_ion_column_names.length];
            String[] mod_losses = new String[fragment_ion_column_names.length];
            int[] ion_charges = new int[fragment_ion_column_names.length];
            for (int j = 0; j < fragment_ion_column_names.length; j++) {
                if (fragment_ion_column_names[j].startsWith("b")) {
                    ion_types[j] = "b";
                } else if (fragment_ion_column_names[j].startsWith("y")) {
                    ion_types[j] = "y";
                } else {
                    System.err.println("Unknown fragment ion type:" + fragment_ion_column_names[j]);
                    System.exit(1);
                }

                if (fragment_ion_column_names[j].endsWith("_z1")) {
                    ion_charges[j] = 1;
                } else if (fragment_ion_column_names[j].endsWith("_z2")) {
                    ion_charges[j] = 2;
                } else {
                    System.err.println("Unknown fragment ion charge:" + fragment_ion_column_names[j]);
                    System.exit(1);
                }

                if (fragment_ion_column_names[j].contains("modloss")) {
                    if (this.mod_ai.equalsIgnoreCase("phosphorylation")) {
                        mod_losses[j] = "H3PO4";
                    }
                } else {
                    mod_losses[j] = "noloss";
                }
            }
            // RT information
            HashMap<Integer, Double> pepID2rt = FileIO.load_rt_data(rt_file, this.rt_max);
            // Ion mobility information
            HashMap<String, Double> pepIDCharge2IonMobility = new HashMap<>();
            if (ccs_enabled) {
                String ccs_file = res_files.get(i).get("ccs");
                pepIDCharge2IonMobility = FileIO.load_ccs_data(ccs_file);
            }
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
            double ionMobility;
            while ((record = reader.read()) != null) {
                // get column "pepID"
                pepID = (int) record.get("pepID");
                frag_start_idx = ((Long) record.get("frag_start_idx")).intValue();
                frag_stop_idx = ((Long) record.get("frag_stop_idx")).intValue();
                sequence = record.get("sequence").toString();
                mods = record.get("mods").toString();
                mod_sites = record.get("mod_sites").toString();
                mz = (double) record.get("mz");
                charge = (int) record.get("charge");
                // decoy = group.getString("decoy",0);
                if (pep2pro.containsKey(sequence)) {
                    pep2pro.get(sequence);
                } else {
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

                if (lines.size() < this.lf_min_n_fragment_ions) {
                    continue;
                }

                // decoy_label = decoy.startsWith("Yes")?1:0;
                decoy_label = 0;
                String mod_pep = get_modified_peptide_skyline(sequence, mods, mod_sites);
                libFragment.ModifiedPeptide = mod_pep;
                RefSpectraID++;
                i_batch++;
                try {
                    skylineIO.pStatementRefSpectra.setString(1, sequence); // peptideSeq VARCHAR(150)
                    skylineIO.pStatementRefSpectra.setDouble(2, mz); // precursorMZ REAL
                    skylineIO.pStatementRefSpectra.setInt(3, charge); // precursorCharge INTEGER
                    skylineIO.pStatementRefSpectra.setString(4, mod_pep); // peptideModSeq VARCHAR(200)
                    // skylineIO.pStatementRefSpectra.setString(5, ""); // prevAA CHAR(1)
                    // skylineIO.pStatementRefSpectra.setString(6, ""); // nextAA CHAR(1)
                    skylineIO.pStatementRefSpectra.setInt(7, lines.size()); // numPeaks INTEGER
                    if (ccs_enabled) {
                        ionMobility = pepIDCharge2IonMobility.get(pepID + String.valueOf(charge));
                        skylineIO.pStatementRefSpectra.setDouble(8, ionMobility); // ionMobility REAL
                        skylineIO.pStatementRefSpectra.setNull(9, Types.DOUBLE); // collisionalCrossSectionSqA REAL
                        skylineIO.pStatementRefSpectra.setNull(10, Types.DOUBLE); // ionMobilityHighEnergyOffset REAL
                        skylineIO.pStatementRefSpectra.setInt(11, SkylineIO.ION_MOBILITY_TYPE_INVERSE_K0); // ionMobilityType TINYINT
                    } else {
                        skylineIO.pStatementRefSpectra.setNull(8, Types.DOUBLE); // ionMobility REAL
                        skylineIO.pStatementRefSpectra.setNull(9, Types.DOUBLE); // collisionalCrossSectionSqA REAL
                        skylineIO.pStatementRefSpectra.setNull(10, Types.DOUBLE); // ionMobilityHighEnergyOffset REAL
                        skylineIO.pStatementRefSpectra.setInt(11, SkylineIO.ION_MOBILITY_TYPE_NONE); // ionMobilityType TINYINT
                    }
                    skylineIO.pStatementRefSpectra.setDouble(12, pepID2rt.get(pepID)); // retentionTime REAL
                    // skylineIO.pStatementRefSpectra.setInt(14, 1); // fileID INTEGER DEFAULT 1
                    // skylineIO.pStatementRefSpectra.setString(15, null); // SpecIDinFile VARCHAR(256)
                    // skylineIO.pStatementRefSpectra.setDouble(16, 0); // score REAL DEFAULT 0
                    // skylineIO.pStatementRefSpectra.setInt(17, 0); // scoreType TINYINT DEFAULT 0
                    skylineIO.pStatementRefSpectra.addBatch();
                } catch (SQLException e) {
                    System.out.println("Error inserting into RefSpectra: " + e.getMessage());
                }

                // save the mz and intensity values to two arrays
                double[] mz_values = new double[lines.size()];
                float[] intensity_values = new float[lines.size()];
                for (int k = 0; k < mz_values.length; k++) {
                    mz_values[k] = lines.get(k).FragmentMz;
                    intensity_values[k] = lines.get(k).RelativeIntensity;
                }
                skylineIO.pStatementRefSpectraPeaks.setInt(1, RefSpectraID); // RefSpectraID INTEGER
                skylineIO.pStatementRefSpectraPeaks.setBytes(2, SkylineIO.doublesToLittleEndianBytes(mz_values)); // mz BLOB
                skylineIO.pStatementRefSpectraPeaks.setBytes(3, SkylineIO.floatsToLittleEndianBytes(intensity_values)); // intensity BLOB
                skylineIO.pStatementRefSpectraPeaks.addBatch();

                // RT
                skylineIO.pStatementRetentionTimes.setInt(1, RefSpectraID); // RefSpectraID INTEGER
                if (ccs_enabled) {
                    ionMobility = pepIDCharge2IonMobility.get(pepID + String.valueOf(charge));
                    skylineIO.pStatementRetentionTimes.setDouble(2, ionMobility); // ionMobility DOUBLE
                    skylineIO.pStatementRetentionTimes.setNull(3, Types.DOUBLE); // collisionalCrossSectionSqA DOUBLE
                    skylineIO.pStatementRetentionTimes.setNull(4, Types.DOUBLE); // ionMobilityHighEnergyOffset DOUBLE
                    skylineIO.pStatementRetentionTimes.setInt(5, SkylineIO.ION_MOBILITY_TYPE_INVERSE_K0); // ionMobilityType INTEGER
                } else {
                    skylineIO.pStatementRetentionTimes.setNull(2, Types.DOUBLE); // ionMobility DOUBLE
                    skylineIO.pStatementRetentionTimes.setNull(3, Types.DOUBLE); // collisionalCrossSectionSqA DOUBLE
                    skylineIO.pStatementRetentionTimes.setNull(4, Types.DOUBLE); // ionMobilityHighEnergyOffset DOUBLE
                    skylineIO.pStatementRetentionTimes.setInt(5, SkylineIO.ION_MOBILITY_TYPE_NONE); // ionMobilityType INTEGER
                }
                skylineIO.pStatementRetentionTimes.setDouble(6, pepID2rt.get(pepID)); // retentionTime DOUBLE
                skylineIO.pStatementRetentionTimes.addBatch();

                // modification
                if (!mods.isEmpty()) {
                    String[] names = mods.split(";");
                    String[] pos = mod_sites.split(";");
                    for (int j = 0; j < pos.length; j++) {
                        skylineIO.pStatementModifications.setInt(1, RefSpectraID); // RefSpectraID INTEGER
                        skylineIO.pStatementModifications.setInt(2, Integer.parseInt(pos[j])); // position INTEGER
                        skylineIO.pStatementModifications.setDouble(3, CModification.getInstance().get_mod_mass_by_psi_name_site(names[j])); // mass REAL
                        skylineIO.pStatementModifications.addBatch();
                    }
                }
                if (i_batch == 10000) {
                    skylineIO.pStatementRefSpectra.executeBatch();
                    skylineIO.pStatementRefSpectraPeaks.executeBatch();
                    skylineIO.pStatementModifications.executeBatch();
                    skylineIO.pStatementRetentionTimes.executeBatch();
                    i_batch = 0;
                }
            }
            reader.close();
        }
        if (i_batch >= 1) {
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

    /**
     * Generate a spectral library in mzSpecLib format.
     * 
     * @param res_files  A Map<String,HashMap<String,String>> containing prediction
     *                   data used for spectral library generation
     * @param out_folder Output folder
     * @param file_name  Output file name
     * @throws IOException  If an I/O error occurs
     * @throws SQLException If an SQL error occurs
     */
    public void generate_spectral_library_parquet_mzSpecLib(Map<String, HashMap<String, String>> res_files,
            String out_folder, String file_name) throws IOException, SQLException {
        // String out_library_file = this.out_dir + File.separator +
        // "carafe_spectral_library.tsv";
        String out_library_file = out_folder + File.separator + file_name;
        if (out_library_file.endsWith("tsv")) {
            out_library_file = out_library_file.replaceAll("tsv$", "mzlib.txt");
        } else if (out_library_file.endsWith("txt")) {
            out_library_file = out_library_file.replaceAll("txt$", "mzlib.txt");
        } else if (out_library_file.endsWith("csv")) {
            out_library_file = out_library_file.replaceAll("csv$", "mzlib.txt");
        } else if (out_library_file.endsWith("parquet")) {
            out_library_file = out_library_file.replaceAll("parquet$", "mzlib.txt");
        }

        BufferedWriter mzLibWriter = new BufferedWriter(new FileWriter(out_library_file));
        mzLibWriter.write("<mzSpecLib 1.0>\n");
        mzLibWriter.write("MS:1003186|library format version=1.0\n");
        File F = new File(out_library_file);
        mzLibWriter.write("MS:1003188|library name=" + F.getName() + "\n");
        mzLibWriter.write("MS:1003207|library creation software=Carafe\n");
        mzLibWriter.write("<AttributeSet Spectrum=all>\n");
        mzLibWriter.write("<AttributeSet Analyte=all>\n");
        mzLibWriter.write("<AttributeSet Interpretation=all>\n");

        DBGear dbGear = new DBGear();
        Map<String, String> pep2pro = new HashMap<>();
        if (CParameter.db.toLowerCase().endsWith(".fa") || CParameter.db.toLowerCase().endsWith(".fasta")) {
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
        for (String i : res_files.keySet()) {
            Cloger.getInstance().logger.info(i);
            String ms2_file = res_files.get(i).get("ms2");
            if (!ms2_file.endsWith("parquet")) {
                if (!get_column_name2index(ms2_file).containsKey("protein")) {
                    if (CParameter.db.toLowerCase().endsWith(".fa") || CParameter.db.toLowerCase().endsWith(".fasta")) {
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
            // b_z1 b_z2 y_z1 y_z2 b_modloss_z1 b_modloss_z2 y_modloss_z1 y_modloss_z2
            // "_ms2_pred.tsv"
            // b_z1 b_z2 y_z1 y_z2 b_modloss_z1 b_modloss_z2 y_modloss_z1 y_modloss_z2
            // "_rt_pred.tsv"
            // pepID sequence mods mod_sites nAA rt_pred rt_norm_pred irt_pred

            // HashMap<String,Integer> ms2_col2index = FileIO.get_column_name2index_from_head_line(ms2_file);
            // HashMap<String,Integer> ms2_intensity_col2index = FileIO.get_column_name2index_from_head_line(ms2_intensity_file);
            // HashMap<String,Integer> rt_col2index = FileIO.get_column_name2index_from_head_line(rt_file);
            String [] fragment_ion_column_names = FileIO.get_column_names_from_parquet(ms2_mz_file);
            //HashMap<String,Integer> ms2_mz_col2index = this.get_column_name2index_from_head_line(ms2mzReader.readLine().trim());

            String[] ion_types = new String[fragment_ion_column_names.length];
            String[] mod_losses = new String[fragment_ion_column_names.length];
            int[] ion_charges = new int[fragment_ion_column_names.length];
            for (int j = 0; j < fragment_ion_column_names.length; j++) {
                if (fragment_ion_column_names[j].startsWith("b")) {
                    ion_types[j] = "b";
                } else if (fragment_ion_column_names[j].startsWith("y")) {
                    ion_types[j] = "y";
                } else {
                    System.err.println("Unknown fragment ion type:" + fragment_ion_column_names[j]);
                    System.exit(1);
                }

                if (fragment_ion_column_names[j].endsWith("_z1")) {
                    ion_charges[j] = 1;
                } else if (fragment_ion_column_names[j].endsWith("_z2")) {
                    ion_charges[j] = 2;
                } else {
                    System.err.println("Unknown fragment ion charge:" + fragment_ion_column_names[j]);
                    System.exit(1);
                }

                if (fragment_ion_column_names[j].contains("modloss")) {
                    if (this.mod_ai.equalsIgnoreCase("phosphorylation")) {
                        mod_losses[j] = "H3PO4";
                    }
                } else {
                    mod_losses[j] = "noloss";
                }
            }
            // RT information
            HashMap<Integer, Double> pepID2rt = FileIO.load_rt_data(rt_file, this.rt_max);
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
                frag_start_idx = ((Long) record.get("frag_start_idx")).intValue();
                frag_stop_idx = ((Long) record.get("frag_stop_idx")).intValue();
                sequence = record.get("sequence").toString();
                mods = record.get("mods").toString();
                mod_sites = record.get("mod_sites").toString();
                mz = (double) record.get("mz");
                charge = (int) record.get("charge");
                // decoy = group.getString("decoy",0);
                if (pep2pro.containsKey(sequence)) {
                    pep2pro.get(sequence);
                } else {
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

                if (lines.size() < this.lf_min_n_fragment_ions) {
                    continue;
                }

                // decoy_label = decoy.startsWith("Yes")?1:0;
                decoy_label = 0;
                String mod_pep = get_modified_peptide_skyline(sequence, mods, mod_sites);
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
                    if (!lines.get(k).FragmentLossType.equals("noloss")) {
                        stringBuilder.append("-").append(lines.get(k).FragmentLossType);
                    }
                    if (lines.get(k).FragmentCharge > 1) {
                        stringBuilder.append("^").append(lines.get(k).FragmentCharge);
                    }
                    stringBuilder.append("/0.0\n");
                }
                mzLibWriter.write(stringBuilder.toString() + "\n");
            }
            reader.close();
        }
        mzLibWriter.close();
    }

    /**
     * Peptide string format conversion.
     * 
     * @param peptide   A Peptide object
     * @param mods      A string containing modifications in the format
     *                  "mod1;mod2;..." (e.g., Oxidation@M:Oxidation@M).
     * @param mod_sites A string containing modification sites in the format
     *                  "site1;site2;..." (e.g., 1;2).
     * @return A formatted peptide string.
     */
    String get_modified_peptide(String peptide, String mods, String mod_sites) {
        if (mods.isEmpty()) {
            return "_" + peptide + "_";
        } else {
            String[] names = mods.split(";");
            String[] pos = mod_sites.split(";");
            String[] aa = peptide.split("");
            for (int i = 0; i < names.length; i++) {
                String ptm_name = "";
                int ptm_pos = Integer.parseInt(pos[i]) - 1;
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
                        String psi_name_site = names[i]; // "Phospho@S"
                        if (CModification.getInstance().psi_name_site2site_psi_name.containsKey(psi_name_site)) {
                            ptm_name = CModification.getInstance().psi_name_site2site_psi_name.get(psi_name_site);
                            aa[ptm_pos] = ptm_name;
                        } else {
                            System.out.println("Error: modification " + names[i] + " is not supported yet!");
                            System.exit(1);
                        }
                }
            }
            return "_" + StringUtils.join(aa, "") + "_";
        }

    }

    /**
     * Peptide string format conversion for DIA-NN library generation.
     * 
     * @param peptide   A Peptide object
     * @param mods      A string containing modifications in the format
     *                  "mod1;mod2;..." (e.g., Oxidation@M:Oxidation@M).
     * @param mod_sites A string containing modification sites in the format
     *                  "site1;site2;..." (e.g., 1;2).
     * @return A formatted peptide string.
     */
    String get_modified_peptide_diann(String peptide, String mods, String mod_sites) {
        if (mods.isEmpty()) {
            return "_" + peptide + "_";
        } else {
            String[] names = mods.split(";");
            String[] pos = mod_sites.split(";");
            String[] aa = peptide.split("");
            for (int i = 0; i < names.length; i++) {
                String ptm_name = "";
                int ptm_pos = Integer.parseInt(pos[i]) - 1;
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
                    case "Acetyl@Protein_N-term":
                        ptm_name = "[UniMod:1]";
                        aa[0] = ptm_name + aa[0];
                        break;
                    default:
                        String psi_name_site = names[i]; // "Phospho@S"
                        if (CModification.getInstance().psi_name_site2site_unimod_acc.containsKey(psi_name_site)) {
                            ptm_name = CModification.getInstance().psi_name_site2site_unimod_acc.get(psi_name_site);
                            aa[ptm_pos] = ptm_name;
                        } else {
                            System.out.println("Error: modification " + names[i] + " is not supported yet!");
                            System.exit(1);
                        }
                }
            }
            return "_" + StringUtils.join(aa, "") + "_";
        }

    }

    /**
     * Peptide string format conversion for EncyclopeDIA library generation.
     * 
     * @param peptide   A Peptide object
     * @param mods      A string containing modifications in the format
     *                  "mod1;mod2;..." (e.g., Oxidation@M:Oxidation@M).
     * @param mod_sites A string containing modification sites in the format
     *                  "site1;site2;..." (e.g., 1;2).
     * @return A formatted peptide string.
     */
    String get_modified_peptide_encyclopedia(String peptide, String mods, String mod_sites) {
        if (mods.isEmpty()) {
            return "_" + peptide + "_";
        } else {
            String[] names = mods.split(";");
            String[] pos = mod_sites.split(";");
            String[] aa = peptide.split("");
            for (int i = 0; i < names.length; i++) {
                String ptm_name = "";
                int ptm_pos = Integer.parseInt(pos[i]) - 1;
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
                        String psi_name_site = names[i]; // "Phospho@S"
                        if(CModification.getInstance().psi_name_site2encyclopedia_mod_name.containsKey(psi_name_site)){
                            ptm_name = CModification.getInstance().psi_name_site2encyclopedia_mod_name.get(psi_name_site);
                            aa[ptm_pos] = ptm_name;
                        } else {
                            System.out.println("Error: modification " + names[i] + " is not supported yet!");
                            System.exit(1);
                        }
                }
            }
            return "_" + StringUtils.join(aa, "") + "_";
        }

    }

    /**
     * Peptide string format conversion for Skyline (.blib) library generation.
     * 
     * @param peptide   A Peptide object
     * @param mods      A string containing modifications in the format
     *                  "mod1;mod2;..." (e.g., Oxidation@M:Oxidation@M).
     * @param mod_sites A string containing modification sites in the format
     *                  "site1;site2;..." (e.g., 1;2).
     * @return A formatted peptide string.
     */
    String get_modified_peptide_skyline(String peptide, String mods, String mod_sites) {
        if (mods.isEmpty()) {
            return peptide;
        } else {
            String[] names = mods.split(";");
            String[] pos = mod_sites.split(";");
            String[] aa = peptide.split("");
            for (int i = 0; i < names.length; i++) {
                String ptm_name = "";
                int ptm_pos = Integer.parseInt(pos[i]) - 1;
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
                        String psi_name_site = names[i]; // "Phospho@S"
                        if (CModification.getInstance().psi_name_site2skyline_mod_name.containsKey(psi_name_site)) {
                            ptm_name = CModification.getInstance().psi_name_site2skyline_mod_name.get(psi_name_site);
                            aa[ptm_pos] = ptm_name;
                        } else {
                            System.out.println("Error: modification " + names[i] + " is not supported yet!");
                            System.exit(1);
                        }
                }
            }
            return StringUtils.join(aa, "");
        }

    }

    /**
     * Print the parameter information to console.
     * 
     * @param cmd The java command line
     * @throws IOException If an I/O error occurs.
     */
    private void print_parameters(String cmd) throws IOException {
        String itol_unit;
        if (CParameter.itolu.equalsIgnoreCase("ppm")) {
            itol_unit = "ppm";
        } else {
            itol_unit = "Da";
        }

        System.out.println("#############################################");
        System.out.println("Parameter:");

        StringBuilder sBuilder = new StringBuilder();
        sBuilder.append("Version: ").append(CParameter.getVersion()).append("\n");
        if (!cmd.equalsIgnoreCase("-")) {
            sBuilder.append("Command line: ").append(cmd).append("\n");
        }
        sBuilder.append("CPU: ").append(CParameter.cpu).append("\n");
        if (cmd.contains(" -i ")) {
            // related to training data generation
            sBuilder.append("## Parameters related to training data generation:\n");
            sBuilder.append("FDR threshold: ").append(fdr_cutoff).append("\n");
            if (!mod_ai.equalsIgnoreCase("general")) {
                sBuilder.append("PTM site probability cutoff: ").append(ptm_site_prob_cutoff).append("\n");
                sBuilder.append("PTM site qvalue cutoff: ").append(ptm_site_qvalue_cutoff).append("\n");
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

    /**
     * Select the best NCE based on the correlation of predicted MS2 spectra and
     * observed MS2 spectra.
     * 
     * @param in_dir  The folder containing all the data required for the analysis
     * @param min_nce The minimum NCE to consider
     * @param max_nce The maximum NCE to consider
     * @return The best NCE.
     * @throws IOException If an I/O error occurs.
     */
    public int select_best_nce(String in_dir, int min_nce, int max_nce) throws IOException {
        int best_nce = 0;
        double best_cor = Double.NEGATIVE_INFINITY;
        // digest proteins and generate peptide forms
        // need to consider for both small and large databases
        long startTime = System.currentTimeMillis();

        Map<String, HashMap<String, String>> res_files = new LinkedHashMap<>();

        AIWorker.fast_mode = this.use_parquet;

        String input_pred_file = in_dir + File.separator + "psm_pdv.txt";
        String i_out_dir = this.out_dir + File.separator + "nce";
        // create the output directory
        File OD = new File(i_out_dir);
        if (!OD.exists()) {
            OD.mkdirs();
        }

        HashMap<String, Integer> hIndex = FileIO.get_column_name2index(input_pred_file);
        // add sequence column and nAA column if they are not present in the file
        if (!hIndex.containsKey("sequence") || !hIndex.containsKey("nAA")) {

            String new_input_pred_file = i_out_dir + File.separator + "psm_pdv_with_sequence_nAA.txt";
            BufferedWriter bWriter = new BufferedWriter(new FileWriter(new_input_pred_file));

            BufferedReader reader = new BufferedReader(new FileReader(input_pred_file));
            String head_line = reader.readLine().trim();
            if (!hIndex.containsKey("sequence")) {
                head_line = head_line + "\t" + "sequence";
            }
            if (!hIndex.containsKey("nAA")) {
                head_line = head_line + "\t" + "nAA";
            }
            if (!hIndex.containsKey("pepID")) {
                head_line = head_line + "\t" + "pepID";
            }
            if (!hIndex.containsKey("protein")) {
                head_line = head_line + "\t" + "protein";
            }
            if (!hIndex.containsKey("decoy")) {
                head_line = head_line + "\t" + "decoy";
            }
            bWriter.write(head_line + "\n");
            String line;
            HashMap<String, Integer> pep_mod2pepID = new HashMap<>();
            int pepID = -1;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                String[] d = line.split("\t");
                if (!hIndex.containsKey("sequence")) {
                    line = line + "\t" + d[hIndex.get("peptide")];
                }
                if (!hIndex.containsKey("nAA")) {
                    line = line + "\t" + d[hIndex.get("peptide")].length();
                }
                String pep_mod = d[hIndex.get("peptide")] + "_" + d[hIndex.get("modification")];
                if (!pep_mod2pepID.containsKey(pep_mod)) {
                    pepID++;
                    pep_mod2pepID.put(pep_mod, pepID);
                }

                if (!hIndex.containsKey("pepID")) {
                    line = line + "\t" + pep_mod2pepID.get(pep_mod);
                }
                // use peptide to fill this column if it is not present
                if (!hIndex.containsKey("protein")) {
                    line = line + "\t" + d[hIndex.get("peptide")];
                }
                if (!hIndex.containsKey("decoy")) {
                    line = line + "\t" + "No";
                }
                bWriter.write(line + "\n");
            }
            reader.close();
            bWriter.close();
            input_pred_file = new_input_pred_file;
        }

        if (this.device.toLowerCase().contains("cpu")) {
            // only use 1 cpu for now
            ExecutorService fixedThreadPool = Executors.newFixedThreadPool(1);
            Cloger.getInstance().logger.info("Number of CPU jobs " + 1);
            AIWorker.python_bin = this.python_bin;
            // perform spectrum and rt prediction.
            String mode = this.mod_ai.equalsIgnoreCase("-") ? "general" : this.mod_ai;
            for (int i_nce = min_nce; i_nce <= max_nce; i_nce++) {
                System.out.println("NCE: " + i_nce);
                // prediction
                if(this.use_user_provided_ms_instrument) {
                    fixedThreadPool.execute(new AIWorker(in_dir, input_pred_file, i_out_dir, i_nce + "", this.device, this.user_provided_ms_instrument, i_nce, this.mod_ai, this.ai_version));
                }else{
                    fixedThreadPool.execute(new AIWorker(in_dir, input_pred_file, i_out_dir, i_nce + "", this.device, this.ms_instrument, i_nce, this.mod_ai, this.ai_version));
                }
                String nce_str = String.valueOf(i_nce);
                res_files.put(nce_str, new HashMap<>());
                res_files.get(nce_str).put("ms2", i_out_dir + File.separator + nce_str + "_ms2_df.tsv");
                res_files.get(nce_str).put("ms2_mz", i_out_dir + File.separator + nce_str + "_ms2_mz_df.tsv");
                res_files.get(nce_str).put("ms2_intensity", i_out_dir + File.separator + nce_str + "_ms2_pred.tsv");
                res_files.get(nce_str).put("rt", i_out_dir + File.separator + nce_str + "_rt_pred.tsv");
            }

            fixedThreadPool.shutdown();

            try {
                fixedThreadPool.awaitTermination(Long.MAX_VALUE, TimeUnit.HOURS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            ExecutorService fixedThreadPool = Executors.newFixedThreadPool(1);
            Cloger.getInstance().logger.info("Number of GPU jobs " + 1);
            AIWorker.python_bin = this.python_bin;
            // perform spectrum and rt prediction.
            String mode = this.mod_ai.equalsIgnoreCase("-") ? "general" : this.mod_ai;
            for (int i_nce = min_nce; i_nce <= max_nce; i_nce++) {
                System.out.println("NCE: " + i_nce);
                // prediction
                if(this.use_user_provided_ms_instrument) {
                    fixedThreadPool.execute(new AIWorker(in_dir, input_pred_file, i_out_dir, i_nce + "", this.device, this.user_provided_ms_instrument, i_nce, this.mod_ai, this.ai_version));
                }else{
                    fixedThreadPool.execute(new AIWorker(in_dir, input_pred_file, i_out_dir, i_nce + "", this.device, this.ms_instrument, i_nce, this.mod_ai, this.ai_version));
                }
                String nce_str = String.valueOf(i_nce);
                res_files.put(nce_str, new HashMap<>());
                res_files.get(nce_str).put("ms2", i_out_dir + File.separator + nce_str + "_ms2_df.tsv");
                res_files.get(nce_str).put("ms2_mz", i_out_dir + File.separator + nce_str + "_ms2_mz_df.tsv");
                res_files.get(nce_str).put("ms2_intensity", i_out_dir + File.separator + nce_str + "_ms2_pred.tsv");
                res_files.get(nce_str).put("rt", i_out_dir + File.separator + nce_str + "_rt_pred.tsv");
            }

            fixedThreadPool.shutdown();

            try {
                fixedThreadPool.awaitTermination(Long.MAX_VALUE, TimeUnit.HOURS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        String mz_df_file = "";
        String rt_pred_file = "";
        for (String i_nce : res_files.keySet()) {
            // generate a hash map for just i_nce and its value
            Map<String, HashMap<String, String>> res_files_nce = new LinkedHashMap<>();
            res_files_nce.put(i_nce, res_files.get(i_nce));
            try {
                generate_spectral_library(res_files_nce, i_out_dir, i_nce + "_carafe_spectral_library.tsv");
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            mz_df_file = res_files.get(i_nce).get("ms2_mz");
            rt_pred_file = res_files.get(i_nce).get("rt");
        }

        // library for observed data
        Map<String, HashMap<String, String>> res_files_ob = new LinkedHashMap<>();
        res_files_ob.put("observed", new HashMap<>());
        res_files_ob.get("observed").put("ms2", input_pred_file);
        res_files_ob.get("observed").put("ms2_mz", mz_df_file);
        res_files_ob.get("observed").put("ms2_intensity", this.out_dir + File.separator + "fragment_intensity_df.tsv");
        res_files_ob.get("observed").put("rt", rt_pred_file);
        try {
            generate_spectral_library(res_files_ob, i_out_dir, "experiment_carafe_spectral_library.tsv");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        //
        String observed_ms2_intensity_file = this.out_dir + File.separator + "fragment_intensity_df.tsv";
        String valid_file = this.out_dir + File.separator + "fragment_intensity_valid.tsv";
        Table observed_ms2_intensity_table = read_table(observed_ms2_intensity_file);
        Table valid_table = read_table(valid_file);
        Table psmTable = read_table(input_pred_file);
        for (String i_nce : res_files.keySet()) {
            String ms2_intensity_file = res_files.get(i_nce).get("ms2_intensity");
            Table ms2_intensity_table = read_table(ms2_intensity_file);
            // add a new column to the psmTable
            String nce_col = "nce_" + i_nce;
            String cor_n_col = "cor_n_" + i_nce;
            psmTable.addColumns(DoubleColumn.create(nce_col, psmTable.rowCount()));
            psmTable.addColumns(IntColumn.create(cor_n_col, psmTable.rowCount()));
            for (int i = 0; i < psmTable.rowCount(); i++) {
                int frag_start_idx = psmTable.row(i).getInt("frag_start_idx");
                int frag_stop_idx = psmTable.row(i).getInt("frag_stop_idx");
                Table a = ms2_intensity_table.inRange(frag_start_idx, frag_stop_idx);
                Table b = observed_ms2_intensity_table.inRange(frag_start_idx, frag_stop_idx);
                Table c = valid_table.inRange(frag_start_idx, frag_stop_idx);
                ArrayList<Double> aList = new ArrayList<>();
                ArrayList<Double> bList = new ArrayList<>();
                for (int j = 0; j < a.rowCount(); j++) {
                    for (int k = 0; k < a.columnCount(); k++) {
                        if (c.row(j).getInt(k) <= 0) {
                            // valid
                            if (a.row(j).getDouble(k) > 0 || b.row(j).getDouble(k) > 0) {
                                aList.add(a.row(j).getDouble(k));
                                bList.add(b.row(j).getDouble(k));
                            }
                        }
                    }
                }
                // calculate the correlation between the predicted and observed fragment ion intensity
                if(aList.size()>=4){
                    double [] a_array = aList.stream().mapToDouble(Double::doubleValue).toArray();
                    double [] b_array = bList.stream().mapToDouble(Double::doubleValue).toArray();
                    if(StatUtils.max(a_array) == 0 || StatUtils.max(b_array) == 0) {
                        psmTable.row(i).setDouble(nce_col,Double.NaN);
                    }else{
                        double cor = calc_unweighted_spectral_entropy(a_array, b_array);
                        psmTable.row(i).setDouble(nce_col, cor);

                        if (cor > 1) {
                            System.out.println(nce_col);
                            System.out.println(psmTable.row(i).getString("peptide"));
                            System.out.println(psmTable.row(i).getInt("charge"));
                            System.out.println(psmTable.row(i).getString("modification"));
                            System.out.println(StringUtils.join(aList, ","));
                            System.out.println(StringUtils.join(bList, ","));
                            a.print();
                            b.print();
                            System.out.println(cor);
                            System.out.println();
                            psmTable.row(i).setDouble(nce_col, Double.NaN);
                        }
                    }

                    // use spearman correlation
                    // SpearmansCorrelation spc = new SpearmansCorrelation();
                    // double cor = spc.correlation(aList.stream().mapToDouble(Double::doubleValue).toArray(), bList.stream().mapToDouble(Double::doubleValue).toArray());
                    // psmTable.row(i).setDouble(nce_col, cor);
                } else {
                    psmTable.row(i).setDouble(nce_col, Double.NaN);
                }
                psmTable.row(i).setInt(cor_n_col, aList.size());
            }
            double median_cor = AggregateFunctions.percentile(psmTable.doubleColumn(nce_col), 50.0);
            if (best_cor <= median_cor) {
                best_cor = median_cor;
                best_nce = Integer.parseInt(i_nce);
            }
            System.out.println("NCE: " + i_nce + " Median correlation: " + median_cor);
        }
        System.out.println("Best NCE:" + best_nce);
        // save the data to a file
        String out_psm_file = this.out_dir + File.separator + "psm_pdv_with_correlation.txt";
        CsvWriteOptions writeOptions = CsvWriteOptions.builder(out_psm_file)
                .separator('\t')
                .header(true)
                .build();
        // Write the table to a TSV file
        psmTable.write().usingOptions(writeOptions);
        long bTime = System.currentTimeMillis();
        Cloger.getInstance().logger.info("Time used for NCE calibration:" + (bTime - startTime) / 1000 + " s.");
        // return res_files;
        return best_nce;
    }

    /**
     * Load the data from a file and save the data as a Table object.
     * 
     * @param file A file containing the data to load
     * @return A Table object.
     */
    private Table read_table(String file) {
        CsvReadOptions.Builder builder = CsvReadOptions.builder(file)
                .maxCharsPerColumn(10000000)
                .separator('\t')
                .header(true);
        CsvReadOptions options = builder.build();
        return (Table.read().usingOptions(options));
    }

    /**
     * Calculate spectral angle between two spectra.
     * 
     * @param a The first spectrum (array of intensities)
     * @param b The second spectrum (array of intensities)
     * @return Spectral angle
     */
    public static double calc_spectral_angle(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        // 1) Compute dot product and norms
        double dot = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        // 2) Compute magnitudes
        normA = Math.sqrt(normA);
        normB = Math.sqrt(normB);

        // 3) Compute cos(theta) = (a dot b) / (|a| * |b|)
        double cosTheta = dot / (normA * normB);

        // 4) Clamp possible floating-point errors
        if (cosTheta > 1.0)
            cosTheta = 1.0;
        if (cosTheta < -1.0)
            cosTheta = -1.0;

        // 5) Angle = arccos(cosTheta)
        double angle = Math.acos(cosTheta);

        // 6) Normalized spectral angle
        return 1.0 - 2.0 * angle / Math.PI;
    }

    /**
     * Calculate the unweighted spectral entropy similarity between two spectra.
     * 
     * @param a The first spectrum (array of intensities)
     * @param b The second spectrum (array of intensities)
     * @return The unweighted spectral entropy similarity
     * @throws IllegalArgumentException if the input arrays have different lengths
     *                                  or are empty
     */
    public static double calc_unweighted_spectral_entropy(double[] a, double[] b) {
        // reference: https://www.nature.com/articles/s41592-021-01331-z#Sec9
        // 1. Compute sums to normalize
        double sumA = 0.0;
        double sumB = 0.0;
        for (int i = 0; i < a.length; i++) {
            sumA += a[i];
            sumB += b[i];
        }

        // 2. Normalize each spectrum so their sums are 1
        double[] aNorm = new double[a.length];
        double[] bNorm = new double[b.length];
        for (int i = 0; i < a.length; i++) {
            aNorm[i] = a[i] / sumA;
            bNorm[i] = b[i] / sumB;
        }

        // 3. Compute individual entropies S_A and S_B
        double sA = 0.0;
        double sB = 0.0;
        for (int i = 0; i < a.length; i++) {
            if (aNorm[i] > 0.0) {
                sA -= aNorm[i] * Math.log(aNorm[i]);
            }
            if (bNorm[i] > 0.0) {
                sB -= bNorm[i] * Math.log(bNorm[i]);
            }
        }

        // 4. Create combined spectrum c = 0.5 * (aNorm + bNorm)
        double[] c = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            c[i] = 0.5 * (aNorm[i] + bNorm[i]);
        }

        // 5. Compute the combined entropy S_AB
        double sAB = 0.0;
        for (int i = 0; i < c.length; i++) {
            if (c[i] > 0.0) {
                sAB -= c[i] * Math.log(c[i]);
            }
        }

        // 6. Compute the unweighted entropy similarity
        // Similarity = 1 - (2 * S_AB - S_A - S_B) / ln(4)
        double similarity = 1.0 - (2.0 * sAB - sA - sB) / Math.log(4.0);

        return similarity;
    }

}
