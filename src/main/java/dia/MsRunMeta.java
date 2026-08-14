package main.java.dia;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONWriter;
import main.java.util.Cloger;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import umich.ms.datatypes.LCMSDataSubset;
import umich.ms.datatypes.lcmsrun.MsSoftware;
import umich.ms.datatypes.lcmsrun.OriginalFile;
import umich.ms.datatypes.scan.IScan;
import umich.ms.datatypes.scan.props.ActivationInfo;
import umich.ms.datatypes.scan.props.Instrument;
import umich.ms.datatypes.scan.props.PrecursorInfo;
import umich.ms.datatypes.scan.props.ScanType;
import umich.ms.fileio.exceptions.FileParsingException;
import umich.ms.fileio.filetypes.mzml.MZMLFile;
import umich.ms.fileio.filetypes.mzml.MZMLIndex;
import umich.ms.fileio.filetypes.mzml.MZMLIndexElement;
import umich.ms.fileio.filetypes.mzml.MZMLRunInfo;
import umich.ms.fileio.filetypes.mzml.jaxb.CVParamType;
import umich.ms.fileio.filetypes.mzml.jaxb.InstrumentConfigurationListType;
import umich.ms.fileio.filetypes.mzml.jaxb.InstrumentConfigurationType;
import umich.ms.fileio.filetypes.mzml.jaxb.MzMLType;
import umich.ms.fileio.filetypes.mzml.jaxb.ReferenceableParamGroupRefType;
import umich.ms.fileio.filetypes.mzml.jaxb.ReferenceableParamGroupType;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Run level metadata read from an mzML header plus a sample of scan headers: the MS instrument,
 * whether the run is DDA or DIA, and - for DIA - whether the isolation scheme uses fixed,
 * variable or overlapped (staggered) windows.
 *
 * <p>Nothing here decodes spectrum binary data. The instrument comes from the mzML header alone;
 * the acquisition call comes from a bounded sample of scan headers selected through the mzML
 * index, so the cost is roughly independent of the file size.
 *
 * <p>The classification steps are static and take plain data, so they can be tested without an
 * mzML file. See {@link #classify_isolation_scheme(List)} and
 * {@link #classify_acquisition(AcquisitionEvidence)}.
 */
public class MsRunMeta {

    public String ms_file = "";

    /**
     * Carafe short instrument name, e.g. "Astral". Empty when unrecognised - this matches the
     * contract of DIAMeta.get_ms_instrument, which callers test with isEmpty().
     */
    public String ms_instrument = "";

    /**
     * Raw model string from the mzML instrument cvParam, e.g. "Orbitrap Astral".
     */
    public String instrument_model = "";
    public String instrument_vendor = "";
    public String instrument_serial = "";
    public String instrument_accession = "";

    /**
     * Where the instrument was resolved from: instrumentConfiguration, refParamGroup, instruments,
     * software, sourceFile or none.
     */
    public String instrument_source = "none";

    public AcquisitionType acquisition_type = AcquisitionType.UNKNOWN;
    public IsolationScheme isolation_scheme = IsolationScheme.UNKNOWN;

    /**
     * True when adjacent windows overlap by roughly half a window (staggered / overlapped DIA).
     * Orthogonal to {@link #is_fixed_width}: staggered schemes usually also use one fixed width.
     */
    public boolean is_overlapped = false;

    /**
     * True when essentially all isolation windows share a single width.
     */
    public boolean is_fixed_width = false;

    /**
     * True when the isolation windows seen at the start, middle and end of the run differ.
     */
    public boolean has_multiple_window_sets = false;

    /**
     * Distinct isolation windows, sorted by mz_lower.
     */
    public List<IsolationWindow> isolation_windows = new ArrayList<>();

    /**
     * Number of sampled MS2 scans per isolation window id.
     */
    public HashMap<String, Integer> isolation_window2n_scan = new HashMap<>();

    public AcquisitionEvidence evidence = null;
    public SchemeMetrics scheme_metrics = null;

    /**
     * Non fatal observations worth showing the user: disagreeing signals, multiple window sets,
     * parsing problems.
     */
    public List<String> notes = new ArrayList<>();

    public int n_ms1 = 0;
    public int n_ms2 = 0;
    public double rt_min = -1;
    public double rt_max = -1;
    public double nce = -1;
    public double precursor_mz_min = -1;
    public double precursor_mz_max = -1;
    public String run_start_time = "";
    public String software = "";

    /**
     * Scan headers read per probe slice. Three slices are taken: head, middle and tail.
     */
    public int probe_scans_per_slice = 2500;

    /**
     * Read every scan header instead of sampling. Slower, but definitive.
     */
    public boolean full_scan = false;

    /**
     * Two windows whose lower and upper edges both agree to within this many m/z are the same
     * window.
     */
    public double window_merge_tol = 0.05;

    public int num_threads = Math.min(4, Runtime.getRuntime().availableProcessors());

    /**
     * How the instrument selected precursors for MS2.
     */
    public enum AcquisitionType {
        DDA("Data-dependent acquisition"),
        DIA("Data-independent acquisition"),
        PRM("Targeted MS2 on a scheduled precursor list (PRM/MRM-HR)"),
        SRM("Selected/multiple reaction monitoring"),
        MS1_ONLY("No MS2 scans present"),
        UNKNOWN("Could not be determined");

        /**
         * The description.
         */
        public final String description;

        /**
         * Constructor.
         *
         * @param description the description
         */
        AcquisitionType(String description) {
            this.description = description;
        }
    }

    /**
     * How the DIA precursor m/z range is divided into isolation windows.
     *
     * <p>OVERLAPPED is not really exclusive of FIXED: staggered schemes almost always step one
     * fixed width by half that width. This enum is the single headline answer;
     * {@link MsRunMeta#is_overlapped} and {@link MsRunMeta#is_fixed_width} carry the two
     * independent facts, and {@link SchemeMetrics} carries the numbers behind them.
     *
     * <p>Note that data already demultiplexed by msconvert has half width, non overlapping
     * windows and so is correctly reported as FIXED. Overlap is only detectable on raw
     * staggered files.
     */
    public enum IsolationScheme {
        FIXED("Uniform window widths, no significant overlap"),
        VARIABLE("Variable window widths, no significant overlap"),
        OVERLAPPED("Staggered windows, adjacent windows overlap by about half a window"),
        NOT_APPLICABLE("Not a DIA run"),
        UNKNOWN("Could not be determined");

        /**
         * The description.
         */
        public final String description;

        /**
         * Constructor.
         *
         * @param description the description
         */
        IsolationScheme(String description) {
            this.description = description;
        }
    }

    /**
     * The scan level measurements the DDA / DIA call is made from.
     *
     * @param n_ms1              number of MS1 scans in the sample
     * @param n_ms2              number of MS2 scans in the sample
     * @param n_distinct_windows number of distinct isolation windows
     * @param distinct_ratio     n_distinct_windows / n_ms2
     * @param periodicity        fraction of MS2 scans whose isolation window repeats exactly one
     *                           cycle later; about 1 for DIA and about 0 for DDA
     * @param median_width       median isolation window width in m/z
     * @param charge_frac        fraction of MS2 scans carrying a precursor charge
     * @param mono_frac          fraction of MS2 scans carrying a monoisotopic precursor m/z
     * @param coverage_fraction  union of window intervals over the full observed precursor range
     * @param median_ms2_per_ms1 median number of MS2 scans between consecutive MS1 scans
     * @param cycles_observed    n_ms2 / n_distinct_windows
     * @param any_srm_scan_type  whether any scan is typed SRM, MRM or CRM
     */
    public record AcquisitionEvidence(int n_ms1, int n_ms2, int n_distinct_windows,
                                      double distinct_ratio, double periodicity,
                                      double median_width,
                                      double charge_frac, double mono_frac,
                                      double coverage_fraction, double median_ms2_per_ms1,
                                      double cycles_observed, boolean any_srm_scan_type) {
    }

    /**
     * The window geometry measurements the isolation scheme call is made from.
     *
     * @param n_windows           number of distinct isolation windows
     * @param modal_width         the most common window width in m/z
     * @param frac_modal          fraction of windows within tolerance of modal_width
     * @param width_cv            coefficient of variation of the window widths
     * @param width_min           narrowest window
     * @param width_max           widest window
     * @param n_width_outliers    number of windows outside the modal width tolerance
     * @param median_offset_ratio median of (step to next window) / (window width)
     * @param coverage_fraction   union of window intervals over the full observed range
     * @param is_fixed_width      whether at least 90% of windows share the modal width
     * @param is_overlapped       whether adjacent windows are staggered by about half a window
     * @param scheme              the headline classification
     */
    public record SchemeMetrics(int n_windows, double modal_width, double frac_modal,
                                double width_cv, double width_min, double width_max,
                                int n_width_outliers, double median_offset_ratio,
                                double coverage_fraction, boolean is_fixed_width,
                                boolean is_overlapped, IsolationScheme scheme) {
    }

    // ------------------------------------------------------------------ instrument

    /**
     * PSI-MS accession to Carafe short instrument name.
     *
     * <p>Kept as a private copy rather than shared with DIAMeta so that class is left untouched.
     * If a new instrument is added here, add it to DIAMeta.get_ms_instrument as well.
     *
     * <p>See https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo
     */
    public static final Map<String, String> CV_ACCESSION_2_INSTRUMENT = Map.ofEntries(
            Map.entry("MS:1003029", "Eclipse"),   // Orbitrap Eclipse
            Map.entry("MS:1003028", "Exploris"),  // Orbitrap Exploris 480
            Map.entry("MS:1003378", "Astral"),    // Orbitrap Astral
            Map.entry("MS:1002416", "Fusion"),    // Orbitrap Fusion
            Map.entry("MS:1002417", "Fusion"),    // Orbitrap Fusion ETD
            Map.entry("MS:1002732", "Lumos"),     // Orbitrap Fusion Lumos
            Map.entry("MS:1001911", "QE"),        // Q Exactive
            Map.entry("MS:1002523", "QEHF"),      // Q Exactive HF
            Map.entry("MS:1002526", "QE+"),       // Exactive Plus
            Map.entry("MS:1002634", "QE+"),       // Q Exactive Plus
            Map.entry("MS:1002877", "QEHFX"),     // Q Exactive HF-X
            Map.entry("MS:1002533", "SciexTOF"),  // TripleTOF 6600
            Map.entry("MS:1000932", "SciexTOF")); // TripleTOF 5600

    /**
     * Model name patterns to Carafe short name, tried in order when the accession is unknown.
     * Order matters: "Orbitrap Fusion Lumos" also matches "Orbitrap Fusion".
     */
    private static final List<Map.Entry<Pattern, String>> MODEL_2_INSTRUMENT = List.of(
            Map.entry(Pattern.compile("(?i)orbitrap\\s+fusion\\s+lumos"), "Lumos"),
            Map.entry(Pattern.compile("(?i)orbitrap\\s+astral"), "Astral"),
            Map.entry(Pattern.compile("(?i)orbitrap\\s+eclipse"), "Eclipse"),
            Map.entry(Pattern.compile("(?i)orbitrap\\s+exploris"), "Exploris"),
            Map.entry(Pattern.compile("(?i)orbitrap\\s+fusion"), "Fusion"),
            Map.entry(Pattern.compile("(?i)q\\s*exactive\\s+hf-?x"), "QEHFX"),
            Map.entry(Pattern.compile("(?i)q\\s*exactive\\s+hf"), "QEHF"),
            Map.entry(Pattern.compile("(?i)(q\\s*exactive\\s+plus|exactive\\s+plus)"), "QE+"),
            Map.entry(Pattern.compile("(?i)q\\s*exactive"), "QE"),
            Map.entry(Pattern.compile("(?i)(tripletof|zenotof|ztof)"), "SciexTOF"),
            Map.entry(Pattern.compile("(?i)timstof"), "timsTOF"));

    /**
     * Instrument serial number.
     */
    private static final String CV_SERIAL_NUMBER = "MS:1000529";

    /**
     * Instrument customization.
     */
    private static final String CV_CUSTOMIZATION = "MS:1000032";

    // ------------------------------------------------------------------ entry points

    /**
     * Reads the run header and a bounded sample of scan headers, then fills in every result field
     * on this object. Parsing problems are recorded in {@link #notes} rather than thrown, so a
     * batch over many files is not aborted by one bad file.
     *
     * @param ms_file path to an mzML file
     * @return this, for chaining
     */
    public MsRunMeta profile(String ms_file) {
        this.ms_file = ms_file;
        if (!is_mzml(ms_file)) {
            this.notes.add("Not an mzML file; only mzML is supported.");
            Cloger.getInstance().logger.error("Only mzML is supported for run profiling: " + ms_file);
            return this;
        }
        MZMLFile source = new MZMLFile(ms_file);
        source.setNumThreadsForParsing(this.num_threads);
        try {
            MZMLRunInfo run_info = source.fetchRunInfo();
            extract_instrument(run_info);

            MZMLIndex index = source.fetchIndex();
            NavigableMap<Integer, MZMLIndexElement> idx = index.getMapByNum();
            if (idx.isEmpty()) {
                this.notes.add("mzML index is empty; no scans could be sampled.");
                return this;
            }
            classify_from_scans(sample_scan_headers(source, idx.firstKey(), idx.lastKey()));
        } catch (FileParsingException | RuntimeException e) {
            this.notes.add("Failed to read the file: " + e.getMessage());
            Cloger.getInstance().logger.error("Failed to profile " + ms_file + ": " + e.getMessage());
        } finally {
            try {
                source.releaseMemory();
            } catch (RuntimeException ignored) {
                // releasing a partially initialised source is not worth failing the run over
            }
        }
        return this;
    }

    /**
     * Whether a path looks like an mzML file.
     *
     * @param ms_file the path
     * @return {@code true} when the name ends with .mzML in any case
     */
    public static boolean is_mzml(String ms_file) {
        return ms_file != null && ms_file.toLowerCase(Locale.ROOT).endsWith(".mzml");
    }

    // ------------------------------------------------------------------ scan sampling

    /**
     * Reads scan headers from three slices of the run - head, middle and tail - using the mzML
     * index so only those byte ranges are touched. No spectrum binary is decoded. Falls back to
     * reading every scan header when the sample does not cover enough acquisition cycles.
     *
     * @param source the open mzML
     * @param lo     the first scan number in the index
     * @param hi     the last scan number in the index
     * @return the sampled scans, ordered by scan number
     * @throws FileParsingException when the underlying reader fails
     */
    private List<IScan> sample_scan_headers(MZMLFile source, int lo, int hi) throws FileParsingException {
        int n_total = hi - lo + 1;
        if (this.full_scan || 3L * this.probe_scans_per_slice >= n_total) {
            return source.parse(LCMSDataSubset.STRUCTURE_ONLY);
        }

        int mid = lo + (n_total - this.probe_scans_per_slice) / 2;
        int tail = hi - this.probe_scans_per_slice + 1;
        List<List<IScan>> slices = new ArrayList<>(3);
        TreeMap<Integer, IScan> by_num = new TreeMap<>();
        for (int start : new int[]{lo, mid, tail}) {
            List<IScan> slice = source.parse(new LCMSDataSubset(start,
                    start + this.probe_scans_per_slice - 1, Collections.<Integer>emptySet(), null));
            slices.add(slice);
            for (IScan scan : slice) {
                by_num.put(scan.getNum(), scan);
            }
        }

        List<IScan> sample = new ArrayList<>(by_num.values());
        if (!sample_is_conclusive(sample)) {
            this.notes.add("The sampled scans did not span a full acquisition cycle; read all "
                    + "scan headers instead.");
            return source.parse(LCMSDataSubset.STRUCTURE_ONLY);
        }
        // only meaningful once the run is known to be DIA - in DDA every slice has different
        // precursors by construction, so classify_from_scans decides whether to keep this
        this.has_multiple_window_sets = window_sets_differ(slices);
        return sample;
    }

    /**
     * Minimum number of scan pairs the periodicity test must be able to compare before its result
     * is trusted.
     */
    private static final int MIN_PERIODICITY_COMPARISONS = 100;

    /**
     * Whether the sample already settles the acquisition type, so that reading every scan header
     * would not change the answer.
     *
     * <p>The periodicity test needs the MS2 sequence to be longer than one cycle before it can
     * compare anything at all. Once it can, its result is conclusive in both directions: a high
     * value means a repeating schedule and a low value means precursors chosen on the fly, and
     * neither verdict moves when more of the same run is added.
     *
     * @param scans the sampled scans
     * @return {@code true} when the sample is sufficient
     */
    private boolean sample_is_conclusive(List<IScan> scans) {
        List<double[]> raw = new ArrayList<>();
        int n_ms2 = 0;
        for (IScan scan : scans) {
            if (is_ms2(scan)) {
                n_ms2++;
                double[] win = isolation_window_of(scan);
                if (win != null) {
                    raw.add(win);
                }
            }
        }
        if (n_ms2 == 0 || raw.isEmpty()) {
            return true; // MS1 only, or no isolation windows at all; more scans will not help
        }
        int n_distinct = merge_windows(raw, this.window_merge_tol).size();
        return raw.size() >= n_distinct + MIN_PERIODICITY_COMPARISONS;
    }

    /**
     * Whether the distinct isolation window sets of the slices disagree.
     *
     * @param slices the per slice scan lists
     * @return {@code true} when any two slices with MS2 scans have different window sets
     */
    private boolean window_sets_differ(List<List<IScan>> slices) {
        Set<String> reference = null;
        for (List<IScan> slice : slices) {
            List<double[]> raw = new ArrayList<>();
            for (IScan scan : slice) {
                if (is_ms2(scan)) {
                    double[] win = isolation_window_of(scan);
                    if (win != null) {
                        raw.add(win);
                    }
                }
            }
            if (raw.isEmpty()) {
                continue;
            }
            Set<String> ids = merge_windows(raw, this.window_merge_tol).stream()
                    .map(w -> w.id).collect(Collectors.toSet());
            if (reference == null) {
                reference = ids;
            } else if (!reference.equals(ids)) {
                return true;
            }
        }
        return false;
    }

    private static boolean is_ms2(IScan scan) {
        Integer level = scan.getMsLevel();
        return level != null && level == 2;
    }

    /**
     * The isolation window of an MS2 scan.
     *
     * @param scan the scan
     * @return {@code {mz_lower, mz_upper}}, or null when the scan declares no isolation window
     */
    private static double[] isolation_window_of(IScan scan) {
        PrecursorInfo precursor = scan.getPrecursor();
        if (precursor == null) {
            return null;
        }
        Double lo = precursor.getMzRangeStart();
        Double hi = precursor.getMzRangeEnd();
        if (lo == null || hi == null || hi < lo) {
            return null;
        }
        return new double[]{lo, hi};
    }

    // ------------------------------------------------------------------ window merging

    /**
     * Collapses per scan isolation windows into the distinct window set, merging windows whose
     * lower and upper edges both agree to within {@code tol}. Merged edges are averaged over their
     * members, which removes converter jitter from the reported bounds.
     *
     * <p>Clustering rather than binning on {@link IsolationWindow#generate_id} matters: that id
     * rounds to 0.1 m/z, so it is a hard bin boundary and would split a window whose edge happens
     * to straddle one.
     *
     * @param raw per scan {@code {lo, hi}} pairs, in any order
     * @param tol edge agreement tolerance in m/z
     * @return the distinct windows, sorted by mz_lower
     */
    public static List<IsolationWindow> merge_windows(List<double[]> raw, double tol) {
        List<double[]> sorted = new ArrayList<>(raw);
        sorted.sort(Comparator.comparingDouble(w -> w[0]));
        // each cluster is {sum_lo, sum_hi, n}
        List<double[]> clusters = new ArrayList<>();
        for (double[] w : sorted) {
            boolean merged = false;
            for (int i = clusters.size() - 1; i >= 0; i--) {
                double[] c = clusters.get(i);
                double lo = c[0] / c[2];
                double hi = c[1] / c[2];
                if (lo < w[0] - tol) {
                    // clusters are created in ascending lo order and we scan backwards, so every
                    // earlier cluster is further below w and cannot match either
                    break;
                }
                if (Math.abs(w[0] - lo) <= tol && Math.abs(w[1] - hi) <= tol) {
                    c[0] += w[0];
                    c[1] += w[1];
                    c[2] += 1;
                    merged = true;
                    break;
                }
            }
            if (!merged) {
                clusters.add(new double[]{w[0], w[1], 1});
            }
        }
        List<IsolationWindow> out = new ArrayList<>(clusters.size());
        for (double[] c : clusters) {
            out.add(new IsolationWindow(c[0] / c[2], c[1] / c[2]));
        }
        out.sort(Comparator.comparingDouble(w -> w.mz_lower));
        return out;
    }

    // ------------------------------------------------------------------ classification

    /**
     * Measures the sampled scans and fills in the acquisition type, isolation scheme and every
     * supporting statistic.
     *
     * @param scans the sampled scan headers
     */
    private void classify_from_scans(List<IScan> input) {
        // the periodicity test reads the window sequence in acquisition order, so make the order
        // explicit rather than relying on whatever the parser returned
        List<IScan> scans = new ArrayList<>(input);
        scans.sort(Comparator.comparingInt(IScan::getNum));
        List<double[]> raw = new ArrayList<>();
        List<Integer> ms2_per_ms1 = new ArrayList<>();
        int n_charge = 0;
        int n_mono = 0;
        int since_ms1 = -1;
        boolean any_srm = false;
        double rt_lo = Double.MAX_VALUE;
        double rt_hi = -Double.MAX_VALUE;
        double mz_lo = Double.MAX_VALUE;
        double mz_hi = -Double.MAX_VALUE;
        int n_no_window = 0;

        for (IScan scan : scans) {
            ScanType type = scan.getScanType();
            if (type == ScanType.SRM || type == ScanType.MRM || type == ScanType.CRM) {
                any_srm = true;
            }
            Double rt = scan.getRt();
            if (rt != null) {
                rt_lo = Math.min(rt_lo, rt);
                rt_hi = Math.max(rt_hi, rt);
            }
            Integer level = scan.getMsLevel();
            if (level == null) {
                continue;
            }
            if (level == 1) {
                this.n_ms1++;
                if (since_ms1 >= 0) {
                    ms2_per_ms1.add(since_ms1);
                }
                since_ms1 = 0;
            } else if (level == 2) {
                this.n_ms2++;
                if (since_ms1 >= 0) {
                    since_ms1++;
                }
                PrecursorInfo precursor = scan.getPrecursor();
                if (precursor != null) {
                    if (precursor.getCharge() != null) {
                        n_charge++;
                    }
                    if (precursor.getMzTargetMono() != null) {
                        n_mono++;
                    }
                    if (this.nce < 0) {
                        ActivationInfo activation = precursor.getActivationInfo();
                        if (activation != null && activation.getActivationEnergyHi() != null) {
                            this.nce = activation.getActivationEnergyHi();
                        }
                    }
                }
                double[] win = isolation_window_of(scan);
                if (win == null) {
                    n_no_window++;
                } else {
                    raw.add(win);
                    mz_lo = Math.min(mz_lo, win[0]);
                    mz_hi = Math.max(mz_hi, win[1]);
                }
            }
        }
        if (since_ms1 > 0) {
            ms2_per_ms1.add(since_ms1);
        }

        this.rt_min = rt_lo == Double.MAX_VALUE ? -1 : rt_lo;
        this.rt_max = rt_hi == -Double.MAX_VALUE ? -1 : rt_hi;
        this.precursor_mz_min = mz_lo == Double.MAX_VALUE ? -1 : mz_lo;
        this.precursor_mz_max = mz_hi == -Double.MAX_VALUE ? -1 : mz_hi;

        if (n_no_window > 0) {
            this.notes.add(n_no_window + " of " + this.n_ms2
                    + " MS2 scans declared no isolation window and were excluded.");
        }

        this.isolation_windows = merge_windows(raw, this.window_merge_tol);
        int[] assigned = assign_windows(raw, this.isolation_windows, this.window_merge_tol);
        count_scans_per_window(assigned);

        int n_distinct = this.isolation_windows.size();
        double distinct_ratio = this.n_ms2 == 0 ? 0 : (double) n_distinct / this.n_ms2;
        double cycles = n_distinct == 0 ? 0 : (double) this.n_ms2 / n_distinct;
        this.evidence = new AcquisitionEvidence(this.n_ms1, this.n_ms2, n_distinct, distinct_ratio,
                periodicity(assigned, n_distinct),
                median_width(this.isolation_windows),
                this.n_ms2 == 0 ? 0 : (double) n_charge / this.n_ms2,
                this.n_ms2 == 0 ? 0 : (double) n_mono / this.n_ms2,
                coverage_fraction(this.isolation_windows), median_int(ms2_per_ms1), cycles, any_srm);

        this.acquisition_type = classify_acquisition(this.evidence);
        cross_check_acquisition();

        if (this.acquisition_type == AcquisitionType.DIA) {
            if (this.has_multiple_window_sets) {
                this.notes.add("MS2 isolation windows differ between the start, middle and end of "
                        + "the run; classification used the union of all observed windows.");
            }
            this.scheme_metrics = classify_isolation_scheme(this.isolation_windows);
            this.isolation_scheme = this.scheme_metrics.scheme();
            this.is_fixed_width = this.scheme_metrics.is_fixed_width();
            this.is_overlapped = this.scheme_metrics.is_overlapped();
            double ratio = this.scheme_metrics.median_offset_ratio();
            if (ratio > 0.70 && ratio < 0.85) {
                this.notes.add(String.format(Locale.ROOT, "Adjacent windows overlap more than a "
                        + "guard band but less than a staggered scheme (offset ratio %.3f); "
                        + "reported as not overlapped.", ratio));
            }
            if (this.scheme_metrics.coverage_fraction() < 0.98) {
                this.notes.add(String.format(Locale.ROOT, "Isolation windows do not fully tile the "
                        + "precursor range (coverage %.3f); multiplexed (MSX) acquisition is not "
                        + "supported and would look like this.",
                        this.scheme_metrics.coverage_fraction()));
            }
        } else {
            this.isolation_scheme = IsolationScheme.NOT_APPLICABLE;
            // differing window sets say nothing outside DIA, so do not report it as a finding
            this.has_multiple_window_sets = false;
        }
    }

    /**
     * Maps each per scan window back to its merged window.
     *
     * <p>Indexed on the lower edge rather than scanned linearly: a DDA run can have thousands of
     * distinct windows and tens of thousands of MS2 scans, where a linear scan per scan would be
     * hundreds of millions of comparisons.
     *
     * @param raw     per scan {@code {lo, hi}} pairs in acquisition order
     * @param merged  the distinct windows, sorted by mz_lower
     * @param tol     edge agreement tolerance in m/z
     * @return the index into {@code merged} per raw window, -1 when unassignable
     */
    public static int[] assign_windows(List<double[]> raw, List<IsolationWindow> merged, double tol) {
        NavigableMap<Double, List<Integer>> by_lower = new TreeMap<>();
        for (int i = 0; i < merged.size(); i++) {
            by_lower.computeIfAbsent(merged.get(i).mz_lower, k -> new ArrayList<>()).add(i);
        }
        int[] out = new int[raw.size()];
        for (int i = 0; i < raw.size(); i++) {
            double[] w = raw.get(i);
            out[i] = -1;
            for (List<Integer> candidates : by_lower.subMap(w[0] - tol, true, w[0] + tol, true).values()) {
                for (int index : candidates) {
                    if (Math.abs(w[1] - merged.get(index).mz_upper) <= tol) {
                        out[i] = index;
                        break;
                    }
                }
                if (out[i] >= 0) {
                    break;
                }
            }
        }
        return out;
    }

    private void count_scans_per_window(int[] assigned) {
        for (int index : assigned) {
            if (index >= 0) {
                this.isolation_window2n_scan.merge(this.isolation_windows.get(index).id, 1, Integer::sum);
            }
        }
    }

    /**
     * Decides DDA against DIA from the scan level evidence.
     *
     * <p>The primary discriminator is window reuse, not window width. Width fails at both ends:
     * narrow window DIA exists (2 m/z Astral methods) and wide window DDA exists. Reuse does not
     * fail that way, because repeating one small window set is structural to what DIA is.
     *
     * @param e the measurements
     * @return the acquisition type
     */
    public static AcquisitionType classify_acquisition(AcquisitionEvidence e) {
        if (e.n_ms2() == 0) {
            return AcquisitionType.MS1_ONLY;
        }
        if (e.any_srm_scan_type()) {
            return AcquisitionType.SRM;
        }
        if (e.n_distinct_windows() == 0) {
            // MS2 present but no isolation windows declared at all - a DDA converter artefact
            return AcquisitionType.DDA;
        }
        if (e.periodicity() >= 0.70) {
            // a repeating window schedule: DIA when it tiles the precursor range, otherwise a
            // scheduled targeted method
            return e.coverage_fraction() >= 0.90 ? AcquisitionType.DIA : AcquisitionType.PRM;
        }
        return AcquisitionType.DDA;
    }

    /**
     * Fraction of MS2 scans whose isolation window is repeated exactly one cycle later, where the
     * cycle length is taken to be the number of distinct windows.
     *
     * <p>This is what actually separates DIA from DDA. DIA steps through one fixed window schedule
     * over and over, so the sequence is periodic with period equal to the window count. DDA picks
     * whatever is most intense, so its sequence has no such period.
     *
     * <p>Window reuse alone - the ratio of distinct windows to MS2 scans - cannot do this job,
     * because it is not invariant to how much of the run is read: over a long gradient DDA
     * re-selects the same precursors many times, so the ratio falls steadily as more scans are
     * sampled and eventually enters the range DIA occupies.
     *
     * @param window_seq the isolation window index per MS2 scan, in acquisition order, -1 for a
     *                   scan whose window could not be assigned
     * @param period     the cycle length to test, normally the number of distinct windows
     * @return the fraction in [0, 1], or 0 when there is not enough data to test
     */
    public static double periodicity(int[] window_seq, int period) {
        if (period <= 0 || window_seq.length <= period) {
            return 0;
        }
        int matches = 0;
        int compared = 0;
        for (int i = 0; i + period < window_seq.length; i++) {
            if (window_seq[i] < 0 || window_seq[i + period] < 0) {
                continue;
            }
            compared++;
            if (window_seq[i] == window_seq[i + period]) {
                matches++;
            }
        }
        return compared == 0 ? 0 : (double) matches / compared;
    }

    /**
     * Records a note when a secondary signal disagrees with the acquisition call. The call itself
     * is never flipped here - these signals are not reliable enough for that.
     */
    private void cross_check_acquisition() {
        if (this.evidence == null) {
            return;
        }
        double charge_frac = this.evidence.charge_frac();
        if (this.acquisition_type == AcquisitionType.DIA && charge_frac >= 0.5) {
            this.notes.add(String.format(Locale.ROOT, "Classified DIA, but %.0f%% of MS2 scans "
                    + "carry a precursor charge, which is unusual for DIA.", 100 * charge_frac));
        } else if (this.acquisition_type == AcquisitionType.DDA && charge_frac < 0.05
                && this.evidence.median_width() >= 4) {
            this.notes.add(String.format(Locale.ROOT, "Classified DDA, but almost no MS2 scan "
                    + "carries a precursor charge and the median isolation width is %.1f m/z.",
                    this.evidence.median_width()));
        }
        if (this.acquisition_type == AcquisitionType.PRM && this.evidence.n_distinct_windows() < 200
                && this.evidence.coverage_fraction() >= 0.85) {
            this.notes.add("Windows nearly tile the precursor range; this could be a dense "
                    + "scheduled PRM panel or a DIA method with gaps.");
        }
        if (this.acquisition_type == AcquisitionType.DDA && this.evidence.median_width() <= 1.0
                && this.evidence.n_distinct_windows() > 50) {
            this.notes.add("Very narrow isolation windows with many distinct values; this could be "
                    + "SRM/MRM data whose scan type was not recorded by the converter.");
        }
    }

    /**
     * Classifies a DIA isolation scheme from its distinct window list. Pure: no file access and no
     * state, so it can be unit tested directly.
     *
     * @param windows the distinct isolation windows; need not be sorted
     * @return the scheme together with every metric it was derived from
     */
    public static SchemeMetrics classify_isolation_scheme(List<IsolationWindow> windows) {
        List<IsolationWindow> sorted = new ArrayList<>(windows);
        sorted.sort(Comparator.comparingDouble(w -> w.mz_lower));
        int n = sorted.size();
        if (n == 0) {
            return new SchemeMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, false, false, IsolationScheme.UNKNOWN);
        }

        double[] widths = new double[n];
        for (int i = 0; i < n; i++) {
            widths[i] = sorted.get(i).mz_upper - sorted.get(i).mz_lower;
        }

        // modal width: the candidate width covering the most windows within tolerance
        double modal_width = widths[0];
        int best_count = 0;
        for (double candidate : widths) {
            double tol = Math.max(0.05, 0.01 * candidate);
            int count = 0;
            for (double w : widths) {
                if (Math.abs(w - candidate) <= tol) {
                    count++;
                }
            }
            if (count > best_count) {
                best_count = count;
                modal_width = candidate;
            }
        }
        double frac_modal = (double) best_count / n;
        boolean is_fixed_width = frac_modal >= 0.90;

        double sum = 0;
        double width_min = Double.MAX_VALUE;
        double width_max = -Double.MAX_VALUE;
        for (double w : widths) {
            sum += w;
            width_min = Math.min(width_min, w);
            width_max = Math.max(width_max, w);
        }
        double mean = sum / n;
        double var = 0;
        for (double w : widths) {
            var += (w - mean) * (w - mean);
        }
        double width_cv = mean == 0 ? 0 : Math.sqrt(var / n) / mean;

        // offset ratio: step to the next window over the current window width.
        // Equals 1 - overlap_fraction when widths are equal, so 0.5 means 50% staggered.
        List<Double> offsets = new ArrayList<>();
        for (int i = 0; i < n - 1; i++) {
            double w = widths[i];
            if (w <= 0) {
                continue;
            }
            double step = sorted.get(i + 1).mz_lower - sorted.get(i).mz_lower;
            if (step < 0.25 * w) {
                continue; // residual near duplicate window, not a real step
            }
            offsets.add(step / w);
        }
        double median_offset_ratio = median(offsets);
        boolean is_overlapped = n >= 4 && median_offset_ratio >= 0.30 && median_offset_ratio <= 0.70;

        IsolationScheme scheme = is_overlapped ? IsolationScheme.OVERLAPPED
                : is_fixed_width ? IsolationScheme.FIXED : IsolationScheme.VARIABLE;

        return new SchemeMetrics(n, modal_width, frac_modal, width_cv, width_min, width_max,
                n - best_count, median_offset_ratio, coverage_fraction(sorted), is_fixed_width,
                is_overlapped, scheme);
    }

    /**
     * The fraction of the observed precursor m/z range that the windows actually cover. Uses the
     * union of the intervals, so an overlapped scheme is not scored above 1.
     *
     * @param windows the isolation windows
     * @return the covered fraction, 0 when there are no windows
     */
    public static double coverage_fraction(List<IsolationWindow> windows) {
        if (windows.isEmpty()) {
            return 0;
        }
        List<IsolationWindow> sorted = new ArrayList<>(windows);
        sorted.sort(Comparator.comparingDouble(w -> w.mz_lower));
        double max_upper = -Double.MAX_VALUE;
        for (IsolationWindow w : sorted) {
            max_upper = Math.max(max_upper, w.mz_upper);
        }
        double span = max_upper - sorted.get(0).mz_lower;
        if (span <= 0) {
            return 0;
        }
        double covered = 0;
        double cur_lo = sorted.get(0).mz_lower;
        double cur_hi = sorted.get(0).mz_upper;
        for (int i = 1; i < sorted.size(); i++) {
            IsolationWindow w = sorted.get(i);
            if (w.mz_lower > cur_hi) {
                covered += cur_hi - cur_lo;
                cur_lo = w.mz_lower;
                cur_hi = w.mz_upper;
            } else {
                cur_hi = Math.max(cur_hi, w.mz_upper);
            }
        }
        covered += cur_hi - cur_lo;
        return covered / span;
    }

    private static double median_width(List<IsolationWindow> windows) {
        List<Double> widths = new ArrayList<>(windows.size());
        for (IsolationWindow w : windows) {
            widths.add(w.mz_upper - w.mz_lower);
        }
        return median(widths);
    }

    private static double median_int(List<Integer> values) {
        List<Double> d = new ArrayList<>(values.size());
        for (int v : values) {
            d.add((double) v);
        }
        return median(d);
    }

    private static double median(List<Double> values) {
        if (values.isEmpty()) {
            return 0;
        }
        List<Double> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        int n = sorted.size();
        return n % 2 == 1 ? sorted.get(n / 2) : 0.5 * (sorted.get(n / 2 - 1) + sorted.get(n / 2));
    }

    // ------------------------------------------------------------------ instrument extraction

    /**
     * Resolves the instrument from the mzML header, trying the inline instrumentConfiguration
     * cvParams first, then referenceableParamGroups, then the parsed Instrument objects.
     *
     * <p>The inline configuration is the primary source because msftbx ships a 2017 copy of the
     * PSI-MS ontology, so its parsed Instrument objects report "Model N/A" for every instrument
     * released since - including the Eclipse, the Exploris and the Astral. The raw cvParam name
     * carries the model regardless of ontology age.
     *
     * @param run_info the parsed mzML run header
     */
    private void extract_instrument(MZMLRunInfo run_info) {
        if (run_info == null) {
            return;
        }
        from_instrument_configuration(run_info);
        if (this.instrument_model.isEmpty()) {
            from_ref_param_groups(run_info);
        }
        if (this.instrument_model.isEmpty()) {
            from_parsed_instruments(run_info);
        }
        if (this.ms_instrument.isEmpty() && !this.instrument_accession.isEmpty()) {
            this.ms_instrument = CV_ACCESSION_2_INSTRUMENT.getOrDefault(this.instrument_accession, "");
        }
        if (this.ms_instrument.isEmpty() && !this.instrument_model.isEmpty()) {
            this.ms_instrument = instrument_from_model(this.instrument_model);
        }
        if (this.instrument_vendor.isEmpty()) {
            this.instrument_vendor = vendor_from_model(this.instrument_model);
        }
        collect_software_and_files(run_info);
        if (this.instrument_vendor.isEmpty()) {
            this.instrument_vendor = vendor_from_software(this.software);
            if (!this.instrument_vendor.isEmpty() && "none".equals(this.instrument_source)) {
                this.instrument_source = "software";
            }
        }
        if (run_info.getRunStartTime() != null) {
            this.run_start_time = run_info.getRunStartTime().toString();
        }
    }

    private void from_instrument_configuration(MZMLRunInfo run_info) {
        MzMLType parsed = run_info.getParsedInfo();
        if (parsed == null) {
            return;
        }
        InstrumentConfigurationListType list = parsed.getInstrumentConfigurationList();
        if (list == null || list.getInstrumentConfiguration() == null) {
            return;
        }
        String default_id = run_info.getDefaultInstrumentID();
        List<InstrumentConfigurationType> configs = new ArrayList<>(list.getInstrumentConfiguration());
        // look at the default configuration first when there is more than one
        configs.sort(Comparator.comparing(c -> !Objects.equals(c.getId(), default_id)));
        for (InstrumentConfigurationType config : configs) {
            List<CVParamType> params = new ArrayList<>();
            if (config.getCvParam() != null) {
                params.addAll(config.getCvParam());
            }
            if (config.getReferenceableParamGroupRef() != null) {
                for (ReferenceableParamGroupRefType ref : config.getReferenceableParamGroupRef()) {
                    params.addAll(resolve_ref(ref, run_info));
                }
            }
            for (CVParamType param : params) {
                String accession = param.getAccession();
                if (accession == null) {
                    continue;
                }
                if (CV_SERIAL_NUMBER.equals(accession)) {
                    if (this.instrument_serial.isEmpty() && param.getValue() != null) {
                        this.instrument_serial = param.getValue();
                    }
                } else if (!CV_CUSTOMIZATION.equals(accession) && this.instrument_model.isEmpty()) {
                    this.instrument_accession = accession;
                    this.instrument_model = param.getName() == null ? "" : param.getName();
                    this.instrument_source = "instrumentConfiguration";
                }
            }
            if (!this.instrument_model.isEmpty()) {
                return;
            }
        }
    }

    private static List<CVParamType> resolve_ref(ReferenceableParamGroupRefType ref,
                                                 MZMLRunInfo run_info) {
        Object target = ref == null ? null : ref.getRef();
        if (target == null) {
            return Collections.emptyList();
        }
        if (target instanceof ReferenceableParamGroupType group) {
            return group.getCvParam() == null ? Collections.emptyList() : group.getCvParam();
        }
        List<CVParamType> params = run_info.getRefParamGroups().get(target.toString());
        return params == null ? Collections.emptyList() : params;
    }

    private void from_ref_param_groups(MZMLRunInfo run_info) {
        for (List<CVParamType> params : run_info.getRefParamGroups().values()) {
            for (CVParamType param : params) {
                String accession = param.getAccession();
                if (accession == null) {
                    continue;
                }
                if (CV_ACCESSION_2_INSTRUMENT.containsKey(accession)) {
                    this.instrument_accession = accession;
                    this.instrument_model = param.getName() == null ? "" : param.getName();
                    this.ms_instrument = CV_ACCESSION_2_INSTRUMENT.get(accession);
                    this.instrument_source = "refParamGroup";
                    return;
                }
                if (CV_SERIAL_NUMBER.equals(accession) && this.instrument_serial.isEmpty()
                        && param.getValue() != null) {
                    this.instrument_serial = param.getValue();
                }
            }
        }
    }

    private void from_parsed_instruments(MZMLRunInfo run_info) {
        Map<String, Instrument> instruments = run_info.getInstruments();
        if (instruments == null || instruments.isEmpty()) {
            return;
        }
        Instrument instrument = run_info.getDefaultInstrument();
        if (instrument == null) {
            instrument = instruments.values().iterator().next();
        }
        // msftbx fills these with "Model N/A" / "Vendor N/A" for anything newer than its 2017
        // ontology copy, so only take values that are actually populated
        if (instrument.getModel() != null && !Instrument.UNKNOWN_MODEL.equals(instrument.getModel())) {
            this.instrument_model = instrument.getModel();
            this.instrument_source = "instruments";
        }
        if (instrument.getManufacturer() != null
                && !Instrument.UNKNOWN_MANUFACTURER.equals(instrument.getManufacturer())) {
            this.instrument_vendor = instrument.getManufacturer();
        }
        if (this.instrument_serial.isEmpty() && instrument.getSerialNumber() != null
                && !Instrument.UNKNOWN_SERIAL_NUMBER.equals(instrument.getSerialNumber())) {
            this.instrument_serial = instrument.getSerialNumber();
        }
    }

    private void collect_software_and_files(MZMLRunInfo run_info) {
        List<MsSoftware> software_list = run_info.getSoftware();
        if (software_list != null && !software_list.isEmpty()) {
            List<String> names = new ArrayList<>();
            for (MsSoftware s : software_list) {
                names.add(s.version == null || s.version.isEmpty() ? s.name : s.name + " " + s.version);
            }
            this.software = String.join(";", names);
        }
        if (this.instrument_vendor.isEmpty()) {
            List<OriginalFile> files = run_info.getOriginalFiles();
            if (files != null) {
                for (OriginalFile f : files) {
                    String vendor = vendor_from_source_file(f.name);
                    if (!vendor.isEmpty()) {
                        this.instrument_vendor = vendor;
                        if ("none".equals(this.instrument_source)) {
                            this.instrument_source = "sourceFile";
                        }
                        break;
                    }
                }
            }
        }
    }

    /**
     * Maps a raw instrument model string to a Carafe short name.
     *
     * @param model the model string, e.g. "Orbitrap Astral"
     * @return the short name, or an empty string when unrecognised
     */
    public static String instrument_from_model(String model) {
        if (model == null || model.isEmpty()) {
            return "";
        }
        for (Map.Entry<Pattern, String> entry : MODEL_2_INSTRUMENT) {
            if (entry.getKey().matcher(model).find()) {
                return entry.getValue();
            }
        }
        return "";
    }

    private static String vendor_from_model(String model) {
        if (model == null || model.isEmpty()) {
            return "";
        }
        String m = model.toLowerCase(Locale.ROOT);
        if (m.contains("orbitrap") || m.contains("exactive") || m.contains("ltq")
                || m.contains("astral")) {
            return "Thermo Fisher Scientific";
        }
        if (m.contains("tripletof") || m.contains("zenotof") || m.contains("quadrupole time-of-flight")) {
            return "SCIEX";
        }
        if (m.contains("timstof") || m.contains("maxis") || m.contains("impact")) {
            return "Bruker Daltonics";
        }
        return "";
    }

    private static String vendor_from_software(String software) {
        if (software == null || software.isEmpty()) {
            return "";
        }
        String s = software.toLowerCase(Locale.ROOT);
        if (s.contains("xcalibur") || s.contains("thermo")) {
            return "Thermo Fisher Scientific";
        }
        if (s.contains("analyst") || s.contains("sciex")) {
            return "SCIEX";
        }
        if (s.contains("compass") || s.contains("timscontrol") || s.contains("otofcontrol")) {
            return "Bruker Daltonics";
        }
        if (s.contains("masslynx")) {
            return "Waters";
        }
        if (s.contains("masshunter")) {
            return "Agilent";
        }
        return "";
    }

    private static String vendor_from_source_file(String name) {
        if (name == null) {
            return "";
        }
        String n = name.toLowerCase(Locale.ROOT);
        if (n.endsWith(".raw")) {
            return "Thermo Fisher Scientific";
        }
        if (n.endsWith(".wiff") || n.endsWith(".wiff2")) {
            return "SCIEX";
        }
        return "";
    }

    // ------------------------------------------------------------------ reporting

    /**
     * A human readable multi line summary of everything that was determined.
     *
     * @return the summary
     */
    public String summary() {
        StringBuilder sb = new StringBuilder();
        sb.append("MS file: ").append(this.ms_file).append("\n");
        sb.append("  Instrument: ").append(this.ms_instrument.isEmpty() ? "-" : this.ms_instrument)
                .append("  model=").append(blank_to_dash(this.instrument_model))
                .append("  vendor=").append(blank_to_dash(this.instrument_vendor))
                .append("  serial=").append(blank_to_dash(this.instrument_serial))
                .append("  source=").append(this.instrument_source).append("\n");
        sb.append("  Acquisition: ").append(this.acquisition_type)
                .append(" (").append(this.acquisition_type.description).append(")\n");
        if (this.acquisition_type == AcquisitionType.DIA) {
            sb.append("  Isolation scheme: ").append(this.isolation_scheme)
                    .append(" (").append(this.isolation_scheme.description).append(")\n");
            if (this.scheme_metrics != null) {
                sb.append(String.format(Locale.ROOT,
                        "    n_windows=%d  modal_width=%.4f  width=[%.4f, %.4f]  width_cv=%.4f  "
                                + "outliers=%d%n    fixed_width=%b  overlapped=%b  "
                                + "median_offset_ratio=%.3f  coverage=%.3f%n",
                        this.scheme_metrics.n_windows(), this.scheme_metrics.modal_width(),
                        this.scheme_metrics.width_min(), this.scheme_metrics.width_max(),
                        this.scheme_metrics.width_cv(), this.scheme_metrics.n_width_outliers(),
                        this.scheme_metrics.is_fixed_width(), this.scheme_metrics.is_overlapped(),
                        this.scheme_metrics.median_offset_ratio(),
                        this.scheme_metrics.coverage_fraction()));
            }
        }
        if (this.evidence != null) {
            sb.append(String.format(Locale.ROOT,
                    "  Evidence: n_ms1=%d  n_ms2=%d  n_windows=%d  periodicity=%.4f  "
                            + "distinct_ratio=%.4f%n            median_width=%.4f  "
                            + "charge_frac=%.3f  mono_frac=%.3f  coverage=%.3f  "
                            + "ms2_per_ms1=%.1f  cycles=%.1f%n",
                    this.evidence.n_ms1(), this.evidence.n_ms2(), this.evidence.n_distinct_windows(),
                    this.evidence.periodicity(), this.evidence.distinct_ratio(),
                    this.evidence.median_width(),
                    this.evidence.charge_frac(), this.evidence.mono_frac(),
                    this.evidence.coverage_fraction(), this.evidence.median_ms2_per_ms1(),
                    this.evidence.cycles_observed()));
        }
        sb.append(String.format(Locale.ROOT, "  Run: rt=[%.2f, %.2f] min  precursor m/z=[%.2f, %.2f]"
                + "  nce=%.1f%n", this.rt_min, this.rt_max, this.precursor_mz_min,
                this.precursor_mz_max, this.nce));
        for (String note : this.notes) {
            sb.append("  NOTE: ").append(note).append("\n");
        }
        return sb.toString();
    }

    /**
     * The result as a flat map, used for the JSON summary and for one row of the folder mode TSV.
     *
     * @return an ordered map of column name to value
     */
    public LinkedHashMap<String, Object> as_row() {
        LinkedHashMap<String, Object> row = new LinkedHashMap<>();
        row.put("ms_file", this.ms_file);
        row.put("file_name", new File(this.ms_file).getName());
        row.put("instrument", blank_to_dash(this.ms_instrument));
        row.put("instrument_model", blank_to_dash(this.instrument_model));
        row.put("instrument_vendor", blank_to_dash(this.instrument_vendor));
        row.put("instrument_serial", blank_to_dash(this.instrument_serial));
        row.put("instrument_source", this.instrument_source);
        row.put("acquisition_type", this.acquisition_type.name());
        row.put("isolation_scheme", this.isolation_scheme.name());
        row.put("is_fixed_width", this.is_fixed_width);
        row.put("is_overlapped", this.is_overlapped);
        row.put("has_multiple_window_sets", this.has_multiple_window_sets);
        if (this.scheme_metrics == null) {
            row.put("n_windows", this.isolation_windows.size());
            row.put("modal_width", "-");
            row.put("width_min", "-");
            row.put("width_max", "-");
            row.put("width_cv", "-");
            row.put("n_width_outliers", "-");
            row.put("median_offset_ratio", "-");
        } else {
            row.put("n_windows", this.scheme_metrics.n_windows());
            row.put("modal_width", fmt(this.scheme_metrics.modal_width()));
            row.put("width_min", fmt(this.scheme_metrics.width_min()));
            row.put("width_max", fmt(this.scheme_metrics.width_max()));
            row.put("width_cv", fmt(this.scheme_metrics.width_cv()));
            row.put("n_width_outliers", this.scheme_metrics.n_width_outliers());
            row.put("median_offset_ratio", fmt(this.scheme_metrics.median_offset_ratio()));
        }
        row.put("coverage_fraction", this.evidence == null ? "-" : fmt(this.evidence.coverage_fraction()));
        row.put("precursor_mz_min", fmt(this.precursor_mz_min));
        row.put("precursor_mz_max", fmt(this.precursor_mz_max));
        row.put("n_ms1", this.n_ms1);
        row.put("n_ms2", this.n_ms2);
        row.put("median_ms2_per_ms1", this.evidence == null ? "-" : fmt(this.evidence.median_ms2_per_ms1()));
        row.put("cycles_observed", this.evidence == null ? "-" : fmt(this.evidence.cycles_observed()));
        row.put("rt_min", fmt(this.rt_min));
        row.put("rt_max", fmt(this.rt_max));
        row.put("nce", fmt(this.nce));
        row.put("run_start_time", blank_to_dash(this.run_start_time));
        row.put("software", blank_to_dash(this.software));
        row.put("periodicity", this.evidence == null ? "-" : fmt(this.evidence.periodicity()));
        row.put("distinct_ratio", this.evidence == null ? "-" : fmt(this.evidence.distinct_ratio()));
        row.put("charge_frac", this.evidence == null ? "-" : fmt(this.evidence.charge_frac()));
        row.put("mono_frac", this.evidence == null ? "-" : fmt(this.evidence.mono_frac()));
        row.put("median_width", this.evidence == null ? "-" : fmt(this.evidence.median_width()));
        row.put("notes", this.notes.isEmpty() ? "-" : String.join("; ", this.notes));
        return row;
    }

    private static String fmt(double v) {
        return String.format(Locale.ROOT, "%.4f", v);
    }

    private static String blank_to_dash(String s) {
        return s == null || s.isEmpty() ? "-" : s;
    }

    /**
     * Writes the JSON summary for one run.
     *
     * @param out_dir the output directory
     * @return the file that was written
     * @throws IOException when the file cannot be written
     */
    public File write_summary_json(String out_dir) throws IOException {
        return write_summary_json(out_dir, strip_extension(new File(this.ms_file).getName()));
    }

    /**
     * Writes the JSON summary for one run under an explicit base name.
     *
     * @param out_dir   the output directory
     * @param base_name the output file name without the suffix
     * @return the file that was written
     * @throws IOException when the file cannot be written
     */
    public File write_summary_json(String out_dir, String base_name) throws IOException {
        File out = new File(out_dir, base_name + "_acquisition.json");
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(out))) {
            writer.write(JSON.toJSONString(as_row(), JSONWriter.Feature.PrettyFormat));
            writer.newLine();
        }
        return out;
    }

    /**
     * Writes the per window detail table. Only meaningful for a DIA run.
     *
     * @param out_dir the output directory
     * @return the file that was written
     * @throws IOException when the file cannot be written
     */
    public File write_isolation_window_tsv(String out_dir) throws IOException {
        return write_isolation_window_tsv(out_dir, strip_extension(new File(this.ms_file).getName()));
    }

    /**
     * Writes the per window detail table under an explicit base name.
     *
     * @param out_dir   the output directory
     * @param base_name the output file name without the suffix
     * @return the file that was written
     * @throws IOException when the file cannot be written
     */
    public File write_isolation_window_tsv(String out_dir, String base_name) throws IOException {
        File out = new File(out_dir, base_name + "_isolation_windows.tsv");
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(out))) {
            writer.write("window_id\tmz_lower\tmz_upper\twidth\tcenter\tn_scans\tstep_to_next\toffset_ratio");
            writer.newLine();
            for (int i = 0; i < this.isolation_windows.size(); i++) {
                IsolationWindow w = this.isolation_windows.get(i);
                double width = w.mz_upper - w.mz_lower;
                String step = "-";
                String ratio = "-";
                if (i < this.isolation_windows.size() - 1) {
                    double s = this.isolation_windows.get(i + 1).mz_lower - w.mz_lower;
                    step = fmt(s);
                    ratio = width > 0 ? fmt(s / width) : "-";
                }
                writer.write(w.id + "\t" + fmt(w.mz_lower) + "\t" + fmt(w.mz_upper) + "\t"
                        + fmt(width) + "\t" + fmt(0.5 * (w.mz_lower + w.mz_upper)) + "\t"
                        + this.isolation_window2n_scan.getOrDefault(w.id, 0) + "\t" + step + "\t"
                        + ratio);
                writer.newLine();
            }
        }
        return out;
    }

    private static String strip_extension(String name) {
        int dot = name.lastIndexOf('.');
        return dot > 0 ? name.substring(0, dot) : name;
    }

    // ------------------------------------------------------------------ CLI

    /**
     * Finds every mzML file under a folder, searching sub folders as well.
     *
     * @param dir the folder to search
     * @return the mzML files, sorted by path
     * @throws IOException when the folder cannot be walked
     */
    public static List<File> find_mzml_files(File dir) throws IOException {
        try (Stream<Path> paths = Files.walk(dir.toPath())) {
            return paths.filter(Files::isRegularFile)
                    .filter(p -> is_mzml(p.getFileName().toString()))
                    .map(Path::toFile)
                    .sorted(Comparator.comparing(File::getAbsolutePath))
                    .collect(Collectors.toList());
        }
    }

    /**
     * Writes one summary row per run into a single TSV.
     *
     * @param results  the profiled runs
     * @param out_file the file to write
     * @throws IOException when the file cannot be written
     */
    public static void write_batch_tsv(List<MsRunMeta> results, File out_file) throws IOException {
        if (results.isEmpty()) {
            return;
        }
        List<String> columns = new ArrayList<>(results.get(0).as_row().keySet());
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(out_file))) {
            writer.write(String.join("\t", columns));
            writer.newLine();
            for (MsRunMeta result : results) {
                LinkedHashMap<String, Object> row = result.as_row();
                List<String> values = new ArrayList<>(columns.size());
                for (String column : columns) {
                    Object v = row.get(column);
                    values.add(v == null ? "-" : String.valueOf(v).replace('\t', ' '));
                }
                writer.write(String.join("\t", values));
                writer.newLine();
            }
        }
    }

    /**
     * Profiles one mzML file, or every mzML file under a folder.
     *
     * <p>With a single mzML the detailed per run files are written: a JSON summary and, for a DIA
     * run, the isolation window table. With a folder a single TSV is written instead, one row per
     * run.
     *
     * @param args the command line
     * @throws ParseException when the command line cannot be parsed
     * @throws IOException    when an output file cannot be written
     */
    public static void main(String[] args) throws ParseException, IOException {
        Options options = new Options();
        options.addOption("ms", true, "A single mzML file, or a folder searched recursively for mzML files");
        options.addOption("o", true, "Output directory, default is the current directory");
        options.addOption("detail", false, "In folder mode, also write the per run detail files");
        options.addOption("full", false, "Read all scan headers instead of sampling");
        options.addOption("h", false, "Help");

        CommandLineParser parser = new DefaultParser(false);
        CommandLine cmd = parser.parse(options, args);
        if (cmd.hasOption("h") || !cmd.hasOption("ms") || args.length == 0) {
            HelpFormatter f = new HelpFormatter();
            f.setWidth(100);
            f.setOptionComparator(null);
            System.out.println("java -cp carafe.jar main.java.dia.MsRunMeta");
            f.printHelp("Options", options);
            return;
        }

        File input = new File(cmd.getOptionValue("ms"));
        if (!input.exists()) {
            Cloger.getInstance().logger.error("Input does not exist: " + input.getAbsolutePath());
            System.exit(1);
        }
        String out_dir = cmd.getOptionValue("o", "./");
        File out_dir_file = new File(out_dir);
        if (!out_dir_file.exists() && !out_dir_file.mkdirs()) {
            Cloger.getInstance().logger.error("Could not create the output directory: " + out_dir);
            System.exit(1);
        }
        boolean full = cmd.hasOption("full");
        boolean detail = cmd.hasOption("detail");

        if (input.isFile()) {
            MsRunMeta meta = new MsRunMeta();
            meta.full_scan = full;
            meta.profile(input.getAbsolutePath());
            System.out.println(meta.summary());
            File json = meta.write_summary_json(out_dir);
            Cloger.getInstance().logger.info("Summary written to " + json.getAbsolutePath());
            if (meta.acquisition_type == AcquisitionType.DIA) {
                File tsv = meta.write_isolation_window_tsv(out_dir);
                Cloger.getInstance().logger.info("Isolation windows written to " + tsv.getAbsolutePath());
            }
            return;
        }

        List<File> files = find_mzml_files(input);
        if (files.isEmpty()) {
            Cloger.getInstance().logger.error("No mzML file found under " + input.getAbsolutePath());
            return;
        }
        Cloger.getInstance().logger.info("Found " + files.size() + " mzML files under "
                + input.getAbsolutePath());
        List<MsRunMeta> results = new ArrayList<>(files.size());
        Set<String> used_names = new HashSet<>();
        for (int i = 0; i < files.size(); i++) {
            File file = files.get(i);
            MsRunMeta meta = new MsRunMeta();
            meta.full_scan = full;
            try {
                meta.profile(file.getAbsolutePath());
            } catch (RuntimeException e) {
                // profile() already traps the expected failures; this is the last resort so one
                // bad file cannot lose the results for all the others
                meta.ms_file = file.getAbsolutePath();
                meta.notes.add("Failed to read the file: " + e.getMessage());
                Cloger.getInstance().logger.warn("Skipped " + file.getAbsolutePath() + ": " + e.getMessage());
            }
            results.add(meta);
            System.out.println("[" + (i + 1) + "/" + files.size() + "] " + file.getName() + " -> "
                    + meta.acquisition_type
                    + (meta.acquisition_type == AcquisitionType.DIA ? "/" + meta.isolation_scheme : ""));
            if (detail) {
                write_detail_files(meta, out_dir, used_names);
            }
        }
        File out_file = new File(out_dir, "mzml_acquisition_summary.tsv");
        write_batch_tsv(results, out_file);
        Cloger.getInstance().logger.info("Summary of " + results.size() + " runs written to "
                + out_file.getAbsolutePath());
    }

    /**
     * Writes the per run detail files in folder mode, disambiguating runs that share a basename.
     */
    private static void write_detail_files(MsRunMeta meta, String out_dir, Set<String> used_names)
            throws IOException {
        String base = strip_extension(new File(meta.ms_file).getName());
        String name = base;
        int suffix = 1;
        while (!used_names.add(name)) {
            name = base + "_" + (++suffix);
        }
        meta.write_summary_json(out_dir, name);
        if (meta.acquisition_type == AcquisitionType.DIA) {
            meta.write_isolation_window_tsv(out_dir, name);
        }
    }
}
