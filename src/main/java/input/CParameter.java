package main.java.input;

import com.compomics.util.experiment.biology.ions.IonFactory;
import java.io.IOException;
import java.util.Properties;

public class CParameter {

    public static double fragment_ion_intensity_cutoff = 0.01;
    /**
     * RT window in minute used to filter peptide candidates
     */
    public static double rt_win = 3.0;
    public static boolean clip_nTerm_M = true;
    public static String db = "";
    public static String cmd = "-";
    public static String decoy_level = "protein";
    public static String decoy_prefix = "rev_";

    /**
     * FDR evaluation using entrapment strategy
     */
    public static boolean fdr_eval = false;

    public static double tol = 10;
    public static String tolu = "ppm";
    public static double itol = 20;
    public static String itolu = "ppm";
    public static int minPeptideLength = 7;
    public static int maxPeptideLength = 35;
    public static double minPeptideMz = 300.0;
    public static double maxPeptideMz = 2000.0;

    /**
     * Any fragment ion with mz <= 200 will not be considered as valid.
     */
    public static double min_fragment_ion_mz = 200.0;
    public static double max_fragment_ion_mz = 2000.0;

    /**
     * The type of transfer learning: ms2, rt, all
     */
    public static String tf_type = "all";

    /**
     * For spectral library generation, the top n fragment ions to consider
     */
    public static int top_n_fragment_ions = 12;

    /**
     * The maximum number of variable modifications
     */
    public static int maxVarMods = 1;

    /**
     * The maximum number of allowed modifications (variable and fixed) on the same amino acid
     * Default only one modification (either variable or fixed) is allowed on the same position
     */
    public static int maxModsPerAA = 1;

    /**
     * The maximum number of allowed missed cleavage sites
     */
    public static int maxMissedCleavages = 2;

    /**
     * options.addOption("e",true,"1:Trypsin (default), 2:Trypsin (no P rule), 3:Arg-C, 4:Arg-C (no P rule), 5:Arg-N, 6:Glu-C, 7:Lys-C");
     */
    public static int enzyme = 1;
    public static int cpu = 0;

    /**
     * Fixed modification
     */
    public static String fixMods = "1";

    /**
     * Variable modification
     */
    public static String varMods = "2";

    /**
     * This parameter is used to control whether add the AA substitution modifications when performing the modification
     * filtering. In default, it's false. But when performing missing protein identification, we can set it as true.
     */
    public static boolean addAAsubstitutionMods = false;

    /**
     * AI: MS instrument
     */
    public static String ms_instrument = "Eclipse";

    /**
     * AI: device, gpu or cpu
     */
    public static String device = "gpu";

    /**
     * AI: NCE, default is 27
     */
    public static double NCE = 27;

    public static double get_mass_error(double precursor_mass, double pep_mass, boolean is_ppm){
        double delta_mass;
        if(is_ppm){
            delta_mass = (1.0e6) * (pep_mass - precursor_mass) / pep_mass;
        }else{
            delta_mass = pep_mass - precursor_mass;
        }
        return delta_mass;
    }

    /**
     * Calculate the left and right values for specified tol.
     * @param msMass Mass of MS/MS spectrum
     * @param tol tol
     * @param isPpm true if the unit of tol is ppm
     * @return
     */
    public static double[] getRangeOfMass(double msMass, double tol, boolean isPpm){
        double [] massRange = new double [2];
        if(isPpm){
            massRange[0] = msMass /(1+ tol /(1.0e6));
            massRange[1] = msMass/(1- tol /(1.0e6));
        }else{
            massRange[0] = msMass-tol;
            massRange[1] = msMass+tol;
        }
        return massRange;
    }

    /**
     * Output folder
     */
    public static String outdir = "./";


    public static void init(){
        IonFactory.getDefaultNeutralLosses();
    }

    /**
     * Get version
     * @return
     */
    public static String getVersion(){
        Properties properties = new Properties();
        try {
            properties.load(CParameter.class.getClassLoader().getResourceAsStream("project.properties"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return properties.getProperty("version");
    }
}
