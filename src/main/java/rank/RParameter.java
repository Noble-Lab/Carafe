package main.java.rank;

class RParameter {

    public static boolean clip_nTerm_M = true;
    public static String db = "";
    public static String cmd = "-";


    public static int minPeptideLength = 7;
    public static int maxPeptideLength = 35;

    public static int minPeptideCharge = 2;
    public static int maxPeptideCharge = 4;

    public static boolean consider_precursor_charge = true;

    /**
     * The maximum number of variable modifications
     */
    public static int maxVarMods = 1;

    /**
     * The maximum number of allowed modifications (variable and fixed) on the same amino acid
     * Default only one modification (either variable or fixed) is allowed on the same position
     */
    public static int maxModsPerAA = 1;

    public static boolean I2L = false;

    /**
     * The maximum number of allowed missed cleavage sites
     */
    public static int maxMissedCleavages = 0;

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

    public static String pair_separator = ":";
    public static String peptide_form_separator = "|";
}
