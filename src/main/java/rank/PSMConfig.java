package main.java.rank;

class PSMConfig {

    /**
     * The search engine name, DIA-NN, Skyline or generic
     */
    public static String search_engine_name = "DIA-NN";

    public static String protein_group_column_name = "Protein.Group";

    /**
     * The column name of stripped peptide sequence in the input file
     */
    public static String stripped_peptide_sequence_column_name = "Stripped.Sequence";

    /**
     * The column name of peptide modification in the input file
     */
    public static String peptide_modification_column_name = "Modified.Sequence";

    /**
     * The column name of precursor charge in the input file
     */
    public static String precursor_charge_column_name = "Precursor.Charge";

    /**
     * The column name of precursor intensity in the input file
     */
    public static String precursor_intensity_column_name = "Precursor.Normalised";

    /**
     * The column name of precursor m/z in the input file
     */
    public static String precursor_mz_column_name = "Precursor.MZ";

    public static String qvalue_column_name = "Q.Value";

    public static String global_qvalue_column_name = "Global.Q.Value";

    public static String lib_qvalue_column_name = "Lib.Q.Value";

    public static String global_pg_column_name = "Global.PG.Q.Value";

    public static String lib_pg_column_name = "Lib.PG.Q.Value";

    public static String ms_file_column_name = "File.Name";

}
