package main.java.ai;

public class PSMConfig {

    /**
     * The search engine name, DIA-NN, Skyline or generic
     */
    public static String search_engine_name = "DIA-NN";

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
     * The column name of precursor m/z in the input file
     */
    public static String precursor_mz_column_name = "Precursor.MZ";

    /**
     * The column name of apex RT in the input file
     */
    public static String rt_column_name = "RT";

    /**
     * The column name of RT start in the input file
     */
    public static String rt_start_column_name = "RT.Start";

    /**
     * The column name of RT end in the input file
     */
    public static String rt_end_column_name = "RT.Stop";

    /**
     * The column name of apex MS2 index in the input file
     */
    public static String ms2_index_column_name = "MS2.Scan"; // this is index not scan for DIA-NN

    public static String ptm_site_confidence_column_name = "PTM.Site.Confidence";

    public static String ptm_site_qvalue_column_name = "PTM.Q.Value";

    public static String qvalue_column_name = "Q.Value";

    public static String im_column_name = "IM";

    public static String ms_file_column_name = "File.Name";

    public static void use_skyline_report_column_names(){
        stripped_peptide_sequence_column_name = "Peptide";
        peptide_modification_column_name = "Peptide Modified Sequence Unimod Ids";
        precursor_charge_column_name = "Precursor Charge";
        rt_column_name = "Best Retention Time";
        rt_start_column_name = "Min Start Time";
        rt_end_column_name = "Max End Time";
        ms2_index_column_name = "ms2index";
        ptm_site_confidence_column_name = "-";
        ptm_site_qvalue_column_name = "-";
        qvalue_column_name = "Detection Q Value";
        im_column_name = "Ion Mobility MS1";
        ms_file_column_name = "File Name";
        search_engine_name = "skyline";
    }
}
