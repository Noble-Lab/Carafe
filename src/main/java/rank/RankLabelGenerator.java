package main.java.rank;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.math.BigInteger;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;
import static java.util.stream.Collectors.toList;
import java.util.zip.GZIPInputStream;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.conf.Configuration;
import org.apache.parquet.hadoop.ParquetFileReader;
import org.apache.parquet.hadoop.util.HadoopInputFile;
import org.apache.parquet.schema.MessageType;

import com.compomics.util.experiment.biology.enzymes.Enzyme;
import com.compomics.util.pride.CvTerm;
import com.google.common.math.BigIntegerMath;

import net.sf.jfasta.FASTAElement;
import net.sf.jfasta.FASTAFileReader;
import net.sf.jfasta.impl.FASTAElementIterator;
import net.sf.jfasta.impl.FASTAFileReaderImpl;
import net.tlabs.tablesaw.parquet.TablesawParquetReadOptions;
import net.tlabs.tablesaw.parquet.TablesawParquetReader;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.LongColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;


class  RankLabelGenerator{


    public HashMap<String,Integer> protein2index = new HashMap<>();
    /**
     * peptide form (peptide + charge: precursor) or stripped peptide. This is only used for
     * detected peptides.
     */
    public HashMap<String,Integer> peptide2index = new HashMap<>();

    /**
     * key: protein ID, value: a set of peptides detected (peptide + charge)
     */
    public HashMap<String,HashSet<String>> protein2detected_peptides = new HashMap<>();

    /**
     *
     */
    public HashMap<Integer,HashMap<Integer,Double>> peptide_index2run2intensity = new HashMap<>();

    public HashMap<Integer,String> index2run = new HashMap<>();
    public HashMap<String,Integer> run2index = new HashMap<>();

    // index -> protein
    // index -> peptide
    public HashMap<Integer,String> index2protein = new HashMap<>();
    public HashMap<Integer,String> index2peptide = new HashMap<>();

    public String db = "";
    public static String protein_separator = ";";

    public double ratio_undetected_peptides = 1.0;
    public int min_detected_peptides = 2;
    /**
     * peptide index -> protein index
     */
    public HashMap<Integer,Integer> peptide_index2protein_index = new HashMap<>();

     public static void main(String [] args) throws ParseException, IOException {

         Options options = new Options();
         options.addOption("i", true, "peptide intensity file (e.g., report.tsv from DIA-NN)");
         options.addOption("prefix", true, "output file prefix");
         options.addOption("pc", true, "peptide intensity column name");
         options.addOption("db", true, "protein database");
         options.addOption("a", true, "save details to files");
         options.addOption("o", true, "output directory");
         options.addOption("enzyme",true,"Enzyme used for protein digestion. 0:Non enzyme, 1:Trypsin (default), 2:Trypsin (no P rule), 3:Arg-C, 4:Arg-C (no P rule), 5:Arg-N, 6:Glu-C, 7:Lys-C");
         options.addOption("miss_c",true,"The max missed cleavages, default is 0");
         options.addOption("min_pep_length", true, "The minimum length of peptide to consider, default is 7");
         options.addOption("max_pep_length", true, "The maximum length of peptide to consider, default is 35");
         options.addOption("min_pep_charge", true, "The minimum precursor charge to consider, default is 2");
         options.addOption("max_pep_charge", true, "The maximum precursor charge to consider, default is 4");
         options.addOption("r", true, "Ratio of undetected peptides to detected peptides, default ratio is 0");
         options.addOption("n", true, "The minimum number of detected peptides for a protein to be considered, default is 2");
         options.addOption("h", false, "Help");

         CommandLineParser parser = new DefaultParser(false);
         CommandLine cmd = parser.parse(options, args);
         if (cmd.hasOption("h") || cmd.hasOption("help") || args.length == 0) {
             HelpFormatter f = new HelpFormatter();
             f.setWidth(100);
             f.setOptionComparator(null);
             System.out.println("java -Xmx4G -jar carafe-rank.jar");
             f.printHelp("Options", options);
             return;
         }

         RankLabelGenerator train = new RankLabelGenerator();
         RankLabelGenerator.init_enzymes();
         if(cmd.hasOption("db")){
            train.db = cmd.getOptionValue("db");
         }
         if(cmd.hasOption("n")){
             train.min_detected_peptides = Integer.parseInt(cmd.getOptionValue("n"));
         }
         if(cmd.hasOption("r")){
             train.ratio_undetected_peptides = Double.parseDouble(cmd.getOptionValue("r"));
         }
         if(cmd.hasOption("enzyme")){
             RParameter.enzyme = Integer.parseInt(cmd.getOptionValue("enzyme"));
         }
         if(cmd.hasOption("miss_c")){
             RParameter.maxMissedCleavages = Integer.parseInt(cmd.getOptionValue("miss_c"));
         }
         if(cmd.hasOption("min_pep_charge")){
             RParameter.minPeptideCharge = Integer.parseInt(cmd.getOptionValue("min_pep_charge"));
         }
         if(cmd.hasOption("max_pep_charge")){
             RParameter.maxPeptideCharge = Integer.parseInt(cmd.getOptionValue("max_pep_charge"));
         }
         if(cmd.hasOption("min_pep_length")){
             RParameter.minPeptideLength = Integer.parseInt(cmd.getOptionValue("min_pep_length"));
         }
         if(cmd.hasOption("max_pep_length")){
             RParameter.maxPeptideLength = Integer.parseInt(cmd.getOptionValue("max_pep_length"));
         }
         String out_dir = "./";
         if(cmd.hasOption("o")){
             out_dir = cmd.getOptionValue("o");
             // make the directory if it does not exist
             File dir = new File(out_dir);
             if(!dir.exists()){
                 dir.mkdirs();
             }
         }
         if(cmd.hasOption("pc")){
             PSMConfig.precursor_intensity_column_name = cmd.getOptionValue("pc");
             System.out.println("The precursor intensity column name is set to:" + PSMConfig.precursor_intensity_column_name);
         }
         String out_prefix = "test";
         if(cmd.hasOption("prefix")){
             out_prefix = cmd.getOptionValue("prefix");
         }
         if(cmd.hasOption("a")){
             LabelGeneratorWorker.save_pair_data_to_file = true;
         }
         if(cmd.hasOption("i")){
             String peptide_intensity_file = cmd.getOptionValue("i");
             // check if the file exists
             File file = new File(peptide_intensity_file);
             if (file.exists()) {
                 train.generate_train_data(peptide_intensity_file, out_dir, out_prefix);
             }else {
                 // protien IDs for prediction
                 train.generate_prediction_data(train.db, cmd.getOptionValue("i"), out_dir, out_prefix);
             }
         }else{
             // generate prediction data
             train.generate_prediction_data(train.db, "", out_dir, out_prefix);
         }

     }

     public void generate_prediction_data(String db, String target_proteins, String out_dir, String out_prefix) throws IOException {
         ArrayList<String> target_proteins_list = new ArrayList<>();
         if(!target_proteins.isEmpty()){
             if(target_proteins.contains(",")){
                 target_proteins_list.addAll(List.of(target_proteins.split(",")));
             }else if(target_proteins.contains(";")){
                 target_proteins_list.addAll(List.of(target_proteins.split(";")));
             }else{
                 target_proteins_list.add(target_proteins);
             }
         }else{
             // use all proteins in the database
             target_proteins_list = get_all_proteins(db);
         }
         HashMap<String, HashSet<String>> protein2peptides = new HashMap<>();
         for(String protein : target_proteins_list){
             protein2peptides.put(protein, new HashSet<>());
         }
         this.protein_digest(this.db, protein2peptides);
         // generate all possible peptide pairs
         int [] charges = new int[RParameter.maxPeptideCharge-RParameter.minPeptideCharge+1];
         for(int i=0;i<charges.length;i++){
             charges[i] = RParameter.minPeptideCharge+i;
         }
         // generate all possible peptide pairs:
         System.out.println("Generate peptide pairs ...");
         ConcurrentHashMap<String, ArrayList<String>> protein2peptide_pairs = new ConcurrentHashMap<>(protein2peptides.size());
         protein2peptides.keySet().parallelStream().forEach(protein -> {
             ArrayList<String> peptide_pairs = new ArrayList<>();
             ArrayList<String> peptide_forms = new ArrayList<>();
             for(String peptide: protein2peptides.get(protein)){
                 for(int charge: charges){
                     peptide_forms.add(peptide + RParameter.peptide_form_separator + charge);
                 }
             }
             String line;
             for(int i=0;i<peptide_forms.size();i++){
                 for(int j=i+1;j<peptide_forms.size();j++){
                     String peptide_form_a = peptide_forms.get(i);
                     String peptide_form_b = peptide_forms.get(j);
                     String pair = peptide_form_a + RParameter.pair_separator + peptide_form_b;
                     line = protein + "\t" + pair + "\t" + peptide_form_a + "\t" + peptide_form_b;
                     peptide_pairs.add(line);
                 }

             }
             protein2peptide_pairs.put(protein, peptide_pairs);
         });
         System.out.println("Generate peptide pairs done.");
         String out_file = out_dir + "/" + out_prefix + "_pairs.txt";
         BufferedWriter writer = new BufferedWriter(new FileWriter(out_file, true));
         writer.write("protein\tpeptide_pair\tpeptide_a\tpeptide_b\n");
         for(String protein: protein2peptide_pairs.keySet()){
             for(String line: protein2peptide_pairs.get(protein)){
                 writer.write(line + "\n");
             }
         }
         writer.close();

     }

     public void generate_train_data(String input_file, String out_dir, String out_prefix) throws IOException {

         BufferedWriter pWriter = null;
         if(LabelGeneratorWorker.save_pair_data_to_file) {
             String out_file = out_dir + "/" + out_prefix + "_train.txt";
             // save the peptide intensity data to a file
             String out_peptide2intensity_file = out_dir + "/" + out_prefix + "_peptide2intensity.txt";
             pWriter = new BufferedWriter(new FileWriter(out_peptide2intensity_file));
             pWriter.write(PSMConfig.ms_file_column_name + "\t" +
                     PSMConfig.protein_group_column_name + "\t" +
                     PSMConfig.stripped_peptide_sequence_column_name + "\t" +
                     PSMConfig.peptide_modification_column_name + "\t" +
                     PSMConfig.precursor_charge_column_name + "\t" +
                     PSMConfig.precursor_intensity_column_name + "\n");
         }

         HashMap<String,Integer> hIndex = get_column_name2index(input_file);
         // run -> protein -> peptide -> intensity
         //HashMap<String,HashMap<Integer,HashMap<Integer, Double>>> dMap = new HashMap<>();
         int protein_index = 0;
         int peptide_index = 0;
         int ms_run_index = 0;
         HashMap<String, HashSet<String>> protein2detected_peptide_sequences = new HashMap<>();
         // peptide sequence -> all peptide forms
         HashMap<String, HashSet<String>> peptide_seq2peptide_forms = new HashMap<>();
         String ms_run;
         String peptide_sequence;
         String modification;
         String protein_id;
         String peptide_form;
         if(!input_file.endsWith(".parquet")) {
             String precursor_charge;
             String intensity;
             BufferedReader readr;
             int BUFFER_SIZE = 8192 * 16; // 128KB buffer instead of default 8KB
             if (input_file.endsWith(".gz")) {
                 // int BUFFER_SIZE = 8192 * 16; // 128KB buffer instead of default 8KB
                 FileInputStream fileStream = new FileInputStream(input_file);
                 InputStream zipStream = new GZIPInputStream(fileStream, BUFFER_SIZE);
                 Reader decoder = new InputStreamReader(zipStream);
                 readr = new BufferedReader(decoder, BUFFER_SIZE);
             } else {
                 readr = new BufferedReader(new FileReader(input_file), BUFFER_SIZE);
             }
             String line = readr.readLine();
             String[] h;
             int n_removed = 0;
             while ((line = readr.readLine()) != null) {
                 h = line.trim().split("\t");
                 if(Double.parseDouble(h[hIndex.get(PSMConfig.global_pg_column_name)]) > 0.01 ||
                         Double.parseDouble(h[hIndex.get(PSMConfig.lib_pg_column_name)]) > 0.01 ||
                         Double.parseDouble(h[hIndex.get(PSMConfig.global_qvalue_column_name)]) > 0.01 ||
                         Double.parseDouble(h[hIndex.get(PSMConfig.lib_qvalue_column_name)]) > 0.01 ||
                         Double.parseDouble(h[hIndex.get(PSMConfig.qvalue_column_name)]) > 0.01){
                     n_removed++;
                     continue;
                 }

                 ms_run = h[hIndex.get(PSMConfig.ms_file_column_name)];
                 peptide_sequence = h[hIndex.get(PSMConfig.stripped_peptide_sequence_column_name)];
                 modification = h[hIndex.get(PSMConfig.peptide_modification_column_name)];
                 precursor_charge = h[hIndex.get(PSMConfig.precursor_charge_column_name)];
                 intensity = h[hIndex.get(PSMConfig.precursor_intensity_column_name)];
                 protein_id = h[hIndex.get(PSMConfig.protein_group_column_name)];
                 peptide_form = peptide_sequence + RParameter.peptide_form_separator + precursor_charge;
                 if (!protein2index.containsKey(protein_id)) {
                     protein2index.put(protein_id, protein_index);
                     index2protein.put(protein_index, protein_id);
                     protein_index++;
                 }
                 if (!peptide2index.containsKey(peptide_form)) {
                     peptide2index.put(peptide_form, peptide_index);
                     index2peptide.put(peptide_index, peptide_form);
                     peptide_index++;
                 }
                 if (!run2index.containsKey(ms_run)) {
                     run2index.put(ms_run, ms_run_index);
                     index2run.put(ms_run_index, ms_run);
                     ms_run_index++;
                 }

                 if(!peptide_seq2peptide_forms.containsKey(peptide_sequence)){
                     peptide_seq2peptide_forms.put(peptide_sequence, new HashSet<>());
                 }
                 peptide_seq2peptide_forms.get(peptide_sequence).add(peptide_form);

                 if (!peptide_index2run2intensity.containsKey(peptide2index.get(peptide_form))) {
                     peptide_index2run2intensity.put(peptide2index.get(peptide_form), new HashMap<>());
                 }

                 peptide_index2run2intensity.get(peptide2index.get(peptide_form)).put(run2index.get(ms_run), Double.parseDouble(intensity));

                 if (!protein2detected_peptides.containsKey(protein_id)) {
                     protein2detected_peptides.put(protein_id, new HashSet<>());
                 }
                 protein2detected_peptides.get(protein_id).add(peptide_form);

                 if (!protein2detected_peptide_sequences.containsKey(protein_id)) {
                     protein2detected_peptide_sequences.put(protein_id, new HashSet<>());
                 }
                 protein2detected_peptide_sequences.get(protein_id).add(peptide_sequence);

                 //if (!dMap.containsKey(ms_run)) {
                 //    dMap.put(ms_run, new HashMap<>());
                 //}
                 //if (!dMap.get(ms_run).containsKey(protein2index.get(protein_id))) {
                 //    dMap.get(ms_run).put(protein2index.get(protein_id), new HashMap<>());
                 //}
                 if (!peptide_index2protein_index.containsKey(peptide2index.get(peptide_form))) {
                     peptide_index2protein_index.put(peptide2index.get(peptide_form), protein2index.get(protein_id));
                 }
                 //dMap.get(ms_run).get(protein2index.get(protein_id)).put(peptide2index.get(peptide_form), Double.parseDouble(intensity));
                 if (LabelGeneratorWorker.save_pair_data_to_file && pWriter != null) {
                     pWriter.write(ms_run + "\t" + protein_id + "\t" + peptide_sequence + "\t" + modification + "\t" + precursor_charge + "\t" + intensity + "\n");
                 }
             }
             readr.close();
             System.out.println("Number of rows removed due to low confidence: " + n_removed);
         }else{
             long precursor_charge;
             double intensity;
             int n_removed  = 0;
             // DIA-NN parquet file
             Table psmTable = new TablesawParquetReader().read(TablesawParquetReadOptions.builder(input_file)
                             .withOnlyTheseColumns(PSMConfig.ms_file_column_name,
                                     PSMConfig.stripped_peptide_sequence_column_name,
                                     PSMConfig.peptide_modification_column_name,
                                     PSMConfig.precursor_charge_column_name,
                                     PSMConfig.precursor_intensity_column_name,
                                     PSMConfig.protein_group_column_name,
                                     PSMConfig.global_pg_column_name,
                                     PSMConfig.lib_pg_column_name,
                                     PSMConfig.global_qvalue_column_name,
                                     PSMConfig.lib_qvalue_column_name,
                                     PSMConfig.qvalue_column_name)
                             .build());
             // Get columns directly for faster access
             int rowCount = psmTable.rowCount();
             DoubleColumn globalPgColumn = (DoubleColumn) psmTable.column(PSMConfig.global_pg_column_name);
             DoubleColumn libPgColumn = (DoubleColumn) psmTable.column(PSMConfig.lib_pg_column_name);
             DoubleColumn globalQvalueColumn = (DoubleColumn) psmTable.column(PSMConfig.global_qvalue_column_name);
             DoubleColumn libQvalueColumn = (DoubleColumn) psmTable.column(PSMConfig.lib_qvalue_column_name);
             DoubleColumn qvalueColumn = (DoubleColumn) psmTable.column(PSMConfig.qvalue_column_name);
             StringColumn msRunColumn = (StringColumn) psmTable.column(PSMConfig.ms_file_column_name);
             StringColumn peptideSequenceColumn = (StringColumn) psmTable.column(PSMConfig.stripped_peptide_sequence_column_name);
             StringColumn modificationColumn = (StringColumn) psmTable.column(PSMConfig.peptide_modification_column_name);
             LongColumn precursorChargeColumn = (LongColumn) psmTable.column(PSMConfig.precursor_charge_column_name);
             DoubleColumn intensityColumn = (DoubleColumn) psmTable.column(PSMConfig.precursor_intensity_column_name);
             StringColumn proteinIdColumn = (StringColumn) psmTable.column(PSMConfig.protein_group_column_name);

             System.out.println("Number of rows: " + rowCount);
             for (int i = 0; i < rowCount; i++) {
                 if (globalPgColumn.getDouble(i) > 0.01 ||
                         libPgColumn.getDouble(i) > 0.01 ||
                         globalQvalueColumn.getDouble(i) > 0.01 ||
                         libQvalueColumn.getDouble(i) > 0.01 ||
                         qvalueColumn.getDouble(i) > 0.01) {
                     n_removed++;
                     continue;
                 }

                 ms_run = msRunColumn.get(i);
                 peptide_sequence = peptideSequenceColumn.get(i);
                 modification = modificationColumn.get(i);
                 precursor_charge = precursorChargeColumn.getLong(i);
                 intensity = intensityColumn.getDouble(i);
                 protein_id = proteinIdColumn.get(i);
                 peptide_form = peptide_sequence + RParameter.peptide_form_separator + precursor_charge;
                 if (!protein2index.containsKey(protein_id)) {
                     protein2index.put(protein_id, protein_index);
                     index2protein.put(protein_index, protein_id);
                     protein_index++;
                 }
                 if (!peptide2index.containsKey(peptide_form)) {
                     peptide2index.put(peptide_form, peptide_index);
                     index2peptide.put(peptide_index, peptide_form);
                     peptide_index++;
                 }
                 if (!run2index.containsKey(ms_run)) {
                     run2index.put(ms_run, ms_run_index);
                     index2run.put(ms_run_index, ms_run);
                     ms_run_index++;
                 }

                 if(!peptide_seq2peptide_forms.containsKey(peptide_sequence)){
                     peptide_seq2peptide_forms.put(peptide_sequence, new HashSet<>());
                 }
                 peptide_seq2peptide_forms.get(peptide_sequence).add(peptide_form);

                 if (!peptide_index2run2intensity.containsKey(peptide2index.get(peptide_form))) {
                     peptide_index2run2intensity.put(peptide2index.get(peptide_form), new HashMap<>());
                 }

                 peptide_index2run2intensity.get(peptide2index.get(peptide_form)).put(run2index.get(ms_run), intensity);

                 if (!protein2detected_peptides.containsKey(protein_id)) {
                     protein2detected_peptides.put(protein_id, new HashSet<>());
                 }
                 protein2detected_peptides.get(protein_id).add(peptide_form);

                 if (!protein2detected_peptide_sequences.containsKey(protein_id)) {
                     protein2detected_peptide_sequences.put(protein_id, new HashSet<>());
                 }
                 protein2detected_peptide_sequences.get(protein_id).add(peptide_sequence);

                 //if (!dMap.containsKey(ms_run)) {
                 //    dMap.put(ms_run, new HashMap<>());
                 //}
                 //if (!dMap.get(ms_run).containsKey(protein2index.get(protein_id))) {
                 //    dMap.get(ms_run).put(protein2index.get(protein_id), new HashMap<>());
                 //}
                 if (!peptide_index2protein_index.containsKey(peptide2index.get(peptide_form))) {
                     peptide_index2protein_index.put(peptide2index.get(peptide_form), protein2index.get(protein_id));
                 }
                 //dMap.get(ms_run).get(protein2index.get(protein_id)).put(peptide2index.get(peptide_form), intensity);
                 if (LabelGeneratorWorker.save_pair_data_to_file && pWriter != null) {
                     pWriter.write(ms_run + "\t" + protein_id + "\t" + peptide_sequence + "\t" + modification + "\t" + precursor_charge + "\t" + intensity + "\n");
                 }
             }

             System.out.println("Number of rows removed due to low confidence: " + n_removed);
         }
         if(LabelGeneratorWorker.save_pair_data_to_file && pWriter != null) {
             pWriter.close();
         }

         System.out.println("Number of proteins: "+protein_index);
         System.out.println("Number of peptides: "+peptide_index);

         // remove peptides presented in multiple proteins
         HashMap<String, HashSet<String>> peptide_seq2remove = new HashMap<>();
         for(String p: protein2detected_peptide_sequences.keySet()){
             for(String peptide_seq: protein2detected_peptide_sequences.get(p)){
                 if(!peptide_seq2remove.containsKey(peptide_seq)){
                     peptide_seq2remove.put(peptide_seq, new HashSet<>());
                 }
                 peptide_seq2remove.get(peptide_seq).add(p);
             }
         }
         int n_shared_peptide_sequences = 0;
         for(String peptide_seq: peptide_seq2remove.keySet()){
             if(peptide_seq2remove.get(peptide_seq).size()>=2){
                 n_shared_peptide_sequences++;
                 // The peptide is presented in multiple proteins
                 for(String p: peptide_seq2remove.get(peptide_seq)){
                     protein2detected_peptide_sequences.get(p).remove(peptide_seq);
                 }
                 for(String p: peptide_seq2remove.get(peptide_seq)){
                     for(String pep_form: peptide_seq2peptide_forms.get(peptide_seq)) {
                         protein2detected_peptides.get(p).remove(pep_form);
                     }
                 }
             }
         }
         System.out.println("Number of detected peptides shared by multiple proteins: "+n_shared_peptide_sequences);

         // remove proteins with less than min_detected_peptides detected peptides
         HashSet<String> protein2remove = new HashSet<>();
         for(String p: protein2detected_peptide_sequences.keySet()){
             if(protein2detected_peptide_sequences.get(p).size()<min_detected_peptides){
                 protein2remove.add(p);
             }
         }
         if(!protein2remove.isEmpty()){
             for(String p: protein2remove) {
                 protein2detected_peptide_sequences.remove(p);
                 protein2detected_peptides.remove(p);
                 protein2index.remove(p);
             }
             System.out.println("The number of proteins removed:" + protein2remove.size());
         }

         // Generate in silico peptides from all detected proteins.
         HashMap<String,HashSet<String>> protein2peptides = new HashMap<>();
         double n_pairs_detected_peptides = 0;
         if(!this.db.isEmpty()){
             for(String proteinID: protein2index.keySet()){
                 if(proteinID.contains(";")){
                     String[] d= proteinID.split(protein_separator);
                     protein2peptides.put(d[0],new HashSet<>());
                 }else if(proteinID.contains(" / ")){
                     // skyline output
                     String[] d= proteinID.split(Pattern.quote(" / "));
                     protein2peptides.put(d[0],new HashSet<>());
                 }else{
                     protein2peptides.put(proteinID,new HashSet<>());
                 }

             }
             this.protein_digest(this.db, protein2peptides);
             // remove peptides presented in multiple proteins
             peptide_seq2remove.clear();
             for(String p: protein2peptides.keySet()){
                 for(String peptide_seq: protein2peptides.get(p)){
                     if(!peptide_seq2remove.containsKey(peptide_seq)){
                         peptide_seq2remove.put(peptide_seq, new HashSet<>());
                     }
                     peptide_seq2remove.get(peptide_seq).add(p);
                 }
             }
             n_shared_peptide_sequences = 0;
             for(String peptide_seq: peptide_seq2remove.keySet()){
                 if(peptide_seq2remove.get(peptide_seq).size()>=2){
                     n_shared_peptide_sequences++;
                     // The peptide is presented in multiple proteins
                     for(String p: peptide_seq2remove.get(peptide_seq)){
                         protein2peptides.get(p).remove(peptide_seq);
                     }
                 }
             }
             System.out.println("Number of in silico peptides shared by multiple proteins: "+n_shared_peptide_sequences);
             //LabelGeneratorWorker.protein2peptides = protein2peptides;
             if(ratio_undetected_peptides>0){
                 // for each protein, get the list of peptides which are not detected.
                 // sample ratio_undetected_peptides*(the number of detected peptides) undetected peptides

                 int n_charges = RParameter.maxPeptideCharge-RParameter.minPeptideCharge+1;
                 for(String proteinID: protein2detected_peptide_sequences.keySet()){
                     String[] d= proteinID.split(protein_separator);
                     if(proteinID.contains(" / ")) {
                         // skyline output
                         d = proteinID.split(Pattern.quote(" / "));
                     }
                     // peptide form here
                     int n_detected_peptides = protein2detected_peptides.get(proteinID).size();
                     BigInteger pairCount = BigInteger.ONE;
                     if(n_detected_peptides>=2) {
                         pairCount = BigIntegerMath.binomial(n_detected_peptides, 2);
                         n_pairs_detected_peptides = n_pairs_detected_peptides + pairCount.intValue();
                     }
                     HashSet<String> undetected_peptides = new HashSet<>(protein2peptides.get(d[0]));
                     undetected_peptides.removeAll(protein2detected_peptide_sequences.get(proteinID));
                     int n_u = (int)Math.ceil(ratio_undetected_peptides*pairCount.intValue()/n_detected_peptides/n_charges);
                     if(n_u > undetected_peptides.size()){
                         // e.g., n_u = 10, n_detected_peptides = 5
                         n_u = undetected_peptides.size();
                     }else{
                         if((n_u <= 1) && (!undetected_peptides.isEmpty())){
                             n_u = 1;
                         }
                     }
                     // sample n_u undetected peptides
                     List<String> undetected_peptide_list = new ArrayList<>(undetected_peptides);
                     Collections.shuffle(undetected_peptide_list);
                     // Take only the first n_u elements
                     List<String> sampled_undetected_peptides = undetected_peptide_list.subList(0, n_u);
                     HashSet<String> sampled_detected_and_undetected_peptides = new HashSet<>(sampled_undetected_peptides);
                     sampled_detected_and_undetected_peptides.addAll(protein2detected_peptide_sequences.get(proteinID));
                     // remove the peptides from LabelGeneratorWorker.protein2peptides for the current protein
                     protein2peptides.put(d[0], sampled_detected_and_undetected_peptides);
                     //System.out.println(n_detected_peptides+"\t"+pairCount.intValue()+"\t"+n_u+"\t"+undetected_peptides.size()+"\t"+protein2detected_peptide_sequences.get(proteinID).size()+"\t"+protein2peptides.get(d[0]).size());
                 }
             }
         }

         System.out.println("Number of peptide pairs (all detected): "+n_pairs_detected_peptides);

         // generate all possible peptide pairs
         int [] charges = new int[RParameter.maxPeptideCharge-RParameter.minPeptideCharge+1];
         for(int i=0;i<charges.length;i++){
             charges[i] = RParameter.minPeptideCharge+i;
         }
         // using parallel stream
         System.out.println("Generate peptide pairs ...");
         protein2detected_peptides.keySet().parallelStream().forEach(protein -> {
             String [] d= protein.split(protein_separator);
             if(protein.contains(" / ")) {
                 // skyline output
                 d = protein.split(Pattern.quote(" / "));
             }
             double intensity_a = 0;
             double intensity_b = 0;
             // Create a local map to collect all pairs for this protein before synchronizing
             HashMap<String, Label> localPairs = new HashMap<>();
             //System.out.print("Generate peptide pairs for protein: "+protein+"\t");
             //System.out.println(protein2detected_peptides.get(protein).size()+"\t"+protein2peptides.get(d[0]).size());
             Integer peptideIndexA = null;
             Integer peptideIndexB = null;
             String peptide_form_b;
             String pair;
             for(String peptide_form_a: protein2detected_peptides.get(protein)){
                 peptideIndexA = this.peptide2index.getOrDefault(peptide_form_a, null);
                 HashMap<Integer, Double> run2intensity = this.peptide_index2run2intensity.get(peptideIndexA);
                 for(String peptide_b: protein2peptides.get(d[0])){
                     for(int charge: charges){
                         peptide_form_b = peptide_b + RParameter.peptide_form_separator + charge;
                         if (peptide_form_a.equals(peptide_form_b)){
                             continue; // skip the same peptide form
                         }
                         peptideIndexB = this.peptide2index.getOrDefault(peptide_form_b, null);
                         pair = peptide_form_a + RParameter.pair_separator +peptide_form_b;
                         localPairs.computeIfAbsent(pair, k -> new Label());
                         if(peptideIndexB!=null){
                             // detected peptide
                             for(int run_index: this.index2run.keySet()){
                                 // Get intensities using the helper method
                                 intensity_a = getIntensityForPeptide(peptideIndexA, run_index);
                                 intensity_b = getIntensityForPeptide(peptideIndexB, run_index);
                                 if(intensity_a > intensity_b){
                                     // A > B: 1
                                     localPairs.get(pair).n_pos++;
                                     if(intensity_b >0 ){
                                         localPairs.get(pair).fully_detected = true;
                                         localPairs.get(pair).a_detected = true;
                                         localPairs.get(pair).b_detected = true;
                                     }else{
                                         localPairs.get(pair).a_detected = true;
                                         localPairs.get(pair).b_detected = false;
                                     }
                                 }else if(intensity_a < intensity_b){
                                     localPairs.get(pair).n_neg++;
                                     if(intensity_a >0 ){
                                         localPairs.get(pair).fully_detected = true;
                                         localPairs.get(pair).a_detected = true;
                                         localPairs.get(pair).b_detected = true;
                                     }else{
                                         localPairs.get(pair).a_detected = false;
                                         localPairs.get(pair).b_detected = true;
                                     }
                                 }
                             }
                         }else{
                             // if peptide_b is not detected, we only need to check the runs in which peptide_a is detected.
                             intensity_b = 0.0;
                             for(int run_index: run2intensity.keySet()){
                                 // Get intensities using the helper method
                                 intensity_a = run2intensity.get(run_index);
                                 // intensity_b = getIntensityForPeptide(peptideIndexB, run_index);
                                 if(intensity_a > intensity_b){
                                     // A > B: 1
                                     localPairs.get(pair).n_pos++;
                                     if(intensity_b >0 ){
                                         localPairs.get(pair).fully_detected = true;
                                         localPairs.get(pair).a_detected = true;
                                         localPairs.get(pair).b_detected = true;
                                     }else{
                                         localPairs.get(pair).a_detected = true;
                                         localPairs.get(pair).b_detected = false;
                                     }
                                 }else if(intensity_a < intensity_b){
                                     localPairs.get(pair).n_neg++;
                                     if(intensity_a >0 ){
                                         localPairs.get(pair).fully_detected = true;
                                         localPairs.get(pair).a_detected = true;
                                         localPairs.get(pair).b_detected = true;
                                     }else{
                                         localPairs.get(pair).a_detected = false;
                                         localPairs.get(pair).b_detected = true;
                                     }
                                 }
                             }
                         }

                     }
                 }
             }
             // Batch update the global map to minimize synchronization
             //synchronized (LabelGeneratorWorker.global_protein2peptide_pair2label) {
             LabelGeneratorWorker.global_protein2peptide_pair2label.put(protein2index.get(protein),localPairs);
             //}
         });
         System.out.println("Generate peptide pairs done.");

         //
         System.out.println("Generate consensus label ...");
         // print out the time: year month day hour minute second
         System.out.println("Start time: "+ LocalDateTime.now());
         generate_consensus_label(LabelGeneratorWorker.global_protein2peptide_pair2label,"majority_vote", out_dir);
         System.out.println("End time: "+ LocalDateTime.now());

     }

    // Helper method to abstract the intensity lookup
    private double getIntensityForPeptide(Integer peptideIndex, int runIndex) {
        if (peptideIndex == null) {
            return 0.0;
        }

        HashMap<Integer, Double> runToIntensity = peptide_index2run2intensity.get(peptideIndex);
        if (runToIntensity == null) {
            return 0.0;
        }

        return runToIntensity.getOrDefault(runIndex, 0.0);
    }

    public HashMap<String,Integer> get_column_name2index(String file) throws IOException {
        HashMap<String, Integer> hIndex = new HashMap<>();
         // DIA-NN parquet file
         if(file.endsWith(".parquet")){
             try (ParquetFileReader reader = ParquetFileReader.open(HadoopInputFile.fromPath(
                     new org.apache.hadoop.fs.Path(file), new Configuration()))) {
                 MessageType schema = reader.getFooter().getFileMetaData().getSchema();
                 for (int i = 0; i < schema.getFieldCount(); i++) {
                     hIndex.put(schema.getFieldName(i), i);
                     // System.out.println(schema.getFieldName(i) + "\t" + i);
                 }
                 if(!hIndex.containsKey(PSMConfig.ms_file_column_name)){
                     if(hIndex.containsKey("Run")){
                         PSMConfig.ms_file_column_name = "Run";
                     }
                 }
             }
         }else {

             BufferedReader reader;
             if (file.endsWith(".gz")) {
                 FileInputStream fileStream = new FileInputStream(file);
                 InputStream zipStream = new GZIPInputStream(fileStream);
                 Reader decoder = new InputStreamReader(zipStream);
                 reader = new BufferedReader(decoder);
             } else {
                 reader = new BufferedReader(new FileReader(file));
             }
             String head_line = reader.readLine().trim();
             hIndex = get_column_name2index_from_head_line(head_line);
             reader.close();
         }
         return hIndex;
    }

    public HashMap<String,Integer> get_column_name2index_from_head_line(String head_line){
        String []h = head_line.split("\t");
        HashMap<String,Integer> hIndex = new HashMap<>();
        for(int i=0;i<h.length;i++){
            hIndex.put(h[i],i);
        }
        return hIndex;
    }

    public void generate_consensus_label_will_delete(ConcurrentHashMap<String, HashMap<Integer,HashMap<String,Boolean>>> pairs2label,String method, String out_dir) throws IOException {
        // ms_run -> protein -> peptide pair -> label
        // pair2label2n
        HashMap<String,HashMap<Integer,Integer>> pair2label2n = new HashMap<>();
        for(String ms_run : pairs2label.keySet()){
            for(int proteinId : pairs2label.get(ms_run).keySet()){
                for(String peptide_pair : pairs2label.get(ms_run).get(proteinId).keySet()){
                    if(!pair2label2n.containsKey(peptide_pair)){
                        pair2label2n.put(peptide_pair, new HashMap<>());
                        pair2label2n.get(peptide_pair).put(0, 0);
                        pair2label2n.get(peptide_pair).put(1, 0);
                    }
                    pair2label2n.get(peptide_pair).put(pairs2label.get(ms_run).get(proteinId).get(peptide_pair)?1:0,pair2label2n.get(peptide_pair).get(pairs2label.get(ms_run).get(proteinId).get(peptide_pair)?1:0)+1);
                }
            }
        }

        String out_file = out_dir + "/" + "consensus_label.txt";
        BufferedWriter writer = new BufferedWriter(new FileWriter(out_file));
        writer.write("protein\tpeptide_pair\tpeptide_a\tpeptide_b\tn_pos\tn_neg\tlabel\n");

        if(method.equalsIgnoreCase("majority_vote")){
            for (String peptide_pair : pair2label2n.keySet()) {
                int n_pos = pair2label2n.get(peptide_pair).get(1);
                int n_neg = pair2label2n.get(peptide_pair).get(0);
                // calculate the consensus label
                int consensus_label = n_pos > n_neg ? 1 : 0;
                String []peptides = peptide_pair.split(Pattern.quote(RParameter.pair_separator));
                String peptide_a = index2peptide.get(Integer.parseInt(peptides[0]));
                String peptide_b = index2peptide.get(Integer.parseInt(peptides[1]));
                writer.write(index2protein.get(peptide_index2protein_index.get(Integer.parseInt(peptides[0])))+"\t"+peptide_pair + "\t" + peptide_a + "\t" + peptide_b + "\t" + n_pos + "\t" + n_neg + "\t" + consensus_label + "\n");
            }
        }else if(method.equalsIgnoreCase("binomial_distribution")) {
            // use the binomial distribution to calculate the probability of the label being 1
            for (String peptide_pair : pair2label2n.keySet()) {
                int n_pos = pair2label2n.get(peptide_pair).get(1);
                int n_neg = pair2label2n.get(peptide_pair).get(0);
                // calculate the consensus label
                int consensus_label = consensusLabel(n_pos, n_neg, 0.05);
                String []peptides = peptide_pair.split(Pattern.quote(RParameter.pair_separator));
                String peptide_a = index2peptide.get(Integer.parseInt(peptides[0]));
                String peptide_b = index2peptide.get(Integer.parseInt(peptides[1]));
                writer.write(index2protein.get(peptide_index2protein_index.get(Integer.parseInt(peptides[0])))+"\t"+peptide_pair + "\t" + peptide_a + "\t" + peptide_b + "\t" + n_pos + "\t" + n_neg + "\t" + consensus_label + "\n");
            }
        }
        writer.close();
    }


    public void generate_consensus_label(ConcurrentHashMap<Integer, HashMap<String,Label>> global_protein2peptide_pair2label,String method, String out_dir) throws IOException {
        String out_file = out_dir + "/" + "consensus_label.txt";
        BufferedWriter writer = new BufferedWriter(new FileWriter(out_file));
        writer.write("protein\tpeptide_pair\tpeptide_a\tpeptide_b\tn_pos\tn_neg\tlabel\ta_detected\tb_detected\tdetection\n");

        if(method.equalsIgnoreCase("majority_vote")){
            for(int proteinId: global_protein2peptide_pair2label.keySet()) {
                for (String peptide_pair : global_protein2peptide_pair2label.get(proteinId).keySet()) {
                    int n_pos = global_protein2peptide_pair2label.get(proteinId).get(peptide_pair).n_pos;
                    int n_neg = global_protein2peptide_pair2label.get(proteinId).get(peptide_pair).n_neg;
                    // calculate the consensus label
                    int consensus_label = n_pos > n_neg ? 1 : 0;
                    String[] peptides = peptide_pair.split(Pattern.quote(RParameter.pair_separator));
                    String peptide_a = peptides[0];
                    String peptide_b = peptides[1];
                    writer.write(index2protein.get(peptide_index2protein_index.get(peptide2index.get(peptides[0]))) + "\t" +
                            peptide_pair + "\t" + peptide_a + "\t" + peptide_b + "\t" + n_pos + "\t" + n_neg + "\t" + consensus_label + "\t"+
                            (global_protein2peptide_pair2label.get(proteinId).get(peptide_pair).a_detected?1:0)+"\t"+
                            (global_protein2peptide_pair2label.get(proteinId).get(peptide_pair).b_detected?1:0)+"\t"+
                            (global_protein2peptide_pair2label.get(proteinId).get(peptide_pair).fully_detected?1:0)+"\n");
                }
            }
        }else if(method.equalsIgnoreCase("binomial_distribution")) {
            // use the binomial distribution to calculate the probability of the label being 1
            for(int proteinId: global_protein2peptide_pair2label.keySet()) {
                for (String peptide_pair : global_protein2peptide_pair2label.get(proteinId).keySet()) {
                    int n_pos = global_protein2peptide_pair2label.get(proteinId).get(peptide_pair).n_pos;
                    int n_neg = global_protein2peptide_pair2label.get(proteinId).get(peptide_pair).n_neg;
                    // calculate the consensus label
                    int consensus_label = consensusLabel(n_pos, n_neg, 0.05);
                    String[] peptides = peptide_pair.split(Pattern.quote(RParameter.pair_separator));
                    String peptide_a = index2peptide.get(Integer.parseInt(peptides[0]));
                    String peptide_b = index2peptide.get(Integer.parseInt(peptides[1]));
                    writer.write(index2protein.get(peptide_index2protein_index.get(Integer.parseInt(peptides[0]))) + "\t" + peptide_pair + "\t" + peptide_a + "\t" + peptide_b + "\t" + n_pos + "\t" + n_neg + "\t" + consensus_label + "\n");
                }
            }
        }
        writer.close();
    }

    public static int consensusLabel(int n_pos, int n_neg, double tau) {
        int n = n_pos + n_neg;
        // count how many runs voted "1"
        int m = n_pos;
        // compute P(X ≤ m) under Binomial(n, 0.5)
        double cdf = 0.0;
        for (int k = 0; k <= m; k++) {
            cdf += binomial(n, k) * Math.pow(0.5, n);
        }

        if (1 - cdf < tau) {
            // very unlikely to get ≥ m heads by chance ⇒ label = 1
            return 1;
        } else if (cdf < tau) {
            // very unlikely to get ≤ m heads by chance ⇒ label = 0
            return 0;
        } else {
            throw new IllegalStateException(
                    String.format("Ambiguous consensus (n=%d, m=%d, P=%.5f)", n, m, cdf)
            );
        }
    }

    // Helper: exact binomial coefficient n choose k
    private static double binomial(int n, int k) {
        // for moderate n this is fine; for large n you'd want a log‐sum or Apache Commons Math
        return factorial(n) / (factorial(k) * factorial(n - k));
    }

    // Helper: simple factorial
    private static double factorial(int x) {
        double f = 1.0;
        for (int i = 2; i <= x; i++) {
            f *= i;
        }
        return f;
    }

    public void protein_digest(String db, HashMap<String,HashSet<String>> protein2peptides) {
        long startTime = System.currentTimeMillis();
        // read protein database
        File dbFile = new File(db);
        FASTAFileReader reader = null;
        try {
            reader = new FASTAFileReaderImpl(dbFile);
        } catch (IOException e) {
            e.printStackTrace();
        }
        FASTAElementIterator it = null;
        try {
            it = reader.getIterator();
        } catch (IOException e) {
            e.printStackTrace();
        }

        int i = 0;
        int n_proteins_in_db = 0;
        HashMap<String,String> proteinID2seq = new HashMap<>();
        try {
            while (it.hasNext()) {
                FASTAElement el = it.next();
                el.setLineLength(1);
                i++;
                String proSeq = el.getSequence();
                String [] d = el.getHeader().split("\\s+",2);
                String proteinID = d[0];
                if(protein2peptides.containsKey(proteinID)){
                    proteinID2seq.put(proteinID,proSeq);
                    n_proteins_in_db++;
                }else{
                    // uniprot ID: sp|A6NCF6|MA13P_HUMAN
                    String [] ids = proteinID.split(Pattern.quote("|"));
                    if(protein2peptides.containsKey(ids[1])) {
                        proteinID2seq.put(ids[1], proSeq);
                        n_proteins_in_db++;
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        reader.close();

        Enzyme enzyme = getEnzymeByIndex(RParameter.enzyme);
        int n_peptides = 0;
        // digest proteins and store the peptides to a map in which the key is the protein id and the value is a set of peptides
        proteinID2seq.keySet().parallelStream().forEach(proteinID -> {
            HashSet<String> peptideSet = digest_protein(enzyme, proteinID2seq.get(proteinID));
            synchronized (protein2peptides) {
                protein2peptides.put(proteinID, peptideSet);
            }
        });
        for(String proteinID: protein2peptides.keySet()){
            n_peptides = n_peptides + protein2peptides.get(proteinID).size();
        }

        long bTime = System.currentTimeMillis();
        System.out.println("Protein sequences:" + proteinID2seq.size() + ", total unique peptide sequences:" + n_peptides);
        if(n_proteins_in_db != proteinID2seq.size()){
            System.out.println("Warning: "+(protein2peptides.size()-n_proteins_in_db)+" proteins are not present in the database.");
        }
        System.out.println("Time used for protein digestion:" + (bTime - startTime) / 1000 + " s.");

    }

    public ArrayList<String> get_all_proteins(String db) {
        ArrayList<String> proteinIDs = new ArrayList<>();
        // read protein database
        File dbFile = new File(db);
        FASTAFileReader reader = null;
        try {
            reader = new FASTAFileReaderImpl(dbFile);
        } catch (IOException e) {
            e.printStackTrace();
        }
        FASTAElementIterator it = null;
        try {
            it = reader.getIterator();
        } catch (IOException e) {
            e.printStackTrace();
        }

        int i = 0;
        int n_proteins_in_db = 0;
        HashMap<String,String> proteinID2seq = new HashMap<>();
        try {
            while (it.hasNext()) {
                FASTAElement el = it.next();
                el.setLineLength(1);
                i++;
                String proSeq = el.getSequence();
                String [] d = el.getHeader().split("\\s+",2);
                String proteinID = d[0];
                proteinIDs.add(proteinID);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        reader.close();
        System.out.println("Total proteins in the database: " + proteinIDs.size());
        return proteinIDs;
    }

    public HashSet<String> digest_protein(Enzyme enzyme, String proteinSequence){
        proteinSequence = proteinSequence.toUpperCase();
        proteinSequence = proteinSequence.replaceAll("\\*$", "");
        if(RParameter.I2L) {
            proteinSequence = proteinSequence.replaceAll("I", "L");
        }
        HashSet<String> peptides = enzyme.digest(proteinSequence, RParameter.maxMissedCleavages, RParameter.minPeptideLength, RParameter.maxPeptideLength);
        if(RParameter.clip_nTerm_M && proteinSequence.startsWith("M")){
            List<String> n_term_peptides = peptides.stream().filter(proteinSequence::startsWith).filter(pep -> pep.length() >= (RParameter.minPeptideLength+1)).map(pep -> pep.substring(1)).collect(toList());
            if(!n_term_peptides.isEmpty()){
                peptides.addAll(n_term_peptides);
            }
        }
        return peptides;
    }

    public static Enzyme getEnzymeByIndex(int ind){

        if(ind < 0 || ind > enzymes.size()){
            System.err.println("Please provide a valid enzyme number:"+ind);
            System.exit(0);
        }
        System.out.println("Use enzyme:"+enzymes.get(ind).getName());

        return(enzymes.get(ind));
    }

    public static int getEnzymeIndexByName(String enzyme_name){

        int ind = -1;
        for(int i=0;i<enzymes.size();i++){
            if(enzymes.get(i).getName().equalsIgnoreCase(enzyme_name)){
                ind = i;
                break;
            }
        }
        if(ind == -1){
            System.err.println("Please provide a valid enzyme name:"+enzyme_name);
            System.exit(0);
        }else{
            System.out.println("Use enzyme:"+enzymes.get(ind).getName());
        }
        return ind;
    }

    private static ArrayList<Enzyme> enzymes = new ArrayList<>();
    public static void init_enzymes(){

        enzymes.clear();
        // 0 non-specific digestion
        Enzyme enzyme = new Enzyme("NoEnzyme");
        String all_aas = "ABCDEFGHIKLMNPQRSTUVWXY";
        for(int i=0;i<all_aas.length();i++){
            enzyme.addAminoAcidBefore(all_aas.charAt(i));
        }
        enzyme.setCvTerm(new CvTerm("PSI-MS", "MS:1001956", "NoEnzyme", null));
        enzymes.add(enzyme);


        enzyme = new Enzyme("Trypsin");
        enzyme.addAminoAcidBefore('R');
        enzyme.addAminoAcidBefore('K');
        enzyme.addRestrictionAfter('P');
        enzyme.setCvTerm(new CvTerm("PSI-MS", "MS:1001251", "Trypsin", null));
        enzymes.add(enzyme);

        enzyme = new Enzyme("Trypsin (no P rule)");
        enzyme.addAminoAcidBefore('R');
        enzyme.addAminoAcidBefore('K');
        enzyme.setCvTerm(new CvTerm("PSI-MS", "MS:1001313", "Trypsin/P", null));
        enzymes.add(enzyme);

        enzyme = new Enzyme("Arg-C");
        enzyme.addAminoAcidBefore('R');
        enzyme.addRestrictionAfter('P');
        enzyme.setCvTerm(new CvTerm("PSI-MS", "MS:1001303", "Arg-C", null));
        enzymes.add(enzyme);

        enzyme = new Enzyme("Arg-C (no P rule)");
        enzyme.addAminoAcidBefore('R');
        enzymes.add(enzyme);

        enzyme = new Enzyme("Arg-N");
        enzyme.addAminoAcidAfter('R');
        enzymes.add(enzyme);

        enzyme = new Enzyme("Glu-C");
        enzyme.addAminoAcidBefore('E');
        enzyme.setCvTerm(new CvTerm("PSI-MS", "MS:1001917", "glutamyl endopeptidase", null));
        enzymes.add(enzyme);

        enzyme = new Enzyme("Lys-C");
        enzyme.addAminoAcidBefore('K');
        enzyme.addRestrictionAfter('P');
        enzyme.setCvTerm(new CvTerm("PSI-MS", "MS:1001309", "Lys-C", null));
        enzymes.add(enzyme);

        enzyme = new Enzyme("Lys-C (no P rule)");
        enzyme.addAminoAcidBefore('K');
        enzyme.setCvTerm(new CvTerm("PSI-MS", "MS:1001310", "Lys-C/P", null));
        enzymes.add(enzyme);

        enzyme = new Enzyme("Lys-N");
        enzyme.addAminoAcidAfter('K');
        enzymes.add(enzyme);

        enzyme = new Enzyme("Asp-N");
        enzyme.addAminoAcidAfter('D');
        enzyme.setCvTerm(new CvTerm("PSI-MS", "MS:1001304", "Asp-N", null));
        enzymes.add(enzyme);

        enzyme = new Enzyme("Asp-N (ambic)");
        enzyme.addAminoAcidAfter('D');
        enzyme.addAminoAcidAfter('E');
        enzyme.setCvTerm(new CvTerm("PSI-MS", "MS:1001305", "Asp-N_ambic", null));
        enzymes.add(enzyme);

        enzyme = new Enzyme("Chymotrypsin");
        enzyme.addAminoAcidBefore('F');
        enzyme.addAminoAcidBefore('Y');
        enzyme.addAminoAcidBefore('W');
        enzyme.addAminoAcidBefore('L');
        enzyme.addRestrictionAfter('P');
        enzyme.setCvTerm(new CvTerm("PSI-MS", "MS:1001306", "Chymotrypsin", null));
        enzymes.add(enzyme);

        enzyme = new Enzyme("Chymotrypsin (no P rule)");
        enzyme.addAminoAcidBefore('F');
        enzyme.addAminoAcidBefore('Y');
        enzyme.addAminoAcidBefore('W');
        enzyme.addAminoAcidBefore('L');
        enzymes.add(enzyme);

        enzyme = new Enzyme("Pepsin A");
        enzyme.addAminoAcidBefore('F');
        enzyme.addAminoAcidBefore('L');
        enzyme.setCvTerm(new CvTerm("PSI-MS", "MS:1001311", "Pepsin A", null));
        enzymes.add(enzyme);

        enzyme = new Enzyme("CNBr");
        enzyme.addAminoAcidBefore('M');
        enzyme.setCvTerm(new CvTerm("PSI-MS", "MS:1001307", "CNBr", null));
        enzymes.add(enzyme);

        enzyme = new Enzyme("Thermolysin");
        enzyme.addAminoAcidAfter('A');
        enzyme.addAminoAcidAfter('F');
        enzyme.addAminoAcidAfter('I');
        enzyme.addAminoAcidAfter('L');
        enzyme.addAminoAcidAfter('M');
        enzyme.addAminoAcidAfter('V');
        enzymes.add(enzyme);

        enzyme = new Enzyme("LysargiNase");
        enzyme.addAminoAcidAfter('R');
        enzyme.addAminoAcidAfter('K');
        enzymes.add(enzyme);

        enzyme = new Enzyme("NoCut");
        enzyme.addAminoAcidAfter('X');
        enzymes.add(enzyme);
    }
}
