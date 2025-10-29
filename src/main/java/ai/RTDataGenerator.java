package main.java.ai;

import com.alibaba.fastjson.JSON;
import com.compomics.util.experiment.biology.modifications.Modification;
import main.java.db.DBGear;
import main.java.dia.DIAMap;
import main.java.dia.DIAMeta;
import main.java.input.CModification;
import main.java.input.CParameter;
import main.java.input.ModificationUtils;
import main.java.util.Cloger;
import org.apache.commons.cli.*;
import org.apache.commons.lang3.StringUtils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class RTDataGenerator {

    public static void main(String[] args) throws ParseException, IOException {
        long startTime = System.currentTimeMillis();
        Options options = new Options();
        options.addOption("i", true, "Peptide detection file from DIA-NN (e.g., report.tsv or report.parquet) or Skyline");
        options.addOption("o", true, "Output directory");
        options.addOption("ms", true, "MS file in mzML format: a single mzML or a folder containing mzML files.");
        options.addOption("fdr", true, "The minimum FDR cutoff to consider, default is 0.01");
        options.addOption("ptm_site_prob", true, "The minimum PTM site score to consider, default is 0.75");
        options.addOption("ptm_site_qvalue", true, "The threshold of PTM site qvalue, default is 1 (no filtering)");
        options.addOption("rt_max", true, "The max RT, default is 0.0, meaning using the max RT from the input MS file");
        options.addOption("data_type", true, "DDA or DIA (default)");
        options.addOption("rt_merge_method",true, "The method to merge RTs from different runs: mean, min, max");
        options.addOption("se", true, "The search engine used to generate the identification result: DIA-NN");
        options.addOption("mode", true, "Data type: general or phosphorylation");
        options.addOption("seed", true, "Random seed, 2024 in default");
        options.addOption("mod2mass",true,"Change the mass of a modification. The format is like: 2@0");
        // Format: mod_name@aa[mass][composition];mod_name@aa[mass][composition]
        // These modifications will be treated as variable modifications and add to the analysis
        options.addOption("user_var_mods",true,"User defined variable modifications");
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

        if(cmd.hasOption("user_var_mods")){
            CParameter.user_var_mods = cmd.getOptionValue("user_var_mods");
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

        //HashMap<Integer,Integer>mod_index2code = new HashMap<>();
        if(cmd.hasOption("varMod")){
            CParameter.varMods = cmd.getOptionValue("varMod");
            for(String mod: cmd.getOptionValue("varMod").split(",")){
                String[]d = mod.split(":");
                // if(d.length>=2) {
                //    mod_index2code.put(Integer.parseInt(d[0]), Integer.parseInt(d[1]));
                //}
            }
        }

        if(cmd.hasOption("user_var_mods")){
            AIWorker.user_mod = cmd.getOptionValue("user_var_mods");
            String[] user_mods = cmd.getOptionValue("user_var_mods").split(";");
            int i=0;
            ArrayList<Integer> mod_i = new ArrayList<>();
            for(String m: user_mods) {
                i++;
                // mod_name,aa,mass,composition
                String[] d = m.split(",");
                String mod_name = d[0];
                String aa = d[1];
                double mass = Double.parseDouble(d[2]);
                Modification ptm = null;
                String ptmName;
                if(CParameter.varMods.matches("^[1-9].*$")){
                    int mi = CModification.getInstance().ptm_name2id.get(mod_name + " of " + aa);
                    mod_i.add(mi);
                }
            }
            String use_var_mod_str = StringUtils.join(mod_i,',');
            if(!CParameter.varMods.isEmpty() && !CParameter.varMods.equals("0") && !CParameter.varMods.equals("no")){
                CParameter.varMods = CParameter.varMods + "," + use_var_mod_str;
            }else{
                CParameter.varMods = use_var_mod_str;
            }
            CModification.getInstance().addVarMods(use_var_mod_str);
            Cloger.getInstance().logger.info(" -varMods "+CParameter.varMods);
        }

        DBGear.init_enzymes();
        String psm_file = cmd.getOptionValue("i");
        CParameter.init();
        ModificationUtils.getInstance();


        AIGear aiGear = new AIGear();
        aiGear.load_mod_map();
        if(cmd.hasOption("mod2mass")){
            // change the mass of a modification
            ArrayList<String> mod2mass_list = new ArrayList<>();
            for(String mod: cmd.getOptionValue("mod2mass").split(",")){
                String[]d = mod.split("@");
                CModification.getInstance().change_mod_mass(Integer.parseInt(d[0]), Double.parseDouble(d[1]));
                String mod_name = CModification.getInstance().id2ptmname.get(Integer.parseInt(d[0]));
                String psi_name_site = aiGear.mod_map.get(mod_name);
                mod2mass_list.add(psi_name_site+"="+d[1]);
            }
            AIWorker.mod2mass = StringUtils.join(mod2mass_list,',');
        }

        if (cmd.hasOption("fdr")) {
            aiGear.fdr_cutoff = Double.parseDouble(cmd.getOptionValue("fdr"));
        }

        if(cmd.hasOption("se")){
            aiGear.search_engine = cmd.getOptionValue("se");
        }else{
            aiGear.search_engine = "DIA-NN";
        }

        if(cmd.hasOption("data_type")){
            aiGear.data_type = cmd.getOptionValue("data_type");
        }else{
            aiGear.data_type = "DIA";
        }

        if(cmd.hasOption("mode")){
            aiGear.mod_ai = cmd.getOptionValue("mode");
        }

        if(cmd.hasOption("ms_instrument")){
            aiGear.user_provided_ms_instrument = cmd.getOptionValue("ms_instrument");
            aiGear.use_user_provided_ms_instrument = true;
        }

        if(cmd.hasOption("rt_max")){
            aiGear.rt_max = Double.parseDouble(cmd.getOptionValue("rt_max"));
        }

        if(cmd.hasOption("rt_merge_method")){
            aiGear.rt_merge_method = cmd.getOptionValue("rt_merge_method");
        }

        if(cmd.hasOption("ptm_site_prob")){
            aiGear.ptm_site_prob_cutoff = Double.parseDouble(cmd.getOptionValue("ptm_site_prob"));
        }

        if(cmd.hasOption("ptm_site_qvalue")){
            aiGear.ptm_site_qvalue_cutoff = Double.parseDouble(cmd.getOptionValue("ptm_site_qvalue"));
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

        if(cmd.hasOption("ms")) {
            String ms_file = cmd.getOptionValue("ms");
            Cloger.getInstance().set_job_start_time();
            if (aiGear.search_engine.equalsIgnoreCase("DIA-NN") || aiGear.search_engine.equalsIgnoreCase("DIANN")) {
                aiGear.load_data(psm_file, ms_file, aiGear.fdr_cutoff);
                if (aiGear.data_type.equalsIgnoreCase("dia")) {
                    RTDataGenerator rtDataGenerator = new RTDataGenerator();
                    rtDataGenerator.generate_rt_data_from_diann_report(aiGear);
                }
            }
        }else{
            Cloger.getInstance().set_job_start_time();
            if (aiGear.search_engine.equalsIgnoreCase("DIA-NN") || aiGear.search_engine.equalsIgnoreCase("DIANN")) {
                aiGear.load_data(psm_file, aiGear.fdr_cutoff);
                if (aiGear.data_type.equalsIgnoreCase("dia")) {
                    RTDataGenerator rtDataGenerator = new RTDataGenerator();
                    rtDataGenerator.generate_rt_data_from_diann_report(aiGear);
                }
            }
        }
        long bTime = System.currentTimeMillis();
        Cloger.getInstance().logger.info("Time used for spectral library generation:" + (bTime - startTime) / 1000 + " s.");
    }


    public void generate_rt_data_from_diann_report(AIGear aiGear) throws IOException {
        CModification.getInstance();
        aiGear.load_mod_map();
        // for RT
        HashMap<String,PeptideRT> peptide2rt = new HashMap<>();
        int n_total_matches = 0;
        int n_total_matches_valid = 0;
        int n_total_psm_matches_valid = 0;
        int n_total_matches_max_fragment_ion_invalid = 0;
        int n_ptm_site_low_confidence = 0;

        // meta information about the MS data and model training
        BufferedWriter metaWriter = new BufferedWriter(new FileWriter(aiGear.out_dir + "/meta.json"));
        metaWriter.write("{\n");
        HashMap<String, JMeta> ms_file2meta = new HashMap<>();
        boolean first_meta = true;

        for(String ms_file: aiGear.ms_file2psm.keySet()){
            System.out.println("Process MS file:"+ms_file);
            ms_file2meta.put(ms_file, new JMeta());
            ms_file2meta.get(ms_file).ms_file = ms_file;
            // For store raw data
            DIAMeta meta = new DIAMeta();
            File MF = new File(ms_file);
            if(MF.isFile()) {
                meta.load_ms_data(ms_file);
                meta.get_ms_run_meta_data();
            }
            if(aiGear.rt_max > meta.rt_max){
                meta.rt_max = aiGear.rt_max;
                System.out.println("Use user-provided RT max:"+aiGear.rt_max);
            }
            String ms_instrument_name = meta.get_ms_instrument(ms_file);
            if(!ms_instrument_name.isEmpty()){
                CParameter.ms_instrument = ms_instrument_name;
                aiGear.ms_instrument = ms_instrument_name;
                System.out.println("MS instrument:"+ms_instrument_name);
            }else{
                System.out.println("No MS instrument detected from MS/MS data!");
            }

            ms_file2meta.get(ms_file).ms_instrument = ms_instrument_name;
            ms_file2meta.get(ms_file).rt_max = meta.rt_max;
            if(first_meta) {
                metaWriter.write("\"" + ms_file + "\":" +JSON.toJSONString(ms_file2meta.get(ms_file)));
                first_meta = false;
            }else{
                metaWriter.write(",\n\"" + ms_file + "\":" + JSON.toJSONString(ms_file2meta.get(ms_file)));
            }

            DIAMap diaMap_tmp = new DIAMap();
            diaMap_tmp.meta = meta;

            if(meta.rt_max > aiGear.rt_max){
                aiGear.rt_max = meta.rt_max;
                System.out.println("RT max:"+aiGear.rt_max);
            }else{
                System.out.println("RT max:"+aiGear.rt_max);
            }

            HashMap<String,ArrayList<String>> isoWinID2PSMs = new HashMap<>();
            boolean show_mod_ai_only_one_time = true;
            for(String line: aiGear.ms_file2psm.get(ms_file)) {
                String []d = line.split("\t");
                String peptide = d[aiGear.hIndex.get(PSMConfig.stripped_peptide_sequence_column_name)];
                String modification = aiGear.get_modification_diann(d[aiGear.hIndex.get(PSMConfig.peptide_modification_column_name)],peptide);
                if(modification.equalsIgnoreCase("null")){
                    continue;
                }
                aiGear.add_peptide(peptide,modification);
                String peptide_mod = peptide + "_" + modification;
                if(aiGear.mod_ai.equalsIgnoreCase("-") || aiGear.mod_ai.equalsIgnoreCase("general")){
                    if(show_mod_ai_only_one_time) {
                        Cloger.getInstance().logger.info("Training data generation for general modeling!");
                        show_mod_ai_only_one_time = false;
                    }
                }else if(aiGear.mod_ai.equalsIgnoreCase("phosphorylation")){
                    String mod_seq = d[aiGear.hIndex.get(PSMConfig.peptide_modification_column_name)];
                    if(show_mod_ai_only_one_time) {
                        Cloger.getInstance().logger.info("Training data generation for phosphorylation modeling!");
                        show_mod_ai_only_one_time = false;
                    }
                    if (aiGear.hIndex.containsKey(PSMConfig.ptm_site_confidence_column_name) && mod_seq.contains("UniMod:21")) {
                        // only filtering out low confidence phosphorylation peptides
                        if (Double.parseDouble(d[aiGear.hIndex.get(PSMConfig.ptm_site_confidence_column_name)]) < aiGear.ptm_site_prob_cutoff) {
                            n_ptm_site_low_confidence++;
                            continue;
                        }
                        if (Double.parseDouble(d[aiGear.hIndex.get(PSMConfig.ptm_site_qvalue_column_name)]) > aiGear.ptm_site_qvalue_cutoff) {
                            n_ptm_site_low_confidence++;
                            continue;
                        }
                    }
                }else{
                    System.err.println("Modification type is not supported:"+aiGear.mod_ai);
                    System.exit(1);
                }

                if (!peptide2rt.containsKey(peptide_mod)) {
                    peptide2rt.put(peptide_mod, new PeptideRT());
                }
                peptide2rt.get(peptide_mod).peptide = peptide;
                peptide2rt.get(peptide_mod).modification = modification;
                peptide2rt.get(peptide_mod).rts.add(Double.parseDouble(d[aiGear.hIndex.get(PSMConfig.rt_column_name)])); // Apex RT
                peptide2rt.get(peptide_mod).scores.add(Double.parseDouble(d[aiGear.hIndex.get(PSMConfig.qvalue_column_name)]));
            }
        }
        metaWriter.write("\n}");
        metaWriter.close();

        System.out.println("Total matches:"+n_total_matches);
        System.out.println("Total valid matches:"+n_total_matches_valid);
        System.out.println("Total valid PSM matches:"+n_total_psm_matches_valid);
        System.out.println("Total matches with invalid max fragment ion intensity:"+n_total_matches_max_fragment_ion_invalid);
        if(n_ptm_site_low_confidence >0){
            System.out.println("Total matches with PTM site low confidence:"+n_ptm_site_low_confidence);
        }
        aiGear.generate_rt_train_data(peptide2rt,aiGear.rt_merge_method,aiGear.out_dir+"/rt_train_data.tsv");
    }
}
