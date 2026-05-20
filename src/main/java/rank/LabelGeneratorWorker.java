package main.java.rank;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;

class LabelGeneratorWorker implements Runnable{


    public static String out_dir = "./";

    public static boolean save_pair_data_to_file = false;

    public static HashMap<String,Integer> protein2index = new HashMap<>();
    public static HashMap<String,Integer> peptide2index = new HashMap<>();
    public static HashMap<Integer,String> index2protein = new HashMap<>();
    public static HashMap<Integer,String> index2peptide = new HashMap<>();
    public static HashMap<String, HashSet<String>> protein2peptides = new HashMap<>();

    public HashMap<Integer,HashMap<Integer, Double>> pMap = new HashMap<>();
    public int run_index;
    ConcurrentHashMap<Integer, String> ms_run_index2file;
    /**
     * save the label for each peptide pair
     */
    public HashMap<Integer, HashMap<String,Boolean>> protein2peptide_pair2label = new HashMap<>();

    public static final ConcurrentHashMap<Integer, HashMap<String,Label>> global_protein2peptide_pair2label = new ConcurrentHashMap<>();

    public LabelGeneratorWorker(HashMap<Integer,HashMap<Integer, Double>> pMap, HashMap<Integer, HashMap<String,Boolean>> protein2peptide_pair2label, int run_index){
        this.pMap = pMap;
        this.protein2peptide_pair2label = protein2peptide_pair2label;
        this.run_index = run_index;
    }

    public LabelGeneratorWorker(HashMap<Integer,HashMap<Integer, Double>> pMap, ConcurrentHashMap<Integer, String> ms_run_index2file, int run_index){
        this.pMap = pMap;
        this.ms_run_index2file = ms_run_index2file;
        this.run_index = run_index;
    }

    @Override
    public void run() {
        // For each protein, get all possible pairs of peptide
        String pair;
        for (int proteinId : pMap.keySet()) {
            HashMap<String,Boolean> peptide_pair2label = new HashMap<>();
            // consider undetected peptides
            String [] protein_ids = index2protein.get(proteinId).split(RankLabelGenerator.protein_separator);
            HashSet<String> all_peptides = protein2peptides.get(protein_ids[0]);
            System.out.println(protein_ids[0]+"\t"+all_peptides.size());
            HashMap<Integer, Double> peptideIntensityMap = pMap.get(proteinId);
            List<Integer> peptideIds = new ArrayList<>(peptideIntensityMap.keySet());

            // Generate all possible pairs
            for (int i = 0; i < peptideIds.size(); i++) {
                for (int j = i + 1; j < peptideIds.size(); j++) {
                    pair = peptideIds.get(i)+RParameter.pair_separator+peptideIds.get(j);
                    if(peptideIntensityMap.get(peptideIds.get(i)) > peptideIntensityMap.get(peptideIds.get(j))){
                        // A > B: 1
                        this.protein2peptide_pair2label.get(proteinId).put(pair, true);
                        peptide_pair2label.put(pair,true);
                        synchronized (global_protein2peptide_pair2label) {
                            if(!global_protein2peptide_pair2label.get(proteinId).containsKey(pair)){
                                global_protein2peptide_pair2label.get(proteinId).put(pair,new Label());
                            }
                            global_protein2peptide_pair2label.get(proteinId).get(pair).n_pos++;
                        }
                    }else{
                        // A < B: 0
                        this.protein2peptide_pair2label.get(proteinId).put(pair, false);
                        synchronized (global_protein2peptide_pair2label) {
                            if(!global_protein2peptide_pair2label.get(proteinId).containsKey(pair)){
                                global_protein2peptide_pair2label.get(proteinId).put(pair,new Label());
                            }
                            global_protein2peptide_pair2label.get(proteinId).get(pair).n_neg++;
                        }
                    }
                }

                for (String peptide : all_peptides) {
                    if(RParameter.consider_precursor_charge) {
                        for(int charge = RParameter.minPeptideCharge; charge <= RParameter.maxPeptideCharge; charge++) {
                            String peptideID = peptide + "|" + charge;
                            if(peptide2index.containsKey(peptideID)){
                                continue;
                            }
                            pair = peptideIds.get(i) + RParameter.pair_separator + peptideID;
                            this.protein2peptide_pair2label.get(proteinId).put(pair, true);
                            synchronized (global_protein2peptide_pair2label) {
                                if (!global_protein2peptide_pair2label.get(proteinId).containsKey(pair)) {
                                    global_protein2peptide_pair2label.get(proteinId).put(pair, new Label());
                                }
                                global_protein2peptide_pair2label.get(proteinId).get(pair).n_pos++;
                            }
                            pair = peptideID + RParameter.pair_separator + peptideIds.get(i);
                            this.protein2peptide_pair2label.get(proteinId).put(pair, false);
                            synchronized (global_protein2peptide_pair2label) {
                                if (!global_protein2peptide_pair2label.get(proteinId).containsKey(pair)) {
                                    global_protein2peptide_pair2label.get(proteinId).put(pair, new Label());
                                }
                                global_protein2peptide_pair2label.get(proteinId).get(pair).n_neg++;
                            }
                        }
                    }
                }
            }
            synchronized (global_protein2peptide_pair2label) {
                global_protein2peptide_pair2label.putIfAbsent(proteinId, new HashMap<>());
                // add all peptide paris to the global map

            }
        }


        if(save_pair_data_to_file){
            String out_file = out_dir + "/pair_data_" + run_index + ".tsv";
            try {
                BufferedWriter writer = new BufferedWriter(new FileWriter(out_file));
                writer.write("protein\tpeptide_a\tpeptide_b\tpeptide_a_intensity\tpeptide_b_intensity\tlabel\n");
                for(int proteinId : protein2peptide_pair2label.keySet()) {
                    for (String peptidePair : this.protein2peptide_pair2label.get(proteinId).keySet()) {
                        String[] peptides = peptidePair.split(Pattern.quote(RParameter.pair_separator));
                        int peptideA = Integer.parseInt(peptides[0]);
                        int peptideB = Integer.parseInt(peptides[1]);
                        double intensityA = pMap.get(proteinId).get(peptideA);
                        double intensityB = pMap.get(proteinId).get(peptideB);
                        boolean label = this.protein2peptide_pair2label.get(proteinId).get(peptidePair);
                        writer.write(index2protein.get(proteinId) + "\t" + index2peptide.get(peptideA) + "\t" + index2peptide.get(peptideB) + "\t" + intensityA + "\t" + intensityB + "\t" + (label ? 1 : 0) + "\n");
                    }
                }
                writer.close();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            this.ms_run_index2file.put(run_index,out_file);
        }
    }
}
