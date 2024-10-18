package main.java.input;

import com.compomics.util.experiment.biology.modifications.Modification;
import com.compomics.util.experiment.biology.proteins.Peptide;
import com.compomics.util.experiment.identification.matches.ModificationMatch;
import com.compomics.util.experiment.identification.protein_sequences.SingleProteinSequenceProvider;
import com.compomics.util.experiment.identification.utils.ModificationUtils;
import com.compomics.util.experiment.io.biology.protein.SequenceProvider;
import com.compomics.util.parameters.identification.advanced.SequenceMatchingParameters;
import com.compomics.util.parameters.identification.search.ModificationParameters;
import org.paukov.combinatorics3.Generator;
import java.util.*;

public final class PeptideUtils {

    public static final SequenceProvider sequenceProvider = new SingleProteinSequenceProvider();
    private static final SequenceMatchingParameters modificationsSequenceMatchingParameters = SequenceMatchingParameters.getDefaultSequenceMatching();
    public static final ModificationParameters modificationParameters = new ModificationParameters();
    public static final SequenceMatchingParameters sequenceMatchingParameters = new SequenceMatchingParameters();

    public static ArrayList<Peptide> calcPeptideIsoforms(String peptideSequence){
        ArrayList<Modification> varMods = CModification.getInstance().get_var_modifications();
        int maxVarMods = CParameter.maxVarMods;
        Peptide peptide = new Peptide(peptideSequence);
        ArrayList<Peptide> peptides = new ArrayList<>();
        if(!varMods.isEmpty()) {
            ArrayList<CPTM> all_mod_sites = new ArrayList<>();
            // get all possible modifications sites for all variable modifications
            for (Modification ptm : varMods) {
                int[] pSites = ModificationUtils.getPossibleModificationSites(peptide, ptm, sequenceProvider, modificationsSequenceMatchingParameters);
                if(pSites.length >= 1){
                    for(int i : pSites){
                        all_mod_sites.add(new CPTM(i,ptm));
                    }
                }
            }
            if(!all_mod_sites.isEmpty()) {
                // >=1 possible modification site
                // the max number of modifications to consider
                int maxNumMods = Math.min(all_mod_sites.size(), maxVarMods);
                for(int k=1;k<=maxNumMods;k++){
                    for (List<CPTM> iCombination : Generator.combination(all_mod_sites).simple(k)) {
                        Peptide modPeptide = new Peptide(peptideSequence);
                        // same site only allows one variable or fixed modification. This is controlled by
                        // CParameter.maxModsPerAA.
                        // The max number of modifications occurring on any individual position
                        int maxModPerAA = 0;
                        // This is used to count how many modifications occurring on each position.
                        HashMap<Integer, Integer> pos2mods = new HashMap<>();
                        for (CPTM mod : iCombination) {
                            modPeptide.addVariableModification(new ModificationMatch(mod.modification.getName(), mod.pos));

                            if (pos2mods.containsKey(mod.pos)) {
                                pos2mods.put(mod.pos, pos2mods.get(mod.pos) + 1);
                            } else {
                                pos2mods.put(mod.pos, 1);
                            }
                            if (pos2mods.get(mod.pos) > maxModPerAA) {
                                maxModPerAA = pos2mods.get(mod.pos);
                            }
                        }

                        if (maxModPerAA <= CParameter.maxModsPerAA) {
                            // add fixed modifications
                            addFixedModification(modPeptide);
                            peptides.add(modPeptide);
                        }


                    }
                }
            }
        }

        // add peptide which only has fixed modifications
        Peptide peptide_no_mod = new Peptide(peptideSequence);
        addFixedModification(peptide_no_mod);
        peptides.add(peptide_no_mod);
        peptides.forEach(pep -> pep.getMass(modificationParameters,sequenceProvider,sequenceMatchingParameters));
        return(peptides);
    }

    public static void addFixedModification(Peptide peptide){
        // 0 means no fixed modification
        if(!CParameter.fixMods.equalsIgnoreCase("0")) {
            ArrayList<Modification> fixedMods = CModification.getInstance().getPTMs(CParameter.fixMods);
            if(!fixedMods.isEmpty()) {
                HashSet<Integer> varModSites = new HashSet<>();
                ModificationMatch []varModificationMatchs =  peptide.getVariableModifications();
                for(ModificationMatch match: varModificationMatchs){
                    varModSites.add(match.getSite());
                }
                for (Modification mod : fixedMods) {
                    int[] possibleSites = ModificationUtils.getPossibleModificationSites(peptide, mod, sequenceProvider, modificationsSequenceMatchingParameters);
                    // The location in the sequence. N-term modification is at index 0, C-term at index sequence length + 1, and other
                    // modifications at amino acid index starting from 1.
                    for (Integer k : possibleSites) {
                        if(!varModSites.contains(k)){
                            peptide.addVariableModification(new ModificationMatch(mod.getName(), k));
                        }

                    }
                }
            }
        }

    }

}
