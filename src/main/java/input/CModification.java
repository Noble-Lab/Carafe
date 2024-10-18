package main.java.input;

import com.compomics.util.experiment.biology.modifications.Modification;
import com.compomics.util.experiment.biology.modifications.ModificationFactory;
import main.java.ai.AIGear;
import org.apache.commons.io.IOUtils;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

public final class CModification {

    private static CModification instance = null;
    private final HashMap<Integer,String> id2ptmname = new HashMap<>();
    private final ArrayList<Modification> var_Modifications = new ArrayList<>();

    private CModification(){
        ModificationFactory ptmFactory = ModificationFactory.getInstance();
        ArrayList<String> ptmNames = ptmFactory.getDefaultModificationsOrdered();
        ArrayList<String> top_mods = load_top_modifications();
        ptmNames.removeAll(top_mods);
        top_mods.addAll(ptmNames);
        int i=0;
        HashSet<String> ptm_name_list = new HashSet<>();
        for(String name : top_mods){
            Modification ptm = ptmFactory.getModification(name);
            if(!ptm.getCategory().name().contains("Nucleotide_Substitution")) {
                i++;
                id2ptmname.put(i, name);
                ptm_name_list.add(name);
            }
        }
        // the alphabetically ordered names of the user defined modifications
        ModificationUtils.getInstance();
        ArrayList<String> usrPTMs = ptmFactory.getUserModificationsOrdered();
        for (String name: usrPTMs){
            if(ptm_name_list.contains(name)){
                continue;
            }
            Modification ptm = ptmFactory.getModification(name);
            if(!ptm.getCategory().name().contains("Nucleotide_Substitution")) {
                i++;
                id2ptmname.put(i, name);
            }
        }

        addVarMods(CParameter.varMods);

    }

    public static CModification getInstance() {
        if (instance == null) {
            instance = new CModification();
        }
        return instance;
    }

    public ArrayList<String> load_top_modifications() {
        ArrayList<String> ptm_names = new ArrayList<>();
        InputStream inputStream = AIGear.class.getResourceAsStream("/main/resources/top_modifications.tsv");
        if (inputStream!=null){
            try {
                List<String> mods = IOUtils.readLines(inputStream, StandardCharsets.UTF_8);
                String[] head = mods.get(0).split("\t");
                HashMap<String, Integer> column2index = new HashMap<>();
                for (int i = 0; i < head.length; i++) {
                    column2index.put(head[i], i);
                }
                for (int i = 1; i < mods.size(); i++) {
                    String[] line = mods.get(i).split("\t");
                    ptm_names.add(line[column2index.get("mod_name")]);
                }

            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return ptm_names;
    }

    public String getPTMname(int index){
        ModificationFactory ptmFactory = ModificationFactory.getInstance();
        return(this.id2ptmname.get(index));
    }

    public ArrayList<Modification> getPTMs(String mod){
        ArrayList<Modification> ptms = new ArrayList<>();
        ModificationFactory ptmFactory = ModificationFactory.getInstance();
        String[] d = mod.split(",");
        for(int i=0;i<d.length;i++){
            ptms.add(ptmFactory.getModification(this.id2ptmname.get(Integer.valueOf(d[i]))));
        }
        return(ptms);
    }

    public Modification getPTMbyName(String mod){
        return ModificationFactory.getInstance().getModification(mod);
    }

    public ArrayList<Modification> get_var_modifications(){
        return(this.var_Modifications);
    }

    private void addVarMods(String mod){
        if(!mod.equalsIgnoreCase("0") && !mod.equalsIgnoreCase("no")){
            ModificationFactory ptmFactory = ModificationFactory.getInstance();
            String[] d = mod.split(",");
            for (int i = 0; i < d.length; i++) {
                this.var_Modifications.add(ptmFactory.getModification(this.id2ptmname.get(Integer.valueOf(d[i]))));
            }
        }

    }

    public void printPTM(){
        ModificationFactory ptmFactory = ModificationFactory.getInstance();
        System.out.println("mod_id\tmod_name\tmod_mass\tmod_type\tmod_category\tunimod_accession");
        for(int ptm_id : id2ptmname.keySet()){
            Modification ptm = ptmFactory.getModification(id2ptmname.get(ptm_id)) ;
            String unimod_acc = "-";
            if(ptm.getUnimodCvTerm()!=null){
                unimod_acc = ptm.getUnimodCvTerm().getAccession();
            }
            System.out.println(ptm_id + "\t" + ptm.getName() + "\t" + ptm.getMass() + "\t" + ptm.getModificationType()+"\t"+ptm.getCategory().name()+"\t"+unimod_acc);

        }
    }
}
