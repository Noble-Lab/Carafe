package main.java.input;

import com.compomics.util.experiment.biology.modifications.Modification;
import com.compomics.util.experiment.biology.modifications.ModificationFactory;
import main.java.ai.AIGear;
import org.apache.commons.io.IOUtils;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.util.*;

public final class CModification {

    private static CModification instance = null;
    public final Map<Integer,String> id2ptmname = new LinkedHashMap<>();
    private final ArrayList<Modification> var_Modifications = new ArrayList<>();

    /**
     * Modification mapping: UniMod accession -> AI modification name
     */
    public HashMap<String,String> unimod2modification_code = new HashMap<>();
    public HashMap<String,String> modification_code2modification = new HashMap<>();

    /**
     * PSI modification name -> site with UniMod accession: e.g., M(UniMod:35) -> Oxidation@M
     */
    public HashMap<String,String> psi_name_site2site_unimod_acc = new HashMap<>();

    /**
     * PSI modification name -> site with UniMod accession: e.g., Oxidation@M -> M[Oxidation]
     */
    public HashMap<String,String> psi_name_site2site_psi_name = new HashMap<>();

    /**
     * PSI modification name -> EncyclopeDIA modification name: e.g., Oxidation@M -> M[Oxidation (M)]
     */
    public HashMap<String,String> psi_name_site2encyclopedia_mod_name = new HashMap<>();


    /**
     * PSI modification name -> Skyline modification name: e.g., Oxidation@M -> M[15.999]
     */
    public HashMap<String,String> psi_name_site2skyline_mod_name = new HashMap<>();

    /**
     * PTM name -> PTM id
     */
    public HashMap<String,Integer> ptm_name2id = new HashMap<>();

    private CModification(){
        // the alphabetically ordered names of the user defined modifications
        ModificationUtils.getInstance();
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
        load_UniMods();
    }

    public static CModification getInstance() {
        if (instance == null) {
            instance = new CModification();
        }
        return instance;
    }

    public String get_mod_name_by_site_unimod_acc(String site_unimod_acc){
        return modification_code2modification.get(unimod2modification_code.get(site_unimod_acc));
    }

    public double get_mod_mass_by_psi_name_site(String psi_name_site){
        String site_unimod_acc = psi_name_site2site_unimod_acc.get(psi_name_site);
        String mod_code = unimod2modification_code.get(site_unimod_acc);
        int mod_id = Integer.parseInt(mod_code);
        return ModificationFactory.getInstance().getModification(id2ptmname.get(mod_id)).getMass();
    }

    private void load_UniMods(){
        // AAAAC(UniMod:4)LDK2
        // AGEVLNQPM(UniMod:35)MMAAR2
        // AAAAAAAATMALAAPS(UniMod:21)SPTPESPTMLTK
        // AAAGPLDMSLPST(UniMod:21)PDLK
        //unimod2modification_code.put("C(UniMod:4)", "0");
        //unimod2modification_code.put("M(UniMod:35)", "1");
        //unimod2modification_code.put("S(UniMod:21)", "2");
        //unimod2modification_code.put("T(UniMod:21)", "3");
        //unimod2modification_code.put("Y(UniMod:21)", "4");

        //modification_code2modification.put("0", "Carbamidomethylation of C");
        //modification_code2modification.put("1", "Oxidation of M");
        //modification_code2modification.put("2", "Phosphorylation of S");
        //modification_code2modification.put("3", "Phosphorylation of T");
        //modification_code2modification.put("4", "Phosphorylation of Y");

        for(int mod_id: id2ptmname.keySet()){
            ptm_name2id.put(id2ptmname.get(mod_id),mod_id);
        }
        for(String mod_name: ModificationUtils.getInstance().mod_name2JMod.keySet()) {
            JMod jMod = ModificationUtils.getInstance().mod_name2JMod.get(mod_name);
            // TODO: need to handle terminal modifications
            if (jMod.position.toLowerCase().contains("term")) {
                System.err.println("Terminal modification is not supported:" + mod_name);
            }
            String site_unimod_acc = jMod.site + "(" + jMod.unimod_accession + ")";
            // take this as modification code (int)
            int mod_id = ptm_name2id.get(mod_name);
            unimod2modification_code.put(site_unimod_acc, String.valueOf(mod_id));
            modification_code2modification.put(String.valueOf(mod_id), mod_name);

            String psi_name = ModificationUtils.getInstance().mod_name2JMod.get(mod_name).psi_ms_name;
            String psi_name_site;
            // TODO: need to update to handle different terminal modifications
            String encyclopedia_mod_name;
            String skyline_mod_name;
            if(mod_name.contains("protein N-term")){
                psi_name_site = psi_name + "@Protein_N-term";
                // TODO: need to check if this works
                skyline_mod_name = "["+jMod.mod_mass+"]";
                encyclopedia_mod_name = "["+jMod.psi_ms_name+"]";
            }else{
                psi_name_site = psi_name + "@" + jMod.site;
                skyline_mod_name = jMod.site+"["+jMod.mod_mass+"]";
                encyclopedia_mod_name = jMod.site+"["+jMod.psi_ms_name+" ("+jMod.site+")]";
            }
            psi_name_site2site_unimod_acc.put(psi_name_site,site_unimod_acc);
            psi_name_site2site_psi_name.put(psi_name_site,psi_name);
            psi_name_site2encyclopedia_mod_name.put(psi_name_site,encyclopedia_mod_name);
            psi_name_site2skyline_mod_name.put(psi_name_site,skyline_mod_name);
        }
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
                System.out.println("Adding variable modification: " + this.id2ptmname.get(Integer.valueOf(d[i])) + " (" + d[i] + ")");
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

    public void change_mod_mass(int ptm_id, double mass){
        ModificationFactory ptmFactory = ModificationFactory.getInstance();
        Modification ptm = ptmFactory.getModification(id2ptmname.get(ptm_id));
        try {
            Field field = Modification.class.getDeclaredField("mass");
            field.setAccessible(true);
            field.set(ptm,mass);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
        System.out.println("Mass of " + ptm.getName() + " changed to " + ptm.getMass());

    }
}
