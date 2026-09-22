package main.java.input;

import com.compomics.util.experiment.biology.modifications.Modification;
import com.compomics.util.experiment.biology.modifications.ModificationFactory;
import com.compomics.util.experiment.identification.matches.ModificationMatch;

import main.java.ai.AIGear;
import main.java.util.Cloger;

import org.apache.commons.io.IOUtils;
import java.io.IOException;
import java.io.InputStream;
import java.math.BigDecimal;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.util.*;

public final class CModification {

    private static CModification instance = null;
    public final Map<Integer,String> id2ptmname = new LinkedHashMap<>();
    private final ArrayList<Modification> var_Modifications = new ArrayList<>();

    /**
     * Modification mapping: UniMod accession -> AI modification name
     * jMod.site + "(" + jMod.unimod_accession + ")";
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
    private final HashMap<String, BigDecimal> preferredMassByModName = new HashMap<>();

    private boolean one_time_message = true;


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
        // This is used to remove duplicated PTMs
        // Key: Unimod accession, amino acid site, position, value: modification delta mass
        HashMap<String, Double> ptm_string2mass = new HashMap<>();
        for(String name : top_mods){
            Modification ptm = ptmFactory.getModification(name);
            if(!ptm.getCategory().name().contains("Nucleotide_Substitution")) {
                String unimod_acc = ptm.getUnimodCvTerm()!=null?ptm.getUnimodCvTerm().getAccession():"-";
                String site;
                if(name.contains(" of ")){
                    String []d=  name.split(" of ",2);
                    site = d[1];
                }else if(name.contains(" on ")){
                    String []d=  name.split(" on ",2);
                    site = d[1];
                }else if(name.contains(" to ")){
                    String []d=  name.split(" to ",2);
                    site = d[1];
                }else if(name.contains(" from ")){
                    String []d=  name.split(" from ",2);
                    site = d[1];
                }else{
                    Cloger.getInstance().logger.debug("Unknown modification name: " + name);
                    continue;
                }
                String ptm_string = unimod_acc + "_" + site;
                if(ptm_string2mass.containsKey(ptm_string) && Math.abs(ptm.getMass()-ptm_string2mass.get(ptm_string))<=0.001){
                    if(!name.startsWith("TMT ")){
                        Cloger.getInstance().logger.debug("Duplicate PTM: " + name + " -> " + ptm_string + " with mass: " + ptm.getMass());
                        continue;
                    }
                }else{
                    ptm_string2mass.put(ptm_string,ptm.getMass());
                }
                if(name.equalsIgnoreCase("Carbamidomethylation of C")){
                    // skip the this one
                    continue;
                }else if(name.equalsIgnoreCase("Phosphorylation of S")){
                    // skip the this one
                    continue;
                }else if(name.equalsIgnoreCase("Phosphorylation of T")){
                    // skip the this one
                    continue;
                }else if(name.equalsIgnoreCase("Phosphorylation of Y")){
                    // skip the this one
                    continue;
                }else{
                    i++;
                    id2ptmname.put(i, name);
                    ptm_name_list.add(name);
                    // for the common PTMs, use the mod mass values from ptmFactory
                    if(name.equalsIgnoreCase("Carbamidomethyl of C")){
                        this.change_mod_mass(i,ptmFactory.getModification("Carbamidomethylation of C").getMass());
                    } else if (name.equalsIgnoreCase("Phospho of S")) {
                        this.change_mod_mass(i,ptmFactory.getModification("Phosphorylation of S").getMass());
                    } else if (name.equalsIgnoreCase("Phospho of T")) {
                        this.change_mod_mass(i,ptmFactory.getModification("Phosphorylation of T").getMass());
                    } else if (name.equalsIgnoreCase("Phospho of Y")) {
                        this.change_mod_mass(i,ptmFactory.getModification("Phosphorylation of Y").getMass());
                    }
                }
            }
        }
        ArrayList<String> usrPTMs = ptmFactory.getUserModificationsOrdered();
        for (String name: usrPTMs){
            if(ptm_name_list.contains(name)){
                continue;
            }
            Modification ptm = ptmFactory.getModification(name);
            if(!ptm.getCategory().name().contains("Nucleotide_Substitution")) {
                String unimod_acc = ptm.getUnimodCvTerm()!=null?ptm.getUnimodCvTerm().getAccession():"-";
                String site;
                if(name.contains(" of ")){
                    String []d=  name.split(" of ",2);
                    site = d[1];
                }else if(name.contains(" on ")){
                    String []d=  name.split(" on ",2);
                    site = d[1];
                }else if(name.contains(" to ")){
                    String []d=  name.split(" to ",2);
                    site = d[1];
                }else if(name.contains(" from ")){
                    String []d=  name.split(" from ",2);
                    site = d[1];
                }else{
                    Cloger.getInstance().logger.debug("Unknown modification name: " + name);
                    continue;
                }
                String ptm_string = unimod_acc + "_" + site;
                if(ptm_string2mass.containsKey(ptm_string) && Math.abs(ptm.getMass()-ptm_string2mass.get(ptm_string))<=0.001){
                    if(!name.startsWith("TMT ")){
                        Cloger.getInstance().logger.debug("Duplicate PTM: " + name + " -> " + ptm_string + " with mass: " + ptm.getMass());
                        continue;
                    }
                }else{
                    ptm_string2mass.put(ptm_string,ptm.getMass());
                }
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

    /**
     * Convert a UniMod specificity site to the site string used by AlphaBase/AlphaPeptDeep.
     * AlphaBase names peptide terminal modifications "Any N-term"/"Any C-term"; UniMod reports
     * the site of those modifications simply as "N-term"/"C-term".
     *
     * @param site A UniMod specificity site, e.g. "K", "N-term" or "Protein N-term".
     * @return The site string used by AlphaBase, e.g. "K", "Any N-term" or "Protein N-term".
     */
    public static String to_alphabase_site(String site){
        if("N-term".equals(site)){
            return "Any N-term";
        }else if("C-term".equals(site)){
            return "Any C-term";
        }
        return site;
    }

    public String get_mod_name_by_site_unimod_acc(String site_unimod_acc){
        return modification_code2modification.get(unimod2modification_code.get(site_unimod_acc));
    }

    public double get_mod_mass_by_psi_name_site(String psi_name_site){
        return get_mod_mass_bigdecimal_by_psi_name_site(psi_name_site).doubleValue();
    }

    public BigDecimal get_mod_mass_bigdecimal_by_psi_name_site(String psi_name_site){
        String mod_name = get_mod_name_by_site_unimod_acc(psi_name_site2site_unimod_acc.get(psi_name_site));
        BigDecimal preferredMass = preferredMassByModName.get(mod_name);
        if (preferredMass != null) {
            return preferredMass;
        }
        String site_unimod_acc = psi_name_site2site_unimod_acc.get(psi_name_site);
        String mod_code = unimod2modification_code.get(site_unimod_acc);
        int mod_id = Integer.parseInt(mod_code);
        return BigDecimal.valueOf(ModificationFactory.getInstance().getModification(id2ptmname.get(mod_id)).getMass());
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
            if(!ptm_name2id.containsKey(mod_name)){
                continue;
            }
            JMod jMod = ModificationUtils.getInstance().mod_name2JMod.get(mod_name);
            // TODO: need to handle terminal modifications
            // mods: Acetyl@Protein_N-term mod_sites: 0
            if (jMod.position.toLowerCase().contains("term") && one_time_message) {
                if (CParameter.verbose == CParameter.VerboseType.DEBUG) {
                    System.err.println("Terminal modification is not well supported:" + mod_name);
                }
                one_time_message = false;
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
                // Acetyl@Protein_N-term, AlphaPeptDeep supported format
                psi_name_site = psi_name + "@Protein_N-term";
                // TODO: need to check if this works
                skyline_mod_name = "["+jMod.mod_mass+"]";
                encyclopedia_mod_name = "["+jMod.psi_ms_name+"]";
            }else if("N-term".equals(jMod.site) || "C-term".equals(jMod.site)){
                // Peptide terminal modification, e.g. TMT6plex@Any N-term
                psi_name_site = psi_name + "@" + to_alphabase_site(jMod.site);
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
        load_isobaric_label_sites();
    }

    /**
     * Register the isobaric label residue sites that the loop above cannot reach.
     * <p>
     * The UniMod-derived name for a lysine label (e.g. "TMT6plex of K") is dropped from
     * {@link #id2ptmname} by the duplicate-PTM check in the constructor, because its
     * UNIMOD:737_K key collides with the modification factory name "TMT 6-plex of K" and the
     * {@code startsWith("TMT ")} exemption does not match "TMT6plex...". Without an entry in
     * {@link #ptm_name2id} the loop above skips it, so "TMT6plex@K" never reaches the
     * psi_name_site maps and the library writers abort with
     * "modification TMT6plex@K is not supported yet".
     * <p>
     * The factory names do have ids, so they are used here to register the AlphaBase names.
     * Existing entries are never overwritten: the peptide N-terminal forms are already
     * registered through UniMod and keep the values they got there.
     */
    private void load_isobaric_label_sites() {
        // factory modification name, AlphaBase name, UniMod accession, UniMod specificity site
        // One row per AlphaBase name. The 10-plex and 11-plex reagents share TMT6plex@K and
        // TMT6plex@Any N-term with the 6-plex, so they need no rows of their own: the guard below
        // would skip them anyway, and listing them only suggested otherwise.
        String[][] labels = {
                {"TMT 6-plex of K",              "TMT6plex@K",           "UniMod:737",  "K"},
                {"TMT 6-plex of peptide N-term", "TMT6plex@Any N-term",  "UniMod:737",  "N-term"},
                {"TMT 2-plex of K",              "TMT2plex@K",           "UniMod:738",  "K"},
                {"TMT 2-plex of peptide N-term", "TMT2plex@Any N-term",  "UniMod:738",  "N-term"},
                {"TMTpro of K",                  "TMTpro@K",             "UniMod:2016", "K"},
                {"TMTpro of peptide N-term",     "TMTpro@Any N-term",    "UniMod:2016", "N-term"},
                {"iTRAQ 4-plex of K",            "iTRAQ4plex@K",         "UniMod:214",  "K"},
                {"iTRAQ 4-plex of Y",            "iTRAQ4plex@Y",         "UniMod:214",  "Y"},
                {"iTRAQ 4-plex of peptide N-term", "iTRAQ4plex@Any N-term", "UniMod:214", "N-term"},
                {"iTRAQ 8-plex of K",            "iTRAQ8plex@K",         "UniMod:730",  "K"},
                {"iTRAQ 8-plex of Y",            "iTRAQ8plex@Y",         "UniMod:730",  "Y"},
                {"iTRAQ 8-plex of peptide N-term", "iTRAQ8plex@Any N-term", "UniMod:730", "N-term"},
        };

        ModificationFactory ptmFactory = ModificationFactory.getInstance();
        for (String[] label : labels) {
            String factory_name = label[0];
            String psi_name_site = label[1];
            String site_unimod_acc = label[3] + "(" + label[2] + ")";

            if (psi_name_site2site_unimod_acc.containsKey(psi_name_site)) {
                // Already registered from UniMod; leave it alone.
                continue;
            }
            Integer mod_id = ptm_name2id.get(factory_name);
            if (mod_id == null) {
                continue;
            }

            unimod2modification_code.putIfAbsent(site_unimod_acc, String.valueOf(mod_id));
            modification_code2modification.putIfAbsent(String.valueOf(mod_id), factory_name);

            BigDecimal mass = preferredMassByModName.get(factory_name);
            if (mass == null) {
                mass = BigDecimal.valueOf(ptmFactory.getModification(factory_name).getMass());
            }
            String psi_name = psi_name_site.split("@")[0];
            String site = label[3];
            boolean terminal = site.endsWith("-term");

            psi_name_site2site_unimod_acc.put(psi_name_site, site_unimod_acc);
            psi_name_site2site_psi_name.put(psi_name_site, psi_name);
            psi_name_site2skyline_mod_name.put(psi_name_site,
                    terminal ? "[" + mass + "]" : site + "[" + mass + "]");
            psi_name_site2encyclopedia_mod_name.put(psi_name_site,
                    terminal ? "[" + psi_name + "]" : site + "[" + psi_name + " (" + site + ")]");
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
                    preferredMassByModName.put(line[column2index.get("mod_name")], new BigDecimal(line[column2index.get("mod_mass")]));
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

    public void addVarMods(String mod){
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

    public LinkedHashMap<Integer, String> get_top_mod_list(int top_n) {
        LinkedHashMap<Integer, String> mod_id2name = new LinkedHashMap<>();
        ModificationFactory ptmFactory = ModificationFactory.getInstance();
        for (int ptm_id : id2ptmname.keySet()) {
            Modification ptm = ptmFactory.getModification(id2ptmname.get(ptm_id));
            mod_id2name.put(ptm_id, ptm.getName() + "[" + String.format("%.4f",ptm.getMass()) + "]");
            if(mod_id2name.size() >= top_n) {
                break;
            }
        }
        return mod_id2name;
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
        if(CParameter.verbose == CParameter.VerboseType.DEBUG) {
            System.out.println("Mass of " + ptm.getName() + " changed to " + ptm.getMass());
        }

    }
}
