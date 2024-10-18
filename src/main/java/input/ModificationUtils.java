package main.java.input;

import com.compomics.util.experiment.biology.modifications.Modification;
import com.compomics.util.experiment.biology.modifications.ModificationCategory;
import com.compomics.util.experiment.biology.modifications.ModificationFactory;
import com.compomics.util.experiment.biology.modifications.ModificationType;
import com.compomics.util.experiment.biology.proteins.Peptide;
import com.compomics.util.experiment.identification.matches.ModificationMatch;
import com.compomics.util.pride.CvTerm;
import main.java.util.Cloger;
import org.apache.commons.lang3.StringUtils;
import uk.ac.ebi.pride.utilities.pridemod.io.unimod.model.Specificity;
import uk.ac.ebi.pride.utilities.pridemod.io.unimod.model.Unimod;
import uk.ac.ebi.pride.utilities.pridemod.io.unimod.model.UnimodModification;
import uk.ac.ebi.pride.utilities.pridemod.io.unimod.xml.UnimodReader;
import javax.xml.bind.JAXBException;
import java.io.*;
import java.util.*;

public class ModificationUtils {

    private static ModificationUtils instance = null;
    private final ModificationFactory ptmFactory;
    public static boolean save_mod2file = false;
    public static double leftMass = -250.0;
    public static double rightMass = 250.0;
    public static String out_dir = "./";

    private ModificationUtils(){
        ptmFactory = ModificationFactory.getInstance();
        try {
            importModFromUnimod();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static ModificationUtils getInstance() {
        if (instance == null) {
            instance = new ModificationUtils();
        }
        return instance;
    }

    public static void main(String[] args) {

    }

    private void importModFromUnimod() throws IOException {
        ArrayList<String > residues;
        ArrayList<String> testSite = new ArrayList<>();
        ArrayList<String> testPosition = new ArrayList<>();
        InputStream inputStream = ModificationUtils.class.getResourceAsStream("/main/resources/unimod.xml");
        UnimodReader unimodreader = null;
        try {
            unimodreader = new UnimodReader(inputStream);
        } catch (JAXBException e) {
            e.printStackTrace();
        }
        Unimod unimod = unimodreader.getUnimodObject();
        List<UnimodModification> unimodModificationList = unimod.getModifications().getMod();
        Cloger.getInstance().logger.info("All modifications in unimod:"+unimodModificationList.size());
        HashMap<String,Integer> modificationClassMap = new HashMap<>();
        ArrayList<String> mod2file_data = new ArrayList<>();
        //Parsing and get each modification;
        int i=0;
        for (UnimodModification unimodModification: unimodModificationList){
            Double monoMass = unimodModification.getDelta().getMonoMass().doubleValue();
            if(monoMass < leftMass || monoMass > rightMass){
                continue;
            }
            String modificationTitle = unimodModification.getTitle();
            String modificationName = unimodModification.getFullName();
            String unimod_accession = "UNIMOD:"+String.valueOf(unimodModification.getRecordId());
            //Get amino acid modification detailed modification;
            List<Specificity> specificityList = unimodModification.getSpecificity();
            for(Specificity specificity: specificityList){
                String site = specificity.getSite();
                String position = specificity.getPosition();
                String classification = specificity.getClassification();
                if(classification.equalsIgnoreCase("AA substitution") && !CParameter.addAAsubstitutionMods){
                    continue;
                }
                if(modificationClassMap.containsKey(classification)){
                    modificationClassMap.put(classification,modificationClassMap.get(classification)+1);
                }else{
                    modificationClassMap.put(classification,1);
                }
                // output
                if(save_mod2file){
                    mod2file_data.add(modificationTitle + " of " + site+"\t"+modificationName+"\t"+monoMass+"\t"+site+"\t"+position+"\t"+classification);
                }
                Modification ptm = null;
                String ptmName;
                i++;

                if(site.equals("N-term")){
                    if(position.equals("Any N-term")){
                        ptmName = modificationTitle + " of " + site;
                        ptm = new Modification(ModificationType.modn_peptide,ptmName,monoMass,null,ModificationCategory.Other);
                        ptm.setShortName(modificationTitle);
                    }else if(position.equals("Protein N-term")){
                        ptmName = modificationTitle + " of protein " + site;
                        ptm = new Modification(ModificationType.modn_protein,ptmName,monoMass,null,ModificationCategory.Other);
                        ptm.setShortName(modificationTitle);
                    }
                }else if(site.equals("C-term")){
                    if(position.equals("Any C-term")){
                        ptmName = modificationTitle + " of " + site;
                        ptm = new Modification(ModificationType.modc_peptide, ptmName,monoMass,null,ModificationCategory.Other);
                        ptm.setShortName(modificationTitle);
                    }else if(position.equals("Protein C-term")){
                        ptmName = modificationTitle + " of protein " + site;
                        ptm = new Modification(ModificationType.modc_protein, ptmName,monoMass,null,ModificationCategory.Other);
                        ptm.setShortName(modificationTitle);
                    }
                }else {
                    residues = new ArrayList<>();
                    residues.add(site);
                    if(position.equals("Any N-term")){
                        ptmName = modificationTitle + " of " + site;
                        // Modification at the N terminus of a peptide at particular amino acids.
                        ptm = new Modification(ModificationType.modnaa_peptide, ptmName, monoMass, residues,ModificationCategory.Other);
                        ptm.setShortName(modificationTitle);
                    }else if(position.equals("Protein N-term")){
                        ptmName = modificationTitle + " of protein " + site;
                        // Modification at the N terminus of a protein at particular amino acids.
                        ptm = new Modification(ModificationType.modnaa_protein, ptmName, monoMass, residues,ModificationCategory.Other);
                        ptm.setShortName(modificationTitle);
                    }else if (position.equals("Any C-term")){
                        ptmName = modificationTitle + " of " + site;
                        ptm = new Modification(ModificationType.modcaa_peptide, ptmName, monoMass, residues,ModificationCategory.Other);
                        ptm.setShortName(modificationTitle);
                    }else if(position.equals("Protein C-term")){
                        ptmName = modificationTitle + " of protein " + site;
                        ptm = new Modification(ModificationType.modcaa_protein, ptmName, monoMass, residues,ModificationCategory.Other);
                        ptm.setShortName(modificationTitle);
                    }else {
                        ptmName = modificationTitle + " of " + site;
                        ptm = new Modification(ModificationType.modaa, ptmName, monoMass, residues,ModificationCategory.Other);
                        ptm.setShortName(modificationTitle);
                    }
                }
                if(ptm!=null) {
                    if(classification.equalsIgnoreCase("AA substitution") && CParameter.addAAsubstitutionMods){
                        ptm.setCategory(ModificationCategory.Nucleotide_Substitution_One);
                    }
                    CvTerm cvTerm = new CvTerm();
                    cvTerm.setAccession(unimod_accession);
                    ptm.setUnimodCvTerm(cvTerm);
                    if(!ptmFactory.containsModification(ptm.getName())){
                        ptmFactory.addUserModification(ptm);
                    }
                }
                if(!testSite.contains(site)){
                    testSite.add(site);
                }
                if(!testPosition.contains(position)){
                    testPosition.add(position);
                }

            }
        }

        if(save_mod2file){
            // Export detailed modification information
            String out_mod_file = out_dir + "/Unimod.tsv";
            BufferedWriter bw = new BufferedWriter(new FileWriter(new File(out_mod_file)));
            bw.write("mod_title\tmod_name\tmono_mass\tsite\tposition\tclassification\n");
            for(String line: mod2file_data) {
                bw.write(line+"\n");
            }
            bw.close();
        }
    }

    public String getModificationString(Peptide peptide){
        String mod = "-";
        ModificationMatch modificationMatchs[] = peptide.getVariableModifications();
        if(modificationMatchs !=null && modificationMatchs.length>=1){
            String d[] = new String[modificationMatchs.length];
            for(int i=0;i<d.length;i++){
                Modification ptm = ptmFactory.getModification(modificationMatchs[i].getModification());
                d[i] = ptm.getName()+"@"+modificationMatchs[i].getSite()+"["+String.format("%.4f",ptm.getMass())+"]";
            }
            mod = StringUtils.join(d,';');
        }

        return(mod);
    }

    public String getSkylineFormatPeptide(Peptide peptide){
        ModificationMatch []modificationMatchs = peptide.getVariableModifications();
        if(modificationMatchs !=null && modificationMatchs.length>=1){
            String [] aa = peptide.getSequence().split("");
            String d[] = new String[modificationMatchs.length];
            for(int i=0;i<d.length;i++){
                Modification ptm = ptmFactory.getModification(modificationMatchs[i].getModification());
                int pos = modificationMatchs[i].getSite();
                if(pos==0){
                    aa[0] = "["+String.format("%.4f",ptm.getMass())+"]" + aa[0];
                }else if(pos > aa.length){
                    aa[pos-1] = aa[pos-1] + "["+String.format("%.4f",ptm.getMass())+"]";
                }else{
                    aa[pos-1] = aa[pos-1] + "["+String.format("%.4f",ptm.getMass())+"]";
                }
            }
            return StringUtils.join(aa,"");
        }else {
            return peptide.getSequence();
        }
    }

    public String getModificationString (String modValue){
        CModification.getInstance();
        ArrayList<String> mods = new ArrayList<>();
        if(!modValue.equalsIgnoreCase("0") && modValue != null && !modValue.isEmpty()){
            String[] fm = modValue.split(",");
            for (String s : fm) {
                String ptMname = CModification.getInstance().getPTMname(Integer.parseInt(s));
                mods.add(ptMname);
            }
        }
        String res = "-";
        if(!mods.isEmpty()){
            res = StringUtils.join(mods,',');
        }
        return(res);

    }

}
