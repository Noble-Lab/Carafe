package main.java.db;
import com.compomics.util.experiment.biology.enzymes.Enzyme;
import com.compomics.util.experiment.biology.ions.impl.ElementaryIon;
import com.compomics.util.pride.CvTerm;
import main.java.input.*;
import main.java.util.Cloger;
import net.sf.jfasta.FASTAElement;
import net.sf.jfasta.FASTAFileReader;
import net.sf.jfasta.impl.FASTAElementIterator;
import net.sf.jfasta.impl.FASTAFileReaderImpl;
import org.apache.commons.io.FileUtils;
import java.io.*;
import java.util.*;
import java.util.stream.Collectors;
import static java.util.stream.Collectors.*;

public class DBGear {

    public String db = "";
    public boolean I2L = false;

    public DBGear() {

    }

    public HashMap<String,String>  generate_pep2pro_from_pep_index_file(String pep_file) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(pep_file));
        String head = reader.readLine();
        head = head.trim();
        String []h = head.split("\t");
        HashMap<String,Integer> hMap = new HashMap<>();
        for(int i=0;i<h.length;i++){
            hMap.put(h[i],i);
        }
        String line;
        String peptideSequence;
        String proteins;
        HashMap<String,String> pep2pro = new HashMap<>(1000);
        while ((line = reader.readLine())!=null){
            String d[] = line.split("\t");
            peptideSequence = d[hMap.get("sequence")];
            proteins = d[hMap.get("proteins")];
            pep2pro.putIfAbsent(peptideSequence,proteins);
        }
        reader.close();
        return pep2pro;
    }

    public Set<String> protein_digest(String db) {
        CModification.getInstance();
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

        HashSet<String> proteins = new HashSet<>(10000);
        int i = 0;
        try {
            while (it.hasNext()) {
                FASTAElement el = it.next();
                el.setLineLength(1);
                i++;
                String proSeq = el.getSequence();
                proteins.add(proSeq);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        reader.close();

        Enzyme enzyme = DBGear.getEnzymeByIndex(CParameter.enzyme);
        Set<String> peptides = proteins.parallelStream()
                .map(seq -> digest_protein(enzyme, seq))
                .flatMap(Set::stream)
                .filter(pep -> !pep.contains("X"))
                .collect(toSet());

        long bTime = System.currentTimeMillis();
        Cloger.getInstance().logger.info("Protein sequences:" + i + ", total unique peptide sequences:" + peptides.size());

        if(CParameter.decoy_level.equalsIgnoreCase("peptide")){
            Cloger.getInstance().logger.info("Add peptide level decoy peptides.");
            Set<String> decoy_peptides = peptides.parallelStream().map(seq -> reverse_peptide(seq,true)).collect(toSet());
            peptides.addAll(decoy_peptides);
        }

        Cloger.getInstance().logger.info("Time used for protein digestion:" + (bTime - startTime) / 1000 + " s.");
        return peptides;
    }

    private String reverse_peptide(String peptide, boolean keep_nc_aa){
        if(keep_nc_aa){
            return peptide.charAt(0) + reverse_peptide(peptide.substring(1, peptide.length() - 1), false) + peptide.charAt(peptide.length() - 1);
        }else{
            StringBuilder decoy = new StringBuilder(peptide);
            return(decoy.reverse().toString());
        }
    }

    public double get_mz(double mass, int charge) {
        return (mass + charge * ElementaryIon.proton.getTheoreticMass()) / charge;
    }

    public HashSet<String> digest_protein(Enzyme enzyme, String proteinSequence){
        proteinSequence = proteinSequence.toUpperCase();
        proteinSequence = proteinSequence.replaceAll("\\*$", "");
        if(this.I2L) {
            proteinSequence = proteinSequence.replaceAll("I", "L");
        }
        HashSet<String> peptides = enzyme.digest(proteinSequence, CParameter.maxMissedCleavages, CParameter.minPeptideLength, CParameter.maxPeptideLength);
        if(CParameter.clip_nTerm_M && proteinSequence.startsWith("M")){
            List<String> n_term_peptides = peptides.stream().filter(proteinSequence::startsWith).filter(pep -> pep.length() >= (CParameter.minPeptideLength+1)).map(pep -> pep.substring(1)).collect(toList());
            if(!n_term_peptides.isEmpty()){
                peptides.addAll(n_term_peptides);
            }
        }
        return peptides;
    }

    public HashMap<String,String> digest_protein(Enzyme enzyme, String proteinID, String proteinSequence, HashMap<String,ArrayList<ProteinHit>> pepMap){
        Set<String> peptides;
        if(CParameter.decoy_level.equalsIgnoreCase("peptide")){
            peptides = digest_protein(enzyme, proteinSequence);
        }else{
            peptides = digest_protein(enzyme,proteinSequence).stream()
                    .filter(pepMap::containsKey)
                    .collect(toSet());
        }

        HashMap<String, String> pep2pro = new HashMap<>();
        if(!peptides.isEmpty()){
            pep2pro = new HashMap<>(peptides.size());
            for(String pep: peptides){
                if(pepMap.containsKey(pep)) {
                    pep2pro.put(pep, proteinID);
                }
            }
        }

        if(CParameter.decoy_level.equalsIgnoreCase("peptide")){
            Set<String> decoy_peptides = peptides.stream().map(seq -> reverse_peptide(seq,true))
                    .filter(pepMap::containsKey)
                    .collect(toSet());
            if(!decoy_peptides.isEmpty()) {
                for (String pep : decoy_peptides) {
                    pep2pro.put(pep, CParameter.decoy_prefix + proteinID);
                }
            }
        }
        return pep2pro;
    }


    public HashMap<String,String> digest_protein(Enzyme enzyme, String proteinID, String proteinSequence){
        Set<String> peptides = digest_protein(enzyme, proteinSequence);
        if(!peptides.isEmpty()){
            HashMap<String, String> pep2pro = new HashMap<>(peptides.size());
            for(String pep: peptides){
                pep2pro.put(pep, proteinID);
            }
            return pep2pro;
        }else{
            return new HashMap<>();
        }
    }

    public Map<String,String> digest_protein(String db) throws IOException {
        File D = new File(db);
        Map<String, String> pep2proteins = new HashMap<>();
        if(!D.isDirectory()) {
            if(!CParameter.fdr_eval) {
                Enzyme enzyme = DBGear.getEnzymeByIndex(CParameter.enzyme);
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


                HashMap<String, String> proteins = new HashMap<>(10000);
                int i = 0;
                try {
                    while (it.hasNext()) {
                        FASTAElement el = it.next();
                        el.setLineLength(1);
                        i++;
                        String headLine[] = el.getHeader().split("\\s+");
                        String proID = headLine[0];
                        String proSeq = el.getSequence();
                        proteins.put(proID, proSeq);
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }

                reader.close();

                pep2proteins = proteins.entrySet().parallelStream()
                        .map(entry -> digest_protein(enzyme, entry.getKey(), entry.getValue()))
                        .flatMap(m -> m.entrySet().stream())
                        .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (c1, c2) -> c1 + ";" + c2));
            }else{
                String pep_index_file = CParameter.outdir + File.separator + "pep_list.tsv";
                pep2proteins = generate_pep2pro_from_pep_index_file(pep_index_file);
            }
        }else{

            String pep_index_file = db + File.separator + "pep_index.tsv";
            pep2proteins = generate_pep2pro_from_pep_index_file(pep_index_file);

        }
        return pep2proteins;

    }

    public void add_protein_to_psm_table(String psm_file, String db) throws IOException {

        String old_psm_rank_file;
        if(psm_file.endsWith(".txt")){
            old_psm_rank_file = psm_file.replaceAll("txt$","") + "tmp";
        } else if (psm_file.endsWith(".tsv")) {
            old_psm_rank_file = psm_file.replaceAll("tsv$","") + "tmp";
        }else{
            old_psm_rank_file = psm_file + ".tmp";
        }
        FileUtils.copyFile(new File(psm_file), new File(old_psm_rank_file));
        BufferedReader psmReader = new BufferedReader(new FileReader(old_psm_rank_file));
        String head = psmReader.readLine();
        head = head.trim();
        String []h = head.split("\t");
        HashMap<String,Integer> hIndex = new HashMap<>();
        for(int i=0;i<h.length;i++){
            hIndex.put(h[i],i);
        }
        String line;
        HashMap<String, ArrayList<ProteinHit>> pepMap = new HashMap<>();
        while((line = psmReader.readLine())!=null){
            line = line.trim();
            String []d = line.split("\t");
            if(hIndex.containsKey("peptide")) {
                pepMap.putIfAbsent(d[hIndex.get("peptide")], new ArrayList<>());
            }else if(hIndex.containsKey("sequence")){
                pepMap.putIfAbsent(d[hIndex.get("sequence")], new ArrayList<>());
            }else{
                System.err.println("Please provide peptide column name in the psm file. The column name should be either \"peptide\" or \"sequence\".");
                System.exit(1);
            }
        }
        psmReader.close();

        File D = new File(db);
        Map<String, String> pep2proteins = new HashMap<>();
        if(!D.isDirectory()) {
            if(!CParameter.fdr_eval) {
                Enzyme enzyme = DBGear.getEnzymeByIndex(CParameter.enzyme);
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


                HashMap<String, String> proteins = new HashMap<>(10000);
                int i = 0;
                try {
                    while (it.hasNext()) {
                        FASTAElement el = it.next();
                        el.setLineLength(1);
                        i++;
                        String headLine[] = el.getHeader().split("\\s+");
                        String proID = headLine[0];
                        String proSeq = el.getSequence();
                        proteins.put(proID, proSeq);
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }

                reader.close();

                pep2proteins = proteins.entrySet().parallelStream()
                        .map(entry -> digest_protein(enzyme, entry.getKey(), entry.getValue(), pepMap))
                        .flatMap(m -> m.entrySet().stream())
                        .collect(toMap(Map.Entry::getKey, Map.Entry::getValue, (c1, c2) -> c1 + ";" + c2));
            }else{
                String pep_index_file = CParameter.outdir + File.separator + "pep_list.tsv";
                pep2proteins = generate_pep2pro_from_pep_index_file(pep_index_file);
            }
        }else{

            String pep_index_file = db + File.separator + "pep_index.tsv";
            pep2proteins = generate_pep2pro_from_pep_index_file(pep_index_file);

        }


        BufferedWriter writer = new BufferedWriter(new FileWriter(psm_file));
        psmReader = new BufferedReader(new FileReader(old_psm_rank_file));
        head = psmReader.readLine().trim();
        writer.write(head+"\tprotein\tdecoy\n");
        while((line = psmReader.readLine())!=null){
            line = line.trim();
            String []d = line.split("\t");
            if(hIndex.containsKey("peptide")) {
                String[] proIDs = pep2proteins.get(d[hIndex.get("peptide")]).split(";");
                String decoy = "Yes";
                for (String pro : proIDs) {
                    if (!pro.startsWith(CParameter.decoy_prefix)) {
                        decoy = "No";
                        break;
                    }
                }
                writer.write(line + "\t" + pep2proteins.get(d[hIndex.get("peptide")]) + "\t" + decoy + "\n");
            }else if(hIndex.containsKey("sequence")) {
                String[] proIDs = pep2proteins.get(d[hIndex.get("sequence")]).split(";");
                String decoy = "Yes";
                for (String pro : proIDs) {
                    if (!pro.startsWith(CParameter.decoy_prefix)) {
                        decoy = "No";
                        break;
                    }
                }
                writer.write(line + "\t" + pep2proteins.get(d[hIndex.get("sequence")]) + "\t" + decoy + "\n");
            }else{
                System.err.println("Please provide peptide column name in the psm file. The column name should be either \"peptide\" or \"sequence\".");
                System.exit(1);
            }
        }
        psmReader.close();
        writer.close();

    }

    public static Enzyme getEnzymeByIndex(int ind){

        ArrayList<Enzyme> enzymes = new ArrayList<>();
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

        if(ind < 0 || ind > enzymes.size()){
            System.err.println("Please provide a valid enzyme number:"+ind);
            System.exit(0);
        }
        Cloger.getInstance().logger.info("Use enzyme:"+enzymes.get(ind).getName());

        return(enzymes.get(ind));
    }

}
