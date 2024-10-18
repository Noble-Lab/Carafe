package main.java.ai;

import com.compomics.util.experiment.io.biology.protein.SequenceProvider;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;

public class JSequenceProvider implements SequenceProvider {

    private HashMap<String, String> acc2seq = new HashMap<>();

    @Override
    public Collection<String> getAccessions() {
        return acc2seq.keySet();
    }

    @Override
    public HashSet<String> getDecoyAccessions() {
        HashSet<String> dacc = new HashSet<>();
        return dacc;
    }

    @Override
    public String getSequence(String proteinAccession) {
        return this.acc2seq.get(proteinAccession);
    }

    @Override
    public String getSubsequence(String accession, int start, int end) {
        return "";
    }

    @Override
    public String getHeaderAsString(String proteinAccession) {
        return null;
    }

    public String getHeader(String proteinAccession) {
        return "";
    }

    public void addProtein(String acc, String seq){
        this.acc2seq.put(acc,seq);
    }
}
