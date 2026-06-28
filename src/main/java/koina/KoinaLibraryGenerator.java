package main.java.koina;

import main.java.db.EntrapmentFastaGear;
import main.java.util.Cloger;
import net.sf.jfasta.FASTAElement;
import net.sf.jfasta.FASTAFileReader;
import net.sf.jfasta.impl.FASTAElementIterator;
import net.sf.jfasta.impl.FASTAFileReaderImpl;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Generates an initial (non-finetuned) DIA-NN-format spectral library for the Osprey
 * workflows by querying the <a href="https://koina.wilhelmlab.org">Koina</a> service for
 * fragment intensities and iRT, instead of running the local AlphaPepDeep model.
 *
 * <p>Input is the peptide-level (target+decoy) FASTA produced by {@link EntrapmentFastaGear}.
 * Each entry is one peptide; modifications matching the configured GUI settings (fixed
 * Carbamidomethyl C, variable Oxidation M up to {@code maxVarMods}) are enumerated, precursors
 * are formed at the requested charges, and the peptidoforms are sent to Koina in ProForma/UNIMOD
 * notation. The result is written as a DIA-NN TSV (same columns Carafe's local library writer
 * uses) that Osprey reads directly; the {@code Decoy}/{@code ProteinID} columns come from
 * the FASTA headers.</p>
 *
 * <p><b>Supported modifications:</b> fixed Carbamidomethyl (C) and variable Oxidation (M), which
 * are the Carafe defaults. The structure extends to more mods, but other PTMs are not yet mapped
 * to UNIMOD here.</p>
 */
public class KoinaLibraryGenerator {

    private static final double PROTON = 1.007276;
    private static final double CARBAMIDOMETHYL_MASS = 57.021464; // UniMod:4 (C)
    private static final double OXIDATION_MASS = 15.994915;       // UniMod:35 (M)

    /** Koina fragment annotation, e.g. {@code y3+1}, {@code b2+2}, {@code y4-H2O+1}. */
    private static final Pattern ANNOTATION =
            Pattern.compile("^([abcxyz])(\\d+)(?:-([A-Za-z0-9]+))?\\+(\\d+)$");

    public static class Config {
        public String peptideFasta;
        public String outputTsv;
        public String koinaUrl = "https://koina.wilhelmlab.org";
        public String ms2Model;            // e.g. Prosit_2020_intensity_HCD
        public String rtModel;             // e.g. Prosit_2019_irt
        public int minCharge = 2;
        public int maxCharge = 3;
        public float nce = 27f;
        public String instrument = "QE";   // Koina instrument code (AlphaPepDeep models)
        public boolean fixedCarbamidomethylC = true;
        public boolean variableOxidationM = true;
        public int maxVarMods = 1;
        public int topNFragments = 20;
        public double minFragmentMz = 200;
        public double minPrecursorMz = 400;
        public double maxPrecursorMz = 900;
        public int batchSize = 1000;
    }

    /** A modified peptide form: stripped sequence, Koina ProForma, DIA-NN UniMod string, mass. */
    private static final class Peptidoform {
        final String stripped;
        final String proforma;
        final String diann;
        final double neutralMass;

        Peptidoform(String stripped, String proforma, String diann, double neutralMass) {
            this.stripped = stripped;
            this.proforma = proforma;
            this.diann = diann;
            this.neutralMass = neutralMass;
        }
    }

    /** A precursor row to predict + write. */
    private static final class Precursor {
        final Peptidoform form;
        final int charge;
        final double mz;
        final String proteinId;
        final int decoy;

        Precursor(Peptidoform form, int charge, double mz, String proteinId, int decoy) {
            this.form = form;
            this.charge = charge;
            this.mz = mz;
            this.proteinId = proteinId;
            this.decoy = decoy;
        }
    }

    public static void run(Config cfg) throws IOException, InterruptedException {
        run(cfg, new KoinaClient(cfg.koinaUrl));
    }

    /**
     * Run with an injected {@link KoinaClient} so the full FASTA -&gt; enumerate -&gt; infer -&gt;
     * write-TSV pipeline can be exercised without a live Koina server (the no-arg overload builds a
     * real client). Package-private for tests.
     */
    static void run(Config cfg, KoinaClient client) throws IOException, InterruptedException {
        Cloger.getInstance().logger.info("Koina library generation: ms2=" + cfg.ms2Model
                + ", rt=" + cfg.rtModel + ", url=" + cfg.koinaUrl);

        // 1. Read peptide FASTA -> precursors.
        List<Precursor> precursors = new ArrayList<>();
        File fasta = new File(cfg.peptideFasta);
        FASTAFileReader reader = new FASTAFileReaderImpl(fasta);
        try {
            FASTAElementIterator it = reader.getIterator();
            while (it.hasNext()) {
                FASTAElement el = it.next();
                el.setLineLength(1);
                String header = el.getHeader();
                String seq = el.getSequence().replaceAll("\\*$", "").replaceAll("^\\*", "").toUpperCase();
                if (seq.isEmpty()) {
                    continue;
                }
                String firstToken = header.trim().split("\\s+")[0];
                if (firstToken.startsWith(">")) {
                    firstToken = firstToken.substring(1);
                }
                int decoy = (firstToken.toLowerCase().startsWith("decoy_")
                        || firstToken.toLowerCase().startsWith("rev_")) ? 1 : 0;
                String proteinId = firstToken;

                for (Peptidoform form : generatePeptidoforms(seq, cfg)) {
                    for (int z = cfg.minCharge; z <= cfg.maxCharge; z++) {
                        double mz = (form.neutralMass + z * PROTON) / z;
                        if (mz < cfg.minPrecursorMz || mz > cfg.maxPrecursorMz) {
                            continue;
                        }
                        precursors.add(new Precursor(form, z, mz, proteinId, decoy));
                    }
                }
            }
        } finally {
            reader.close();
        }
        Cloger.getInstance().logger.info("Koina: built " + precursors.size() + " precursors from "
                + cfg.peptideFasta);
        if (precursors.isEmpty()) {
            throw new IOException("No precursors generated from " + cfg.peptideFasta);
        }

        // 2. Resolve model inputs once.
        Set<String> ms2Inputs = client.getModelInputNames(cfg.ms2Model);
        Set<String> rtInputs = client.getModelInputNames(cfg.rtModel);

        // 3. iRT per unique peptidoform (charge-independent).
        Map<String, Float> proforma2irt = predictRt(client, cfg, precursors, rtInputs);

        // 4. MS2 per precursor, batched; write rows as we go.
        File out = new File(cfg.outputTsv);
        if (out.getParentFile() != null) {
            out.getParentFile().mkdirs();
        }
        int written = 0;
        try (BufferedWriter w = Files.newBufferedWriter(out.toPath(), StandardCharsets.UTF_8)) {
            w.write("ModifiedPeptide\tStrippedPeptide\tPrecursorMz\tPrecursorCharge\tTr_recalibrated\t"
                    + "ProteinID\tDecoy\tFragmentMz\tRelativeIntensity\tFragmentType\tFragmentNumber\t"
                    + "FragmentCharge\tFragmentLossType\n");

            for (int start = 0; start < precursors.size(); start += cfg.batchSize) {
                int end = Math.min(start + cfg.batchSize, precursors.size());
                List<Precursor> batch = precursors.subList(start, end);
                List<String> seqs = new ArrayList<>(batch.size());
                List<Integer> charges = new ArrayList<>(batch.size());
                List<Float> ces = new ArrayList<>(batch.size());
                List<String> instruments = new ArrayList<>(batch.size());
                for (Precursor p : batch) {
                    seqs.add(p.form.proforma);
                    charges.add(p.charge);
                    ces.add(cfg.nce);
                    instruments.add(cfg.instrument);
                }
                List<KoinaClient.Ms2> preds = client.inferMs2(cfg.ms2Model, seqs, charges, ces,
                        instruments, ms2Inputs);
                for (int i = 0; i < batch.size(); i++) {
                    written += writePrecursor(w, batch.get(i), preds.get(i),
                            proforma2irt.getOrDefault(batch.get(i).form.proforma, 0f), cfg);
                }
                Cloger.getInstance().logger.info("Koina: predicted " + end + "/" + precursors.size()
                        + " precursors");
            }
        }
        Cloger.getInstance().logger.info("Koina: wrote " + written + " fragment rows to " + cfg.outputTsv);
    }

    /** Predict iRT for each unique peptidoform (RT is charge-independent), in batches. */
    private static Map<String, Float> predictRt(KoinaClient client, Config cfg,
            List<Precursor> precursors, Set<String> rtInputs) throws IOException, InterruptedException {
        // Unique ProForma sequences.
        List<String> unique = new ArrayList<>();
        Map<String, Float> map = new LinkedHashMap<>();
        for (Precursor p : precursors) {
            if (!map.containsKey(p.form.proforma)) {
                map.put(p.form.proforma, 0f);
                unique.add(p.form.proforma);
            }
        }
        for (int start = 0; start < unique.size(); start += cfg.batchSize) {
            int end = Math.min(start + cfg.batchSize, unique.size());
            List<String> batch = unique.subList(start, end);
            float[] irt = client.inferRt(cfg.rtModel, batch, rtInputs);
            for (int i = 0; i < batch.size(); i++) {
                map.put(batch.get(i), irt[i]);
            }
        }
        return map;
    }

    /** Write the top-N fragment rows for one precursor; returns the number of rows written. */
    private static int writePrecursor(BufferedWriter w, Precursor p, KoinaClient.Ms2 pred, float irt,
            Config cfg) throws IOException {
        // Collect valid fragments.
        List<float[]> frags = new ArrayList<>(); // [mz, intensity, type, number, charge, loss-index]
        List<String[]> fragMeta = new ArrayList<>(); // [type, lossType]
        float maxInt = 0f;
        for (int j = 0; j < pred.annotation.length; j++) {
            float intensity = pred.intensity[j];
            float mz = pred.mz[j];
            if (intensity <= 0 || mz <= 0 || mz < cfg.minFragmentMz) {
                continue;
            }
            Matcher m = ANNOTATION.matcher(pred.annotation[j]);
            if (!m.matches()) {
                continue;
            }
            String type = m.group(1);
            int number = Integer.parseInt(m.group(2));
            String loss = m.group(3);
            int fcharge = Integer.parseInt(m.group(4));
            frags.add(new float[] { mz, intensity, number, fcharge });
            fragMeta.add(new String[] { type, loss == null ? "noloss" : loss });
            maxInt = Math.max(maxInt, intensity);
        }
        if (frags.isEmpty() || maxInt <= 0) {
            return 0;
        }
        // Sort by intensity desc and take top-N.
        Integer[] order = new Integer[frags.size()];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, (a, b) -> Float.compare(frags.get(b)[1], frags.get(a)[1]));
        int limit = Math.min(cfg.topNFragments, order.length);

        int written = 0;
        for (int oi = 0; oi < limit; oi++) {
            int idx = order[oi];
            float[] f = frags.get(idx);
            String[] meta = fragMeta.get(idx);
            float relInt = f[1] / maxInt;
            w.write(p.form.diann + "\t" + p.form.stripped + "\t"
                    + String.format(Locale.ROOT, "%.5f", p.mz) + "\t" + p.charge + "\t"
                    + String.format(Locale.ROOT, "%.4f", irt) + "\t" + p.proteinId + "\t" + p.decoy + "\t"
                    + String.format(Locale.ROOT, "%.5f", f[0]) + "\t" + String.format(Locale.ROOT, "%.6f", relInt) + "\t"
                    + meta[0] + "\t" + (int) f[2] + "\t" + (int) f[3] + "\t" + meta[1] + "\n");
            written++;
        }
        return written;
    }

    /**
     * Enumerate peptidoforms for a peptide: fixed Carbamidomethyl on every C, plus 0..maxVarMods
     * Oxidations on M. Returns at least the all-fixed form. Null mass (unknown residue) -> empty.
     */
    private static List<Peptidoform> generatePeptidoforms(String pep, Config cfg) {
        List<Peptidoform> out = new ArrayList<>();
        Double base = EntrapmentFastaGear.peptideNeutralMass(pep);
        if (base == null) {
            return out;
        }
        // Fixed Carbamidomethyl C positions (always modified).
        List<Integer> cPos = new ArrayList<>();
        List<Integer> mPos = new ArrayList<>();
        for (int i = 0; i < pep.length(); i++) {
            char c = pep.charAt(i);
            if (c == 'C') {
                cPos.add(i);
            } else if (c == 'M') {
                mPos.add(i);
            }
        }
        boolean fixC = cfg.fixedCarbamidomethylC;
        int nFixedC = fixC ? cPos.size() : 0;
        double fixedMass = base + nFixedC * CARBAMIDOMETHYL_MASS;

        // Variable Oxidation subsets of M positions, size 0..maxVarMods.
        List<List<Integer>> oxSubsets = new ArrayList<>();
        oxSubsets.add(new ArrayList<>()); // no oxidation
        if (cfg.variableOxidationM && !mPos.isEmpty()) {
            int maxVar = Math.min(cfg.maxVarMods, mPos.size());
            for (int k = 1; k <= maxVar; k++) {
                combinations(mPos, k, 0, new ArrayList<>(), oxSubsets);
            }
        }

        for (List<Integer> oxSet : oxSubsets) {
            // mod position -> unimod id; build proforma + diann.
            Map<Integer, Integer> pos2unimod = new LinkedHashMap<>();
            if (fixC) {
                for (int p : cPos) {
                    pos2unimod.put(p, 4);
                }
            }
            for (int p : oxSet) {
                pos2unimod.put(p, 35);
            }
            double mass = fixedMass + oxSet.size() * OXIDATION_MASS;
            StringBuilder pf = new StringBuilder();
            StringBuilder dn = new StringBuilder();
            for (int i = 0; i < pep.length(); i++) {
                pf.append(pep.charAt(i));
                dn.append(pep.charAt(i));
                Integer u = pos2unimod.get(i);
                if (u != null) {
                    pf.append("[UNIMOD:").append(u).append("]");
                    dn.append("(UniMod:").append(u).append(")");
                }
            }
            out.add(new Peptidoform(pep, pf.toString(), dn.toString(), mass));
        }
        return out;
    }

    private static void combinations(List<Integer> items, int k, int startIdx, List<Integer> current,
            List<List<Integer>> out) {
        if (current.size() == k) {
            out.add(new ArrayList<>(current));
            return;
        }
        for (int i = startIdx; i < items.size(); i++) {
            current.add(items.get(i));
            combinations(items, k, i + 1, current, out);
            current.remove(current.size() - 1);
        }
    }
}
