package test.java.dia;

import com.compomics.util.experiment.mass_spectrometry.spectra.Spectrum;
import main.java.dia.DIAIndex;
import main.java.dia.DIAMeta;
import main.java.dia.IsolationWindow;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;

/**
 * Tests for the dia/ spectrum layer ({@link DIAMeta} parsing + {@link DIAIndex} scan lookup) on a
 * tiny hand-built mzML written at test time. A real DIA mzML is far too large to commit (an Astral
 * file is ~145 MB for under a minute), so the fixture is generated in-process: a 3-scan run -- one
 * MS1 and two MS2 in adjacent isolation windows, with base64-encoded peak arrays -- which exercises
 * the real MSFTBX parse path used to read acquisition mzMLs in production.
 */
public class DIASpectrumTest {

    private static String b64f64(double[] v) {
        ByteBuffer b = ByteBuffer.allocate(v.length * 8).order(ByteOrder.LITTLE_ENDIAN);
        for (double x : v) b.putDouble(x);
        return Base64.getEncoder().encodeToString(b.array());
    }

    private static String b64f32(double[] v) {
        ByteBuffer b = ByteBuffer.allocate(v.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (double x : v) b.putFloat((float) x);
        return Base64.getEncoder().encodeToString(b.array());
    }

    private static String spectrum(int idx, int msLevel, double rtMin, double[] mz, double[] inten,
            double isoTarget, double isoLo, double isoHi) {
        StringBuilder s = new StringBuilder();
        s.append("<spectrum index=\"").append(idx).append("\" id=\"scan=").append(idx + 1)
                .append("\" defaultArrayLength=\"").append(mz.length).append("\">");
        s.append("<cvParam cvRef=\"MS\" accession=\"MS:1000511\" name=\"ms level\" value=\"")
                .append(msLevel).append("\"/>");
        if (msLevel == 1) {
            s.append("<cvParam cvRef=\"MS\" accession=\"MS:1000579\" name=\"MS1 spectrum\" value=\"\"/>");
        } else {
            s.append("<cvParam cvRef=\"MS\" accession=\"MS:1000580\" name=\"MSn spectrum\" value=\"\"/>");
        }
        s.append("<scanList count=\"1\"><scan>");
        s.append("<cvParam cvRef=\"MS\" accession=\"MS:1000016\" name=\"scan start time\" value=\"")
                .append(rtMin).append("\" unitCvRef=\"UO\" unitAccession=\"UO:0000031\" unitName=\"minute\"/>");
        double scanLo = msLevel == 1 ? 380.0 : 150.0;
        double scanHi = msLevel == 1 ? 980.0 : 2000.0;
        s.append("<scanWindowList count=\"1\"><scanWindow>");
        s.append("<cvParam cvRef=\"MS\" accession=\"MS:1000501\" name=\"scan window lower limit\" value=\"")
                .append(scanLo).append("\" unitCvRef=\"MS\" unitAccession=\"MS:1000040\" unitName=\"m/z\"/>");
        s.append("<cvParam cvRef=\"MS\" accession=\"MS:1000500\" name=\"scan window upper limit\" value=\"")
                .append(scanHi).append("\" unitCvRef=\"MS\" unitAccession=\"MS:1000040\" unitName=\"m/z\"/>");
        s.append("</scanWindow></scanWindowList>");
        s.append("</scan></scanList>");
        if (msLevel == 2) {
            s.append("<precursorList count=\"1\"><precursor><isolationWindow>");
            s.append("<cvParam cvRef=\"MS\" accession=\"MS:1000827\" name=\"isolation window target m/z\" value=\"")
                    .append(isoTarget).append("\"/>");
            s.append("<cvParam cvRef=\"MS\" accession=\"MS:1000828\" name=\"isolation window lower offset\" value=\"")
                    .append(isoTarget - isoLo).append("\"/>");
            s.append("<cvParam cvRef=\"MS\" accession=\"MS:1000829\" name=\"isolation window upper offset\" value=\"")
                    .append(isoHi - isoTarget).append("\"/>");
            s.append("</isolationWindow><selectedIonList count=\"1\"><selectedIon>");
            s.append("<cvParam cvRef=\"MS\" accession=\"MS:1000744\" name=\"selected ion m/z\" value=\"")
                    .append(isoTarget).append("\"/>");
            s.append("</selectedIon></selectedIonList>");
            s.append("<activation><cvParam cvRef=\"MS\" accession=\"MS:1000133\" name=\"collision-induced dissociation\" value=\"\"/></activation>");
            s.append("</precursor></precursorList>");
        }
        String mzB = b64f64(mz);
        String inB = b64f32(inten);
        s.append("<binaryDataArrayList count=\"2\">");
        s.append("<binaryDataArray encodedLength=\"").append(mzB.length()).append("\">");
        s.append("<cvParam cvRef=\"MS\" accession=\"MS:1000523\" name=\"64-bit float\" value=\"\"/>");
        s.append("<cvParam cvRef=\"MS\" accession=\"MS:1000576\" name=\"no compression\" value=\"\"/>");
        s.append("<cvParam cvRef=\"MS\" accession=\"MS:1000514\" name=\"m/z array\" value=\"\" unitCvRef=\"MS\" unitAccession=\"MS:1000040\" unitName=\"m/z\"/>");
        s.append("<binary>").append(mzB).append("</binary></binaryDataArray>");
        s.append("<binaryDataArray encodedLength=\"").append(inB.length()).append("\">");
        s.append("<cvParam cvRef=\"MS\" accession=\"MS:1000521\" name=\"32-bit float\" value=\"\"/>");
        s.append("<cvParam cvRef=\"MS\" accession=\"MS:1000576\" name=\"no compression\" value=\"\"/>");
        s.append("<cvParam cvRef=\"MS\" accession=\"MS:1000515\" name=\"intensity array\" value=\"\" unitCvRef=\"MS\" unitAccession=\"MS:1000131\" unitName=\"number of detector counts\"/>");
        s.append("<binary>").append(inB).append("</binary></binaryDataArray>");
        s.append("</binaryDataArrayList></spectrum>");
        return s.toString();
    }

    private static String buildMzml() {
        double[] mz1 = { 400.1, 500.2, 600.3 };
        double[] in1 = { 1000, 2000, 3000 };
        double[] mz2 = { 150.05, 250.10, 350.15 };
        double[] in2 = { 500, 1500, 2500 };
        StringBuilder s = new StringBuilder();
        s.append("<?xml version=\"1.0\" encoding=\"utf-8\"?>");
        s.append("<mzML xmlns=\"http://psi.hupo.org/ms/mzml\" version=\"1.1.0\">");
        s.append("<cvList count=\"2\">"
                + "<cv id=\"MS\" fullName=\"PSI-MS\" URI=\"http://psidev.info/ms/mzML/psi-ms.obo\"/>"
                + "<cv id=\"UO\" fullName=\"Unit Ontology\" URI=\"http://obo.cvs.sourceforge.net/obo/obo/ontology/phenotype/unit.obo\"/>"
                + "</cvList>");
        s.append("<fileDescription><fileContent>"
                + "<cvParam cvRef=\"MS\" accession=\"MS:1000579\" name=\"MS1 spectrum\" value=\"\"/>"
                + "<cvParam cvRef=\"MS\" accession=\"MS:1000580\" name=\"MSn spectrum\" value=\"\"/>"
                + "</fileContent></fileDescription>");
        s.append("<softwareList count=\"1\"><software id=\"sw\" version=\"1\">"
                + "<cvParam cvRef=\"MS\" accession=\"MS:1000799\" name=\"custom unreleased software tool\" value=\"\"/>"
                + "</software></softwareList>");
        s.append("<instrumentConfigurationList count=\"1\"><instrumentConfiguration id=\"IC1\">"
                + "<cvParam cvRef=\"MS\" accession=\"MS:1000031\" name=\"instrument model\" value=\"\"/>"
                + "</instrumentConfiguration></instrumentConfigurationList>");
        s.append("<dataProcessingList count=\"1\"><dataProcessing id=\"dp\">"
                + "<processingMethod order=\"0\" softwareRef=\"sw\">"
                + "<cvParam cvRef=\"MS\" accession=\"MS:1000544\" name=\"Conversion to mzML\" value=\"\"/>"
                + "</processingMethod></dataProcessing></dataProcessingList>");
        s.append("<run id=\"probe\" defaultInstrumentConfigurationRef=\"IC1\">"
                + "<spectrumList count=\"3\" defaultDataProcessingRef=\"dp\">");
        // 1 MS1, then 2 MS2 in two isolation windows at slightly later RT.
        s.append(spectrum(0, 1, 0.100, mz1, in1, 0, 0, 0));
        s.append(spectrum(1, 2, 0.101, mz2, in2, 702.0, 700.0, 704.0));
        s.append(spectrum(2, 2, 0.102, mz2, in2, 706.0, 704.0, 708.0));
        s.append("</spectrumList></run></mzML>");
        return s.toString();
    }

    private static Path writeMzml() throws IOException {
        Path p = Files.createTempFile("synthetic", ".mzML");
        Files.write(p, buildMzml().getBytes(StandardCharsets.UTF_8));
        return p;
    }

    @Test
    public void diaMetaParsesScansRtAndIsolationWindows() throws IOException {
        DIAMeta meta = new DIAMeta();
        meta.load_ms_data(writeMzml().toString());
        meta.get_ms_run_meta_data();

        Assert.assertEquals(meta.num2scanMap.size(), 3, "one MS1 + two MS2 scans");
        int ms2 = 0;
        for (int scanNum : meta.num2scanMap.keySet()) {
            if (meta.num2scanMap.get(scanNum).getMsLevel() == 2) {
                ms2++;
            }
        }
        Assert.assertEquals(ms2, 2, "two MS2 scans");

        // The two MS2 isolation windows [700,704] and [704,708] are discovered, keyed by id.
        Assert.assertEquals(meta.isolationWindowMap.size(), 2);
        IsolationWindow w = meta.isolationWindowMap.get(IsolationWindow.generate_id(700.0, 704.0));
        Assert.assertNotNull(w, "window 700-704 must be present");
        Assert.assertEquals(w.mz_lower, 700.0, 1e-6);
        Assert.assertEquals(w.mz_upper, 704.0, 1e-6);
        Assert.assertTrue(
                meta.isolationWindowMap.containsKey(IsolationWindow.generate_id(704.0, 708.0)));
    }

    @Test
    public void diaIndexReadsPeaksAndResolvesSpectrumByScan() throws IOException {
        DIAMeta meta = new DIAMeta();
        meta.load_ms_data(writeMzml().toString());
        meta.get_ms_run_meta_data();

        DIAIndex idx = new DIAIndex();
        idx.meta = meta;
        String win = IsolationWindow.generate_id(700.0, 704.0);
        idx.target_isolation_wins.add(win);
        idx.index();

        // Exactly the one MS2 scan in that window is indexed, keyed by its native scan number.
        Assert.assertEquals(idx.scan2spectrum.size(), 1, "one MS2 scan in window " + win);
        int scanNum = idx.scan2spectrum.keySet().iterator().next();
        Spectrum sp = idx.get_spectrum_by_scan(scanNum);
        Assert.assertNotNull(sp, "get_spectrum_by_scan must resolve the indexed MS2 scan");
        Assert.assertEquals(sp.getNPeaks(), 3, "the MS2 spectrum's three peaks must round-trip");
        // An unknown scan number resolves to null -- the upstream "spectrum not found" guard.
        Assert.assertNull(idx.get_spectrum_by_scan(999999));
    }
}
