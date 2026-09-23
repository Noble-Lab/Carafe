package test.java.ai;

import main.java.ai.FileIO;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

/**
 * The mzML-to-MGF step of the generic DDA route ({@code -se generic -data_type DDA -ms <mzML>}).
 * <p>
 * It used to label each spectrum with its position in the file rather than its scan number. The two
 * agree only when the mzML holds every scan; an mzML converted with an MS-level filter (MS3 dropped
 * from an SPS-MS3 TMT run, say) has gaps, and past the first gap every PSM was silently paired with
 * a different spectrum. On a real TMT run that left 1,647 of 5,337 PSMs matched, to the wrong
 * spectra (base-model PCC 0.15 instead of 0.75).
 */
public class GenericDdaMgfTest {

    private static String floats(double... values) {
        ByteBuffer b = ByteBuffer.allocate(values.length * 8).order(ByteOrder.LITTLE_ENDIAN);
        for (double v : values) {
            b.putDouble(v);
        }
        return Base64.getEncoder().encodeToString(b.array());
    }

    private static String spectrum(int index, int scan, double precursorMz, int charge, double fragment) {
        String mz = floats(fragment, fragment + 100.0);
        String inten = floats(1000.0, 500.0);
        return "<spectrum index=\"" + index + "\" id=\"controllerType=0 controllerNumber=1 scan=" + scan
                + "\" defaultArrayLength=\"2\">\n"
                + "<cvParam cvRef=\"MS\" accession=\"MS:1000511\" name=\"ms level\" value=\"2\"/>\n"
                + "<cvParam cvRef=\"MS\" accession=\"MS:1000580\" name=\"MSn spectrum\" value=\"\"/>\n"
                + "<cvParam cvRef=\"MS\" accession=\"MS:1000127\" name=\"centroid spectrum\" value=\"\"/>\n"
                + "<scanList count=\"1\"><cvParam cvRef=\"MS\" accession=\"MS:1000795\" name=\"no combination\" value=\"\"/>\n"
                + "<scan><cvParam cvRef=\"MS\" accession=\"MS:1000016\" name=\"scan start time\" value=\"" + (scan / 10.0)
                + "\" unitCvRef=\"UO\" unitAccession=\"UO:0000031\" unitName=\"minute\"/></scan></scanList>\n"
                + "<precursorList count=\"1\"><precursor><selectedIonList count=\"1\"><selectedIon>\n"
                + "<cvParam cvRef=\"MS\" accession=\"MS:1000744\" name=\"selected ion m/z\" value=\"" + precursorMz
                + "\" unitCvRef=\"MS\" unitAccession=\"MS:1000040\" unitName=\"m/z\"/>\n"
                + "<cvParam cvRef=\"MS\" accession=\"MS:1000041\" name=\"charge state\" value=\"" + charge + "\"/>\n"
                + "</selectedIon></selectedIonList><activation>\n"
                + "<cvParam cvRef=\"MS\" accession=\"MS:1000133\" name=\"collision-induced dissociation\" value=\"\"/>\n"
                + "</activation></precursor></precursorList>\n"
                + "<binaryDataArrayList count=\"2\">\n"
                + "<binaryDataArray encodedLength=\"" + mz.length() + "\">"
                + "<cvParam cvRef=\"MS\" accession=\"MS:1000523\" name=\"64-bit float\" value=\"\"/>"
                + "<cvParam cvRef=\"MS\" accession=\"MS:1000576\" name=\"no compression\" value=\"\"/>"
                + "<cvParam cvRef=\"MS\" accession=\"MS:1000514\" name=\"m/z array\" value=\"\" unitCvRef=\"MS\" unitAccession=\"MS:1000040\" unitName=\"m/z\"/>"
                + "<binary>" + mz + "</binary></binaryDataArray>\n"
                + "<binaryDataArray encodedLength=\"" + inten.length() + "\">"
                + "<cvParam cvRef=\"MS\" accession=\"MS:1000523\" name=\"64-bit float\" value=\"\"/>"
                + "<cvParam cvRef=\"MS\" accession=\"MS:1000576\" name=\"no compression\" value=\"\"/>"
                + "<cvParam cvRef=\"MS\" accession=\"MS:1000515\" name=\"intensity array\" value=\"\" unitCvRef=\"MS\" unitAccession=\"MS:1000131\" unitName=\"number of counts\"/>"
                + "<binary>" + inten + "</binary></binaryDataArray>\n"
                + "</binaryDataArrayList></spectrum>\n";
    }

    /**
     * Without -nce, the generic DDA route takes the NCE from the PSM table. It used to stay at the
     * default of 27, so the library was predicted at an NCE the model had not been trained on.
     */
    @Test
    public void testMostCommonNce() throws Exception {
        java.lang.reflect.Method m = main.java.ai.AIGear.class.getDeclaredMethod(
                "most_common_nce", List.class, int.class);
        m.setAccessible(true);
        Assert.assertEquals(m.invoke(null, List.of("A\t2\t34", "B\t2\t34", "C\t3\t30"), 2), 34.0);
        // Ties go to the lower value, independent of row order.
        Assert.assertEquals(m.invoke(null, List.of("A\t2\t34", "C\t3\t30"), 2), 30.0);
        Assert.assertEquals(m.invoke(null, List.of("C\t3\t30", "A\t2\t34"), 2), 30.0);
        // Blank and non-numeric cells are ignored; nothing usable gives null (keep the default).
        Assert.assertEquals(m.invoke(null, List.of("A\t2\t", "B\t2\tx", "C\t3\t32"), 2), 32.0);
        Assert.assertNull(m.invoke(null, List.of("A\t2\t", "B\t2"), 2));
    }

    /**
     * The RT maximum for an MGF comes from the table's rt column. It used to stay 0, which made every
     * RT training target rt / 0 = Infinity.
     */
    @Test
    public void testMaxColumnValue() throws Exception {
        java.lang.reflect.Method m = main.java.ai.AIGear.class.getDeclaredMethod(
                "max_column_value", List.class, int.class);
        m.setAccessible(true);
        Assert.assertEquals(m.invoke(null, List.of("A\t33.9", "B\t61.4", "C\t48.6"), 1), 61.4);
        Assert.assertEquals(m.invoke(null, List.of("A\t", "B\tx", "C\t12.5"), 1), 12.5);
        Assert.assertNull(m.invoke(null, List.of("A\t", "B"), 1));
    }

    @Test
    public void testScanNumbersNotPositionsAndPsmCharge() throws Exception {
        File dir = Files.createTempDirectory("carafe_generic_dda").toFile();
        dir.deleteOnExit();

        // Native scans 1, 5 and 9 at positions 1, 2 and 3, as after an MS-level filter.
        String mzml = "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
                + "<mzML xmlns=\"http://psi.hupo.org/ms/mzml\" version=\"1.1.0\">\n"
                + "<cvList count=\"2\"><cv id=\"MS\" fullName=\"PSI-MS\" URI=\"https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo\"/>"
                + "<cv id=\"UO\" fullName=\"Unit Ontology\" URI=\"http://ontologies.berkeleybop.org/uo.obo\"/></cvList>\n"
                + "<fileDescription><fileContent>"
                + "<cvParam cvRef=\"MS\" accession=\"MS:1000580\" name=\"MSn spectrum\" value=\"\"/>"
                + "</fileContent></fileDescription>\n"
                + "<softwareList count=\"1\"><software id=\"sw\" version=\"1\">"
                + "<cvParam cvRef=\"MS\" accession=\"MS:1000799\" name=\"custom unreleased software tool\" value=\"test\"/>"
                + "</software></softwareList>\n"
                + "<instrumentConfigurationList count=\"1\"><instrumentConfiguration id=\"IC1\">"
                + "<cvParam cvRef=\"MS\" accession=\"MS:1000031\" name=\"instrument model\" value=\"\"/>"
                + "</instrumentConfiguration></instrumentConfigurationList>\n"
                + "<dataProcessingList count=\"1\"><dataProcessing id=\"dp\">"
                + "<processingMethod order=\"0\" softwareRef=\"sw\">"
                + "<cvParam cvRef=\"MS\" accession=\"MS:1000544\" name=\"Conversion to mzML\" value=\"\"/>"
                + "</processingMethod></dataProcessing></dataProcessingList>\n"
                + "<run id=\"fixture\" defaultInstrumentConfigurationRef=\"IC1\">"
                + "<spectrumList count=\"3\" defaultDataProcessingRef=\"dp\">\n"
                + spectrum(0, 1, 400.0, 2, 200.0)
                + spectrum(1, 5, 500.0, 2, 300.0)
                + spectrum(2, 9, 600.0, 3, 400.0) // the mzML says 3+
                + "</spectrumList></run></mzML>\n";
        File mzmlFile = new File(dir, "fixture.mzML");
        Files.write(mzmlFile.toPath(), mzml.getBytes(StandardCharsets.UTF_8));

        File psm = new File(dir, "psms.tsv");
        Files.write(psm.toPath(), ("peptide\tmodification\tcharge\tspectrum_title\n"
                + "PEPTIDEK\t-\t2\tfixture:9:2\n"      // the search engine assigned 2+
                + "PEPTIDER\t-\t2\tfixture:5:2\n").getBytes(StandardCharsets.UTF_8));

        File mgf = new File(dir, "out.mgf");
        FileIO.generate_mgf_for_PSMs(psm.getAbsolutePath(), mzmlFile.getAbsolutePath(), mgf.getAbsolutePath());

        List<String> titles = new ArrayList<>();
        List<String> pepmass = new ArrayList<>();
        for (String line : Files.readAllLines(mgf.toPath())) {
            if (line.startsWith("TITLE=")) titles.add(line.substring(6));
            if (line.startsWith("PEPMASS=")) pepmass.add(line.substring(8).split(" ")[0]);
        }
        Assert.assertEquals(titles.size(), 2, "both PSMs' spectra must be found: " + titles);
        int i5 = titles.indexOf("fixture:5:2");
        int i9 = titles.indexOf("fixture:9:2");
        Assert.assertTrue(i5 >= 0 && i9 >= 0, "titles use native scan numbers and the PSM charge: " + titles);
        // Each title is attached to the spectrum with that scan number, not to the one at that position.
        Assert.assertEquals(Double.parseDouble(pepmass.get(i5)), 500.0, 1e-6);
        Assert.assertEquals(Double.parseDouble(pepmass.get(i9)), 600.0, 1e-6);
    }
}
