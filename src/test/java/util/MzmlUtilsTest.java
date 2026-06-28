package test.java.util;

import main.java.util.MzmlUtils;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Tests for {@link MzmlUtils#readNce(String)} using small synthetic mzML files.
 */
public class MzmlUtilsTest {

    private Path writeMzml(String body) throws Exception {
        Path p = Files.createTempFile("test", ".mzML");
        Files.writeString(p, body, StandardCharsets.UTF_8);
        return p;
    }

    @Test
    public void readsFirstCollisionEnergyByAccession() throws Exception {
        // An MS1 spectrum (no CE) followed by an MS2 with collision energy 28.0.
        String mzml = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                + "<mzML><run><spectrumList count=\"2\">"
                + "<spectrum index=\"0\"><cvParam accession=\"MS:1000511\" name=\"ms level\" value=\"1\"/></spectrum>"
                + "<spectrum index=\"1\"><cvParam accession=\"MS:1000511\" name=\"ms level\" value=\"2\"/>"
                + "<precursorList><precursor><activation>"
                + "<cvParam accession=\"MS:1000045\" name=\"collision energy\" value=\"28.0\" unitName=\"electronvolt\"/>"
                + "</activation></precursor></precursorList></spectrum>"
                + "</spectrumList></run></mzML>";
        Path p = writeMzml(mzml);
        Assert.assertEquals(MzmlUtils.readNce(p.toString()), 28.0, 1e-9);
    }

    @Test
    public void readsCollisionEnergyByNameWhenAccessionDiffers() throws Exception {
        String mzml = "<mzML><run><spectrumList>"
                + "<spectrum><cvParam name=\"collision energy\" value=\"30.5\"/></spectrum>"
                + "</spectrumList></run></mzML>";
        Path p = writeMzml(mzml);
        Assert.assertEquals(MzmlUtils.readNce(p.toString()), 30.5, 1e-9);
    }

    @Test
    public void returnsMinusOneWhenNoCollisionEnergy() throws Exception {
        String mzml = "<mzML><run><spectrumList>"
                + "<spectrum><cvParam accession=\"MS:1000511\" name=\"ms level\" value=\"2\"/></spectrum>"
                + "</spectrumList></run></mzML>";
        Path p = writeMzml(mzml);
        Assert.assertEquals(MzmlUtils.readNce(p.toString()), -1.0, 1e-9);
    }

    @Test
    public void returnsMinusOneForMissingFile() {
        Assert.assertEquals(MzmlUtils.readNce("/no/such/file.mzML"), -1.0, 1e-9);
    }
}
