package test.java.gui;

import main.java.gui.CarafeGUI;
import main.java.input.CParameter;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

import javax.swing.JComboBox;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import java.awt.GraphicsEnvironment;
import java.io.File;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Workflow 5, "Spectral library generation: start with DDA report": trains from a DDA PSM table in
 * Carafe's generic format ({@code -se generic -data_type DDA}) and the mzML files it was searched on.
 */
public class DdaReportWorkflowTest {

    private static String problem(File f) throws Exception {
        Method m = CarafeGUI.class.getDeclaredMethod("ddaReportProblem", File.class);
        m.setAccessible(true);
        return (String) m.invoke(null, f);
    }

    private static String analyzer(File f) throws Exception {
        Method m = CarafeGUI.class.getDeclaredMethod("ms2Analyzer", File.class);
        m.setAccessible(true);
        return (String) m.invoke(null, f);
    }

    private static File write(String name, String suffix, String content) throws Exception {
        File f = Files.createTempFile("carafe_" + name, suffix).toFile();
        f.deleteOnExit();
        Files.write(f.toPath(), content.getBytes(StandardCharsets.UTF_8));
        return f;
    }

    private static Object field(CarafeGUI gui, String name) throws Exception {
        Field f = CarafeGUI.class.getDeclaredField(name);
        f.setAccessible(true);
        return f.get(gui);
    }

    @Test
    public void testDdaReportHeaderCheck() throws Exception {
        Assert.assertNull(problem(write("ok", ".tsv",
                "peptide\tmodification\tcharge\tspectrum_title\tscore\trt\tnce\nAAK\t-\t2\tf:1:2\t1\t2\t34\n")));
        // A byte order mark must not hide the first column.
        Assert.assertNull(problem(write("bom", ".tsv", "\uFEFFpeptide\tmodification\tcharge\tspectrum_title\n")));

        // A search engine's own output chosen instead of the converted table (Sage's header).
        String sage = problem(write("sage", ".tsv", "psm_id\tpeptide\tproteins\tfilename\tscannr\tcharge\n"));
        Assert.assertNotNull(sage);
        Assert.assertTrue(sage.contains("modification") && sage.contains("spectrum_title"), sage);

        Assert.assertNotNull(problem(write("empty", ".tsv", "")));
        Assert.assertNotNull(problem(new File("Z:/nope/x_" + System.nanoTime() + ".tsv")));
    }

    private static String mgfProblem(File mgf, File report) throws Exception {
        Method m = CarafeGUI.class.getDeclaredMethod("mgfTitleProblem", File.class, File.class);
        m.setAccessible(true);
        return (String) m.invoke(null, mgf, report);
    }

    @Test
    public void testMgfTitlesMustMatchTheReport() throws Exception {
        File report = write("rep", ".tsv", "peptide\tmodification\tcharge\tspectrum_title\n"
                + "AAK\t-\t2\trunA:1571:2\nPEPK\t-\t3\trunB:20000:3\n");
        String record = "BEGIN IONS\nTITLE=%s\nPEPMASS=500.0\nCHARGE=2+\n200.1 10\nEND IONS\n";

        // Titles in the report's format, naming a file the report uses.
        Assert.assertNull(mgfProblem(write("ok", ".mgf",
                String.format(record, "runA:1571:2") + String.format(record, "runB:20000:3")), report));

        // A converter's own title scheme: nothing would match.
        String msconvert = mgfProblem(write("msc", ".mgf",
                String.format(record, "runA.1571.1571.2 File:\"runA.raw\", NativeID:\"scan=1571\"")), report);
        Assert.assertNotNull(msconvert);
        Assert.assertTrue(msconvert.contains("runA:1571:2"), "shows the expected form: " + msconvert);

        // Right format, but spectra from a different run.
        Assert.assertNotNull(mgfProblem(write("other", ".mgf", String.format(record, "runC:5:2")), report));

        // Not an MGF at all.
        Assert.assertNotNull(mgfProblem(write("none", ".mgf", "hello\n"), report));
    }

    private static final String ITMS_TRFP =
            "<cvParam cvRef=\"MS\" accession=\"MS:1000512\" value=\"FTMS + c NSI Full ms [400.0000-1600.0000]\" name=\"filter string\" />\n"
            + "<cvParam cvRef=\"MS\" accession=\"MS:1000512\" value=\"ITMS + c NSI t d Full ms2 745.3972@cid34.00 [400.0000-1600.0000]\" name=\"filter string\" />\n";

    @Test
    public void testMs2AnalyzerDetection() throws Exception {
        // ThermoRawFileParser order (value before name). The MS1 line before it must not decide.
        Assert.assertEquals(analyzer(write("trfp", ".mzML", ITMS_TRFP)), "ITMS");
        // msconvert order (name before value).
        Assert.assertEquals(analyzer(write("msconvert", ".mzML",
                "<cvParam cvRef=\"MS\" accession=\"MS:1000512\" name=\"filter string\" "
                        + "value=\"FTMS + p NSI d Full ms2 500.2500@hcd30.00 [110.0000-1500.0000]\"/>\n")), "FTMS");
        // No filter string (e.g. a non-Thermo file): unknown, so no check is applied.
        Assert.assertNull(analyzer(write("none", ".mzML", "<spectrum id=\"index=0\"/>\n")));

        // A folder: the first mzML in it.
        File dir = Files.createTempDirectory("carafe_ms").toFile();
        dir.deleteOnExit();
        File f = new File(dir, "a.mzML");
        Files.write(f.toPath(), ITMS_TRFP.getBytes(StandardCharsets.UTF_8));
        f.deleteOnExit();
        Assert.assertEquals(analyzer(dir), "ITMS");
    }

    @Test
    public void testWorkflowIndicesAndCommand() throws Exception {
        if (GraphicsEnvironment.isHeadless()) {
            throw new SkipException("no display available");
        }
        CParameter.init();
        AtomicReference<CarafeGUI> ref = new AtomicReference<>();
        SwingUtilities.invokeAndWait(() -> ref.set(new CarafeGUI()));
        CarafeGUI gui = ref.get();
        Method build = CarafeGUI.class.getDeclaredMethod("buildCarafeCommand");
        build.setAccessible(true);
        try {
            SwingUtilities.invokeAndWait(() -> {
                try {
                    JComboBox<?> workflow = (JComboBox<?>) field(gui, "workflowCombo");
                    // Saved settings store the workflow by index, and the code compares against fixed
                    // indices, so the existing entries must never move.
                    String[] expected = {
                            "1. Spectral library generation: start with DIA-NN search",
                            "2. Spectral library generation: start with DIA-NN report",
                            "3. End-to-end DIA search",
                            "4. Spectral library generation: predict with a pretrained model (no fine-tuning)",
                            "5. Spectral library generation: start with DDA report"};
                    Assert.assertEquals(workflow.getItemCount(), expected.length);
                    for (int i = 0; i < expected.length; i++) {
                        Assert.assertEquals(workflow.getItemAt(i), expected[i], "workflow index " + i);
                    }

                    ((JTextField) field(gui, "diannReportFileField")).setText("C:/data/report.parquet");
                    ((JTextField) field(gui, "ddaReportFileField")).setText("C:/data/psms.tsv");
                    ((JTextField) field(gui, "trainMsFileField")).setText("C:/data/mzml");

                    workflow.setSelectedIndex(4);
                    Object task = build.invoke(gui);
                    String dda = (String) task.getClass().getField("cmd").get(task);
                    Assert.assertTrue(dda.contains("-i C:/data/psms.tsv"), dda);
                    Assert.assertTrue(dda.contains("-se generic -data_type DDA"), dda);
                    Assert.assertTrue(dda.contains("-ms C:/data/mzml"), dda);
                    Assert.assertTrue(dda.contains("-tf all"), dda);
                    // The hidden DIA-NN report field still holds text; it must not leak in.
                    Assert.assertFalse(dda.contains("report.parquet"), dda);
                    Assert.assertFalse(dda.contains("-se DIA-NN"), dda);

                    // And the other way round: workflow 2 is unchanged.
                    workflow.setSelectedIndex(1);
                    task = build.invoke(gui);
                    String dia = (String) task.getClass().getField("cmd").get(task);
                    Assert.assertTrue(dia.contains("-i C:/data/report.parquet"), dia);
                    Assert.assertTrue(dia.contains("-se DIA-NN"), dia);
                    Assert.assertFalse(dia.contains("psms.tsv"), dia);
                    Assert.assertFalse(dia.contains("-data_type"), dia);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            });
        } finally {
            SwingUtilities.invokeAndWait(gui::dispose);
        }
    }
}
