package test.java.gui;

import org.apache.tools.ant.types.Commandline;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.Arrays;
import java.util.List;

/**
 * Verifies the command-line tokenization used for the GUI's free-form "additional options"
 * fields. Renamed from {@code testGUI} so it matches the Surefire {@code **<!-- -->/*Test.java}
 * include pattern and actually runs under {@code mvn test}.
 */
public class GuiCommandLineTest {

    /** Value token immediately following {@code flag} in {@code argv}, or null if absent. */
    private static String valueAfter(String[] argv, String flag) {
        List<String> args = Arrays.asList(argv);
        int i = args.indexOf(flag);
        return (i >= 0 && i + 1 < args.size()) ? args.get(i + 1) : null;
    }

    @Test
    public void testTextInputCommandLineOptions() {
        // A DIA-NN-style options string as a user would type into the GUI's "additional options"
        // field. Commandline.translateCommandline must split on whitespace while keeping quoted
        // values (with embedded spaces) as a single token and stripping the surrounding quotes.
        String userInput = "--lib \"d:/test output/\" --threads 20 --verbose 1 --qvalue 0.01 --matrices  --gen-spec-lib --predictor --fasta-search --met-excision --min-pep-len 7 --max-pep-len 30 --min-pr-mz 300 --max-pr-mz 1800 --min-pr-charge 1 --max-pr-charge 4 --min-fr-mz 200 --max-fr-mz 1800 --cut \"K*,R*\" --missed-cleavages 1 --unimod4 --var-mods 1 --var-mod UniMod:21,79.966331,STY --peptidoforms --reanalyse --rt-profiling";
        String[] argv = Commandline.translateCommandline(userInput);

        // Quoted value with an embedded space is one token, quotes stripped.
        Assert.assertEquals(valueAfter(argv, "--lib"), "d:/test output/");
        // Quoted enzyme-cut spec stays a single token.
        Assert.assertEquals(valueAfter(argv, "--cut"), "K*,R*");
        // Comma-separated modification spec is not split.
        Assert.assertEquals(valueAfter(argv, "--var-mod"), "UniMod:21,79.966331,STY");
        Assert.assertEquals(valueAfter(argv, "--threads"), "20");
        // Flags without values are still present as tokens.
        Assert.assertTrue(Arrays.asList(argv).contains("--unimod4"));
        Assert.assertTrue(Arrays.asList(argv).contains("--rt-profiling"));

        // A second input beginning with '!' keeps '!' as part of the first token.
        String[] argv2 = Commandline.translateCommandline("!--met-excision --min-pep-len 7");
        Assert.assertEquals(argv2[0], "!--met-excision");
        Assert.assertEquals(valueAfter(argv2, "--min-pep-len"), "7");
    }
}
