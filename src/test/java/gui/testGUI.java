package test.java.gui;

import org.apache.tools.ant.types.Commandline;
import org.testng.annotations.Test;

public class testGUI {

    @Test
    void testTextInputCommandLineOptions() {
        String userInput = "--lib \"d:/test output/\" --threads 20 --verbose 1 --qvalue 0.01 --matrices  --gen-spec-lib --predictor --fasta-search --met-excision --min-pep-len 7 --max-pep-len 30 --min-pr-mz 300 --max-pr-mz 1800 --min-pr-charge 1 --max-pr-charge 4 --min-fr-mz 200 --max-fr-mz 1800 --cut \"K*,R*\" --missed-cleavages 1 --unimod4 --var-mods 1 --var-mod UniMod:21,79.966331,STY --peptidoforms --reanalyse --rt-profiling";
        String[] argv = Commandline.translateCommandline(userInput);
        // print argv
        for (String arg : argv) {
            System.out.println(arg);
        }

        System.out.println("!------------------!");
        userInput = "!--met-excision --min-pep-len 7";
        argv = Commandline.translateCommandline(userInput);
        // print argv
        for (String arg : argv) {
            System.out.println(arg);
        }
    }

}
