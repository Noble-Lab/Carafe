package main.java.gui;

import java.security.cert.CertPath;
import java.util.Arrays;

import main.java.ai.AIGear;
import main.java.input.CParameter;
import main.java.util.GenericUtils;

/**
 * CarafeLauncher - Unified entry point for Carafe
 * 
 * This launcher allows users to start either the GUI or CLI version of Carafe.
 * - Without arguments or with --gui: Launches the graphical user interface
 * - With command line arguments: Runs in command line mode (original AIGear behavior)
 */
public class CarafeLauncher {

    private static final String BANNER = """
            AI-Powered Spectral Library Generator for DIA Proteomics
            Version: %s
            
            """.formatted(CParameter.getVersion());
    
    // Static initializer - runs before main() and before any Hadoop classes load
    static {
        // Apply Java 24+ Hadoop fix BEFORE any Hadoop classes are loaded
        // This is critical because AIGear imports Hadoop/Parquet classes
        GenericUtils.applyJava25HadoopFix();
    }

    public static void main(String[] args) {
        // Check for GUI flag or no arguments
        if (args.length == 0 || containsGuiFlag(args)) {
            launchGUI(filterGuiFlags(args));
        } else if (containsHelpFlag(args) && args.length == 1) {
            printUsage();
        } else {
            // Launch CLI mode
            launchCLI(args);
        }
    }

    private static boolean containsGuiFlag(String[] args) {
        for (String arg : args) {
            if (arg.equalsIgnoreCase("--gui") || arg.equalsIgnoreCase("-gui")) {
                return true;
            }
        }
        return false;
    }

    private static boolean containsHelpFlag(String[] args) {
        for (String arg : args) {
            if (arg.equalsIgnoreCase("-h") || arg.equalsIgnoreCase("--help") || arg.equalsIgnoreCase("-help")) {
                return true;
            }
        }
        return false;
    }

    private static String[] filterGuiFlags(String[] args) {
        return Arrays.stream(args)
                .filter(arg -> !arg.equalsIgnoreCase("--gui") && !arg.equalsIgnoreCase("-gui"))
                .toArray(String[]::new);
    }

    private static void launchGUI(String[] args) {
        System.out.println(BANNER);
        System.out.println("Launching Carafe GUI...\n");
        CarafeGUI.main(args);
    }

    private static void launchCLI(String[] args) {
        System.out.println(BANNER);
        try {
            AIGear.main(args);
        } catch (Exception e) {
            System.err.println("Error running Carafe: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static void printUsage() {
        System.out.println(BANNER);
        System.out.println("""
            Usage: java -jar carafe.jar [options]
            
            Launch Modes:
              --gui, -gui          Launch the graphical user interface
              (no arguments)       Launch the graphical user interface
              (with CLI options)   Run in command line mode
            
            For CLI options, use: java -jar carafe.jar -h
            
            Examples:
              java -jar carafe.jar                    # Launch GUI
              java -jar carafe.jar --gui              # Launch GUI
              java -jar carafe.jar -db protein.fasta  # Run CLI mode
            
            For more information, visit: https://github.com/Noble-Lab/Carafe
            """);
    }
}
