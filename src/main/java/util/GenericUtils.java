package main.java.util;

import oshi.SystemInfo;
import oshi.hardware.GlobalMemory;

public class GenericUtils {


    /**
     * Gets the major version of the running Java runtime.
     * @return Java major version (e.g., 17, 21, 25)
     */
    public static int getJavaMajorVersion() {
        String version = System.getProperty("java.version");
        if (version.startsWith("1.")) {
            // Java 8 or earlier: "1.8.0_xxx"
            return Integer.parseInt(version.substring(2, 3));
        } else {
            // Java 9+: "9", "11.0.1", "17.0.2", "21", "25.0.1"
            int dotIndex = version.indexOf(".");
            if (dotIndex > 0) {
                return Integer.parseInt(version.substring(0, dotIndex));
            }
            // Handle versions like "21" without dots
            try {
                return Integer.parseInt(version.split("-")[0]);
            } catch (NumberFormatException e) {
                return 17; // Default fallback
            }
        }
    }

    public static void applyJava25HadoopFix() {
        int javaVersion = getJavaMajorVersion();
        
        // Set a default user name for Hadoop (useful in all Java versions)
        if (System.getProperty("HADOOP_USER_NAME") == null) {
            System.setProperty("HADOOP_USER_NAME", System.getProperty("user.name", "hadoop"));
        }
        
        if (javaVersion >= 24) {
            System.out.println("[INFO] Java " + javaVersion + " detected. Using Hadoop 3.4.1+ for compatibility.");
        } else {
            System.out.println("[INFO] Java " + javaVersion + " detected.");
        }
    }
    
    /**
     * Checks if the current Java version is 24 or higher.
     * @return true if Java 24+, false otherwise
     */
    public static boolean isJava24OrHigher() {
        return getJavaMajorVersion() >= 24;
    }

    /**
     * Retrieves the available system memory in gigabytes.
     * @return Available memory in GB
     */
    public static double get_system_memory_available(){
        SystemInfo si = new SystemInfo();
        GlobalMemory mem = si.getHardware().getMemory();

        double total = 1.0*mem.getTotal() / 1024 / 1024 / 1024; // Gb
        double available = 1.0*mem.getAvailable() / 1024 / 1024 / 1024; // Gb

        System.out.printf("Total RAM: %.2f GB%n", total);
        System.out.printf("Available RAM: %.2f GB%n", available);
        return available;
    }

}