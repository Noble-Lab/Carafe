package main.java.util;

import java.io.BufferedReader;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

public class StreamLog extends Thread {
    InputStream stream;
    String type;
    boolean isOut;
    List<Pattern> filterPatterns;

    public StreamLog(InputStream is, String type, boolean isOut) {
        this(is, type, isOut, new ArrayList<>());
    }

    public StreamLog(InputStream is, String type, boolean isOut, List<String> filterPatterns) {
        this.stream = is;
        this.type = type;
        this.isOut = isOut;
        // Compile regex patterns for better performance
        this.filterPatterns = new ArrayList<>();
        if (filterPatterns != null) {
            for (String pattern : filterPatterns) {
                this.filterPatterns.add(Pattern.compile(pattern));
            }
        }
    }

    private boolean shouldFilter(String line) {
        for (Pattern pattern : filterPatterns) {
            if (pattern.matcher(line).find()) {
                return true;
            }
        }
        return false;
    }

    public void run() {
        try {
            InputStreamReader isr = new InputStreamReader(stream);
            BufferedReader br = new BufferedReader(isr);
            String line;
            while ((line = br.readLine()) != null) {
                if (this.isOut && !shouldFilter(line)) {
                    System.out.println(type + " > " + line);
                }
            }
        } catch (IOException ioe) {
            ioe.printStackTrace();
            System.exit(1);
        }
    }
}
