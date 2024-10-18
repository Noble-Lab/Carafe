package main.java.util;

import java.io.BufferedReader;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class StreamLog extends Thread {
    InputStream stream;
    String type;
    boolean isOut;

    public StreamLog(InputStream is, String type, boolean isOut) {
        this.stream = is;
        this.type = type;
        this.isOut = isOut;
    }

    public void run() {
        try {
            InputStreamReader isr = new InputStreamReader(stream);
            BufferedReader br = new BufferedReader(isr);
            String line;
            while ((line = br.readLine()) != null) {
                if (this.isOut) {
                    System.out.println(type + " > " + line);
                }
            }
        } catch (IOException ioe) {
            ioe.printStackTrace();
            System.exit(1);
        }
    }
}
