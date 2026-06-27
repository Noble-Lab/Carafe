package main.java.util;

import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamConstants;
import javax.xml.stream.XMLStreamReader;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.zip.GZIPInputStream;

/**
 * Lightweight mzML utilities. Currently reads the normalized collision energy (NCE) used for
 * "auto" NCE selection when generating an initial library (local AlphaPepDeep or Koina).
 *
 * <p>The reader streams the mzML with StAX and returns the value of the first
 * {@code collision energy} cvParam (PSI-MS accession {@code MS:1000045}) it finds, which for a DIA
 * run is identical across precursors. It stops as soon as that value is found.</p>
 */
public final class MzmlUtils {

    private MzmlUtils() {
    }

    /**
     * Read the collision energy (NCE) from an mzML file, or {@code -1} if it could not be found.
     *
     * @param mzmlPath path to a (optionally gzip-compressed) mzML file
     * @return the first collision-energy value, or -1 if absent/unreadable
     */
    public static double readNce(String mzmlPath) {
        File f = new File(mzmlPath);
        if (!f.isFile()) {
            return -1;
        }
        XMLStreamReader r = null;
        try (InputStream raw = new BufferedInputStream(new FileInputStream(f))) {
            InputStream in = mzmlPath.toLowerCase().endsWith(".gz") ? new GZIPInputStream(raw) : raw;
            XMLInputFactory factory = XMLInputFactory.newInstance();
            // Harden against XXE and keep the parser lightweight.
            factory.setProperty(XMLInputFactory.IS_SUPPORTING_EXTERNAL_ENTITIES, false);
            factory.setProperty(XMLInputFactory.SUPPORT_DTD, false);
            r = factory.createXMLStreamReader(in);
            while (r.hasNext()) {
                if (r.next() == XMLStreamConstants.START_ELEMENT && "cvParam".equals(r.getLocalName())) {
                    String accession = r.getAttributeValue(null, "accession");
                    String name = r.getAttributeValue(null, "name");
                    if ("MS:1000045".equals(accession)
                            || (name != null && name.equalsIgnoreCase("collision energy"))) {
                        String value = r.getAttributeValue(null, "value");
                        if (value != null && !value.isBlank()) {
                            try {
                                return Double.parseDouble(value.trim());
                            } catch (NumberFormatException ignore) {
                                return -1;
                            }
                        }
                    }
                }
            }
        } catch (Exception e) {
            return -1;
        } finally {
            if (r != null) {
                try {
                    r.close();
                } catch (Exception ignore) {
                    // ignore
                }
            }
        }
        return -1;
    }
}
