package main.java.ai;

import main.java.dia.IsolationWindow;
import main.java.util.Cloger;
import org.apache.avro.Schema;
import org.apache.avro.SchemaBuilder;
import org.apache.avro.generic.GenericRecord;
import org.apache.hadoop.conf.Configuration;
import org.apache.parquet.avro.AvroParquetReader;
import org.apache.parquet.hadoop.ParquetFileReader;
import org.apache.parquet.hadoop.ParquetReader;
import org.apache.parquet.hadoop.api.ReadSupport;
import org.apache.parquet.io.LocalInputFile;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.Type;
import umich.ms.datatypes.LCMSDataSubset;
import umich.ms.datatypes.lcmsrun.LCMSRunInfo;
import umich.ms.datatypes.scan.IScan;
import umich.ms.datatypes.scan.StorageStrategy;
import umich.ms.datatypes.scancollection.IScanCollection;
import umich.ms.datatypes.scancollection.impl.ScanCollectionDefault;
import umich.ms.fileio.exceptions.FileParsingException;
import umich.ms.fileio.filetypes.mzml.MZMLFile;
import umich.ms.fileio.filetypes.mzml.MZMLIndex;

import java.io.*;
import java.nio.file.Paths;
import java.util.*;

public class FileIO {

    public static Schema getSchema4PredictionInput(){
        // Input file for MS2 and RT prediction
        // "pepID sequence mz charge mods mod_sites"
        return SchemaBuilder.record("PeptideForm")
                .fields()
                .requiredInt("pepID")
                .requiredString("sequence")
                .requiredDouble("mz")
                .requiredInt("charge")
                .requiredString("mods")
                .requiredString("mod_sites")
                .endRecord();
    }

    public static Schema getSchema4SpectralLib(){
        // ModifiedPeptide	StrippedPeptide	PrecursorMz	PrecursorCharge	Tr_recalibrated	ProteinID	Decoy	FragmentMz	RelativeIntensity	FragmentType	FragmentNumber	FragmentCharge	FragmentLossType
        return SchemaBuilder.record("SpectralLib")
                .fields()
                .requiredString("ModifiedPeptide")
                .requiredString("StrippedPeptide")
                .requiredFloat("PrecursorMz")
                .requiredInt("PrecursorCharge")
                .requiredFloat("Tr_recalibrated")
                .requiredString("ProteinID")
                .requiredInt("Decoy")
                .requiredFloat("FragmentMz")
                .requiredFloat("RelativeIntensity")
                .requiredString("FragmentType")
                .requiredInt("FragmentNumber")
                .requiredInt("FragmentCharge")
                .requiredString("FragmentLossType")
                .endRecord();
    }

    public static ArrayList<double[]> load_matrix(String file) throws IOException {
        // Load fragment ion intensity data from file
        Configuration conf = new Configuration();
        LocalInputFile inputFile = new LocalInputFile(Paths.get(file));
        ParquetReader<GenericRecord> reader = AvroParquetReader.<GenericRecord>builder(inputFile).withConf(conf).build();

        ArrayList<double[]> rows = new ArrayList<>();
        GenericRecord record;
        // get the number of rows
        int fieldCount=0;
        boolean is_first_row = true;
        while ((record = reader.read()) != null) {
            // get the number of rows
            if(is_first_row){
                fieldCount = record.getSchema().getFields().size();
                is_first_row = false;
            }
            double[] row = new double[fieldCount];
            for (int i = 0; i < fieldCount; i++) {
                row[i] = (float) record.get(i);
            }
            rows.add(row);
        }
        reader.close();
        return rows;
    }

    public static HashMap<Integer, Double> load_rt_data(String file, double rt_max) throws IOException {
        Schema schema = null;
        if(rt_max>0){
            schema = SchemaBuilder.record("RTData")
                    .fields()
                    .requiredInt("pepID")
                    .requiredDouble("rt_pred")
                    .endRecord();
        }else{
            schema = SchemaBuilder.record("RTData")
                    .fields()
                    .requiredInt("pepID")
                    .requiredDouble("irt_pred")
                    .endRecord();
        }
        Configuration conf = new Configuration();
        conf.set(ReadSupport.PARQUET_READ_SCHEMA, schema.toString());
        LocalInputFile inputFile = new LocalInputFile(Paths.get(file));
        ParquetReader<GenericRecord> reader = AvroParquetReader.<GenericRecord>builder(inputFile).withConf(conf).build();
        HashMap<Integer, Double> data = new HashMap<>();
        GenericRecord record;
        while ((record = reader.read()) != null) {
            Integer pepID = (Integer) record.get("pepID");
            if(rt_max>0){
                data.put(pepID,rt_max * ((Double) record.get("rt_pred")));
            }else{
                data.put(pepID,(Double) record.get("irt_pred"));
            }
        }
        reader.close();
        return data;
    }

    public static String[] get_column_names_from_parquet(String file) throws IOException {
        LocalInputFile inputFile = new LocalInputFile(Paths.get(file));
        ArrayList<String> col_names = new ArrayList<>();
        try (ParquetFileReader reader = ParquetFileReader.open(inputFile)) {
            MessageType schema = reader.getFooter().getFileMetaData().getSchema();
            for (Type field : schema.getFields()) {
                col_names.add(field.getName());
            }
        }
        String []cols = new String[col_names.size()];
        for(int i=0;i<col_names.size();i++){
            cols[i] = col_names.get(i);
        }
        return cols;
    }

    /**
     *
     * @param psm_file A PSM table in TSV format.
     * @param ms_file A MS2 file in mzML or a folder containing mzML files.
     */
    public static void generate_mgf_for_PSMs(String psm_file, String ms_file, String out_file) throws IOException {
        HashMap<String,Integer> hIndex = get_column_name2index(psm_file);
        BufferedReader psmReader = new BufferedReader(new FileReader(psm_file));
        psmReader.readLine();
        String line;
        HashMap<String, HashSet<Integer>> ms_file2scan = new HashMap<>();
        while((line=psmReader.readLine())!=null) {
            line = line.trim();
            String[] d = line.split("\t");
            // spectrum_title: ms_file:scan:charge
            String spectrum_title = d[hIndex.get("spectrum_title")];
            String[] a = spectrum_title.split(":");
            String ms_file_name = a[0];
            int scan = Integer.parseInt(a[1]);
            if(!ms_file2scan.containsKey(ms_file_name)){
                ms_file2scan.put(ms_file_name, new HashSet<>());
            }
            ms_file2scan.get(ms_file_name).add(scan);
        }
        psmReader.close();
        Cloger.getInstance().logger.info("The total number of MS2 files in "+psm_file + ": " +ms_file2scan.size());
        // if ms_file is a folder, get all mzML files in the folder
        File F = new File(ms_file);
        BufferedWriter bWriter = new BufferedWriter(new FileWriter(out_file));
        if(F.isDirectory()){
            File[] files = F.listFiles();
            assert files != null;
            for(File f:files){
                if(f.getName().endsWith(".mzML")){
                    String ms_file_name = f.getName().replaceAll(".mzML","");
                    if(ms_file2scan.containsKey(ms_file_name)){
                        HashSet<Integer> scans = ms_file2scan.get(ms_file_name);
                        Cloger.getInstance().logger.info("Generating mgf for "+ms_file_name+" with "+scans.size()+" scans ...");
                        generate_mgf_from_mzML(f.getAbsolutePath(),new HashSet<>(scans),bWriter);
                    }
                }
            }
        }else{
            if(F.getName().endsWith(".mzML")){
                String ms_file_name = F.getName().replaceAll(".mzML","");
                if(ms_file2scan.containsKey(ms_file_name)){
                    HashSet<Integer> scans = ms_file2scan.get(ms_file_name);
                    Cloger.getInstance().logger.info("Generating mgf for "+ms_file_name+" with "+scans.size()+" scans ...");
                    generate_mgf_from_mzML(F.getAbsolutePath(), scans,bWriter);
                }
            }
        }
        bWriter.close();
    }

    public static void generate_mgf_from_mzML(String ms_file, HashSet<Integer> scan_list, BufferedWriter bWriter) throws IOException {
        MZMLFile source = null;
        if (ms_file.endsWith("mzML") || ms_file.endsWith("mzml")) {
            source = new MZMLFile(ms_file);
        } else {
            Cloger.getInstance().logger.error("The input MS data format is not supported:" + ms_file);
            System.exit(1);
        }

        LCMSRunInfo lcmsRunInfo = null;
        try {
            lcmsRunInfo = source.fetchRunInfo();
        } catch (FileParsingException e) {
            e.printStackTrace();
        }
        Cloger.getInstance().logger.info(lcmsRunInfo.toString());
        source.setNumThreadsForParsing(Runtime.getRuntime().availableProcessors());

        MZMLIndex mzMLindex = null;
        try {
            mzMLindex = source.fetchIndex();
        } catch (FileParsingException e) {
            e.printStackTrace();
        }

        if (mzMLindex.size() > 0) {

        } else {
            System.err.println("Parsed index was empty!");
        }

        IScanCollection scans;

        scans = new ScanCollectionDefault(true);
        scans.setDataSource(source);
        try {
            scans.loadData(LCMSDataSubset.MS2_WITH_SPECTRA, StorageStrategy.STRONG);
        } catch (FileParsingException e) {
            e.printStackTrace();
        }

        TreeMap<Integer, IScan> num2scanMap = scans.getMapNum2scan();
        Set<Map.Entry<Integer, IScan>> num2scanEntries = num2scanMap.entrySet();

        File F = new File(ms_file);
        String ms_file_name = F.getName().replaceAll(".mzML","");
        int num = 0;
        for (Map.Entry<Integer, IScan> next : num2scanEntries) {
            IScan scan = next.getValue();
            int scan_num = scan.getNum();
            if (scan.getSpectrum() != null) {
                if (scan.getMsLevel() == 2) {
                    if(scan_list.contains(scan_num)) {
                        // System.out.println(scan.getNum()+":"+scan.getPrecursor().getCharge()+":"+scan.getPrecursor().getMzTarget());
                        if (scan.getPrecursor().getCharge() != null) {
                            int charge = scan.getPrecursor().getCharge();
                            // scan number is original scan number
                            //double mz = scan.getPrecursor().getMzTarget();
                            double mz = scan.getPrecursor().getMzTargetMono();
                            try {
                                asMgf(scan, ms_file, charge);
                                bWriter.write(asMgf(scan, ms_file_name, charge) + "\n");
                                num++;
                            } catch (IOException e) {
                                e.printStackTrace();
                            }
                        } else {
                            Cloger.getInstance().logger.error("The charge of the precursor is null for scan " + scan.getNum());
                            System.exit(1);
                        }
                    }
                }
            }
        }
        if(num != scan_list.size()){
            Cloger.getInstance().logger.warn("The number of MS2 spectra not found in the mzML file: "+(scan_list.size()-num));
        }
    }


    private static String asMgf(IScan scan, String filename, int charge){
        // minute
        double rt = scan.getRt();
        double intensity = 0;
        if(scan.getPrecursor().getIntensity() != null){
            intensity = scan.getPrecursor().getIntensity();
        }
        //double mz = scan.getPrecursor().getMzTarget();
        double mz;
        if(scan.getPrecursor().getMzTargetMono()!=null){
            mz = scan.getPrecursor().getMzTargetMono();
        }else{
            mz = scan.getPrecursor().getMzTarget();
        }

        // System.out.println(scan.getPrecursor().getMzTarget()+"\t"+scan.getPrecursor().getMzTargetMono());
        double[] intArray = scan.getSpectrum().getIntensities();
        double[] mzArray = scan.getSpectrum().getMZs();
        int scan_number = scan.getNum();

        StringBuilder stringBuilder = new StringBuilder();

        stringBuilder.append("BEGIN IONS\n");
        stringBuilder.append("TITLE=").append(filename).append(":").append(scan_number).append(":").append(charge).append("\n");
        stringBuilder.append("PEPMASS=").append(mz).append(" ").append(intensity).append("\n");
        // is there a way to check rt unit from scan.getRt()?
        stringBuilder.append("RTINSECONDS=").append(rt*60).append("\n");
        stringBuilder.append("CHARGE=").append(charge).append("+\n");
        stringBuilder.append("SCANS=").append(scan_number).append("\n");
        for(int i=0;i<intArray.length;i++){
            stringBuilder.append(String.format("%.4f",mzArray[i])).append(" ").append(String.format("%.2f",intArray[i])).append("\n");
        }
        stringBuilder.append("END IONS\n");
        return(stringBuilder.toString());

    }

    public static HashMap<String,Integer> get_column_name2index(String file) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String head_line= reader.readLine().trim();
        HashMap<String,Integer> hIndex = get_column_name2index_from_head_line(head_line);
        reader.close();
        return hIndex;
    }

    public static HashMap<String,Integer> get_column_name2index_from_head_line(String head_line){
        String []h = head_line.split("\t");
        HashMap<String,Integer> hIndex = new HashMap<>();
        for(int i=0;i<h.length;i++){
            hIndex.put(h[i],i);
        }
        return hIndex;
    }


}
