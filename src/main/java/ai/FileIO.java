package main.java.ai;

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
import java.io.*;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;

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


}
