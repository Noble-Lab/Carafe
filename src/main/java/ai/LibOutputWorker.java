package main.java.ai;

import org.apache.avro.Schema;
import org.apache.avro.reflect.ReflectData;
import org.apache.hadoop.conf.Configuration;
import org.apache.parquet.avro.AvroParquetWriter;
import org.apache.parquet.example.data.Group;
import org.apache.parquet.hadoop.ParquetFileWriter;
import org.apache.parquet.hadoop.ParquetReader;
import org.apache.parquet.hadoop.ParquetWriter;
import org.apache.parquet.hadoop.example.GroupReadSupport;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.apache.parquet.io.LocalOutputFile;
import org.apache.parquet.io.OutputFile;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class LibOutputWorker implements Runnable{

    private static final Schema SCHEMA = ReflectData.AllowNull.get().getSchema(LibFragment.class);
    private final String out_library_file;
    String ms2_intensity_file;
    String rt_file;
    String ms2_mz_file;
    String ms2_file;
    public static Map<String, String> pep2pro = new HashMap<>();
    public static AIGear aiGear = new AIGear();

    public LibOutputWorker(String ms2_file, String ms2_mz_file, String ms2_intensity_file, String rt_file, String out_library_file){
        this.ms2_file = ms2_file;
        this.ms2_mz_file = ms2_mz_file;
        this.ms2_intensity_file = ms2_intensity_file;
        this.rt_file = rt_file;
        this.out_library_file = out_library_file;

    }

    @Override
    public void run() {
        BufferedWriter libWriter = null;
        boolean export_tsv = true;
        ParquetWriter<LibFragment> pWriter = null;
        if(aiGear.export_spectral_library_file_format.equalsIgnoreCase("parquet")){
            export_tsv = false;
            // Schema schema = FileIO.getSchema4SpectralLib();
            // org.apache.hadoop.fs.Path path = new org.apache.hadoop.fs.Path(out_library_file);
            OutputFile out_file = null;
            // try {
            //    out_file = HadoopOutputFile.fromPath(path, new Configuration());
            //} catch (IOException e) {
            //    throw new RuntimeException(e);
            // }
            out_file = new LocalOutputFile(Paths.get(out_library_file));
            try {
                pWriter = AvroParquetWriter.<LibFragment>builder(out_file)
                        .withSchema(SCHEMA)
                        .withDataModel(ReflectData.get())
                        //.withCompressionCodec(CompressionCodecName.SNAPPY)
                        .withCompressionCodec(CompressionCodecName.ZSTD)
                        .withPageSize(ParquetWriter.DEFAULT_PAGE_SIZE)
                        //.withConf(new Configuration())
                        .withValidation(false)
                        // override when existing
                        .withWriteMode(ParquetFileWriter.Mode.OVERWRITE)
                        .withDictionaryEncoding(false)
                        .build();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }else {
            try {
                libWriter = new BufferedWriter(new FileWriter(out_library_file));
                libWriter.write("ModifiedPeptide\tStrippedPeptide\tPrecursorMz\tPrecursorCharge\tTr_recalibrated\tProteinID\tDecoy\tFragmentMz\tRelativeIntensity\tFragmentType\tFragmentNumber\tFragmentCharge\tFragmentLossType\n");
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        boolean export_diann_format = false;
        boolean export_EncyclopeDIA_format = false;
        boolean export_generic_format = false;
        if(aiGear.export_spectral_library_format.equalsIgnoreCase("DIANN") || aiGear.export_spectral_library_format.equalsIgnoreCase("DIA-NN")){
            export_diann_format = true;
        }else if(aiGear.export_spectral_library_format.equalsIgnoreCase("EncyclopeDIA")){
            export_EncyclopeDIA_format = true;
        }else{
            export_generic_format = true;
        }
        int pepID;
        String sequence;
        String mods;
        String mod_sites;
        double rt;
        String rt_str;
        double mz;
        int charge;
        String decoy;
        int decoy_label;
        String protein;
        int frag_start_idx;
        int frag_stop_idx;
        LibFragment libFragment = new LibFragment();
        String [] fragment_ion_column_names = null;
        try {
            fragment_ion_column_names = FileIO.get_column_names_from_parquet(ms2_mz_file);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        String []ion_types = new String[fragment_ion_column_names.length];
        String []mod_losses = new String[fragment_ion_column_names.length];
        int []ion_charges = new int[fragment_ion_column_names.length];
        for(int j=0;j<fragment_ion_column_names.length;j++){
            if(fragment_ion_column_names[j].startsWith("b")){
                ion_types[j] = "b";
            } else if (fragment_ion_column_names[j].startsWith("y")) {
                ion_types[j] = "y";
            }else{
                System.err.println("Unknown fragment ion type:"+fragment_ion_column_names[j]);
                System.exit(1);
            }

            if(fragment_ion_column_names[j].endsWith("_z1")){
                ion_charges[j] = 1;
            }else if(fragment_ion_column_names[j].endsWith("_z2")){
                ion_charges[j] = 2;
            }else{
                System.err.println("Unknown fragment ion charge:"+fragment_ion_column_names[j]);
                System.exit(1);
            }

            if(fragment_ion_column_names[j].contains("modloss")) {
                if(aiGear.mod_ai.equalsIgnoreCase("phosphorylation")){
                    mod_losses[j] = "H3PO4";
                }
            }else{
                mod_losses[j] = "noloss";
            }
        }
        // RT information
        HashMap<Integer,Double> pepID2rt = null;
        try {
            pepID2rt = FileIO.load_rt_data(rt_file,aiGear.rt_max);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        // MS intensity
        ArrayList<double[]> ms2_intensity_lines = null;
        try {
            ms2_intensity_lines = FileIO.load_matrix(ms2_intensity_file);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        // mz intensity
        ArrayList<double[]> ms2_mz_lines = null;
        try {
            ms2_mz_lines = FileIO.load_matrix(ms2_mz_file);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        // MS2 information
        String line;
        Configuration conf = new Configuration();
        // Set the configuration to use built-in Java classes
        conf.set("fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem");
        conf.set("fs.hdfs.impl", "org.apache.hadoop.hdfs.DistributedFileSystem");
        org.apache.hadoop.fs.Path path = new org.apache.hadoop.fs.Path(ms2_file);
        ParquetReader<Group> reader = null;
        try {
            reader = ParquetReader.builder(new GroupReadSupport(), path)
                    .withConf(conf)
                    .build();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        // Read the records one row at a time
        Group group;
        try {
            while ((group = reader.read()) != null) {
                // get column "pepID"
                pepID = group.getInteger("pepID",0);
                frag_start_idx = (int)group.getLong("frag_start_idx",0);
                frag_stop_idx = (int)group.getLong("frag_stop_idx",0);
                sequence = group.getString("sequence",0);
                mods = group.getString("mods",0);
                mod_sites = group.getString("mod_sites",0);
                mz = group.getDouble("mz",0);
                charge = group.getInteger("charge",0);
                // decoy = group.getString("decoy",0);
                protein = pep2pro.get(sequence);
                libFragment.StrippedPeptide = sequence;
                libFragment.PrecursorMz = (float) mz;
                libFragment.PrecursorCharge = charge;
                libFragment.ProteinID = protein;
                libFragment.Decoy = 0;
                libFragment.Tr_recalibrated = pepID2rt.get(pepID).floatValue();
                ArrayList<LibFragment> lines = aiGear.get_fragment_ion_intensity4parquet_all(ms2_mz_lines,
                        ms2_intensity_lines,
                        fragment_ion_column_names,
                        frag_start_idx,
                        frag_stop_idx,
                        aiGear.lf_top_n_fragment_ions,
                        ion_types,
                        mod_losses,
                        ion_charges,
                        aiGear.lf_frag_n_min);

                // decoy_label = decoy.startsWith("Yes")?1:0;
                decoy_label = 0;
                rt_str = String.format("%.2f",pepID2rt.get(pepID));
                String mod_pep;
                if(export_diann_format){
                    mod_pep = aiGear.get_modified_peptide_diann(sequence,mods,mod_sites);
                }else if(export_EncyclopeDIA_format){
                    mod_pep = aiGear.get_modified_peptide_encyclopedia(sequence,mods,mod_sites);
                }else{
                    mod_pep = aiGear.get_modified_peptide(sequence,mods,mod_sites);
                }
                libFragment.ModifiedPeptide = mod_pep;
                for(LibFragment l: lines) {
                    if(export_tsv) {
                        StringBuilder ob = new StringBuilder();
                        ob.append(mod_pep).append("\t")
                                .append(sequence).append("\t")
                                .append(mz).append("\t")
                                .append(charge).append("\t")
                                .append(rt_str).append("\t")
                                .append(protein).append("\t")
                                .append(decoy_label).append("\t")
                                // FragmentMz	RelativeIntensity	FragmentType	FragmentNumber	FragmentCharge	FragmentLossType
                                .append(l.FragmentMz).append("\t")
                                .append(String.format("%.4f",l.RelativeIntensity)).append("\t")
                                .append(l.FragmentType).append("\t")
                                .append(l.FragmentNumber).append("\t")
                                .append(l.FragmentCharge).append("\t")
                                .append(l.FragmentLossType).append("\n");
                        libWriter.write(ob.toString());
                    }else{
                        // write to parquet
                        // FragmentMz	RelativeIntensity	FragmentType	FragmentNumber	FragmentCharge	FragmentLossType
                        libFragment.FragmentMz = l.FragmentMz;
                        libFragment.RelativeIntensity = l.RelativeIntensity;
                        libFragment.FragmentType = l.FragmentType;
                        libFragment.FragmentNumber = l.FragmentNumber;
                        libFragment.FragmentCharge = l.FragmentCharge;
                        libFragment.FragmentLossType = l.FragmentLossType;
                        pWriter.write(libFragment);
                    }
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        try {
            reader.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        //}
        if(export_tsv) {
            try {
                libWriter.close();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }else{
            try {
                pWriter.close();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

    }
}
