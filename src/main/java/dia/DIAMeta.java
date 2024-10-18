package main.java.dia;

import main.java.input.CParameter;
import main.java.util.Cloger;
import umich.ms.datatypes.LCMSDataSubset;
import umich.ms.datatypes.lcmsrun.LCMSRunInfo;
import umich.ms.datatypes.scan.IScan;
import umich.ms.datatypes.scan.StorageStrategy;
import umich.ms.datatypes.scan.props.Instrument;
import umich.ms.datatypes.scancollection.IScanCollection;
import umich.ms.datatypes.scancollection.impl.ScanCollectionDefault;
import umich.ms.fileio.exceptions.FileParsingException;
import umich.ms.fileio.exceptions.RunHeaderParsingException;
import umich.ms.fileio.filetypes.mzml.MZMLFile;
import umich.ms.fileio.filetypes.mzml.MZMLIndex;
import umich.ms.fileio.filetypes.mzml.MZMLRunHeaderParser;
import umich.ms.fileio.filetypes.mzml.MZMLRunInfo;
import umich.ms.fileio.filetypes.mzml.jaxb.CVParamType;
import java.util.*;


public class DIAMeta {

    public int ms_level = 2;
    public double fragment_ion_mz_bin_size = 0.05;
    public double fragment_ion_mz_min = Double.MAX_VALUE;
    public double fragment_ion_mz_max = 0;
    public double precursor_ion_mz_min = Double.MAX_VALUE;
    public double precursor_ion_mz_max = 0;
    public double nce = 0;
    public double rt_min = Double.MAX_VALUE;
    public double rt_max = 0;
    public TreeMap<Integer, IScan> num2scanMap = new TreeMap<>();
    public HashMap<String,IsolationWindow> isolationWindowMap = new HashMap<>();
    public HashMap<String,Integer> isolationWindow2n_scan = new HashMap<>();
    public double isolation_win_mz_max = -1;

    public DIAMeta(){

    }

    public void load_ms_data(String ms_file){
        if(this.num2scanMap.isEmpty()) {
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
                if(this.ms_level==2) {
                    scans.loadData(LCMSDataSubset.MS2_WITH_SPECTRA, StorageStrategy.STRONG);
                }else{
                    scans.loadData(LCMSDataSubset.MS1_WITH_SPECTRA, StorageStrategy.STRONG);
                }
            } catch (FileParsingException e) {
                e.printStackTrace();
            }

            this.num2scanMap = scans.getMapNum2scan();
        }
    }

    public String get_ms_instrument(MZMLRunInfo mzmlRunInfo){

        // https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo
        HashMap<String,String> instrumentInfo = new HashMap<>();
        instrumentInfo.put("MS:1003029", "Eclipse"); // Orbitrap Eclipse
        instrumentInfo.put("MS:1003028", "Exploris"); // Orbitrap Exploris 480
        instrumentInfo.put("MS:1003378", "Astral"); // Orbitrap Astral
        instrumentInfo.put("MS:1002416", "Fusion"); // Orbitrap Fusion
        instrumentInfo.put("MS:1002732", "Lumos"); // Orbitrap Fusion Lumos
        instrumentInfo.put("MS:1001911", "QE"); // Q Exactive
        instrumentInfo.put("MS:1002523", "QEHF"); // Q Exactive HF
        instrumentInfo.put("MS:1002526", "QE+"); // Exactive Plus
        instrumentInfo.put("MS:1002634", "QE+"); // Q Exactive Plus
        instrumentInfo.put("MS:1002877", "QEHFX"); // Q Exactive HF-X
        instrumentInfo.put("MS:1002533", "SciexTOF"); // TripleTOF 6600

        //referenceableParamGroup
        String instrument_name = "";
        Map<String, List<CVParamType>> stringListHashMap =  mzmlRunInfo.getRefParamGroups();
        for (String key : stringListHashMap.keySet()){
            for (CVParamType cvParamType : stringListHashMap.get(key)){
                String cv_accession = cvParamType.getAccession();
                System.out.println("MS instrument: "+cvParamType.getName() + " (" + cv_accession + ")");
                if(instrumentInfo.containsKey(cv_accession)){
                    instrument_name =  instrumentInfo.get(cv_accession);
                    break;
                }
            }
        }

        Map<String, Instrument> instrumentMap =  mzmlRunInfo.getInstruments();
        for (String key : instrumentMap.keySet()){
            if(instrumentMap.get(key).getModel().equalsIgnoreCase("TripleTOF 6600")){
                //instrument_name = "SciexTOF";
                break;
            }
        }

        return instrument_name;
    }


    public String get_ms_instrument(String ms_file){

        MZMLFile mzmlFile = new MZMLFile(ms_file);
        MZMLRunHeaderParser mzmlRunHeaderParser = new MZMLRunHeaderParser(mzmlFile);
        String instrument_name = "";
        try {
            MZMLRunInfo mzmlRunInfo = mzmlRunHeaderParser.parse();
            instrument_name = get_ms_instrument(mzmlRunInfo);
        } catch (RunHeaderParsingException e) {
            throw new RuntimeException(e);
        }
        return instrument_name;
    }

    public void get_ms_run_meta_data(){
        Set<Map.Entry<Integer, IScan>> num2scanEntries = num2scanMap.entrySet();
        int total_spectra = 0;
        double mz_start;
        double mz_end;
        String isoWinID;
        for (Map.Entry<Integer, IScan> next : num2scanEntries) {
            IScan scan = next.getValue();
            if (scan.getSpectrum() != null) {
                if (scan.getMsLevel() == this.ms_level) {
                    total_spectra++;

                    double mz_lower = scan.getScanMzWindowLower();
                    double mz_upper = scan.getScanMzWindowUpper();

                    if(this.fragment_ion_mz_min > mz_lower){
                        this.fragment_ion_mz_min = mz_lower;
                    }
                    if(this.fragment_ion_mz_max < mz_upper){
                        this.fragment_ion_mz_max = mz_upper;
                    }

                    double rt = scan.getRt();

                    if(this.rt_min > rt){
                        this.rt_min = rt;
                    }
                    if(this.rt_max < rt){
                        this.rt_max = rt;
                    }


                    if(this.ms_level == 2) {
                        mz_start = scan.getPrecursor().getMzRangeStart();
                        mz_end = scan.getPrecursor().getMzRangeEnd();
                        isoWinID = IsolationWindow.generate_id(mz_start,mz_end);
                        if(this.precursor_ion_mz_max < mz_end){
                            this.precursor_ion_mz_max = mz_end;
                        }
                        if(this.precursor_ion_mz_min > mz_start){
                            this.precursor_ion_mz_min = mz_start;
                        }
                        if(this.nce==0){
                            System.out.println(scan.getPrecursor().getActivationInfo().toString());
                            if(scan.getPrecursor().getActivationInfo().getActivationEnergyHi()!=null){
                                this.nce = scan.getPrecursor().getActivationInfo().getActivationEnergyHi();
                            }else{
                                this.nce = CParameter.NCE;
                            }

                        }else{
                            if(scan.getPrecursor().getActivationInfo().getActivationEnergyHi()!=null) {
                                if (this.nce != scan.getPrecursor().getActivationInfo().getActivationEnergyHi()) {
                                    Cloger.getInstance().logger.error("The input MS data has different NCE values!");
                                    System.out.println(this.nce + " vs " + scan.getPrecursor().getActivationInfo().getActivationEnergyHi());
                                }
                            }
                        }

                    }else{
                        isoWinID = "0";
                        mz_start = mz_lower;
                        mz_end = mz_upper;
                    }

                    if(this.ms_level == 2) {
                        if(this.isolation_win_mz_max >0 &&
                                (mz_end - mz_start) > this.isolation_win_mz_max){
                            System.out.println("Ignore - the isolation window m/z range is too large: "+mz_start+"-"+mz_end);
                            continue;
                        }

                    }

                    if(!this.isolationWindowMap.containsKey(isoWinID)){
                        this.isolationWindowMap.put(isoWinID,new IsolationWindow(mz_start,mz_end));
                    }

                    if(!isolationWindow2n_scan.containsKey(isoWinID)){
                        isolationWindow2n_scan.put(isoWinID,0);
                    }
                    isolationWindow2n_scan.put(isoWinID,isolationWindow2n_scan.get(isoWinID)+1);

                }
            }
        }

        System.out.println("Total MS/MS spectra:"+total_spectra);
        this.rt_max = this.rt_max + 0.1;
        for(String isoWin: this.isolationWindowMap.keySet()){
            System.out.println("Valid isolation window: "+isoWin + " -> "+isolationWindowMap.get(isoWin).mz_lower+" - "+isolationWindowMap.get(isoWin).mz_upper);
        }
        if(this.ms_level == 2) {
            System.out.println("Fragment ion m/z range: " + this.fragment_ion_mz_min + "-" + this.fragment_ion_mz_max);
            System.out.println("Precursor ion m/z range: " + this.precursor_ion_mz_min + "-" + this.precursor_ion_mz_max);
        }
        System.out.println("Retention time range: " + this.rt_min + "-" + this.rt_max);
    }

    public long get_fragment_ion_mz_bin_index(double mz){
        return Math.round((mz - this.fragment_ion_mz_min) / this.fragment_ion_mz_bin_size);
    }

}
