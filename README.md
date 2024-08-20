# [Carafe](https://github.com/Noble-Lab/Carafe)
**Carafe** is a tool for experimental specific *in silico* spectral library generation for DIA data analysis.

![Downloads](https://img.shields.io/github/downloads/Noble-Lab/Carafe/total.svg) ![Release](https://img.shields.io/github/release/Noble-Lab/Carafe.svg)![Downloads](https://img.shields.io/github/downloads/Noble-Lab/Carafe/latest/total)

## Installation

Carafe is written using Java and can be run on Windows, Mac OS and Linux. Both Java and python are required to be installed to run Carafe. If java is not installed, please install Java by following the instruction at https://openjdk.org/install/ or https://www.oracle.com/java/technologies/downloads/. After java is installed, Carafe can be downloaded at https://github.com/Noble-Lab/Carafe/releases.

### Install AlphaPeptDeep-DIA

Carafe uses a customized version of AlphaPeptDeep (AlphaPeptDeep-DIA) for model training using DIA data. Please follow the instruction at https://github.com/wenbostar/alphapeptdeep_dia to install AlphaPeptDeep-DIA.

## Usage

```
$ java -jar carafe.jar
usage: Options
 -i <arg>                PSM file
 -ms <arg>               MS file in mzML format
 -fixMod <arg>           Fixed modification, the format is like : 1,2,3. Use '-printPTM' to show all
                         supported modifications. Default is 1 (Carbamidomethylation(C)[57.02]). If
                         there is no fixed modification, set it as '-fixMod no' or '-fixMod 0'.
 -varMod <arg>           Variable modification, the format is the same with -fixMod. Default is 2
                         (Oxidation(M)[15.99]). If there is no variable modification, set it as
                         '-varMod no' or '-varMod 0'.
 -maxVar <arg>           Max number of variable modifications, default is 1
 -db <arg>               Protein database
 -o <arg>                Output directory
 -itol <arg>             Fragment ion m/z tolerance in ppm, default is 20
 -itolu <arg>            Fragment ion m/z tolerance unit, default is ppm
 -sg <arg>               The number of data points for XIC smoothing, it's 3 in default
 -nm                     Perform fragment ion intensity normalization or not
 -nf <arg>               The minimum number of matched fragment ions to consider, it's 4 in default
 -cs                     Fragment ion charge less than precursor charge or not
 -ez                     Export fragment ion mz to file or not
 -skyline                Export skyline transition list file or not
 -valid                  Only export valid matches or not
 -na <arg>               The number of adjacent scans to match: default is 0
 -fdr <arg>              The minimum FDR cutoff to consider, default is 0.01
 -cor <arg>              The minimum correlation cutoff to consider, default is 0.75
 -ptm_site_prob <arg>    The minimum PTM site score to consider, default is 0.75
 -use_all_peaks          Use all peaks for training
 -min_mz <arg>           The minimum fragment ion m/z to consider, default is 200.0
 -min_n <arg>            The minimum high quality fragment ion number to consider, default is 4
 -enzyme <arg>           Enzyme used for protein digestion. 0:Non enzyme, 1:Trypsin (default),
                         2:Trypsin (no P rule), 3:Arg-C, 4:Arg-C (no P rule), 5:Arg-N, 6:Glu-C,
                         7:Lys-C
 -miss_c <arg>           The max missed cleavages, default is 1
 -I2L                    Convert I to L
 -clip_n_m               When digesting a protein starting with amino acid M, two copies of the
                         leading peptides (with and without the N-terminal M) are considered or not.
                         Default is false.
 -minLength <arg>        The minimum length of peptide to consider, default is 7
 -maxLength <arg>        The maximum length of peptide to consider, default is 35
 -min_pep_mz <arg>       The minimum mz of peptide to consider, default is 400
 -max_pep_mz <arg>       The maximum mz of peptide to consider, default is 1000
 -min_pep_charge <arg>   The minimum precursor charge to consider, default is 2
 -max_pep_charge <arg>   The maximum precursor charge to consider, default is 4
 -lf_type <arg>          Spectral library format: DIA-NN (default) or EncyclopeDIA
 -lf_format <arg>        Spectral library file format: tsv (default) or parquet
 -lf_frag_mz_min <arg>   The minimum mz of fragment to consider for library generation, default is
                         200
 -lf_frag_mz_max <arg>   The minimum mz of fragment to consider for library generation, default is
                         1800
 -lf_top_n_frag <arg>    The maximum number of fragment ions to consider for library generation,
                         default is 20
 -lf_frag_n_min <arg>    The minimum number of fragment ions to consider for library generation,
                         default is 2
 -rf                     Refine peak boundary or not
 -rf_rt_win <arg>        RT window for refine peak boundary, default is 3 minutes
 -rt_win_offset <arg>    RT window offset for XIC extraction, default is 1 minute
 -xic                    Export XIC to file or not
 -export_mgf             Export spectra to a mgf file or not
 -y1                     Don't use y1 ion in training
 -n_ion_min <arg>        For n-terminal fragment ions (such as b-ion) with number <= n_ion_min, they
                         will be considered as invalid. Default is 0.
 -c_ion_min <arg>        For c-terminal fragment ions (such as y-ion) with number <= n_ion_min, they
                         will be considered as invalid. Default is 0.
 -nce <arg>              NCE for in-silico spectral library
 -ms_instrument <arg>    MS instrument for in-silico spectral library: default is Eclipse
 -device <arg>           device for in-silico spectral library: default is gpu
 -se <arg>               The search engine used to generate the identification result: DIA-NN
 -mode <arg>             Data type: general or phosphorylation
 -tf <arg>               Fine tune type: ms2, rt, all (default)
 -seed <arg>             Random seed, 2024 in default
 -fast                   Save data to parquet format for speeding up reading and writing
 -h                      Help
```

#### *In silico* spectral library generation using Carafe

The following example shows how to generate a spectral library for yeast proteome (**UP000002311_559292.fasta**). The training DIA data (**train_dia_file.mzML**) is a human DIA file and peptide identification is performed using DIA-NN (**diann/report.tsv**).

```shell
java -jar carafe-0.0.1.jar -db UP000002311_559292.fasta -fixMod 1 -varMod 0 -maxVar 1 -o test_ai_all -min_mz 200 -maxLength 35 -min_pep_mz 400 -max_pep_mz 1000 -i diann/report.tsv -ms train_dia_file.mzML -itol 20 -itolu ppm -nm -nf 4 -ez -skyline -valid -enzyme 2 -miss_c 1 -se "DIA-NN" -mode general -minLength 7 -lf_type diann -rf -tf all -na 0 -cor 0.8 -lf_top_n_frag 20 -lf_frag_n_min 2 -rf_rt_win 1 -n_ion_min 2 -c_ion_min 2 -seed 2000
```

The output spectral library is in a tsv format compatible with DIA-NN. The content looks like below:

<code>
<pre>
ModifiedPeptide          StrippedPeptide  PrecursorMz        PrecursorCharge  Tr_recalibrated  ProteinID             Decoy  FragmentMz  RelativeIntensity  FragmentType  FragmentNumber  FragmentCharge  FragmentLossType
_KLWWDC[UniMod:4]YWWDR_  KLWWDCYWWDR      571.9258823279321  3                117.52           sp|P39961|TOG1_YEAST  0      476.22522   1.0000             y             3               1               noloss
_KLWWDC[UniMod:4]YWWDR_  KLWWDCYWWDR      571.9258823279321  3                117.52           sp|P39961|TOG1_YEAST  0      662.3045    0.9709             y             4               1               noloss
_KLWWDC[UniMod:4]YWWDR_  KLWWDCYWWDR      571.9258823279321  3                117.52           sp|P39961|TOG1_YEAST  0      825.36786   0.2221             y             5               1               noloss
_KLWWDC[UniMod:4]YWWDR_  KLWWDCYWWDR      571.9258823279321  3                117.52           sp|P39961|TOG1_YEAST  0      290.1459    0.1924             y             2               1               noloss
_KLWWDC[UniMod:4]YWWDR_  KLWWDCYWWDR      571.9258823279321  3                117.52           sp|P39961|TOG1_YEAST  0      889.4025    0.1285             b             6               1               noloss
_KLWWDC[UniMod:4]YWWDR_  KLWWDCYWWDR      571.9258823279321  3                117.52           sp|P39961|TOG1_YEAST  0      729.3719    0.1169             b             5               1               noloss
_KLWWDC[UniMod:4]YWWDR_  KLWWDCYWWDR      571.9258823279321  3                117.52           sp|P39961|TOG1_YEAST  0      428.26562   0.0727             b             3               1               noloss
_KLWWDC[UniMod:4]YWWDR_  KLWWDCYWWDR      571.9258823279321  3                117.52           sp|P39961|TOG1_YEAST  0      1052.4658   0.0624             b             7               1               noloss
_KLWWDC[UniMod:4]YWWDR_  KLWWDCYWWDR      571.9258823279321  3                117.52           sp|P39961|TOG1_YEAST  0      614.3449    0.0437             b             4               1               noloss
</code>
</pre>

## How to cite:

