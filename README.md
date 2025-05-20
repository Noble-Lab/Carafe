# [Carafe](https://doi.org/10.1101/2024.10.15.618504)

![Downloads](https://img.shields.io/github/downloads/Noble-Lab/Carafe/total.svg) ![Release](https://img.shields.io/github/release/Noble-Lab/Carafe.svg)![Downloads](https://img.shields.io/github/downloads/Noble-Lab/Carafe/latest/total)

**Carafe** is a tool for experimental specific *in silico* spectral library generation using deep learning for DIA data analysis. Carafe generates an experimental specific *in silico* spectral library by directly training both RT and fragment ion intensity prediction models on DIA data generated from a specific DIA experiment setting of interest. More details about how Carafe works can be found in the following manuscript:

Wen, Bo, Chris Hsu, Wen-Feng Zeng, Michael Riffle, Alexis Chang, Miranda Mudge, Brook L. Nunn, et al. “[Carafe Enables High Quality in Silico Spectral Library Generation for Data-Independent Acquisition Proteomics](https://doi.org/10.1101/2024.10.15.618504).” **bioRxiv**, 2024.10.15.618504.

## Installation

Carafe is written using Java and can be run on Windows, Mac OS and Linux. Both Java and python are required to be installed to run Carafe. If java is not installed, please install Java by following the instruction at https://openjdk.org/install/ or https://www.oracle.com/java/technologies/downloads/. After java is installed, Carafe can be downloaded at https://github.com/Noble-Lab/Carafe/releases.

### Install AlphaPeptDeep-DIA

Carafe uses a customized version of AlphaPeptDeep (AlphaPeptDeep-DIA) for model training using DIA data. Please follow the instruction at https://github.com/wenbostar/alphapeptdeep_dia to install AlphaPeptDeep-DIA. In default, the python package will be installed into a conda environment named "carafe". To run the Carafe java program, the conda environment "carafe" should be activated by using the following command:

```shell
conda activate carafe
```

## Usage

```
$ java -jar carafe.jar
usage: Options
 -i <arg>                PSM file
 -ms <arg>               MS file in mzML format
 -fixMod <arg>           Fixed modification, the format is like : 1,2,3. Default is 1 (Carbamidomethylation(C)[57.02]). 
                         If there is no fixed modification, set it as '-fixMod no' or '-fixMod 0'.
 -varMod <arg>           Variable modification, the format is the same with -fixMod. Default is 2
                         (Oxidation(M)[15.99]). If there is no variable modification, set it as
                         '-varMod no' or '-varMod 0'. For phosphorylation, the code is "7,8,9".
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
 -lf_type <arg>          Spectral library format: DIA-NN (default), EncyclopeDIA, Skyline (blib) or
                         mzSpecLib
 -lf_format <arg>        Spectral library file format: tsv (default) or parquet
 -lf_frag_mz_min <arg>   The minimum mz of fragment to consider for library generation, default is
                         200
 -lf_frag_mz_max <arg>   The minimum mz of fragment to consider for library generation, default is
                         1800
 -lf_top_n_frag <arg>    The maximum number of fragment ions to consider for library generation,
                         default is 20
 -lf_min_n_frag <arg>    The minimum number of fragment ions to consider for library generation,
                         default is 2
 -lf_frag_n_min <arg>    The minimum fragment ion number to consider for library generation, default
                         is 2
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

#### Experiment specific *in silico* spectral library generation using Carafe

The following example shows how to generate a spectral library for yeast proteome ([UP000002311_559292.fasta](https://panoramaweb.org/_webdav/Panorama%20Public/2024/MacCoss%20-%20Carafe/%40files/SupplementaryFiles/ProteinDatabases/UP000002311_559292.fasta)). The training DIA data ([Crucios_20240320_CH_15_HeLa_CID_27NCE_01.mzML](https://panoramaweb.org/_webdav/Panorama%20Public/2024/MacCoss%20-%20Carafe/%40files/RawFiles/Lumos/8mz_staggered_reCID/Crucios_20240320_CH_15_HeLa_CID_27NCE_01.mzML)) is a human cell line DIA file and peptide identification is performed using DIA-NN ([report.tsv](https://panoramaweb.org/_webdav/Panorama%20Public/2024/MacCoss%20-%20Carafe/%40files/SupplementaryFiles/SearchResults/Lumos_8mz_staggered_reCID_human/report.tsv)). Carafe has been tested on using peptide identification result from DIA-NN search for model training.

```shell
# please make sure that the carafe conda environment is activated (conda activate carafe) before run the following java command line.
java -jar carafe-0.0.1.jar -db UP000002311_559292.fasta -fixMod 1 -varMod 0 -maxVar 1 -o test_ai_all -min_mz 200 -maxLength 35 -min_pep_mz 400 -max_pep_mz 1000 -i report.tsv -ms Crucios_20240320_CH_15_HeLa_CID_27NCE_01.mzML -itol 20 -itolu ppm -nm -nf 4 -ez -skyline -valid -enzyme 2 -miss_c 1 -se DIA-NN -mode general -minLength 7 -lf_type diann -rf -tf all -na 0 -cor 0.8 -lf_top_n_frag 20 -lf_frag_n_min 0 -rf_rt_win 1.5 -n_ion_min 2 -c_ion_min 2 -seed 2000 -lf_min_n_frag 2
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


The above example command line took about 8 minutes on a Linux server (CPU: 36 threads, 128G RAM) using GPU (one Nvidia Quadro RTX4000): set parameter **-device gpu**. It took less than 14 minutes using CPU only on the same server: set parameter **-device cpu**.

##### Modifications supported in Carafe

The following modifications have been tested in Carafe:
Carbamidomethylation(C) as fixed modification and Oxidation(M), Phosphorylation (STY) as variable modifications.

Add parameter setting ``-fixMod 1`` to set **Carbamidomethylation(C)** as the fixed modification in Carafe. Add parameter setting ``-varMod 2`` to set **Oxidation(M)** as a variable modification. Add parameter setting ``-varMod 7,8,9`` to set **Phosphorylation(STY)** as a variable modification. Add parameter setting ``-varMod 2,7,8,9`` to set both **Phosphorylation(STY)** and **Oxidation(M)** as variable modifications. When variable modification is considered, ``-maxVar`` is recommended to set as ``-maxVar 1``: the max number of variable modifications allowed for each peptide is 1. For **phosphorylation**, the command line parameter ``-mode`` needs to set as ``-mode phosphorylation``.

#### *In silico* spectral library generation using Carafe with pretrained DDA models

The following example shows how to generate a spectral library for yeast proteome ([UP000002311_559292.fasta](https://panoramaweb.org/_webdav/Panorama%20Public/2024/MacCoss%20-%20Carafe/%40files/SupplementaryFiles/ProteinDatabases/UP000002311_559292.fasta)) **without fine-tuning pretrained models using DIA data**. No DIA data is required.

```shell
java -jar carafe-0.0.1.jar -db UP000002311_559292.fasta -fixMod 1 -varMod 0 -maxVar 1 -o test_ai_all -min_mz 200 -maxLength 35 -min_pep_mz 400 -max_pep_mz 1000 -enzyme 2 -miss_c 1 -mode general -minLength 7 -lf_type diann -lf_top_n_frag 20 -lf_frag_n_min 2 -nce 27 -ms_instrument Lumos -seed 2000
```

The output spectral library is in a tsv format compatible with DIA-NN.

The above example command line took about 3 minutes on a Linux server (CPU: 36 threads, 128G RAM) using GPU (one Nvidia Quadro RTX4000): set parameter **-device gpu**. It took about 6 minutes using CPU only on the same server: set parameter **-device cpu**.

#### An end-to-end workflow for *in silico* spectral library generation using Carafe

An end-to-end workflow is also available to run Carafe for *in silico* spectral library generation. The workflow is available at https://nf-carafe-ai-ms.readthedocs.io. The workflow is built using [Nextflow](https://www.nextflow.io/) and [Docker](https://www.docker.com/). It is developed to go from a DIA RAW MS/MS file to an experiment-specific *in silico* spectral library for DIA data analysis. The following input files are typically required to run the workflow:

```
1. A DIA MS/MS file generated using an experiment setting of interest. Both ".raw" and ".mzML" formats are supported.
2. A protein database in FASTA format used for peptide detection on the DIA file;
3. A protein database in FASTA format used for *in silico* spectral library generation.
4. A parameter file.
```

The workflow can be run on both Windows and Linux systems. It can also be run on both local computer and cloud computer ([AWS](https://aws.amazon.com/)). GPU is not needed to run this workflow.

Details are available at https://nf-carafe-ai-ms.readthedocs.io.

## How to cite:

Wen, Bo, Chris Hsu, Wen-Feng Zeng, Michael Riffle, Alexis Chang, Miranda Mudge, Brook L. Nunn, et al. “[Carafe Enables High Quality in Silico Spectral Library Generation for Data-Independent Acquisition Proteomics](https://doi.org/10.1101/2024.10.15.618504).” **bioRxiv**, 2024.10.15.618504.

