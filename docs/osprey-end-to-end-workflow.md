# The OspreySharp end-to-end workflow

This document describes, step by step, what Carafe2 does when it runs a spectral-library
workflow with **OspreySharp** as the search engine — from the input protein FASTA all the way
to the final fine-tuned spectral library (and, in the end-to-end variant, the search of your
project files with that library).

It is written for users and developers who want to understand *exactly* what happens between
clicking **Run Carafe** and getting a library, including the parts that are normally hidden:
the peptide-FASTA digestion, the initial library prediction, the MSConvert conversions, the
decoys-in-library / pairing-manifest mechanism, and the optional entrapment peptides.

> **Background.** Unlike DIA-NN, OspreySharp does not build its own AI-predicted library.
> Carafe2 therefore has to *supply* OspreySharp with a library (targets **and** decoys), let
> OspreySharp search the training runs against it, and then fine-tune Carafe2's RT / fragment
> intensity / ion-mobility models on OspreySharp's identifications. The fine-tuned models are
> what generate the final, experiment-specific library.

---

## Which workflows use OspreySharp

Two entries in the **Workflow** dropdown drive OspreySharp:

| # | Workflow | What it does |
|---|----------|--------------|
| **4** | OspreySharp: search, finetune, build new library | Searches the **training** runs, fine-tunes the models, builds a new fine-tuned library. **Stops there.** |
| **5** | OspreySharp: end-to-end (finetune, then search project files) | Everything Workflow 4 does, **plus** a second OspreySharp search of your **project** runs using the new fine-tuned library. |

The only difference between them is the final search stage (Stage 6 below). Everything up to
and including building the fine-tuned library is identical.

---

## Inputs

| Input | Purpose | Formats |
|-------|---------|---------|
| **Train MS file(s)** | Runs that are searched and used to fine-tune the models. | mzML, Thermo `.raw`, Bruker `.d` (a file, several files, or a folder) |
| **Train protein database** | FASTA digested to build the library that the **training** runs are searched against. | FASTA |
| **Library protein database** | FASTA digested to build the **final** library. Often the same file as the train DB. | FASTA |
| **Project MS file(s)** *(Workflow 5 only)* | Runs searched at the end with the fine-tuned library. | mzML, `.raw`, `.d` |

When the train and library databases are the **same file**, Carafe2 digests it once and reuses
the resulting peptide FASTA and pairing manifest for both the initial and the final library.

---

## The stages, in order

```
            ┌─────────────────────────────────────────────────────────────────────┐
 Phase 1    │  MSConvert (raw/.d → mzML)   │   Digest FASTA → peptide FASTA +      │
 (parallel) │  one process per file        │   pairing manifest  (Stage 1)         │
            └──────────────┬───────────────┴───────────────┬───────────────────────┘
                           │                                │
 Phase 2                   ▼                                ▼
 (sequential)   Stage 2  Build INITIAL library  ◀──────────┘
                Stage 3  OspreySharp search of TRAIN runs  →  osprey.blib
                Stage 4  Fine-tune RT / MS2 / CCS models on osprey.blib + mzML
                Stage 5  Build FINAL (fine-tuned) library
                Stage 6  (Workflow 5 only) OspreySharp search of PROJECT runs
```

MSConvert conversions and the FASTA digestion run **in parallel** up front (Phase 1); the
search/fine-tune/build chain that depends on them runs **sequentially** (Phase 2). The number
of concurrent MSConvert processes is set on the OspreySharp tab (default 4).

---

### Stage 0 — MSConvert: raw files → mzML

OspreySharp reads **mzML**. Any training or project input that is not already mzML (Thermo
`.raw`, Bruker `.d`) is converted first. Carafe2 launches **one MSConvert process per file** so
conversions run in parallel, throttled to the configured thread count.

The command is essentially:

```
msconvert --filter "peakPicking true 1-2" --mzML "<input.raw>" -o "<out_dir>"
```

- `--filter "peakPicking true 1-2"` centroids MS1 and MS2 during conversion.
- The MSConvert executable is taken from the OspreySharp tab (or the saved preference, or
  `msconvert` on `PATH`). MSConvert ships with ProteoWizard; it must be installed to convert
  Thermo/Bruker data. Files that are already mzML skip this stage entirely.

The converted mzMLs land in a `*_mzML` subfolder of the output directory and are what every
later stage (search, fragment extraction) reads.

---

### Stage 1 — Digest the protein FASTA into a peptide FASTA (+ pairing manifest)

This is the step that makes OspreySharp's library. Carafe2 digests the protein FASTA into
peptides and, for **each target peptide**, deterministically generates a **decoy** (and,
optionally, **entrapment** peptides — see [Entrapment](#the-entrapment-option) below). It writes:

1. A **peptide-level FASTA** — one entry per peptide (target, decoy, and any entrapment forms).
   Each entry's accession is its source protein accession, optionally with a per-source
   `_pepNNNNN` counter so library predictors that de-duplicate by accession accept every entry.
2. A **5-column pairing manifest** (TSV) that records, for every peptide, whether it is a decoy
   and which target it is paired with (see [The pairing manifest](#the-pairing-manifest)).

Digestion uses the GUI's configured digest options — enzyme, missed cleavages, peptide-length
range, and N-terminal-M clipping — so the peptide set matches what you'd expect from the
selected protease. Decoys are produced by a **deterministic shuffle that preserves the
C-terminal residue** (keeping the peptide tryptic-like); any synthetic peptide that happens to
collide with a real target sequence is dropped (the FDRBench collision-drop policy).

The equivalent command line is:

```
java -jar carafe.jar -build_entrapment_fasta peptides.fasta \
     -db proteins.fasta -manifest pairing.tsv \
     -enzyme 1 -miss_c 1 [-entrapment] [-no_decoys] [-mz_filter]
```

This stage runs once for the train DB and (if the library DB differs) once for the library DB.

---

### Stage 2 — Build the initial in-silico library

Carafe2 now predicts spectra and retention times for **every** peptide in the Stage 1 FASTA
(targets and decoys alike), producing the library OspreySharp will search against. There are two
predictor options, chosen on the OspreySharp tab:

- **Local AlphaPepDeep (Carafe).** Carafe runs its own library generation over the peptide
  FASTA with **`-enzyme NoCut`** — the peptides are already digested, so "NoCut" tells Carafe to
  predict each FASTA entry as-is rather than re-digesting it. AlphaPepDeep predicts the fragment
  intensities and RT.
- **Koina model.** Carafe queries a remote [Koina](https://koina.wilhelmlab.org) model
  (e.g. a Prosit/AlphaPepDeep fragment-intensity model plus an iRT model) for the predictions.
  When auto-NCE is enabled, a reference mzML is used to pick the collision energy.

Either way the output is a **DIA-NN-format TSV** spectral library containing both target and
decoy precursors. This is the *initial*, generic library — not yet tuned to your data.

---

### Stage 3 — OspreySharp search of the training runs

OspreySharp searches the converted training mzMLs against the initial library. The key part is
that the library already contains decoys, and the pairing manifest tells OspreySharp how targets
and decoys correspond — see [Decoys in the library](#decoys-in-the-library-and-the-pairing-manifest).

The command is shaped like:

```
OspreySharp -i <run1.mzML> <run2.mzML> ... \
            -l <initial_library.tsv> \
            -o osprey.blib \
            --decoys-in-library \
            --decoy-pairing-manifest pairing.tsv \
            --resolution <auto|unit|hram> \
            --fragment-tolerance <tol> --fragment-unit <ppm|mz> \
            --run-fdr <q> --experiment-fdr <q> --protein-fdr <q> \
            --fdr-method percolator \
            --fdr-level <precursor|peptide|both> \
            --shared-peptides <policy> \
            --threads <N>
```

- `-i` takes **all** of the run mzMLs (variadic).
- `-l` is the Stage 2 initial library; `-o` is the output **`osprey.blib`** (a BiblioSpec/SQLite
  spectral-library file holding the identifications).
- FDR is always estimated with **Percolator**; the run / experiment / protein FDR thresholds and
  the FDR level come from the OspreySharp tab.
- `--threads` defaults to the number of available processors.

The output `osprey.blib` holds the confident identifications: peptide sequence, modifications,
charge, and apex retention time (and ion mobility for timsTOF), at the configured FDR.

---

### Stage 4 — Fine-tune the models on the search results

This is where the experiment-specific learning happens. Carafe2 feeds `osprey.blib` back into its
own pipeline as the identification input (the role a DIA-NN report plays in Workflows 1–3):

1. **Read the identifications.** `osprey.blib` is converted to a DIA-NN-style report internally,
   so the rest of Carafe2 can treat it exactly like a DIA-NN search result. The blib supplies the
   peptide sequence, modifications, charge, and apex RT/IM.
2. **Extract measured fragments.** From the **training mzMLs**, Carafe2 extracts the *observed*
   fragment-ion intensities for each identified precursor and performs transition masking — just
   as it does for a DIA-NN report. (OspreySharp gives the IDs; the mzML gives the empirical
   intensities used as training targets.)
3. **Train.** Carafe2 invokes its Python training script to fine-tune, by transfer learning, the
   prediction models on this measured data:
   - **RT** (retention time) model,
   - **MS2** (fragment-intensity) model,
   - **CCS** (ion-mobility) model — only when ion-mobility data is present (timsTOF) and there are
     enough training examples.

   What gets trained is controlled by the transfer-learning type (`-tf`): `all` (default) trains
   RT + MS2 + CCS-if-available, or you can restrict it to `ms2`, `rt`, or `ccs`. Training device
   (`gpu`/`cpu`), instrument, and NCE are configurable; epochs and learning rate are auto-tuned to
   a roughly fixed training budget based on the number of identified peptides.

The result is a set of fine-tuned model weights written into the output directory.

---

### Stage 5 — Build the final, fine-tuned library

Using the fine-tuned models from Stage 4, Carafe2 generates the final spectral library. It digests
the **library** protein database (Stage 1's library-DB peptide FASTA, when the library DB differs
from the train DB) and predicts RT, fragment intensities, and ion mobility for every precursor with
the newly tuned models.

The output is `carafe_spectral_library.tsv` (plus a Parquet copy). The export format is selectable:
**DIA-NN** (default), **EncyclopeDIA**, **Skyline** (a BiblioSpec `.blib`), or **mzSpecLib**.

For **Workflow 4**, this is the end of the run — the fine-tuned library is the deliverable.

---

### Stage 6 — OspreySharp search of the project runs *(Workflow 5 only)*

In the end-to-end workflow, Carafe2 takes one more step: it runs OspreySharp again, this time
searching your **project** mzMLs against the **fine-tuned** library from Stage 5 (the same
`OspreySharp -i … -l … -o … --decoys-in-library --decoy-pairing-manifest …` shape as Stage 3,
using the library-DB pairing manifest). The output is a second `osprey.blib` containing the
peptide detections for your project, produced with a library tuned to your experimental
conditions.

---

## How the Carafe GUI calls each tool

The GUI does **not** run any of the science inside its own process. When you click **Run
Carafe**, `runCarafe()` builds a list of small command objects (`CmdTask`) — one per external
tool invocation — each with a deterministic output path, and hands them to
`executeParallelWorkflow()`, which launches them as **separate OS processes**. Four executables
are involved:

| Tool | How the GUI launches it | Builder method |
|------|-------------------------|----------------|
| **MSConvert** | A shell command string run through the OS shell. | `buildMsConvertCommand` |
| **OspreySharp** | An argument list run with `ProcessBuilder`; binary located by `resolveOspreyBinary()`. | `buildOspreyCommand` |
| **Carafe** (entrapment FASTA, Koina library, initial/final library + finetune) | A **fresh Carafe process** — `java -jar carafe.jar …` (or the bundled `Carafe.exe`). The GUI re-invokes the Carafe CLI as a child process. | `buildEntrapmentFastaCommand`, `buildKoinaLibraryCommand`, `buildCarafeCommand` |
| **Python** (`ai.py` training) | Spawned *by the child Carafe process* during fine-tuning, not by the GUI directly. | (inside `AIGear`) |

> **Key point.** The peptide-FASTA build, the Koina/initial-library build, the
> fine-tune-and-build-new-library step, and the project library are each run by spawning a **new**
> Carafe JVM (`java -Xmx<~80% of RAM>G -jar carafe.jar …`). The GUI only orchestrates; the actual
> digestion, prediction, and training happen in those child processes (and the Python subprocesses
> they spawn). That is why each step can be skipped/reused independently and why **Stop** has to
> track and kill child processes.

### The `CmdTask` abstraction and result reuse

Each step is a `CmdTask` carrying its argument list / command string, a human-readable
description, its input files, its output directory, and a `skip_check_file` — the output whose
existence means "this step is already done." When **Reuse existing results** is enabled,
`skipIfResultPresent()` skips a step only if its `skip_check_file` exists **and** a stored
signature (a hash of the command + arguments + input files) still matches; otherwise the step
re-runs. The signature is computed at run time, so a changed upstream output cascades a re-run
through every downstream step. After a step succeeds, `writeStepSignature()` records the new
signature next to its output.

### The phase / lane execution model

`executeParallelWorkflow()` runs a list of **phases** sequentially. Within a phase, **lanes** run
concurrently on a thread pool; within a single lane, tasks run sequentially. MSConvert tasks must
acquire a **semaphore** sized to the "parallel MSConvert processes" setting, so only N convert at
once. The first task to exit non-zero aborts the run. The OspreySharp workflow uses two phases:

- **Phase 1 (parallel lanes):** one lane per MSConvert file, plus the entrapment-FASTA build(s).
  When the initial library is a Koina model (and isn't waiting on `auto` NCE, which needs a
  converted mzML first), the Koina prediction shares a lane with its FASTA and runs here too.
- **Phase 2 (one sequential lane):** the deferred local/Koina initial library → OspreySharp train
  search → fine-tune + new library → (Workflow 5) OspreySharp project search.

### How each builder maps GUI widgets to a command line

**MSConvert** (`buildMsConvertCommand`) — a shell string:

```
msconvert --filter "peakPicking true 1-2" --mzML "<input>" -o "<out_dir>"
```

The executable is resolved from the OspreySharp-tab combo → the saved preference → `msconvert` on
`PATH`. `resolveOspreyMsInputs()` produces **one task per input file** (so they parallelize):
`.mzML` inputs pass through untouched; `.raw` and `.d` inputs are queued for conversion.

**OspreySharp** (`buildOspreyCommand`) — an argument list. The binary is found by
`resolveOspreyBinary()` in order: (1) saved path preference / OspreySharp-tab combo, (2) bundled
`<jarDir>/osprey/<rid>/OspreySharp(.exe)`, (3) `~/.carafe/osprey/<rid>/…`, (4) `where`/`which` on
`PATH`, where `<rid>` is `win-x64` / `osx-arm64` / `osx-x64` / `linux-x64`. Field → flag:

| GUI field | OspreySharp flag |
|-----------|------------------|
| (all run mzMLs) | `-i <f1> <f2> …` (variadic) |
| (initial / final library TSV) | `-l` |
| (`osprey_*/osprey.blib`) | `-o` |
| pairing manifest present | `--decoys-in-library` + `--decoy-pairing-manifest <tsv>` |
| Resolution combo | `--resolution` |
| Fragment tolerance + unit fields | `--fragment-tolerance` + `--fragment-unit` (ppm or mz) |
| Run / Experiment / Protein FDR fields | `--run-fdr` / `--experiment-fdr` / `--protein-fdr` |
| (always) | `--fdr-method percolator` |
| FDR level combo | `--fdr-level` |
| Shared-peptides combo | `--shared-peptides` |
| (automatic) | `--threads <available CPUs>` |
| OspreySharp additional-options field | appended verbatim |

**Carafe entrapment FASTA** (`buildEntrapmentFastaCommand`) — `java -jar carafe.jar`:

| GUI field | Flag |
|-----------|------|
| (target / library DB) | `-db` |
| (deterministic outputs) | `-build_entrapment_fasta <fasta>` + `-manifest <tsv>` |
| Enzyme combo | `-enzyme` |
| Missed-cleavage spinner | `-miss_c` |
| Min / Max length spinners | `-minLength` / `-maxLength` |
| Min / Max charge spinners | `-min_pep_charge` / `-max_pep_charge` |
| "Include entrapment" checkbox | `-entrapment` (added only when checked) |

**Carafe Koina library** (`buildKoinaLibraryCommand`) — `java -jar carafe.jar -build_koina_library
…`. The "Initial library predictor" combo maps to a `{ms2 model, rt model}` pair — e.g. *Koina:
Prosit 2020 HCD* → `Prosit_2020_intensity_HCD` + `Prosit_2019_irt`. The local option returns
`null`, which tells the workflow to use `buildCarafeCommand` instead. It passes
`-koina_ms2_model` / `-koina_rt_model` / `-koina_url`, the NCE (`auto` adds `-nce_ms <reference
mzML>`), instrument, charge and m/z ranges, modifications, and fragment options.

**Carafe initial/final library + fine-tune** (`buildCarafeCommand`) — this is the **same** builder
the DIA-NN workflows use. Rather than write a second command builder, the OspreySharp path
re-points it through six **override fields**, which are set just before the task is built and
cleared (`clearCarafeOverrides()`) right after:

| Override field | Normally comes from | Set to (OspreySharp) |
|----------------|---------------------|----------------------|
| `carafeDbOverride` | `-db` (library DB field) | the peptide FASTA |
| `carafeIOverride` | `-i` (DIA-NN report field) | `osprey.blib` (fine-tune input) |
| `carafeSeOverride` | `-se` (search-engine combo) | `OspreySharp` |
| `carafeEnzymeOverride` | `-enzyme` | `NoCut` |
| `carafeLfTypeOverride` | `-lf_type` | `DIA-NN` |
| `carafeOutSubdirOverride` | output subdir | `osprey_initial_library` / `osprey_new_library` |

The initial-library call passes an empty `-ms` (so the child Carafe only predicts a library); the
fine-tune call passes the training MS folder so the child Carafe extracts measured fragments from
the mzML and runs `ai.py` to train the models before generating the new library.

---

## Decoys in the library, and the pairing manifest

OspreySharp expects a library that **already contains decoys**, with retention times and spectra
predicted for them exactly as for targets. That is why Stage 1 generates a decoy for every target
and Stage 2 predicts both. Two flags wire this up at search time:

- **`--decoys-in-library`** — tells OspreySharp that decoy entries are present in the `-l` library
  itself (rather than asking OspreySharp to generate decoys on the fly). FDR is then estimated
  against those library decoys.
- **`--decoy-pairing-manifest pairing.tsv`** — points OspreySharp at the manifest so it knows
  which library entry is the decoy *partner* of which target.

### The pairing manifest

The manifest is a 5-column, FDRBench-style TSV written in Stage 1. Its header is:

```
sequence	decoy	proteins	peptide_type	peptide_pair_index
```

| Column | Meaning |
|--------|---------|
| `sequence` | The peptide sequence (target, decoy, or entrapment form). |
| `decoy` | `Yes` for decoy / p_decoy rows, `No` for target / p_target rows. |
| `proteins` | Source protein accession(s); shared peptides list all sources joined by `;`. Decoy rows carry the `decoy_` prefix; entrapment rows carry the `_p_target` suffix. |
| `peptide_type` | One of `target`, `decoy`, `p_target`, `p_decoy`. |
| `peptide_pair_index` | An integer that **groups all forms derived from the same target peptide**. Every group contains exactly one `target` row. |

Each `peptide_pair_index` group is a *pair* (target + decoy) or, with entrapment on, a *quartet*
(target + p_target + decoy + p_decoy). The index is what lets OspreySharp pair a decoy back to its
target for FDR estimation. Schematically (sequences shown are illustrative — real decoy/entrapment
sequences are deterministic C-term-preserving shuffles):

```
sequence    decoy  proteins                    peptide_type  peptide_pair_index
SAMPLEPEPK  No     sp|P12345|PROT              target        0
PELPMEPSAK  Yes    decoy_sp|P12345|PROT        decoy         0
EXAMPLEPER  No     sp|P67890|PROT2             target        1
RELPMEPXAE  Yes    decoy_sp|P67890|PROT2       decoy         1
```

The manifest and the peptide FASTA are always written together and always agree — that
consistency is all OspreySharp requires.

---

## The entrapment option

"Add entrapment" extends the target/decoy *pair* into a *quartet* for FDR-validation experiments
(the FDRBench scheme). When enabled, Stage 1 emits, per target peptide:

| Type | What it is | `decoy` column |
|------|-----------|----------------|
| `target` | The real peptide. | No |
| `p_target` | An **entrapment** peptide: a deterministic shuffle of the target (different seed from the decoy), treated as a "fake target." | No |
| `decoy` | A deterministic shuffle of the target. | Yes |
| `p_decoy` | A deterministic shuffle of the *entrapment* (`p_target`). | Yes |

Entrapment peptides let you empirically measure FDR: because `p_target` sequences are not real
biology, the rate at which they are "identified" estimates how well the reported FDR holds. They
are flagged in the FASTA header with the `_p_target` suffix and appear as their own rows in the
manifest, grouped under the same `peptide_pair_index` as their target.

Relevant flags (defaults shown):

| Flag | Effect |
|------|--------|
| `-entrapment` | Turn entrapment on (emit `p_target`/`p_decoy`). **Off by default** — target+decoy only. |
| `-no_decoys` | Do **not** add decoys (default is to add them). |
| `-mz_filter` | Restrict to peptides whose precursor m/z falls in the configured window (`-min_pep_mz`/`-max_pep_mz` at `-min_pep_charge..-max_pep_charge`). |
| `-entrapment_seed <int>` | Master RNG seed for entrapment shuffles (default `42`). |
| `-decoy_seed <int>` | Master RNG seed for decoy shuffles (default `24`). |

The shuffles are deterministic: the same FASTA, seeds, and digest options always produce the same
manifest, so a run is reproducible.

---

## Output directory layout

A typical OspreySharp run (Workflow 5) produces:

```
<output_dir>/
├── train_mzML/                         # MSConvert output for the training runs (if converted)
├── project_mzML/                       # MSConvert output for the project runs (Workflow 5)
├── osprey_train_db_peptides.fasta      # Stage 1: peptide FASTA (train DB)
├── osprey_train_db_pairing.tsv         # Stage 1: pairing manifest (train DB)
├── osprey_library_db_peptides.fasta    # Stage 1: peptide FASTA (library DB; only if it differs)
├── osprey_library_db_pairing.tsv       # Stage 1: pairing manifest (library DB; only if it differs)
├── osprey_initial_library/             # Stage 2: initial in-silico library (targets + decoys)
│   └── carafe_spectral_library.tsv
├── osprey_train/                       # Stage 3: training search
│   └── osprey.blib
├── (fine-tuned model weights)          # Stage 4: RT / MS2 / CCS models + training report
├── osprey_new_library/                 # Stage 5: final fine-tuned library
│   └── carafe_spectral_library.tsv
├── osprey_project/                     # Stage 6 (Workflow 5): project search w/ fine-tuned library
│   └── osprey.blib
└── carafe_log.txt                      # Full log, including every command line that was run
```

(Exact subfolder names may vary; `carafe_log.txt` records the actual paths and commands for each
run.)

---

## CLI parameter quick reference

The GUI builds and runs these commands for you; the table is for reproducing or scripting a run.

| Parameter | Used in | Meaning |
|-----------|---------|---------|
| `-se OspreySharp` | search/fine-tune | Select OspreySharp as the search engine. |
| `-build_entrapment_fasta <path>` | Stage 1 | Output peptide-level FASTA path. |
| `-db <fasta>` | Stage 1 / 5 | Input protein database to digest. |
| `-manifest <path>` | Stage 1 | Output pairing manifest (used by Osprey's `--decoy-pairing-manifest`). |
| `-entrapment` | Stage 1 | Emit entrapment `p_target`/`p_decoy` peptides. |
| `-no_decoys` | Stage 1 | Skip decoy generation. |
| `-mz_filter` | Stage 1 | Apply the precursor m/z window filter. |
| `-entrapment_seed` / `-decoy_seed` | Stage 1 | RNG seeds for entrapment / decoy shuffles. |
| `-enzyme`, `-miss_c`, `-minLength`, `-maxLength` | Stage 1 | Digestion options. |
| `-enzyme NoCut` | Stage 2 | Predict each peptide-FASTA entry as-is (no re-digestion). |
| `--decoys-in-library`, `--decoy-pairing-manifest` | Stage 3 / 6 | Tell OspreySharp the library contains decoys and how they pair. |
| `-i <blib>` | Stage 4 | Fine-tune on an OspreySharp `.blib` (in place of a DIA-NN report). |
| `-ms <files>` | Stage 4 | Training MS data for fragment extraction. |
| `-tf <all\|ms2\|rt\|ccs>` | Stage 4 | Which models to fine-tune (default `all`). |
| `-device <gpu\|cpu>`, `-nce`, `-ms_instrument` | Stage 4 | Training device / collision energy / instrument. |
| `-lf_type <DIA-NN\|EncyclopeDIA\|Skyline\|mzSpecLib>` | Stage 5 | Final library export format. |

---

## See also

- `README.md` → **"Using OspreySharp as the search engine"** for the GUI walkthrough and
  OspreySharp installation/bundling details.
- `scripts/build_ospreysharp.sh` / `.bat` for building the self-contained OspreySharp binary.
