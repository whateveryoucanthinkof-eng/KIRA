# Datasets

Put the public datasets here, in exactly this layout. Everything under `data/` except this
file is git-ignored. To keep the data elsewhere, either symlink these folders or set
`PCAP_ROOT`, `CIC2018_CSV_DIR`, `CIC2017_DIR` and `CTU_DIR`.

```text
data/
├── cic2018/
│   ├── pcap/                    CSE-CIC-IDS2018 raw PCAPs, one folder per capture day
│   │   ├── wed_14_pcap/pcap/…   (per-host capture files, as distributed)
│   │   ├── thu_15_pacap/pcap/…
│   │   ├── fri_16_pcap/pcap/…
│   │   ├── tue_20_pcap/pcap/…
│   │   ├── wed_21_pcap/pcap/…
│   │   ├── thu_22_pcap/pcap/…
│   │   ├── fri_23_pacap/pcap/…
│   │   ├── wed_28_pcap/pcap/…
│   │   ├── thu_1_pcap/pcap/…
│   │   └── fri_2_pcap/pcap/…
│   └── csv/                     CSE-CIC-IDS2018 labelled CSVs (labels for the PCAPs)
│       ├── wed_14_csv.csv  thu_15_csv.csv  fri_16_csv.csv  tue_20_csv.csv  wed_21_csv.csv
│       └── thu_22_csv.csv  fri_23_csv.csv  wed_29_csv.csv  thu_1_csv.csv   fri_2_csv.csv
├── cic2017/
│   └── TrafficLabelling/        CIC-IDS2017 "GeneratedLabelledFlows" CSVs (the variant WITH IPs)
│       ├── Monday-WorkingHours.pcap_ISCX.csv
│       ├── Tuesday-WorkingHours.pcap_ISCX.csv
│       ├── Wednesday-workingHours.pcap_ISCX.csv
│       ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
│       ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
│       ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
│       ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
│       └── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
└── ctu13/                       CTU-13 bidirectional NetFlow, one folder per scenario
    ├── 1/capture20110810.binetflow
    ├── 2/capture20110811.binetflow
    ├── …
    └── 13/capture20110815-3.binetflow
```

The names are fixed: the train / validation / test assignment of every capture is frozen in
`data_unification/splits.lock.json`, and the CIC-2018 folder names follow it (including the
upstream spellings `thu_15_pacap` and `fri_23_pacap`). The 28 Feb PCAP day is labelled by
`wed_29_csv.csv`, as in the published CSVs.

Sources (all public):

| Dataset | Where |
|---|---|
| CSE-CIC-IDS2018 (PCAP + CSV) | https://www.unb.ca/cic/datasets/ids-2018.html (AWS bucket `cse-cic-ids2018`) |
| CIC-IDS2017 (`GeneratedLabelledFlows`) | https://www.unb.ca/cic/datasets/ids-2017.html |
| CTU-13 | https://www.stratosphereips.org/datasets-ctu13 |

Check the layout before training:

```bash
./train.sh check
```

It verifies every capture by name, pairs each PCAP day with its label CSV, and reports PCAP files
known to be truncated in the upstream release (`scripts/dataset_integrity/`).
