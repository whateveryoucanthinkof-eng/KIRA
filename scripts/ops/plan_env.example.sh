# Optional machine-specific settings for ./train.sh. Every value has a default;
# copy this file, adjust what you need, and `source` it before ./train.sh.
#
# Datasets (default: data/ in the repository, see README.md "Datasets"):
# export PCAP_ROOT=/path/to/cic2018/pcap
# export CIC2018_CSV_DIR=/path/to/cic2018/csv
# export CIC2017_DIR=/path/to/cic2017/TrafficLabelling
# export CTU_DIR=/path/to/ctu13
#
# export PYTHON=/path/to/venv/bin/python
# export MEM_MAX=17G            # memory cap per heavy job
# export INGEST_WORKERS=8       # parallel capture parsers
# export OUT=results/training_plan
