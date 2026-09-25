# Machine-specific settings for scripts/run_training_plan.sh. Copy to
# ~/.config/cyberworld/plan_env.sh, adjust the paths, then launch as a user
# service (survives the terminal; see scripts/ops/watchdog.py to monitor):
#   systemd-run --user --unit=train-plan --working-directory="$PWD" --setenv=PATH="$PATH" \
#     -p StandardOutput=append:$PWD/plan.log -p StandardError=append:$PWD/plan.log \
#     bash -c 'source ~/.config/cyberworld/plan_env.sh && exec bash scripts/run_training_plan.sh all'
export PCAP_ROOT=$HOME/Documents/SIH/DATA/pcap
export CIC2018_CSV_DIR=$HOME/Documents/SIH/DATA/CSV
export CIC2017_DIR="$HOME/Documents/SIH/test/extracted_flows/TrafficLabelling /"
export CTU_DIR=$HOME/Documents/SIH/CTU-13-Dataset
export PYTHON=$HOME/.pyenv/versions/3.12.14/bin/python
export LANES=2
export INGEST_WORKERS=8
export PLAN_ENCODER_EXTRA="--fast_step --fast_step_level 3 --batch_planner"
