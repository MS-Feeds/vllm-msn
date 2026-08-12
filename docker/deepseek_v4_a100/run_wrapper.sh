#!/bin/bash
set -x
set -e

# --- 1. Environment and Variable Setup ---
if [[ -z "${_ModelDataPath_}" ]]; then
  echo "_ModelDataPath_ is not set or is empty. Assuming Azure Machine Learning environment."
  ws_dir=$(pwd)

  # Map Azure ML incoming positional args.
  input_aml_path=$1
  output_dir=$2
  model_dir=$3
  model_name=$4
  run_type=$5

  echo "Current working directory: ${ws_dir}"
else
  echo "_ModelDataPath_ is set and not empty: ${_ModelDataPath_}"
  ws_dir=${_ModelDataPath_}/code

  # Default fallbacks for local/DLIS environments.
  input_aml_path="${_InputFilePath_:-none}"
  output_dir="${_OutputFilePath_:-none}"
  model_dir=${_ModelDataPath_}/model
  model_name="model_release"
  run_type="http"

  echo "Current working directory: ${ws_dir}"
fi

# Keep these variables available to run.sh and child processes.
export input_aml_path output_dir model_dir model_name

# --- 2. Build the Correct Arguments for run.sh ---
case "$run_type" in
  "grpc"|"http"|"udp")
    RUN_ARGS=("$run_type")
    ;;
  "offline")
    RUN_ARGS=("offline" "$input_aml_path" "$output_dir")
    ;;
  "offline_profile")
    RUN_ARGS=("offline_profile" "$input_aml_path" "$output_dir" "profile_output")
    ;;
  *)
    echo "Warning: run_type '$run_type' not directly supported by run.sh. Defaulting to http."
    RUN_ARGS=("http")
    ;;
esac

# --- 3. Execution ---
bash "$ws_dir/run.sh" "${RUN_ARGS[@]}" &
pid_bash=$!

wait "$pid_bash"
