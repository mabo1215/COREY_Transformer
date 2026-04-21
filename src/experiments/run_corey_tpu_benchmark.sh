# run_corey_tpu_benchmark.sh
#
# Example shell script to run the new benchmark on Google Cloud TPU VM or Colab with service account token.
#
# Usage:
#   bash run_corey_tpu_benchmark.sh <path_to_service_account_json>

SERVICE_ACCOUNT_JSON=$1

# Activate service account (if on GCP VM)
gcloud auth activate-service-account --key-file="$SERVICE_ACCOUNT_JSON"

# Install torch_xla (if not already installed)
pip install torch_xla torch torchvision

# Run the benchmark on TPU
echo "Running COREY selective_scan_fn benchmark on TPU..."
python src/experiments/run_corey_tpu_benchmark.py --device tpu --model mamba-370m --chunk-size 512 --seq-len 4096 --dtype float16 --repeat 30 --output-dir src/outputs/corey_tpu_benchmark

# Upload results to GCS (optional)
# gsutil cp src/outputs/corey_tpu_benchmark/summary.json gs://<your-bucket>/corey_tpu_benchmark/

echo "Done. Results in src/outputs/corey_tpu_benchmark/summary.json"
