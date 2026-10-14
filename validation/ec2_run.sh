#!/bin/bash
# EC2 user-data for the baseline feature run. Runs as root on Ubuntu 24.04.
# Expects the packed tiles at s3://$BUCKET/terravector/tiles_q/ (validation/pack_tiles.py).
# Writes results to s3://$BUCKET/terravector/baselines/<mode>_p<ps>/ and logs to
# s3://$BUCKET/terravector/logs/, then powers off (launch with
# --instance-initiated-shutdown-behavior terminate).
set -u
BUCKET="3body-compute-290318"
PREFIX="terravector"
WORKERS="${WORKERS:-30}"
LOG=/root/run.log
exec > >(tee -a $LOG) 2>&1
echo "=== start $(date -u)"

push_log() { aws s3 cp $LOG s3://$BUCKET/$PREFIX/logs/run.log --only-show-errors || true; }
trap 'echo "=== exiting $(date -u)"; push_log; poweroff' EXIT

export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q python3.12-venv git libgl1 unzip curl
curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip && (cd /tmp && unzip -q awscliv2.zip && ./aws/install) || true

cd /root
git clone -q https://github.com/bshepp/RESIDUALS.git
git clone -q https://github.com/bshepp/terravector.git
python3.12 -m venv venv
. venv/bin/activate
pip install -q --upgrade pip
pip install -q numpy scipy scikit-image opencv-python-headless PyWavelets scikit-learn tqdm matplotlib PyYAML zstandard psutil
pip install -q -e /root/RESIDUALS
python -c "import residuals, numpy, scipy, skimage, cv2, pywt; print('imports ok')"
push_log

mkdir -p /root/tiles_q /root/out
# wait for the uploader to mark the tile set complete
until aws s3 ls s3://$BUCKET/$PREFIX/tiles_q/_COMPLETE >/dev/null 2>&1; do echo "waiting for tiles... $(date -u)"; sleep 60; done
aws s3 sync s3://$BUCKET/$PREFIX/tiles_q/ /root/tiles_q/ --only-show-errors --exclude _COMPLETE
echo "tiles: $(ls /root/tiles_q | wc -l)"
push_log

cd /root/terravector
for spec in "simple 64" "lrm 64" "full 128"; do
  set -- $spec
  MODE=$1; PS=$2
  echo "=== $MODE p$PS start $(date -u)"
  python validation/baseline_features.py --tiles /root/tiles_q --out /root/out --mode $MODE --patch-size $PS --workers $WORKERS
  aws s3 sync /root/out/${MODE}_p${PS}/ s3://$BUCKET/$PREFIX/baselines/${MODE}_p${PS}/ --only-show-errors
  echo "=== $MODE p$PS done $(date -u)"
  push_log
done
echo "=== ALL DONE $(date -u)"
touch /root/ALL_DONE && aws s3 cp /root/ALL_DONE s3://$BUCKET/$PREFIX/baselines/ALL_DONE --only-show-errors
