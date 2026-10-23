#!/usr/bin/env bash
# Chuẩn bị 1 máy EC2 Ubuntu 22.04/24.04 (khuyến nghị c6i.2xlarge: 8 vCPU / 16 GB, ổ gp3 50 GB)
# để chạy toàn bộ: kind + Istio + Prometheus + Online Boutique + Locust.
# Chạy:  bash deploy/ec2_bootstrap.sh   (sau đó logout/login lại 1 lần để nhóm docker có hiệu lực)
set -euo pipefail
sudo apt-get update -y
sudo apt-get install -y docker.io python3-pip python3-venv git tmux curl jq
sudo usermod -aG docker "$USER"
sudo systemctl enable --now docker
# kind cần nhiều inotify watcher khi chạy nhiều pod
echo -e "fs.inotify.max_user_watches=524288\nfs.inotify.max_user_instances=512" | sudo tee /etc/sysctl.d/99-kind.conf
sudo sysctl --system >/dev/null
python3 -m venv ~/dsenv
~/dsenv/bin/pip install -q --upgrade pip
~/dsenv/bin/pip install -q locust requests pandas numpy pyyaml matplotlib
echo 'source ~/dsenv/bin/activate' >> ~/.bashrc
echo "Xong. Logout rồi SSH lại, sau đó: cd DeepScaler && deploy/00_kind_cluster.sh"
