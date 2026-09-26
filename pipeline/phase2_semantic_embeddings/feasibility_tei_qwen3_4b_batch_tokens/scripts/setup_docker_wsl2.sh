#!/usr/bin/env bash
# Installs Docker Engine and the NVIDIA Container Toolkit inside WSL2 Ubuntu-24.04 (systemd enabled).
# Run as root:  wsl -d Ubuntu-24.04 -u root -- bash /mnt/c/.../setup_docker_wsl2.sh
# Sources: https://docs.docker.com/engine/install/ubuntu/ (apt repository method) and
#          https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
TARGET_USER="${TARGET_USER:-spence}"
NCT_VERSION="${NVIDIA_CONTAINER_TOOLKIT_VERSION:-1.20.0-1}"

echo "== prerequisites =="
apt-get update -q
apt-get install -y -q --no-install-recommends ca-certificates curl gnupg2

echo "== Docker apt repository =="
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
. /etc/os-release
CODENAME="${UBUNTU_CODENAME:-$VERSION_CODENAME}"
cat > /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: ${CODENAME}
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
EOF
apt-get update -q
apt-get install -y -q docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "== NVIDIA Container Toolkit apt repository =="
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -q
if ! apt-get install -y -q \
      nvidia-container-toolkit="${NCT_VERSION}" \
      nvidia-container-toolkit-base="${NCT_VERSION}" \
      libnvidia-container-tools="${NCT_VERSION}" \
      libnvidia-container1="${NCT_VERSION}"; then
  echo "pinned toolkit version ${NCT_VERSION} not available; installing the repository's current version"
  apt-get install -y -q nvidia-container-toolkit
fi

echo "== configure the Docker runtime =="
nvidia-ctk runtime configure --runtime=docker
systemctl enable docker
systemctl restart docker
usermod -aG docker "${TARGET_USER}"

echo "== versions =="
docker --version
containerd --version
nvidia-ctk --version
dpkg-query -W -f='${Package} ${Version}\n' docker-ce docker-ce-cli containerd.io nvidia-container-toolkit nvidia-container-toolkit-base libnvidia-container-tools libnvidia-container1
cat /etc/docker/daemon.json
systemctl is-active docker
echo "setup_docker_wsl2.sh: done"
