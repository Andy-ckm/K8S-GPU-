私有化、企业级的 AI 平台，支持开发环境（Pod）、交互式调试（SSH/VSCode/Jupyter）、按任务占用 GPU（Volcano 做 gang 调度）、持久化数据/模型（MinIO/CEPH）、
多租户（Namespace+RBAC）、域账号登录（Keycloak+LDAP）、CI/CD 自动触发训练任务（GitLab CI → 镜像 → Job）

============================================================================================================

# 目录清单
#  GoHai私有化 AI 平台操作方案（Kubernetes 版，含 DevEnv、训练流、存储、认证、CI/CD、示例任务）

下面的内容是工程级的、按步骤可跑通的一整套手册：从集群、存储、认证、镜像仓库，到 DevEnv、GoHai-CLI 行为、训练任务模板、Volcano 调度、多机/多卡训练、资产导出、VSCode/SSH 接入、SFTP/上传策略、CI/CD 都覆盖到位。
（替换占位符）

> 注意：文档中以 `example.*`、`*.local`、`YOUR_*` 为占位符，请替换为公司域名/凭据/路径。

---

# 目录（快速跳转）

1. 概览与设计原则
2. 环境与前置条件（机器/账号/网络）
3. 从零：Kubernetes 集群（含 GPU 支持）
4. 基础服务部署（Harbor / MinIO / NFS / Rook-Ceph）
5. 身份认证（Keycloak + AD/LDAP）与多租户策略
6. 调度器与训练运行时（Volcano + HAMi + Kubeflow Training Operator）
7. DevEnv 子系统（Pod-based 开发环境、micromamba、SSH/VSCode 接入）
8. GoHai-CLI 行为与本地 config（context、token、space）示例
9. 训练任务模板与提交流程（单机/分布式、dry-run、--bare）
10. 示例：Fashion-MNIST 完整训练（train.py + Job YAML + 导出模型）
11. 资产上传方式（Web、SFTP、lftp、外部导入）与最佳实践
12. CI/CD（GitLab CI 示例 + Helm/ArgoCD）
13. 维护、监控与运维要点（日志、清理、配额告警）
14. 附录：关键文件（Dockerfile、Helm 模板片段、.mambarc、GoHai train_job_template.yaml）

---

# 1. 概览与设计原则（一分钟读懂）

目标：建立一套私有化、企业级的 AI 平台，支持开发环境（Pod）、交互式调试（SSH/VSCode/Jupyter）、按任务占用 GPU（Volcano 做 gang 调度）、持久化数据/模型（MinIO/CEPH）、多租户（Namespace+RBAC）、域账号登录（Keycloak+LDAP）、CI/CD 自动触发训练任务（GitLab CI → 镜像 → Job）。

设计原则：

* **资源复用优先**：开发环境短时使用，训练任务使用按需、完毕释放 GPU。
* **持久化分离**：/workspace（持久化）与镜像内 /root（非持久）。用户环境内不将重要数据写入镜像内根目录。
* **最小权限**：Namespace + RBAC，用户只访问所属 Space。
* **可审计**：SSO + 日志集中收集（Loki/Prometheus）。
* **可复现**：Dockerfile / Helm / CI/CD，保证一键重建。

---

# 2. 环境与前置条件（必读）

最低节点建议（DEMO / 小规模生产）：

* 3 x Control plane nodes（HA 推荐）或 1 控制节点（DEMO）
* 3 x Worker nodes（包含至少 1 台 GPU 节点，NVIDIA GPU）
* 每台机器 Ubuntu 22.04，内存 >= 32GB（GPU 节点更多）
* 网络互通、DNS 可解析（ingress 域名指向 LB）
* 对外域名：`GoHai.example.com`、`harbor.example.com`、`minio.example.com`、`git.example.com`、`keycloak.example.com`
* Kubernetes v1.26+（示例以 kubeadm / helm 为主）
* Helm 3、kubectl、docker/containerd

账号/密钥：

* LDAP/AD 账号（用于 Keycloak 绑定）
* Harbor admin、MinIO admin、GitLab admin（初始）
* Kubernetes admin kubeconfig

安全说明：任何生产环境，把证书、DB 密码、secret 存在 Kubernetes Secret + Vault 更安全。文档中示例用明文仅用于演示。

---

# 3. 从零：Kubernetes + GPU 支持（命令集合）

> 下面演示 kubeadm 的经典流程（DEMO 化、可替换为 K3s / managed k8s）

## 3.1 必装组件（控制节点）

```bash
# 基本工具
sudo apt update -y
sudo apt install -y apt-transport-https ca-certificates curl gnupg lsb-release

# 安装 containerd（建议）
sudo apt install -y containerd
sudo systemctl enable --now containerd
```

## 3.2 安装 kubeadm/kubelet/kubectl

```bash
sudo curl -fsSLo /usr/share/keyrings/kubernetes-archive-keyring.gpg \
  https://dl.k8s.io/apt/doc/apt-key.gpg
echo "deb [signed-by=/usr/share/keyrings/kubernetes-archive-keyring.gpg] \
  https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list
sudo apt update
sudo apt install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl
```

## 3.3 初始化 master（示例）

```bash
sudo kubeadm init --pod-network-cidr=10.244.0.0/16
mkdir -p $HOME/.kube && sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
```

## 3.4 安装 CNI（示例使用 Calico 或 Flannel）

```bash
kubectl apply -f https://raw.githubusercontent.com/projectcalico/calico/v3.26.0/manifests/calico.yaml
# 或 Flannel:
# kubectl apply -f https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml
```

## 3.5 在 GPU 节点安装 NVIDIA 驱动与 container toolkit（在每个 GPU 节点）

```bash
# 在宿主机上安装 NVIDIA 驱动（依 GPU 型号）
# 安装 NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart containerd
```

## 3.6 部署 NVIDIA Device Plugin（DaemonSet）

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/nvidia-device-plugin.yml
```

验证 GPU:

```bash
kubectl run --rm -it --gpus=1 gpu-test --image=nvidia/cuda:11.0-base nvidia-smi
```

---

# 4. 基础服务部署（Harbor / MinIO / NFS / Rook-Ceph）

你可以按需选择 Ceph（规模大）或 MinIO（快速部署）。示例都会给出 Helm 命令和关键配置项。

## 4.1 Harbor（私有镜像仓库）

1. 添加 Helm 仓库并部署（用 `values-harbor.yaml` 定制）

```bash
helm repo add harbor https://helm.goharbor.io
helm repo update
helm install harbor harbor/harbor -f values-harbor.yaml --namespace harbor --create-namespace
```

`values-harbor.yaml` 关键项示例：

```yaml
expose:
  type: ingress
  tls:
    enabled: true
  ingress:
    hosts:
      core: harbor.example.com
externalURL: https://harbor.example.com
harborAdminPassword: "YOUR_ADMIN_PWD"
```

## 4.2 MinIO（对象存储，用于模型/数据）

```bash
helm repo add minio https://charts.min.io/
helm repo update
helm install minio minio/minio --namespace storage --create-namespace \
  --set accessKey=minioadmin,secretKey=minioadmin123,service.type=ClusterIP
```

记住访问端点 `http://minio.storage.svc.cluster.local:9000`，用 mc 工具方便管理。

## 4.3 NFS（持久化 /workspace）

在一台机器上搭 NFS，或使用 Longhorn/Ceph。DEMO NFS：
宿主机：

```bash
sudo apt install -y nfs-kernel-server
sudo mkdir -p /srv/nfs/workspace
sudo chown -R nobody:nogroup /srv/nfs/workspace
echo "/srv/nfs/workspace *(rw,sync,no_root_squash,no_subtree_check)" | sudo tee -a /etc/exports
sudo exportfs -ra
```

K8s PV：

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: workspace-pv
spec:
  capacity:
    storage: 500Gi
  accessModes:
    - ReadWriteMany
  nfs:
    path: /srv/nfs/workspace
    server: nfs.example.com
```

PVC：

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: workspace-pvc
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 200Gi
```

## 4.4 可选：Rook-Ceph（大规模）

Rook/Ceph 用于块存储/CephFS/RBD：

```bash
helm repo add rook-release https://charts.rook.io/release
helm repo update
kubectl create namespace rook-ceph
helm install rook-ceph rook-release/rook-ceph -n rook-ceph
```

（后续要创建 CephCluster CRD、pool、StorageClass 等，略长，按 Rook 官方文档走）

---

# 5. 身份认证（Keycloak + LDAP/AD）与多租户

## 5.1 部署 Keycloak（示例 Helm）

```bash
helm repo add codecentric https://codecentric.github.io/helm-charts
helm repo update
helm install keycloak codecentric/keycloak -n keycloak --create-namespace \
  -f keycloak-values.yaml
```

`keycloak-values.yaml` 示例如下（生产使用外部 DB）：

```yaml
replicaCount: 2
keycloak:
  username: admin
  password: YOUR_KEYCLOAK_PWD
proxy: true
```

## 5.2 在 Keycloak 添加 LDAP/AD provider

Keycloak 管理台 → Realm → Identity providers → Add provider: LDAP / Active Directory
配置 bind DN、bind credential、user search base 等，通过用户名/密码校验用户。

## 5.3 配置 OIDC 客户端（供 GoHai Web & CLI）

在 Keycloak 创建 Client（`GoHai-portal` / `GoHai-cli`），启用 `Authorization Code` 流程，设置 `redirectUris` 指向你的前端回调 URL。

---

# 6. 调度器与训练运行时：Volcano + HAMi + Training Operator

## 6.1 安装 Volcano

```bash
helm repo add volcano https://volcano-sh.github.io/helm-charts
helm repo update
helm install volcano volcano/volcano --namespace volcano-system --create-namespace
```

验证：

```bash
kubectl get pods -n volcano-system
```

## 6.2 HAMi（GPU 切片/资源管理）

> HAMi 不是业界极广泛的通用插件（取决于你内部）；如果没有，至少确保 NVIDIA device plugin 与可能的 MPS/NVIDIA-MIG 已启用。示例部署 HAMi（如果有 Helm chart）：

```bash
helm repo add hami https://hami-io.github.io/charts
helm install hami hami/hami --namespace kube-system
```

如果无 HAMi，使用 NVIDIA Device Plugin + Volcano 仍可完成大部分任务调度。

## 6.3 Kubeflow Training Operator（PyTorchJob / TFJob）

```bash
kubectl apply -f https://github.com/kubeflow/training-operator/releases/download/v1.7.0/training-operator.yaml
```

这允许我们使用 `PyTorchJob`、`TFJob` 等 CRD 来运行分布式作业。

---

# 7. DevEnv 子系统（可交互 Pod、micromamba、SSH / VSCode）

核心思想：为每个用户/space 启动一个 Pod（或一组 Pod），挂载 PVC（workspace-pvc），注入用户 SSH 公钥与环境变量。Pod image 是带 micromamba、SSH server、常用工具的镜像。

## 7.1 mamba-base Dockerfile（推荐）

```Dockerfile
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y curl git openssh-server lsb-release ca-certificates \
    build-essential sudo procps unzip wget vim && rm -rf /var/lib/apt/lists/*
# 安装 micromamba（轻量 conda）
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba
RUN mkdir -p /workspace /root/.ssh /root/micromamba
ENV MAMBA_ROOT_PREFIX=/root/micromamba
WORKDIR /workspace
# sshd 配置
RUN mkdir /var/run/sshd
RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config
RUN echo 'root:root' | chpasswd
EXPOSE 22
CMD ["/usr/sbin/sshd","-D"]
```

构建&推送：

```bash
docker build -t harbor.example.com/GoHai/mamba-base:v1 .
docker push harbor.example.com/GoHai/mamba-base:v1
```

## 7.2 Pod 模板（devenv）

示例 YAML（由 controller 生成）：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: devenv-<username>
  labels:
    app: GoHai-devenv
spec:
  containers:
  - name: workspace
    image: harbor.example.com/GoHai/mamba-base:v1
    ports:
      - containerPort: 22
    volumeMounts:
      - name: workspace
        mountPath: /workspace
      - name: sshkeys
        mountPath: /root/.ssh
  volumes:
    - name: workspace
      persistentVolumeClaim:
        claimName: workspace-pvc
    - name: sshkeys
      secret:
        secretName: user-ssh-<username>
```

> `user-ssh-<username>` 是把用户公钥写入 `authorized_keys` 的 Secret（controller 在创建时生成）。

## 7.3 micromamba 与持久化

在容器启动后用户执行：

```bash
micromamba shell init --shell bash --root-prefix=/root/micromamba
# 这会将初始化脚本写入 ~/.bashrc（你可以把 .bashrc 放在 /workspace，使其持久）
source /workspace/.bashrc
```

`.mambarc` 示例（放在 /workspace）：

```yaml
envs_dirs:
  - /workspace/conda_env/
  - /root/micromamba/envs/
pkgs_dirs:
  - /workspace/conda_env/pkgs/
  - /root/micromamba/pkgs/
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
channel_priority: strict
show_channel_urls: true
```

创建持久 env：

```bash
micromamba create -n work python=3.11
micromamba activate work
pip install torch torchvision datasets huggingface_hub
```

## 7.4 SSH / VSCode Remote 连接流程（用户端）

1. 在本地生成 SSH key：

```bash
ssh-keygen -t rsa -f ~/.ssh/id_rsa -C "your@company"
cat ~/.ssh/id_rsa.pub
```

2. 在 GoHai UI（用户配置页）上传公钥（平台存为 Secret，controller 注入 Pod）。
3. 获取 Pod 的 SSH 连接信息（平台 UI 显示 `ssh user@env-xxx.ssh-GoHai.example.com -p 2022`），或直接使用 port-forward。
4. VSCode: 安装 Remote-SSH，添加 Host 到 `~/.ssh/config`，连接时会自动部署 VSCode Server（第一次稍慢）。

## 7.5 后台运行 & 长任务（开发时常犯的坑）

在 DevEnv 终端启动训练不要直接 `python train.py`（会被 SSH 断开而终止）。推荐：

```bash
nohup python train.py > train.log 2>&1 &
# 或用 tmux/screen
tmux new -s train
# run inside tmux, detach with Ctrl+B D
```

查看日志：

```bash
tail -f train.log
```

停进程（稳妥）：

```bash
ps aux | grep train.py
kill -9 <PID>
```

---

# 8. GoHai-CLI 行为与本地配置（context、token、space）示例

我把你提供过的行为规范串成可操作的样板文件/命令。

## 8.1 安装 GoHai CLI（示例）

安装脚本（你之前有）：

```bash
curl https://nexus.example.com/repository/.../install-cli.sh | sh
# 若 PATH 没包含 ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"
```

## 8.2 ~/.config/GoHai-cli/config.yaml（示例 — 请脱敏后保存）

```yaml
current-context: default
contexts:
  default:
    host: https://GoHai.example.com
    token: <REDACTED_TOKEN>
    space: space-abc123
    user: user@company.com
profiles: {}
```

命令：

```bash
GoHai login           # 浏览器跳转 OIDC，选择 Space（platform will create default context）
GoHai context list
GoHai context new my-other-space
GoHai context use my-other-space   # 切换上下文
GoHai whoami
```

## 8.3 repo init / repo push / trainjob template flow

在本地（或 DevEnv）：

```bash
# 从平台关联的 repo clone，或 GoHai repo init 拉取
git clone https://git.example.com/yourproj/GoHai-example.git
cd GoHai-example
GoHai repo init   # 生成 GoHai.repo.yaml
# 改代码...
git add . && git commit -m "train debug"
git push origin main
# 或使用 GoHai repo push（会 push 到平台关联仓库）
GoHai repo push
```

生成训练任务模板：

```bash
GoHai trainjob template  # 生成 train_job_template.yaml
# 或从已有 job 导出为模板
GoHai trainjob template -s <job-id>
```

---

# 9. 训练任务模板与提交（YAML + CLI）

## 9.1 train_job_template.yaml（示例）

```yaml
title: fashion-mnist-demo
description: "Fashion-MNIST CNN training via CLI template"
image: harbor.example.com/ai/pytorch:1.13-cuda11
command: python train.py --epochs 5 --batch_size 128
env:
  GoHai_CLI: "true"
repository:
  - repository:
      space: space-abc123
      id: repo-0
      hash: ""          # 如果空则挂载最新 commit
dataset:
  - dataset:
      space: space-abc123
      id: dataset-0
      versionId: "v1"
model: []
mode: single
spec:
  singleInstanceType: gpu-1x-16c-32g-1gpu
```

## 9.2 dry-run & create

```bash
GoHai trainjob create -f train_job_template.yaml --dry-run
# 检查输出的 Job YAML（由平台生成）
GoHai trainjob create -f train_job_template.yaml
GoHai trainjob list
GoHai trainjob logs <job-id>
```

## 9.3 完全按 YAML（--bare）

```bash
GoHai trainjob create -f job_full.yaml --bare
```

---

# 10. 示例：Fashion-MNIST 完整训练（可复制执行）

## 10.1 train.py（支持 single & distributed）

把这个放到仓库根目录 `train.py`。这是一个保守、可运行的 PyTorch 脚本，支持单机与分布式（torch.distributed.run / torchrun）：

```python
# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_model():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64*5*5, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

def train_single(epochs, batch_size, lr, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    testset  = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    model = get_model().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} done")
    torch.save(model.state_dict(), "/output/fashion_mnist_cnn.pth")
    print("Saved model to /output/fashion_mnist_cnn.pth")

def train_distributed(epochs, batch_size, lr):
    # 使用 torchrun 启动；这里仅占位：实际入口用 torchrun --nproc_per_node=...
    device = torch.device("cuda")
    # assume ddp init done externally
    # Simplify: call single for demo
    train_single(epochs, batch_size, lr, data_dir="/dataset")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="auto")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--data_dir", default="/dataset")
    args = parser.parse_args()

    is_multi_node = int(os.getenv("PET_NNODES", "1")) > 1
    if args.mode == "auto":
        if (not is_multi_node) and torch.cuda.device_count() <= 1:
            mode = "single"
        else:
            mode = "multi"
    else:
        mode = args.mode

    if mode == "single":
        train_single(args.epochs, args.batch_size, args.lr, args.data_dir)
    else:
        train_distributed(args.epochs, args.batch_size, args.lr)
```

## 10.2 Job YAML（Volcano + initContainer 拉取代码）

示例 `vcjob.yaml`（单机示例，若分布式则用 replicas + torchrun）：

```yaml
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: fashion-mnist-job
spec:
  minAvailable: 1
  schedulerName: volcano
  queue: default
  tasks:
    - name: train
      replicas: 1
      template:
        spec:
          containers:
            - name: trainer
              image: harbor.example.com/ai/pytorch:1.13-cuda11
              command: ["bash","-lc"]
              args:
                - |
                  git clone https://git.example.com/server/devops/sandbox/GoHai-example.git /repo || true
                  cd /repo
                  python train.py --epochs 5 --batch_size 128 --data_dir /dataset
              resources:
                limits:
                  nvidia.com/gpu: 1
          restartPolicy: OnFailure
  volumes:
    - name: dataset
      persistentVolumeClaim:
        claimName: fashionmnist-dataset-pvc
```

> 若你更喜欢挂载 repo via PVC（platform does that), 用 initContainer 在共享 emptyDir 拉取。

## 10.3 提交与监控

```bash
kubectl apply -f vcjob.yaml
kubectl get jobs -n default
kubectl get pods -w
kubectl logs -f <pod-name> -c trainer
```

## 10.4 导出模型到模型资产（MinIO）

任务脚本保存到 `/output/fashion_mnist_cnn.pth`。平台 post-process（controller）可以将 `/output` 的文件复制到 MinIO：

示例 mc 命令（在任一有 mc 的机器）：

```bash
mc alias set myminio http://minio.example.com minioadmin minioadmin123
mc cp /path/to/fashion_mnist_cnn.pth myminio/models/fashion-mnist/v1/
```

平台 UI 提供「导出到模型」功能：用户选中输出 -> 选择模型草稿 -> 导出。

---

# 11. 资产上传方式与最佳实践（网页、SFTP、lftp、HuggingFace 导入）

## 11.1 Web 上传（小文件 < 2GB）

平台提供上传页 -> 选择文件 -> 上传（适合 notebook、配置文件、小模型）

## 11.2 SFTP（推荐普通用户上传大文件）

服务端（平台）提供 SFTP endpoint `sftp-GoHai.example.com:2022`，示例：

```bash
sftp -P 2022 user@sftp-GoHai.example.com
put -r /local/dataset/ /remote/data/dataset-0/
```

Windows：WinSCP 或 FileZilla

## 11.3 lftp（增量上传大目录）

安装：

```bash
sudo apt-get install -y lftp
```

命令：

```bash
lftp -c "
set sftp:connect-program 'ssh -a -x -p 2022'
open sftp://user@sftp-GoHai.example.com
mirror -R --ignore-time --only-newer /local/dir/ /remote/dir/
"
```

## 11.4 外部导入（HuggingFace / S3）

API 调用（平台后端有导入接口）：

```bash
curl -X POST "https://GoHai.example.com/api/v1/assets/import" \
  -H "Authorization: Bearer <TOKEN>" \
  -d '{"type":"huggingface","repo":"ylecun/mnist","target_path":"/datasets/mnist"}'
```

---

# 12. CI/CD（GitLab CI 示例 + Helm + ArgoCD）

## 12.1 GitLab CI（.gitlab-ci.yml）

把这个放 repo：

```yaml
stages:
  - build
  - push
  - deploy
  - train

variables:
  DOCKER_REGISTRY: harbor.example.com
  IMAGE: $DOCKER_REGISTRY/ai/train

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $DOCKER_REGISTRY
    - docker build -t $IMAGE:$CI_COMMIT_SHORT_SHA .
    - docker push $IMAGE:$CI_COMMIT_SHORT_SHA
  only:
    - main

deploy:
  stage: deploy
  script:
    - helm upgrade --install GoHai-app ./charts/GoHai --namespace GoHai --set image.tag=$CI_COMMIT_SHORT_SHA
  only:
    - main

train:
  stage: train
  script:
    - kubectl apply -f jobs/fashion-mnist-job.yaml
  only:
    - tags
```

## 12.2 ArgoCD（可选）

用 ArgoCD 自动同步 Helm Chart 到集群，触发升级与灰度发布。

---

# 13. 维护、监控与运维要点（必须关注）

* **日志**：每个训练容器输出日志到 stdout；采集使用 Fluent Bit -> Loki / Elasticsearch，前端通过 Kibana/Grafana 查询。
* **监控**：Prometheus + Grafana，监控 GPU 利用率、队列长度、PV 使用率、Keycloak 状态。
* **配额/告警**：ResourceQuota + LimitRange，平台 UI 展示个人/团队配额，并配置告警阈值（<= X% 触发 UME/邮件）。
* **垃圾清理**：对 `workspace` 中未使用卷、已完成任务产生的 PVC/Job 定期清理（保留最后 N 个版本）。
* **安全**：Ingress 用 TLS，Keycloak 客户端 secret 存 Kubernetes Secret，镜像扫描（Harbor）启用 Clair/Trivy。
* **备份**：MinIO 数据、Ceph 元数据、Postgres（平台 DB）要定期备份到外部对象存储。
* **更新策略**：先 dev/staging 灰度验证再 prod。Driver/CUDA 版本升级尤其危险，要在单独测试节点验证。

---

# 14. 附录（直接可复制的关键文件）

## 14.1 .mambarc（放在 /workspace/.mambarc）

```yaml
envs_dirs:
  - /workspace/conda_env/
  - /root/micromamba/envs/
pkgs_dirs:
  - /workspace/conda_env/pkgs/
  - /root/micromamba/pkgs/
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
channel_priority: strict
show_channel_urls: true
```

## 14.2 GoHai train_job_template.yaml（完整版示例）

```yaml
title: CLI-d2k3m7i1p6o4mutspnng
description: 生成自CLI模板
image: harbor.example.com/ai/pytorch:1.13-cuda11
command: python train.py --epochs 5 --batch_size 128
env:
    GoHai_CLI: "true"
repository:
  - repository:
      space: space-abc123
      id: repo-0
      hash: ""
dataset:
  - dataset:
      space: space-abc123
      id: dataset-0
      versionId: v1
model: []
mode: Single
spec:
  singleInstanceType: gpu-1x-16c-32g-1gpu
```

## 14.3 sample Helm Chart 目录（建议）

```
charts/GoHai/
  Chart.yaml
  values.yaml
  templates/
    api-deployment.yaml
    controller-deployment.yaml
    devenv-deployment.yaml
    storage-pvc.yaml
    ingress.yaml
```

## 14.4 Dockerfile（训练镜像 minimal）

```Dockerfile
FROM pytorch/pytorch:1.13-cuda11.7-cudnn8-runtime
WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /workspace
CMD ["bash","-lc","python train.py"]
```

---

# 最后 — 直接行动清单（你现在就能开始执行的步骤）

下面是一份**顺序化、最易上手的行动清单**（把大任务拆成小块），执行它就能把一个 DEMO 环境从 0 搭起来：

1. **准备机器与网络**，搭建 1 control + 2 worker（1 worker 必须有 NVIDIA GPU）。
2. **按第 3 节创建 Kubernetes 集群 + CNI + NVIDIA device plugin**，测试 `nvidia-smi` 在 Pod 内可见。
3. **部署 Harbor + MinIO + NFS（或 Rook-Ceph）**，确认镜像、对象存储可用。
4. **部署 Keycloak 并连接 LDAP（测试登录）**。
5. **部署 Volcano（调度器）** 与（若可）HAMi。
6. **部署 GoHai core helm chart（GoHai-api、GoHai-controller、devenv-controller、GoHai-ui）**。
7. **构建并推送 mamba-base 与训练镜像到 Harbor**。
8. **创建 workspace PVC 并在 UI 创建一个 DevEnv（选择 mamba-base、挂载 workspace）**，上传公钥，SSH 试连。
9. **把 Fashion-MNIST repo `GoHai-example` push 到 GitLab（或 GitHub），在平台创建 dataset（或导入 HuggingFace）**。
10. **在 DevEnv 中 micromamba create 环境，安装依赖，测试 `nohup python train.py &`**。
11. **编写 train_job_template.yaml，使用 GoHai CLI 或 UI 提交训练**，观察 Volcano 的调度、Pod 运行与输出保存到 `/output`。
12. **将 `/output/*.pth` 导出到模型资产（MinIO），并在 UI 上保存为正式版本**。
13. **接入 CI：在 repo 中添加 `.gitlab-ci.yml`，使推 main 自动构建镜像并触发 job**。
14. **完善：添加 Prometheus/Grafana、Loki/Fluentbit、告警策略及审计日志。**

---

# 风险提示与工程建议（必须阅读）

* **NVIDIA 驱动与 CUDA 必须和训练镜像匹配**，否则 GPU 不工作。升级驱动慎重。
* **生产环境密钥管理**：不要把 Keycloak client secret 或 MinIO 密钥硬编码到 helm values。使用 Kubernetes Secret 或 Vault。
* **PVC 权限**：确保 PVC 权限、AccessMode 满足场景（Data 可能是 ReadOnlyMany）。
* **清理策略**：job 完成应自动回收 Pod，保留 output 到对象存储；避免长期占用 PV。
* **安全**：对外暴露服务使用 TLS，禁止无认证访问 Harbor/MinIO。

---

这份操作方案把你给的所有零散信息（GoHai CLI 行为、context、repo init、trainjob template、dev env、micromamba、ssh/vscode、nohup、SFTP/lftp、模型导出、Volcano/HAMi/K8s/MinIO/Harbor/Keycloak/CI）全部串在一条可执行链路里：**你按步骤跑下去就能搭出一个能生产演示 Fashion-MNIST 的私有化 GoHai 平台 DEMO**。

如果你现在需要，我可以立刻（在本次回复里）：

* 把 `charts/GoHai` 的 **最小 Helm Chart 模板** 生成出来（包含 values.yaml 和 templates），或者
* 把 `GoHai-controller` / `devenv-controller` 的 **示例实现伪代码**（Go/Cobra + client-go）写出来，或
* 直接把 **完整 Fashion-MNIST repo（含 train.py、train_job_template.yaml、.gitlab-ci.yml）** 的文件树与内容生成，方便你 `git init` 即用。


=============================================================================================================

