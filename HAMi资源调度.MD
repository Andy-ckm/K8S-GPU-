私有化、企业级的 AI 平台，支持开发环境（Pod）、交互式调试（SSH/VSCode/Jupyter）、按任务占用 GPU（Volcano 做 gang 调度）、持久化数据/模型（MinIO/CEPH）、
多租户（Namespace+RBAC）、域账号登录（Keycloak+LDAP）、CI/CD 自动触发训练任务（GitLab CI → 镜像 → Job）
