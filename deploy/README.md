# Transcriber server runtime bundle

Copy this directory to the server (for example `/opt/transcriber`).

Server uses only these files and Docker/GHCR:
- no git clone
- no image build on server
- no python/pip setup on server

Operator commands:
```bash
cd /opt/transcriber
./install.sh prod-<sha>
./run_job.sh /data/input/file.json
./rollback.sh
```
