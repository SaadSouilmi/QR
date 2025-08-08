```bash
curl -LO https://github.com/SaadSouilmi/QR/releases/download/v1.0.0/data.tar.gz.gpg
gpg -d data.tar.gz.gpg > data.tar.gz
```

### uv stuff 
```bash 

uv venv 
soruce .venv/bin/activate

# Installing depencies 
uv sync 
uv pip install -e . 
```