# subliminAI

## Installation

```bash
pip install -r requirements.txt
llm install llm-replicate
llm keys set replicate # set your key
llm replicate add \
  replicate/llama70b-v2-chat \
  --chat --alias llama70b
```

Export API tokens:

```
export REPLICATE_API_TOKEN=r8_*************************************

```

## Test installation

```bash
llm -m llama2 "Ten great names for a pet pelican"
```
