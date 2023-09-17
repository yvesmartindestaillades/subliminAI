# subliminAI

## Installation

```bash
pip install -r requirements.txt
llm install llm-replicate
llm keys set replicate
llm replicate add a16z-infra/llama13b-v2-chat \
  --chat --alias llama2

```

Export API tokens:

```
export REPLICATE_API_TOKEN=r8_*************************************

```

## Test installation

```bash
llm -m llama2 "Ten great names for a pet pelican"
```
