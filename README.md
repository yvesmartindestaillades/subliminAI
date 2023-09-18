# subliminAI

## Installation

```bash
pip install -r requirements.txt
```

Export API tokens:

```
export REPLICATE_API_TOKEN=r8_*************************************

```

## Usage Examples

```bash
python main.py
```

Will create the following output and image:

```
Generating a single image with text: If you can read this raise your hands
prompt: A forest full of mystery and magic. Elves and fairies live here. A blue river with red stones.
control_strength: 0.8
```

```bash
python main.py -t "AI Arts & Crafts Hackathon" -p "Craft an artistic representation of a landscape graced by mountains and the flow of serene rivers." -n 3 -c 0.95
```
