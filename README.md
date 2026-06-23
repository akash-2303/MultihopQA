# Cross-Document Bridge Multi-Hop Question Generation

Implementation of the bridge question method from **Pan et al. (2021)** — *"Unsupervised Multi-hop Question Answering by Question Generation"* — adapted to generate multi-hop questions across pairs of PDF documents within a topic folder, rather than from a pre-existing curated corpus.

## What this does

Given a folder of PDFs on the same topic, the pipeline:
1. Extracts text from every PDF
2. Runs Named Entity Recognition (NER) on each document
3. Finds entities shared across pairs of PDFs (bridge entities)
4. Generates a single-hop question per PDF using a T5 model fine-tuned on SQuAD
5. Converts the second question into a declarative sentence (QA2D)
6. Fuses the two questions into one multi-hop question using BERT-Large mask filling
7. Saves every question with full provenance: which two PDFs it came from, the bridge entity, the answer, and the supporting context sentences from each PDF

The result is questions that require reading **both** documents to answer — the bridge entity connects the two.

## Folder structure

```
MultiHopQA/
├── Input/
│   ├── Characterization/          # place PDFs here
│   └── Bean Shape Array/          # place PDFs here
├── output/                        # generated questions saved here (gitignored)
│   ├── Characterization/
│   │   └── bridge_questions.jsonl
│   └── Bean Shape Array/
│       └── bridge_questions.jsonl
├── notebooks/
│   └── bridge_mhqg_pipeline.ipynb
├── environment.yml
└── README.md
```

## Setup

Tested on RHEL with Python 3.10, CUDA 12.4, 4x NVIDIA RTX A6000.

### 1. Create conda environment

```bash
conda create -n mhqg python=3.10 -y
conda activate mhqg
```

### 2. Install PyTorch with CUDA

Check your driver version first:
```bash
nvidia-smi
```

Then install matching wheel (adjust `cu124` if your CUDA version differs):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install remaining dependencies

```bash
pip install transformers sentencepiece spacy pdfplumber pandas \
            nltk rouge-score protobuf tiktoken jupyter ipykernel
python -m spacy download en_core_web_sm
```

### 4. Register Jupyter kernel

```bash
python -m ipykernel install --user --name mhqg --display-name "Python (mhqg)"
```

### 5. Verify

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected: `2.6.x+cu124 True`

## Usage

1. Place your PDFs inside the relevant topic subfolder under `Input/`
2. Launch Jupyter and select the **Python (mhqg)** kernel
3. Open `notebooks/bridge_mhqg_pipeline.ipynb`
4. Update the paths in **Section 1** (Cell 1):
```python
INPUT_ROOT = Path("/your/path/to/MultiHopQA/Input")
OUTPUT_ROOT = Path("/your/path/to/MultiHopQA/output")
TOPICS = ["Characterization", "Bean Shape Array"]
```
5. Run all cells top to bottom

On first run, the notebook downloads ~3GB of model weights from HuggingFace:
- `valhalla/t5-base-qg-hl` — question generation
- `MarkS/bart-base-qa2d` — question to declarative sentence
- `bert-large-uncased` — mask fill fusion
- `gpt2` — fluency evaluation

Weights are cached in `~/.cache/huggingface/` after first download.

## Output format

Each line of `bridge_questions.jsonl` is a JSON record:

```json
{
  "topic": "Characterization",
  "source_pdf_a": "A Novel Approach to Photonic Packaging Leveraging.pdf",
  "source_pdf_b": "Photonic plug for scalable silicon photonics packaging.pdf",
  "bridge_entity": "silicon photonics",
  "answer_entity": "Intel",
  "question": "Who developed the coupling method in the work where silicon photonics was applied to wafer-scale integration?",
  "answer": "Intel",
  "context_a": "Intel developed the evanescent coupling method for silicon photonics integration.",
  "context_b": "Silicon photonics was applied to wafer-scale integration by leveraging existing CMOS fabs."
}
```

## Evaluation

The notebook includes reference-free evaluation metrics (Section 9) that do not require gold reference questions:

| Metric | What it measures |
|--------|-----------------|
| `distinct-1 / distinct-2` | Vocabulary diversity — closer to 1.0 means less repetitive output |
| `% answer in context` | Whether the answer entity appears in the stored supporting contexts |
| `% bridge in both contexts` | Whether the bridge entity genuinely appears in both PDF contexts — the real cross-document check |
| `% questions over 30 words` | The original paper flags these as usually low quality |
| `mean GPT-2 perplexity` | Fluency proxy — lower is more natural-sounding |

If you have hand-written gold reference questions, `reference_based_metrics()` in the evaluation section also computes BLEU and ROUGE-L.

## Method overview

This follows the bridge question generation approach from:

> Pan, B., et al. (2021). *Unsupervised Multi-hop Question Answering by Question Generation*. NAACL 2021.

The pipeline adapts their method from a clean pre-tokenised corpus setting to raw PDF input, and extends it to operate across all pairwise combinations of PDFs within a topic folder rather than a fixed two-passage setup.

**Key adaptations from the paper:**
- PDF text extraction and cleanup added upstream (the paper assumes clean text)
- Pairwise combination across N PDFs in a folder (not just fixed pairs)
- Full provenance tracking per generated question (source PDF filenames, contexts)
- `MarkS/bart-base-qa2d` used for QA2D step (original `domenicrosati/QA2D` checkpoint no longer available on HuggingFace)
- GPT-2 fluency filtering and BART paraphrasing (described in the paper) not yet implemented

**Known limitations:**
- `en_core_web_sm` NER is generic — for scientific/technical PDFs, domain NER (e.g. sciSpaCy) would find better bridge entities
- Bridge entities are matched by exact string — aliases and abbreviations are missed
- Multi-column PDF layouts cause spacing artifacts in extracted text that degrade downstream quality
- Generated questions are cross-document but not strictly multi-hop in the reasoning-chain sense — bridge entities are lexical connectors, not verified reasoning dependencies

## Requirements

```
python=3.10
torch>=2.6.0
transformers>=4.30.0
sentencepiece>=0.1.99
spacy>=3.5.0
pdfplumber>=0.10.0
pandas
nltk
rouge-score
protobuf
tiktoken
jupyterlab
ipykernel
```

## Citation

```bibtex
@inproceedings{pan2021unsupervised,
  title     = {Unsupervised Multi-hop Question Answering by Question Generation},
  author    = {Pan, Liangming and Chen, Wenhu and Xiong, Wenhan and Kan, Min-Yen and Wang, William Yang},
  booktitle = {Proceedings of NAACL},
  year      = {2021}
}
```
