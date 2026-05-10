# Models Benchmark for Audio Storage in Vector Databases — Milvus

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Milvus](https://img.shields.io/badge/Vector%20Database-Milvus-green)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![Docker](https://img.shields.io/badge/Infrastructure-Docker-lightgrey)
![Grade](https://img.shields.io/badge/Final%20Grade-17%2F20-success)

Benchmarking framework for comparing audio embedding models when storing and querying audio representations in a vector database. The project uses **Milvus** as the vector database backend and provides a **Streamlit dashboard** for analysing extraction time, insertion time, memory usage, CPU/RAM/GPU behaviour, and vector-search results.

> **Academic grade:** 17/20  
> **Academic context:** MSc in Advanced Computing — University of Minho

---

## Table of Contents

- [Project Overview](#project-overview)
- [Main Goals](#main-goals)
- [System Architecture](#system-architecture)
- [Benchmark Pipeline](#benchmark-pipeline)
- [Audio Embedding Models](#audio-embedding-models)
- [Metrics Collected](#metrics-collected)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Dashboard](#dashboard)
- [Milvus Collections](#milvus-collections)
- [Outputs](#outputs)
- [Example Workflow](#example-workflow)
- [Technical Notes](#technical-notes)
- [Future Work](#future-work)
- [Authors](#authors)

---

## Project Overview

Vector databases are increasingly important in AI systems that depend on similarity search, retrieval-augmented generation, recommendation, and multimedia search. While text embeddings are now common, audio embeddings introduce additional constraints: models have different sampling-rate requirements, embedding dimensionalities, memory footprints, extraction latency, and retrieval behaviour.

This project benchmarks different **audio embedding models** for storage and similarity search in **Milvus**. It processes audio files, extracts vector representations, stores them with metadata, performs search queries, and generates benchmark artefacts for comparison.

The project is relevant for:

- audio similarity search;
- music recommendation systems;
- speech and voice retrieval;
- multimedia search engines;
- AI retrieval pipelines using non-textual data;
- benchmarking vector database workloads.

---

## Main Goals

The benchmark evaluates the complete path from raw audio to vector search:

1. **Audio loading and preprocessing**
   - Load `.wav`, `.mp3`, `.flac`, and `.ogg` files.
   - Resample according to each model requirement.
   - Extract audio duration, sample rate, channel count, and metadata.

2. **Embedding extraction**
   - Compare multiple pre-trained audio embedding models.
   - Measure extraction latency and resource usage.
   - Store model-specific vector dimensions.

3. **Vector storage**
   - Insert embeddings into Milvus collections.
   - Store audio metadata alongside vectors.
   - Use cosine similarity as the search metric.

4. **Vector search**
   - Run similarity searches per model.
   - Store the score of the first retrieved result.
   - Compare retrieval behaviour across embeddings.

5. **Result analysis**
   - Save CSV benchmark results.
   - Generate static plots.
   - Provide an interactive Streamlit dashboard.

---

## System Architecture

```text
Audio Dataset
     │
     ▼
Audio Metadata Extraction
     │
     ▼
Embedding Models
     │
     ▼
Benchmark Engine
     │
     ├── Extraction time
     ├── Insertion time
     ├── Peak memory
     ├── CPU / RAM / GPU usage
     └── Audio metadata
     │
     ▼
Milvus Vector Database
     │
     ▼
Similarity Search
     │
     ▼
Benchmark Reports + Streamlit Dashboard
```

---

## Benchmark Pipeline

The benchmark is implemented mainly in `audio_benchmark.py` and follows this execution flow:

```text
1. Load embedding models
2. Discover audio files in audio_data/
3. For each model:
      3.1 Create a dedicated Milvus collection
      3.2 For each audio file:
              - extract embedding
              - measure time and resources
              - extract metadata
              - insert vector + metadata into Milvus
4. Save detailed results to CSV
5. Generate visual plots
6. Run vector search tests
7. Store search-score summaries
8. Visualise results through Streamlit
```

---

## Audio Embedding Models

The benchmark compares the following models:

| Model | Framework / Source | Embedding Dimension | Notes |
|---|---:|---:|---|
| `wav2vec2` | Hugging Face / PyTorch | 768 | Speech-oriented representation model. |
| `vggish` | TensorFlow Hub | 128 | Compact audio embedding model. |
| `openl3` | OpenL3 / TensorFlow | 512 | General-purpose audio/music embeddings. |
| `yamnet` | TensorFlow Hub | 1024 | Audio event model trained on AudioSet-style labels. |
| `clap` | Hugging Face / PyTorch | 512 | Contrastive language-audio pretraining model. |
| `ast` | Hugging Face / PyTorch | 768 | Audio Spectrogram Transformer model. |

Each model produces vectors with different dimensionality, which directly affects memory usage, insertion cost, and search behaviour inside Milvus.

---

## Metrics Collected

For each processed audio file and embedding model, the benchmark records:

| Metric | Description |
|---|---|
| `modelo` | Name of the embedding model. |
| `ficheiro` | Processed audio file. |
| `tempo_extracao` | Time required to generate the embedding. |
| `tempo_insercao` | Time required to insert the vector into Milvus. |
| `memoria_peak_mb` | Peak memory usage during extraction. |
| `dimensao` | Embedding vector dimension. |
| `tamanho_audio_mb` | Audio file size. |
| `duracao_audio_s` | Audio duration in seconds. |
| `cpu_percent_before` / `cpu_percent_after` | CPU usage before and after extraction. |
| `ram_percent_before` / `ram_percent_after` | RAM usage before and after extraction. |
| `gpu_usage` | GPU utilisation when available. |
| `embedding` | Stored vector representation. |

---

## Repository Structure

```text
Models-Benchmark-for-Audio-Storage-in-Vectorial-Databases-Milvus/
│
├── audio_benchmark.py          # Main benchmark pipeline
├── audio_db.py                 # Audio metadata and Milvus insertion helpers
├── dashboard.py                # Streamlit interactive dashboard
├── db_config.py                # Milvus client configuration
├── main.py                     # Additional project entry point
├── test_connection.py          # Milvus connection test
│
├── docker-compose.yml          # Milvus / database infrastructure
├── makefile                    # Utility commands
├── requirements.txt            # Python dependencies
├── requirements_jupyter.txt    # Notebook dependencies
│
├── benchmark_relatorio.ipynb   # Notebook-based analysis
├── benchmarks/                 # Generated benchmark results
├── audio_data/                 # Input audio dataset
├── volumes/                    # Milvus persistent data volumes
│
└── Apresentação Projeto.pptx   # Project presentation
```

---

## Requirements

The project uses Python, Milvus, Docker, Streamlit, and several machine-learning/audio-processing libraries.

Recommended environment:

- Python 3.10+
- Docker
- Docker Compose
- Milvus running locally on port `19530`
- Optional GPU support for model inference acceleration

Core Python libraries used by the project include:

```text
numpy
pandas
librosa
pymilvus
torch
transformers
tensorflow
tensorflow-hub
openl3
mutagen
psutil
GPUtil
matplotlib
seaborn
plotly
streamlit
tqdm
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/diogocsilva12/Models-Benchmark-for-Audio-Storage-in-Vectorial-Databases-Milvus.git
cd Models-Benchmark-for-Audio-Storage-in-Vectorial-Databases-Milvus
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If the requirements file is incomplete in your local copy, install the core dependencies manually:

```bash
pip install numpy pandas librosa pymilvus torch transformers tensorflow tensorflow-hub openl3 mutagen psutil GPUtil matplotlib seaborn plotly streamlit tqdm
```

---

## Running the Project

### 1. Start Milvus

```bash
docker-compose up -d
```

The default Milvus URI used by the project is:

```text
http://localhost:19530
```

If the deployment includes Attu, the web interface is usually available at:

```text
http://localhost:8000
```

### 2. Test the Milvus connection

```bash
python test_connection.py
```

### 3. Add audio files

Place the audio dataset inside:

```text
audio_data/
```

Supported formats:

```text
.wav
.mp3
.flac
.ogg
```

### 4. Run the benchmark

```bash
python audio_benchmark.py --audio_dir audio_data --max_files 30 --repeat 1
```

Example with more files and repeated runs:

```bash
python audio_benchmark.py --audio_dir audio_data --max_files 100 --repeat 3
```

Arguments:

| Argument | Default | Description |
|---|---:|---|
| `--audio_dir` | `audio_data` | Directory containing input audio files. |
| `--max_files` | `30` | Maximum number of audio files to process. |
| `--repeat` | `1` | Number of repetitions per file/model. |

---

## Dashboard

Run the Streamlit dashboard with:

```bash
streamlit run dashboard.py
```

The dashboard allows the user to:

- select benchmark result folders;
- load and inspect CSV results;
- filter models dynamically;
- analyse extraction time, insertion time, memory, CPU, RAM, and GPU usage;
- compare models with bar charts, boxplots, scatter plots, violin plots, pairplots, 3D plots, and heatmaps;
- export filtered benchmark data.

---

## Milvus Collections

The benchmark creates one Milvus collection per model and processed dataset size.

Collection naming pattern:

```text
audio_<model_name>_<number_of_files>
```

Examples:

```text
audio_wav2vec2_30
audio_vggish_30
audio_openl3_30
audio_yamnet_30
audio_clap_30
audio_ast_30
```

Each inserted record contains:

- integer ID;
- embedding vector;
- filename;
- title;
- duration;
- authors;
- channels;
- sample rate;
- album;
- genre.

The vector search metric used in the benchmark is:

```text
COSINE
```

---

## Outputs

Benchmark outputs are written to folders such as:

```text
benchmarks/benchmark_30/
benchmarks/benchmark_30_1/
benchmarks/benchmark_100/
```

Typical generated files include:

```text
resultados_benchmark.csv
benchmark.log
tempos_pesquisa.csv
tempo_extracao_seaborn.png
tempo_insercao_seaborn.png
memoria_pico_seaborn.png
scatter_tamanho_vs_tempo.png
correlacao_duracao_tempo.png
boxplot_tempo_extracao.png
scatter3d_tempo_memoria_tamanho.png
evolucao_cpu.png
evolucao_ram.png
evolucao_gpu.png
score_primeiro_resultado.png
```

---

## Example Workflow

```bash
# Start Milvus
Docker-compose up -d

# Activate Python environment
source .venv/bin/activate

# Run benchmark on 30 audio files
python audio_benchmark.py --audio_dir audio_data --max_files 30 --repeat 1

# Open dashboard
streamlit run dashboard.py
```

Then open the Streamlit URL shown in the terminal and select the generated `benchmarks/benchmark_*` folder.

---

## Technical Notes

### Model loading

The benchmark loads all embedding models at startup to avoid reloading models for every audio file. This improves fairness in the per-file extraction benchmark, because the measured extraction time is focused on inference rather than repeated model initialization.

### Audio metadata

Metadata extraction combines `mutagen` and `librosa`:

- `mutagen` extracts title, artist, album, and genre when available;
- `librosa` extracts duration, sample rate, and channel information.

### Fallback behaviour

The implementation includes fallback dummy classes for TensorFlow Hub models such as VGGish and YAMNet when model loading fails. This allows the pipeline to continue executing, but results produced under fallback mode should not be interpreted as valid model-performance results for those models.

### Milvus storage design

Each model receives its own collection because embedding dimensionality differs between models. This avoids dimension conflicts and keeps the benchmark comparison clean.

### Evaluation limitation

The benchmark currently records the score of the first result returned by Milvus. For a more rigorous retrieval-quality evaluation, future versions should include labelled query sets and metrics such as Recall@K, Precision@K, mAP, NDCG, and embedding clustering quality.

---

## Future Work

Potential improvements:

- add labelled audio datasets for retrieval-quality evaluation;
- implement Recall@K, Precision@K, NDCG, and mAP;
- benchmark different Milvus index types such as IVF_FLAT, HNSW, DISKANN, and AUTOINDEX;
- compare CPU vs GPU inference paths;
- add batch insertion and batch-query benchmarks;
- separate model-loading time from embedding-extraction time explicitly;
- add reproducible experiment configuration files;
- include dataset cards and benchmark manifests;
- add automatic report generation in Markdown/PDF;
- integrate experiment tracking with MLflow or Weights & Biases;
- add Docker image for the full benchmark application;
- extend the dashboard with query-by-example audio upload.

---

## Authors

- **Diogo Coelho da Silva**
- **João Barbosa**
- **Pedro Oliveira**

MSc in Advanced Computing  
University of Minho

---

## Academic Context

This project was developed as part of academic coursework at the University of Minho and received a final grade of **17/20**.
