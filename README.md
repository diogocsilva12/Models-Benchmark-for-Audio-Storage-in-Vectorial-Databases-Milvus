# Models Benchmark for Audio Storage in Vector Databases (Milvus)

Benchmarking framework to evaluate **audio embedding models for storage
and retrieval in vector databases**, using **Milvus** as the backend.

**Final Grade:** 17/20

This project explores how different embedding models perform when
storing and querying **audio representations in a vector database**,
focusing on performance, scalability, and retrieval quality.

------------------------------------------------------------------------

# Overview

Vector databases are increasingly used for **similarity search and AI
applications** such as:

-   audio search
-   speech recognition systems
-   multimedia retrieval
-   AI recommendation systems

This project benchmarks different **audio embedding models** to
determine how efficiently they can be stored and retrieved using
**Milvus**, a high-performance vector database.

The system allows:

-   generating embeddings from audio data
-   storing vectors in Milvus
-   benchmarking retrieval performance
-   visualizing results through an interactive dashboard

------------------------------------------------------------------------

# Architecture

    Audio Dataset
          │
          ▼
    Embedding Models
          │
          ▼
    Vector Storage (Milvus)
          │
          ▼
    Similarity Search
          │
          ▼
    Interactive Dashboard (Streamlit)

------------------------------------------------------------------------

# Repository Structure

    Models-Benchmark-for-Audio-Storage-in-Vectorial-Databases-Milvus/
    │
    ├── src/
    │   ├── dashboard.py          # Interactive dashboard (Streamlit)
    │   ├── test_connection.py    # Test connection to Milvus
    │
    ├── docker-compose.yml        # Milvus and dependencies
    │
    ├── notebooks/                # Experiments and analysis
    │
    └── README.md

------------------------------------------------------------------------

# Technologies Used

-   Python
-   Milvus (Vector Database)
-   Docker / Docker Compose
-   Streamlit
-   Machine Learning Embeddings
-   Audio Processing

------------------------------------------------------------------------

# Setup and Execution

## 1 Clone the repository

``` bash
git clone https://github.com/diogocsilva12/Models-Benchmark-for-Audio-Storage-in-Vectorial-Databases-Milvus.git
cd Models-Benchmark-for-Audio-Storage-in-Vectorial-Databases-Milvus
```

------------------------------------------------------------------------

# Running the System

## 1 Start Milvus and dependencies

``` bash
docker-compose up -d
```

You can access the **Milvus Attu interface** at:

http://localhost:8000

------------------------------------------------------------------------

## 2 Run the interactive dashboard

``` bash
streamlit run src/dashboard.py
```

------------------------------------------------------------------------

## 3 Test the Milvus connection

``` bash
python src/test_connection.py
```

------------------------------------------------------------------------

# Benchmark Goals

The benchmark evaluates:

-   vector insertion performance
-   search latency
-   similarity search accuracy
-   scalability for large embedding collections

------------------------------------------------------------------------

# Applications

This type of system is relevant for:

-   audio similarity search
-   music recommendation engines
-   voice recognition systems
-   multimedia search engines
-   AI retrieval systems

------------------------------------------------------------------------

# Learning Outcomes

This project demonstrates skills in:

-   vector databases
-   embedding-based retrieval systems
-   containerized infrastructure
-   AI system benchmarking
-   interactive ML dashboards

------------------------------------------------------------------------

# Author

Diogo Coelho da Silva
João Barbosa
Pedro Oliveira

MSc in Advanced Computing\
University of Minho

------------------------------------------------------------------------

# Academic Context

Developed as part of academic coursework at the **University of Minho**.
