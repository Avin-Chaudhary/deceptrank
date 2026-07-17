# DeceptRank

**A Veracity-Aware Node2Vec Framework for Identifying Misinformation Super-Spreaders in Social Interaction Networks**

DeceptRank analyzes how misinformation spreads through social networks and ranks users by their structural influence as misinformation super-spreaders — not by looking at what they said, but by _how_ they're connected and _how_ they behave in the network.

Instead of content-based fake news detection, this project focuses purely on **network influence analysis**: building a directed interaction graph from real Twitter data, assigning veracity-aware weights to edges, learning structural embeddings with Node2Vec, and scoring every user by their potential to spread misinformation.

---

## How It Works

1. **Data** — Real Twitter rumor interactions from the [PHEME dataset](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078), each interaction labeled `fake`, `real`, or `unverified`.
2. **Storage** — Data is stored and served through **Hadoop HDFS**.
3. **Preprocessing** — **PySpark** cleans the data and aggregates interactions into weighted edges — fake interactions get higher weight than real ones.
4. **Graph Construction** — Users become nodes, interactions become directed weighted edges, forming a directed graph with **NetworkX**.
5. **Node2Vec Embeddings** — Biased random walks (preferring high-weight, misinformation-heavy edges) generate a 128-dimensional structural fingerprint for every user, trained via **Gensim Word2Vec**.
6. **Influence Scoring** — Each user's embedding norm, PageRank, and weighted out-degree are combined into a single composite **influence score**.
7. **Output** — A ranked list of the most influential misinformation super-spreaders, saved as CSV, with a bar chart visualization.

---

## Tech Stack

| Component            | Tool                                             |
| -------------------- | ------------------------------------------------ |
| Distributed storage  | Hadoop HDFS (WSL Ubuntu)                         |
| Data processing      | PySpark (local mode)                             |
| Graph representation | NetworkX                                         |
| Embeddings           | Node2Vec (custom biased walks) + Gensim Word2Vec |
| Scoring              | scikit-learn (MinMaxScaler)                      |
| Visualization        | Matplotlib                                       |
| Language             | Python 3.12                                      |

---

## Project Structure

```
deceptrank/
├── data/
│   └── raw/
│       └── interactions.csv      # PHEME-derived interaction data
├── src/
│   ├── config.py                 # all tunable settings
│   ├── utils.py                  # logging, timers, helpers
│   ├── spark_session.py          # PySpark session management
│   ├── hdfs_upload.py            # HDFS upload/check utilities
│   ├── preprocess.py             # load, clean, weight, aggregate edges
│   ├── graph_builder.py          # build directed graph, PageRank, out-degree
│   ├── node2vec_runner.py        # biased random walks + Word2Vec training
│   ├── influence_scorer.py       # composite influence score computation
│   └── visualize.py              # top spreaders bar chart
├── output/
│   ├── spreaders_ranked.csv      # final ranked output
│   └── top_spreaders.png         # bar chart of top spreaders
├── main.py                       # pipeline entry point
└── requirements.txt
```

---

## Data Schema

`interactions.csv` — one row per tweet/reply/retweet:

| Column             | Description                                      |
| ------------------ | ------------------------------------------------ |
| `tweet_id`         | Unique ID for the interaction                    |
| `src_user_id`      | The user who performed the action (the spreader) |
| `dst_user_id`      | The user being replied to / retweeted            |
| `interaction_type` | `tweet`, `reply`, or `retweet`                   |
| `veracity`         | `fake`, `real`, or `unverified`                  |
| `timestamp`        | Unix timestamp                                   |

Edge direction always flows `src → dst`, and `veracity` determines the edge weight — `fake` interactions are weighted highest since they're what drive the random walk toward misinformation spreaders.

---

## Setup

### 1. Prerequisites

- Python 3.12
- Hadoop installed on WSL Ubuntu
- Java (JDK) installed

### 2. Clone and set up the environment

```bash
git clone https://github.com/yourname/deceptrank.git
cd deceptrank

python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. Start Hadoop (in WSL)

```bash
start-dfs.sh
start-yarn.sh
```

### 4. Add your data

Place your `interactions.csv` in `data/raw/interactions.csv` following the schema above.

### 5. Run the pipeline

```bash
python main.py
```

---

## Output

Running the pipeline produces:

- **`output/spreaders_ranked.csv`** — every user ranked by influence score, highest first
- **`output/top_spreaders.png`** — bar chart of the top misinformation super-spreaders

---

## Key Design Choices

- **Veracity-weighted edges** — fake-news interactions get significantly higher edge weight than real ones, so Node2Vec's random walks naturally gravitate toward misinformation-heavy paths in the network.
- **p = 1.0, q = 0.5** — a DFS-biased walk strategy that favors discovering users who play structurally similar "hub" roles across different parts of the network, which is what super-spreader detection needs.
- **Composite influence score** — combines embedding norm (structural centrality), PageRank (global importance), and weighted out-degree (misinformation volume actually pushed) into one interpretable score.

---

## Dataset

This project uses the [PHEME dataset](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078) — a collection of Twitter rumor threads labeled for veracity, developed for rumor detection and veracity classification research.
