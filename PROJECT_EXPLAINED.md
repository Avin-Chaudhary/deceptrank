# DeceptRank — Complete Project Explanation

## What Is This Project?

**DeceptRank** is a tool that finds **misinformation super-spreaders** on social media. Think of it like this: on platforms like Twitter, some users are really good at spreading fake news — they share it widely, they're connected to lots of people, and their posts reach many communities. DeceptRank finds those users and ranks them by how dangerous they are.

It does this by:
1. Reading a big CSV file of social media interactions (who replied to whom, who retweeted whom, etc.)
2. Building a network/graph of users (like a web of connections)
3. Using AI to understand each user's position and role in the network
4. Scoring each user on how much misinformation they spread
5. Grouping users into communities and finding "bridge" users who connect multiple groups
6. Creating a bar chart of the top offenders

---

## The Input Data

The whole pipeline starts with one file: **`data/interactions.csv`**

Each row represents one social media interaction with these columns:

| Column | What it means |
|---|---|
| `tweet_id` | Unique ID of the tweet |
| `src_user_id` | The user who did the action (e.g., retweeted, replied) |
| `dst_user_id` | The user whose tweet was interacted with |
| `interaction_type` | What kind of interaction — reply, retweet, etc. |
| `veracity` | Is the content **fake**, **real**, or **unverified**? |
| `timestamp` | When it happened (Unix timestamp) |

---

## The Output

After the pipeline runs, you get these files in the `output/` folder:

| File | What it is |
|---|---|
| `spreaders_ranked.csv` | Every user ranked by their influence score (most dangerous first) |
| `bridge_nodes.csv` | Users who connect multiple communities (most dangerous bridge users first) |
| `top_spreaders.png` | A bar chart showing the top 10 super-spreaders visually |

---

## How To Run It

```bash
python main.py
```

That's it. The `main.py` file runs the entire 7-step pipeline from start to finish.

---

## Project Structure — File by File

```
deceptrank/
├── main.py                  ← Entry point, runs the full 7-step pipeline
├── requirements.txt         ← Python packages needed
├── data/
│   └── interactions.csv     ← Input data (social media interactions)
├── output/                  ← Where results go
└── src/
    ├── config.py            ← All settings and constants in one place
    ├── utils.py             ← Helper functions (logging, timing, saving files)
    ├── hdfs_upload.py       ← Step 1: Try to upload data to Hadoop (optional)
    ├── spark_session.py     ← Creates the PySpark engine
    ├── preprocess.py        ← Step 2: Clean and prepare the raw data
    ├── graph_builder.py     ← Step 3: Build the user network graph
    ├── node2vec_runner.py   ← Step 4: AI walks through the graph to learn user roles
    ├── influence_scorer.py  ← Step 5: Calculate influence scores
    ├── clustering.py        ← Step 6: Group users into communities
    └── visualize.py         ← Step 7: Create the bar chart
```

---

## The 7-Step Pipeline — In Detail

### Step 1: HDFS Upload (`hdfs_upload.py`)

**What it does:** Checks if Hadoop HDFS (a distributed file system) is running on your machine. If it is, it uploads the CSV file there so Spark can read it faster. If HDFS is not available (which is the common case), it just skips this step and uses the local CSV file directly.

**How it works:**
- `check_hdfs_available()` tries to connect to `localhost:9870` (Hadoop's web port). If it can't connect within 3 seconds, HDFS is considered unavailable.
- If available, `create_hdfs_dirs()` creates the folders on HDFS, and `upload_to_hdfs()` copies the CSV file there.
- All HDFS commands run through **WSL** (Windows Subsystem for Linux) because Hadoop runs on Linux.

**You can think of it as:** "Let me check if the fast highway is open. If yes, I'll use it. If not, I'll take the normal road."

---

### Step 2: Preprocessing (`preprocess.py`)

**What it does:** Reads the raw CSV and cleans it up using **PySpark** (a big data processing engine), then converts the raw interactions into weighted edges for the graph.

**How it works, function by function:**

1. **`load_interactions(spark, path)`** — Reads the CSV file into a Spark DataFrame (like a super-powered spreadsheet). It uses a predefined schema so Spark knows which columns to expect and what type each column is (string, number, etc.).

2. **`clean_interactions(df)`** — Removes garbage data:
   - Drops rows where `src_user_id`, `dst_user_id`, or `veracity` is missing
   - Removes **self-loops** (a user interacting with themselves makes no sense)
   - Removes duplicate tweets (same `tweet_id` appearing twice)
   - Keeps only interactions labeled as `fake`, `real`, or `unverified`

3. **`assign_veracity_weights(df)`** — Adds a numeric "weight" column based on the veracity label:
   - `fake` → **1.0** (highest weight, because we care most about fake content spreading)
   - `unverified` → **0.5** (might be fake, so it gets some weight)
   - `real` → **0.2** (even real content interactions show user activity, so gets a small weight)

4. **`aggregate_edges(df)`** — Combines multiple interactions between the same two users into a single edge:
   - If UserA retweeted UserB 5 times (3 fake, 2 real), instead of 5 separate records, this creates one edge from A→B.
   - The edge weight = sum of veracity weights + log(interaction count). The log part means: "if you interacted 100 times instead of 10, you don't get 10x the weight — you get a little bit more." This prevents one hyper-active user from dominating.
   - Edges with weight below `MIN_EDGE_WEIGHT` (0.1) are thrown out to remove noise.

**The Spark session** is created by `spark_session.py`:
- `get_spark()` creates a PySpark session running locally on all CPU cores (`local[*]`)
- Allocates 2GB of memory for processing
- Suppresses noisy Spark logs so you only see important messages

**You can think of it as:** "Take the raw messy data, clean it up, and figure out how strongly each user is connected to each other user."

---

### Step 3: Graph Construction (`graph_builder.py`)

**What it does:** Takes the cleaned edges from Step 2 and builds a **directed weighted graph** — a network where users are dots (nodes) and their interactions are arrows (edges) with a weight showing how "suspicious" the connection is.

**How it works, function by function:**

1. **`build_networkx_graph(edge_df)`** — Converts the Spark DataFrame to a Pandas table, then loops through every row and adds an edge to a NetworkX `DiGraph` (directed graph) object. Each edge has:
   - `src_user_id` → `dst_user_id` (direction: who interacted with whom)
   - `weight` = the edge weight calculated in preprocessing

2. **`compute_pagerank(G)`** — Runs Google's **PageRank** algorithm on the graph. PageRank was originally designed to rank web pages — "if many important pages link to you, you must be important." Here it means: **if many active users interact with you, you are an important node in the misinformation network.** A user with high PageRank receives a lot of attention from other high-PageRank users.

3. **`get_weighted_outdegree(G)`** — For each user, adds up the weights of all their outgoing edges. This directly measures: **how much fake content did this user push out?** A user who retweeted 50 fake tweets has a much higher weighted out-degree than one who retweeted 2.

4. **`print_graph_stats(G)`** — Prints a summary: total nodes, total edges, whether it's directed, and the top 5 users by weighted out-degree.

**You can think of it as:** "Draw the web of connections and figure out who's the most connected and who's pushing out the most fake content."

---

### Step 4: Node2Vec Embeddings (`node2vec_runner.py`)

**What it does:** This is the **AI/machine learning** step. It converts each user's position in the network into a vector of 64 numbers (called an "embedding"). Users who are structurally similar in the network (similar neighbors, similar roles) get similar vectors.

**Why?** Because a user's position in a network tells you a lot. Are they in the center connected to everyone? Are they a bridge between two groups? Are they on the edges? The embedding captures all of this as numbers that a computer can work with.

**How it works, function by function:**

1. **`biased_walk(G, start_node, walk_length, p, q)`** — Performs a single **random walk** starting from a user. Imagine you're standing on a user in the network and you take 30 steps, randomly choosing which neighbor to visit next. But it's not completely random — it's **biased**:
   - **Edge weight matters:** You prefer to walk along heavier edges (more suspicious connections)
   - **Parameter `p` (return, = 1.0):** Controls how likely you are to go back to where you just came from. Low p = stay local, high p = don't care about going back.
   - **Parameter `q` (in-out, = 0.5):** Controls whether you explore far away or stay close. Low q (like 0.5) = explore far. High q = stay close.
   
   So `q=0.5` means the walker tends to **explore outward** — it's biased towards discovering far-away parts of the network, which is good for finding super-spreaders who have wide reach.

2. **`generate_walks(G)`** — Runs `biased_walk` for **every** user in the graph, **10 times each** (`NUM_WALKS=10`), with each walk being 30 steps long (`WALK_LENGTH=30`). So if there are 1000 users, this generates 10,000 walks. The nodes are shuffled each round for better coverage.

3. **`train_embeddings(walks)`** — Treats each walk as a "sentence" and each user as a "word", then trains a **Word2Vec skip-gram model** on them. Word2Vec is normally used in language — "words that appear in similar sentences have similar meanings." Here: **users that appear in similar random walks have similar network positions.** The output is a 64-dimensional vector for each user.

**You can think of it as:** "Send AI scouts on random walks through the network so they can learn what role each user plays — who's central, who's peripheral, who bridges multiple groups."

---

### Step 5: Influence Scoring (`influence_scorer.py`)

**What it does:** Combines three different signals into one final **influence score** for each user: how structurally important they are (embedding), how much attention they receive (PageRank), and how much content they push out (out-degree).

**How it works, function by function:**

1. **`compute_embedding_norms(model)`** — For each user, calculates the **L2 norm** (length) of their 64-d embedding vector. A larger norm generally means the user is more structurally active — they appeared in many walks in distinctive positions. Think of it as: "how much structural significance does this user have?"

2. **`build_score_dataframe(model, pagerank, outdegree)`** — Puts three metrics into one table per user:
   - `emb_norm` — structural significance from embeddings
   - `pagerank` — how much important attention they receive
   - `weighted_outdegree` — raw volume of suspicious content they push

3. **`normalize_and_score(df)`** — 
   - First, normalizes all three columns to **0–1 range** using MinMaxScaler (so they're comparable)
   - Then calculates the final score as a weighted sum:
     ```
     influence_score = 0.3 × emb_norm + 0.3 × pagerank + 0.4 × outdegree
     ```
   - **Why 0.4 for outdegree?** Because directly pushing out fake content is the strongest signal of being a super-spreader. PageRank and embedding norm are supporting signals.
   - Users are then sorted by this score, highest first. Rank 1 = most dangerous user.

4. **`get_top_spreaders(df, top_n)`** — Simply returns the top N rows (default 10).

After scoring, `main.py` prints the top 10 to the terminal and saves the full ranked list to `spreaders_ranked.csv`.

**You can think of it as:** "We measured three things about each user — their network position, how much attention they get, and how much fake stuff they share. Now we mix these together into one final danger score."

---

### Step 6: Clustering (`clustering.py`)

**What it does:** Groups users into **communities** (clusters) based on how similar their network roles are, then identifies the most dangerous "bridge nodes" — users who connect multiple communities.

**How it works, function by function:**

1. **`get_embeddings_matrix(model)`** — Extracts all the 64-d embedding vectors from the Node2Vec model into a NumPy matrix (rows = users, columns = embedding dimensions). This is just a format conversion so KMeans can work with it.

2. **`cluster_nodes(model)`** — Runs **KMeans clustering** with `N_CLUSTERS=3` on the embedding vectors. KMeans groups similar vectors together — so users with similar network roles end up in the same community. The output is a dictionary mapping each user to a cluster number (0, 1, or 2).

3. **`find_bridge_nodes(G, cluster_map, influence_df)`** — This is the key analysis:
   - For each user, it looks at all their neighbors in the graph
   - It counts how many **different communities** those neighbors belong to
   - A user connected to 3 different communities is more dangerous than one connected to just 1, because they can spread misinformation across group boundaries
   - The **bridge score** = `influence_score × number_of_communities_connected`
   - Users are ranked by bridge score (highest = most dangerous bridge)

4. **`run_clustering(model, G, influence_df)`** — Runs the full pipeline: cluster → find bridges. Returns the cluster map and the bridge dataframe.

After clustering, `main.py` saves the bridge analysis to `bridge_nodes.csv`.

**You can think of it as:** "First, group users into communities. Then find the users who have tentacles reaching into multiple communities — those are the super-spreaders who can infect the widest audience."

---

### Step 7: Visualizations (`visualize.py`)

**What it does:** Creates a horizontal bar chart of the top 10 super-spreaders.

**How it works:**

**`plot_top_spreaders_bar(influence_df, top_n=10)`** — Takes the influence dataframe, gets the top 10 rows, and creates a horizontal bar chart:
- Each bar represents one user
- Bar length = influence score
- Bars are colored with a red gradient (lighter to darker)
- The exact score value is printed on each bar
- Saved as `top_spreaders.png` in the output folder

---

## Supporting Files

### `config.py` — All Settings

Every tunable number in the project lives here so you don't have to hunt through code to change something:

| Setting | Value | What it controls |
|---|---|---|
| `VERACITY_WEIGHTS` | fake=1.0, unverified=0.5, real=0.2 | How much weight each content type gets |
| `MIN_EDGE_WEIGHT` | 0.1 | Edges below this are thrown out |
| `WALK_LENGTH` | 30 | How many steps each random walk takes |
| `NUM_WALKS` | 10 | How many walks per user |
| `P` | 1.0 | Random walk return parameter |
| `Q` | 0.5 | Random walk explore parameter |
| `EMBEDDING_DIM` | 64 | Size of each user's AI-learned vector |
| `WINDOW_SIZE` | 5 | Word2Vec context window |
| `EPOCHS` | 3 | How many times Word2Vec trains over the data |
| `ALPHA, BETA, GAMMA` | 0.3, 0.3, 0.4 | Weights for the 3 components of influence score |
| `N_CLUSTERS` | 3 | Number of communities to detect |
| `TOP_N` | 10 | How many top spreaders to show |

### `utils.py` — Helpers

- **`logger`** — Logs messages with timestamps (e.g., `12:30:45  INFO  Loading data...`)
- **`Timer`** — A context manager that measures how long each step takes. Used as `with Timer("step name"):` throughout the code.
- **`ensure_dir(path)`** — Creates a folder if it doesn't exist yet
- **`print_top_spreaders(df)`** — Prints a nice formatted table of the top 10 users to the terminal
- **`save_csv(df, filename, output_dir)`** — Saves a pandas DataFrame to a CSV file

---

## Technologies Used

| Technology | What it is | Why it's used here |
|---|---|---|
| **Python** | Programming language | Everything is written in Python |
| **PySpark** | Big data processing engine | To efficiently clean and transform millions of interaction rows |
| **NetworkX** | Graph/network library | To build and analyze the user interaction graph |
| **Gensim (Word2Vec)** | NLP/embedding library | To train Node2Vec embeddings (treating graph walks as "sentences") |
| **scikit-learn** | Machine learning library | For KMeans clustering and MinMaxScaler normalization |
| **Matplotlib** | Plotting library | To create the bar chart visualization |
| **Pandas / NumPy** | Data manipulation | Dataframes, arrays, and math operations throughout |
| **HDFS (optional)** | Hadoop file system | Distributed storage for large datasets (falls back to local) |

---

## The Big Picture Flow

```
interactions.csv
      │
      ▼
┌─────────────────┐
│  1. HDFS Upload  │  Try to use Hadoop, fall back to local
└────────┬────────┘
         ▼
┌─────────────────┐
│  2. Preprocess   │  Clean data → assign weights → aggregate edges
└────────┬────────┘
         ▼
┌─────────────────┐
│  3. Build Graph  │  Create user network + compute PageRank + out-degree
└────────┬────────┘
         ▼
┌─────────────────┐
│  4. Node2Vec     │  Random walks → Word2Vec → 64-d embedding per user
└────────┬────────┘
         ▼
┌─────────────────┐
│  5. Score Users  │  Mix embedding norm + PageRank + out-degree → influence score
└────────┬────────┘
         ▼
┌─────────────────┐
│  6. Clustering   │  KMeans communities + find bridge nodes
└────────┬────────┘
         ▼
┌─────────────────┐
│  7. Visualize    │  Bar chart of top 10 super-spreaders
└────────┬────────┘
         ▼
   output/ folder
   ├── spreaders_ranked.csv
   ├── bridge_nodes.csv
   └── top_spreaders.png
```

---

## Key Concepts Explained Simply

**Graph / Network:** A collection of dots (users) connected by lines (interactions). In this project, the lines have arrows (who interacted with whom) and weights (how suspicious the interaction is).

**PageRank:** Google's algorithm for ranking importance. "If important people point to you, you must be important too." Applied to users instead of web pages.

**Node2Vec / Embeddings:** A way to convert a user's position in the network into a list of numbers. Users in similar positions get similar numbers. This lets us use math to compare users.

**KMeans Clustering:** An algorithm that groups similar things together. Here it groups users with similar network roles into communities.

**Bridge Nodes:** Users who are connected to multiple different communities. They're dangerous because they can spread misinformation from one group to another, like a virus jumping between populations.

**Influence Score:** The final "danger rating" — a number between 0 and 1 that combines three signals: network significance (embedding), attention received (PageRank), and content volume pushed (out-degree).
