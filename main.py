import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import os
import hashlib
import secrets
import io
import csv
import random
from node2vec import Node2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np
from matplotlib.patches import Patch
import matplotlib.cm as cm
from datetime import date


import tensorflow as tf
from spektral.layers import GCNConv
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from spektral.data import Graph
from spektral.data import Dataset


# ------------------ CONFIG ------------------
BASE_DIR = Path(r"C:\RumourApp")
AUTH_DIR = BASE_DIR / "auth"
DATA_DIR = BASE_DIR / "data"
USERS_FILE = AUTH_DIR / "users.json"

# Default simulation params
DEFAULT_INF_PROB = 0.1


# ------------------ STORAGE INIT ------------------
def init_storage():
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not USERS_FILE.exists():
        USERS_FILE.write_text(json.dumps({}))


init_storage()

# ------------------ AUTH HELPERS ------------------


def hash_password(password: str, salt: str) -> str:
    """Derive a password hash using pbkdf2_hmac."""
    pwd = password.encode("utf-8")
    saltb = salt.encode("utf-8")
    dk = hashlib.pbkdf2_hmac("sha256", pwd, saltb, 100_000)
    return dk.hex()


def load_users() -> dict:
    try:
        return json.loads(USERS_FILE.read_text())
    except Exception:
        return {}


def save_users(users: dict):
    USERS_FILE.write_text(json.dumps(users, indent=2))


def register_user(username: str, password: str) -> tuple[bool, str]:
    users = load_users()
    if username in users:
        return False, "User already exists"
    salt = secrets.token_hex(16)
    pwd_hash = hash_password(password, salt)
    users[username] = {
        "salt": salt,
        "pwd_hash": pwd_hash,
        "created_at": datetime.utcnow().isoformat(),
    }
    save_users(users)
    return True, "Registered"


def authenticate_user(username: str, password: str) -> bool:
    users = load_users()
    if username not in users:
        return False
    rec = users[username]
    return hash_password(password, rec["salt"]) == rec["pwd_hash"]


# ------------------ DATASET STORAGE (filesystem-first) ------------------


def list_datasets(owner: str | None = None) -> pd.DataFrame:
    rows = []
    for p in DATA_DIR.iterdir():
        if p.is_dir():
            meta = p / "metadata.json"
            if meta.exists():
                try:
                    m = json.loads(meta.read_text())
                    if owner and m.get("owner") != owner:
                        continue
                    rows.append(m)
                except Exception:
                    continue
    if rows:
        return pd.DataFrame(sorted(rows, key=lambda r: r["created_at"], reverse=True))
    return pd.DataFrame(columns=["id", "name", "owner", "created_at"])


def create_dataset(name: str, owner: str) -> dict:
    # create unique id
    ds_id = secrets.token_hex(8)
    ds_dir = DATA_DIR / ds_id
    ds_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": ds_id,
        "name": name,
        "owner": owner,
        "created_at": datetime.utcnow().isoformat(),
    }
    (ds_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def save_edges_csv(ds_id: str, file) -> int:
    ds_dir = DATA_DIR / ds_id
    dest = ds_dir / "edges.csv"
    # read uploaded file and normalize to two-column csv
    content = file.read().decode("utf-8")
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    rows = []
    for ln in lines:
        parts = [p for p in re_split(ln) if p != ""]
        if len(parts) >= 2:
            rows.append((parts[0], parts[1]))
    # write
    with open(dest, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return len(rows)


def save_seeds_csv(ds_id: str, file) -> int:
    ds_dir = DATA_DIR / ds_id
    dest = ds_dir / "seeds.csv"
    content = file.read().decode("utf-8")
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    with open(dest, "w", newline="") as f:
        writer = csv.writer(f)
        for ln in lines:
            # if space separated, take first token
            parts = [p for p in re_split(ln) if p != ""]
            if parts:
                writer.writerow([parts[0]])
    return len(lines)


def read_edges(ds_id: str) -> list[tuple[str, str]]:
    dest = DATA_DIR / ds_id / "edges.csv"
    if not dest.exists():
        return []
    rows = []
    with open(dest, newline="") as f:
        reader = csv.reader(f)
        for r in reader:
            if len(r) >= 2:
                rows.append((r[0], r[1]))
    return rows


def read_seeds(ds_id: str) -> list[str]:
    dest = DATA_DIR / ds_id / "seeds.csv"
    if not dest.exists():
        return []
    vals = []
    with open(dest, newline="") as f:
        reader = csv.reader(f)
        for r in reader:
            if r:
                vals.append(r[0])
    return vals


# small helper
import re


def re_split(s: str) -> list[str]:
    return re.split(r"[\s,]+", s.strip())


# ------------------ SIMULATION MODELS ------------------


def independent_cascade(
    G: nx.Graph, seeds: list[str], p: float = DEFAULT_INF_PROB, max_steps: int = 100
) -> list[set]:
    """Return list of sets: infected nodes at each timestep (including time 0 seeds)."""
    infected = set(seeds)
    layers = [set(seeds)]
    active = set(seeds)
    steps = 0
    while active and steps < max_steps:
        new_active = set()
        for u in active:
            for v in G.neighbors(u):
                if v in infected:
                    continue
                if random.random() <= p:
                    new_active.add(v)
                    infected.add(v)
        if not new_active:
            break
        layers.append(new_active)
        active = new_active
        steps += 1
    return layers


def si_model(
    G: nx.Graph, seeds: list[str], beta: float = DEFAULT_INF_PROB, max_steps: int = 100
) -> list[set]:
    """Simple SI model: once infected, stays infected; each infected tries to infect neighbors each step with prob beta."""
    infected = set(seeds)
    layers = [set(seeds)]
    for _ in range(max_steps):
        newly = set()
        for u in list(infected):
            for v in G.neighbors(u):
                if v not in infected and random.random() <= beta:
                    newly.add(v)
        if not newly:
            break
        infected |= newly
        layers.append(newly)
    return layers


def train_gnn(G, y_binary, epochs=50, hidden_units=32, learning_rate=0.01):
    """
    Train a simple Graph Convolutional Network (GCN) to predict rumor spread.
    """
    nodes = list(G.nodes())
    N = len(nodes)

    # Build adjacency matrix
    A = nx.to_numpy_array(G, nodelist=nodes)
    A_norm = normalize_adjacency(A)  # symmetric normalization

    # Node features: simple one-hot if no features
    X = np.eye(N)

    # Labels
    y = np.array([y_binary[nodes.index(n)] for n in nodes])

    # TensorFlow inputs
    X_input = tf.convert_to_tensor(X, dtype=tf.float32)
    A_input = tf.convert_to_tensor(A_norm, dtype=tf.float32)
    y_input = tf.convert_to_tensor(y.reshape(-1, 1), dtype=tf.float32)

    # GCN Model
    class GCN(Model):
        def __init__(self):
            super().__init__()
            self.gcn1 = GCNConv(hidden_units, activation="relu")
            self.gcn2 = GCNConv(1, activation="sigmoid")

        def call(self, inputs):
            X, A = inputs
            h = self.gcn1([X, A])
            return self.gcn2([h, A])

    model = GCN()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    # Training loop
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            y_pred = model([X_input, A_input])
            loss = loss_fn(y_input, y_pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, loss: {loss.numpy():.4f}")

    # Return predicted probabilities
    y_prob = model([X_input, A_input]).numpy().flatten()
    prob_series = dict(zip(nodes, y_prob))
    return prob_series


def normalize_adjacency(A):
    """
    Symmetric normalization: D^-1/2 * A * D^-1/2
    """
    I = np.eye(A.shape[0])
    A_hat = A + I  # add self-loops
    D = np.diag(np.sum(A_hat, axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


# ------------------ BLOCKING STRATEGIES ------------------


def block_high_degree(G: nx.Graph, k: int) -> list[str]:
    return [n for n, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:k]]


def block_betweenness(G: nx.Graph, k: int) -> list[str]:
    bc = nx.betweenness_centrality(G)
    return sorted(bc, key=bc.get, reverse=True)[:k]


def block_random(G: nx.Graph, k: int) -> list[str]:
    nodes = list(G.nodes())
    random.shuffle(nodes)
    return nodes[:k]


# ------------------ UTIL / VISUALIZATION ------------------


def build_graph(edges: list[tuple[str, str]]) -> nx.Graph:
    G = nx.Graph()
    G.add_edges_from(edges)
    return G


def plot_graph(G, seeds=None, blocked=None):
    """
    Plot the graph with NetworkX inside Streamlit.
    - Seeds are red
    - Blocked nodes are gray
    - Others are lightblue
    """
    if G is None or len(G) == 0:
        st.warning("Empty graph — nothing to visualize.")
        return None

    # Compute layout
    pos = nx.spring_layout(G, seed=42)

    # Color scheme
    node_colors = []
    for n in G.nodes():
        if seeds and n in seeds:
            node_colors.append("red")
        elif blocked and n in blocked:
            node_colors.append("gray")
        else:
            node_colors.append("skyblue")

    # Create a fresh figure (avoid stale pyplot state)
    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw(G, pos, node_color=node_colors, with_labels=False, node_size=80, ax=ax)
    ax.set_title("Network Visualization", fontsize=12)
    ax.axis("off")

    # Return the figure for Streamlit display
    st.pyplot(fig)
    return fig


def plot_graph_new(
    G, affected=None, title="Rumor Spread Visualization", figsize=(4, 4)
):
    """
    Visualize affected (red) and unaffected (blue) nodes with black circular borders,
    a left-aligned legend, and a bordered Streamlit container.
    """
    if G is None or G.number_of_nodes() == 0:
        st.warning("⚠️ Empty graph — nothing to visualize.")
        return None

    affected = set(affected or [])
    # ✅ Increased spacing using k and iterations
    pos = nx.spring_layout(G, seed=42, k=0.75, iterations=120)

    # Node color scheme
    node_colors = ["#e74c3c" if n in affected else "#3498db" for n in G.nodes()]

    # ---- Create Matplotlib figure ----
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_edges(G, pos, edge_color="lightgray", ax=ax, alpha=0.6)

    # ✅ Draw edges in DEEP BLACK (strong visibility)
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="black",  # strong black
        width=1.2,  # slightly thicker for clarity
        ax=ax,
        alpha=0.9,  # near solid opacity
    )

    # ✅ Draw nodes with BLACK borders
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=100,
        edgecolors="black",  # black outline
        linewidths=0.8,  # outline thickness
        ax=ax,
    )

    ax.set_title(title, fontsize=12, pad=8)
    ax.axis("off")

    # Legend (top-left)
    legend_elements = [
        Patch(facecolor="#e74c3c", edgecolor="black", label="Affected"),
        Patch(facecolor="#3498db", edgecolor="black", label="Unaffected"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0, 1.05),
        fontsize=8,
        frameon=False,
    )

    # ---- HTML bordered container ----
    st.markdown(
        """
        <div style="
            border: 2px solid #000000;   /* black border */
            border-radius: 10px;
            padding: 12px;
            margin: 10px 0px 20px 0px;
            background-color: #ffffff;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.15);
            text-align: center;
        ">
        """,
        unsafe_allow_html=True,
    )

    # Display inside bordered box
    st.pyplot(fig, use_container_width=False)
    st.markdown("</div>", unsafe_allow_html=True)

    return fig


def plot_graph_with_probs(G, nodes, probs_all, title="Rumor Spread Prediction"):
    """
    Visualize the rumor spread probability over the network
    using a continuous colormap (Reds) inside a bordered container.
    """
    if G is None or len(G) == 0:
        st.warning("⚠️ Empty graph — nothing to visualize.")
        return

    # Normalize probabilities to [0, 1]
    probs = np.array(
        [probs_all[nodes.index(n)] if n in nodes else 0 for n in G.nodes()]
    )
    probs = np.clip(probs, 0, 1)

    cmap = cm.get_cmap("Reds")
    node_colors = cmap(probs)

    # Compute layout
    pos = nx.spring_layout(G, seed=42)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw(
        G,
        pos,
        node_color=node_colors,
        edge_color="lightgray",
        with_labels=False,
        node_size=100,
        ax=ax,
    )
    ax.set_title(title, fontsize=13)
    ax.axis("off")

    # Add colorbar to indicate rumor probability intensity
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Rumor Probability", fontsize=8)

    # Enclosed visualization box (HTML border)
    box_style = """
    <div style="
        border: 1px solid #e6e6e6;
        padding: 10px;
        border-radius: 8px;
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    ">
    """
    st.markdown(box_style, unsafe_allow_html=True)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------- helper utilities ----------
def generate_ic_labels(G, seeds, p=0.1, runs=200, max_steps=100):
    """
    Run IC model 'runs' times and return label frequencies:
    freq[node] = fraction of runs where node became infected.
    Also returns list of sets with infected nodes per run.
    """
    nodes = list(G.nodes())
    freq = {n: 0 for n in nodes}
    runs_infected = []
    for _ in range(runs):
        layers = independent_cascade(G, seeds, p=p, max_steps=max_steps)
        infected = set().union(*layers)
        runs_infected.append(infected)
        for n in infected:
            freq[n] += 1
    for n in freq:
        freq[n] /= runs
    return freq, runs_infected


def simulate_ic_spread(G, seeds, p, blocked=None):
    blocked = set(blocked or [])
    infected = set(seeds) - blocked
    active = infected.copy()

    while active:
        new_active = set()
        for u in list(active):
            for v in G.neighbors(u):
                if v in infected or v in blocked:
                    continue
                if np.random.rand() <= p:
                    new_active.add(v)
                    infected.add(v)
        active = new_active
    return infected


def simulate_ic_predicted(G, seeds, prob_dict, base_p=0.1):
    infected = set(seeds)
    active = set(seeds)

    while active:
        new_active = set()
        for u in list(active):
            for v in G.neighbors(u):
                if v in infected:
                    continue
                # Infection prob = base * predicted susceptibility
                p_v = base_p * prob_dict[v]
                if np.random.rand() <= p_v:
                    infected.add(v)
                    new_active.add(v)
        active = new_active
    return infected


def compute_node_features(G, node2vec_dim=64, walks=10, walk_length=80):
    """
    Return DataFrame indexed by node with features:
    - degree, clustering, betweenness, embedding dims...
    """
    nodes = list(G.nodes())
    # structural features
    degree = dict(G.degree())
    clustering = nx.clustering(G)
    # betweenness can be expensive for very large graphs; consider approximate if needed
    betweenness = (
        nx.betweenness_centrality(G)
        if G.number_of_nodes() < 2000
        else nx.betweenness_centrality_subset(
            G, sources=nodes[:100], targets=nodes[:100]
        )
    )

    # node2vec embeddings
    node2vec = Node2Vec(
        G,
        dimensions=node2vec_dim,
        walk_length=walk_length,
        num_walks=walks,
        workers=1,
        quiet=True,
    )
    model = node2vec.fit(window=10, min_count=1)  # gensim Word2Vec model
    embeddings = {
        n: model.wv[str(n)] if str(n) in model.wv else np.zeros(node2vec_dim)
        for n in nodes
    }

    # assemble DataFrame
    rows = []
    for n in nodes:
        row = {
            "node": n,
            "degree": degree.get(n, 0),
            "clustering": clustering.get(n, 0.0),
            "betweenness": betweenness.get(n, 0.0),
        }
        emb = embeddings[n]
        for i, val in enumerate(emb):
            row[f"emb_{i}"] = float(val)
        rows.append(row)
    feats = pd.DataFrame(rows).set_index("node")
    return feats


def train_predict_model(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=(y > 0)
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    # compute metrics
    auc_score = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else None
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)
    return clf, (X_test, y_test, probs), {"auc": auc_score, "pr_auc": pr_auc}


def precision_at_k(y_true, y_score, k):
    """
    y_true, y_score arrays for all nodes: compute Precision@k (top-k predicted nodes).
    """
    idx = np.argsort(y_score)[::-1][:k]
    return np.mean(y_true[idx] == 1)


# ------------------ STREAMLIT UI ------------------

st.set_page_config(page_title="Rumour Blocking Simulator", layout="wide")

if "user" not in st.session_state:
    st.session_state["user"] = None

menu = st.sidebar.selectbox(
    "Menu",
    [
        "Home",
        "Register",
        "Login",
        "Create Dataset",
        "Datasets",
        "Visualization",
        "Simulation",
        "Containment",
        "Reports",
    ],
)

# --- Home ---
if menu == "Home":
    st.title("🕸️ Rumour Blocking Simulator")
    st.markdown(
        """
    This app demonstrates rumour spread simulations on user-uploaded networks and allows testing blocking strategies.

    Flow:
    1. Register / Login
    2. Create a dataset (name) and upload two files: edges and initial seeds
    3. Visualize the network
    4. Simulate spread and test containment strategies
    5. Download reports
    """
    )
    if st.session_state["user"]:
        st.success(f"Logged in as: {st.session_state['user']}")
    else:
        st.info("Please register or login to create datasets.")

# --- Register ---

elif menu == "Register":
    st.title("Create an Account")

    full_name = st.text_input("Full Name")
    username = st.text_input("Username")
    email = st.text_input("Email")
    dob = st.date_input(
        "Date of Birth",
        min_value=date(1900, 1, 1),  # Allow dates as far back as 1900
        max_value=date.today(),  # Prevent selecting future dates
        key="dob_input",
    )
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Register", key="register_btn"):
        users = load_users()

        # -------- VALIDATIONS -------- #

        # Full name
        if len(full_name.strip()) < 3:
            st.error("Full Name must be at least 3 characters.")
            st.stop()

        # Username
        if len(username.strip()) < 3:
            st.error("Username must be at least 3 characters.")
            st.stop()

        if username in users:
            st.error("Username already exists. Choose a different one.")
            st.stop()

        # Email
        import re

        email_pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
        if not re.match(email_pattern, email):
            st.error("Invalid email format.")
            st.stop()

        # DOB validation (age ≥ 13)
        from datetime import date

        today = date.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

        if age < 13:
            st.error("You must be at least 13 years old to register.")
            st.stop()

        # Password
        if len(password) < 6:
            st.error("Password must be at least 6 characters.")
            st.stop()

        if password != confirm_password:
            st.error("Passwords do not match.")
            st.stop()

        # -------- SAVE USER -------- #
        salt = secrets.token_hex(16)
        pwd_hash = hash_password(password, salt)

        users[username] = {
            "full_name": full_name,
            "email": email,
            "dob": str(dob),
            "salt": salt,
            "pwd_hash": pwd_hash,
            "created_at": datetime.utcnow().isoformat(),
        }

        save_users(users)

        st.success("✅ Registration successful! You can now log in.")
        st.info(f"Welcome, {full_name}!")

# --- Login ---
elif menu == "Login":
    st.title("Login")
    u = st.text_input("Username", key="login_user")
    p = st.text_input("Password", type="password", key="login_pass")
    if st.button("Login"):
        if authenticate_user(u, p):
            st.session_state["user"] = u
            st.success("Logged in")
        else:
            st.error("Invalid credentials")

# --- Create Dataset ---
elif menu == "Create Dataset":
    st.title("Create a new dataset")
    if not st.session_state["user"]:
        st.warning("You must be logged in to create datasets.")
    else:
        name = st.text_input("Dataset name")
        if st.button("Create"):
            if not name.strip():
                st.error("Name required")
            else:
                meta = create_dataset(name.strip(), st.session_state["user"])
                st.success(f"Dataset created: {meta['id']}")

# --- Datasets (upload files) ---
elif menu == "Datasets":
    st.title("Datasets")
    if not st.session_state["user"]:
        st.warning("Login first")
    else:
        df = list_datasets(owner=st.session_state["user"])
        st.dataframe(df)
        if df.empty:
            st.info("No datasets yet. Create one first.")
        else:
            selected = st.selectbox("Select dataset", df["name"] + " — " + df["id"])
            ds_id = selected.split(" — ")[-1]
            st.write("Dataset ID:", ds_id)
            st.subheader("Upload edge list (file with two columns, no header)")
            edges_file = st.file_uploader(
                "Edge file", type=["csv", "txt"], key=f"edges_{ds_id}"
            )
            st.subheader("Upload seed list (one node per line)")
            seeds_file = st.file_uploader(
                "Seed file", type=["csv", "txt"], key=f"seeds_{ds_id}"
            )
            if edges_file and st.button("Save edges", key=f"save_edges_{ds_id}"):
                cnt = save_edges_csv(ds_id, edges_file)
                st.success(f"Saved {cnt} edges")
            if seeds_file and st.button("Save seeds", key=f"save_seeds_{ds_id}"):
                cnt = save_seeds_csv(ds_id, seeds_file)
                st.success(f"Saved {cnt} seed rows")

# --- Visualization ---
elif menu == "Visualization":
    st.title("Network Visualization")
    if not st.session_state["user"]:
        st.warning("Login first")
    else:
        df = list_datasets(owner=st.session_state["user"])
        if df.empty:
            st.info("No datasets yet")
        else:
            selected = st.selectbox("Select dataset", df["name"] + " — " + df["id"])
            ds_id = selected.split(" — ")[-1]
            edges = read_edges(ds_id)
            seeds = read_seeds(ds_id)
            if not edges:
                st.warning("No edges uploaded for this dataset")
            else:
                st.info("Building Graph1")
                G = build_graph(edges)
                blocked = []
                plt_obj = plot_graph_new(
                    G, affected=seeds, title="Visualization of Network"
                )
                st.info("Building Graph Successful")
                st.write(
                    f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}, Seeds: {len(seeds)}"
                )

# --- Simulation ---
elif menu == "Simulation":
    st.title("Simulate Rumor Spread")

    if not st.session_state["user"]:
        st.warning("⚠️ Please login first.")
        st.stop()

    df = list_datasets(owner=st.session_state["user"])
    if df.empty:
        st.info("No datasets yet.")
        st.stop()

    # Dataset selector
    selected = st.selectbox(
        "Select dataset", df["name"] + " — " + df["id"], key="select_dataset_simulation"
    )
    ds_id = selected.split(" — ")[-1]

    edges = read_edges(ds_id)
    seeds = read_seeds(ds_id)

    if not edges:
        st.warning("⚠️ No edges uploaded for this dataset.")
        st.stop()

    G = build_graph(edges)
    nodes = list(G.nodes())

    st.subheader("Simulation Settings")

    # SLIDERS
    runs_mc = st.slider("Monte Carlo runs for label generation", 50, 1000, 200)
    p_sim = st.slider("Infection probability", 0.01, 1.0, 0.1)
    emb_dim = st.selectbox("Node2Vec embedding dim", [16, 32, 64, 128], index=2)
    rf_split = st.slider("Train/Test Split (%)", 10, 50, 20)

    method = st.selectbox(
        "Select simulation method",
        ["Node2Vec + Random Forest", "Graph Neural Network (GNN)"],
        key="simulation_method",
    )

    if st.button("Run Simulation", key="run_sim_btn"):

        # ==========================================================
        # ✅ STEP 1 — Generate Labels (Monte Carlo IC Spread)
        # ==========================================================
        with st.spinner("Generating labels using Monte Carlo IC..."):
            freq, _ = generate_ic_labels(G, seeds, p=p_sim, runs=runs_mc)

        y_freq = np.array([freq[n] for n in nodes])
        y_binary = (y_freq > 0).astype(int)

        st.write("### 📊 Label Statistics")
        st.dataframe(pd.Series(y_freq).describe())

        # ==========================================================
        # ✅ STEP 2 — MODEL TRAINING (NODE2VEC or GNN)
        # ==========================================================
        if method == "Node2Vec + Random Forest":
            st.subheader("Training Node2Vec + Random Forest Model")

            # Compute node2vec embeddings
            X = compute_node_features(G, node2vec_dim=emb_dim)
            X = X.reindex(nodes).fillna(0.0)

            clf, test_data, metrics = train_predict_model(
                X.values, y_binary, test_size=rf_split / 100.0
            )

            # Show Metrics
            auc_val = metrics.get("auc")
            pr_auc = metrics.get("pr_auc")
            st.metric("AUC (test)", f"{auc_val:.4f}" if auc_val else "N/A")
            st.metric("PR AUC (test)", f"{pr_auc:.4f}" if pr_auc else "N/A")

            probs_all = clf.predict_proba(X.values)[:, 1]
            prob_dict = {n: probs_all[i] for i, n in enumerate(nodes)}

        else:
            st.subheader("Training Graph Neural Network (GNN) Model")

            # 🔥 GNN training returns dict: node → risk probability
            prob_dict = train_gnn(G, y_binary, epochs=50)

        # ==========================================================
        # ✅ STEP 3 — VISUALIZE HIGH RISK NODES (PREDICTION ONLY)
        # ==========================================================
        st.write("### 🔥 High-Risk Nodes (Prediction Only)")
        high_risk = sorted(prob_dict, key=prob_dict.get, reverse=True)[:20]

        plot_graph_new(
            G, affected=high_risk, title="High-Risk Nodes (Prediction, NOT Infection)"
        )

        # ==========================================================
        # ✅ STEP 4 — RUN PREDICTED FLOW SIMULATION (Actual Infection)
        # ==========================================================
        st.write("### ✅ Actual Spread Simulation (Using Predicted Susceptibility)")

        mc_runs = st.number_input(
            "Monte Carlo runs for predicted-flow", min_value=10, max_value=300, value=50
        )

        infected_final = set()
        infected_list = []

        for _ in range(mc_runs):
            infected = simulate_ic_predicted(G, seeds, prob_dict, base_p=p_sim)
            infected_list.append(len(infected))
            infected_final |= infected

        st.write("### 📊 Infection Distribution Summary")
        st.dataframe(pd.Series(infected_list).describe())

        # Visualization of final spread
        plot_graph_new(
            G, affected=infected_final, title="Final Infected Nodes (Actual Spread)"
        )

        # ==========================================================
        # ✅ STEP 5 — DOWNLOAD RESULTS
        # ==========================================================
        buf = io.StringIO()
        pd.Series(infected_list).to_csv(buf, index=False)

        st.download_button(
            "⬇️ Download infection distribution CSV",
            buf.getvalue(),
            file_name=f"predicted_flow_{ds_id}.csv",
            mime="text/csv",
        )


# --- Containment ---
elif menu == "Containment":
    st.title("Rumor Containment Strategies")

    if not st.session_state["user"]:
        st.warning("⚠️ Please login first.")
        st.stop()

    df = list_datasets(owner=st.session_state["user"])
    if df.empty:
        st.info("No datasets yet.")
        st.stop()

    # Select dataset
    selected = st.selectbox(
        "Select dataset",
        df["name"] + " — " + df["id"],
        key="select_dataset_containment",
    )
    ds_id = selected.split(" — ")[-1]

    edges = read_edges(ds_id)
    seeds = read_seeds(ds_id)
    if not edges:
        st.warning("⚠️ No edges uploaded for this dataset.")
        st.stop()

    G = build_graph(edges)

    st.subheader("Containment Strategy Settings")

    p_sim = st.slider("Infection probability", 0.01, 1.0, 0.1)
    runs_containment = st.slider("Monte Carlo runs", 50, 500, 200)
    k_block = st.number_input(
        "Number of nodes to BLOCK", min_value=1, max_value=50, value=5
    )

    containment_method = st.selectbox(
        "Select containment strategy",
        ["Degree Centrality", "Betweenness Centrality", "PageRank"],
        key="containment_method",
    )

    if st.button("Run Containment Simulation", key="containment_run_btn"):
        # ---- Baseline spread (no blocking) ------------------------------------
        with st.spinner("Running baseline (no containment)..."):
            infected_baseline = set()
            for _ in range(runs_containment):
                infected = simulate_ic_spread(G, seeds, p_sim)
                infected_baseline |= infected

        st.write("### ✅ Baseline Spread (No Containment)")
        plot_graph_new(
            G, affected=infected_baseline, title="Baseline Spread (No Blocking)"
        )

        # ---- Determine nodes to block ----------------------------------------
        st.subheader("Blocked Nodes Based on Centrality")

        if containment_method == "Degree Centrality":
            centrality = nx.degree_centrality(G)
        elif containment_method == "Betweenness Centrality":
            centrality = nx.betweenness_centrality(G)
        else:
            centrality = nx.pagerank(G)

        # Pick top-k
        blocked_nodes = sorted(centrality, key=centrality.get, reverse=True)[:k_block]

        st.success(f"Blocked Nodes ({containment_method}): {blocked_nodes}")

        # ---- Contained Spread (after blocking) --------------------------------
        with st.spinner("Running spread with containment..."):
            infected_contained = set()
            for _ in range(runs_containment):
                infected = simulate_ic_spread(G, seeds, p_sim, blocked=blocked_nodes)
                infected_contained |= infected

        st.write("### ✅ Spread After Containment")
        plot_graph_new(
            G, affected=infected_contained, title="Spread After Blocking Key Nodes"
        )

        # ---- Comparison -------------------------------------------------------
        st.subheader("📊 Containment Effectiveness")

        baseline_count = len(infected_baseline)
        contained_count = len(infected_contained)
        reduction = baseline_count - contained_count
        reduction_pct = (reduction / baseline_count) * 100 if baseline_count > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Baseline Infected", baseline_count)
        col2.metric("After Containment", contained_count)
        col3.metric("Reduction (%)", f"{reduction_pct:.2f}%")

        st.write("### ✅ Visual Comparison")

        compare_fig = plt.figure(figsize=(4, 2))
        st.bar_chart({"Baseline": baseline_count, "After Containment": contained_count})

# --- Reports ---
elif menu == "Reports":
    st.title("Reports")
    if not st.session_state["user"]:
        st.warning("Login first")
    else:
        df = list_datasets(owner=st.session_state["user"])
        st.dataframe(df)
        if not df.empty:
            selected = st.selectbox("Select dataset", df["name"] + " — " + df["id"])
            ds_id = selected.split(" — ")[-1]
            edges = read_edges(ds_id)
            seeds = read_seeds(ds_id)
            if edges:
                G = build_graph(edges)
                # basic report
                report = {
                    "dataset_id": ds_id,
                    "dataset_name": json.loads(
                        (DATA_DIR / ds_id / "metadata.json").read_text()
                    )["name"],
                    "created_at": json.loads(
                        (DATA_DIR / ds_id / "metadata.json").read_text()
                    )["created_at"],
                    "nodes": G.number_of_nodes(),
                    "edges": G.number_of_edges(),
                    "seeds": len(seeds),
                }
                st.json(report)
                # offer downloads
                buf = io.StringIO()
                writer = csv.writer(buf)
                writer.writerow(["metric", "value"])
                for k, v in report.items():
                    writer.writerow([k, v])
                st.download_button(
                    "Download report CSV",
                    buf.getvalue(),
                    file_name=f"report_{ds_id}.csv",
                )

# End of file

# Ensure session state is initialized
if "page" not in st.session_state:
    st.session_state.page = "login"


# # Simple navigation function
# def go_to(page_name):
#     st.session_state.page = page_name
#     st.experimental_rerun()


# # Page router
# if st.session_state.page == "login":
#     st.title("🔐 Login Page")
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
#     if st.button("Login"):
#         # (replace with your auth check)
#         if username == "admin" and password == "admin":
#             st.success("Login successful!")
#             go_to("home")
#         else:
#             st.error("Invalid credentials")

# elif st.session_state.page == "register":
#     st.title("📝 Register Page")
#     new_user = st.text_input("New Username")
#     new_pass = st.text_input("Password", type="password")
#     if st.button("Register"):
#         st.success(f"User {new_user} registered! Now login.")
#         go_to("login")

# elif st.session_state.page == "home":
#     st.title("🏠 Dashboard")
#     st.write("Welcome to the Rumour Spread Simulation App!")
#     if st.button("Create Dataset"):
#         go_to("dataset")

# elif st.session_state.page == "dataset":
#     st.title("📂 Create Dataset")
#     st.write("Upload edges and rumour node files here.")
#     if st.button("Back"):
#         go_to("home")
