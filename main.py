import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# import torch.nn.functional as F
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
import reportlab

import tensorflow as tf
from spektral.layers import GCNConv

from tensorflow.keras import Model, Input

# from tensorflow.keras.layers import Dense
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


def save_json(ds_id: str, filename: str, payload: dict):
    path = DATA_DIR / ds_id / filename
    path.write_text(json.dumps(payload, indent=2))


def get_centrality(G, method: str):
    if method == "Degree Centrality":
        return nx.degree_centrality(G)
    elif method == "Betweenness Centrality":
        return nx.betweenness_centrality(G)
    elif method == "PageRank":
        return nx.pagerank(G)
    else:
        raise ValueError(f"Unknown method: {method}")


# class GCN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, out_channels)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         return x


# @st.cache_resource
# def load_model():
#     model = GCN(1, 16, 2)
#     model.load_state_dict(torch.load("gcn_model.pth", map_location="cpu"))
#     model.eval()
#     return model


def ai_select_nodes(prob_dict, G, k):
    """
    Combine ML prediction + graph importance
    """
    degree = nx.degree_centrality(G)
    pagerank = nx.pagerank(G)

    score = {}

    for n in G.nodes():
        score[n] = (
            0.6 * prob_dict.get(n, 0)  # ML prediction
            + 0.2 * degree.get(n, 0)
            + 0.2 * pagerank.get(n, 0)
        )

    ranked = sorted(score, key=score.get, reverse=True)
    return ranked[:k]


def load_json(ds_id: str, filename: str) -> dict | None:
    path = DATA_DIR / ds_id / filename
    if not path.exists():
        return None
    return json.loads(path.read_text())


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


import traceback


def show_exception(e: Exception, context: str = ""):
    st.error(f"❌ Error occurred {f'in {context}' if context else ''}")
    st.code(str(e), language="text")

    with st.expander("🔍 Show full traceback"):
        st.code(traceback.format_exc(), language="python")


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
    st.session_state.user = None

st.sidebar.markdown("### 👤 Session")

if st.session_state.user:
    st.sidebar.success(f"Logged in as\n**{st.session_state.user}**")

    if st.sidebar.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()
else:
    st.sidebar.info("Not logged in")

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
        "AI Containment",
        "Reports",
    ],
)

# --- Home ---
if menu == "Home":
    try:
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

    except Exception as e:
        show_exception(e, "Home Page")

# --- Register ---

elif menu == "Register":
    try:
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
            age = (
                today.year
                - dob.year
                - ((today.month, today.day) < (dob.month, dob.day))
            )

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

    except Exception as e:
        show_exception(e, "Register Page")


# --- Login ---
elif menu == "Login":
    try:
        st.title("Login")

        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            if authenticate_user(u, p):
                st.session_state.user = u
                st.success("✅ Logged in successfully")
                st.rerun()  # 🔥 IMPORTANT
            else:
                st.error("❌ Invalid credentials")

    except Exception as e:
        show_exception(e, "Login Page")

# --- Create Dataset ---
elif menu == "Create Dataset":
    try:
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
    except Exception as e:
        show_exception(e, "Dataset Page")

# --- Datasets (upload files) ---
elif menu == "Datasets":
    try:
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

    except Exception as e:
        show_exception(e, "Datasets Page")

# --- Visualization ---
elif menu == "Visualization":
    try:
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
                    st.info("Building Graph...")
                    G = build_graph(edges)
                    blocked = []
                    plt_obj = plot_graph_new(
                        G, affected=seeds, title="Visualization of Network"
                    )
                    st.info("Building Graph Successful")
                    st.write(
                        f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}, Seeds: {len(seeds)}"
                    )
                if st.button("💾 Save Visualization", key=f"save_vis_{ds_id}"):
                    density = nx.density(G)
                    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
                    clustering = nx.average_clustering(G)

                    payload = {
                        "dataset_id": ds_id,
                        "saved_at": datetime.utcnow().isoformat(),
                        "nodes": G.number_of_nodes(),
                        "edges": G.number_of_edges(),
                        "seeds": seeds,
                        "graph_metrics": {
                            "density": density,
                            "avg_degree": avg_degree,
                            "clustering": clustering,
                        },
                    }

                    save_json(ds_id, "visualization.json", payload)
                    st.success("Visualization saved successfully")
    except Exception as e:
        show_exception(e, "Visualization Page")

# --- Simulation ---
elif menu == "Simulation":
    try:
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
            "Select dataset",
            df["name"] + " — " + df["id"],
            key="select_dataset_simulation",
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
                G,
                affected=high_risk,
                title="High-Risk Nodes (Prediction, NOT Infection)",
            )

            # ==========================================================
            # ✅ STEP 4 — RUN PREDICTED FLOW SIMULATION (Actual Infection)
            # ==========================================================
            st.write("### ✅ Actual Spread Simulation (Using Predicted Susceptibility)")

            mc_runs = st.number_input(
                "Monte Carlo runs for predicted-flow",
                min_value=10,
                max_value=300,
                value=50,
            )

            infected_final = set()
            infected_list = []
            spread_runs = []
            for _ in range(mc_runs):
                infected = simulate_ic_predicted(G, seeds, prob_dict, base_p=p_sim)
                infected_list.append(len(infected))
                infected_final |= infected
                layers = independent_cascade(G, seeds, p=p_sim)
                spread_runs.append([list(layer) for layer in layers])

            st.write("### 📊 Infection Distribution Summary")
            st.dataframe(pd.Series(infected_list).describe())

            # Visualization of final spread
            plot_graph_new(
                G, affected=infected_final, title="Final Infected Nodes (Actual Spread)"
            )
            st.session_state["sim_data"] = {
                "method": method,
                "infection_probability": p_sim,
                "mc_runs": mc_runs,
                "high_risk_nodes": high_risk,
                "infected_final_count": len(infected_final),
                "infection_distribution": infected_list,
                "spread_layers": spread_runs,  # 🔥 NEW
            }

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

        if st.button("💾 Save Simulation Results", key=f"save_sim_{ds_id}"):

            if "sim_data" not in st.session_state:
                st.error("❌ Run simulation first.")
                st.stop()

            payload = {
                "dataset_id": ds_id,
                "saved_at": datetime.utcnow().isoformat(),
                **st.session_state["sim_data"],
            }
            st.success("Going to save Simulation results saved")
            save_json(ds_id, "simulation.json", payload)
            st.success("Simulation results saved")
    except Exception as e:
        show_exception(e, "Simulation Page")


# --- Containment ---
elif menu == "Containment":
    try:
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

        containment_methods = st.multiselect(
            "Select containment strategy (one or more)",
            ["Degree Centrality", "Betweenness Centrality", "PageRank"],
            default=["Degree Centrality"],
        )

        if st.button("Run Containment Simulation", key="containment_run_btn"):

            if not containment_methods:
                st.error("❌ Please select at least one containment strategy.")
                st.stop()

            # ---- Baseline (run once) -----------------------------------
            with st.spinner("Running baseline (no containment)..."):
                infected_baseline = set()
                for _ in range(runs_containment):
                    infected_baseline |= simulate_ic_spread(G, seeds, p_sim)

            baseline_count = len(infected_baseline)

            st.write("### ✅ Baseline Spread (No Containment)")
            plot_graph_new(G, affected=infected_baseline, title="Baseline Spread")

            # ---- Run each containment strategy -------------------------
            results = {}

            for method in containment_methods:
                st.subheader(f"🛡️ Containment: {method}")

                centrality = get_centrality(G, method)
                blocked_nodes = sorted(centrality, key=centrality.get, reverse=True)[
                    :k_block
                ]

                with st.spinner(f"Running containment using {method}..."):
                    infected_contained = set()
                    for _ in range(runs_containment):
                        infected_contained |= simulate_ic_spread(
                            G, seeds, p_sim, blocked=blocked_nodes
                        )

                contained_count = len(infected_contained)
                reduction = baseline_count - contained_count
                reduction_pct = (
                    (reduction / baseline_count) * 100 if baseline_count else 0
                )

                # ---- Visuals per method --------------------------------
                plot_graph_new(
                    G,
                    affected=infected_contained,
                    title=f"Spread After {method}",
                )

                col1, col2, col3 = st.columns(3)
                col1.metric("Baseline", baseline_count)
                col2.metric("After Containment", contained_count)
                col3.metric("Reduction (%)", f"{reduction_pct:.2f}%")

                # ---- Store result --------------------------------------
                results[method] = {
                    "blocked_nodes": blocked_nodes,
                    "baseline_infected": baseline_count,
                    "contained_infected": contained_count,
                    "reduction_pct": reduction_pct,
                    "infection_probability": p_sim,
                    "runs": runs_containment,
                }

            # ---- Comparison Mode --------------------------------------
            if len(results) > 1:
                st.subheader("📊 Containment Strategy Comparison")

                comp_df = pd.DataFrame.from_dict(results, orient="index")[
                    ["contained_infected", "reduction_pct"]
                ]

                st.dataframe(comp_df)

                st.bar_chart(comp_df["contained_infected"])

            # ---- Persist in session -----------------------------------
            st.session_state["con_data"] = results

        if st.button("💾 Save Containment Results", key=f"save_cont_{ds_id}"):
            if "con_data" not in st.session_state:
                st.error("❌ Run Containment first.")
                st.stop()

            payload = {
                "dataset_id": ds_id,
                "saved_at": datetime.utcnow().isoformat(),
                **st.session_state["con_data"],
            }
            st.success("Going to save Containment results saved")
            save_json(ds_id, "containment.json", payload)

    except Exception as e:
        show_exception(e, "Containment Page")


elif menu == "AI Containment":
    try:
        st.title("🤖 AI-Based Rumor Containment (ML-Based)")

        if not st.session_state["user"]:
            st.warning("⚠️ Please login first.")
            st.stop()

        df = list_datasets(owner=st.session_state["user"])
        if df.empty:
            st.info("No datasets yet.")
            st.stop()

        selected = st.selectbox(
            "Select dataset",
            df["name"] + " — " + df["id"],
        )
        ds_id = selected.split(" — ")[-1]

        edges = read_edges(ds_id)
        seeds = read_seeds(ds_id)

        if not edges:
            st.warning("No edges found")
            st.stop()

        G = build_graph(edges)
        nodes = list(G.nodes())

        p_sim = st.slider("Infection probability", 0.01, 1.0, 0.1)
        k_block = st.number_input("Nodes to block", 1, 50, 5)

        if st.button("🚀 Run AI Containment"):

            with st.spinner("Running baseline (no containment)..."):
                infected_baseline = set()
                runs_containment = st.slider("Monte Carlo runs", 50, 500, 200)
                for _ in range(runs_containment):
                    infected_baseline |= simulate_ic_spread(G, seeds, p_sim)

            baseline_count = len(infected_baseline)

            st.write("### ✅ Baseline Spread (No Containment)")
            plot_graph_new(G, affected=infected_baseline, title="Baseline Spread")

            # 🔹 STEP 1: Generate labels
            freq, _ = generate_ic_labels(G, seeds, p=p_sim, runs=200)
            y_binary = np.array([(freq[n] > 0) for n in nodes]).astype(int)

            # 🔹 STEP 2: Train ML model
            X = compute_node_features(G)
            X = X.reindex(nodes).fillna(0.0)

            clf, _, _ = train_predict_model(X.values, y_binary)

            probs = clf.predict_proba(X.values)[:, 1]
            prob_dict = {n: probs[i] for i, n in enumerate(nodes)}

            # 🔹 STEP 3: AI Node Selection
            blocked_nodes = ai_select_nodes(prob_dict, G, k_block)

            st.write("### 🚫 AI Selected Nodes")
            st.write(blocked_nodes)

            # 🔹 STEP 4: Baseline
            baseline = set()
            for _ in range(100):
                baseline |= simulate_ic_spread(G, seeds, p_sim)

            # 🔹 STEP 5: Containment
            contained = set()
            for _ in range(100):
                contained |= simulate_ic_spread(G, seeds, p_sim, blocked=blocked_nodes)

            # 🔹 STEP 6: Metrics
            b = len(baseline)
            c = len(contained)
            reduction = ((b - c) / b) * 100 if b else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Baseline", b)
            col2.metric("After AI", c)
            col3.metric("Reduction %", f"{reduction:.2f}")

            results = {
                "blocked_nodes": blocked_nodes,
                "baseline_infected": b,
                "contained_infected": c,
                "reduction_pct": reduction,
                "infection_probability": p_sim,
                "runs": runs_containment,
                "model_type": "Node2Vec + RandomForest",  # NEW
            }
            st.session_state["AI_con_data"] = results

            # 🔹 STEP 7: Visualization
            plot_graph_new(G, affected=contained, title="AI Containment Result")

        if st.button("💾 Save AI Containment Results", key=f"save_cont_{ds_id}"):
            if "AI_con_data" not in st.session_state:
                st.error("❌ Run AI Containment first.")
                st.stop()

            payload = {
                "dataset_id": ds_id,
                "saved_at": datetime.utcnow().isoformat(),
                **st.session_state["AI_con_data"],
            }
            st.success("Going to save AI Containment results saved")
            save_json(ds_id, "AIcontainment.json", payload)

    except Exception as e:
        show_exception(e, "AI Containment")

# --- Reports ---
# --- Reports ---
elif menu == "Reports":
    try:
        st.title("📑 Comprehensive Rumor Analysis Report")

        if not st.session_state["user"]:
            st.warning("Login first")
            st.stop()

        df = list_datasets(owner=st.session_state["user"])
        if df.empty:
            st.info("No datasets yet.")
            st.stop()

        selected = st.selectbox("Select dataset", df["name"] + " — " + df["id"])
        ds_id = selected.split(" — ")[-1]

        meta = json.loads((DATA_DIR / ds_id / "metadata.json").read_text())
        vis = load_json(ds_id, "visualization.json")
        sim = load_json(ds_id, "simulation.json")
        cont = load_json(ds_id, "containment.json")
        ai = load_json(ds_id, "AIcontainment.json")

        # Load graph for additional analysis
        edges = read_edges(ds_id)
        seeds = read_seeds(ds_id)
        G = build_graph(edges) if edges else None

        # ================= STYLE =================
        st.markdown(
            """
        <style>
        .card {
            background: #ffffff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .section-title {
            font-size: 22px;
            font-weight: bold;
            color: #2c3e50;
        }
        .executive-summary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #667eea;
        }
        .recommendation-box {
            background: #e8f5e9;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #4caf50;
            margin: 10px 0;
        }
        .warning-box {
            background: #fff3e0;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #ff9800;
            margin: 10px 0;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # ================= EXECUTIVE SUMMARY =================
        st.markdown('<div class="executive-summary">', unsafe_allow_html=True)
        st.markdown("## 📋 Executive Summary")

        # Generate dynamic summary
        summary_points = []

        if vis:
            network_size = (
                "large"
                if vis["nodes"] > 1000
                else "medium" if vis["nodes"] > 100 else "small"
            )
            summary_points.append(
                f"Network contains **{vis['nodes']} nodes** and **{vis['edges']} edges** ({network_size} scale)"
            )

        if sim:
            spread_severity = (
                "high"
                if sim["infected_final_count"] > vis["nodes"] * 0.5
                else (
                    "moderate"
                    if sim["infected_final_count"] > vis["nodes"] * 0.2
                    else "low"
                )
            )
            summary_points.append(
                f"Rumor spread severity: **{spread_severity}** ({sim['infected_final_count']} nodes infected)"
            )

        if cont:
            best_method = max(
                cont.items(),
                key=lambda x: (
                    x[1].get("reduction_pct", 0) if isinstance(x[1], dict) else 0
                ),
            )
            if isinstance(best_method[1], dict):
                summary_points.append(
                    f"Most effective containment: **{best_method[0]}** ({best_method[1]['reduction_pct']:.1f}% reduction)"
                )

        if ai:
            summary_points.append(
                f"AI containment achieved **{ai['reduction_pct']:.1f}%** reduction"
            )

        for point in summary_points:
            st.markdown(f"• {point}")

        st.markdown("</div>", unsafe_allow_html=True)

        # ================= 1. DATASET DETAILS =================
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">📊 Dataset Details</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Dataset Name", meta["name"])
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Owner", meta["owner"])
            st.markdown("</div>", unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Created", meta["created_at"][:10])
            st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("ℹ️ What is this dataset?"):
            st.write(
                """
            This dataset represents a **network graph** where:
            - **Nodes** = Users, entities, or endpoints in the network
            - **Edges** = Connections, relationships, or communication channels between nodes
            - **Seeds** = Initial rumor sources (where the misinformation originates)

            The analysis examines how information (or misinformation) propagates through this network structure.
            """
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # ================= 2. NETWORK TOPOLOGY ANALYSIS (NEW) =================
        if G and vis:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-title">🔬 Network Topology Analysis</div>',
                unsafe_allow_html=True,
            )

            col1, col2, col3, col4 = st.columns(4)

            density = nx.density(G)
            avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
            clustering = nx.average_clustering(G)

            # Calculate additional metrics
            try:
                if nx.is_connected(G):
                    diameter = nx.diameter(G)
                    avg_path = nx.average_shortest_path_length(G)
                else:
                    largest_cc = max(nx.connected_components(G), key=len)
                    subgraph = G.subgraph(largest_cc)
                    diameter = nx.diameter(subgraph)
                    avg_path = nx.average_shortest_path_length(subgraph)
            except:
                diameter = "N/A"
                avg_path = "N/A"

            num_components = nx.number_connected_components(G)

            with col1:
                st.metric("Density", f"{density:.4f}")
            with col2:
                st.metric("Avg Degree", f"{avg_degree:.2f}")
            with col3:
                st.metric("Clustering Coef.", f"{clustering:.4f}")
            with col4:
                st.metric("Components", num_components)

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Network Diameter",
                    diameter if isinstance(diameter, str) else f"{diameter}",
                )
            with col2:
                st.metric(
                    "Avg Path Length",
                    avg_path if isinstance(avg_path, str) else f"{avg_path:.2f}",
                )

            # Degree Distribution
            st.subheader("Degree Distribution")
            degrees = [d for n, d in G.degree()]
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))

            ax[0].hist(degrees, bins=30, edgecolor="black", alpha=0.7, color="#667eea")
            ax[0].set_xlabel("Degree")
            ax[0].set_ylabel("Frequency")
            ax[0].set_title("Degree Distribution (Linear)")

            # Log-log plot for power law detection
            from collections import Counter

            degree_count = Counter(degrees)
            deg, cnt = zip(*sorted(degree_count.items()))
            ax[1].loglog(deg, cnt, "o", markersize=5, color="#764ba2")
            ax[1].set_xlabel("Degree (log)")
            ax[1].set_ylabel("Count (log)")
            ax[1].set_title("Degree Distribution (Log-Log)")

            plt.tight_layout()
            st.pyplot(fig)

            with st.expander("ℹ️ Understanding Network Metrics"):
                st.write(
                    """
                - **Density**: Ratio of actual edges to possible edges. Higher density = more interconnected network.
                - **Average Degree**: Mean number of connections per node.
                - **Clustering Coefficient**: Probability that neighbors of a node are also connected.
                - **Diameter**: Longest shortest path between any two nodes.
                - **Average Path Length**: Mean shortest path between all node pairs.
                - **Components**: Number of disconnected subgraphs.

                **Implications for Rumor Spread:**
                - High density → Faster spread, harder to contain
                - High clustering → Local containment possible
                - Small diameter → Rapid network-wide propagation
                """
                )

            st.markdown("</div>", unsafe_allow_html=True)

        # ================= 3. NETWORK VISUALIZATION =================
        if vis:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-title">🌐 Network Visualization</div>',
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Nodes", vis["nodes"])
            with col2:
                st.metric("Total Edges", vis["edges"])
            with col3:
                st.metric("Seed Nodes", len(vis["seeds"]))

            if "graph_metrics" in vis:
                st.subheader("Graph Metrics at Visualization Time")
                gm = vis["graph_metrics"]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Density", f"{gm.get('density', 0):.4f}")
                with col2:
                    st.metric("Avg Degree", f"{gm.get('avg_degree', 0):.2f}")
                with col3:
                    st.metric("Clustering", f"{gm.get('clustering', 0):.4f}")

            st.markdown("</div>", unsafe_allow_html=True)

        # ================= 4. SPREAD DYNAMICS =================
        if sim and "spread_layers" in sim:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-title">📈 Spread Dynamics Analysis</div>',
                unsafe_allow_html=True,
            )

            runs = sim["spread_layers"]
            curves = []

            for run in runs:
                c = 0
                temp = []
                for layer in run:
                    c += len(layer)
                    temp.append(c)
                curves.append(temp)

            max_len = max(len(c) for c in curves)

            # Calculate statistics
            avg_curve = []
            min_curve = []
            max_curve = []
            std_curve = []

            for i in range(max_len):
                vals = [c[i] if i < len(c) else c[-1] for c in curves]
                avg_curve.append(np.mean(vals))
                min_curve.append(np.min(vals))
                max_curve.append(np.max(vals))
                std_curve.append(np.std(vals))

            # Calculate velocity (rate of change)
            velocity = [0] + [
                avg_curve[i] - avg_curve[i - 1] for i in range(1, len(avg_curve))
            ]
            peak_velocity_step = velocity.index(max(velocity))
            peak_velocity = max(velocity)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Peak Spread Velocity", f"{peak_velocity:.1f} nodes/step")
            with col2:
                st.metric("Peak at Step", peak_velocity_step)
            with col3:
                saturation_pct = (avg_curve[-1] / vis["nodes"] * 100) if vis else 0
                st.metric("Saturation", f"{saturation_pct:.1f}%")

            # Plot with confidence interval
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Spread curve with confidence band
            x = range(len(avg_curve))
            axes[0].fill_between(
                x, min_curve, max_curve, alpha=0.3, color="#667eea", label="Range"
            )
            axes[0].plot(avg_curve, color="#667eea", linewidth=2, label="Average")
            axes[0].set_title("Rumor Spread Over Time")
            axes[0].set_xlabel("Time Steps")
            axes[0].set_ylabel("Cumulative Infected Nodes")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Velocity curve
            axes[1].bar(range(len(velocity)), velocity, color="#764ba2", alpha=0.7)
            axes[1].axhline(
                y=np.mean(velocity),
                color="red",
                linestyle="--",
                label=f"Avg: {np.mean(velocity):.1f}",
            )
            axes[1].set_title("Spread Velocity (New Infections per Step)")
            axes[1].set_xlabel("Time Steps")
            axes[1].set_ylabel("New Infections")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

            with st.expander("ℹ️ Interpreting Spread Dynamics"):
                st.write(
                    """
                **Spread Curve Analysis:**
                - **S-curve pattern**: Typical epidemic spread (slow start → rapid growth → saturation)
                - **Linear growth**: Indicates constant spread rate, possibly due to network structure
                - **Exponential growth**: Highly concerning, requires immediate intervention

                **Velocity Analysis:**
                - Peak velocity indicates when the rumor spreads fastest
                - Early peak → Network has high initial connectivity from seeds
                - Late peak → Rumor takes time to reach highly connected nodes
                """
                )

            st.markdown("</div>", unsafe_allow_html=True)

        # ================= 5. SIMULATION DETAILS =================
        if sim:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-title">🔥 Rumor Spread Simulation Results</div>',
                unsafe_allow_html=True,
            )

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Method", sim["method"].split()[0])
            with col2:
                st.metric("Infection Prob.", f"{sim['infection_probability']:.2f}")
            with col3:
                st.metric("MC Runs", sim["mc_runs"])
            with col4:
                st.metric("Final Infected", sim["infected_final_count"])

            # Distribution analysis
            st.subheader("Infection Distribution Analysis")

            infection_data = sim["infection_distribution"]
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            # Histogram
            axes[0].hist(
                infection_data, bins=20, edgecolor="black", alpha=0.7, color="#e74c3c"
            )
            axes[0].axvline(
                np.mean(infection_data),
                color="blue",
                linestyle="--",
                label=f"Mean: {np.mean(infection_data):.1f}",
            )
            axes[0].axvline(
                np.median(infection_data),
                color="green",
                linestyle="--",
                label=f"Median: {np.median(infection_data):.1f}",
            )
            axes[0].set_xlabel("Infected Count")
            axes[0].set_ylabel("Frequency")
            axes[0].set_title("Infection Distribution")
            axes[0].legend()

            # Box plot
            axes[1].boxplot(infection_data, vert=True)
            axes[1].set_ylabel("Infected Count")
            axes[1].set_title("Distribution Box Plot")

            # CDF
            sorted_data = np.sort(infection_data)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            axes[2].plot(sorted_data, cdf, color="#3498db", linewidth=2)
            axes[2].set_xlabel("Infected Count")
            axes[2].set_ylabel("Cumulative Probability")
            axes[2].set_title("Cumulative Distribution")
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

            # Statistics table
            st.subheader("Statistical Summary")
            stats_df = pd.DataFrame(
                {
                    "Statistic": [
                        "Mean",
                        "Median",
                        "Std Dev",
                        "Min",
                        "Max",
                        "25th Percentile",
                        "75th Percentile",
                    ],
                    "Value": [
                        f"{np.mean(infection_data):.2f}",
                        f"{np.median(infection_data):.2f}",
                        f"{np.std(infection_data):.2f}",
                        f"{np.min(infection_data)}",
                        f"{np.max(infection_data)}",
                        f"{np.percentile(infection_data, 25):.2f}",
                        f"{np.percentile(infection_data, 75):.2f}",
                    ],
                }
            )
            st.table(stats_df)

            with st.expander("ℹ️ Understanding the Simulation"):
                st.write(
                    """
                **Monte Carlo Simulation** runs the spread process multiple times with randomization to:
                - Capture inherent uncertainty in real-world spreading
                - Estimate average and worst-case scenarios
                - Provide confidence in results

                **Model Types:**
                - **Node2Vec + Random Forest**: Converts graph structure to numerical embeddings, then uses ensemble learning
                - **GNN (Graph Neural Network)**: Deep learning directly on graph structure

                **Infection Probability**: The likelihood that an infected node transmits to each neighbor per time step.
                """
                )

            st.markdown("</div>", unsafe_allow_html=True)

        # ================= 6. HIGH RISK NODES =================
        if sim and G:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-title">🎯 High Risk Node Analysis</div>',
                unsafe_allow_html=True,
            )

            high_risk = sim["high_risk_nodes"]

            # Create detailed table
            risk_data = []
            for i, node in enumerate(high_risk):
                node_degree = G.degree(node)
                neighbors = list(G.neighbors(node))
                risk_data.append(
                    {
                        "Rank": i + 1,
                        "Node": node,
                        "Degree": node_degree,
                        "Neighbors": len(neighbors),
                        "Is Seed": "✓" if node in seeds else "✗",
                    }
                )

            risk_df = pd.DataFrame(risk_data)
            st.dataframe(risk_df, use_container_width=True)

            # Visualize high-risk node degrees
            fig, ax = plt.subplots(figsize=(10, 4))
            colors = [
                "#e74c3c" if row["Is Seed"] == "✓" else "#3498db"
                for _, row in risk_df.iterrows()
            ]
            ax.bar(range(len(risk_df)), risk_df["Degree"], color=colors)
            ax.set_xticks(range(len(risk_df)))
            ax.set_xticklabels(risk_df["Node"], rotation=45, ha="right")
            ax.set_xlabel("Node")
            ax.set_ylabel("Degree")
            ax.set_title("High-Risk Nodes by Degree (Red = Seed)")
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.write(
                f"⚠️ **Critical Finding**: The top {len(high_risk)} high-risk nodes should be prioritized for monitoring or intervention."
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # ================= 7. CONTAINMENT STRATEGIES =================
        if cont:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-title">🛡️ Containment Strategy Analysis</div>',
                unsafe_allow_html=True,
            )

            methods = []
            baseline_vals = []
            contained_vals = []
            reductions = []
            blocked_counts = []

            for m, d in cont.items():
                if m in ["dataset_id", "saved_at"]:
                    continue
                if isinstance(d, dict):
                    methods.append(m)
                    baseline_vals.append(d.get("baseline_infected", 0))
                    contained_vals.append(d.get("contained_infected", 0))
                    reductions.append(d.get("reduction_pct", 0))
                    blocked_counts.append(len(d.get("blocked_nodes", [])))

            if methods:
                # Summary table
                summary_df = pd.DataFrame(
                    {
                        "Strategy": methods,
                        "Nodes Blocked": blocked_counts,
                        "Baseline Infected": baseline_vals,
                        "After Containment": contained_vals,
                        "Reduction (%)": [f"{r:.2f}%" for r in reductions],
                    }
                )
                st.table(summary_df)

                # Comparison chart
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                x = np.arange(len(methods))
                width = 0.35

                axes[0].bar(
                    x - width / 2,
                    baseline_vals,
                    width,
                    label="Baseline",
                    color="#e74c3c",
                )
                axes[0].bar(
                    x + width / 2,
                    contained_vals,
                    width,
                    label="After Containment",
                    color="#27ae60",
                )
                axes[0].set_xticks(x)
                axes[0].set_xticklabels(methods, rotation=45, ha="right")
                axes[0].set_ylabel("Infected Nodes")
                axes[0].set_title("Baseline vs Contained Infections")
                axes[0].legend()

                axes[1].bar(methods, reductions, color="#3498db")
                axes[1].set_ylabel("Reduction (%)")
                axes[1].set_title("Effectiveness by Strategy")
                for i, v in enumerate(reductions):
                    axes[1].text(i, v + 0.5, f"{v:.1f}%", ha="center")

                plt.tight_layout()
                st.pyplot(fig)

                # Best strategy recommendation
                best_idx = reductions.index(max(reductions))
                st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                st.write(
                    f"✅ **Recommended Strategy**: {methods[best_idx]} achieves the highest reduction of {reductions[best_idx]:.2f}%"
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("ℹ️ Understanding Containment Methods"):
                st.write(
                    """
                **Degree Centrality**: Blocks nodes with the most connections. Simple and effective for hub-based networks.

                **Betweenness Centrality**: Blocks nodes that act as bridges between communities. Effective for networks with clear community structure.

                **PageRank**: Blocks nodes with high global importance (like Google's algorithm). Considers both direct and indirect influence.

                **When to use each:**
                - Dense networks → Degree Centrality
                - Community-structured networks → Betweenness Centrality
                - Complex hierarchical networks → PageRank
                """
                )

            st.markdown("</div>", unsafe_allow_html=True)

        # ================= 8. AI CONTAINMENT =================
        if ai:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-title">🤖 AI-Based Containment Results</div>',
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Baseline Infected", ai.get("baseline_infected", "N/A"))
            with col2:
                st.metric("After AI Containment", ai.get("contained_infected", "N/A"))
            with col3:
                st.metric("Reduction", f"{ai.get('reduction_pct', 0):.2f}%")

            st.subheader("AI-Selected Blocking Nodes")
            st.write(ai.get("blocked_nodes", []))

            # Compare AI vs traditional (if both available)
            if cont and methods:
                st.subheader("AI vs Traditional Methods Comparison")

                all_methods = methods + ["AI-Based"]
                all_reductions = reductions + [ai.get("reduction_pct", 0)]

                fig, ax = plt.subplots(figsize=(10, 5))
                colors = ["#3498db"] * len(methods) + ["#9b59b6"]
                bars = ax.bar(all_methods, all_reductions, color=colors)

                # Highlight best
                best_overall = max(all_reductions)
                for bar, red in zip(bars, all_reductions):
                    if red == best_overall:
                        bar.set_edgecolor("gold")
                        bar.set_linewidth(3)

                ax.set_ylabel("Reduction (%)")
                ax.set_title("All Containment Methods Comparison")
                for i, v in enumerate(all_reductions):
                    ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")

                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)

            with st.expander("ℹ️ How AI Containment Works"):
                st.write(
                    f"""
                **Model Used**: {ai.get('model_type', 'ML-Based')}

                **Process:**
                1. Train a machine learning model on network structure and spread patterns
                2. Predict which nodes are most likely to spread the rumor
                3. Select blocking candidates based on predicted risk AND network position
                4. Simulate containment effectiveness

                **Advantages over traditional methods:**
                - Adapts to specific network characteristics
                - Considers both local and global features
                - Can learn from historical spread patterns
                """
                )

            st.markdown("</div>", unsafe_allow_html=True)

        # ================= 9. RECOMMENDATIONS =================
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">💡 Recommendations & Action Items</div>',
            unsafe_allow_html=True,
        )

        recommendations = []

        # Generate dynamic recommendations
        if vis:
            density = vis.get("graph_metrics", {}).get("density", 0)
            if density > 0.1:
                recommendations.append(
                    {
                        "priority": "High",
                        "category": "Network Structure",
                        "recommendation": "Network is highly dense. Consider implementing broad monitoring systems.",
                        "action": "Deploy network-wide alerts for unusual activity patterns.",
                    }
                )
            else:
                recommendations.append(
                    {
                        "priority": "Medium",
                        "category": "Network Structure",
                        "recommendation": "Network has moderate density. Targeted interventions should be effective.",
                        "action": "Focus resources on high-centrality nodes.",
                    }
                )

        if sim:
            if sim["infected_final_count"] > vis["nodes"] * 0.5:
                recommendations.append(
                    {
                        "priority": "Critical",
                        "category": "Spread Risk",
                        "recommendation": f"Potential spread affects over 50% of network ({sim['infected_final_count']} nodes).",
                        "action": "Implement immediate containment measures on identified high-risk nodes.",
                    }
                )

        if cont:
            best_method = max(
                [
                    (m, d.get("reduction_pct", 0))
                    for m, d in cont.items()
                    if isinstance(d, dict)
                ],
                key=lambda x: x[1],
                default=None,
            )
            if best_method:
                recommendations.append(
                    {
                        "priority": "High",
                        "category": "Containment",
                        "recommendation": f"{best_method[0]} is the most effective traditional method ({best_method[1]:.1f}% reduction).",
                        "action": f"Prioritize blocking nodes identified by {best_method[0]} analysis.",
                    }
                )

        if ai and ai.get("reduction_pct", 0) > 0:
            recommendations.append(
                {
                    "priority": "High",
                    "category": "AI Strategy",
                    "recommendation": f"AI containment achieves {ai['reduction_pct']:.1f}% reduction.",
                    "action": f"Block the following nodes: {', '.join(map(str, ai.get('blocked_nodes', [])[:5]))}...",
                }
            )

        # Display recommendations
        for rec in recommendations:
            priority_color = {
                "Critical": "#e74c3c",
                "High": "#f39c12",
                "Medium": "#3498db",
                "Low": "#27ae60",
            }
            st.markdown(
                f"""
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid {priority_color.get(rec['priority'], '#gray')}">
                <strong>[{rec['priority']}] {rec['category']}</strong><br>
                {rec['recommendation']}<br>
                <em>Action: {rec['action']}</em>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # ================= 10. FINAL INSIGHTS =================
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">🧠 Key Insights Summary</div>',
            unsafe_allow_html=True,
        )

        insights = [
            "Dense networks lead to faster rumor spread and require broader containment strategies",
            "A small number of highly-connected nodes (hubs) control most of the spread",
            "Centrality-based methods are effective for targeted intervention",
            "AI-based methods can adapt to specific network patterns for improved results",
            "Early intervention at peak velocity yields maximum impact",
            "Monte Carlo simulations provide confidence intervals for risk assessment",
        ]

        for insight in insights:
            st.markdown(f"• {insight}")

        st.markdown("</div>", unsafe_allow_html=True)

        # ================= PDF EXPORT =================
        st.markdown("---")
        st.subheader("📥 Export Report")

        def generate_pdf_report():
            """Generate a comprehensive PDF report"""
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import (
                SimpleDocTemplate,
                Paragraph,
                Spacer,
                Table,
                TableStyle,
                Image,
                PageBreak,
            )
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            import io
            import matplotlib

            matplotlib.use("Agg")

            buffer = io.BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72,
            )

            styles = getSampleStyleSheet()
            # Use unique names to avoid conflicts with built-in styles
            styles.add(ParagraphStyle(name='CustomTitle', fontSize=24, spaceAfter=30,
                                    textColor=colors.HexColor('#2c3e50'), fontName='Helvetica-Bold'))
            styles.add(ParagraphStyle(name='SectionHeader', fontSize=16, spaceAfter=12, spaceBefore=20,
                                    textColor=colors.HexColor('#667eea'), fontName='Helvetica-Bold'))
            styles.add(ParagraphStyle(name='SubHeaderCustom', fontSize=12, spaceAfter=6, spaceBefore=10,
                                    fontName='Helvetica-Bold'))
            styles.add(ParagraphStyle(name='BodyTextCustom', fontSize=10, spaceAfter=6, leading=14))

            story = []

            # Title
            story.append(
                Paragraph("Comprehensive Rumor Analysis Report", styles["CustomTitle"])
            )
            story.append(Paragraph(f"Dataset: {meta['name']}", styles["SubHeaderCustom"]))
            story.append(
                Paragraph(
                    f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
                    styles["BodyTextCustom"],
                )
            )
            story.append(Spacer(1, 20))

            # Executive Summary
            story.append(Paragraph("Executive Summary", styles["SectionHeader"]))
            for point in summary_points:
                clean_point = point.replace("**", "")
                story.append(Paragraph(f"• {clean_point}", styles["BodyTextCustom"]))
            story.append(Spacer(1, 12))

            # Dataset Details
            story.append(Paragraph("1. Dataset Details", styles["SectionHeader"]))
            dataset_data = [
                ["Property", "Value"],
                ["Dataset Name", meta["name"]],
                ["Owner", meta["owner"]],
                ["Created At", meta["created_at"]],
            ]
            t = Table(dataset_data, colWidths=[2 * inch, 4 * inch])
            t.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#667eea")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
                        ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#ddd")),
                    ]
                )
            )
            story.append(t)
            story.append(Spacer(1, 12))

            # Network Metrics
            if vis:
                story.append(Paragraph("2. Network Topology", styles["SectionHeader"]))
                network_data = [
                    ["Metric", "Value"],
                    ["Total Nodes", str(vis["nodes"])],
                    ["Total Edges", str(vis["edges"])],
                    ["Seed Nodes", str(len(vis["seeds"]))],
                ]
                if "graph_metrics" in vis:
                    gm = vis["graph_metrics"]
                    network_data.extend(
                        [
                            ["Density", f"{gm.get('density', 0):.4f}"],
                            ["Average Degree", f"{gm.get('avg_degree', 0):.2f}"],
                            [
                                "Clustering Coefficient",
                                f"{gm.get('clustering', 0):.4f}",
                            ],
                        ]
                    )
                t = Table(network_data, colWidths=[2.5 * inch, 3.5 * inch])
                t.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#667eea")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                            (
                                "BACKGROUND",
                                (0, 1),
                                (-1, -1),
                                colors.HexColor("#f8f9fa"),
                            ),
                            ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#ddd")),
                        ]
                    )
                )
                story.append(t)
                story.append(Spacer(1, 12))

            # Simulation Results
            if sim:
                story.append(
                    Paragraph("3. Simulation Results", styles["SectionHeader"])
                )
                sim_data = [
                    ["Parameter", "Value"],
                    ["Method", sim["method"]],
                    ["Infection Probability", str(sim["infection_probability"])],
                    ["Monte Carlo Runs", str(sim["mc_runs"])],
                    ["Final Infected Count", str(sim["infected_final_count"])],
                    ["Mean Infection", f"{np.mean(sim['infection_distribution']):.2f}"],
                    ["Std Deviation", f"{np.std(sim['infection_distribution']):.2f}"],
                ]
                t = Table(sim_data, colWidths=[2.5 * inch, 3.5 * inch])
                t.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e74c3c")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                            (
                                "BACKGROUND",
                                (0, 1),
                                (-1, -1),
                                colors.HexColor("#f8f9fa"),
                            ),
                            ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#ddd")),
                        ]
                    )
                )
                story.append(t)
                story.append(Spacer(1, 12))

                # High Risk Nodes
                story.append(Paragraph("High Risk Nodes", styles["SubHeaderCustom"]))
                risk_text = ", ".join(map(str, sim["high_risk_nodes"][:10]))
                story.append(Paragraph(f"Top 10: {risk_text}", styles["BodyTextCustom"]))
                story.append(Spacer(1, 12))

            # Containment Results
            if cont:
                story.append(PageBreak())
                story.append(
                    Paragraph(
                        "4. Containment Strategy Results", styles["SectionHeader"]
                    )
                )

                cont_data = [
                    ["Strategy", "Nodes Blocked", "Baseline", "After", "Reduction"]
                ]
                for m, d in cont.items():
                    if m in ["dataset_id", "saved_at"]:
                        continue
                    if isinstance(d, dict):
                        cont_data.append(
                            [
                                m,
                                str(len(d.get("blocked_nodes", []))),
                                str(d.get("baseline_infected", "N/A")),
                                str(d.get("contained_infected", "N/A")),
                                f"{d.get('reduction_pct', 0):.2f}%",
                            ]
                        )

                if len(cont_data) > 1:
                    t = Table(
                        cont_data,
                        colWidths=[1.5 * inch, 1 * inch, 1 * inch, 1 * inch, 1 * inch],
                    )
                    t.setStyle(
                        TableStyle(
                            [
                                (
                                    "BACKGROUND",
                                    (0, 0),
                                    (-1, 0),
                                    colors.HexColor("#27ae60"),
                                ),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                ("FONTSIZE", (0, 0), (-1, -1), 9),
                                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                                (
                                    "BACKGROUND",
                                    (0, 1),
                                    (-1, -1),
                                    colors.HexColor("#f8f9fa"),
                                ),
                                ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#ddd")),
                            ]
                        )
                    )
                    story.append(t)
                    story.append(Spacer(1, 12))

            # AI Containment
            if ai:
                story.append(
                    Paragraph("5. AI-Based Containment", styles["SectionHeader"])
                )
                ai_data = [
                    ["Metric", "Value"],
                    ["Model Type", ai.get("model_type", "ML-Based")],
                    ["Baseline Infected", str(ai.get("baseline_infected", "N/A"))],
                    ["After AI Containment", str(ai.get("contained_infected", "N/A"))],
                    ["Reduction", f"{ai.get('reduction_pct', 0):.2f}%"],
                    [
                        "Blocked Nodes",
                        ", ".join(map(str, ai.get("blocked_nodes", [])[:5])) + "...",
                    ],
                ]
                t = Table(ai_data, colWidths=[2.5 * inch, 3.5 * inch])
                t.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#9b59b6")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                            (
                                "BACKGROUND",
                                (0, 1),
                                (-1, -1),
                                colors.HexColor("#f8f9fa"),
                            ),
                            ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#ddd")),
                        ]
                    )
                )
                story.append(t)
                story.append(Spacer(1, 12))

            # Recommendations
            story.append(Paragraph("6. Recommendations", styles["SectionHeader"]))
            for rec in recommendations:
                story.append(
                    Paragraph(
                        f"[{rec['priority']}] {rec['category']}", styles["SubHeaderCustom"]
                    )
                )
                story.append(Paragraph(rec["recommendation"], styles["BodyTextCustom"]))
                story.append(Paragraph(f"Action: {rec['action']}", styles["BodyTextCustom"]))
                story.append(Spacer(1, 6))

            # Key Insights
            story.append(Paragraph("7. Key Insights", styles["SectionHeader"]))
            for insight in insights:
                story.append(Paragraph(f"• {insight}", styles["BodyTextCustom"]))

            # Build PDF
            doc.build(story)
            buffer.seek(0)
            return buffer

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📄 Generate PDF Report", key="gen_pdf"):
                with st.spinner("Generating PDF..."):
                    try:
                        pdf_buffer = generate_pdf_report()
                        st.session_state["pdf_buffer"] = pdf_buffer.getvalue()
                        st.success("✅ PDF generated successfully!")
                    except ImportError:
                        st.error(
                            "❌ reportlab library not installed. Run: `pip install reportlab`"
                        )
                    except Exception as e:
                        st.error(f"❌ Error generating PDF: {str(e)}")

        with col2:
            if "pdf_buffer" in st.session_state:
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=st.session_state["pdf_buffer"],
                    file_name=f"rumor_analysis_report_{ds_id}_{datetime.utcnow().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                )

        # Also offer JSON export
        st.markdown("---")
        if st.button("📊 Export Raw Data (JSON)", key="export_json"):
            export_data = {
                "metadata": meta,
                "visualization": vis,
                "simulation": sim,
                "containment": cont,
                "ai_containment": ai,
                "generated_at": datetime.utcnow().isoformat(),
            }
            st.download_button(
                label="⬇️ Download JSON",
                data=json.dumps(export_data, indent=2, default=str),
                file_name=f"rumor_data_{ds_id}.json",
                mime="application/json",
            )

    except Exception as e:
        show_exception(e, "Report Page")

# End of file

# Ensure session state is initialized
if "page" not in st.session_state:
    st.session_state.page = "login"
