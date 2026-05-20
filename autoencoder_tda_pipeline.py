import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Model
except ImportError:
    tf = None

try:
    import gudhi as gd
except ImportError:
    gd = None

try:
    import kmapper as km
except ImportError:
    km = None

try:
    import umap
except ImportError:
    umap = None

RANDOM_STATE = 42


def load_data(csv_path, feature_mode, n_features):
    raw = pd.read_csv(csv_path)
    numeric = raw.select_dtypes(include=[np.number]).dropna()
    if numeric.empty:
        raise ValueError('No usable numerical columns found.')
    if feature_mode == 'first':
        selected = numeric.iloc[:, :n_features]
    elif feature_mode == 'last':
        selected = numeric.iloc[:, -n_features:]
    else:
        selected = numeric
    scaled = StandardScaler().fit_transform(selected.values)
    return raw, selected, scaled


def train_autoencoder(x, latent_dim, epochs, batch_size):
    if tf is None:
        raise ImportError('TensorFlow is not installed.')
    inp = Input(shape=(x.shape[1],))
    z = Dense(64, activation='relu')(inp)
    z = Dense(32, activation='relu')(z)
    latent = Dense(latent_dim, activation='linear', name='latent_space')(z)
    z = Dense(32, activation='relu')(latent)
    z = Dense(64, activation='relu')(z)
    out = Dense(x.shape[1], activation='linear')(z)
    autoencoder = Model(inp, out)
    encoder = Model(inp, latent)
    autoencoder.compile(optimizer='adam', loss='mse')
    history = autoencoder.fit(x, x, validation_split=0.2, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)
    latent_x = encoder.predict(x, verbose=0)
    reconstructed = autoencoder.predict(x, verbose=0)
    return latent_x, mean_squared_error(x, reconstructed), history.history


def save_embedding(embedding, path, prefix):
    cols = [f'{prefix}_{i + 1}' for i in range(embedding.shape[1])]
    pd.DataFrame(embedding, columns=cols).to_csv(path, index=False)


def plot_embedding(embedding, title, path):
    plt.figure(figsize=(7, 5))
    if embedding.shape[1] == 3:
        ax = plt.axes(projection='3d')
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], s=12)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=12)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def run_pca(x, dim):
    dim = min(dim, x.shape[1])
    model = PCA(n_components=dim, random_state=RANDOM_STATE)
    return model.fit_transform(x), model.explained_variance_ratio_


def run_tsne(x, dim):
    perplexity = max(2, min(30, x.shape[0] - 1))
    model = TSNE(n_components=dim, perplexity=perplexity, random_state=RANDOM_STATE, init='pca', learning_rate='auto')
    return model.fit_transform(x)


def run_persistence(x, max_edge_length, max_dimension):
    if gd is None:
        raise ImportError('GUDHI is not installed.')
    rips = gd.RipsComplex(points=x, max_edge_length=max_edge_length)
    tree = rips.create_simplex_tree(max_dimension=max_dimension)
    tree.persistence()
    intervals_by_dim = {}
    summary = {}
    for dim in range(max_dimension + 1):
        intervals = tree.persistence_intervals_in_dimension(dim)
        intervals_by_dim[dim] = intervals
        finite = intervals[np.isfinite(intervals[:, 1])] if len(intervals) else np.empty((0, 2))
        summary[f'H{dim}_finite_intervals'] = int(len(finite))
        summary[f'H{dim}_total_intervals'] = int(len(intervals))
    return intervals_by_dim, summary


def save_persistence(intervals_by_dim, summary, output_dir):
    pd.DataFrame([summary]).to_csv(output_dir / 'betti_summary.csv', index=False)
    rows = []
    for dim, intervals in intervals_by_dim.items():
        for birth, death in intervals:
            rows.append({'dimension': dim, 'birth': birth, 'death': death})
    pd.DataFrame(rows).to_csv(output_dir / 'persistence_intervals.csv', index=False)


def run_mapper(x, output_dir, dim):
    if km is None or umap is None:
        raise ImportError('KeplerMapper or UMAP is not installed.')
    mapper = km.KeplerMapper(verbose=0)
    isomap_projection = Isomap(n_components=min(dim, x.shape[1])).fit_transform(x)
    projection = umap.UMAP(n_components=dim, random_state=RANDOM_STATE).fit_transform(isomap_projection)
    graph = mapper.map(projection, x, clusterer=DBSCAN(eps=0.5, min_samples=3), cover=km.Cover(n_cubes=10, perc_overlap=0.2))
    mapper.visualize(graph, path_html=str(output_dir / 'mapper_graph.html'), title='Mapper Graph for Dynamical-System Data')
    return {'nodes': len(graph.get('nodes', {})), 'links': sum(len(v) for v in graph.get('links', {}).values())}


def run_svr(raw):
    numeric = raw.select_dtypes(include=[np.number]).dropna()
    if numeric.shape[1] < 2:
        raise ValueError('SVR requires at least two numerical columns.')
    x = numeric.iloc[:, :-1]
    y = numeric.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = SVR(kernel='poly', degree=3, C=1.0, epsilon=0.1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return {'target_column': y.name, 'mse': float(mean_squared_error(y_test, y_pred)), 'r2': float(r2_score(y_test, y_pred))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--feature-mode', default='last', choices=['first', 'last', 'all'])
    parser.add_argument('--n-features', type=int, default=10)
    parser.add_argument('--latent-dim', type=int, default=2, choices=[2, 3])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-edge-length', type=float, default=2.0)
    parser.add_argument('--max-homology-dim', type=int, default=2)
    parser.add_argument('--output-dir', default='outputs')
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw, selected, x = load_data(args.csv, args.feature_mode, args.n_features)
    pd.DataFrame({'selected_features': list(selected.columns)}).to_csv(output_dir / 'selected_features.csv', index=False)
    try:
        latent, mse, history = train_autoencoder(x, args.latent_dim, args.epochs, args.batch_size)
        save_embedding(latent, output_dir / 'autoencoder_latent_space.csv', 'latent')
        plot_embedding(latent, 'Autoencoder Latent Space', output_dir / 'autoencoder_latent_space.png')
        pd.DataFrame(history).to_csv(output_dir / 'autoencoder_training_history.csv', index=False)
        pd.DataFrame([{'reconstruction_mse': mse}]).to_csv(output_dir / 'autoencoder_metrics.csv', index=False)
    except Exception as error:
        print(f'Autoencoder step skipped: {error}')
    pca_embedding, pca_variance = run_pca(x, args.latent_dim)
    save_embedding(pca_embedding, output_dir / 'pca_embedding.csv', 'pca')
    plot_embedding(pca_embedding, 'PCA Projection', output_dir / 'pca_projection.png')
    pd.DataFrame([pca_variance]).to_csv(output_dir / 'pca_explained_variance.csv', index=False)
    try:
        tsne_embedding = run_tsne(x, args.latent_dim)
        save_embedding(tsne_embedding, output_dir / 'tsne_embedding.csv', 'tsne')
        plot_embedding(tsne_embedding, 't-SNE Projection', output_dir / 'tsne_projection.png')
    except Exception as error:
        print(f't-SNE step skipped: {error}')
    try:
        intervals, summary = run_persistence(x, args.max_edge_length, args.max_homology_dim)
        save_persistence(intervals, summary, output_dir)
    except Exception as error:
        print(f'Persistent homology step skipped: {error}')
    try:
        mapper_summary = run_mapper(x, output_dir, args.latent_dim)
        pd.DataFrame([mapper_summary]).to_csv(output_dir / 'mapper_summary.csv', index=False)
    except Exception as error:
        print(f'Mapper step skipped: {error}')
    try:
        results = run_svr(raw)
        pd.DataFrame([results]).to_csv(output_dir / 'svr_results.csv', index=False)
        print('SVR results:', results)
    except Exception as error:
        print(f'SVR step skipped: {error}')


if __name__ == '__main__':
    main()
