# clustering.py
from sklearn.cluster import KMeans
import logging
from sklearn.metrics import silhouette_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def perform_clustering(df, n_clusters):
    try:
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++').fit(df[['Annual_Income', 'Spending_Score']])
        df['Cluster'] = kmeans.labels_
        logging.info(f'Clustering performed with {n_clusters} clusters')
        return df, kmeans
    except Exception as e:
        logging.error(f'Error performing clustering: {e}')
        raise

def calculate_wcss(df, k_range):
    try:
        WCSS = []
        for k in k_range:
            kmodel = KMeans(n_clusters=k).fit(df[['Annual_Income', 'Spending_Score']])
            WCSS.append(kmodel.inertia_)
        logging.info(f'WCSS calculation completed successfully: {WCSS}')
        return WCSS
    except Exception as e:
        logging.error(f'Error calculating WCSS: {e}')
        raise

def calculate_silhouette(df, k_range):
    try:
        ss = []
        for k in k_range:
            kmodel = KMeans(n_clusters=k).fit(df[['Annual_Income', 'Spending_Score']])
            ypred = kmodel.labels_
            sil_score = silhouette_score(df[['Annual_Income', 'Spending_Score']], ypred)
            ss.append(sil_score)
        logging.info(f'Silhouette score calculation completed successfully: {ss}')
        return ss
    except Exception as e:
        logging.error(f'Error calculating silhouette scores: {e}')
        raise
