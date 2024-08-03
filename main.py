# main.py
import logging
from src.data_loader import load_data
from src.clustering import perform_clustering, calculate_silhouette, calculate_wcss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info('Starting main function')
    try:
        df = load_data('src/dataset/mall_customers.csv')
        
        # Clustering
        df, kmeans = perform_clustering(df, n_clusters=5)
        
        # Visualization
        #visualization.plot_clusters(df)
        
        # WCSS and Silhouette Analysis
        k_range = range(3, 9)
        WCSS = calculate_wcss(df, k_range)
        #visualization.plot_wcss(k_range, WCSS)
        
        ss = calculate_silhouette(df, k_range)
        #visualization.plot_silhouette(k_range, ss)
       
    except Exception as e:
        logging.error(f'Error occurred: {e}')

if __name__ == '__main__':
    main()
