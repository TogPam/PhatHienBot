import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
import warnings

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

# ==================================================
# 0. LOAD DATA
# ==================================================
df = pd.read_csv('data/mastodon_features_scaled.csv')
features = ['follow_ratio', 'avg_time_diff_minutes', 'engagement_rate', 'night_post_ratio', 'lexical_diversity']

# Initialize columns
df['hybrid_cluster'] = 0      # 0 = normal, -1 = outlier
df['detection_source'] = ''   # 'rule' | 'dbscan' | 'iso' | 'consensus'
df['outlier_reason'] = ''

# ==================================================
# 1. RULE-BASED LAYER (Pre-filter heuristics)
# ==================================================
rule_mask = pd.Series(False, index=df.index)

# Rule 1: API Declared Bot
mask1 = df['is_bot_declared'] == 1
df.loc[mask1, 'outlier_reason'] += 'API Declared Bot | '
rule_mask |= mask1

# Rule 2: "bot" in username (case-insensitive)
mask2 = df['username'].str.contains('bot', case=False, na=False)
df.loc[mask2, 'outlier_reason'] += 'Bot in Username | '
rule_mask |= mask2

# Rule 3: Extreme repetitive content
# Note: lexical_diversity in features CSV is scaled; check raw threshold via unscaled CSV
# We use unscaled features_csv instead, load raw scaled - column may be z-scored.
# To apply raw thresholds, we load the unscaled mastodon_features_scaled is already scaled.
# However, feature_engineering also saves username+is_bot_declared alongside scaled values.
# We rely on the scale: lexical_diversity mean~0.5, std~0.3 -> raw<0.1 ~ scaled < (0.1-0.5)/0.3 = -1.33
mask3 = df['lexical_diversity'] < -1.33
df.loc[mask3, 'outlier_reason'] += 'Extreme Repetitive Content | '
rule_mask |= mask3

# Rule 4: Inhuman posting speed
# avg_time_diff_minutes mean~60, std~200 -> raw<0.5 ~ scaled < (0.5-60)/200 = -0.298
mask4 = df['avg_time_diff_minutes'] < -0.298
df.loc[mask4, 'outlier_reason'] += 'Inhuman Posting Speed | '
rule_mask |= mask4

# Apply rule-based outlier labels
df.loc[rule_mask, 'hybrid_cluster'] = -1
df.loc[rule_mask, 'detection_source'] = 'rule'

rule_count = rule_mask.sum()
print(f"[Rule-Based Layer] Pre-filtered {rule_count} accounts as outliers.")

# ==================================================
# 2. ML LAYER — applied ONLY on non-rule-flagged users
# ==================================================
normal_mask = ~rule_mask
df_ml = df[normal_mask].copy()
X_ml = df_ml[features].values

if len(df_ml) > 0:
    # --- KMeans (for cluster profiling on the clean subset) ---
    optimal_k = 3
    wcss = []
    for i in range(2, 9):
        km_test = KMeans(n_clusters=i, init='k-means++', random_state=42)
        km_test.fit(X_ml)
        wcss.append(km_test.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(range(2, 9), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method')
    plt.xlabel('K')
    plt.ylabel('WCSS')
    plt.savefig('docs/elbow_method.png')
    plt.close()
    print("1. Elbow chart saved to 'docs/elbow_method.png'.")

    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
    ml_kmeans_labels = kmeans.fit_predict(X_ml)
    df_ml['kmeans_cluster'] = ml_kmeans_labels

    score = silhouette_score(X_ml, ml_kmeans_labels)
    print(f"2. K-Means (K={optimal_k}) Silhouette Score (on clean subset): {score:.4f}")

    # --- DBSCAN ---
    dbscan = DBSCAN(eps=0.8, min_samples=2)
    dbscan_labels = dbscan.fit_predict(X_ml)
    df_ml['dbscan_cluster'] = dbscan_labels
    dbscan_outlier_mask = dbscan_labels == -1
    df_ml.loc[dbscan_outlier_mask, 'hybrid_cluster'] = -1
    df_ml.loc[dbscan_outlier_mask, 'detection_source'] = 'dbscan'
    df_ml.loc[dbscan_outlier_mask, 'outlier_reason'] += 'Density Outlier (DBSCAN) | '

    # --- Isolation Forest ---
    iso = IsolationForest(contamination=0.1, random_state=42)
    iso_labels = iso.fit_predict(X_ml)
    df_ml['iso_outlier'] = iso_labels
    iso_outlier_mask = iso_labels == -1

    # Consensus: flagged by BOTH DBSCAN and Isolation Forest
    consensus_mask = dbscan_outlier_mask & iso_outlier_mask
    df_ml.loc[consensus_mask, 'detection_source'] = 'consensus'

    # Iso-only (not already DBSCAN-flagged)
    iso_only_mask = iso_outlier_mask & ~dbscan_outlier_mask
    df_ml.loc[iso_only_mask, 'hybrid_cluster'] = -1
    df_ml.loc[iso_only_mask, 'detection_source'] = 'iso'
    df_ml.loc[iso_only_mask, 'outlier_reason'] += 'Statistical Outlier (Isolation Forest) | '

    dbscan_count = dbscan_outlier_mask.sum()
    iso_count = iso_outlier_mask.sum()
    consensus_count = consensus_mask.sum()
    print(f"3. DBSCAN detected: {dbscan_count} anomalies in clean subset.")
    print(f"4. Isolation Forest detected: {iso_count} anomalies in clean subset.")
    print(f"5. Consensus (both): {consensus_count} anomalies.")

    # Write ml results back to df
    df.loc[normal_mask, 'kmeans_cluster'] = df_ml['kmeans_cluster'].values
    df.loc[normal_mask, 'dbscan_cluster'] = df_ml['dbscan_cluster'].values
    df.loc[normal_mask, 'iso_outlier'] = df_ml['iso_outlier'].values
    df.loc[normal_mask, 'hybrid_cluster'] = df_ml['hybrid_cluster'].values
    df.loc[normal_mask, 'detection_source'] = df_ml['detection_source'].values
    df.loc[normal_mask, 'outlier_reason'] = df_ml['outlier_reason'].values
else:
    print("[WARNING] No users remain after rule-based filtering for ML layer.")

# Clean up trailing separators in reason strings
df['outlier_reason'] = df['outlier_reason'].str.strip(' |').str.strip()

# ==================================================
# 3. MERGE & SAVE
# ==================================================
total_outliers = (df['hybrid_cluster'] == -1).sum()
print(f"\n{'='*50}")
print("🏆 KẾT QUẢ HYBRID DETECTION SYSTEM")
print(f"{'='*50}")
print(f"Rule-Based flagged:  {rule_count} accounts")
print(f"ML-Based flagged:    {total_outliers - rule_count} additional accounts")
print(f"Total Bot/Outliers:  {total_outliers} / {len(df)} accounts")
print(f"{'='*50}")

df.to_csv('data/mastodon_clustered_results.csv', index=False)
print("6. Results saved to 'data/mastodon_clustered_results.csv'.")

# Export blacklist
blacklist_cols = ['username', 'follow_ratio', 'avg_time_diff_minutes', 'engagement_rate',
                  'night_post_ratio', 'lexical_diversity', 'detection_source', 'outlier_reason']
available_cols = [c for c in blacklist_cols if c in df.columns]
blacklist_df = df[df['hybrid_cluster'] == -1][available_cols]
blacklist_df.to_csv('data/bot_blacklist.csv', index=False)
print(f"7. Blacklist of {len(blacklist_df)} accounts saved to 'data/bot_blacklist.csv'.")
