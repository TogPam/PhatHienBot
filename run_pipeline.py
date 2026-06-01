import os
import shutil
import subprocess
import sys

# Reconfigure stdout to use UTF-8 to prevent UnicodeEncodeError on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def move_file(filename, target_dir):
    if os.path.exists(filename) and os.path.isfile(filename):
        os.makedirs(target_dir, exist_ok=True)
        dest = os.path.join(target_dir, filename)
        if os.path.exists(dest):
            os.remove(dest)
        shutil.move(filename, target_dir)
        print(f"Moved {filename} to {target_dir}/")

def main():
    print("=== STEP 1: Reorganizing existing files ===")
    data_files = [
        'bot_blacklist.csv',
        'mastodon_clientcred.secret',
        'mastodon_clustered_results.csv',
        'mastodon_dataset.json',
        'mastodon_features_scaled.csv',
        'mastodon_tok.txt',
        'mastodon_usercred.secret'
    ]
    doc_files = [
        'elbow_method.png',
        'cluster_visualization_2d.png'
    ]
    
    for f in data_files:
        move_file(f, 'data')
    for f in doc_files:
        move_file(f, 'docs')
        
    print("\n=== STEP 2: Running Pipeline Sequentially ===")
    
    python_exe = sys.executable
    
    # We will pass PYTHONUTF8=1 to sub-processes so they encode output in UTF-8
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    
    pipeline = [
        ('Feature Engineering', [python_exe, 'src/feature_engineering.py']),
        ('Clustering Model', [python_exe, 'src/clustering_model.py']),
        ('Cluster Analysis', [python_exe, 'src/cluster_analysis.py'])
    ]
    
    for name, cmd in pipeline:
        print(f"\n--- Running {name} ---")
        res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=env)
        if res.returncode != 0:
            print(f"[ERROR] Failed running {name}:")
            print(res.stderr)
            sys.exit(res.returncode)
        else:
            print(f"[SUCCESS] Finished {name} successfully.")
            print(res.stdout)

    print("\n[SUCCESS] Pipeline executed successfully!")

if __name__ == '__main__':
    main()
