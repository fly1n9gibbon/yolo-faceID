import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def generate_summary_report(df, output_folder):
    """Generates a text-based summary report of the analysis."""
    report_path = os.path.join(output_folder, "summary_report.txt")
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Face Recognition Performance Analysis Report\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        f.write("--- High-Level Summary by Model ---\n\n")
        
        model_summary = df.groupby('face_recognition_model_name').agg(
            total_images=('timestamp', 'count'),
            recognition_rate=('status', lambda x: (x == 'Recognized').sum() / len(x)),
            avg_confidence=('confidence', lambda x: x[df['status'] == 'Recognized'].mean())
        ).sort_values(by='avg_confidence', ascending=False)
        model_summary['recognition_rate'] = model_summary['recognition_rate'].map('{:.2%}'.format)
        model_summary['avg_confidence'] = model_summary['avg_confidence'].map('{:.4f}'.format)
        
        f.write(model_summary.to_string())
        f.write("\n\n" + "="*80 + "\n\n")
        f.write("--- Detailed Breakdown by Individual Run ---\n\n")
        
        for run_label in sorted(df['run_label'].unique()):
            run_df = df[df['run_label'] == run_label].copy()
            total_run_images = len(run_df)
            model_name = run_df['face_recognition_model_name'].iloc[0]
            f.write("-" * 60 + "\n")
            f.write(f"Analysis for Run: {run_label}\n")
            f.write(f"(Model: {model_name})\n")
            f.write("-" * 60 + "\n")
            rec_rate = (run_df['status'] == 'Recognized').sum() / total_run_images if total_run_images > 0 else 0
            avg_confidence = run_df[run_df['status'] == 'Recognized']['confidence'].mean()
            f.write(f"  - Total Images: {total_run_images}\n")
            f.write(f"  - Recognition Rate: {rec_rate:.2%}\n")
            f.write(f"  - Avg. Confidence (Recognized only): {avg_confidence:.4f}\n\n")
            f.write("  Breakdown by Person:\n")
            person_summary = run_df[run_df['status'] == 'Recognized'].groupby('recognized_person')['confidence'].agg(['count', 'mean']).sort_values(by='count', ascending=False)
            if not person_summary.empty:
                f.write(person_summary.to_string())
            else:
                f.write("    No individuals were recognized in this run.")
            f.write("\n\n")
    print(f"✅ Summary report saved to: {report_path}")

def plot_status_comparison(df, output_folder):
    """Plots a bar chart comparing recognition statuses across runs."""
    plt.figure(figsize=(14, 8))
    sns.countplot(data=df, x='run_label', hue='status', order=sorted(df['run_label'].unique()))
    plt.title('Comparison of Recognition Status Across Runs', fontsize=16)
    plt.ylabel('Number of Images')
    plt.xlabel('Run (Model_Timestamp)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Status')
    plt.tight_layout()
    plot_path = os.path.join(output_folder, "plot_status_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Status comparison plot saved to: {plot_path}")

def plot_confidence_distribution(df, output_folder):
    """Plots a boxplot of confidence scores for 'Recognized' images."""
    recognized_df = df[df['status'] == 'Recognized'].copy()
    if recognized_df.empty:
        print("⚠️ Skipping confidence distribution plot: No 'Recognized' images found.")
        return
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=recognized_df, x='run_label', y='confidence', order=sorted(recognized_df['run_label'].unique()))
    plt.title('Distribution of Confidence Scores Across Runs', fontsize=16)
    plt.ylabel('Confidence Score')
    plt.xlabel('Run (Model_Timestamp)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plot_path = os.path.join(output_folder, "plot_confidence_distribution.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Confidence distribution plot saved to: {plot_path}")

def main(args):
    print(f"Starting analysis of log files in: {args.input_folder}")
    log_files = glob.glob(os.path.join(args.input_folder, "**", "*.csv"), recursive=True)
    if not log_files:
        print(f"❌ Error: No CSV files found in '{args.input_folder}' or its subdirectories.")
        return

    all_logs = []
    for log_file in log_files:
        try:
            df = pd.read_csv(log_file)
            if 'face_recognition_model_name' not in df.columns:
                print(f"⚠️ Warning: Skipping '{log_file}' as it's an old log format without 'face_recognition_model_name'.")
                continue
            
            # Extract timestamp from the parent folder's name
            folder_name = os.path.basename(os.path.dirname(log_file))
            try:
                run_id = folder_name.split('_')[-1].replace('-results', '')
            except IndexError:
                run_id = "unknown_run"
            
            df['run_id'] = run_id
            df['run_label'] = df['face_recognition_model_name'] + '_' + df['run_id']
            all_logs.append(df)
        except Exception as e:
            print(f"⚠️ Warning: Could not process '{log_file}'. Error: {e}")
            
    if not all_logs:
        print("❌ Error: Failed to load any valid log files with model name data.")
        return

    master_df = pd.concat(all_logs, ignore_index=True)
    print(f"Successfully loaded and combined {len(all_logs)} log files.")
    
    output_dir = os.path.join(args.input_folder, "analysis_results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Analysis results will be saved in: {output_dir}\n")
    
    generate_summary_report(master_df, output_dir)
    plot_status_comparison(master_df, output_dir)
    plot_confidence_distribution(master_df, output_dir)
    
    print("\n✅ Analysis complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze face recognition log files from multiple models and runs.")
    parser.add_argument("-i", "--input-folder", required=True, help="Path to the top-level folder (e.g., 'recognition_runs') containing the result directories.")
    args = parser.parse_args()
    main(args)