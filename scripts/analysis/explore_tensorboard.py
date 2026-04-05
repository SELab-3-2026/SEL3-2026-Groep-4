#!/usr/bin/env python3
"""
Reproducible CLI tool to explore TensorBoard logs.
Designed for both local development and HPC diagnostics.

Requirements:
    pip install tensorboard

Usage:
    python explore_tensorboard.py <path_to_run_directory> [--csv output.csv]
"""

import argparse
import os
import sys
import csv

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("Error: Missing dependency. Please run: pip install tensorboard")
    sys.exit(1)

def explore_run(log_dir):
    """
    Extracts and displays a summary of scalar metrics from a TensorBoard log directory.
    """
    print(f"\n{'='*20} Exploring Run {'='*20}")
    print(f"Directory: {log_dir}")
    print(f"{'='*55}\n")

    if not os.path.exists(log_dir):
        print(f"Error: Directory '{log_dir}' does not exist.")
        return None

    # Initialize EventAccumulator
    # size_guidance=0 loads all data points for each tag.
    ea = event_accumulator.EventAccumulator(log_dir, size_guidance={
        event_accumulator.SCALARS: 0,
        event_accumulator.TENSORS: 0,
    })
    
    print("Loading event files (this may take a moment for large runs)...")
    ea.Reload()

    tags = ea.Tags()
    scalar_tags = tags.get('scalars', [])
    
    if not scalar_tags:
        print("No scalar metrics found in this directory.")
        return None

    print(f"Found {len(scalar_tags)} scalar metrics.\n")

    data = {}
    summary = []

    # Process scalar values
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        if not events:
            continue
        
        values = [e.value for e in events]
        last_event = events[-1]
        data[tag] = values
        
        summary.append({
            "Metric": tag,
            "Steps": len(events),
            "Last Value": f"{last_event.value:.4f}",
            "Max": f"{max(values):.4f}",
            "Min": f"{min(values):.4f}"
        })

    # Display summary table formatted manually
    summary = sorted(summary, key=lambda x: x['Metric'])
    print(f"{'Metric':<30} {'Steps':>10} {'Last':>12} {'Max':>12} {'Min':>12}")
    print("-" * 80)
    for row in summary:
        print(f"{row['Metric']:<30} {row['Steps']:>10} {row['Last Value']:>12} {row['Max']:>12} {row['Min']:>12}")

    # Calculate and display global metadata
    if 'charts/SPS' in data:
        sps_events = ea.Scalars('charts/SPS')
        if len(sps_events) > 1:
            total_duration_hours = (sps_events[-1].wall_time - sps_events[0].wall_time) / 3600
            print(f"\nTotal Recorded Duration: {total_duration_hours:.2f} hours")
            
            # Estimate completion if total_timesteps is available in hyperparameters
            try:
                hp_tags = [t for t in tags.get('tensors', []) if 'hyperparameters' in t]
                if hp_tags:
                    hp_event = ea.Tensors(hp_tags[0])[0]
                    hp_text = hp_event.tensor_proto.string_val[0].decode('utf-8')
                    if 'total_timesteps' in hp_text:
                        for line in hp_text.split('\n'):
                            if 'total_timesteps' in line:
                                target = int(line.split('|')[2].strip())
                                current = ea.Scalars(scalar_tags[0])[-1].step
                                percent = (current / target) * 100
                                print(f"Progress: {current:,} / {target:,} steps ({percent:.1f}%)")
            except Exception:
                pass

    return data

def main():
    parser = argparse.ArgumentParser(description="Clean, reproducible TensorBoard exploration tool.")
    parser.add_argument("log_dir", help="Path to the TensorBoard run directory.")
    parser.add_argument("--csv", help="Optional: Path to save all scalar data as a CSV.", default=None)
    
    args = parser.parse_args()
    
    scalar_data = explore_run(args.log_dir)
    
    if args.csv and scalar_data:
        # Reloading for wall_time and steps
        ea = event_accumulator.EventAccumulator(args.log_dir).Reload()
        with open(args.csv, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["tag", "step", "value", "wall_time"])
            writer.writeheader()
            for tag in scalar_data.keys():
                for e in ea.Scalars(tag):
                    writer.writerow({
                        "tag": tag, 
                        "step": e.step, 
                        "value": e.value, 
                        "wall_time": e.wall_time
                    })
        
        print(f"\nData exported to: {args.csv}")

if __name__ == "__main__":
    main()
