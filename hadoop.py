import subprocess
import json

def check_hdfs_health():
    # Get filesystem statistics
    fs_stats = subprocess.check_output(
        "hdfs dfsadmin -report -json", 
        shell=True
    ).decode('utf-8')
    
    stats = json.loads(fs_stats)
    
    # Extract key metrics
    total_capacity = stats['StorageTypeStats']['DISK']['capacity'] / (1024**3)  # GB
    used_capacity = stats['StorageTypeStats']['DISK']['used'] / (1024**3)  # GB
    usage_percent = (used_capacity / total_capacity) * 100 if total_capacity > 0 else 0
    
    live_nodes = len(stats['LiveNodes'])
    dead_nodes = len(stats['DeadNodes'])
    total_blocks = stats['BlocksCount']
    
    # Get count of under-replicated blocks
    fsck_output = subprocess.check_output(
        "hdfs fsck / -blocks -files | grep 'Under replicated'",
        shell=True
    ).decode('utf-8')
    
    under_replicated = int(fsck_output.split(':')[1].strip())
    
    # Print report
    print(f"HDFS Cluster Health Report")
    print(f"=======================")
    print(f"Storage: {used_capacity:.2f}GB / {total_capacity:.2f}GB ({usage_percent:.2f}%)")
    print(f"Nodes: {live_nodes} live, {dead_nodes} dead")
    print(f"Total Blocks: {total_blocks}")
    print(f"Under-replicated Blocks: {under_replicated}")
    
    # Recommendations based on metrics
    if usage_percent > 80:
        print("\nWARNING: Storage utilization high, consider adding capacity")
    
    if under_replicated > 0:
        print(f"\nWARNING: {under_replicated} under-replicated blocks detected")
        print("Run: hdfs fsck / -under -files to identify affected files")
    
    if dead_nodes > 0:
        print(f"\nWARNING: {dead_nodes} dead nodes detected")
        print("Run: hdfs dfsadmin -refreshNodes to update node status")

if __name__ == "__main__":
    check_hdfs_health()
