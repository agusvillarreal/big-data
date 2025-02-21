# Spark JupyterLab Docker Setup

This setup provides a Docker environment with Apache Spark (Master + Worker) and JupyterLab integration.

## Prerequisites
- Docker
- Docker Compose

## Setup Instructions

1. Create project structure:
   ```bash
   mkdir spark-jupyter-docker
   cd spark-jupyter-docker
   mkdir workspace
   ```

2. Place all the configuration files in the project root directory.

3. Build and start the containers:
   ```bash
   docker-compose up --build
   ```

4. Access the UIs:
   - JupyterLab: http://localhost:8888 (token: token-uag)
   - Spark Master UI: http://localhost:8080
   - Spark Worker UI: http://localhost:8081
   - Spark Application UI: http://localhost:4040 (when running a Spark job)

## Usage Example

```python
from pyspark.sql import SparkSession

# Initialize Spark session with cluster master
spark = SparkSession.builder \
    .appName("SparkExample") \
    .master("spark://spark-master:7077") \
    .config("spark.driver.memory", "1g") \
    .config("spark.executor.memory", "1g") \
    .getOrCreate()

# Create a sample DataFrame
data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
df = spark.createDataFrame(data, ["name", "age"])

# Show the DataFrame
df.show()
```

## Configuration

- Spark Master UI: http://localhost:8080
- Spark Worker UI: http://localhost:8081
- Spark Application UI: http://localhost:4040
- Jupyter token: token-uag
- Workspace directory is mounted at `/workspace` in the container
- Default Spark configuration:
  - Driver Memory: 1GB
  - Executor Memory: 1GB
  - Worker Cores: 1
  - Worker Memory: 1GB

## Note
This configuration uses:
- Apache Spark 3.5.4 with Python 3 support (scala2.12-java11-python3-ubuntu)
- JupyterLab instead of Jupyter Notebook for a more modern interface