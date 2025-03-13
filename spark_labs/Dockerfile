FROM apache/spark:3.5.4-scala2.12-java11-python3-ubuntu


USER root

# Install necessary packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install JupyterLab and PySpark
RUN pip3 install --no-cache-dir \
    jupyterlab \
    pyspark==3.5.4 \
    pandas \
    numpy \
    matplotlib \
    panel \
    watchfiles

# Create working directory with permissive permissions
WORKDIR /workspace
RUN mkdir -p /workspace && chmod 777 /workspace

# Create a non-root user
RUN useradd -m -s /bin/bash jupyter

# Make sure all needed directories have appropriate permissions
RUN mkdir -p /tmp/spark-events && chmod 777 /tmp/spark-events
RUN mkdir -p /opt/spark/work && chmod 777 /opt/spark/work
RUN mkdir -p /opt/spark/logs && chmod 777 /opt/spark/logs

# Switch to non-root user
USER jupyter

# Expose Jupyter port
EXPOSE 8888

# Start JupyterLab with custom token
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=token-uag"]