# Use a base image with PyLucene installed
FROM pylucene/jcc:latest

# Set working directory
WORKDIR /app

# Copy your Python script to the container
COPY lucene_example.py /app/

# Create a directory for the index
RUN mkdir -p /app/lucene_index

# Make the container run the script when started
CMD ["python3", "lucene_example.py"]