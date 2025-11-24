# Use the official Jupyter image (Contains Python + Java + Spark)
FROM jupyter/pyspark-notebook:latest

WORKDIR /app

# Switch to root to install small libraries (kafka-python)
USER root

# Install dependencies (pyspark is NOT needed here, it's built-in!)
COPY requirements.txt .
# We add a timeout increase just in case, but it should be fast now
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy your scripts
COPY preprocessor.py .

# Fix permissions
RUN chown -R jovyan:users /app

# Switch back to default user
USER jovyan

# Keep container running
CMD ["tail", "-f", "/dev/null"]