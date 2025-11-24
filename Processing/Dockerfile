# Use the official Jupyter image (Contains Python + Java + Spark)
FROM jupyter/pyspark-notebook:latest

WORKDIR /app

# Switch to root to install small libraries (pandas, etc.)
USER root

# Install dependencies from requirements.txt
COPY requirements.txt .
# We add a timeout increase just in case, but it should be fast now
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy your scripts
COPY preprocessor.py .
COPY start.sh .

# Fix permissions
RUN chown -R jovyan:users /app
RUN chmod +x start.sh

# Switch back to default user
USER jovyan

# Keep container running
CMD ["./start.sh"]