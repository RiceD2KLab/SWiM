# Use python:3.11-slim-bookworm as the base image
FROM python:3.11-bookworm

# Switch to the root user
USER root

# Update package list and install required packages
RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils libglib2.0-0 libgl1 && \
    apt-get clean 

# Copy requirements.txt to the container
COPY requirements.txt /tmp/requirements.txt

# Install Python packages from requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# Set the working directory
WORKDIR /code_execution

# Copy all test Python files to the container
COPY tests/*.py /code_execution/tests/

# Copy the benchmark Python file to the container
COPY benchmark.py /code_execution/benchmark.py
COPY run_benchmark.sh /code_execution/run_benchmark.sh

# Run tests to verify the environment
RUN python3 /code_execution/tests/test_packages.py \
    && python3 /code_execution/tests/test_environment.py

# Command to execute when running the container
CMD ["/bin/bash", "/code_execution/run_benchmark.sh"]

