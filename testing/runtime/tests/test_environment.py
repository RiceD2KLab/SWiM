"""
This script checks the resources available on the system where the tests are run.
It checks the number of CPU cores, the total and available memory, and the speed of the cpu core.

Usage:
    python test_environment.py
"""

import os
import psutil
from loguru import logger


def test_resources():
    """
    Test the resources available on the system.
    """
    # Get the number of CPU cores
    cpu_count = psutil.cpu_count(logical=False)

    # Get memory information
    memory_info = psutil.virtual_memory()

    # Get CPU frequency (in MHz)
    cpu_frequency = psutil.cpu_freq()

    logger.info(f"Number of CPU cores available: {cpu_count}")
    logger.info(f"Total memory available: {memory_info.total / (1024 ** 2)} MB")
    logger.info(f"Available memory: {memory_info.available / (1024 ** 2)} MB")

    # Check if the CPU frequency is available
    if cpu_frequency:
        logger.info(f"CPU speed: {cpu_frequency.current:.2f} MHz")
    else:
        logger.info("CPU frequency information is not available.")


def main():
    """
    Main function to test the resources available on the system
    """
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Contents in the WORKDIR: {os.listdir('/code_execution')}")
    logger.info("Testing the resources available on the system.")

    test_resources()


if __name__ == "__main__":
    main()
