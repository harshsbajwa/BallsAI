#!/usr/bin/env python3
"""Initialize database with historical data"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.database import create_tables, engine
from src.utils.loader import DataLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Initialize database with historical data"""
    try:
        logger.info("Creating database tables...")
        create_tables()
        
        logger.info("Loading historical data...")
        loader = DataLoader(data_path="./data")
        loader.load_initial_data()
        
        logger.info("Database initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
