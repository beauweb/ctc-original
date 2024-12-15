import sys
import logging
from app import app

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    app.run()
