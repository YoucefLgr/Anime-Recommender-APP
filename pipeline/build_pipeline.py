from src.DataLoader import AnimeDataLoader
from src.vector_store import VectorStoreBuilder
from dotenv import load_dotenv
from utils.logger import get_logger
from utils.custom_exception import CustomException

load_dotenv()

logger = get_logger(__name__)

def main():
    try:
        logger.info("Starting to build pipeline")

        loader = AnimeDataLoader(original_csv="../data/anime_with_synopsis.csv", processed_csv="../data/anime_updated.csv")
        processed_data = loader.load_and_process()

        logger.info("Data loaded and Processed...")

        vector_builder = VectorStoreBuilder(csv_path=processed_data)
        vector_builder.build_and_save_vector_store()

        logger.info("Vector store built successfully !")

        logger.info("pipeline built successfully!")

    except Exception as e:
        logger.error("Failed Building the pipeling ")
        raise CustomException("Error during Building Pipeline")