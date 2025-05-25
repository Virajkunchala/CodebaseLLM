import os
from git import Repo
from threading import Lock
from config.config import logger, settings

class RepoCloner:
    """
    Class to handle cloning of a GitHub repository to a local directory.
    Optimized for scalability and speed using shallow clone and thread safety.
    """
    _lock = Lock()  # Class-level lock for thread safety

    def __init__(self, repo_url: str, target_dir: str):
        self.repo_url = repo_url
        self.target_dir = target_dir

    def clone(self) -> str:
        """
        Clone the repository if not already present.
        Returns the path to the cloned repository.
        Optimized for scalability and speed.
        """
        with RepoCloner._lock:
            if os.path.exists(self.target_dir):
                logger.info(f"Repository already exists at {self.target_dir}.")
                return self.target_dir
            logger.info(f"Cloning repository from {self.repo_url} into {self.target_dir} (shallow clone)...")
            Repo.clone_from(self.repo_url, self.target_dir, depth=1, multi_options=["--no-single-branch"])
            logger.info("Clone completed.")
            return self.target_dir

if __name__ == "__main__":
    REPO_URL = settings.REPO_URL
    TARGET_DIR = settings.TARGET_DIR
    if not REPO_URL or not TARGET_DIR:
        logger.error("REPO_URL and TARGET_DIR must be set in the .env file.")
        raise ValueError("REPO_URL and TARGET_DIR must be set in the .env file.")
    cloner = RepoCloner(REPO_URL, TARGET_DIR)
    cloner.clone()