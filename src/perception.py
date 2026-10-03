from abc import ABC, abstractmethod
from pathlib import Path

class PerceptionGenerationError(ValueError):
    def __init__(self, message: str, replies: list[str], errors: list[str]) -> None:
        """Preserve model replies and validation errors when perception fails"""
        super().__init__(message)
        self.replies = replies
        self.errors = errors


class PerceptionModel(ABC):
    @abstractmethod
    def generate_scene_graph(
        self,
        image_path: Path,
        width: int,
        height: int,
    ) -> tuple[dict, list[str]]:
        """Run perception and return a validated scene graph plus diagnostic replies

        Use the original image's width and height for all pixel coordinates
        Validate the graph with validate_scene_graph before returning it
        Return an empty replies list when the backend produces no text responses
        Raise ValueError if a valid scene graph cannot be produced
        """
        # Loading, inference, and graph construction are specific to each backend
        raise NotImplementedError
