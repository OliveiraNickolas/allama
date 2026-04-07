"""
Task Classifier: Auto-detect which profile to use based on user query
"""
import re
from typing import Optional, Tuple
from core.config import logger

class TaskClassifier:
    """Classify user tasks to optimal model profiles."""

    # Keywords mapping task types to profiles
    TASK_KEYWORDS = {
        "coding": {
            "keywords": ["code", "programming", "algorithm", "debug", "function", "class",
                        "variable", "loop", "condition", "regex", "compile", "python", "javascript",
                        "java", "golang", "rust", "cpp", "c++", "sql", "database"],
            "profile": "coding",
            "confidence_boost": 0.3
        },
        "ocr": {
            "keywords": ["extract text", "ocr", "apostila", "scan", "pdf", "document", "page",
                        "read text", "transcript", "handwriting", "recognize text", "convert to word",
                        "digitize", "transcribe"],
            "profile": "ocr",
            "confidence_boost": 0.4
        },
        "video": {
            "keywords": ["video", "keyframe", "frame", "analyze video", "video analysis",
                        "summarize video", "extract frames", "video summary", "motion", "scene",
                        "shot", "video content", "video understanding", "action recognition"],
            "profile": "video",
            "confidence_boost": 0.35
        },
        "comics": {
            "keywords": ["comic", "quadrinho", "manga", "graphic novel", "panel", "speech bubble",
                        "balloon", "comic panel", "comic text", "cartoon", "illustration text",
                        "extract dialogue", "comic dialogue"],
            "profile": "comics",
            "confidence_boost": 0.35
        },
        "research": {
            "keywords": ["research", "search", "mcp", "tool", "websearch", "web search",
                        "find", "lookup", "investigate", "analyze research", "scientific",
                        "study", "paper", "information gathering"],
            "profile": "research",
            "confidence_boost": 0.25
        }
    }

    # Image/vision keywords that boost vision profiles
    VISION_KEYWORDS = ["image", "picture", "photo", "visual", "see", "look at", "view", "show"]

    # Context keywords for different profiles
    CONTEXT_HINTS = {
        "coding": ["syntax", "error", "bug", "optimize", "refactor", "architecture"],
        "ocr": ["quality", "accuracy", "format", "layout", "spacing"],
        "video": ["duration", "fps", "frames", "chunk", "segment", "timeline"],
        "comics": ["text extraction", "dialogue", "narrative", "visual story"],
        "research": ["source", "verify", "fact check", "credible", "reference"]
    }

    @classmethod
    def classify(cls, query: str, filename: Optional[str] = None) -> Tuple[str, float]:
        """
        Classify a task and return (profile_name, confidence_score).

        Args:
            query: User query or task description
            filename: Optional filename that might hint at task type

        Returns:
            (profile_name, confidence) where confidence is 0.0-1.0
        """
        # Combine query with filename for better detection
        full_text = query.lower()
        if filename:
            full_text += " " + filename.lower()

        scores = {}

        # Score each task type
        for task_type, config in cls.TASK_KEYWORDS.items():
            score = cls._score_profile(full_text, config)
            scores[task_type] = score

        # Find best match
        best_profile = max(scores, key=scores.get)
        best_score = scores[best_profile]

        # If confidence is too low, check for vision keywords
        if best_score < 0.5 and any(kw in full_text for kw in cls.VISION_KEYWORDS):
            if filename and cls._is_image_file(filename):
                best_profile = "ocr"  # Default to OCR for images
                best_score = 0.6

        logger.info(f"🎯 Task Classification: {best_profile} (confidence: {best_score:.1%})")
        return best_profile, best_score

    @classmethod
    def _score_profile(cls, text: str, config: dict) -> float:
        """Score how well a profile matches the query."""
        score = 0.0

        # Count matching keywords
        matching_keywords = sum(1 for kw in config["keywords"] if kw in text)
        if matching_keywords > 0:
            score = min(matching_keywords * 0.15, 1.0)  # Cap at 1.0

        # Check context hints
        profile_name = config["profile"]
        context_hints = cls.CONTEXT_HINTS.get(profile_name, [])
        matching_hints = sum(1 for hint in context_hints if hint in text)
        score += matching_hints * 0.1

        # Boost for file extensions
        if profile_name == "video" and any(ext in text for ext in [".mp4", ".avi", ".mov"]):
            score += 0.3
        elif profile_name == "ocr" and any(ext in text for ext in [".pdf", ".jpg", ".png"]):
            score += 0.3

        return min(score, 1.0)

    @staticmethod
    def _is_image_file(filename: str) -> bool:
        """Check if filename is an image."""
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
        return any(filename.lower().endswith(ext) for ext in image_extensions)

    @classmethod
    def get_profile_for_file(cls, filename: str) -> Optional[str]:
        """Auto-detect profile from file extension."""
        filename_lower = filename.lower()

        if any(filename_lower.endswith(ext) for ext in [".pdf", ".jpg", ".png", ".tiff"]):
            return "ocr"
        elif any(filename_lower.endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv"]):
            return "video"
        elif any(filename_lower.endswith(ext) for ext in [".py", ".js", ".java", ".go", ".rs"]):
            return "coding"

        return None


def auto_detect_profile(query: str, filename: Optional[str] = None) -> str:
    """
    Convenience function: auto-detect profile and return it.
    """
    profile, confidence = TaskClassifier.classify(query, filename)

    if confidence < 0.3:
        logger.warning(f"⚠️  Low confidence profile selection ({confidence:.1%})")
        logger.info("ℹ️  Tip: Use --profile flag to override (--profile coding/ocr/video/research/comics)")

    return profile
