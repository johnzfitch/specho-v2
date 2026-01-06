"""Utility functions for SpecHO watermark detection system.

This module provides essential utilities for file I/O, logging, and error handling
that are used throughout the SpecHO pipeline. These are foundational helpers that
enable the main detection components to focus on their core responsibilities.

Tier: 1 (MVP)
Task: 7.3
Dependencies: Task 1.1 (models.py)
"""

import logging
import json
from pathlib import Path
from typing import Optional, Callable, Any
from functools import wraps
from dataclasses import asdict

from models import DocumentAnalysis


# ============================================================================
# FILE I/O UTILITIES
# ============================================================================

def load_text_file(path: str, encoding: str = "utf-8") -> str:
    """Load text from a file with error handling.

    Reads the entire contents of a text file into a string. Handles common
    file errors gracefully and provides informative error messages.

    Args:
        path: Path to the text file to load
        encoding: Text encoding to use (default: 'utf-8')

    Returns:
        String containing the full file contents

    Raises:
        FileNotFoundError: If the file does not exist
        IOError: If the file cannot be read
        UnicodeDecodeError: If the file encoding is incorrect

    Examples:
        >>> text = load_text_file("sample.txt")
        >>> len(text) > 0
        True

        >>> text = load_text_file("data/corpus/article.txt", encoding="utf-8")
    """
    file_path = Path(path)

    # Validate file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")

    # Validate it's actually a file (not a directory)
    if not file_path.is_file():
        raise IOError(f"Path is not a file: {path}")

    # Read file contents
    try:
        with open(file_path, "r", encoding=encoding) as f:
            content = f.read()
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(
            e.encoding,
            e.object,
            e.start,
            e.end,
            f"Failed to decode file with {encoding} encoding: {path}"
        )
    except IOError as e:
        raise IOError(f"Failed to read file {path}: {e}")

    # Warn if file is empty
    if not content.strip():
        logging.warning(f"Loaded empty file: {path}")

    return content


def save_analysis_results(analysis: DocumentAnalysis, output_path: str, format: str = "json") -> None:
    """Save DocumentAnalysis results to a file.

    Serializes a DocumentAnalysis object to disk in the specified format.
    Creates parent directories if they don't exist.

    Args:
        analysis: DocumentAnalysis object to save
        output_path: Path where results should be written
        format: Output format ('json' or 'txt'). Default is 'json'.

    Raises:
        IOError: If the file cannot be written
        ValueError: If format is not supported

    Examples:
        >>> analysis = DocumentAnalysis(...)
        >>> save_analysis_results(analysis, "output/results.json")
        >>> save_analysis_results(analysis, "output/results.txt", format="txt")
    """
    output_file = Path(output_path)

    # Create parent directories if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Validate format
    if format not in ["json", "txt"]:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'txt'.")

    try:
        if format == "json":
            _save_as_json(analysis, output_file)
        elif format == "txt":
            _save_as_text(analysis, output_file)

        logging.info(f"Saved analysis results to: {output_path}")

    except IOError as e:
        raise IOError(f"Failed to save analysis results to {output_path}: {e}")


def _save_as_json(analysis: DocumentAnalysis, output_file: Path) -> None:
    """Internal helper to save analysis as JSON.

    Args:
        analysis: DocumentAnalysis object to serialize
        output_file: Path object for output file
    """
    # Convert dataclass to dictionary (recursively handles nested dataclasses)
    analysis_dict = asdict(analysis)

    # Write formatted JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(analysis_dict, f, indent=2, ensure_ascii=False)


def _save_as_text(analysis: DocumentAnalysis, output_file: Path) -> None:
    """Internal helper to save analysis as human-readable text.

    Args:
        analysis: DocumentAnalysis object to format
        output_file: Path object for output file
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("SpecHO Watermark Detection Analysis\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Document Length: {len(analysis.text)} characters\n")
        f.write(f"Clause Pairs Analyzed: {len(analysis.clause_pairs)}\n")
        f.write(f"Echo Scores Computed: {len(analysis.echo_scores)}\n\n")

        f.write("-" * 80 + "\n")
        f.write("RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Final Score:    {analysis.final_score:.4f}\n")
        f.write(f"Z-Score:        {analysis.z_score:.4f}\n")
        f.write(f"Confidence:     {analysis.confidence:.2%}\n\n")

        # Interpretation
        if analysis.z_score > 3.0:
            verdict = "HIGHLY LIKELY watermarked (z > 3.0)"
        elif analysis.z_score > 2.0:
            verdict = "LIKELY watermarked (z > 2.0)"
        elif analysis.z_score > 1.0:
            verdict = "POSSIBLY watermarked (z > 1.0)"
        else:
            verdict = "UNLIKELY watermarked (z <= 1.0)"

        f.write(f"Verdict: {verdict}\n\n")

        # Individual echo scores summary
        if analysis.echo_scores:
            f.write("-" * 80 + "\n")
            f.write("ECHO SCORES SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Pair':<6} {'Phonetic':<10} {'Structural':<12} {'Semantic':<10} {'Combined':<10}\n")
            f.write("-" * 80 + "\n")

            for idx, score in enumerate(analysis.echo_scores[:10]):  # Show first 10
                f.write(
                    f"{idx+1:<6} "
                    f"{score.phonetic_score:<10.3f} "
                    f"{score.structural_score:<12.3f} "
                    f"{score.semantic_score:<10.3f} "
                    f"{score.combined_score:<10.3f}\n"
                )

            if len(analysis.echo_scores) > 10:
                f.write(f"... ({len(analysis.echo_scores) - 10} more pairs)\n")

        f.write("\n" + "=" * 80 + "\n")


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

def setup_logging(level: str = "INFO", log_file: Optional[str] = None, format_style: str = "detailed") -> None:
    """Configure logging for the SpecHO system.

    Sets up Python's logging module with appropriate handlers, formatters,
    and log levels. Can log to console, file, or both.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional path to write logs to file (in addition to console)
        format_style: Log format style ('simple' or 'detailed')

    Raises:
        ValueError: If level is not a valid logging level

    Examples:
        >>> setup_logging("INFO")  # Console logging only
        >>> setup_logging("DEBUG", log_file="logs/debug.log")  # Console + file
        >>> setup_logging("WARNING", format_style="simple")  # Minimal format
    """
    # Validate and convert level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    if level.upper() not in level_map:
        valid_levels = ", ".join(level_map.keys())
        raise ValueError(f"Invalid log level: {level}. Use one of: {valid_levels}")

    log_level = level_map[level.upper()]

    # Choose format based on style
    if format_style == "simple":
        log_format = "%(levelname)s: %(message)s"
    else:  # detailed
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Remove existing handlers (avoid duplicates)
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    if format_style == "simple":
        console_formatter = logging.Formatter(log_format)
    else:
        console_formatter = logging.Formatter(log_format, datefmt=date_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        if format_style == "simple":
            file_formatter = logging.Formatter(log_format)
        else:
            file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        logging.info(f"Logging to file: {log_file}")

    logging.info(f"Logging configured at level: {level}")


# ============================================================================
# ERROR HANDLING DECORATORS
# ============================================================================

def handle_errors(default_return: Any = None, log_errors: bool = True) -> Callable:
    """Decorator to catch and handle exceptions in functions.

    Wraps a function with try-except logic that logs errors and returns
    a default value instead of propagating exceptions. Useful for making
    pipeline components more robust to errors.

    Args:
        default_return: Value to return if an exception occurs
        log_errors: Whether to log caught exceptions

    Returns:
        Decorator function

    Examples:
        >>> @handle_errors(default_return=[])
        ... def risky_function():
        ...     raise ValueError("Something broke")
        ...     return [1, 2, 3]
        >>> result = risky_function()
        >>> result
        []

        >>> @handle_errors(default_return=0.0, log_errors=True)
        ... def calculate_score(x):
        ...     return 1.0 / x
        >>> calculate_score(0)  # Would normally raise ZeroDivisionError
        0.0
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logging.error(
                        f"Error in {func.__name__}: {type(e).__name__}: {e}",
                        exc_info=True
                    )
                return default_return
        return wrapper
    return decorator


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)) -> Callable:
    """Decorator to retry a function on failure.

    Attempts to execute a function multiple times if it raises specified
    exceptions. Useful for operations that might fail due to transient issues
    (e.g., network requests, file locks).

    Args:
        max_attempts: Maximum number of attempts (including initial try)
        delay: Seconds to wait between attempts
        exceptions: Tuple of exception types to catch and retry

    Returns:
        Decorator function

    Examples:
        >>> @retry_on_failure(max_attempts=3, delay=0.1)
        ... def flaky_function():
        ...     import random
        ...     if random.random() < 0.7:
        ...         raise IOError("Temporary failure")
        ...     return "success"

        >>> @retry_on_failure(max_attempts=2, exceptions=(ValueError,))
        ... def validate_input(x):
        ...     if x < 0:
        ...         raise ValueError("Must be positive")
        ...     return x * 2
    """
    import time

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logging.warning(
                            f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        logging.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )

            # If we exhausted all attempts, raise the last exception
            raise last_exception

        return wrapper
    return decorator


def validate_input(**validators) -> Callable:
    """Decorator to validate function arguments.

    Checks that function arguments meet specified validation criteria before
    executing the function. Useful for ensuring data quality at component
    boundaries.

    Args:
        **validators: Keyword arguments mapping parameter names to validation functions
                     Validation functions should take the argument value and return bool

    Returns:
        Decorator function

    Raises:
        ValueError: If any validation fails

    Examples:
        >>> @validate_input(x=lambda x: x > 0, name=lambda n: isinstance(n, str))
        ... def process(x, name):
        ...     return f"{name}: {x}"

        >>> @validate_input(text=lambda t: len(t) > 0)
        ... def analyze(text):
        ...     return len(text.split())
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature to map positional args to names
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate each specified argument
            for param_name, validator_func in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator_func(value):
                        raise ValueError(
                            f"Validation failed for parameter '{param_name}' in {func.__name__}: "
                            f"value={value}"
                        )

            return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# HELPER UTILITIES
# ============================================================================

def ensure_directory(path: str) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists

    Returns:
        Path object for the directory

    Examples:
        >>> ensure_directory("data/output")
        PosixPath('data/output')
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_extension(path: str) -> str:
    """Get the file extension from a path.

    Args:
        path: File path

    Returns:
        Lowercase file extension without the dot (e.g., 'txt', 'json')

    Examples:
        >>> get_file_extension("data/sample.txt")
        'txt'
        >>> get_file_extension("results.JSON")
        'json'
    """
    return Path(path).suffix.lstrip(".").lower()
