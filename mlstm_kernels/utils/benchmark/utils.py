from pathlib import Path


def setup_output_folder(output_dir: str = "./outputs_kernel_benchmarks") -> Path:
    import logging
    import sys
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_folder = Path(output_dir) / timestamp

    output_folder.mkdir(parents=True, exist_ok=False)

    logfile = output_folder / "benchmark.log"
    file_handler = logging.FileHandler(filename=logfile)
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        handlers=[file_handler, stdout_handler],
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        force=True,
    )
    LOGGER = logging.getLogger(__name__)
    LOGGER.info(f"Logging to {logfile}")
    return output_folder
