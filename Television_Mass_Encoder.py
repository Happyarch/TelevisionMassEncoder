import os
import subprocess
import random
import time
import fcntl
import shutil
import argparse
import shlex
import multiprocessing
import queue
import signal
import sys
import logging


FFMPEG_PATH = "ffmpeg"
MAX_RETRIES = 10
NUM_WORKERS = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed AV1 encoder client with locking."
    )
    parser.add_argument(
        "--source-dir", type=str, required=True, help="Path to input videos directory"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Path to output videos directory"
    )
    parser.add_argument(
        "--ffmpeg-flags", type=str, required=True, help="Flags to pass to ffmpeg"
    )
    # --- REFACTORED: Changed --debug_enable to behave as a simple boolean and removed the str2bool function ---
    parser.add_argument(
        "--debug_enable",
        action="store_true",
        help="Enable debug messaging",
    )
    # --- REFACTORED: Added --use-ramdisk argument ---
    parser.add_argument(
        "--use-ramdisk",
        action="store_true",
        help="Copy source file to /tmp for potentially faster processing.",
    )
    parser.add_argument(
        "--output-extension",
        type=str,
        help="Specify the file extension of the output/its container. Default: .mkv",
    )
    parser.add_argument(
        "--input-extensions",
        type=str,
        help=(
            "Colon-separated list of file extensions to process (without leading dots). "
            "Example: mp4:mkv:avi. Defaults to all common FFmpeg-supported video/audio formats."
        ),
    )
    return parser.parse_args()


def detach():
    """Detach from shell but keep stdout/stderr, ignoring SIGHUP."""
    # Ignore hangup signal (shell closing)
    try:
        os.setsid()
    except Exception:
        pass  # Not critical if already in a session

    # Double-fork to prevent a defunct process (zombie)
    try:
        if os.fork() > 0:
            os._exit(0)  # Exit the first child
    except OSError as e:
        print(f"Failed to double-fork: {e}")
        os._exit(1)

    # Close standard file descriptors to completely detach
    try:
        os.close(0)
        os.close(1)
        os.close(2)
    except OSError:
        pass  # Ignore if they're already closed

    # Ignore hangup signal again after setsid() and fork()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)


def get_media_files(source_dir, input_exts=None):
    """
    Return shuffled list of media files in source_dir matching input_exts.

    input_exts: colon-separated string of extensions without leading dot, e.g. "mp4:mkv"
                If None, defaults to common FFmpeg-supported formats.
    """
    # Default extensions
    default_exts = (
        ".mkv",
        ".mp4",
        ".mov",
        ".avi",
        ".webm",
        ".divx",
        ".vob",
        ".evo",
        ".ogv",
        ".ogx",
        ".flv",
        ".f4v",
        ".aac",
        ".flac",
        ".mp3",
        ".ogg",
        ".opus",
        ".alac",
        ".mka",
        ".pcm",
        ".aiff",
        ".wav",
        ".cda",
        ".ape",
    )

    if input_exts:
        # Split on colon, strip whitespace, and ensure each starts with "."
        exts_to_use = tuple(
            f".{ext.strip().lstrip('.')}"
            for ext in input_exts.split(":")
            if ext.strip()
        )
    else:
        exts_to_use = default_exts

    files = [f for f in os.listdir(source_dir) if f.lower().endswith(exts_to_use)]
    random.shuffle(files)
    return files


# --- REFACTORED: Removed unused lock_file() function ---


def get_output_path(output_dir, file_name, output_extension):
    if not output_extension:
        output_extension = ".mkv"
    output_dir_tmp = os.path.join(output_dir, "tmp")
    os.makedirs(output_dir_tmp, exist_ok=True)
    return os.path.join(
        output_dir_tmp, os.path.splitext(file_name)[0] + output_extension
    )


def setup_logging(main_pid=None, is_main=False):
    """Setup logging for this Python process (one log file per process)."""
    pid = os.getpid()

    if is_main:
        log_file = f"main_{pid}.log"
    elif main_pid:
        log_dir = f"{main_pid}_worker_logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"transcode_{main_pid}_{pid}.log")
    else:
        raise ValueError("Must provide either is_main=True or main_pid for worker.")

    logger = logging.getLogger(f"transcoder_{pid}")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # File handler (always active)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(logging.INFO)
        fh_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

        # Stream handler (active for main process and for a brief time in workers)
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        sh.setFormatter(sh_formatter)
        if not is_main:
            sh.set_name("temporary_stream_handler")  # Set a name to find it later
        logger.addHandler(sh)

    return logger


# --- REFACTORED: Changed ram_disk parameter to use_ramdisk ---
def process_file(
    file_name,
    source_dir,
    output_dir,
    output_extension,
    ffmpeg_flags,
    use_ramdisk,
    logger,
    msg_queue,
    detach_event,  # Added detach_event parameter
):
    input_path = os.path.join(source_dir, file_name)
    working_input_path = input_path  # default

    if use_ramdisk:
        tmp_path = os.path.join("/tmp", file_name)
        logger.info(f"Copying {file_name} to /tmp/ for faster encoding...")
        shutil.copy2(input_path, tmp_path)
        working_input_path = tmp_path

    output_path = get_output_path(output_dir, file_name, output_extension)

    # ffmpeg_flags is a single string, so split it safely into list
    flags_list = shlex.split(ffmpeg_flags)

    cmd = [FFMPEG_PATH, "-i", working_input_path] + flags_list + [output_path]

    # --- Conditional output based on detach_event ---
    if detach_event.is_set():
        # Once detached, capture output for logging
        stdout_dest = subprocess.PIPE
        stderr_dest = subprocess.PIPE
    else:
        # Before detachment, send output directly to the terminal
        stdout_dest = None
        stderr_dest = None

    logger.info(f"[{os.uname().nodename}] Encoding: {file_name}")
    logger.info(f"[DEBUG] Command: {' '.join(cmd)}")  # Debug: show full command

    try:
        result = subprocess.run(
            cmd, check=True, stdout=stdout_dest, stderr=stderr_dest, text=True
        )
        logger.info(f"[{os.uname().nodename}] Finished: {file_name}")
        msg_queue.put(("info", os.getpid(), file_name, "completed"))

        if detach_event.is_set():
            if result.stdout:
                logger.info("FFmpeg STDOUT:\n" + result.stdout)
            if result.stderr:
                logger.error("FFmpeg STDERR:\n" + result.stderr)

    except FileNotFoundError:
        logger.error(f"[{os.uname().nodename}] FFmpeg not found. Is it in your PATH?")
        msg_queue.put(("error", os.getpid(), file_name, "ffmpeg not found"))
        return False
    except subprocess.CalledProcessError as e:
        logger.error(
            f"[{os.uname().nodename}] FFmpeg failed with exit code {e.returncode}"
        )
        if detach_event.is_set():
            if e.stdout:
                logger.info("FFmpeg STDOUT:\n" + e.stdout)
            if e.stderr:
                logger.error("FFmpeg STDERR:\n" + e.stderr)

        msg_queue.put(
            ("error", os.getpid(), file_name, f"ffmpeg failed (code {e.returncode})")
        )
        return False

    if use_ramdisk:
        try:
            os.remove(tmp_path)
            logger.info(f"Removed temp copy: {tmp_path}")
        except OSError as e:
            logger.warning(f"Warning: Could not remove temp file {tmp_path}: {e}")

    return True


def worker(args, detach_event, main_pid, msg_queue):
    logger = setup_logging(main_pid=main_pid)
    retries = 0

    temp_sh = None  # Get a reference to the temporary stream handler
    for handler in logger.handlers:
        if handler.get_name() == "temporary_stream_handler":
            temp_sh = handler
            break

    # First log message will go to both file and stream
    logger.info(f"Worker {os.getpid()} starting...")
    # Remove the temporary stream handler after the first log message
    if temp_sh:
        logger.removeHandler(temp_sh)

    while not detach_event.is_set():
        files = get_media_files(args.source_dir, args.input_extensions)
        if not files:
            # All subsequent log messages will go to the file only
            detach_event.set()
            logger.info("No files found at all, detaching immediately.")
            return

        picked_file = None
        lock_fd = None

        os.makedirs(os.path.join(args.output_dir, "tmp"), exist_ok=True)

        for f in files:
            # --- FIX: Use the helper function for a consistent check ---
            output_tmp_path = get_output_path(args.output_dir, f, args.output_extension)
            if os.path.exists(output_tmp_path):
                continue

            lock_path = os.path.join(args.output_dir, "tmp", f"{f}.lock")
            try:
                lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(lock_fd, f"{os.uname().nodename}:{os.getpid()}".encode())
                picked_file = f
                break
            except FileExistsError:
                continue

        if picked_file:
            # If we find a file, reset retries
            retries = 0
            try:
                success = process_file(
                    picked_file,
                    args.source_dir,
                    args.output_dir,
                    args.output_extension,
                    args.ffmpeg_flags,
                    args.use_ramdisk,
                    logger,
                    msg_queue,
                    detach_event,  # Corrected: Pass detach_event here
                )
                if not success:
                    msg_queue.put(("error", os.getpid(), picked_file, "failed"))
            finally:
                if lock_fd:
                    os.close(lock_fd)
                    os.remove(lock_path)
        else:
            # If no files are available, increment retries and sleep
            retries += 1
            if retries >= MAX_RETRIES:
                logger.info(
                    f"[{os.uname().nodename}] No unlocked files found after {MAX_RETRIES} retries, signaling detach."
                )
                detach_event.set()
                return  # Exit worker as there's no work

            sleep_time = random.randint(1, 4)
            logger.info(
                f"[{os.uname().nodename}] No unlocked files, retry {retries}/{MAX_RETRIES} in {sleep_time}s"
            )
            time.sleep(sleep_time)

    # Clean up and exit if detach_event is set
    logger.info(f"[{os.uname().nodename}] Exiting due to detach signal.")


def main():
    try:
        args = parse_args()
        main_pid = os.getpid()
        logger = setup_logging(is_main=True)
        detach_event = multiprocessing.Event()
        msg_queue = multiprocessing.Queue()

        if args.debug_enable:
            logger.info(f"[DEBUG] Source dir: {args.source_dir}")
            logger.info(f"[DEBUG] Output dir: {args.output_dir}")
            logger.info(f"[DEBUG] FFmpeg flags: {args.ffmpeg_flags}")
            logger.info(f"[DEBUG] Use Ramdisk: {args.use_ramdisk}")

        processes = []
        for _ in range(NUM_WORKERS):
            p = multiprocessing.Process(
                target=worker,
                args=(args, detach_event, main_pid, msg_queue),
            )
            p.start()
            processes.append(p)

        while not detach_event.is_set():
            try:
                level, worker_pid, file_name, text = msg_queue.get(timeout=1)
                msg = f"Worker {worker_pid} {text} on {file_name}"
                if level == "error":
                    logger.error(msg)
                else:
                    logger.info(msg)
            except queue.Empty:
                continue

        # --- FIX: Remove the stream handler before detaching ---
        main_logger = logging.getLogger(f"transcoder_{main_pid}")
        for handler in main_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                main_logger.removeHandler(handler)

        logger.info("Detaching: Workers will continue in the background.")
        detach()
        # The main process will now exit, leaving the workers to finish.
        # It's important to not have any further logging or output after detach()
        # to ensure the terminal is released.

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt caught: exiting...")
        for p in processes:
            p.terminate()
        exit()


if __name__ == "__main__":
    main()
