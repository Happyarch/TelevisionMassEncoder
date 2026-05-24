#!/usr/bin/env python3
import os
import subprocess
import random
import time
import shutil
import argparse
import shlex
import multiprocessing
import queue
import signal
import sys
import logging
import select
import termios
import tty
import configparser


MAX_RETRIES = 10
NUM_WORKERS = 5
CONFIG_PATH = os.path.expanduser("~/.config/dtme.conf")


def load_config():
    """
    Parse ~/.config/dtme.conf and return a dict of overrides ready for
    parser.set_defaults().  Unknown keys and malformed values are silently
    skipped with a warning so a bad config never prevents the tool from running.
    """
    config = configparser.RawConfigParser()
    if not config.read(CONFIG_PATH):
        return {}
    if "defaults" not in config:
        return {}

    section = config["defaults"]
    overrides = {}

    for key in ("ffmpeg_binary", "ffmpeg_flags", "output_extension", "input_extensions"):
        if key in section:
            overrides[key] = section[key].strip()

    for key in ("num_workers", "max_retries"):
        if key in section:
            try:
                overrides[key] = section.getint(key)
            except ValueError:
                print(f"dtme: config warning: '{key}' is not a valid integer, ignoring",
                      file=sys.stderr)

    for key in ("use_ramdisk", "debug_enable"):
        if key in section:
            try:
                overrides[key] = section.getboolean(key)
            except ValueError:
                print(f"dtme: config warning: '{key}' is not a valid boolean, ignoring",
                      file=sys.stderr)

    return overrides


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed video encoder with live terminal output and detach support."
    )
    parser.add_argument(
        "--source-dir", type=str, required=True, help="Path to input videos directory"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Path to output videos directory"
    )
    parser.add_argument(
        "--ffmpeg-flags",
        type=str,
        default=None,
        help="Flags to pass to ffmpeg (required if not set in config file)",
    )
    parser.add_argument(
        "--ffmpeg-binary",
        type=str,
        default="ffmpeg",
        help="Path to the ffmpeg executable (default: ffmpeg)",
    )
    parser.add_argument(
        "--debug_enable",
        action="store_true",
        help="Write the full FFmpeg command to the log for each file",
    )
    parser.add_argument(
        "--use-ramdisk",
        action="store_true",
        help="Copy source file to /tmp for potentially faster processing.",
    )
    parser.add_argument(
        "--output-extension",
        type=str,
        help="Output container extension (default: .mkv)",
    )
    parser.add_argument(
        "--input-extensions",
        type=str,
        help=(
            "Colon-separated list of file extensions to process (without leading dots). "
            "Example: mp4:mkv:avi. Defaults to all common FFmpeg-supported video/audio formats."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Number of parallel worker processes (default: {NUM_WORKERS})",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES,
        help=f"Max retry attempts per worker before giving up (default: {MAX_RETRIES})",
    )

    # Config file values become defaults; CLI args override them.
    parser.set_defaults(**load_config())
    args = parser.parse_args()

    if not args.ffmpeg_flags:
        parser.error(
            "--ffmpeg-flags is required (or set ffmpeg_flags in ~/.config/dtme.conf)"
        )

    return args


def get_media_files(source_dir, input_exts=None):
    """Return a shuffled list of media files in source_dir matching input_exts."""
    default_exts = (
        ".mkv", ".mp4", ".mov", ".avi", ".webm", ".divx", ".vob", ".evo",
        ".ogv", ".ogx", ".flv", ".f4v", ".aac", ".flac", ".mp3", ".ogg",
        ".opus", ".alac", ".mka", ".pcm", ".aiff", ".wav", ".cda", ".ape",
    )

    if input_exts:
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


def get_output_path(output_dir, file_name, output_extension):
    if not output_extension:
        output_extension = ".mkv"
    if not output_extension.startswith("."):
        output_extension = "." + output_extension
    output_dir_tmp = os.path.join(output_dir, "tmp")
    os.makedirs(output_dir_tmp, exist_ok=True)
    return os.path.join(
        output_dir_tmp, os.path.splitext(file_name)[0] + output_extension
    )


def setup_logging(main_pid=None, is_main=False):
    """Set up a file-only logger for this process."""
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
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)

    return logger


# Queue message format used throughout: (worker_id, level, text)
#   worker_id : int   — index 0..N-1, used for [Worker N] display prefix
#   level     : str   — "info" or "error"
#   text      : str   — human-readable event description


def process_file(
    file_name,
    source_dir,
    output_dir,
    output_extension,
    ffmpeg_binary,
    ffmpeg_flags,
    use_ramdisk,
    debug_enable,
    worker_id,
    logger,
    msg_queue,
):
    input_path = os.path.join(source_dir, file_name)
    working_input_path = input_path

    if use_ramdisk:
        tmp_path = os.path.join("/tmp", file_name)
        msg_queue.put((worker_id, "info", f"Copying to /tmp: {file_name}"))
        logger.info(f"Copying {file_name} to /tmp for faster encoding.")
        shutil.copy2(input_path, tmp_path)
        working_input_path = tmp_path

    output_path = get_output_path(output_dir, file_name, output_extension)
    flags_list = shlex.split(ffmpeg_flags)
    cmd = [ffmpeg_binary, "-i", working_input_path] + flags_list + [output_path]

    msg_queue.put((worker_id, "info", f"Encoding: {file_name}"))
    logger.info(f"Encoding: {file_name}")
    if debug_enable:
        logger.info(f"[DEBUG] Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        msg_queue.put((worker_id, "info", f"Finished: {file_name}"))
        logger.info(f"Finished: {file_name}")
        if result.stdout:
            logger.info("FFmpeg STDOUT:\n" + result.stdout)
        if result.stderr:
            logger.info("FFmpeg STDERR:\n" + result.stderr)

    except FileNotFoundError:
        msg_queue.put((worker_id, "error", f"FFmpeg binary not found: {ffmpeg_binary!r}"))
        logger.error(f"FFmpeg binary not found: {ffmpeg_binary!r}")
        return False
    except subprocess.CalledProcessError as e:
        msg_queue.put((worker_id, "error", f"FFmpeg failed (code {e.returncode}): {file_name}"))
        logger.error(f"FFmpeg failed with exit code {e.returncode}: {file_name}")
        if e.stdout:
            logger.info("FFmpeg STDOUT:\n" + e.stdout)
        if e.stderr:
            logger.error("FFmpeg STDERR:\n" + e.stderr)
        return False

    if use_ramdisk:
        try:
            os.remove(tmp_path)
            logger.info(f"Removed temp copy: {tmp_path}")
        except OSError as exc:
            logger.warning(f"Could not remove temp file {tmp_path}: {exc}")

    return True


def worker(args, detach_event, main_pid, msg_queue, worker_id):
    logger = setup_logging(main_pid=main_pid)
    retries = 0

    msg_queue.put((worker_id, "info", f"Started (PID {os.getpid()})"))

    while not detach_event.is_set():
        files = get_media_files(args.source_dir, args.input_extensions)
        if not files:
            detach_event.set()
            msg_queue.put((worker_id, "info", "No files found, stopping."))
            return

        picked_file = None
        lock_fd = None

        os.makedirs(os.path.join(args.output_dir, "tmp"), exist_ok=True)

        for f in files:
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
            retries = 0
            try:
                success = process_file(
                    picked_file,
                    args.source_dir,
                    args.output_dir,
                    args.output_extension,
                    args.ffmpeg_binary,
                    args.ffmpeg_flags,
                    args.use_ramdisk,
                    args.debug_enable,
                    worker_id,
                    logger,
                    msg_queue,
                )
                if not success:
                    msg_queue.put((worker_id, "error", f"Failed: {picked_file}"))
            finally:
                if lock_fd is not None:
                    os.close(lock_fd)
                    os.remove(lock_path)
        else:
            retries += 1
            if retries >= args.max_retries:
                msg_queue.put(
                    (worker_id, "info",
                     f"No unlocked files after {args.max_retries} retries, stopping.")
                )
                detach_event.set()
                return

            sleep_time = random.randint(1, 4)
            msg_queue.put(
                (worker_id, "info",
                 f"No unlocked files, retry {retries}/{args.max_retries} in {sleep_time}s")
            )
            time.sleep(sleep_time)

    msg_queue.put((worker_id, "info", "Exiting."))


def _display_loop(parent_conn, child_pid):
    """
    Read messages from the manager and print them to the terminal.

    Watches stdin for keypresses when running interactively:
      'd'    — detach (return "detach")
      Ctrl+C — abort   (return "interrupt")

    Returns "detach", "interrupt", or "done" (pipe closed / encoding finished).
    """
    is_tty = sys.stdin.isatty()
    old_settings = None

    if is_tty:
        stdin_fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(stdin_fd)
        tty.setcbreak(stdin_fd)

    print(f"Encoding started (manager PID {child_pid}).", flush=True)
    if is_tty:
        print("Press 'd' to detach, Ctrl+C to abort.", flush=True)

    poll_sources = [parent_conn]
    if is_tty:
        poll_sources.append(sys.stdin)

    try:
        while True:
            try:
                ready = select.select(poll_sources, [], [], 0.5)[0]
            except (ValueError, OSError):
                return "done"

            for source in ready:
                if is_tty and source is sys.stdin:
                    key = os.read(sys.stdin.fileno(), 1)
                    if key.lower() == b"d":
                        return "detach"
                elif source is parent_conn:
                    try:
                        worker_id, level, text = parent_conn.recv()
                        print(f"[Worker {worker_id}] {text}", flush=True)
                    except EOFError:
                        return "done"

    except KeyboardInterrupt:
        return "interrupt"
    finally:
        if old_settings is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)


def main():
    args = parse_args()

    # Pipe: manager (child) writes display messages; UI (parent) reads and prints them.
    # Using duplex=False gives a unidirectional OS pipe whose read end supports select().
    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)

    try:
        child_pid = os.fork()
    except OSError as e:
        print(f"Fork failed: {e}", file=sys.stderr)
        sys.exit(1)

    # -------------------------------------------------------------------------
    # MANAGER PROCESS
    # -------------------------------------------------------------------------
    if child_pid == 0:
        parent_conn.close()
        os.setsid()
        signal.signal(signal.SIGHUP, signal.SIG_IGN)

        # Manager has no use for terminal I/O; redirect to avoid stray output.
        devnull_fd = os.open(os.devnull, os.O_RDWR)
        for fd in (0, 1, 2):
            try:
                os.dup2(devnull_fd, fd)
            except OSError:
                pass
        os.close(devnull_fd)

        main_pid = os.getpid()
        logger = setup_logging(is_main=True)
        detach_event = multiprocessing.Event()
        msg_queue = multiprocessing.Queue()

        if args.debug_enable:
            logger.info(f"[DEBUG] Config file: {CONFIG_PATH if os.path.exists(CONFIG_PATH) else 'not found'}")
            logger.info(f"[DEBUG] Source dir: {args.source_dir}")
            logger.info(f"[DEBUG] Output dir: {args.output_dir}")
            logger.info(f"[DEBUG] FFmpeg binary: {args.ffmpeg_binary}")
            logger.info(f"[DEBUG] FFmpeg flags: {args.ffmpeg_flags}")
            logger.info(f"[DEBUG] Use Ramdisk: {args.use_ramdisk}")

        processes = []
        for i in range(args.num_workers):
            p = multiprocessing.Process(
                target=worker,
                args=(args, detach_event, main_pid, msg_queue, i),
            )
            p.start()
            processes.append(p)

        # Relay messages from workers to the UI pipe and the log file.
        # If the UI process has detached (pipe broken), keep logging to file only.
        pipe_alive = True

        def relay(wid, level, text):
            nonlocal pipe_alive
            if level == "error":
                logger.error(f"[Worker {wid}] {text}")
            else:
                logger.info(f"[Worker {wid}] {text}")
            if pipe_alive:
                try:
                    child_conn.send((wid, level, text))
                except (BrokenPipeError, OSError):
                    pipe_alive = False

        while any(p.is_alive() for p in processes):
            try:
                wid, level, text = msg_queue.get(timeout=1)
                relay(wid, level, text)
            except queue.Empty:
                continue

        # Drain any messages queued after the last liveness check.
        while True:
            try:
                wid, level, text = msg_queue.get_nowait()
                relay(wid, level, text)
            except queue.Empty:
                break

        for p in processes:
            p.join()

        logger.info("All workers finished.")
        try:
            child_conn.close()
        except OSError:
            pass
        os._exit(0)

    # -------------------------------------------------------------------------
    # UI PROCESS
    # -------------------------------------------------------------------------
    else:
        child_conn.close()

        result = _display_loop(parent_conn, child_pid)
        parent_conn.close()

        if result == "detach":
            print(f"\nDetached. Manager continues in background (PID {child_pid}).")
            print(f"To stop all workers: kill {child_pid}")
        elif result == "interrupt":
            print("\nInterrupted. Stopping all workers...")
            try:
                os.killpg(child_pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        elif result == "done":
            print("\nAll encoding complete.")

        os._exit(0)


if __name__ == "__main__":
    main()
