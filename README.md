# TelevisionMassEncoder

A parallel, distributed video encoding pipeline built on top of FFmpeg. Designed for mass-encoding large libraries of television or other video content, it coordinates multiple worker processes using filesystem-level locking — meaning it can safely run across multiple machines sharing a common network drive simultaneously.

## How It Works

At startup, the script spawns **5 worker processes** that race to claim and encode files from a source directory. Claiming is done by atomically creating a `.lock` file (using `O_CREAT | O_EXCL` — a Linux syscall that fails if the file already exists) in the output `tmp/` directory. This guarantees that no two workers — whether on the same machine or different ones — will encode the same file at the same time.

Workers shuffle the file list before iterating, which distributes load more evenly when running multiple instances of the script across machines. Once encoding is complete, the lock file is deleted and the finished output is left in `output_dir/tmp/`.

After all files have been claimed or a retry threshold is hit, the main process **detaches from the terminal** via a double-fork (a standard Unix daemonization technique), leaving the remaining workers to finish in the background without holding up your shell session.

## Features

- **Parallel encoding** — 5 concurrent FFmpeg worker processes by default
- **Distributed-safe locking** — filesystem lock files work across NFS/SMB mounts
- **Ramdisk support** — optionally copies source files to `/tmp` before encoding for faster I/O
- **Flexible format support** — configurable input extensions; defaults to a wide range of common video and audio containers
- **Configurable output** — pass any FFmpeg flags directly; output container is configurable
- **Per-process logging** — each worker writes its own log file; the main process writes a separate one
- **Graceful detachment** — once all files are picked up, the main process daemonizes and exits the terminal cleanly
- **Retry logic** — workers retry up to 10 times with randomized backoff before giving up

## Requirements

- Python 3.7+
- `ffmpeg` available in your `PATH`

No third-party Python packages are required — only the standard library.

## Usage

```bash
python Television_Mass_Encoder.py \
  --source-dir /path/to/source/videos \
  --output-dir /path/to/output \
  --ffmpeg-flags "-c:v libsvtav1 -crf 28 -preset 6 -c:a libopus -b:a 192k"
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--source-dir` | Yes | Directory containing input video files |
| `--output-dir` | Yes | Directory where encoded files will be written (into a `tmp/` subdirectory) |
| `--ffmpeg-flags` | Yes | FFmpeg encoding flags as a single quoted string, passed directly to FFmpeg |
| `--output-extension` | No | Output container extension (default: `.mkv`) |
| `--input-extensions` | No | Colon-separated list of extensions to process, e.g. `mp4:mkv:avi` (default: common video/audio formats) |
| `--use-ramdisk` | No | Copy each source file to `/tmp` before encoding for potentially faster reads |
| `--debug_enable` | No | Print extra debug information to the log |
| `--num-workers` | No | Number of parallel worker processes (default: 5) |
| `--max-retries` | No | Max retry attempts per worker before giving up (default: 10) |

### Example: AV1 encoding with Opus audio

```bash
python Television_Mass_Encoder.py \
  --source-dir /mnt/nas/shows \
  --output-dir /mnt/nas/encoded \
  --ffmpeg-flags "-c:v libsvtav1 -crf 30 -preset 5 -g 240 -c:a libopus -b:a 128k" \
  --output-extension .mkv \
  --input-extensions mkv:mp4:avi
```

### Example: Running across two machines on a shared NAS

Start the script with identical arguments on both machines. The `.lock` file mechanism will prevent double-encoding automatically, as long as both machines have read/write access to `--output-dir`.

```bash
# Machine A
python Television_Mass_Encoder.py --source-dir /mnt/nas/shows --output-dir /mnt/nas/encoded --ffmpeg-flags "..."

# Machine B (same command simultaneously)
python Television_Mass_Encoder.py --source-dir /mnt/nas/shows --output-dir /mnt/nas/encoded --ffmpeg-flags "..."
```

## Output Structure

```
output-dir/
└── tmp/
    ├── Show.S01E01.mkv
    ├── Show.S01E01.mkv.lock   ← present only while encoding; deleted on completion
    ├── Show.S01E02.mkv
    └── ...
```

Encoded files are placed in `output-dir/tmp/`. You can move or rename them once encoding finishes.

## Logging

The main process writes a log named `main_<pid>.log` in the current working directory. Each worker writes its log to a subdirectory named `<main_pid>_worker_logs/`.

```
./
├── main_12345.log
└── 12345_worker_logs/
    ├── transcode_12345_12346.log
    ├── transcode_12345_12347.log
    └── ...
```

## Notes

- The script is opinionated about output going into a `tmp/` subdirectory of `--output-dir`. Post-processing (moving files to a final destination, renaming, etc.) is left to the user.
- The number of worker processes and max retry count default to 5 and 10 respectively, and can be overridden with `--num-workers` and `--max-retries`.
- Pressing `Ctrl+C` before detachment will terminate all workers cleanly.
- Once the main process detaches, it cannot be stopped with `Ctrl+C`. Use `kill <pid>` on the individual worker PIDs (visible in the log files) to stop them.

## License

MIT — see [LICENSE](LICENSE) for details.
