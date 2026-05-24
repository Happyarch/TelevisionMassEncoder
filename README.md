# TelevisionMassEncoder

A parallel, distributed video encoding pipeline built on top of FFmpeg. Designed for mass-encoding large libraries of television or other video content, it coordinates multiple worker processes using filesystem-level locking — meaning it can safely run across multiple machines sharing a common network drive simultaneously.

## How It Works

At startup, the script forks into two processes:

- The **UI process** (the one your shell launched) stays in the foreground. It displays live encoding events in the terminal, prefixed with `[Worker N]` so you always know which worker is responsible. It also watches for keyboard input.
- The **manager process** (the forked child) calls `setsid()` to detach from the controlling terminal, then spawns all worker processes as its own children and waits for them to finish.

Workers race to claim files by atomically creating a `.lock` file (using `O_CREAT | O_EXCL` — a Linux syscall that fails if the file already exists) in the output `tmp/` directory. This guarantees that no two workers — whether on the same machine or different ones — will encode the same file at the same time.

Workers shuffle the file list before iterating, which distributes load more evenly when running multiple instances across machines. Once encoding is complete, the lock file is deleted and the finished output is left in `output_dir/tmp/`.

### Detaching

While the script is running, press **`d`** to detach. The UI process exits (returning your shell prompt), while the manager and all workers continue running in the background. Because workers are always children of the manager, the entire encoding job can be stopped at any time with a single `kill <manager_pid>`.

## Features

- **Parallel encoding** — 5 concurrent FFmpeg worker processes by default
- **Distributed-safe locking** — filesystem lock files work across NFS/SMB mounts
- **Live terminal output** — encoding events streamed to the terminal with `[Worker N]` identifiers while running in the foreground
- **Press `d` to detach** — hands off to the background manager at any time, freeing the shell without interrupting encoding
- **Ramdisk support** — optionally copies source files to `/tmp` before encoding for faster I/O
- **Flexible format support** — configurable input extensions; defaults to a wide range of common video and audio containers
- **Configurable output** — pass any FFmpeg flags directly; output container is configurable
- **Per-process logging** — each worker writes its own log file; the manager writes a separate one
- **Retry logic** — workers retry up to 10 times with randomized backoff before giving up

## Requirements

- Python 3.7+
- `ffmpeg` available in your `PATH`

No third-party Python packages are required — only the standard library.

## Installation

```bash
git clone https://github.com/Happyarch/TelevisionMassEncoder.git
cd TelevisionMassEncoder
sudo make install
```

This installs `dtme` to `/usr/local/bin` and the man page to `/usr/local/share/man/man1/`. The `check` step will warn you if `ffmpeg` is missing.

**Custom install prefix** (e.g. user-local, no `sudo` needed):

```bash
make install PREFIX=~/.local
```

Ensure `~/.local/bin` is in your `PATH`.

**Uninstall:**

```bash
sudo make uninstall
# or, if installed to a custom prefix:
make uninstall PREFIX=~/.local
```

## Configuration File

`dtme` reads `~/.config/dtme.conf` on startup. Values there become the defaults for every run; command-line arguments always take precedence.

```bash
cp dtme.conf.example ~/.config/dtme.conf
$EDITOR ~/.config/dtme.conf
```

Example config:

```ini
[defaults]
ffmpeg_binary = /usr/local/bin/ffmpeg
ffmpeg_flags = -c:v libsvtav1 -crf 28 -preset 6 -c:a libopus -b:a 192k
output_extension = .mkv
num_workers = 8
use_ramdisk = true
```

With `ffmpeg_flags` set in the config, `--ffmpeg-flags` becomes optional on the command line:

```bash
dtme --source-dir /nas/shows --output-dir /nas/encoded
```

All available keys are documented with comments in `dtme.conf.example`.

## Usage

```bash
dtme \
  --source-dir /path/to/source/videos \
  --output-dir /path/to/output \
  --ffmpeg-flags "-c:v libsvtav1 -crf 28 -preset 6 -c:a libopus -b:a 192k"
```

Or run directly without installing:

```bash
python3 Television_Mass_Encoder.py \
  --source-dir /path/to/source/videos \
  --output-dir /path/to/output \
  --ffmpeg-flags "-c:v libsvtav1 -crf 28 -preset 6 -c:a libopus -b:a 192k"
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--source-dir` | Yes | Directory containing input video files |
| `--output-dir` | Yes | Directory where encoded files will be written (into a `tmp/` subdirectory) |
| `--ffmpeg-flags` | Yes* | FFmpeg encoding flags as a single quoted string, passed directly to FFmpeg |
| `--ffmpeg-binary` | No | Path to the ffmpeg executable (default: `ffmpeg`) |
| `--output-extension` | No | Output container extension (default: `.mkv`) |
| `--input-extensions` | No | Colon-separated list of extensions to process, e.g. `mp4:mkv:avi` or `.mp4:.mkv:.avi` — leading dot is optional (default: common video/audio formats) |
| `--use-ramdisk` | No | Copy each source file to `/tmp` before encoding for potentially faster reads |
| `--debug_enable` | No | Write the full FFmpeg command to the log for each file |
| `--num-workers` | No | Number of parallel worker processes (default: 5) |
\* Required unless `ffmpeg_flags` is set in `~/.config/dtme.conf`.
| `--max-retries` | No | Max retry attempts per worker before giving up (default: 10) |

### Example: AV1 encoding with Opus audio

```bash
dtme \
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
dtme --source-dir /mnt/nas/shows --output-dir /mnt/nas/encoded --ffmpeg-flags "..."

# Machine B (same command simultaneously)
dtme --source-dir /mnt/nas/shows --output-dir /mnt/nas/encoded --ffmpeg-flags "..."
```

## Terminal Output

While running in the foreground, events are printed as they happen:

```
Encoding started (manager PID 12346). Press 'd' to detach, Ctrl+C to abort.
[Worker 0] Started (PID 12347)
[Worker 1] Started (PID 12348)
[Worker 0] Encoding: Show.S01E01.mkv
[Worker 1] Encoding: Show.S01E02.mkv
[Worker 0] Finished: Show.S01E01.mkv
[Worker 0] Encoding: Show.S01E03.mkv
...
```

Press **`d`** at any time to detach:

```
Detached. Manager continues in background (PID 12346).
To stop all workers: kill 12346
$
```

Press **`Ctrl+C`** to stop all workers immediately and exit.

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

The manager writes a log named `main_<pid>.log` in the current working directory. Each worker writes its log to a subdirectory named `<manager_pid>_worker_logs/`. Worker logs include the full FFmpeg stdout/stderr for each file.

```
./
├── main_12346.log
└── 12346_worker_logs/
    ├── transcode_12346_12347.log
    ├── transcode_12346_12348.log
    └── ...
```

## Notes

- The script is opinionated about output going into a `tmp/` subdirectory of `--output-dir`. Post-processing (moving files to a final destination, renaming, etc.) is left to the user.
- The number of worker processes and max retry count default to 5 and 10 respectively, and can be overridden with `--num-workers` and `--max-retries`.
- The manager PID is printed at startup. Use `kill <manager_pid>` to stop the manager and all workers together; because workers share the manager's process group, a single `kill` is sufficient.
- If stdin is not a terminal (e.g. piped input or a non-interactive context), the `d` key and `Ctrl+C` handling are skipped and the UI process simply mirrors output until encoding is complete.

## License

MIT — see [LICENSE](LICENSE) for details.
