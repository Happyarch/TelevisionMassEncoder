complete -c dtme -f

complete -c dtme -l source-dir       -r -a '(__fish_complete_directories)' -d 'Path to input videos directory'
complete -c dtme -l output-dir       -r -a '(__fish_complete_directories)' -d 'Path to output videos directory'
complete -c dtme -l ffmpeg-flags     -r                                     -d 'Flags to pass to ffmpeg'
complete -c dtme -l ffmpeg-binary    -r -F                                  -d 'Path to the ffmpeg executable'
complete -c dtme -l debug_enable                                             -d 'Write full FFmpeg command to log'
complete -c dtme -l use-ramdisk                                              -d 'Copy source file to /tmp for faster processing'
complete -c dtme -l output-extension -r                                      -d 'Output container extension (default: .mkv)'
complete -c dtme -l input-extensions -r                                      -d 'Colon-separated list of input extensions (e.g. mp4:mkv:avi)'
complete -c dtme -l num-workers      -r                                      -d 'Number of parallel worker processes (default: 5)'
complete -c dtme -l max-retries      -r                                      -d 'Max retry attempts per worker (default: 10)'
