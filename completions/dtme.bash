_dtme() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    opts="--source-dir --output-dir --ffmpeg-flags --ffmpeg-binary --debug_enable
          --use-ramdisk --output-extension --input-extensions --num-workers
          --max-retries --help"

    case "${prev}" in
        --source-dir|--output-dir)
            COMPREPLY=( $(compgen -d -- "${cur}") )
            return 0
            ;;
        --ffmpeg-binary)
            COMPREPLY=( $(compgen -f -- "${cur}") )
            return 0
            ;;
    esac

    COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
    return 0
}

complete -F _dtme dtme
