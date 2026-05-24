PREFIX  ?= /usr/local
BINDIR  := $(DESTDIR)$(PREFIX)/bin
MANDIR  := $(DESTDIR)$(PREFIX)/share/man/man1
SCRIPT  := Television_Mass_Encoder.py
CMD     := dtme
MANPAGE := man/dtme.1

.PHONY: install uninstall check

install: check
	install -d $(BINDIR)
	install -m 755 $(SCRIPT) $(BINDIR)/$(CMD)
	install -d $(MANDIR)
	install -m 644 $(MANPAGE) $(MANDIR)/$(CMD).1

uninstall:
	rm -f $(BINDIR)/$(CMD)
	rm -f $(MANDIR)/$(CMD).1

check:
	@command -v python3 >/dev/null 2>&1 || \
		{ echo "Error: python3 not found in PATH"; exit 1; }
	@python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,7) else 1)" || \
		{ echo "Error: python3 3.7 or higher required"; exit 1; }
	@command -v ffmpeg >/dev/null 2>&1 || \
		{ echo "Warning: ffmpeg not found in PATH — install it before running dtme"; }
