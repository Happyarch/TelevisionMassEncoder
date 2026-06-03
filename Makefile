PREFIX      ?= /usr/local
BINDIR      := $(DESTDIR)$(PREFIX)/bin
MANDIR      := $(DESTDIR)$(PREFIX)/share/man/man1
BASHCOMPDIR := $(DESTDIR)$(PREFIX)/share/bash-completion/completions
ZSHCOMPDIR  := $(DESTDIR)$(PREFIX)/share/zsh/site-functions
FISHCOMPDIR := $(DESTDIR)$(PREFIX)/share/fish/vendor_completions.d

USER_BINDIR      := $(HOME)/.local/bin
USER_MANDIR      := $(HOME)/.local/share/man/man1
USER_BASHCOMPDIR := $(HOME)/.local/share/bash-completion/completions
USER_ZSHCOMPDIR  := $(HOME)/.local/share/zsh/site-functions
USER_FISHCOMPDIR := $(HOME)/.config/fish/completions

SCRIPT    := Television_Mass_Encoder.py
CMD       := dtme
MANPAGE   := man/dtme.1
BASH_COMP := completions/dtme.bash
ZSH_COMP  := completions/_dtme
FISH_COMP := completions/dtme.fish

.PHONY: install uninstall reinstall install-user uninstall-user reinstall-user check

install: check
	install -d $(BINDIR)
	install -m 755 $(SCRIPT) $(BINDIR)/$(CMD)
	install -d $(MANDIR)
	install -m 644 $(MANPAGE) $(MANDIR)/$(CMD).1
	install -d $(BASHCOMPDIR)
	install -m 644 $(BASH_COMP) $(BASHCOMPDIR)/$(CMD)
	install -d $(ZSHCOMPDIR)
	install -m 644 $(ZSH_COMP) $(ZSHCOMPDIR)/_$(CMD)
	install -d $(FISHCOMPDIR)
	install -m 644 $(FISH_COMP) $(FISHCOMPDIR)/$(CMD).fish

uninstall:
	rm -f $(BINDIR)/$(CMD)
	rm -f $(MANDIR)/$(CMD).1
	rm -f $(BASHCOMPDIR)/$(CMD)
	rm -f $(ZSHCOMPDIR)/_$(CMD)
	rm -f $(FISHCOMPDIR)/$(CMD).fish

reinstall:
	$(MAKE) uninstall
	$(MAKE) install

install-user: check
	install -d $(USER_BINDIR)
	install -m 755 $(SCRIPT) $(USER_BINDIR)/$(CMD)
	install -d $(USER_MANDIR)
	install -m 644 $(MANPAGE) $(USER_MANDIR)/$(CMD).1
	install -d $(USER_BASHCOMPDIR)
	install -m 644 $(BASH_COMP) $(USER_BASHCOMPDIR)/$(CMD)
	install -d $(USER_ZSHCOMPDIR)
	install -m 644 $(ZSH_COMP) $(USER_ZSHCOMPDIR)/_$(CMD)
	install -d $(USER_FISHCOMPDIR)
	install -m 644 $(FISH_COMP) $(USER_FISHCOMPDIR)/$(CMD).fish

uninstall-user:
	rm -f $(USER_BINDIR)/$(CMD)
	rm -f $(USER_MANDIR)/$(CMD).1
	rm -f $(USER_BASHCOMPDIR)/$(CMD)
	rm -f $(USER_ZSHCOMPDIR)/_$(CMD)
	rm -f $(USER_FISHCOMPDIR)/$(CMD).fish

reinstall-user:
	$(MAKE) uninstall-user
	$(MAKE) install-user

check:
	@command -v python3 >/dev/null 2>&1 || \
		{ echo "Error: python3 not found in PATH"; exit 1; }
	@python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,7) else 1)" || \
		{ echo "Error: python3 3.7 or higher required"; exit 1; }
	@command -v ffmpeg >/dev/null 2>&1 || \
		{ echo "Warning: ffmpeg not found in PATH — install it before running dtme"; }
