SHELL = /usr/bin/zsh -o errreturn

site  ?= docs
css   ?= assets/sakura.css

notes != print notes/**/*.qmd(.:t)
html  := $(notes:%.qmd=$(site)/notes/%.html)

all: build
.PHONY: build build-index clean fix-links

clean:
	rm -rf $(site)

build: $(html) \
	build-index \
	fix-links \
	$(site)/$(css)

fix-links: $(html)
	zsh scripts/fix-links $(site)/notes/

build-index: $(html)
	zsh scripts/build-index $(site)/notes > index.md
	quarto render index.md --to html --output-dir $(site) -o index.html --css $(css)

$(site)/assets/%.css: assets/%.css
	@mkdir -p $(notes)/assets
	cp $< $@

$(site)/notes/%.html: notes/*/%.qmd
	@mkdir -p $(site)/{notes,assets}
	quarto render $< --to html --output-dir $(site)/notes -o $(@F) --css ../$(css)
	if [[ -d $(<D)/$(basename $(<F))_files ]]; then \
		cp -r $(<D)/$(basename $(<F))_files $(site)/assets; \
	fi
