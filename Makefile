SHELL = /usr/bin/zsh -o errreturn

output_dir ?= docs
note_dir   := $(output_dir)/notes
asset_dir  := $(output_dir)/assets

css   ?= assets/sakura.css

notes != print notes/**/*.qmd(.:t)
html  := $(notes:%.qmd=$(note_dir)/%.html)

all: build
.PHONY: build build-index clean fix-links

clean:
	rm -rf $(output_dir)

build: $(html) \
	build-index \
	fix-links \
	$(output_dir)/$(css)

fix-links: $(html)
	zsh scripts/fix-links $(note_dir)

build-index: $(html)
	zsh scripts/build-index $(note_dir) > index.md
	quarto render index.md --to html --output-dir $(output_dir) -o index.html --css $(css)

$(asset_dir)/sakura.css: assets/sakura.css
	@mkdir -p $(asset_dir)
	cp $< $@

$(note_dir)/%.html: notes/*/%.qmd
	@mkdir -p $(note_dir)
	@mkdir -p $(asset_dir)
	quarto render $< --to html --output-dir $(note_dir) -o $(@F) --css ../$(css)
	if [[ -d $(<D)/$(basename $(<F))_files ]]; then \
		cp -r $(<D)/$(basename $(<F))_files $(asset_dir)/; \
	fi
