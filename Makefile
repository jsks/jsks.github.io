SHELL = /usr/bin/zsh -o errreturn

site   ?= docs
assets := $(site)/assets
pages  := $(site)/notes

css    := styles/sakura.css styles/site.css
js     := $(wildcard scripts/*.js)
static := $(css:styles/%=$(assets)/css/%) $(js:scripts/%=$(assets)/js/%)

notes != print notes/**/*.qmd(.:t)
html  := $(notes:%.qmd=$(pages)/%.html)

all: build
.PHONY: build build-index clean fix-links
.NOTPARALLEL:

clean:
	rm -rf $(site)

build: $(html) \
	build-index \
	fix-links \
	$(static)

fix-links: $(html)
	zsh bin/fix-links $(pages)

build-index: $(html)
	zsh bin/build-index $(pages) > index.md
	quarto render index.md --to html --output-dir $(site) -o index.html

$(assets)/css/%.css: styles/%.css
	@mkdir -p $(@D)
	cp $< $@

$(assets)/js/%.js: scripts/%.js
	@mkdir -p $(@D)
	cp $< $@

$(pages)/%.html: notes/*/%.qmd
	@mkdir -p $(@D) $(assets)
	quarto render $< --profile notes --to html --output-dir $(site)/notes -o $(@F)
	if [[ -d $(<D)/$(basename $(<F))_files ]]; then \
		cp -r $(<D)/$(basename $(<F))_files $(assets); \
	fi
