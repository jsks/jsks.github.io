/*
 * This file implements dynamic highlighting of the table of contents
 * entries as a user scrolls through the page. Also dynamically
 * unrolls/rolls TOC submenus depending on the current section.
 */

function visible(elements) {
    const windowHeight = window.innerHeight

    const heights = Array.from(elements).map(elem => {
        const rect = elem.getBoundingClientRect(),
                top = Math.max(0, rect.top),
                bottom = Math.min(windowHeight, rect.bottom)

        return Math.max(0, bottom - top)
    })

    // None of the elements are visible
    if (heights.every(x => x == 0))
        return null

    const idx = heights.reduce((aux, height, i) => {
        if (height > heights[aux] ||
            (elements[aux].contains(elements[i]) &&
                height > 0.5 * heights[aux]))
            return i
        else
            return aux
    }, 0)

    return elements[idx]
}

function submenu(target) {
    const ul = target.closest("ul")

    // In case current selection is hidden, unroll the parent
    ul.style.display = "block"

    ul.querySelectorAll(":scope > li > a")
        .forEach(elem => {
            const next = elem.nextElementSibling
            if (!next || next.tagName !== "UL")
                return

            next.style.display = (elem.classList.contains('active') &&
                elem === target) ? "block" : "none"
        })
}

function throttle(fn, ms) {
    let block = null
    return (...args) => {
        if (!block) {
            fn.apply(this, args)
            block = setTimeout(() => block = null, ms)
        }
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const sections = document.querySelectorAll('section:not(#footnotes)'),
            links = document.querySelectorAll('aside nav ul li a')

    if (!sections.length || !links.length)
        return

    // This is ugly, but we want to block scroll events when a link
    // in the TOC is clicked. Otherwise, the callback will update
    // the ui during the smooth scroll.
    let clickEventFlag = false

    // Function to update the active highlighting in the TOC
    // depending on which section is most visible. Allows for nested
    // sections.
    const updateLinks = () => {
        if (clickEventFlag)
            return

        const visibleSection = visible(sections)
        for (elem of links) {
            elem.classList.remove('active')
            if (visibleSection && elem.getAttribute('href') == `#${visibleSection.id}`) {
                elem.classList.add('active')
                submenu(elem)
            }
        }
    }

    // When a link is clicked in the TOC, highlight as active and
    // unroll any submenus
    links.forEach(link => {
        link.addEventListener("click", event => {
            clickEventFlag = true

            links.forEach(link => link.classList.remove('active'))
            link.classList.add('active')
            submenu(link)

            // Note: `scrollend` is not implemented on safari
            addEventListener("scrollend", () => {
                clickEventFlag = false
                updateLinks()
            }, {once: true})
        })
    })

    // Update the TOC when the page is scrolled or resized
    window.addEventListener('scroll', throttle(updateLinks, 50))
    window.addEventListener('resize', updateLinks)

    // Initial update when page is loaded
    updateLinks()
})
