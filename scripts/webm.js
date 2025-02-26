// Auto-loop all webm videos
document.addEventListener("DOMContentLoaded", () => {
    elems = document.getElementsByTagName("video")
    for (let el of elems) {
        if (el.querySelector("source[type='video/webm']"))
            el.setAttribute("loop", "")
    }
})
