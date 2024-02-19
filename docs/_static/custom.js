window.onload = () => {
    // detect enums and make sure their description is easily to style
    document.querySelectorAll(".class").forEach(class_elt => {
        if (isEnum(class_elt)) {
            title = class_elt.querySelector("dt");
            title.innerHTML = title.innerHTML.replace(/, /g, "<span>, </span>");
            class_elt.classList.add("enum");
        }
    })
    
    // remove the dark theme watermark added in js
    footer = document.querySelector("footer")
    dark_theme_link = (
        '<a href="https://github.com/MrDogeBro/sphinx_rtd_dark_mode">Dark theme</a>'
        + ' provided by '
        + '<a href="http://mrdogebro.com">MrDogeBro</a>.'
    )
    whenCondition(
        () => footer.innerHTML = footer.innerHTML.replace(dark_theme_link, ''), 
        () => footer.innerHTML.includes(dark_theme_link)
    )
	
	r = document.querySelector(':root')
	r.style.setProperty('--dark-link-color', '#80cb53');
	
	// t = document.querySelector('html[data-theme="dark"] .wy-side-nav-search')
	// t.style.setProperty('background-color', '#0d0d0d');

    // move the attribute and properties of a class to appear first in the class
    document.querySelectorAll(".py.class").forEach(pythonClass => {
        if (!isEnum(pythonClass)) {
            appendAfter = getEndOfClassHeader(pythonClass)
            pythonClass.querySelectorAll(".py.attribute").forEach(attribute => {
                appendAfter.insertAdjacentElement('afterend', attribute)
                appendAfter = attribute
            })
            pythonClass.querySelectorAll(".py.property").forEach(property => {
                appendAfter.insertAdjacentElement('afterend', property)
                appendAfter = property
            })
        }
    })

    // move the constants of a module to appear first in the module
    document.querySelectorAll(".target").forEach(moduleMarker => {
        if (moduleMarker.id.startsWith("module-")) {
            appendAfter = moduleMarker
            sibling = moduleMarker.nextSibling
            for (; sibling; ) {
                next = sibling.nextSibling
                if (sibling.nodeName === "DL" && sibling.classList.contains("data")) {
                    appendAfter.insertAdjacentElement('afterend', sibling)
                    appendAfter = sibling
                }
                sibling = next
            }
        }
    })

    // move the abstract classes of a module to appear first in the module
    document.querySelectorAll(".section").forEach(section => {
        if (section.id.startsWith("module-")) {
            descriptionStart = section.querySelector("dl")
            appendAfter = descriptionStart.previousElementSibling
            sibling = descriptionStart
            for (; sibling; ) {
                next = sibling.nextElementSibling
                if (sibling.classList.contains("class") && isAbstract(sibling)) {
                    appendAfter.insertAdjacentElement('afterend', sibling)
                    appendAfter = sibling
                }
                sibling = next
            }
        }
    })
	
	
}

// const themeButton = document.querySelector('themeSwitcher')
// themeButton.addEventListener('click', () => {
    //
// });


function getEndOfClassHeader(elt) {
    admonition = elt.querySelectorAll(".admonition")
    if (admonition.length != 0) return admonition[admonition.length - 1]
    examples = elt.querySelectorAll("dd>div.doctest")
    if (examples.length != 0) return examples[examples.length - 1]
    return elt.querySelector(".field-list")
}

function isEnum(elt, explored = []) {
    id = elt.querySelector("dt").id
    if (explored.includes(id)) return false
    
    explored.push(id)

    parents = elt.querySelector("dd > p:first-child")
    if (parents && parents.innerHTML.includes("Enum")) {
        return true
    }

    parents_contain_enum = false
    elt.querySelectorAll("dd > p:first-child > a").forEach(link => {
        parent_id = link.href.split("#")[1]
        new_elt = document.querySelector(`.class:has([id='${parent_id}'])`)
        if (new_elt && isEnum(new_elt, explored)) parents_contain_enum = true
    })

    return parents_contain_enum
}

function isAbstract(elt) {
    parents = elt.querySelector("dd > p:first-child")
    if (parents && parents.innerHTML.includes("ABC")) {
        return true
    }
    return false
}

function whenCondition(action, condition) {
    function watcher() {
        if(condition()) {
            action()
            clearInterval(interval)
        }
    }
    
    interval = setInterval(watcher, 50);
}