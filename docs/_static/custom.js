window.onload = () => {
  // detect enums and make sure their description is easily to style
  document.querySelectorAll(".class").forEach((class_elt) => {
    if (isEnum(class_elt)) {
      title = class_elt.querySelector("dt");
      title.innerHTML = title.innerHTML.replace(/, /g, "<span>, </span>");
      class_elt.classList.add("enum");
    }
  });

  // remove the dark theme watermark added in js
  footer = document.querySelector("footer");
  dark_theme_link =
    '<a href="https://github.com/MrDogeBro/sphinx_rtd_dark_mode">Dark theme</a>' +
    " provided by " +
    '<a href="http://mrdogebro.com">MrDogeBro</a>.';
  whenCondition(
    () => (footer.innerHTML = footer.innerHTML.replace(dark_theme_link, "")),
    () => footer.innerHTML.includes(dark_theme_link)
  );

  r = document.querySelector(":root");
  r.style.setProperty("--dark-link-color", "#80cb53");

  // t = document.querySelector('html[data-theme="dark"] .wy-side-nav-search')
  // t.style.setProperty('background-color', '#0d0d0d');

  // move the attribute and properties of a class to appear first in the class
  document.querySelectorAll(".py.class").forEach((pythonClass) => {
    if (!isEnum(pythonClass)) {
      appendAfter = getEndOfClassHeader(pythonClass);
      pythonClass.querySelectorAll(".py.attribute").forEach((attribute) => {
        appendAfter.insertAdjacentElement("afterend", attribute);
        appendAfter = attribute;
      });
      pythonClass.querySelectorAll(".py.property").forEach((property) => {
        appendAfter.insertAdjacentElement("afterend", property);
        appendAfter = property;
      });
    }
  });

  // move the constants of a module to appear first in the module
  document.querySelectorAll(".target").forEach((moduleMarker) => {
    if (moduleMarker.id.startsWith("module-")) {
      module_description =
        moduleMarker.parentElement.querySelectorAll(".target ~ p");
      if (module_description.length != 0)
        appendAfter = module_description[module_description.length - 1];
      else appendAfter = moduleMarker;
      sibling = moduleMarker.nextSibling;
      for (; sibling; ) {
        next = sibling.nextSibling;
        if (sibling.nodeName === "DL" && sibling.classList.contains("data")) {
          appendAfter.insertAdjacentElement("afterend", sibling);
          appendAfter = sibling;
        }
        sibling = next;
      }
    }
  });

  // move the abstract classes of a module to appear first in the module
  document.querySelectorAll(".section").forEach((section) => {
    if (section.id.startsWith("module-")) {
      descriptionStart = section.querySelector("dl");
      appendAfter = descriptionStart.previousElementSibling;
      sibling = descriptionStart;
      for (; sibling; ) {
        next = sibling.nextElementSibling;
        if (sibling.classList.contains("class") && isAbstract(sibling)) {
          appendAfter.insertAdjacentElement("afterend", sibling);
          appendAfter = sibling;
        }
        sibling = next;
      }
    }
  });

  // in the native gates section, add the list of native gates
  ngListLocation = document.getElementById("native-gates-list");
  if (ngListLocation) {
    ngSection = ngListLocation.parentElement;
    ngClasses = ngSection.querySelectorAll("dl.py.class");
    ngLinks = "";

    ngClasses.forEach(function (c) {
      if (c.textContent.includes("ABC")) return;
      ngLinks += `<a href="#${c.querySelector("dt.sig.sig-object.py").id}">
        ${c.querySelector("span.descname").innerText}
      </a>`;
    });
    ngListLocation.innerHTML = ngLinks;
  }

  // we add a note for abstract classes to remind that they cannot be
  // implemented directly
  document.querySelectorAll(".class").forEach((class_elt) => {
    if (isABC(class_elt)) {
      parents = class_elt.querySelector("dd > p:first-child");
      template = document.createElement("template");
      template.innerHTML = `
      <div class="admonition note">
        <p class="admonition-title"><span>Note</span></p>
        <p>
          Abstract classes (ABCs) are not meant to be instantiated as is. See 
          classes that inherits from this one to check how to instantiate it.
        </p>
      </div>`;
      parents.insertAdjacentElement("afterend", template.content.children[0]);
    }
  });
};

function getEndOfClassHeader(elt) {
  const endElements = [elt];
  admonition = elt.querySelectorAll(":scope>dd>.admonition");
  if (admonition.length != 0)
    endElements.push(admonition[admonition.length - 1]);
  field_list = elt.querySelector(".class>dd>.field-list");
  if (field_list !== null)
    endElements.push(elt.querySelector(".class>dd>.field-list"));
  examples = elt.querySelectorAll(".class>dd>div.doctest");
  if (examples.length != 0) endElements.push(examples[examples.length - 1]);
  offsets = endElements.map((elt) => elt.offsetTop);
  return endElements[offsets.indexOf(Math.max(...offsets))];
}

function isABC(elt) {
  parents = elt.querySelector("dd > p:first-child");
  return parents && parents.innerHTML.includes("ABC");
}

function isEnum(elt, explored = []) {
  id = elt.querySelector("dt").id;
  if (explored.includes(id)) return false;

  explored.push(id);

  parents = elt.querySelector("dd > p:first-child");
  if (parents && parents.innerHTML.includes("Enum")) {
    return true;
  }

  parents_contain_enum = false;
  elt.querySelectorAll("dd > p:first-child > a").forEach((link) => {
    parent_id = link.href.split("#")[1];
    new_elt = document.querySelector(`.class:has([id='${parent_id}'])`);
    if (new_elt && isEnum(new_elt, explored)) parents_contain_enum = true;
  });

  return parents_contain_enum;
}

function isAbstract(elt) {
  parents = elt.querySelector("dd > p:first-child");
  if (parents && parents.innerHTML.includes("ABC")) {
    return true;
  }
  return false;
}

function whenCondition(action, condition) {
  function watcher() {
    if (condition()) {
      action();
      clearInterval(interval);
    }
  }

  interval = setInterval(watcher, 50);
}
