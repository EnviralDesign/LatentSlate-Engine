const $ = (id) => document.getElementById(id);
const state = {
  builtins: [], users: [], operations: new Map(), document: null,
  builtinKey: null, revision: null, hash: null, dirty: false, busy: false,
  validation: null, token: sessionStorage.getItem("latentslate.authoring.token") || "",
  picker: null, searchSequence: 0, searchTimer: null,
  imports: [], importBusy: false,
  publication: null,
};

// Browser-native source-aware JSON keeps the existing unsigned 64-bit integer
// domain intact. Number coercion must not silently round saved seeds/bounds.
function parseJSON(text) {
  return JSON.parse(text, (key, value, context) => {
    if (typeof value === "number" && Number.isInteger(value) && !Number.isSafeInteger(value)) {
      if (!context?.source) throw new Error("This browser cannot preserve large integers. Please use a current browser.");
      if (/^-?\d+$/.test(context.source)) return BigInt(context.source);
    }
    return value;
  });
}

function stringifyJSON(value) {
  return JSON.stringify(value, (key, item) => typeof item === "bigint" ? JSON.rawJSON(item.toString()) : item);
}

function element(tag, attributes = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (["value", "checked", "disabled", "hidden", "readOnly"].includes(key)) node[key] = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  return node;
}

function label(key) {
  return key.replaceAll("_", " ").replace(/\bloras\b/gi, "LoRAs").replace(/\bcfg\b/gi, "CFG").replace(/^./, (letter) => letter.toUpperCase());
}

function pill(text, kind = "") {
  return element("span", { class: `status-pill ${kind}`, text });
}

function notice(message = "", error = false, action = null) {
  const node = $("notice");
  node.hidden = !message;
  node.className = `notice${error ? " error" : ""}`;
  node.replaceChildren(document.createTextNode(message));
  if (action) node.append(element("button", { class: "secondary", text: action.text, onclick: action.run }));
}

async function api(path, options = {}, textResponse = false) {
  const headers = {};
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  if (options.body !== undefined) headers["Content-Type"] = "application/json";
  let response;
  try {
    response = await fetch(`/v1/authoring${path}`, { ...options, headers, body: options.body === undefined ? undefined : typeof options.body === "string" ? options.body : stringifyJSON(options.body) });
  } catch {
    throw new Error("Cannot reach this Engine. Check the connection and try again.");
  }
  const text = await response.text();
  if (response.ok && textResponse) return text;
  let data;
  try { data = parseJSON(text); } catch (error) {
    if (response.ok) throw error;
    data = {};
  }
  if (!response.ok) {
    const error = new Error(data.error?.message || `Engine request failed (${response.status})`);
    error.status = response.status;
    error.data = data;
    throw error;
  }
  return data;
}

function showError(error) {
  if (error.data?.validation) {
    state.validation = error.data.validation;
    renderValidation(true);
  }
  if (error.status === 401) {
    $("connection-status").textContent = "Authentication required";
    $("connection-button").classList.remove("connected");
    $("connection-error").textContent = "This Engine requires a bearer token.";
    openConnection();
  } else if (error.status === 409 && state.document && !state.builtinKey) {
    notice(`${error.message}. Your unsaved edits are still here.`, true, { text: "Reload latest", run: reloadLatest });
  } else notice(error.message, true);
}

async function work(action) {
  if (state.busy) return;
  state.busy = true;
  document.querySelector(".workspace").inert = true;
  updateToolbar();
  try { await action(); } catch (error) { showError(error); }
  finally {
    state.busy = false;
    document.querySelector(".workspace").inert = false;
    updateToolbar();
  }
}

function updateToolbar() {
  const builtin = Boolean(state.builtinKey);
  $("duplicate-button").hidden = !builtin;
  $("save-button").hidden = builtin;
  $("export-button").hidden = builtin || !state.document;
  $("export-button").disabled = state.busy || state.dirty;
  $("export-button").title = state.dirty ? "Save your edits before exporting" : "Download the saved recipe definition";
  $("save-button").disabled = state.busy || !state.dirty;
  $("validate-button").disabled = state.busy;
  $("duplicate-button").disabled = state.busy;
  if (state.document) $("recipe-meta").textContent = `${state.document.operation} · ${builtin ? "Certified built-in" : `Revision ${state.revision}${state.dirty ? " · Unsaved edits" : " · Saved"}`}`;
  $("publication").hidden = builtin || !state.document;
  const published = state.publication;
  $("publication-button").textContent = published?.enabled ? "Disable tool" : "Enable tool";
  $("publication-button").disabled = state.busy || state.dirty || !published;
  $("publication-status").textContent = !published ? "Loading publication state…" : published.enabled ? "Enabled in Engine catalog" : "Disabled · Not in Engine catalog";
  $("publication-detail").textContent = !published ? "" : `Tool ${published.tool.id} · ${published.tool.available ? "Dependencies resolved" : published.tool.unavailable_reason}${state.dirty ? " · Save edits before changing publication." : ""}`;
}

async function loadPublication() {
  state.publication = state.builtinKey ? null : await api(`/recipes/${state.document.id}/publication`);
  updateToolbar();
}

$("publication-button").addEventListener("click", () => work(async () => {
  state.publication = await api(`/recipes/${state.document.id}/publication`, { method: "PUT", body: { enabled: !state.publication.enabled } });
  updateToolbar();
  notice(state.publication.enabled ? "Recipe enabled. Refresh the Engine catalog in your client to use it." : "Recipe disabled. Previously accepted jobs keep their saved revision.");
}));

function markDirty() {
  state.dirty = true;
  state.validation = null;
  updateToolbar();
  renderValidation();
}

async function loadLibrary() {
  const [operations, builtins, users] = await Promise.all([api("/operations"), api("/builtins"), api("/recipes")]);
  state.operations = new Map(operations.operations.map((operation) => [operation.key, operation]));
  state.builtins = builtins.recipes;
  state.users = users.recipes;
  $("connection-status").textContent = "Engine connected";
  $("connection-button").classList.add("connected");
  renderLibrary();
}

function renderLibrary() {
  const node = $("recipe-library");
  node.replaceChildren();
  for (const [title, entries, builtin] of [["Built-in recipes", state.builtins, true], ["Your recipes", state.users, false]]) {
    const group = element("div", { class: "library-group" });
    group.append(element("div", { class: "library-label" }, [title, element("span", { text: entries.length })]));
    if (!entries.length) group.append(element("p", { class: "no-recipes", text: "Duplicate a built-in to start your first recipe." }));
    for (const item of entries) {
      const selected = builtin ? state.builtinKey === item.key : !state.builtinKey && state.document?.id === item.document.id;
      group.append(element("button", {
        class: `recipe-link${selected ? " active" : ""}`,
        "data-recipe-id": item.document.id,
        ...(builtin ? { "data-builtin-key": item.key } : {}),
        onclick: () => work(async () => {
          if (state.dirty && !confirm("Discard unsaved edits and open another recipe?")) return;
          const record = builtin ? item : await api(`/recipes/${item.document.id}`);
          selectRecipe(record, builtin ? item.key : null);
          await validate(false);
        }),
      }, [item.document.name, element("small", { text: builtin ? "BUILT-IN · READ-ONLY" : `Revision ${item.revision}` })]));
    }
    node.append(group);
  }
}

function selectRecipe(record, builtinKey = null) {
  state.document = structuredClone(record.document);
  state.builtinKey = builtinKey;
  state.revision = record.revision ?? null;
  state.hash = record.definition_hash ?? null;
  state.dirty = false;
  state.validation = null;
  state.publication = null;
  notice();
  renderLibrary();
  renderEditor();
}

async function validate(openIssues = true) {
  state.validation = await api("/validate", { method: "POST", body: state.document });
  await loadPublication();
  renderValidation(openIssues);
  return state.validation;
}

async function save() {
  await work(async () => {
    const result = await validate(true);
    if (!result.document_valid || !result.recipe_compiles) {
      notice("Fix the validation issues below before saving.", true);
      return;
    }
    const record = await api(`/recipes/${state.document.id}`, { method: "PUT", body: { base_revision: state.revision, document: state.document } });
    const validation = state.validation;
    selectRecipe(record);
    await loadPublication();
    state.validation = validation;
    renderValidation();
    await loadLibrary();
    notice(`Saved revision ${record.revision}. Previous published revisions are preserved.`);
  });
}

async function reloadLatest() {
  await work(async () => {
    if (state.dirty && !confirm("Discard unsaved edits and reload the latest saved revision?")) return;
    selectRecipe(await api(`/recipes/${state.document.id}`));
    await loadLibrary();
    await validate(false);
  });
}

async function exportRecipe() {
  await work(async () => {
    const content = await api(`/recipes/${state.document.id}/export`, {}, true);
    const url = URL.createObjectURL(new Blob([content], { type: "application/json" }));
    const link = element("a", { href: url, download: `recipe-${state.document.id}.json` });
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
}

function issueList(issues) {
  const list = element("ul", { class: "issue-list" });
  for (const issue of issues) list.append(element("li", {}, [
    element("code", { text: `${issue.stage} · ${issue.path} · ${issue.code}` }),
    issue.message, element("small", { text: issue.remediation }),
  ]));
  return list;
}

function renderImports() {
  $("import-files").disabled = state.importBusy;
  $("import-summary").textContent = state.importBusy ? "Checking recipe files…" : `${state.imports.length} files staged. Confirm each import below.`;
  const list = $("import-list");
  list.replaceChildren();
  for (const item of state.imports) {
    const card = element("section", { class: "import-card", "data-import-file": item.name });
    const preview = item.preview;
    const doc = preview?.document;
    card.append(element("span", { class: "eyebrow", text: item.name }));
    card.append(element("h3", { text: doc?.name || "Recipe file" }));
    if (doc) card.append(element("p", { class: "muted path-text", text: `${doc.operation} · ${doc.id}` }));
    if (preview) {
      const validation = preview.validation;
      const labels = { new: "New recipe", identical: "Already present · No changes", conflict: "UUID exists · Copy required", builtin: "Reserved built-in ID · Copy required", invalid: "Cannot import" };
      const statuses = element("div", { class: "import-statuses" }, [
        pill(labels[preview.status], ["conflict", "builtin", "invalid"].includes(preview.status) ? "warning" : ""),
        pill(validation.recipe_compiles ? "Policy valid" : "Policy invalid", validation.recipe_compiles ? "" : "error"),
        pill(validation.artifact_resolution.status === "resolved" ? "Dependencies resolved" : "Unresolved artifacts", validation.artifact_resolution.status === "resolved" ? "" : "warning"),
      ]);
      card.append(statuses);
      if (validation.issues.length) {
        const details = element("details", { class: "import-issues", ...(!validation.recipe_compiles ? { open: "" } : {}) }, [element("summary", { text: `${validation.issues.length} validation ${validation.issues.length === 1 ? "issue" : "issues"}` }), issueList(validation.issues)]);
        card.append(details);
      }
      if (validation.recipe_compiles && validation.artifact_resolution.status !== "resolved") card.append(element("p", { class: "field-footnote", text: "You can import this definition with unresolved paths. Artifact files are not copied." }));
      if (item.result) {
        const record = item.result.record;
        card.append(element("p", { class: "import-result", text: record ? `${item.result.status === "copied" ? "Imported as copy" : "Imported"} · Revision ${record.revision} · ${record.document.id}` : "Already present. No revision was written." }));
      } else if (["new", "conflict", "builtin"].includes(preview.status)) {
        const copy = preview.status !== "new";
        card.append(element("button", {
          class: "primary", text: copy ? "Import as copy" : "Import recipe",
          "aria-label": copy ? `Import ${item.name} as copy` : `Import ${item.name}`,
          disabled: state.importBusy, onclick: () => commitImport(item, copy),
        }));
      }
    }
    if (item.error) card.append(element("p", { class: "error-text", text: item.error }));
    if (!preview && !item.error) card.append(element("p", { class: "muted", text: "Reading and validating…" }));
    list.append(card);
  }
}

async function previewImportFiles() {
  if (state.importBusy) return;
  const files = Array.from($("import-files").files);
  state.importBusy = true;
  state.imports = files.map((file) => ({ name: file.name }));
  renderImports();
  try {
    for (let index = 0; index < files.length; index++) {
      const item = state.imports[index];
      try {
        item.source = await files[index].text();
        // Check JSON syntax locally but forward the original numeric literals.
        // Re-serializing in JavaScript changes 1.0 to 1 and the canonical hash.
        parseJSON(item.source);
        item.preview = await api("/imports/preview", { method: "POST", body: `{"document":${item.source}}` });
      } catch (error) {
        item.error = error instanceof SyntaxError ? `Invalid JSON: ${error.message}` : error.message;
        if (error.status === 401) showError(error);
      }
      renderImports();
    }
  } finally {
    state.importBusy = false;
    $("import-files").value = "";
    renderImports();
  }
}

async function commitImport(item, asCopy) {
  if (state.importBusy) return;
  state.importBusy = true;
  item.error = null;
  renderImports();
  try {
    item.result = await api("/imports", { method: "POST", body: `{"document":${item.source},"as_copy":${asCopy}}` });
    await loadLibrary();
  } catch (error) {
    item.error = error.message;
    if (error.status === 401) showError(error);
    if (error.status === 409) {
      try { item.preview = await api("/imports/preview", { method: "POST", body: `{"document":${item.source}}` }); }
      catch (previewError) { item.error = previewError.message; }
    }
  } finally {
    state.importBusy = false;
    renderImports();
  }
}

function numericValue(raw, type) {
  const text = raw.trim();
  if (!text) return null;
  if (type === "integer") {
    if (!/^-?\d+$/.test(text)) return raw;
    const integer = BigInt(text);
    return integer >= BigInt(Number.MIN_SAFE_INTEGER) && integer <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(integer) : integer;
  }
  const number = Number(text);
  return Number.isFinite(number) ? number : raw;
}

function scalarControl(descriptor, value, set, accessibleLabel, disabled) {
  const type = descriptor.value_type;
  if (descriptor.choices?.length) {
    const control = element("select", { class: "value-control", "aria-label": accessibleLabel, disabled });
    descriptor.choices.forEach((choice, index) => control.append(element("option", { value: index, text: String(choice) })));
    const selected = descriptor.choices.findIndex((choice) => stringifyJSON(choice) === stringifyJSON(value));
    control.value = selected < 0 ? "" : String(selected);
    control.addEventListener("change", () => set(descriptor.choices[Number(control.value)]));
    return control;
  }
  if (type === "boolean") {
    const checkbox = element("input", { type: "checkbox", "aria-label": accessibleLabel, checked: value === true, disabled });
    checkbox.addEventListener("change", () => set(checkbox.checked));
    return element("label", { class: "boolean-control" }, [checkbox, "Enabled"]);
  }
  if (type === "text") {
    const control = element("textarea", { "aria-label": accessibleLabel, value: value ?? "", disabled });
    control.addEventListener("input", () => set(control.value));
    return control;
  }
  const numeric = ["number", "integer"].includes(type);
  const control = element("input", {
    type: "text", class: `value-control${numeric ? " numeric-input" : ""}`,
    "aria-label": accessibleLabel, value: value === null || value === undefined ? "" : String(value),
    ...(numeric ? { inputmode: type === "integer" ? "numeric" : "decimal" } : {}), disabled,
  });
  control.addEventListener("input", () => set(numeric ? numericValue(control.value, type) : control.value));
  return control;
}

function pathControl(descriptor, reference, set, accessibleLabel, disabled) {
  const input = element("input", { type: "text", class: "path-input", "aria-label": accessibleLabel, value: reference?.path ?? "", placeholder: "Paste an absolute local path", spellcheck: "false", disabled });
  input.addEventListener("input", () => set({ source: "local", path: input.value }));
  const button = element("button", {
    class: "secondary", text: "Find", "aria-label": `Find ${accessibleLabel}`, disabled,
    onclick: () => openPicker(descriptor, (artifact) => { set(artifact); renderEditor(); }),
  });
  return element("div", { class: "input-action" }, [input, button]);
}

function valueControl(descriptor, value, set, accessibleLabel, disabled) {
  if (descriptor.value_type === "artifact") return pathControl(descriptor, value, set, `${accessibleLabel} path`, disabled);
  if (descriptor.value_type === "adapter") {
    let current = value ?? { artifact: { source: "local", path: "" }, strength: 1 };
    const row = element("div");
    row.append(pathControl(descriptor, current.artifact, (artifact) => { current = { ...current, artifact }; set(current); }, `${accessibleLabel} path`, disabled));
    row.append(element("div", { class: "adapter-strength" }, [
      element("label", { text: "Strength" }),
      scalarControl({ value_type: "number" }, current.strength, (strength) => { current = { ...current, strength }; set(current); }, `${accessibleLabel} strength`, disabled),
    ]));
    return row;
  }
  return scalarControl(descriptor, value, set, `${accessibleLabel} value`, disabled);
}

function collectionControl(descriptor, field, disabled) {
  const box = element("div", { class: "collection" });
  const values = Array.isArray(field.value) ? field.value : [];
  const title = label(field.key);
  if (!values.length) box.append(element("p", { class: "empty-collection", text: "No items selected." }));
  values.forEach((value, index) => {
    const row = element("div", { class: "collection-row" });
    row.append(valueControl(descriptor, value, (next) => { field.value[index] = next; markDirty(); }, `${title} ${index + 1}`, disabled));
    const move = (offset) => {
      [field.value[index], field.value[index + offset]] = [field.value[index + offset], field.value[index]];
      markDirty(); renderEditor();
    };
    row.append(element("div", { class: "collection-controls" }, [
      element("span", { class: "row-index", text: `ITEM ${String(index + 1).padStart(2, "0")}` }),
      element("button", { class: "quiet", text: "↑", "aria-label": `Move ${title} ${index + 1} up`, disabled: disabled || index === 0, onclick: () => move(-1) }),
      element("button", { class: "quiet", text: "↓", "aria-label": `Move ${title} ${index + 1} down`, disabled: disabled || index === values.length - 1, onclick: () => move(1) }),
      element("button", { class: "quiet", text: "Remove", "aria-label": `Remove ${title} ${index + 1}`, disabled, onclick: () => { field.value.splice(index, 1); markDirty(); renderEditor(); } }),
    ]));
    box.append(row);
  });
  box.append(element("button", {
    class: "collection-add", text: "+ Add item", "aria-label": `Add ${title} item`, disabled,
    onclick: () => {
      let value;
      if (descriptor.value_type === "artifact") value = { source: "local", path: "" };
      else if (descriptor.value_type === "adapter") value = { artifact: { source: "local", path: "" }, strength: 1 };
      else if (["number", "integer"].includes(descriptor.value_type)) value = 1;
      else if (descriptor.value_type === "boolean") value = false;
      else value = "";
      field.value = [...values, value]; markDirty(); renderEditor();
    },
  }));
  return box;
}

function constraintsControl(descriptor, field, disabled) {
  const details = element("details", { class: "constraint-details" }, [element("summary", { text: "Narrowed constraints" })]);
  const grid = element("div", { class: "constraint-grid" });
  if (["number", "integer"].includes(descriptor.value_type)) {
    for (const key of ["minimum", "maximum", "step"]) {
      const name = `${label(field.key)} ${key}`;
      const input = element("input", { "aria-label": name, value: field[key] === undefined ? "" : String(field[key]), placeholder: descriptor[key] === null || descriptor[key] === undefined ? "Inherit" : `Inherit ${descriptor[key]}`, inputmode: "decimal", disabled });
      input.addEventListener("input", () => {
        if (!input.value.trim()) delete field[key];
        else field[key] = numericValue(input.value, descriptor.value_type);
        markDirty();
      });
      grid.append(element("label", {}, [element("span", { class: "control-label", text: label(key) }), input]));
    }
  }
  const choices = element("input", { "aria-label": `${label(field.key)} choices`, value: field.choices === undefined ? "" : stringifyJSON(field.choices), placeholder: "Inherit · or a JSON array", disabled });
  choices.addEventListener("input", () => {
    if (!choices.value.trim()) delete field.choices;
    else { try { field.choices = parseJSON(choices.value); } catch { field.choices = choices.value; } }
    markDirty();
  });
  grid.append(element("label", { class: "full-width" }, [element("span", { class: "control-label", text: "Choices" }), choices]));
  if (descriptor.optional) {
    const nullable = element("select", { "aria-label": `${label(field.key)} nullability`, disabled }, [
      element("option", { value: "inherit", text: "Inherit family nullability" }),
      element("option", { value: "true", text: "Allow an empty value" }),
      element("option", { value: "false", text: "Require a value" }),
    ]);
    nullable.value = field.nullable === undefined ? "inherit" : String(field.nullable);
    nullable.addEventListener("change", () => {
      if (nullable.value === "inherit") delete field.nullable;
      else field.nullable = nullable.value === "true";
      markDirty();
    });
    grid.append(element("label", { class: "full-width" }, [element("span", { class: "control-label", text: "Empty values" }), nullable]));
  }
  details.append(grid);
  return details;
}

function renderEditor() {
  $("empty-state").hidden = Boolean(state.document);
  $("editor").hidden = !state.document;
  if (!state.document) return;
  const disabled = Boolean(state.builtinKey);
  $("recipe-kind").textContent = disabled ? "BUILT-IN RECIPE" : "USER RECIPE";
  $("recipe-name").value = state.document.name;
  $("recipe-name").readOnly = disabled;
  $("readonly-banner").hidden = !disabled;
  $("artifact-fields").replaceChildren();
  $("policy-fields").replaceChildren();
  $("caller-fields").replaceChildren();
  const operation = state.operations.get(state.document.operation);
  const descriptors = new Map(operation.fields.map((field) => [field.key, field]));
  for (const field of state.document.fields) {
    const descriptor = descriptors.get(field.key);
    if (!descriptor || descriptor.owner === "host") continue;
    if (descriptor.owner === "caller") {
      $("caller-fields").append(element("div", { class: "caller-chip" }, [label(field.key), element("span", { text: descriptor.value_type })]));
      continue;
    }
    const artifact = descriptor.owner === "artifact";
    const card = element("div", { class: `field-card${descriptor.ordered || descriptor.value_type === "text" ? " wide-field" : ""}`, "data-field": field.key });
    const heading = element("div", { class: "field-heading" }, [element("h3", { text: label(field.key) })]);
    if (artifact) heading.append(element("span", { class: "slot-status", "data-slot": field.key, text: "Not checked" }));
    else {
      const mode = element("select", { class: "mode-select", "aria-label": `${label(field.key)} mode`, disabled }, [element("option", { value: "fixed", text: "Fixed" }), element("option", { value: "exposed", text: "Exposed" })]);
      mode.value = field.mode;
      mode.addEventListener("change", () => { field.mode = mode.value; markDirty(); renderEditor(); });
      heading.append(mode);
    }
    card.append(heading);
    const presentation = descriptor.presentation;
    const warning = presentation?.advanced_warning ? element("p", { class: "field-footnote", text: presentation.advanced_warning }) : null;
    const updateWarning = () => {
      if (warning) warning.hidden = field.mode === "fixed" && field.value === presentation.certified_value;
    };
    if (!artifact) card.append(element("span", { class: "control-label", text: field.mode === "fixed" ? "Fixed value" : "Default value" }));
    if (descriptor.ordered) card.append(collectionControl(descriptor, field, disabled));
    else card.append(valueControl(descriptor, field.value, (value) => { field.value = value; markDirty(); updateWarning(); }, label(field.key), disabled));
    if (warning) { updateWarning(); card.append(warning); }
    if (artifact) {
      const companions = descriptor.artifact.required_files;
      if (companions?.length) card.append(element("p", { class: "field-footnote", text: `Directory requires ${companions.length} companion files. Validate to check the selected path.` }));
    } else {
      if (descriptor.choices?.length) card.append(element("p", { class: "field-footnote", text: `Family supports: ${descriptor.choices.join(", ")}` }));
      card.append(constraintsControl(descriptor, field, disabled));
    }
    $(artifact ? "artifact-fields" : "policy-fields").append(card);
  }
  updateToolbar();
  renderValidation();
}

function renderValidation(openIssues = false) {
  const summary = $("validation-summary");
  const details = $("validation-details");
  summary.replaceChildren(); details.replaceChildren();
  const result = state.validation;
  if (!result) summary.append(pill(state.dirty ? "Unvalidated edits" : "Not validated", "neutral"), element("span", { text: "Validate to check policy and local paths." }));
  else {
    summary.append(pill(result.recipe_compiles ? "Policy valid" : "Policy needs attention", result.recipe_compiles ? "" : "error"));
    summary.append(pill(result.artifact_resolution.status === "resolved" ? "Dependencies resolved" : "Unresolved artifacts", result.artifact_resolution.status === "resolved" ? "" : "warning"));
    summary.append(element("span", { text: "Execution unverified · Backend not checked" }));
    if (result.execution_readiness.status === "blocked") summary.lastChild.textContent = "Execution blocked · Backend not checked";
    if (result.issues.length) {
      const panel = element("details", { ...(openIssues ? { open: "" } : {}) }, [element("summary", { text: `${result.issues.length} validation ${result.issues.length === 1 ? "issue" : "issues"}` })]);
      panel.append(issueList(result.issues)); details.append(panel);
    }
  }
  document.querySelectorAll("[data-slot]").forEach((node) => {
    const slots = result?.artifact_resolution.slots.filter((slot) => slot.key === node.dataset.slot);
    const resolved = slots && slots.every((slot) => slot.status === "resolved");
    node.textContent = !result ? "Not checked" : resolved ? "Resolved" : "Unresolved";
    node.className = `slot-status${result ? resolved ? " resolved" : " unresolved" : ""}`;
  });
}

function openConnection() {
  $("engine-address").textContent = window.location.origin;
  $("engine-token").value = state.token;
  if (!$("connection-dialog").open) $("connection-dialog").showModal();
}

async function connect(event) {
  event?.preventDefault();
  state.token = $("engine-token").value;
  if (state.token) sessionStorage.setItem("latentslate.authoring.token", state.token);
  else sessionStorage.removeItem("latentslate.authoring.token");
  $("connection-error").textContent = "";
  try {
    await loadLibrary();
    $("connection-dialog").close();
    notice();
    if (!state.document && state.builtins.length) {
      selectRecipe(state.builtins[0], state.builtins[0].key);
      await work(() => validate(false));
    }
  } catch (error) {
    $("connection-error").textContent = error.message;
    showError(error);
  }
}

async function loadRoots() {
  const { roots } = await api("/roots");
  const node = $("root-list");
  node.replaceChildren();
  if (!roots.length) node.append(element("p", { class: "muted", text: "No model folders registered. Manual recipe paths work without a model folder." }));
  for (const root of roots) {
    const remove = element("button", {
      class: "quiet", text: "Remove", "aria-label": `Remove model folder ${root.name || root.path}`,
      onclick: async () => {
        remove.disabled = true;
        try {
          await api(`/roots/${root.id}`, { method: "DELETE" });
          await loadRoots();
          $("root-index-status").textContent = "Index will refresh on the next search.";
        } catch (error) { $("root-error").textContent = error.message; }
        finally { remove.disabled = false; }
      },
    });
    node.append(element("div", { class: "root-item" }, [element("div", {}, [
      element("strong", { text: root.name || root.path.split(/[\\/]/).filter(Boolean).at(-1) || root.path }),
      element("p", { class: "path-text", text: root.path }),
      pill(root.available ? "Available" : "Unavailable · Still registered", root.available ? "" : "warning"),
    ]), remove]));
  }
}

async function openRoots() {
  $("root-error").textContent = "";
  if (!$("roots-dialog").open) $("roots-dialog").showModal();
  try { await loadRoots(); } catch (error) { $("root-error").textContent = error.message; showError(error); }
}

async function addRoot(event) {
  event.preventDefault();
  const button = $("root-form").querySelector("button[type=submit]");
  button.disabled = true; $("root-error").textContent = "";
  try {
    await api("/roots", { method: "POST", body: { path: $("root-path").value, name: $("root-name").value || null } });
    $("root-path").value = ""; $("root-name").value = "";
    await loadRoots();
    $("root-index-status").textContent = "Folder registered. Search will build its index.";
  } catch (error) { $("root-error").textContent = error.message; }
  finally { button.disabled = false; }
}

async function refreshIndex() {
  const buttons = [$("refresh-roots-button"), $("refresh-picker-button")];
  buttons.forEach((button) => { button.disabled = true; });
  $("root-index-status").textContent = "Indexing registered folders…";
  $("search-summary").textContent = "Indexing registered folders…";
  try {
    const index = await api("/artifacts/refresh", { method: "POST" });
    $("root-index-status").textContent = `${index.files} files · ${index.directories} folders indexed`;
    if (state.picker) await searchArtifacts();
  } catch (error) {
    $("root-index-status").textContent = error.message;
    $("search-summary").textContent = error.message;
    showError(error);
  } finally { buttons.forEach((button) => { button.disabled = false; }); }
}

function openPicker(descriptor, select) {
  state.picker = { descriptor, select, operation: state.document.operation };
  $("picker-title").textContent = label(descriptor.key);
  $("artifact-query").value = "";
  $("search-results").replaceChildren();
  $("picker-dialog").showModal();
  $("artifact-query").focus();
  searchArtifacts();
}

async function searchArtifacts() {
  if (!state.picker) return;
  const sequence = ++state.searchSequence;
  const picker = state.picker;
  $("search-summary").textContent = "Searching local artifacts…";
  try {
    const parameters = new URLSearchParams({ operation: picker.operation, field: picker.descriptor.key, q: $("artifact-query").value, limit: "50" });
    const data = await api(`/artifacts/search?${parameters}`);
    if (sequence !== state.searchSequence || state.picker !== picker) return;
    const results = $("search-results");
    results.replaceChildren();
    const indexed = `${data.index.files} files · ${data.index.directories} folders indexed`;
    $("search-summary").textContent = data.total ? `${Math.min(data.total, 50)} of ${data.total} results · ${indexed}` : `No results. Try a shorter filename or add a model folder. · ${indexed}`;
    if (data.index.issues.length) $("search-summary").append(document.createTextNode(` · ${data.index.issues.map((issue) => issue.message).join("; ")}`));
    for (const candidate of data.results) {
      const info = element("div", {}, [
        element("strong", { text: candidate.relative_path === "." ? candidate.absolute_path : candidate.relative_path }),
        element("p", { text: `${candidate.root_name} · ${candidate.kind}` }),
        element("p", { text: candidate.absolute_path }),
        pill(candidate.structural_status === "resolved" ? "Structure found" : "Structure incomplete", candidate.structural_status === "resolved" ? "" : "warning"),
      ]);
      const missing = candidate.checks.filter((check) => !check.present);
      if (missing.length) info.append(element("div", { class: "missing-checks", text: `Missing: ${missing.map((check) => check.path).join(", ")}` }));
      results.append(element("div", { class: "search-result" }, [info, element("button", {
        class: "secondary", text: "Select", "aria-label": `Select ${candidate.relative_path}`,
        onclick: () => { picker.select(candidate.artifact); $("picker-dialog").close(); },
      })]));
    }
  } catch (error) {
    if (sequence !== state.searchSequence) return;
    $("search-summary").textContent = error.message;
    showError(error);
  }
}

$("recipe-name").addEventListener("input", () => { state.document.name = $("recipe-name").value; markDirty(); });
$("validate-button").addEventListener("click", () => work(() => validate(true)));
$("save-button").addEventListener("click", save);
$("export-button").addEventListener("click", exportRecipe);
$("import-button").addEventListener("click", () => $("import-dialog").showModal());
$("import-files").addEventListener("change", previewImportFiles);
$("duplicate-button").addEventListener("click", () => work(async () => {
  const record = await api(`/builtins/${state.builtinKey}/duplicate`, { method: "POST", body: {} });
  selectRecipe(record);
  await loadLibrary();
  await validate(false);
  notice("User recipe created. Edit its name, artifacts, and policy, then save a new revision.");
}));
$("connection-button").addEventListener("click", openConnection);
$("connection-form").addEventListener("submit", connect);
$("clear-token-button").addEventListener("click", () => { $("engine-token").value = ""; connect(); });
$("roots-button").addEventListener("click", openRoots);
$("picker-roots-button").addEventListener("click", openRoots);
$("root-form").addEventListener("submit", addRoot);
$("refresh-roots-button").addEventListener("click", refreshIndex);
$("refresh-picker-button").addEventListener("click", refreshIndex);
$("artifact-query").addEventListener("input", () => {
  clearTimeout(state.searchTimer); state.searchSequence++;
  state.searchTimer = setTimeout(searchArtifacts, 220);
});
$("picker-dialog").addEventListener("close", () => { state.picker = null; state.searchSequence++; clearTimeout(state.searchTimer); });
$("roots-dialog").addEventListener("close", () => { if (state.picker) searchArtifacts(); });
document.querySelectorAll("[data-close]").forEach((button) => button.addEventListener("click", () => $(button.dataset.close).close()));
window.addEventListener("beforeunload", (event) => { if (state.dirty) { event.preventDefault(); event.returnValue = ""; } });

$("engine-token").value = state.token;
connect();
